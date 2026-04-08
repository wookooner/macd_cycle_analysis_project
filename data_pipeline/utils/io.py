import logging
import os
import shutil
import uuid
from pathlib import Path

import pandas as pd


BACKUP_KEEP_OLDEST = 3
BACKUP_KEEP_NEWEST = 5
_VALIDATION_SAMPLE_BYTES = 4096
logger = logging.getLogger(__name__)


def _has_nul_bytes(path: Path, sample_bytes: int = _VALIDATION_SAMPLE_BYTES) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(sample_bytes)
    except OSError:
        return True


def is_valid_csv_file(path: str | Path, required_columns: list[str] | None = None) -> bool:
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return False
    if candidate.stat().st_size == 0 or _has_nul_bytes(candidate):
        return False

    try:
        sample = pd.read_csv(candidate, nrows=5)
    except Exception:
        return False

    if required_columns:
        missing = [col for col in required_columns if col not in sample.columns]
        if missing:
            return False
    return True


def _iter_recovery_candidates(path: Path):
    candidates: list[Path] = []

    candidates.extend(
        sorted(
            path.parent.glob(f"{path.name}.*.tmp"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    )

    backup_dir = path.parent.parent / "backup_data"
    if backup_dir.exists():
        candidates.extend(
            sorted(
                backup_dir.glob(f"{path.name}.backup_*"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        )
        candidates.extend(
            sorted(
                backup_dir.glob(f"{path.name}.backfill_backup_*"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or candidate == path:
            continue
        seen.add(candidate)
        yield candidate


def load_csv_with_recovery(
    path: str | Path,
    required_columns: list[str] | None = None,
    restore_in_place: bool = False,
) -> pd.DataFrame:
    csv_path = Path(path)
    if is_valid_csv_file(csv_path, required_columns=required_columns):
        return pd.read_csv(csv_path)

    if csv_path.exists():
        logger.warning("Corrupted CSV detected: %s", csv_path)

    for candidate in _iter_recovery_candidates(csv_path):
        if not is_valid_csv_file(candidate, required_columns=required_columns):
            continue

        logger.warning("Recovering %s from %s", csv_path.name, candidate.name)
        recovered_df = pd.read_csv(candidate)
        if restore_in_place:
            atomic_write_csv(recovered_df, csv_path)
        return recovered_df

    return pd.DataFrame()


def atomic_write_csv(df: pd.DataFrame, output_path: str | Path, **to_csv_kwargs) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    recovery_path = None

    try:
        if path.exists():
            recovery_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.recovery")
            shutil.copy2(path, recovery_path)

        df.to_csv(temp_path, index=False, **to_csv_kwargs)
        if not is_valid_csv_file(temp_path, required_columns=list(df.columns)):
            raise IOError(f"Temporary CSV validation failed: {temp_path}")

        os.replace(temp_path, path)
        if not is_valid_csv_file(path, required_columns=list(df.columns)):
            raise IOError(f"Written CSV validation failed: {path}")
    except Exception:
        if recovery_path is not None and recovery_path.exists():
            os.replace(recovery_path, path)
        raise
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        if recovery_path is not None and recovery_path.exists():
            recovery_path.unlink(missing_ok=True)


def prune_backup_files(backup_dir: str | Path, keep_oldest: int = BACKUP_KEEP_OLDEST, keep_newest: int = BACKUP_KEEP_NEWEST) -> None:
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return

    groups: dict[str, list[Path]] = {}
    for file_path in backup_path.iterdir():
        if not file_path.is_file():
            continue

        name = file_path.name
        if ".backfill_backup_" in name:
            base_name = name.split(".backfill_backup_", 1)[0]
        elif ".backup_" in name:
            base_name = name.split(".backup_", 1)[0]
        else:
            continue

        groups.setdefault(base_name, []).append(file_path)

    for files in groups.values():
        ordered = sorted(files, key=lambda item: (item.stat().st_mtime, item.name))
        keep_count = min(len(ordered), keep_oldest + keep_newest)
        if len(ordered) <= keep_count:
            continue

        keep_names = {item.name for item in ordered[:keep_oldest]}
        keep_names.update(item.name for item in ordered[-keep_newest:])
        for file_path in ordered:
            if file_path.name not in keep_names:
                file_path.unlink(missing_ok=True)
