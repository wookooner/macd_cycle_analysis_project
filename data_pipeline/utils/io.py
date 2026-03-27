from pathlib import Path
import os
import uuid

import pandas as pd


BACKUP_KEEP_OLDEST = 3
BACKUP_KEEP_NEWEST = 5


def atomic_write_csv(df: pd.DataFrame, output_path: str | Path, **to_csv_kwargs) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    df.to_csv(temp_path, index=False, **to_csv_kwargs)
    os.replace(temp_path, path)


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
