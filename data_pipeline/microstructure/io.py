from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now_ns() -> int:
    return int(pd.Timestamp.now(tz="UTC").value)


def event_date_from_ns(timestamp_ns: int | None) -> str:
    if timestamp_ns is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return pd.Timestamp(timestamp_ns, unit="ns", tz="UTC").strftime("%Y-%m-%d")


def write_partitioned_parquet(
    rows: list[dict[str, Any]],
    base_dir: Path,
    timestamp_col: str = "event_time_ns",
    prefix: str = "part",
    keep_recent_files: int | None = None,
) -> Path | None:
    if not rows:
        return None

    df = pd.DataFrame(rows)
    if timestamp_col in df.columns:
        ts_ns = pd.to_numeric(df[timestamp_col], errors="coerce").dropna()
        event_date = event_date_from_ns(int(ts_ns.iloc[-1])) if not ts_ns.empty else event_date_from_ns(None)
    else:
        event_date = event_date_from_ns(None)

    out_dir = base_dir / f"date={event_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:10]}.parquet"
    df.to_parquet(out_path, index=False)
    prune_recent_parquet_files(base_dir, keep_recent_files)
    return out_path


def prune_recent_parquet_files(base_dir: Path, keep_recent_files: int | None = None) -> None:
    if keep_recent_files is None or keep_recent_files <= 0 or not base_dir.exists():
        return
    files = sorted(
        base_dir.rglob("*.parquet"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    for path in files[keep_recent_files:]:
        try:
            path.unlink()
            parent = path.parent
            while parent != base_dir and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
        except OSError:
            continue


def read_parquet_tree(base_dir: Path) -> pd.DataFrame:
    files = sorted(base_dir.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True, sort=False)


def read_parquet_trees_prefer_first(base_dirs: tuple[Path, ...]) -> pd.DataFrame:
    """Read compatible directory trees, preferring files from the first tree.

    A layout migration copies the same ``date=.../filename.parquet`` hierarchy
    to its canonical destination. Selecting by relative path prevents those
    migrated files from being read twice while still allowing a partial
    migration to fall back to the older tree.
    """
    selected: dict[Path, Path] = {}
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.rglob("*.parquet")):
            selected.setdefault(path.relative_to(base_dir), path)
    if not selected:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in selected.values()), ignore_index=True, sort=False)
