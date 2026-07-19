from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest_entry(
    path: Path,
    *,
    purpose: str,
    derived_from: str,
    view_of: str,
    filter_clause: str = "",
    sort_clause: str = "",
    code_version: str = "shared_backtest_v1",
    sample_split_thresholds: dict[str, int] | None = None,
) -> dict[str, Any]:
    row_count = column_count = None
    if path.exists() and path.suffix.lower() == ".csv":
        row_count = sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1
        column_count = len(pd.read_csv(path, nrows=0).columns)
    return {
        "file_name": path.name,
        "purpose": purpose,
        "derived_from": derived_from,
        "view_of": view_of,
        "filter_clause": filter_clause,
        "sort_clause": sort_clause,
        "row_count": row_count,
        "column_count": column_count,
        "code_version": code_version,
        "data_snapshot_date": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S %Z"),
        "sample_split_thresholds": sample_split_thresholds or {"low_lt": 100, "reliable_gte": 1000},
        "sha256": sha256_file(path),
    }


def write_manifest(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
