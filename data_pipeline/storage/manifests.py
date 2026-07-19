from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.common.paths import PROJECT_PATHS


def _component(value: str) -> str:
    normalized = str(value).strip().lower().replace(" ", "_")
    if not normalized or normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError(f"Invalid manifest component: {value!r}")
    return normalized


def _relative_to_data_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_PATHS.data_root.resolve()))
    except ValueError:
        return str(path)


def _event_time_bounds(rows: pd.DataFrame, timestamp_col: str) -> dict[str, int | None]:
    if timestamp_col not in rows.columns:
        return {"min_event_time_ns": None, "max_event_time_ns": None}
    timestamps = pd.to_numeric(rows[timestamp_col], errors="coerce").dropna()
    if timestamps.empty:
        return {"min_event_time_ns": None, "max_event_time_ns": None}
    return {
        "min_event_time_ns": int(timestamps.min()),
        "max_event_time_ns": int(timestamps.max()),
    }


def write_ingestion_manifest(
    *,
    provider: str,
    market: str,
    symbol: str,
    dataset: str,
    data_path: Path,
    rows: pd.DataFrame,
    timestamp_col: str = "event_time_ns",
    source: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write one immutable metadata record for a raw-data write.

    Keeping a small JSON file beside each ingestion event avoids mutable global
    state and lets a catalog be rebuilt by scanning ``metadata/manifests``.
    """
    event_date = data_path.parent.name.removeprefix("date=")
    manifest_dir = PROJECT_PATHS.manifests_root.joinpath(
        _component(provider),
        _component(market),
        _component(symbol).upper(),
        _component(dataset),
        f"date={event_date}",
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{data_path.stem}.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "provider": _component(provider),
        "market": _component(market),
        "symbol": _component(symbol).upper(),
        "dataset": _component(dataset),
        "source": source,
        "data_path": _relative_to_data_root(data_path),
        "row_count": int(len(rows)),
        "columns": list(rows.columns),
        **_event_time_bounds(rows, timestamp_col),
    }
    if extra:
        payload["extra"] = dict(extra)

    temp_path = manifest_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(manifest_path)
    return manifest_path
