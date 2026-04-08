import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.paths import PROJECT_PATHS


PROJECT_ROOT = PROJECT_PATHS.project_root
STRUCTURED_DIR = PROJECT_PATHS.cycle_structured_dir
DASHBOARD_DATA_DIR = PROJECT_PATHS.dashboard_root / "candles"
DASHBOARD_META_DIR = PROJECT_PATHS.dashboard_root / "meta"
SUPPORTED_TIMEFRAMES = ("1m", "1h", "4h", "1d", "1w")
BASE_FIELDS = {
    "cycle_id",
    "asset",
    "timeframe",
    "start_date",
    "end_date",
    "cycle_type",
    "duration_candles",
    "category",
    "algorithm_used",
}


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return value
    return str(value)


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        safe_key = str(key).strip().replace(" ", "_")
        next_prefix = f"{prefix}_{safe_key}" if prefix else safe_key
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, next_prefix))
        else:
            flattened[next_prefix] = _normalize_scalar(value)
    return flattened


def _find_cycle_parquet(asset: str, timeframe: str) -> Path | None:
    candidates = [
        STRUCTURED_DIR / asset / f"cycles_{timeframe}.parquet",
        STRUCTURED_DIR / f"cycles_{timeframe}.parquet",
        STRUCTURED_DIR / asset / f"cycles_{timeframe}_enriched.parquet",
        STRUCTURED_DIR / f"cycles_{timeframe}_enriched.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_hierarchy_map(asset: str) -> Path | None:
    candidates = [
        STRUCTURED_DIR / asset / "cycle_hierarchy_map.json",
        STRUCTURED_DIR / "cycle_hierarchy_map.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_hierarchy(asset: str) -> dict[str, Any]:
    hierarchy_path = _find_hierarchy_map(asset)
    if not hierarchy_path:
        return {}
    return json.loads(hierarchy_path.read_text(encoding="utf-8"))


def _extract_structure_fields(
    hierarchy: dict[str, Any],
    timeframe: str,
    cycle_id: str,
) -> dict[str, Any]:
    node = hierarchy.get(timeframe, {}).get(cycle_id, {})
    if not node:
        return {}

    fields: dict[str, Any] = {}
    parent_ids = node.get("parent_cycle_ids", {})
    child_ids = node.get("child_cycle_ids", {})

    for parent_tf, ids in parent_ids.items():
        first_id = ids[0] if ids else None
        fields[f"struct_parent_{parent_tf}_cycle_id"] = first_id
        if first_id:
            parent_node = hierarchy.get(parent_tf, {}).get(first_id, {})
            fields[f"struct_parent_{parent_tf}_cycle_type"] = parent_node.get("cycle_type")

    for child_tf, ids in child_ids.items():
        fields[f"struct_child_{child_tf}_count"] = len(ids or [])

    return fields


def _extract_cycle_feature_fields(row: pd.Series) -> dict[str, Any]:
    features = row.get("cycle_features")
    if not isinstance(features, dict):
        return {}
    return _flatten_dict(features)


def _build_record(
    row: pd.Series,
    asset: str,
    timeframe: str,
    hierarchy: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "cycle_id": row.get("cycle_id"),
        "asset": asset,
        "timeframe": timeframe,
        "start_date": _normalize_scalar(row.get("start_date")),
        "end_date": _normalize_scalar(row.get("end_date")),
        "cycle_type": _normalize_scalar(row.get("cycle_type")),
        "duration_candles": _normalize_scalar(row.get("duration_candles")),
        "category": _normalize_scalar(row.get("category")),
        "algorithm_used": _normalize_scalar(row.get("algorithm_used")),
    }
    record.update(_extract_structure_fields(hierarchy, timeframe, str(row.get("cycle_id")),))
    record.update(_extract_cycle_feature_fields(row))
    return record


def build_dashboard_dataframe(asset: str, timeframe: str) -> pd.DataFrame:
    source_path = _find_cycle_parquet(asset, timeframe)
    if not source_path:
        raise FileNotFoundError(f"No cycle parquet found for asset={asset}, timeframe={timeframe}")

    hierarchy = _load_hierarchy(asset)
    source_df = pd.read_parquet(source_path)
    records = [
        _build_record(row, asset=asset, timeframe=timeframe, hierarchy=hierarchy)
        for _, row in source_df.iterrows()
    ]
    dashboard_df = pd.DataFrame(records)

    for column in ("start_date", "end_date"):
        if column in dashboard_df.columns:
            dashboard_df[column] = pd.to_datetime(dashboard_df[column], errors="coerce")

    return dashboard_df


def _infer_field_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"
    if pd.api.types.is_bool_dtype(non_null):
        return "boolean"
    if pd.api.types.is_numeric_dtype(non_null):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    sample = non_null.head(min(len(non_null), 25)).astype(str).str.strip()
    datetime_like_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?")
    if not sample.empty and sample.map(lambda value: bool(datetime_like_pattern.match(value))).all():
        parsed_dt = pd.to_datetime(sample, errors="coerce", format="mixed")
        if not parsed_dt.isna().all():
            return "datetime"
    return "string"


def _build_number_meta(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"min": None, "max": None}
    return {
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def _build_string_meta(series: pd.Series, max_options: int = 100) -> dict[str, Any]:
    values = series.dropna().astype(str)
    unique_values = sorted(values.unique().tolist())
    meta = {"distinct_count": len(unique_values)}
    if len(unique_values) <= max_options:
        meta["options"] = unique_values
    return meta


def build_feature_catalog(df: pd.DataFrame) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for field in df.columns:
        series = df[field]
        field_type = _infer_field_type(series)
        if field in BASE_FIELDS:
            field_group = "base"
            category = "base"
            feature_name = field
        elif field.startswith("struct_"):
            field_group = "structure"
            category = "structure"
            feature_name = field.removeprefix("struct_")
        else:
            field_group = "feature"
            if "_" in field:
                category, feature_name = field.split("_", 1)
            else:
                category, feature_name = "feature", field

        item = {
            "field": field,
            "label": field.replace("_", " "),
            "field_group": field_group,
            "category": category,
            "feature_name": feature_name,
            "data_type": field_type,
            "null_count": int(series.isna().sum()),
            "filterable": field_type in {"number", "string", "datetime", "boolean"},
        }

        if field_type == "number":
            item["filter_type"] = "range"
            item["filter_ops"] = ["between", "gte", "lte", "gt", "lt", "eq"]
            item.update(_build_number_meta(series))
        elif field_type == "string":
            item["filter_type"] = "select"
            item["filter_ops"] = ["in", "eq", "neq"]
            item.update(_build_string_meta(series))
        elif field_type == "datetime":
            item["filter_type"] = "date_range"
            item["filter_ops"] = ["between", "gte", "lte"]
            non_null = pd.to_datetime(series, errors="coerce").dropna()
            item["min"] = non_null.min().isoformat() if not non_null.empty else None
            item["max"] = non_null.max().isoformat() if not non_null.empty else None
        elif field_type == "boolean":
            item["filter_type"] = "boolean"
            item["filter_ops"] = ["eq"]
            item["options"] = [True, False]
        else:
            item["filter_type"] = "unsupported"
            item["filter_ops"] = []

        catalog.append(item)

    return sorted(catalog, key=lambda item: (item["field_group"], item["category"], item["field"]))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_and_save(asset: str, timeframe: str) -> tuple[Path, Path]:
    df = build_dashboard_dataframe(asset=asset, timeframe=timeframe)
    catalog = build_feature_catalog(df)

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_META_DIR.mkdir(parents=True, exist_ok=True)

    data_path = DASHBOARD_DATA_DIR / f"{asset}_{timeframe}.parquet"
    meta_path = DASHBOARD_META_DIR / f"{asset}_{timeframe}_features.json"

    df.to_parquet(data_path, index=False)
    _write_json(meta_path, {
        "asset": asset,
        "timeframe": timeframe,
        "row_count": len(df),
        "field_count": len(df.columns),
        "fields": catalog,
    })
    return data_path, meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build flat dashboard datasets and field metadata from cycle parquet files.",
    )
    parser.add_argument("--asset", default="btc", help="Asset namespace under data/cycle_data/structured.")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h", "4h", "1d", "1w"],
        help="Timeframes to build. Example: --timeframes 1h 4h 1d",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_timeframes = [tf for tf in args.timeframes if tf in SUPPORTED_TIMEFRAMES]
    if not requested_timeframes:
        raise ValueError(f"No supported timeframes requested. Supported: {', '.join(SUPPORTED_TIMEFRAMES)}")

    for timeframe in requested_timeframes:
        data_path, meta_path = build_and_save(asset=args.asset, timeframe=timeframe)
        print(f"[OK] {args.asset} {timeframe} data -> {data_path}")
        print(f"[OK] {args.asset} {timeframe} meta -> {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
