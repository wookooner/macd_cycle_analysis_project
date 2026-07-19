from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from fastapi import HTTPException

from src.common.paths import PROJECT_PATHS
from src.dashboard_api.query_engine import (
    DuckDBDashboardQueryEngine,
    PandasDashboardQueryEngine,
    get_configured_query_engine,
)


DASHBOARD_DATA_DIR = PROJECT_PATHS.dashboard_root / "candles"
MAX_PREVIEW_ROWS = 200
MAX_FILTER_OPTIONS = 80
MAX_GROUPABLE_NUMERIC_VALUES = 24
TIMEFRAME_ORDER = ("1min", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M")
SCATTER_DOWNSAMPLE_THRESHOLD = 2400
SCATTER_GRID_X = 120
SCATTER_GRID_Y = 72

BOUNDARY_LABELS = {
    0: "normal",
    1: "straddle",
    2: "transition_trigger",
}
PARENT_ASSIGN_RULE_LABELS = {
    0: "contained",
    1: "by_start",
}
TYPE_CODE_LABELS = {
    -1: "down",
    1: "up",
}


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    asset: str
    timeframe: str
    source: str
    label: str
    path: Path
    row_count: int
    child_timeframe: str | None = None


_DATASET_CATALOG_CACHE: dict[str, DatasetInfo] | None = None
_BASE_DATAFRAME_CACHE: dict[str, pd.DataFrame] = {}
_QUERY_DATAFRAME_CACHE: dict[str, pd.DataFrame] = {}
_FEATURE_RESPONSE_CACHE: dict[str, dict[str, Any]] = {}


def _compact_label(value: str) -> str:
    return value.replace("__", " / ").replace("_", " ").strip().title()


def _timeframe_sort_key(timeframe: str) -> int:
    try:
        return TIMEFRAME_ORDER.index(timeframe)
    except ValueError:
        return len(TIMEFRAME_ORDER) + 1


def _parent_timeframes(timeframe: str) -> list[str]:
    if timeframe not in TIMEFRAME_ORDER:
        return []
    idx = TIMEFRAME_ORDER.index(timeframe)
    return list(TIMEFRAME_ORDER[idx + 1 :])


def _child_timeframe(timeframe: str) -> str | None:
    if timeframe not in TIMEFRAME_ORDER:
        return None
    idx = TIMEFRAME_ORDER.index(timeframe)
    return TIMEFRAME_ORDER[idx - 1] if idx > 0 else None


def _relation_prefix(parent_timeframe: str) -> str:
    return f"parent_{parent_timeframe}__"


def _read_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _build_dataset_catalog() -> dict[str, DatasetInfo]:
    catalog: dict[str, DatasetInfo] = {}
    cycle_root = PROJECT_PATHS.cycle_structured_dir

    if cycle_root.exists():
        for asset_dir in sorted(cycle_root.iterdir()):
            if not asset_dir.is_dir() or asset_dir.name == "archive":
                continue
            asset = asset_dir.name
            for path in sorted(asset_dir.glob("cycles_*.parquet")):
                timeframe = path.stem.replace("cycles_", "", 1)
                dataset_id = f"{asset}_{timeframe}"
                catalog[dataset_id] = DatasetInfo(
                    id=dataset_id,
                    asset=asset,
                    timeframe=timeframe,
                    source="cycle",
                    label=f"{asset.upper()} {timeframe} cycles",
                    path=path,
                    row_count=_read_row_count(path),
                    child_timeframe=_child_timeframe(timeframe),
                )

    processed_root = PROJECT_PATHS.processed_root / "context"
    if processed_root.exists():
        for asset_dir in sorted(processed_root.iterdir()):
            if not asset_dir.is_dir():
                continue
            asset = asset_dir.name
            for path in sorted(asset_dir.glob("timeframe_context_*.parquet")):
                timeframe = path.stem.replace("timeframe_context_", "", 1)
                dataset_id = f"{asset}_context_{timeframe}"
                catalog[dataset_id] = DatasetInfo(
                    id=dataset_id,
                    asset=asset,
                    timeframe=timeframe,
                    source="context",
                    label=f"{asset.upper()} context {timeframe}",
                    path=path,
                    row_count=_read_row_count(path),
                )

    if DASHBOARD_DATA_DIR.exists():
        for path in sorted(DASHBOARD_DATA_DIR.glob("*.parquet")):
            dataset_id = path.stem
            if dataset_id in catalog or "_" not in dataset_id:
                continue
            asset, timeframe = dataset_id.split("_", 1)
            catalog[dataset_id] = DatasetInfo(
                id=dataset_id,
                asset=asset,
                timeframe=timeframe,
                source="legacy_dashboard",
                label=f"{asset.upper()} {timeframe} dashboard",
                path=path,
                row_count=_read_row_count(path),
            )

    return dict(
        sorted(
            catalog.items(),
            key=lambda item: (
                item[1].asset,
                0 if item[1].source == "cycle" else 1,
                _timeframe_sort_key(item[1].timeframe),
                item[1].id,
            ),
        )
    )


def _dataset_catalog() -> dict[str, DatasetInfo]:
    global _DATASET_CATALOG_CACHE
    if _DATASET_CATALOG_CACHE is None:
        _DATASET_CATALOG_CACHE = _build_dataset_catalog()
    return _DATASET_CATALOG_CACHE


def _get_dataset_info(dataset: str) -> DatasetInfo:
    info = _dataset_catalog().get(dataset)
    if not info:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")
    return info


def _flatten_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    if "cycle_features" not in df.columns:
        return df

    normalized = pd.json_normalize(
        [item if isinstance(item, dict) else {} for item in df["cycle_features"]],
        sep="__",
    )
    if normalized.empty:
        return df.drop(columns=["cycle_features"])

    normalized = normalized.rename(columns=lambda column: f"feature__{column}")
    normalized.index = df.index
    return pd.concat([df.drop(columns=["cycle_features"]), normalized], axis=1)


def _add_cycle_derived_columns(df: pd.DataFrame, asset: str, timeframe: str) -> pd.DataFrame:
    df["asset"] = asset
    df["timeframe"] = timeframe

    if "candle_data" in df.columns:
        df = df.drop(columns=["candle_data"])

    df = _flatten_cycle_features(df)

    if {"start_date", "end_date"}.issubset(df.columns):
        starts = pd.to_datetime(df["start_date"], errors="coerce")
        ends = pd.to_datetime(df["end_date"], errors="coerce")
        span_hours = (ends - starts).dt.total_seconds() / 3600.0
        df["duration_hours"] = span_hours.astype("float32")
        df["duration_days"] = (span_hours / 24.0).astype("float32")

    if "parent_key" in df.columns:
        df["has_parent"] = df["parent_key"].notna()
    if "child_count" in df.columns:
        child_numeric = pd.to_numeric(df["child_count"], errors="coerce").fillna(0)
        df["has_children"] = child_numeric.gt(0)

    if "boundary_type" in df.columns:
        boundary_numeric = pd.to_numeric(df["boundary_type"], errors="coerce")
        df["boundary_label"] = boundary_numeric.map(BOUNDARY_LABELS).fillna("unknown")

    if "parent_assign_rule" in df.columns:
        assign_numeric = pd.to_numeric(df["parent_assign_rule"], errors="coerce")
        df["parent_assign_rule_label"] = assign_numeric.map(PARENT_ASSIGN_RULE_LABELS).fillna("unknown")

    for type_column in ("prev_type", "parent_type", "parent_prev_type", "parent_next_type"):
        if type_column in df.columns:
            numeric = pd.to_numeric(df[type_column], errors="coerce")
            df[f"{type_column}_label"] = numeric.map(TYPE_CODE_LABELS).fillna("unknown")

    return df


def _load_prepared_base_dataset(dataset: str) -> pd.DataFrame:
    if dataset in _BASE_DATAFRAME_CACHE:
        return _BASE_DATAFRAME_CACHE[dataset].copy()

    info = _get_dataset_info(dataset)
    df = pd.read_parquet(info.path)

    if info.source == "cycle":
        df = _add_cycle_derived_columns(df, asset=info.asset, timeframe=info.timeframe)
    elif info.source == "context":
        df["asset"] = info.asset
        df["context_resolution"] = info.timeframe
    elif "asset" not in df.columns:
        df["asset"] = info.asset

    _BASE_DATAFRAME_CACHE[dataset] = df
    return df.copy()


def _load_query_dataset(dataset: str) -> pd.DataFrame:
    if dataset in _QUERY_DATAFRAME_CACHE:
        return _QUERY_DATAFRAME_CACHE[dataset].copy()

    info = _get_dataset_info(dataset)
    base_df = _load_prepared_base_dataset(dataset)

    if info.source != "cycle":
        _QUERY_DATAFRAME_CACHE[dataset] = base_df
        return base_df.copy()

    joined = base_df.copy()
    relation_key = "parent_key"

    for parent_timeframe in _parent_timeframes(info.timeframe):
        if relation_key not in joined.columns:
            break

        parent_dataset = f"{info.asset}_{parent_timeframe}"
        if parent_dataset not in _dataset_catalog():
            continue

        parent_df = _load_prepared_base_dataset(parent_dataset)
        prefix = _relation_prefix(parent_timeframe)
        parent_df = parent_df.rename(columns={column: f"{prefix}{column}" for column in parent_df.columns})
        joined = joined.merge(
            parent_df,
            how="left",
            left_on=relation_key,
            right_on=f"{prefix}cycle_key",
        )
        relation_key = f"{prefix}parent_key"

    _QUERY_DATAFRAME_CACHE[dataset] = joined
    return joined.copy()

def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _serialize_preview(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit, MAX_PREVIEW_ROWS))
    preview = df.head(safe_limit).copy()
    for column in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[column]):
            preview[column] = preview[column].astype("string")
    return preview.where(pd.notna(preview), None).to_dict(orient="records")


def _serialize_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    safe = df.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].astype("string")
    return safe.where(pd.notna(safe), None).to_dict(orient="records")


def _infer_field_group(field_name: str, relation_scope: str) -> str:
    if field_name.startswith("feature__"):
        return "features"
    if field_name in {"n_up_4", "combo_4", "n_up_8", "combo_8", "major_up_count", "minor_up_count"}:
        return "context"
    if field_name.startswith("child_") or field_name in {"has_children", "opposite_child_ratio", "max_opposite_child_streak"}:
        return "child_summary"
    if field_name in {
        "cycle_key",
        "parent_key",
        "prev_key",
        "prev_type",
        "prev_dur",
        "prev_price_pct",
        "parent_type",
        "order_in_parent",
        "total_siblings",
        "parent_progress_at_start",
        "parent_progress_at_end",
        "parent_assign_rule",
        "parent_assign_rule_label",
        "boundary_type",
        "boundary_label",
        "parent_prev_key",
        "parent_prev_type",
        "parent_prev_type_label",
        "parent_next_key",
        "parent_next_type",
        "parent_next_type_label",
        "overlap_prev_ratio",
        "overlap_next_ratio",
        "has_parent",
    }:
        return "relationship"
    if relation_scope == "context":
        return "context"
    return "base"


def _infer_field_location(field_name: str, dataset: str) -> tuple[str, str, str | None]:
    info = _get_dataset_info(dataset)
    if field_name.startswith("parent_") and "__" in field_name:
        prefix, inner = field_name.split("__", 1)
        relation_timeframe = prefix.replace("parent_", "", 1)
        field_group = _infer_field_group(inner, relation_scope="parent")
        return "parent", field_group, relation_timeframe

    if info.source == "context":
        return "context", _infer_field_group(field_name, relation_scope="context"), None

    return "current", _infer_field_group(field_name, relation_scope="current"), info.timeframe


def _infer_data_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "string"


def _field_chart_roles(data_type: str, distinct_count: int | None) -> dict[str, bool]:
    roles = {"x": False, "y": False, "group": False, "color": False, "filter": True}
    if data_type == "number":
        roles["x"] = True
        roles["y"] = True
        if distinct_count is not None and distinct_count <= MAX_GROUPABLE_NUMERIC_VALUES:
            roles["group"] = True
            roles["color"] = True
        return roles
    if data_type == "datetime":
        roles["x"] = True
        return roles
    if data_type in {"string", "boolean"}:
        roles["group"] = True
        roles["color"] = True
        return roles
    roles["filter"] = False
    return roles


def _field_category_label(field_name: str, field_group: str) -> str:
    if field_group == "features" and field_name.startswith("feature__"):
        parts = field_name.split("__")
        return parts[1] if len(parts) > 1 else "features"
    if field_group == "relationship":
        return "relationships"
    if field_group == "child_summary":
        return "child_summary"
    if field_group == "context":
        return "context"
    return "base"


def _build_field_metadata(dataset: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for column in df.columns:
        if column == "candle_data":
            continue

        series = df[column]
        if pd.api.types.is_object_dtype(series) and series.map(lambda item: isinstance(item, (dict, list, tuple))).any():
            continue

        data_type = _infer_data_type(series)
        non_null = series.dropna()
        distinct_count = int(non_null.nunique()) if len(non_null) else 0
        relation_scope, field_group, relation_timeframe = _infer_field_location(column, dataset)
        category = _field_category_label(column, field_group)
        roles = _field_chart_roles(data_type, distinct_count)

        filterable = roles["filter"]
        filter_type = "select"
        meta: dict[str, Any] = {
            "field": column,
            "label": _compact_label(column.replace("feature__", "")),
            "field_group": field_group,
            "category": category,
            "data_type": data_type,
            "null_count": int(series.isna().sum()),
            "distinct_count": distinct_count,
            "filterable": filterable,
            "relation_scope": relation_scope,
            "relation_timeframe": relation_timeframe,
            "chart_roles": roles,
        }

        if data_type == "number":
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                meta["min"] = float(numeric.min())
                meta["max"] = float(numeric.max())
                filter_type = "range"
            else:
                filterable = False
        elif data_type == "datetime":
            dt = _coerce_datetime(series)
            if dt.notna().any():
                meta["min"] = str(dt.min())
                meta["max"] = str(dt.max())
                filter_type = "date_range"
            else:
                filterable = False
        elif data_type == "boolean":
            filter_type = "boolean"
        else:
            if 0 < distinct_count <= MAX_FILTER_OPTIONS:
                meta["options"] = sorted(str(item) for item in non_null.astype(str).unique().tolist())
            else:
                filterable = False
                roles["filter"] = False

        meta["filterable"] = filterable
        if filterable:
            meta["filter_type"] = filter_type
            if data_type == "number":
                meta["filter_ops"] = ["between", "gte", "lte", "gt", "lt", "eq"]
            elif data_type == "datetime":
                meta["filter_ops"] = ["between", "gte", "lte"]
            elif data_type == "boolean":
                meta["filter_ops"] = ["eq", "neq"]
            else:
                meta["filter_ops"] = ["in", "eq", "neq"]
        fields.append(meta)

    return fields


def _choose_default_field(fields: list[dict[str, Any]], *, role: str, preferred: list[str]) -> str:
    role_fields = [field for field in fields if field.get("chart_roles", {}).get(role)]
    field_map = {field["field"]: field for field in role_fields}
    for name in preferred:
        if name in field_map:
            return name
    return role_fields[0]["field"] if role_fields else ""


def _build_analysis_presets(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    field_names = {field["field"] for field in fields}

    def first_of(names: list[str]) -> str:
        for name in names:
            if name in field_names:
                return name
        return ""

    presets = []
    distribution_x = first_of([
        "feature__change__price_pct",
        "duration_candles",
        "duration_hours",
        "child_count",
        "n_up_4",
    ])
    if distribution_x:
        presets.append({
            "key": "distribution",
            "label": "Distribution",
            "description": "Inspect one numeric field as a distribution.",
            "chart_type": "histogram",
            "x": distribution_x,
            "y": "",
            "group_by": "",
            "color": "",
            "metric": "count",
        })

    relation_group = first_of([
        "combo_4",
        "boundary_label",
        "cycle_type",
        "parent_4h__cycle_type",
        "parent_1d__cycle_type",
    ])
    if relation_group:
        presets.append({
            "key": "relation_mix",
            "label": "Relation Mix",
            "description": "Compare counts or aggregates by relation or context buckets.",
            "chart_type": "bar",
            "x": "",
            "y": "",
            "group_by": relation_group,
            "color": "",
            "metric": "count",
        })

    scatter_x = first_of(["parent_progress_at_start", "child_count", "duration_candles"])
    scatter_y = first_of(["feature__change__price_pct", "feature__strength__direction_pct", "duration_hours"])
    if scatter_x and scatter_y:
        presets.append({
            "key": "structure_vs_result",
            "label": "Structure vs Result",
            "description": "See how relation structure aligns with cycle outcome.",
            "chart_type": "scatter",
            "x": scatter_x,
            "y": scatter_y,
            "group_by": "",
            "color": first_of(["cycle_type", "combo_4", "boundary_label"]),
            "metric": "count",
        })

    line_y = first_of(["feature__change__price_pct", "duration_candles", "child_count"])
    if "start_date" in field_names and line_y:
        presets.append({
            "key": "timeline",
            "label": "Timeline",
            "description": "Track a metric over time, optionally colored by regime.",
            "chart_type": "line",
            "x": "start_date",
            "y": line_y,
            "group_by": "",
            "color": first_of(["cycle_type", "combo_4"]),
            "metric": "count",
        })

    return presets


def get_feature_response(dataset: str) -> dict[str, Any]:
    if dataset in _FEATURE_RESPONSE_CACHE:
        return _FEATURE_RESPONSE_CACHE[dataset]

    info = _get_dataset_info(dataset)
    df = _load_query_dataset(dataset)
    fields = _build_field_metadata(dataset, df)
    available_parent_timeframes = sorted(
        {
            field["relation_timeframe"]
            for field in fields
            if field.get("relation_scope") == "parent" and field.get("relation_timeframe")
        },
        key=_timeframe_sort_key,
    )
    payload = {
        "dataset": info.id,
        "label": info.label,
        "asset": info.asset,
        "timeframe": info.timeframe,
        "source": info.source,
        "row_count": int(len(df)),
        "field_count": len(fields),
        "fields": fields,
        "available_parent_timeframes": available_parent_timeframes,
        "child_timeframe": info.child_timeframe,
        "default_chart_state": {
            "chart_type": "histogram",
            "x": _choose_default_field(fields, role="x", preferred=["feature__change__price_pct", "duration_candles", "start_date"]),
            "y": _choose_default_field(fields, role="y", preferred=["feature__change__price_pct", "duration_candles", "child_count"]),
            "group_by": _choose_default_field(fields, role="group", preferred=["cycle_type", "combo_4", "boundary_label", "category"]),
            "color": _choose_default_field(fields, role="color", preferred=["cycle_type", "combo_4", "boundary_label", "category"]),
            "metric": "count",
        },
        "analysis_presets": _build_analysis_presets(fields),
    }
    _FEATURE_RESPONSE_CACHE[dataset] = payload
    return payload


def _field_meta_map(dataset: str) -> dict[str, dict[str, Any]]:
    meta = get_feature_response(dataset)
    return {item["field"]: item for item in meta.get("fields", [])}


def _field_type(dataset: str, field: str) -> str:
    return _field_meta_map(dataset).get(field, {}).get("data_type", "string")

def _apply_filter(df: pd.DataFrame, condition: Any, field_meta: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if condition.field not in df.columns:
        raise HTTPException(status_code=400, detail=f"Unknown field: {condition.field}")

    series = df[condition.field]
    meta = field_meta.get(condition.field, {})
    data_type = meta.get("data_type", "string")
    op = condition.op
    value = condition.value

    if data_type == "datetime":
        series = _coerce_datetime(series)
        if op == "between":
            start, end = value
            mask = (series >= pd.to_datetime(start)) & (series <= pd.to_datetime(end))
        elif op == "gte":
            mask = series >= pd.to_datetime(value)
        elif op == "lte":
            mask = series <= pd.to_datetime(value)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported datetime op: {op}")
        return df[mask]

    if data_type == "number":
        numeric = pd.to_numeric(series, errors="coerce")
        if op == "between":
            low, high = value
            mask = (numeric >= low) & (numeric <= high)
        elif op == "gte":
            mask = numeric >= value
        elif op == "lte":
            mask = numeric <= value
        elif op == "gt":
            mask = numeric > value
        elif op == "lt":
            mask = numeric < value
        elif op == "eq":
            mask = numeric == value
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported numeric op: {op}")
        return df[mask]

    if data_type == "boolean":
        if op == "eq":
            return df[series == value]
        if op == "neq":
            return df[series != value]
        raise HTTPException(status_code=400, detail=f"Unsupported boolean op: {op}")

    comparable = series.astype(str)
    if op == "in":
        mask = comparable.isin([str(item) for item in value])
    elif op == "eq":
        mask = comparable == str(value)
    elif op == "neq":
        mask = comparable != str(value)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported categorical op: {op}")
    return df[mask]


def apply_filters(df: pd.DataFrame, filters: list[Any], dataset: str) -> pd.DataFrame:
    field_meta = _field_meta_map(dataset)
    filtered = df
    for condition in filters:
        filtered = _apply_filter(filtered, condition, field_meta)
    return filtered


def _downsample_scatter_numeric(
    subset: pd.DataFrame,
    x: str,
    y: str,
    color: str | None,
    grid_x: int = SCATTER_GRID_X,
    grid_y: int = SCATTER_GRID_Y,
) -> pd.DataFrame:
    sampled = subset.copy()
    x_min = sampled[x].min()
    x_max = sampled[x].max()
    y_min = sampled[y].min()
    y_max = sampled[y].max()

    if x_min == x_max or y_min == y_max:
        sampled["count"] = 1
        return sampled

    sampled["_x_bin"] = (
        ((sampled[x] - x_min) / (x_max - x_min) * max(grid_x - 1, 1))
        .clip(lower=0, upper=max(grid_x - 1, 1))
        .round()
        .astype(int)
    )
    sampled["_y_bin"] = (
        ((sampled[y] - y_min) / (y_max - y_min) * max(grid_y - 1, 1))
        .clip(lower=0, upper=max(grid_y - 1, 1))
        .round()
        .astype(int)
    )

    group_columns = ["_x_bin", "_y_bin"] + ([color] if color else [])
    aggregated = (
        sampled.groupby(group_columns, dropna=False)
        .agg(**{x: (x, "mean"), y: (y, "mean"), "count": (y, "size")})
        .reset_index()
    )
    return aggregated.drop(columns=["_x_bin", "_y_bin"], errors="ignore")


def _build_histogram(df: pd.DataFrame, x: str, bins: int) -> dict[str, Any]:
    series = pd.to_numeric(df[x], errors="coerce").dropna()
    counts, _ = pd.cut(series, bins=bins, retbins=True, include_lowest=True)
    result = (
        pd.DataFrame({"bucket": counts.astype(str)})
        .value_counts()
        .reset_index(name="count")
        .sort_values("bucket")
    )
    return {"type": "histogram", "x": x, "bins": int(bins), "rows": result.to_dict(orient="records")}


def _build_scatter(df: pd.DataFrame, dataset: str, x: str, y: str, color: str | None) -> dict[str, Any]:
    columns = [x, y] + ([color] if color else [])
    subset = df[columns].copy()
    subset[y] = pd.to_numeric(subset[y], errors="coerce")
    x_type = _field_type(dataset, x)
    if x_type == "datetime" or pd.api.types.is_datetime64_any_dtype(df[x]):
        subset[x] = pd.to_datetime(subset[x], errors="coerce")
        x_type = "datetime"
    else:
        subset[x] = pd.to_numeric(subset[x], errors="coerce")
        x_type = "number"
    subset = subset.dropna(subset=[x, y])
    if color:
        subset[color] = subset[color].astype(str)

    source_row_count = int(len(subset))
    downsampled = False
    mode = "raw"
    if x_type == "number" and source_row_count > SCATTER_DOWNSAMPLE_THRESHOLD:
        subset = _downsample_scatter_numeric(subset, x=x, y=y, color=color)
        downsampled = True
        mode = "binned"
    else:
        subset["count"] = 1

    subset = subset.where(pd.notna(subset), None)
    return {
        "type": "scatter",
        "x": x,
        "y": y,
        "color": color,
        "x_type": x_type,
        "mode": mode,
        "downsampled": downsampled,
        "source_row_count": source_row_count,
        "rendered_row_count": int(len(subset)),
        "rows": _serialize_rows(subset),
    }


def _aggregate_metric(grouped: pd.core.groupby.generic.SeriesGroupBy, metric: str) -> pd.Series:
    if metric == "count":
        return grouped.count()
    if metric == "mean":
        return grouped.mean()
    if metric == "median":
        return grouped.median()
    if metric == "sum":
        return grouped.sum()
    raise HTTPException(status_code=400, detail=f"Unsupported metric: {metric}")

def _build_bar(df: pd.DataFrame, group_by: str, y: str | None, metric: str) -> dict[str, Any]:
    grouped_df = df.copy()
    grouped_df[group_by] = grouped_df[group_by].astype(str)
    if metric == "count":
        result = grouped_df.groupby(group_by).size().reset_index(name="value")
    else:
        if not y:
            raise HTTPException(status_code=400, detail="y is required for this metric")
        grouped_df[y] = pd.to_numeric(grouped_df[y], errors="coerce")
        result = _aggregate_metric(grouped_df.groupby(group_by)[y], metric).reset_index(name="value")
    return {"type": "bar", "group_by": group_by, "metric": metric, "y": y, "rows": result.to_dict(orient="records")}


def _build_boxplot(df: pd.DataFrame, group_by: str, y: str) -> dict[str, Any]:
    subset = df[[group_by, y]].copy()
    subset[group_by] = subset[group_by].astype(str)
    subset[y] = pd.to_numeric(subset[y], errors="coerce")
    subset = subset.dropna(subset=[y])
    rows = []
    for key, group in subset.groupby(group_by):
        stats = group[y].describe(percentiles=[0.25, 0.5, 0.75])
        rows.append(
            {
                group_by: key,
                "count": int(stats["count"]),
                "min": float(stats["min"]),
                "q1": float(stats["25%"]),
                "median": float(stats["50%"]),
                "q3": float(stats["75%"]),
                "max": float(stats["max"]),
            }
        )
    return {"type": "boxplot", "group_by": group_by, "y": y, "rows": rows}


def _build_line(df: pd.DataFrame, dataset: str, x: str, y: str, color: str | None) -> dict[str, Any]:
    columns = [x, y] + ([color] if color else [])
    subset = df[columns].copy()
    subset[y] = pd.to_numeric(subset[y], errors="coerce")
    subset = subset.dropna(subset=[y])

    x_type = _field_type(dataset, x)
    if x_type == "datetime" or pd.api.types.is_datetime64_any_dtype(df[x]):
        subset[x] = pd.to_datetime(subset[x], errors="coerce")
        x_type = "datetime"
    else:
        subset[x] = pd.to_numeric(subset[x], errors="coerce")
        x_type = "number"

    subset = subset.dropna(subset=[x])
    subset = subset.sort_values(x)
    if color:
        subset[color] = subset[color].astype(str)
    return {"type": "line", "x": x, "y": y, "color": color, "x_type": x_type, "rows": _serialize_rows(subset)}


def _build_pie(df: pd.DataFrame, group_by: str, y: str | None, metric: str) -> dict[str, Any]:
    grouped_df = df.copy()
    grouped_df[group_by] = grouped_df[group_by].astype(str)
    if metric == "count" or not y:
        result = grouped_df.groupby(group_by).size().reset_index(name="value")
    else:
        grouped_df[y] = pd.to_numeric(grouped_df[y], errors="coerce")
        result = _aggregate_metric(grouped_df.groupby(group_by)[y], metric).reset_index(name="value")
    return {"type": "pie", "group_by": group_by, "metric": metric, "y": y, "rows": result.to_dict(orient="records")}


def _build_heatmap(df: pd.DataFrame, x: str, y: str, bins: int) -> dict[str, Any]:
    subset = df[[x, y]].copy()
    subset[x] = pd.to_numeric(subset[x], errors="coerce")
    subset[y] = pd.to_numeric(subset[y], errors="coerce")
    subset = subset.dropna(subset=[x, y])
    if subset.empty:
        return {"type": "heatmap", "x": x, "y": y, "bins": bins, "rows": []}

    subset["x_bin"] = pd.cut(subset[x], bins=bins, include_lowest=True).astype(str)
    subset["y_bin"] = pd.cut(subset[y], bins=bins, include_lowest=True).astype(str)
    result = subset.groupby(["x_bin", "y_bin"]).size().reset_index(name="value")
    return {
        "type": "heatmap",
        "x": x,
        "y": y,
        "bins": bins,
        "source_row_count": int(len(subset)),
        "rendered_row_count": int(len(result)),
        "rows": result.to_dict(orient="records"),
    }


def _build_table(df: pd.DataFrame, limit: int) -> dict[str, Any]:
    preview = df.head(max(1, min(limit, MAX_PREVIEW_ROWS))).copy()
    return {"type": "table", "rows": _serialize_rows(preview)}


def list_datasets() -> dict[str, Any]:
    datasets = []
    for info in _dataset_catalog().values():
        datasets.append(
            {
                "id": info.id,
                "label": info.label,
                "asset": info.asset,
                "timeframe": info.timeframe,
                "source": info.source,
                "row_count": info.row_count,
                "child_timeframe": info.child_timeframe,
            }
        )
    return {"datasets": datasets}


def get_dataset_features(dataset: str) -> dict[str, Any]:
    return get_feature_response(dataset)


def preview_dataset(dataset: str, filters: list[Any], limit: int) -> dict[str, Any]:
    df = _load_query_dataset(dataset)
    filtered = apply_filters(df, filters, dataset)
    return {
        "dataset": dataset,
        "total_rows": int(len(df)),
        "filtered_rows": int(len(filtered)),
        "rows": _serialize_preview(filtered, limit),
    }


def build_chart(
    dataset: str,
    filters: list[Any],
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    group_by: str | None = None,
    metric: str | None = "count",
    bins: int | None = 20,
    limit: int | None = 100,
) -> dict[str, Any]:
    df = _load_query_dataset(dataset)
    filtered = apply_filters(df, filters, dataset)

    if chart_type == "histogram":
        if not x:
            raise HTTPException(status_code=400, detail="x is required for histogram")
        chart = _build_histogram(filtered, x, bins or 20)
    elif chart_type == "scatter":
        if not x or not y:
            raise HTTPException(status_code=400, detail="x and y are required for scatter")
        chart = _build_scatter(filtered, dataset, x, y, color)
    elif chart_type == "bar":
        if not group_by:
            raise HTTPException(status_code=400, detail="group_by is required for bar")
        chart = _build_bar(filtered, group_by, y, metric or "count")
    elif chart_type == "boxplot":
        if not group_by or not y:
            raise HTTPException(status_code=400, detail="group_by and y are required for boxplot")
        chart = _build_boxplot(filtered, group_by, y)
    elif chart_type == "line":
        if not x or not y:
            raise HTTPException(status_code=400, detail="x and y are required for line")
        chart = _build_line(filtered, dataset, x, y, color)
    elif chart_type == "pie":
        if not group_by:
            raise HTTPException(status_code=400, detail="group_by is required for pie")
        chart = _build_pie(filtered, group_by, y, metric or "count")
    elif chart_type == "heatmap":
        if not x or not y:
            raise HTTPException(status_code=400, detail="x and y are required for heatmap")
        chart = _build_heatmap(filtered, x, y, bins or 12)
    elif chart_type == "table":
        chart = _build_table(filtered, limit or 100)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported chart_type: {chart_type}")

    return {
        "dataset": dataset,
        "total_rows": int(len(df)),
        "filtered_rows": int(len(filtered)),
        "chart": chart,
    }


_PANDAS_ENGINE = PandasDashboardQueryEngine(
    list_datasets_fn=list_datasets,
    get_dataset_features_fn=get_dataset_features,
    preview_dataset_fn=preview_dataset,
    build_chart_fn=build_chart,
)


def get_query_engine():
    configured = get_configured_query_engine()
    if configured == "duckdb":
        return DuckDBDashboardQueryEngine()
    return _PANDAS_ENGINE
