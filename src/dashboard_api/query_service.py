import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException

from src.common.paths import PROJECT_PATHS
from src.dashboard_api.query_engine import (
    DuckDBDashboardQueryEngine,
    PandasDashboardQueryEngine,
    get_configured_query_engine,
)


DASHBOARD_DATA_DIR = PROJECT_PATHS.dashboard_root / "candles"
DASHBOARD_META_DIR = PROJECT_PATHS.dashboard_root / "meta"
MAX_PREVIEW_ROWS = 200
TIMEFRAME_ORDER = ("1m", "1h", "4h", "1d", "1w")
SCATTER_DOWNSAMPLE_THRESHOLD = 2400
SCATTER_GRID_X = 120
SCATTER_GRID_Y = 72

_DATAFRAME_CACHE: dict[str, pd.DataFrame] = {}
_META_CACHE: dict[str, dict[str, Any]] = {}
_QUERY_DATAFRAME_CACHE: dict[str, pd.DataFrame] = {}
_FEATURE_RESPONSE_CACHE: dict[str, dict[str, Any]] = {}


def _dataset_path(dataset: str) -> Path:
    return DASHBOARD_DATA_DIR / f"{dataset}.parquet"


def _meta_path(dataset: str) -> Path:
    return DASHBOARD_META_DIR / f"{dataset}_features.json"


def _load_dataset(dataset: str) -> pd.DataFrame:
    if dataset in _DATAFRAME_CACHE:
        return _DATAFRAME_CACHE[dataset].copy()

    path = _dataset_path(dataset)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")

    df = pd.read_parquet(path)
    _DATAFRAME_CACHE[dataset] = df
    return df.copy()


def _load_meta(dataset: str) -> dict[str, Any]:
    if dataset in _META_CACHE:
        return _META_CACHE[dataset]

    path = _meta_path(dataset)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Metadata not found: {dataset}")

    meta = json.loads(path.read_text(encoding="utf-8"))
    _META_CACHE[dataset] = meta
    return meta


def _parse_dataset_name(dataset: str) -> tuple[str, str]:
    if "_" not in dataset:
        raise HTTPException(status_code=400, detail=f"Invalid dataset name: {dataset}")
    return dataset.split("_", 1)


def _parent_timeframes(timeframe: str) -> list[str]:
    if timeframe not in TIMEFRAME_ORDER:
        return []
    idx = TIMEFRAME_ORDER.index(timeframe)
    return [tf for tf in TIMEFRAME_ORDER[idx + 1 :] if tf in {"1h", "4h", "1d", "1w"}]


def _relation_prefix(parent_timeframe: str) -> str:
    return f"parent_{parent_timeframe}__"


def _build_relation_fields(dataset: str) -> list[dict[str, Any]]:
    asset, timeframe = _parse_dataset_name(dataset)
    relation_fields: list[dict[str, Any]] = []

    for parent_timeframe in _parent_timeframes(timeframe):
        parent_dataset = f"{asset}_{parent_timeframe}"
        parent_meta_path = _meta_path(parent_dataset)
        if not parent_meta_path.exists():
            continue

        parent_meta = _load_meta(parent_dataset)
        prefix = _relation_prefix(parent_timeframe)
        for item in parent_meta.get("fields", []):
            cloned = dict(item)
            cloned["field"] = f"{prefix}{item['field']}"
            cloned["label"] = f"{parent_timeframe} / {item['label']}"
            cloned["field_group"] = "parent_cycle"
            cloned["category"] = f"parent_{parent_timeframe}"
            cloned["relation_scope"] = "parent"
            cloned["relation_timeframe"] = parent_timeframe
            cloned["source_dataset"] = parent_dataset
            relation_fields.append(cloned)

    return relation_fields


def get_feature_response(dataset: str) -> dict[str, Any]:
    if dataset in _FEATURE_RESPONSE_CACHE:
        return _FEATURE_RESPONSE_CACHE[dataset]

    meta = dict(_load_meta(dataset))
    fields = list(meta.get("fields", []))
    relation_fields = _build_relation_fields(dataset)
    payload = {
        **meta,
        "fields": fields + relation_fields,
        "field_count": len(fields) + len(relation_fields),
        "available_parent_timeframes": [item["relation_timeframe"] for item in relation_fields if item.get("field", "").endswith("__cycle_id")],
    }
    _FEATURE_RESPONSE_CACHE[dataset] = payload
    return payload


def _field_meta_map(dataset: str) -> dict[str, dict[str, Any]]:
    meta = get_feature_response(dataset)
    return {item["field"]: item for item in meta.get("fields", [])}


def _field_type(dataset: str, field: str) -> str:
    return _field_meta_map(dataset).get(field, {}).get("data_type", "string")


def _load_query_dataset(dataset: str) -> pd.DataFrame:
    if dataset in _QUERY_DATAFRAME_CACHE:
        return _QUERY_DATAFRAME_CACHE[dataset].copy()

    base_df = _load_dataset(dataset)
    asset, timeframe = _parse_dataset_name(dataset)
    joined = base_df.copy()

    for parent_timeframe in _parent_timeframes(timeframe):
        link_field = f"struct_parent_{parent_timeframe}_cycle_id"
        if link_field not in joined.columns:
            continue

        parent_dataset = f"{asset}_{parent_timeframe}"
        parent_data_path = _dataset_path(parent_dataset)
        if not parent_data_path.exists():
            continue

        parent_df = _load_dataset(parent_dataset).copy()
        prefix = _relation_prefix(parent_timeframe)
        parent_df = parent_df.rename(columns={column: f"{prefix}{column}" for column in parent_df.columns})
        joined = joined.merge(
            parent_df,
            how="left",
            left_on=link_field,
            right_on=f"{prefix}cycle_id",
        )

    _QUERY_DATAFRAME_CACHE[dataset] = joined
    return joined.copy()


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


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

    if data_type in {"string", "boolean"}:
        comparable = series.astype(str) if data_type == "string" else series
        if op == "in":
            values = [str(item) for item in value] if data_type == "string" else value
            mask = comparable.isin(values)
        elif op == "eq":
            mask = comparable == (str(value) if data_type == "string" else value)
        elif op == "neq":
            mask = comparable != (str(value) if data_type == "string" else value)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported categorical op: {op}")
        return df[mask]

    return df


def apply_filters(df: pd.DataFrame, filters: list[Any], dataset: str) -> pd.DataFrame:
    field_meta = _field_meta_map(dataset)
    filtered = df
    for condition in filters:
        filtered = _apply_filter(filtered, condition, field_meta)
    return filtered


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
        .agg(
            **{
                x: (x, "mean"),
                y: (y, "mean"),
                "count": (y, "size"),
            }
        )
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
    datasets = sorted(path.stem for path in DASHBOARD_DATA_DIR.glob("*.parquet"))
    return {"datasets": datasets}


def get_dataset_features(dataset: str) -> dict[str, Any]:
    payload = get_feature_response(dataset)
    available_parent_timeframes = sorted(
        {
            item.get("relation_timeframe")
            for item in payload.get("fields", [])
            if item.get("relation_scope") == "parent" and item.get("relation_timeframe")
        },
        key=lambda timeframe: TIMEFRAME_ORDER.index(timeframe) if timeframe in TIMEFRAME_ORDER else 999,
    )
    return {
        **payload,
        "available_parent_timeframes": available_parent_timeframes,
    }


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
