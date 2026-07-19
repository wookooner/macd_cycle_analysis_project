from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from ..services.data_access import DataAccessService


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        items = [item.strip() for item in raw[1:-1].split(",") if item.strip()]
        return [_coerce_value(item.strip("\"'")) for item in items]
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip("\"'")


def _parse_filter_group(filters: str) -> list[dict[str, Any]]:
    if not filters.strip():
        return []

    operators = (" contains ", " in ", ">=", "<=", "==", "!=", ">", "<")
    parsed: list[dict[str, Any]] = []
    for chunk in [item.strip() for item in filters.split(";") if item.strip()]:
        matched = False
        for operator in operators:
            stripped = operator.strip()
            if stripped in {"contains", "in"}:
                if operator in chunk:
                    column, value = chunk.split(operator, 1)
                    parsed.append(
                        {
                            "column": column.strip(),
                            "operator": stripped,
                            "value": _coerce_value(value.strip()),
                        }
                    )
                    matched = True
                    break
            else:
                if operator in chunk:
                    column, value = chunk.split(operator, 1)
                    parsed.append(
                        {
                            "column": column.strip(),
                            "operator": operator,
                            "value": _coerce_value(value.strip()),
                        }
                    )
                    matched = True
                    break
        if not matched:
            parsed.append({"column": "", "operator": "", "value": chunk})
    return parsed


def _parse_metrics(metrics: str) -> list[str]:
    return [item.strip() for item in metrics.split(",") if item.strip()]


@tool
def analyze_feature_combinations(condition: str) -> str:
    """Return a placeholder summary for one feature-combination condition."""
    return (
        f"Condition '{condition}' analysis is still a placeholder.\n"
        "Use compare_groups for concrete two-group analysis.\n"
        "Use rank_features to compare one focused subset against the remainder of a timeframe.\n"
        "Use describe_available_data, build_analysis_frame, and filter_frame to confirm datasets, columns, and subsets before broader feature-combination workflows."
    )


@tool
def compare_groups(
    timeframe: str,
    metrics: str,
    group_a_filters: str,
    group_b_filters: str,
    row_cap: int = 5000,
    preview_rows: int = 5,
    asset: str = "btc",
) -> str:
    """Compare two filtered groups on one or more metric columns."""
    result = DataAccessService().compare_groups(
        timeframe=timeframe,
        asset=asset,
        group_a_filters=_parse_filter_group(group_a_filters),
        group_b_filters=_parse_filter_group(group_b_filters),
        metrics_columns=_parse_metrics(metrics),
        row_cap=row_cap,
        preview_rows=preview_rows,
    )
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


@tool
def rank_features(
    timeframe: str,
    focus_filters: str,
    candidate_columns: str = "",
    top_k: int = 10,
    row_cap: int = 5000,
    preview_rows: int = 5,
    asset: str = "btc",
) -> str:
    """Rank feature columns by how strongly a focused subset differs from the remainder."""
    result = DataAccessService().rank_features(
        timeframe=timeframe,
        asset=asset,
        focus_filters=_parse_filter_group(focus_filters),
        candidate_columns=_parse_metrics(candidate_columns) or None,
        top_k=top_k,
        row_cap=row_cap,
        preview_rows=preview_rows,
    )
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)
