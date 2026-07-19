from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from ..services.data_access import DataAccessService


def _parse_columns(columns: str) -> list[str] | None:
    parsed = [item.strip() for item in columns.split(",") if item.strip()]
    return parsed or None


def _coerce_filter_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        items = [item.strip() for item in raw[1:-1].split(",") if item.strip()]
        return [_coerce_filter_value(item.strip("\"'")) for item in items]
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip("\"'")


def _parse_filters(filters: str) -> list[dict[str, Any]]:
    if not filters.strip():
        return []

    operators = (" contains ", " in ", ">=", "<=", "==", "!=", ">", "<")
    parsed_filters: list[dict[str, Any]] = []

    for chunk in [item.strip() for item in filters.split(";") if item.strip()]:
        matched = False
        for operator in operators:
            if operator.strip() in {"contains", "in"}:
                if operator in chunk:
                    column, value = chunk.split(operator, 1)
                    parsed_filters.append(
                        {
                            "column": column.strip(),
                            "operator": operator.strip(),
                            "value": _coerce_filter_value(value.strip()),
                        }
                    )
                    matched = True
                    break
            else:
                if operator in chunk:
                    column, value = chunk.split(operator, 1)
                    parsed_filters.append(
                        {
                            "column": column.strip(),
                            "operator": operator,
                            "value": _coerce_filter_value(value.strip()),
                        }
                    )
                    matched = True
                    break
        if not matched:
            parsed_filters.append(
                {
                    "column": "",
                    "operator": "",
                    "value": chunk,
                }
            )
    return parsed_filters


@tool
def filter_frame(
    timeframe: str,
    filters: str,
    columns: str = "",
    row_cap: int = 2000,
    preview_rows: int = 5,
    asset: str = "btc",
) -> str:
    """Filter one timeframe analysis frame using semicolon-separated conditions."""
    result = DataAccessService().filter_analysis_frame(
        timeframe=timeframe,
        asset=asset,
        filters=_parse_filters(filters),
        columns=_parse_columns(columns),
        row_cap=row_cap,
        preview_rows=preview_rows,
    )
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)
