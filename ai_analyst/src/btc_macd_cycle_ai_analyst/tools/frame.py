from __future__ import annotations

import json

from langchain_core.tools import tool

from ..services.data_access import DataAccessService


def _parse_columns(columns: str) -> list[str] | None:
    parsed = [item.strip() for item in columns.split(",") if item.strip()]
    return parsed or None


@tool
def build_analysis_frame(
    timeframe: str,
    columns: str = "",
    row_cap: int = 2000,
    preview_rows: int = 5,
    asset: str = "btc",
) -> str:
    """Build a flattened analysis-frame preview for one cycle timeframe."""
    result = DataAccessService().build_analysis_frame(
        timeframe=timeframe,
        asset=asset,
        columns=_parse_columns(columns),
        row_cap=row_cap,
        preview_rows=preview_rows,
    )
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)
