from __future__ import annotations

import json

from langchain_core.tools import tool

from ..services.data_access import DataAccessService


@tool
def describe_available_data(asset: str = "btc") -> str:
    """Describe canonical cycle and context datasets available for one asset."""
    result = DataAccessService().describe_available_data(asset=asset)
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)
