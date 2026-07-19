from __future__ import annotations

from typing import Any


def create_collector(asset: str) -> Any:
    """Create the active market collector from a single, extensible registry."""
    normalized = asset.strip().lower()
    if normalized == "btc":
        from data_pipeline.collectors.binance_market import AdvancedBTCDataCollectorV2

        return AdvancedBTCDataCollectorV2()
    if normalized == "gold":
        from data_pipeline.collectors.gold import GoldDataCollector

        return GoldDataCollector()
    raise ValueError(f"No collector registered for asset={asset!r}")
