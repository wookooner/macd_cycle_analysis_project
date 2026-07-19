"""Canonical Binance market-collector import path.

The implementation retains its historical module name during the migration so
external callers keep working. New code should import from this module.
"""

from data_pipeline.collectors.new_collcetor import AdvancedBTCDataCollectorV2

__all__ = ["AdvancedBTCDataCollectorV2"]
