"""
Central schema definitions for market_state.

Keep this module as the single source of truth for:
- timeframe ordering
- analysis column lists
- top-level market_state keys
- cycle payload keys

This makes it easier to extend the data contract without scattering
hard-coded field names across the bot.
"""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_TIMEFRAME_ORDER = ["1w", "1d", "4h", "1h"]

ANALYSIS_COLUMNS = [
    "taker_buy_base",
    "volume_delta",
    "cvd",
    "cvd_rolling",
    "ppo",
    "ppo_signal",
    "ppo_hist",
    "delta",
    "ma_7",
    "ma_25",
    "ma_99",
    "oi",
    "oi_usd",
    "funding_rate",
]

TOP_LEVEL_KEYS = [
    "timestamp",
    "price",
    "timeframes",
    "chain",
    "cycle_context",
    "current_position",
    "recent_signals",
]

TIMEFRAME_STATE_KEYS = [
    "cycle_id",
    "cycle_type",
    "duration",
    "start_date",
    "end_date",
    "start_rsi",
    "start_macd",
    "start_hist",
    "start_price",
    "noise_count",
    "position_pct",
    "is_current",
    "analysis_snapshot",
]

CYCLE_PAYLOAD_KEYS = [
    "timeframe",
    "cycle_id",
    "cycle_type",
    "start_date",
    "end_date",
    "duration_candles",
    "is_current",
    "latest_close",
    "latest_timestamp",
    "latest_candle",
    "parent_cycle_ids",
    "child_cycle_ids",
    "cycle_features",
    "candle_count",
    "candle_data",
    "analysis_fields",
    "analysis_latest",
    "analysis_rows",
]


@dataclass(slots=True)
class MarketStateSchema:
    timeframe_order: list[str] = field(default_factory=lambda: list(DEFAULT_TIMEFRAME_ORDER))
    analysis_columns: list[str] = field(default_factory=lambda: list(ANALYSIS_COLUMNS))
    top_level_keys: list[str] = field(default_factory=lambda: list(TOP_LEVEL_KEYS))
    timeframe_state_keys: list[str] = field(default_factory=lambda: list(TIMEFRAME_STATE_KEYS))
    cycle_payload_keys: list[str] = field(default_factory=lambda: list(CYCLE_PAYLOAD_KEYS))


DEFAULT_SCHEMA = MarketStateSchema()
