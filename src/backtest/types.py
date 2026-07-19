from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

import pandas as pd


class RuleKind(StrEnum):
    PRODUCTION = "production"
    PROXY = "proxy"
    RESEARCH_ONLY = "research_only"


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"


class EntryPolicyKind(StrEnum):
    IMMEDIATE = "immediate"
    WAIT_N_BARS = "wait_n_bars"
    CONFIRMED_AFTER_EVENT = "confirmed_after_event"


class ExitPolicyKind(StrEnum):
    OPPOSITE_TRUE = "opposite_true"
    PARTIAL_TP = "partial_tp"
    TP_SL = "tp_sl"
    FIXED_SL = "fixed_sl"
    ATR_STOP = "atr_stop"
    STRUCTURAL_INVALIDATION = "structural_invalidation"


@dataclass(frozen=True)
class Costs:
    fee_pct: float = 0.08
    slippage_pct_per_side: float = 0.02

    @property
    def round_trip_pct(self) -> float:
        return self.fee_pct + self.slippage_pct_per_side * 2.0


@dataclass(frozen=True)
class EntryPolicy:
    kind: EntryPolicyKind = EntryPolicyKind.IMMEDIATE
    wait_bars: int = 0
    confirm_within_bars: int | None = None


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    kind: ExitPolicyKind
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SizingPolicy:
    fixed_size: float = 1.0


@dataclass(frozen=True)
class EntrySignal:
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    direction: Direction | str
    entry_tf: str
    bar_index: int
    rule_name: str
    rule_kind: RuleKind | str = RuleKind.RESEARCH_ONLY
    size_hint: float = 1.0
    entry_delay_bars: int = 0
    missed_move_before_entry: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketContext:
    candles: dict[str, pd.DataFrame]
    opposite_event_resolver: Callable[[str, str, pd.Timestamp], tuple[pd.Timestamp | None, float | None]]
    structural_levels: Callable[[EntrySignal, ExitPolicy], tuple[float | None, float | None]] | None = None


@dataclass(frozen=True)
class BasisConfig:
    metric_basis: str = "realized_with_cost"
    unit_basis: str = "pct"
    sample_low_n: int = 100
    sample_reliable_n: int = 1000


@dataclass
class MetricsRow:
    values: dict[str, Any]


@dataclass
class BacktestResults:
    ledger: pd.DataFrame
    comparison: pd.DataFrame
    reliable: pd.DataFrame
    medium: pd.DataFrame
    low_sample: pd.DataFrame
    manifest: list[dict[str, Any]]
    validation: dict[str, Any]
    output_dir: Path | None = None
