"""Shared backtesting primitives for cycle/PPO research and live parity checks."""

from src.backtest.metrics import compute_metrics
from src.backtest.runner import run_backtest_grid
from src.backtest.simulator import simulate_trades
from src.backtest.types import (
    BacktestResults,
    Costs,
    EntryPolicy,
    EntrySignal,
    ExitPolicy,
    MarketContext,
    MetricsRow,
    SizingPolicy,
)

__all__ = [
    "BacktestResults",
    "Costs",
    "EntryPolicy",
    "EntrySignal",
    "ExitPolicy",
    "MarketContext",
    "MetricsRow",
    "SizingPolicy",
    "compute_metrics",
    "run_backtest_grid",
    "simulate_trades",
]
