from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class ConsistencyReport:
    start: pd.Timestamp
    end: pd.Timestamp
    compared: int
    mismatches: int
    mismatch_rate: float
    passed: bool
    details: pd.DataFrame


def verify_bot_backtest_consistency(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    backtest_decider: Callable[[pd.Timestamp], object] | None = None,
    bot_decider: Callable[[pd.Timestamp], object] | None = None,
    frequency: str = "1h",
    max_mismatch_rate: float = 0.01,
) -> ConsistencyReport:
    """Scaffold for future bot/backtest decision parity checks."""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    times = pd.date_range(start_ts, end_ts, freq=frequency)
    rows = []
    for ts in times:
        backtest_value = backtest_decider(ts) if backtest_decider else None
        bot_value = bot_decider(ts) if bot_decider else None
        rows.append({"timestamp": ts, "backtest_decision": backtest_value, "bot_decision": bot_value, "same": backtest_value == bot_value})
    details = pd.DataFrame(rows)
    compared = len(details)
    mismatches = int((~details["same"]).sum()) if compared else 0
    rate = mismatches / compared if compared else 0.0
    return ConsistencyReport(start_ts, end_ts, compared, mismatches, rate, rate <= max_mismatch_rate, details)
