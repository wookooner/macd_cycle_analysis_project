from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.backtest.types import Costs, EntrySignal, ExitPolicy, ExitPolicyKind, MarketContext, SizingPolicy


def _direction_return(entry: float, exit_price: float, direction: str) -> float:
    if not entry or not np.isfinite(entry) or not np.isfinite(exit_price):
        return np.nan
    if str(direction) == "long":
        return (exit_price / entry - 1.0) * 100.0
    return (entry / exit_price - 1.0) * 100.0


def _first_hit(mask: np.ndarray) -> int | None:
    if len(mask) == 0 or not mask.any():
        return None
    return int(np.argmax(mask))


def _window_arrays(candles: pd.DataFrame, idx: int, end_time: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = candles["timestamp"].to_numpy()
    end_pos = int(np.searchsorted(times.astype("datetime64[ns]"), np.datetime64(end_time), side="right"))
    start = min(idx + 1, len(candles))
    end = max(start + 1, min(len(candles), end_pos))
    segment = candles.iloc[start:end]
    return (
        segment["timestamp"].to_numpy(),
        segment["high"].to_numpy(dtype="float64"),
        segment["low"].to_numpy(dtype="float64"),
        segment["close"].to_numpy(dtype="float64"),
    )


def _window_32_mfe_mae(candles: pd.DataFrame, idx: int, entry: float, direction: str) -> tuple[float, float]:
    segment = candles.iloc[idx + 1 : min(len(candles), idx + 33)]
    if segment.empty:
        return np.nan, np.nan
    if direction == "long":
        return (
            float((segment["high"].max() / entry - 1.0) * 100.0),
            float((segment["low"].min() / entry - 1.0) * 100.0),
        )
    return (
        float((entry / segment["low"].min() - 1.0) * 100.0),
        float((entry / segment["high"].max() - 1.0) * 100.0),
    )


def simulate_trades(
    entry_signal_stream: Iterable[EntrySignal],
    entry_policy: Any,
    exit_policy: ExitPolicy,
    sizing_policy: SizingPolicy,
    costs: Costs,
    market_context: MarketContext,
) -> pd.DataFrame:
    """Simulate a stream of already-materialized entry signals into a trade ledger.

    Entry variants such as immediate/wait/confirmed are represented on the signal
    itself in v1, so this function owns realized exit path accounting.
    """

    rows: list[dict[str, Any]] = []
    for signal in entry_signal_stream:
        tf = signal.entry_tf
        direction = str(signal.direction)
        candles = market_context.candles[tf]
        idx = int(signal.bar_index)
        entry = float(signal.entry_price)
        fallback_time, fallback_price = market_context.opposite_event_resolver(tf, direction, signal.entry_time)
        if fallback_time is None or fallback_price is None or pd.isna(fallback_time):
            end_idx = min(len(candles) - 1, idx + 32)
            fallback_time = pd.Timestamp(candles.iloc[end_idx]["timestamp"])
            fallback_price = float(candles.iloc[end_idx]["close"])

        times, highs, lows, closes = _window_arrays(candles, idx, pd.Timestamp(fallback_time))
        if len(closes) == 0:
            continue
        fav = (highs / entry - 1.0) * 100.0 if direction == "long" else (entry / lows - 1.0) * 100.0
        adv = (lows / entry - 1.0) * 100.0 if direction == "long" else (entry / highs - 1.0) * 100.0

        gross = _direction_return(entry, float(fallback_price), direction)
        exit_idx = len(closes) - 1
        exit_time = pd.Timestamp(fallback_time)
        exit_reason = "opposite_true_reversal"
        first_leg_hit = False
        first_leg_return = 0.0
        remaining_leg_return = gross

        if exit_policy.kind == ExitPolicyKind.TP_SL:
            tp = float(exit_policy.params["tp_pct"])
            sl = float(exit_policy.params["sl_pct"])
            hit = _first_hit((adv <= -sl) | (fav >= tp))
            if hit is not None:
                exit_idx = hit
                exit_time = pd.Timestamp(times[hit])
                if adv[hit] <= -sl:
                    gross = -sl
                    exit_reason = "stop_loss"
                else:
                    gross = tp
                    exit_reason = "take_profit"
        elif exit_policy.kind == ExitPolicyKind.FIXED_SL:
            sl = float(exit_policy.params["sl_pct"])
            hit = _first_hit(adv <= -sl)
            if hit is not None:
                exit_idx = hit
                exit_time = pd.Timestamp(times[hit])
                gross = -sl
                exit_reason = "stop_loss"
        elif exit_policy.kind == ExitPolicyKind.PARTIAL_TP:
            tp = float(exit_policy.params["tp_pct"])
            ratio = float(exit_policy.params.get("partial_ratio", 0.5))
            hit = _first_hit(fav >= tp)
            if hit is not None:
                first_leg_hit = True
                first_leg_return = tp
                remaining_leg_return = gross
                gross = ratio * first_leg_return + (1.0 - ratio) * remaining_leg_return
                exit_reason = f"partial_tp_{tp}_then_opposite"
        elif exit_policy.kind == ExitPolicyKind.STRUCTURAL_INVALIDATION:
            invalid_low = invalid_high = None
            if market_context.structural_levels is not None:
                invalid_low, invalid_high = market_context.structural_levels(signal, exit_policy)
            if direction == "long" and invalid_low is not None:
                hit = _first_hit(lows <= invalid_low)
            elif direction == "short" and invalid_high is not None:
                hit = _first_hit(highs >= invalid_high)
            else:
                hit = None
            if hit is not None:
                exit_idx = hit
                exit_time = pd.Timestamp(times[hit])
                gross = _direction_return(entry, float(closes[hit]), direction)
                exit_reason = exit_policy.name

        exit_price = entry * (1.0 + gross / 100.0) if direction == "long" else entry / (1.0 + gross / 100.0)
        metric_end = exit_idx + 1 if exit_reason != "opposite_true_reversal" else len(fav)
        mfe32, mae32 = _window_32_mfe_mae(candles, idx, entry, direction)
        holding_hours = (exit_time - signal.entry_time).total_seconds() / 3600.0
        size = sizing_policy.fixed_size * signal.size_hint
        gross_sized = gross * size
        cost = costs.round_trip_pct * abs(size)

        rows.append(
            {
                "rule_name": signal.rule_name,
                "rule_kind": str(signal.rule_kind),
                "entry_signal_time": signal.signal_time,
                "entry_time": signal.entry_time,
                "exit_time": exit_time,
                "entry_price": entry,
                "exit_price": exit_price,
                "direction": direction,
                "entry_tf": tf,
                "size": size,
                "fee": costs.fee_pct * abs(size),
                "slippage": costs.slippage_pct_per_side * 2.0 * abs(size),
                "gross_return": gross_sized,
                "net_return": gross_sized - cost,
                "exit_policy": exit_policy.name,
                "exit_reason": exit_reason,
                "mfe_until_exit": float(np.nanmax(fav[:metric_end])),
                "mae_until_exit": float(np.nanmin(adv[:metric_end])),
                "holding_bars": int(metric_end),
                "holding_hours": holding_hours,
                "mfe_window_32bars": mfe32,
                "mae_window_32bars": mae32,
                "entry_delay_bars": signal.entry_delay_bars,
                "missed_move_before_entry": signal.missed_move_before_entry,
                "first_leg_hit": first_leg_hit,
                "first_leg_return": first_leg_return,
                "remaining_leg_return": remaining_leg_return,
                "metric_basis": "realized_with_cost",
                **{f"meta_{k}": v for k, v in signal.metadata.items() if isinstance(k, str)},
            }
        )
    return pd.DataFrame(rows)
