from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.ppo_reversal_candidate_backtest import (  # noqa: E402
    EXTREME_THRESHOLDS,
    LOW_SAMPLE_N,
    MAX_TOLERANCE,
    PROJECT_PATHS,
    ROUND_TRIP_FEE_PCT,
    SLIPPAGE_PER_SIDE_PCT,
    TF_SECONDS,
    _candidate_sign,
    _direction_return,
    _raw_market_path,
    _read_timestamp,
    _tf_delta,
)


BASE_ANALYSIS_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_reversal_candidate_backtest"
OUT_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_reversal_practical_followup"
TIMEFRAMES = ("15m", "1h", "4h", "1d")
FULL_STRATEGIES = ("S3", "S7", "S5", "S2", "S9")
TP_SL_GRID = {
    "15m": {"tp": (0.8, 1.2, 1.8, 2.5), "sl": (0.4, 0.6, 0.8, 1.2)},
    "1h": {"tp": (1.5, 2.5, 4.0, 6.0), "sl": (0.8, 1.2, 1.8, 2.5)},
}
POSITION_COST_PCT = ROUND_TRIP_FEE_PCT + SLIPPAGE_PER_SIDE_PCT * 2


@dataclass(frozen=True)
class EntrySpec:
    name: str
    entry_tf: str
    exit_tf: str
    base: str


ENTRY_SPECS = (
    EntrySpec("S3_15m_1h_4h_alignment", "15m", "1h", "S3"),
    EntrySpec("S7_1h_4h_bias", "1h", "4h", "S7"),
    EntrySpec("S5_15m_4h_extreme", "15m", "1h", "S5"),
    EntrySpec("S2_15m_4h_bias", "15m", "1h", "S2"),
    EntrySpec("S9_4h_1d_bias", "4h", "1d", "S9"),
    EntrySpec("S3E1_15m_1h_extreme_4h_bias", "15m", "1h", "S3E1"),
    EntrySpec("S3E2_15m_1h_bias_4h_fixed_extreme", "15m", "1h", "S3E2"),
    EntrySpec("S7E1_1h_candidate_extreme_4h_bias", "1h", "4h", "S7E1"),
    EntrySpec("W1_15m_1h_4h_align_1bar_extreme", "15m", "1h", "W1"),
    EntrySpec("W2_15m_1h_4h_align_2bar_extreme", "15m", "1h", "W2"),
    EntrySpec("WTOL_15m_1h_4h_align_tolerance_extreme", "15m", "1h", "WTOL"),
)


def load_candidates() -> pd.DataFrame:
    path = BASE_ANALYSIS_DIR / "20_reversal_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing candidate file: {path}. Run ppo_reversal_candidate_backtest.py first.")
    usecols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "ppo",
        "ppo_hist",
        "ppo_hist_diff",
        "raw_direction",
        "bar_index",
        "candidate_run_length_afterward",
        "candidate_tf",
        "candidate_direction",
        "close_at_entry",
        "ppo_bin",
        "hist_bin",
        "cycle_type",
        "cycle_progress",
        "is_noise",
        "is_true_reversal",
        "upper_1h_ppo",
        "upper_1h_hist",
        "upper_1h_ppo_bin",
        "upper_1h_hist_bin",
        "upper_1h_align",
        "upper_4h_ppo",
        "upper_4h_hist",
        "upper_4h_ppo_bin",
        "upper_4h_hist_bin",
        "upper_4h_align",
        "upper_1d_ppo",
        "upper_1d_hist",
        "upper_1d_ppo_bin",
        "upper_1d_hist_bin",
        "upper_1d_align",
        "has_supportive_4h_extreme",
        "has_contra_4h_extreme",
        "align_count",
        "major_align_count",
    ]
    available = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[col for col in usecols if col in available], engine="python")
    df["timestamp"] = _read_timestamp(df["timestamp"])
    for col in ("bar_index", "raw_direction", "candidate_run_length_afterward", "is_noise", "is_true_reversal"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["candidate_tf", "timestamp"]).reset_index(drop=True)


def load_candles(timeframe: str) -> pd.DataFrame:
    path = _raw_market_path(timeframe)
    df = pd.read_csv(path, usecols=lambda col: col in {"date", "timestamp", "open_time", "open", "high", "low", "close", "ppo", "ppo_hist"})
    ts_col = next(col for col in ("timestamp", "open_time", "date") if col in df.columns)
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _read_timestamp(df["timestamp"])
    for col in ("open", "high", "low", "close", "ppo", "ppo_hist"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "ppo", "ppo_hist"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df), dtype=np.int64)
    df["ppo_hist_diff"] = df["ppo_hist"].diff()
    df["raw_direction"] = np.sign(df["ppo_hist_diff"]).replace(0, np.nan).ffill().fillna(0).astype("int8")
    return df


def is_bottom20(series: pd.Series) -> pd.Series:
    return series.isin(["bottom10", "bottom20"])


def is_top20(series: pd.Series) -> pd.Series:
    return series.isin(["top10", "top20"])


def direction_extreme(df: pd.DataFrame, ppo_bin: str, hist_bin: str, direction_col: str = "candidate_direction") -> pd.Series:
    return (
        df[direction_col].eq("long") & is_bottom20(df[ppo_bin]) & is_bottom20(df[hist_bin])
    ) | (
        df[direction_col].eq("short") & is_top20(df[ppo_bin]) & is_top20(df[hist_bin])
    )


def fixed_4h_extreme(df: pd.DataFrame) -> pd.Series:
    return (
        df["candidate_direction"].eq("long")
        & (pd.to_numeric(df["upper_4h_ppo"], errors="coerce") <= EXTREME_THRESHOLDS["4h"]["long"])
        & (pd.to_numeric(df["upper_4h_hist"], errors="coerce") < 0)
    ) | (
        df["candidate_direction"].eq("short")
        & (pd.to_numeric(df["upper_4h_ppo"], errors="coerce") >= EXTREME_THRESHOLDS["4h"]["short"])
        & (pd.to_numeric(df["upper_4h_hist"], errors="coerce") > 0)
    )


def entry_frame(candidates: pd.DataFrame, spec: EntrySpec, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = candidates[candidates["candidate_tf"].eq(spec.entry_tf)].copy()
    if spec.base == "S3":
        df = df[df["upper_1h_align"].eq(1) & df["upper_4h_align"].eq(1)]
    elif spec.base == "S7":
        df = df[df["upper_4h_align"].eq(1)]
    elif spec.base == "S5":
        df = df[df["has_supportive_4h_extreme"].eq(1)]
    elif spec.base == "S2":
        df = df[df["upper_4h_align"].eq(1)]
    elif spec.base == "S9":
        df = df[df["upper_1d_align"].eq(1)]
    elif spec.base == "S3E1":
        df = df[df["upper_4h_align"].eq(1) & direction_extreme(df, "upper_1h_ppo_bin", "upper_1h_hist_bin")]
    elif spec.base == "S3E2":
        df = df[df["upper_1h_align"].eq(1) & fixed_4h_extreme(df)]
    elif spec.base == "S7E1":
        df = df[df["upper_4h_align"].eq(1) & direction_extreme(df, "ppo_bin", "hist_bin")]
    elif spec.base in {"W1", "W2", "WTOL"}:
        df = df[df["upper_1h_align"].eq(1) & df["upper_4h_align"].eq(1)]
        own_extreme = direction_extreme(df, "ppo_bin", "hist_bin")
        upper_extreme = direction_extreme(df, "upper_1h_ppo_bin", "upper_1h_hist_bin")
        df = df[own_extreme | upper_extreme].copy()
        wait_bars = {"W1": 1, "W2": 2, "WTOL": MAX_TOLERANCE}.get(spec.base, 0)
        df = apply_wait_entry(df, candles[spec.entry_tf], wait_bars=wait_bars, require_tolerance=spec.base == "WTOL")
        return df
    return df.sort_values("timestamp").reset_index(drop=True)


def apply_wait_entry(entries: pd.DataFrame, candles: pd.DataFrame, wait_bars: int, require_tolerance: bool) -> pd.DataFrame:
    if entries.empty:
        return entries
    rows = []
    max_idx = len(candles) - 1
    for row in entries.itertuples(index=False):
        direction = _candidate_sign(row.candidate_direction)
        if require_tolerance and row.candidate_run_length_afterward <= MAX_TOLERANCE:
            continue
        entry_idx = int(row.bar_index) + wait_bars
        if entry_idx > max_idx:
            continue
        check = candles.iloc[int(row.bar_index) : entry_idx + 1]
        if len(check) < wait_bars + 1 or not (check["raw_direction"].to_numpy() == direction).all():
            continue
        item = row._asdict()
        entry_candle = candles.iloc[entry_idx]
        item["signal_time"] = item["timestamp"]
        item["timestamp"] = entry_candle["timestamp"]
        item["bar_index"] = int(entry_candle["bar_index"])
        item["close_at_entry"] = float(entry_candle["close"])
        item["entry_delay_bars"] = wait_bars
        item["entry_delay_hours"] = wait_bars * TF_SECONDS[item["candidate_tf"]] / 3600
        rows.append(item)
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def candidate_event_tables(candidates: pd.DataFrame) -> dict[tuple[str, str, bool], pd.DataFrame]:
    out = {}
    for tf in TIMEFRAMES:
        frame = candidates[candidates["candidate_tf"].eq(tf)].sort_values("timestamp").reset_index(drop=True)
        for direction in ("long", "short"):
            subset = frame[frame["candidate_direction"].eq(direction)].reset_index(drop=True)
            out[(tf, direction, False)] = subset
            out[(tf, direction, True)] = subset[subset["is_true_reversal"].eq(1)].reset_index(drop=True)
    return out


def next_event(events: pd.DataFrame, when: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    if events.empty:
        return None, None
    times = events["timestamp"].to_numpy(dtype="datetime64[ns]")
    idx = int(np.searchsorted(times, np.datetime64(when), side="right"))
    if idx >= len(events):
        return None, None
    row = events.iloc[idx]
    return row["timestamp"], float(row["close_at_entry"])


def tp_sl_before_exit(
    candles: pd.DataFrame,
    entry_bar: int,
    entry_price: float,
    direction: str,
    stop_time: pd.Timestamp | None,
    tp: float,
    sl: float,
) -> tuple[pd.Timestamp | None, float | None, str]:
    sign = _candidate_sign(direction)
    start = entry_bar + 1
    end = len(candles)
    if stop_time is not None:
        stop_pos = np.searchsorted(candles["timestamp"].to_numpy(dtype="datetime64[ns]"), np.datetime64(stop_time), side="right")
        end = min(end, int(stop_pos))
    for idx in range(start, end):
        row = candles.iloc[idx]
        if sign > 0:
            stop_hit = (row["low"] / entry_price - 1.0) * 100.0 <= -sl
            take_hit = (row["high"] / entry_price - 1.0) * 100.0 >= tp
            stop_price = entry_price * (1 - sl / 100)
            take_price = entry_price * (1 + tp / 100)
        else:
            stop_hit = (entry_price / row["high"] - 1.0) * 100.0 <= -sl
            take_hit = (entry_price / row["low"] - 1.0) * 100.0 >= tp
            stop_price = entry_price * (1 + sl / 100)
            take_price = entry_price * (1 - tp / 100)
        if stop_hit:
            return row["timestamp"], float(stop_price), f"sl_{sl}"
        if take_hit:
            return row["timestamp"], float(take_price), f"tp_{tp}"
    return None, None, "cycle_exit"


def run_strategy(
    spec: EntrySpec,
    entries: pd.DataFrame,
    candidates: pd.DataFrame,
    event_tables: dict[tuple[str, str, bool], pd.DataFrame],
    candles: dict[str, pd.DataFrame],
    exit_method: str,
    tp: float | None = None,
    sl: float | None = None,
    direction_filter: str = "all",
) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()
    entries = entries.sort_values("timestamp").reset_index(drop=True)
    if direction_filter != "all":
        entries = entries[entries["candidate_direction"].eq(direction_filter)].reset_index(drop=True)
    rows = []
    last_exit = pd.Timestamp.min
    for entry in entries.itertuples(index=False):
        if entry.timestamp <= last_exit:
            continue
        direction = str(entry.candidate_direction)
        opposite = "short" if direction == "long" else "long"
        exit_time: pd.Timestamp | None = None
        exit_price: float | None = None
        reason = exit_method
        if exit_method == "upper_opposite_true_reversal_confirmed":
            exit_time, exit_price = next_event(event_tables[(spec.exit_tf, opposite, True)], entry.timestamp)
        elif exit_method == "same_tf_opposite_true_reversal_confirmed":
            exit_time, exit_price = next_event(event_tables[(spec.entry_tf, opposite, True)], entry.timestamp)
        elif exit_method == "cycle_or_tp_sl":
            cycle_time, cycle_price = next_event(event_tables[(spec.exit_tf, opposite, True)], entry.timestamp)
            hit_time, hit_price, hit_reason = tp_sl_before_exit(
                candles[spec.entry_tf],
                int(entry.bar_index),
                float(entry.close_at_entry),
                direction,
                cycle_time,
                float(tp),
                float(sl),
            )
            if hit_time is not None:
                exit_time, exit_price, reason = hit_time, hit_price, f"cycle_or_tp{tp}_sl{sl}_{hit_reason}"
            else:
                exit_time, exit_price, reason = cycle_time, cycle_price, f"cycle_or_tp{tp}_sl{sl}_cycle"
        if exit_time is None or exit_price is None or exit_time <= entry.timestamp:
            continue
        gross = _direction_return(float(entry.close_at_entry), float(exit_price), direction)
        rows.append(
            {
                "strategy_name": spec.name,
                "base_strategy": spec.base,
                "entry_mode": "wait" if spec.base.startswith("W") else "immediate",
                "exit_method": reason,
                "entry_tf": spec.entry_tf,
                "exit_tf": spec.exit_tf,
                "entry_time": entry.timestamp,
                "exit_time": exit_time,
                "direction": direction,
                "entry_price": float(entry.close_at_entry),
                "exit_price": float(exit_price),
                "gross_return": gross,
                "net_return": gross - POSITION_COST_PCT,
                "holding_bars": float((exit_time - entry.timestamp) / _tf_delta(spec.entry_tf)),
                "holding_hours": float((exit_time - entry.timestamp).total_seconds() / 3600),
                "entry_delay_bars": float(getattr(entry, "entry_delay_bars", 0.0)),
                "entry_delay_hours": float(getattr(entry, "entry_delay_hours", 0.0)),
                "candidate_ppo": float(entry.ppo),
                "candidate_hist": float(entry.ppo_hist),
                "candidate_ppo_bin": getattr(entry, "ppo_bin", None),
                "candidate_hist_bin": getattr(entry, "hist_bin", None),
                "upper_1h_ppo_bin": getattr(entry, "upper_1h_ppo_bin", None),
                "upper_1h_hist_bin": getattr(entry, "upper_1h_hist_bin", None),
                "upper_4h_ppo_bin": getattr(entry, "upper_4h_ppo_bin", None),
                "upper_4h_hist_bin": getattr(entry, "upper_4h_hist_bin", None),
                "is_noise": int(entry.is_noise),
                "is_true_reversal": int(entry.is_true_reversal),
            }
        )
        last_exit = exit_time
    return pd.DataFrame(rows)


def max_drawdown(returns: pd.Series) -> float:
    net = pd.to_numeric(returns, errors="coerce").fillna(0) / 100
    equity = (1 + net).cumprod()
    if equity.empty:
        return np.nan
    dd = equity / equity.cummax() - 1
    return float(dd.min() * 100)


def summarize(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in trades.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        net = group["net_return"].dropna()
        wins = net[net > 0]
        losses = net[net < 0]
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n_trades": len(group),
                "win_rate": float((net > 0).mean() * 100) if len(net) else np.nan,
                "avg_return_net": float(net.mean()) if len(net) else np.nan,
                "median_return_net": float(net.median()) if len(net) else np.nan,
                "total_return_compounded": float((np.prod(1 + net / 100) - 1) * 100) if len(net) else np.nan,
                "profit_factor": float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else np.inf if wins.sum() > 0 else np.nan,
                "max_drawdown": max_drawdown(group.sort_values("entry_time")["net_return"]),
                "sharpe_like": float(net.mean() / net.std() * math.sqrt(len(net))) if len(net) > 1 and net.std() else np.nan,
                "avg_holding_hours": float(group["holding_hours"].mean()),
                "avg_entry_delay_bars": float(group["entry_delay_bars"].mean()),
                "noise_trade_pct": float(group["is_noise"].mean() * 100),
                "low_sample": len(group) < LOW_SAMPLE_N,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("avg_return_net", ascending=False).reset_index(drop=True)


def add_regime(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    trades = trades.copy()
    long_ext = trades["upper_4h_ppo_bin"].isin(["bottom10", "bottom20"]) & trades["upper_4h_hist_bin"].isin(["bottom10", "bottom20"])
    short_ext = trades["upper_4h_ppo_bin"].isin(["top10", "top20"]) & trades["upper_4h_hist_bin"].isin(["top10", "top20"])
    one_d_long = trades.get("upper_1d_ppo_bin", pd.Series(index=trades.index, dtype=object)).isin(["bottom10", "bottom20"])
    one_d_short = trades.get("upper_1d_ppo_bin", pd.Series(index=trades.index, dtype=object)).isin(["top10", "top20"])
    trades["market_regime"] = np.select(
        [long_ext, short_ext, one_d_long, one_d_short],
        ["4h_long_extreme", "4h_short_extreme", "1d_long_regime", "1d_short_regime"],
        default="neutral",
    )
    return trades


def walk_forward_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["entry_time"]).dt.year
    rows = []
    windows = [
        ("2017_2020", 2017, 2020),
        ("2021", 2021, 2021),
        ("2022", 2022, 2022),
        ("2023_2025", 2023, 2025),
        ("2026", 2026, 2026),
    ]
    for label, start, end in windows:
        part = trades[(trades["year"] >= start) & (trades["year"] <= end)]
        if part.empty:
            continue
        summ = summarize(part, ["strategy_name", "exit_method"])
        summ.insert(0, "period", label)
        rows.append(summ)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Practical follow-up full backtests for PPO reversal candidates.")
    parser.add_argument("--skip-tpsl", action="store_true", help="Skip TP/SL + cycle grid.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    candles = {tf: load_candles(tf) for tf in TIMEFRAMES}
    events = candidate_event_tables(candidates)

    entry_counts = []
    all_trades = []
    for spec in ENTRY_SPECS:
        entries = entry_frame(candidates, spec, candles)
        entry_counts.append({"strategy_name": spec.name, "entry_tf": spec.entry_tf, "candidate_entries": len(entries)})
        exit_method = "upper_opposite_true_reversal_confirmed"
        all_trades.append(run_strategy(spec, entries, candidates, events, candles, exit_method))
        all_trades.append(run_strategy(spec, entries, candidates, events, candles, "same_tf_opposite_true_reversal_confirmed"))
        if not args.skip_tpsl and spec.base in {"S3", "S7", "S5", "S2"} and spec.entry_tf in TP_SL_GRID:
            for tp in TP_SL_GRID[spec.entry_tf]["tp"]:
                for sl in TP_SL_GRID[spec.entry_tf]["sl"]:
                    all_trades.append(run_strategy(spec, entries, candidates, events, candles, "cycle_or_tp_sl", tp=tp, sl=sl))
        for direction in ("long", "short"):
            side = run_strategy(spec, entries, candidates, events, candles, exit_method, direction_filter=direction)
            if not side.empty:
                side["strategy_name"] = side["strategy_name"] + f"_{direction}_only"
                all_trades.append(side)

    trades = pd.concat([t for t in all_trades if not t.empty], ignore_index=True) if all_trades else pd.DataFrame()
    trades = add_regime(trades)
    trades.to_csv(OUT_DIR / "31_practical_full_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(entry_counts).to_csv(OUT_DIR / "31_practical_entry_counts.csv", index=False, encoding="utf-8-sig")

    summary = summarize(trades, ["strategy_name", "entry_tf", "exit_method"])
    summary.to_csv(OUT_DIR / "32_practical_full_summary.csv", index=False, encoding="utf-8-sig")
    regime = summarize(trades, ["strategy_name", "exit_method", "market_regime"])
    regime.to_csv(OUT_DIR / "33_practical_regime_summary.csv", index=False, encoding="utf-8-sig")
    side = summarize(trades, ["strategy_name", "direction", "exit_method"])
    side.to_csv(OUT_DIR / "34_practical_long_short_summary.csv", index=False, encoding="utf-8-sig")
    wf = walk_forward_summary(trades)
    wf.to_csv(OUT_DIR / "35_practical_period_summary.csv", index=False, encoding="utf-8-sig")

    report = f"""# PPO Reversal Practical Follow-up

This follow-up focuses on full-entry practical checks for S3/S7/S5/S2/S9 plus the requested S3E/S7E and wait variants.

## Entry Counts

{pd.DataFrame(entry_counts).to_markdown(index=False)}

## Top Strategies

{summary[~summary["low_sample"]].head(15).to_markdown(index=False)}

## Worst Strategies

{summary[~summary["low_sample"]].sort_values("avg_return_net").head(15).to_markdown(index=False)}

## Notes

- Full candidate entries are used; no strategy-level sampling cap is applied in this follow-up.
- Wait strategies enter after confirmation bars, so their entry price is the later candle close.
- TP/SL combinations are combined with the upper-TF opposite true reversal exit. Stop is assumed before take-profit if both are hit inside the same candle.
- The output includes period, regime, and long/short splits for the practical checks.
"""
    (OUT_DIR / "PPO_reversal_practical_followup_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote practical follow-up outputs to {OUT_DIR}")
    print(f"Trades: {len(trades):,}; summary rows: {len(summary):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
