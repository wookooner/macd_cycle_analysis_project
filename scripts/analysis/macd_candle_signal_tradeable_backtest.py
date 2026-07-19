"""Tradeable MACD candle-signal horizon backtest.

This is intentionally candle-signal based, not "true finished cycle" based.
At the time of trading we only see that MACD histogram direction has changed;
we do not know whether that move will become a durable cycle or a noise wiggle.

Signal:
  - MACD histogram delta turns positive -> long candidate
  - MACD histogram delta turns negative -> short candidate

Entry:
  - signal candle index + entry_offset_candles
  - by default the signal direction must still be alive at entry

Exits compared:
  - fixed N-candle close
  - next opposite MACD-hist-delta signal close
  - TP/SL first hit, with same-candle ambiguity counted as SL first
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import PROJECT_PATHS  # noqa: E402


TIMEFRAMES = ("5m", "15m", "30m", "1h", "4h", "1d", "1w")
DEFAULT_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d", "1w")
DEFAULT_HORIZONS = (1, 2, 3, 5, 8, 13, 21, 34)
DEFAULT_ENTRY_OFFSETS = (2, 3)
DEFAULT_RISK_LEVELS = (1.0,)
DEFAULT_FEE_BPS_PER_SIDE = 4.0
DEFAULT_SLIPPAGE_BPS_PER_SIDE = 1.0
CONTEXT_TFS = ("5m", "15m", "1h", "4h")


@dataclass(frozen=True)
class ContextFrame:
    timeframe: str
    intervals: pd.DataFrame


def output_dir() -> Path:
    path = PROJECT_PATHS.outputs_root / "analysis_results" / "macd_candle_signal_tradeable_backtest"
    path.mkdir(parents=True, exist_ok=True)
    return path


def market_path(timeframe: str) -> Path:
    candidates = [
        PROJECT_PATHS.base_data_dir / f"BTCUSD_{timeframe}.csv",
        PROJECT_PATHS.base_data_dir / f"BTCUSDT_{timeframe}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing market CSV for {timeframe}: tried {candidates}")


def cycle_path(timeframe: str) -> Path | None:
    candidates = [
        PROJECT_PATHS.asset_cycle_dir("btc") / f"cycles_{timeframe}.parquet",
        PROJECT_PATHS.cycle_structured_dir / "btc" / f"cycles_{timeframe}.parquet",
        PROJECT_PATHS.cycle_structured_dir / f"cycles_{timeframe}.parquet",
    ]
    return next((path for path in candidates if path.exists()), None)


def read_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def signed_direction(values: pd.Series) -> pd.Series:
    signs = np.sign(pd.to_numeric(values, errors="coerce"))
    signs = pd.Series(signs, index=values.index).replace(0, np.nan).ffill()
    return signs


def load_market(timeframe: str) -> pd.DataFrame:
    path = market_path(timeframe)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    ts_col = next((col for col in ("date", "timestamp", "open_time") if col in header), None)
    if ts_col is None:
        raise ValueError(f"{path} has no timestamp column")

    wanted = {
        ts_col,
        "open",
        "high",
        "low",
        "close",
        "macd",
        "macd_signal",
        "macd_hist",
        "ppo",
        "ppo_hist",
        "rsi",
    }
    df = pd.read_csv(path, usecols=[col for col in header if col in wanted]).rename(columns={ts_col: "timestamp"})
    df["timestamp"] = read_timestamp(df["timestamp"])
    for col in ("open", "high", "low", "close", "macd", "macd_signal", "macd_hist", "ppo", "ppo_hist", "rsi"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    required = {"timestamp", "open", "high", "low", "close", "macd_hist"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "macd_hist"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["hist_delta"] = df["macd_hist"].diff()
    df["hist_delta_direction"] = signed_direction(df["hist_delta"])
    return df.dropna(subset=["hist_delta_direction"]).reset_index(drop=True)


def load_context_frames() -> dict[str, ContextFrame]:
    frames: dict[str, ContextFrame] = {}
    for tf in CONTEXT_TFS:
        path = cycle_path(tf)
        if path is None:
            continue
        df = pd.read_parquet(path, columns=["cycle_id", "start_date", "end_date", "cycle_type"]).copy()
        df["start_date"] = read_timestamp(df["start_date"])
        df["end_date"] = read_timestamp(df["end_date"])
        df["cycle_sign"] = df["cycle_type"].astype(str).str.lower().map({"up": 1, "down": -1})
        df = df.dropna(subset=["start_date", "end_date", "cycle_sign"]).sort_values("start_date").reset_index(drop=True)
        frames[tf] = ContextFrame(timeframe=tf, intervals=df)
    return frames


def context_at(timestamp: pd.Timestamp, direction_sign: float, frames: dict[str, ContextFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n_up = 0
    known = 0
    for tf in CONTEXT_TFS:
        frame = frames.get(tf)
        label = "unknown"
        aligned: bool | None = None
        if frame is not None and not frame.intervals.empty:
            starts = frame.intervals["start_date"].to_numpy(dtype="datetime64[ns]")
            idx = int(np.searchsorted(starts, np.datetime64(timestamp.to_datetime64()), side="right") - 1)
            if idx >= 0:
                row = frame.intervals.iloc[idx]
                if row["end_date"] >= timestamp:
                    sign = int(row["cycle_sign"])
                    label = "up" if sign > 0 else "down"
                    aligned = bool(sign == int(direction_sign))
                    n_up += int(sign > 0)
                    known += 1
        out[f"context_{tf}_cycle_type"] = label
        out[f"context_{tf}_aligned"] = aligned
    out["n_up_4"] = n_up if known == len(CONTEXT_TFS) else np.nan
    out["h4_alignment"] = out.get("context_4h_aligned")
    return out


def attach_context(market: pd.DataFrame, frames: dict[str, ContextFrame]) -> pd.DataFrame:
    out = market[["timestamp"]].copy()
    for tf in CONTEXT_TFS:
        frame = frames.get(tf)
        if frame is None or frame.intervals.empty:
            out[f"context_{tf}_cycle_type"] = "unknown"
            out[f"context_{tf}_cycle_sign"] = np.nan
            continue
        intervals = frame.intervals[["start_date", "end_date", "cycle_sign"]].sort_values("start_date").copy()
        merged = pd.merge_asof(
            out[["timestamp"]].sort_values("timestamp"),
            intervals,
            left_on="timestamp",
            right_on="start_date",
            direction="backward",
        ).sort_index()
        valid = merged["end_date"].notna() & (merged["end_date"] >= merged["timestamp"])
        sign = merged["cycle_sign"].where(valid)
        out[f"context_{tf}_cycle_sign"] = sign.to_numpy()
        out[f"context_{tf}_cycle_type"] = np.where(sign > 0, "up", np.where(sign < 0, "down", "unknown"))
    sign_cols = [f"context_{tf}_cycle_sign" for tf in CONTEXT_TFS]
    known_count = out[sign_cols].notna().sum(axis=1)
    out["n_up_4"] = (out[sign_cols] > 0).sum(axis=1).where(known_count == len(CONTEXT_TFS), np.nan)
    return out


def regime_label(timestamp: pd.Timestamp) -> str:
    year = timestamp.year
    if year in (2020, 2021):
        return "bull_2020_2021"
    if year == 2022:
        return "bear_2022"
    if year in (2023, 2024):
        return "recovery_2023_2024"
    if year >= 2025:
        return "recent_2025_plus"
    return "pre_2020"


def find_signal_indices(directions: np.ndarray) -> np.ndarray:
    changed = np.r_[False, directions[1:] != directions[:-1]]
    return np.flatnonzero(changed)


def first_hit_exit(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_close: float,
    sign: float,
    tp_pct: float,
    sl_pct: float,
) -> tuple[int, str, float]:
    if sign > 0:
        tp_price = entry_close * (1.0 + tp_pct / 100.0)
        sl_price = entry_close * (1.0 - sl_pct / 100.0)
        for offset, (high, low, close) in enumerate(zip(highs, lows, closes), start=1):
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
            if hit_sl:
                return offset, "sl_first", sl_price
            if hit_tp:
                return offset, "tp_first", tp_price
            exit_price = close
    else:
        tp_price = entry_close * (1.0 - tp_pct / 100.0)
        sl_price = entry_close * (1.0 + sl_pct / 100.0)
        for offset, (high, low, close) in enumerate(zip(highs, lows, closes), start=1):
            hit_tp = low <= tp_price
            hit_sl = high >= sl_price
            if hit_sl:
                return offset, "sl_first", sl_price
            if hit_tp:
                return offset, "tp_first", tp_price
            exit_price = close
    return len(closes), "horizon_close", float(exit_price)


def build_timeframe_ledger(
    timeframe: str,
    horizons: tuple[int, ...],
    entry_offsets: tuple[int, ...],
    risk_levels: tuple[float, ...],
    total_cost_pct: float,
    context_frames: dict[str, ContextFrame],
    require_confirmation: bool,
) -> pd.DataFrame:
    market = load_market(timeframe)
    market_context = attach_context(market, context_frames)
    directions = market["hist_delta_direction"].to_numpy(dtype="float64")
    signal_indices = find_signal_indices(directions)
    if len(signal_indices) == 0:
        return pd.DataFrame()

    times = market["timestamp"].to_numpy(dtype="datetime64[ns]")
    highs = market["high"].to_numpy(dtype="float64")
    lows = market["low"].to_numpy(dtype="float64")
    closes = market["close"].to_numpy(dtype="float64")
    context_arrays = {
        column: market_context[column].to_numpy()
        for column in market_context.columns
        if column != "timestamp" and not column.endswith("_cycle_sign")
    }
    indicator_cols = [col for col in ("macd", "macd_signal", "macd_hist", "ppo", "ppo_hist", "rsi") if col in market.columns]
    indicator_arrays = {col: market[col].to_numpy(dtype="float64") for col in indicator_cols}

    rows: list[dict[str, Any]] = []
    max_horizon = max(horizons)
    for signal_idx in signal_indices:
        signal_sign = directions[signal_idx]
        next_opposites = signal_indices[signal_indices > signal_idx]
        cycle_exit_idx = int(next_opposites[0]) if len(next_opposites) else None
        for entry_offset in entry_offsets:
            entry_idx = int(signal_idx + entry_offset)
            if entry_idx + max_horizon >= len(market):
                continue
            if require_confirmation and directions[entry_idx] != signal_sign:
                continue

            entry_close = closes[entry_idx]
            if not np.isfinite(entry_close) or entry_close <= 0:
                continue

            entry_time = pd.Timestamp(times[entry_idx])
            base = {
                "timeframe": timeframe,
                "signal_index": int(signal_idx),
                "signal_time": pd.Timestamp(times[signal_idx]),
                "entry_index": int(entry_idx),
                "entry_time": entry_time,
                "entry_offset_candles": int(entry_offset),
                "direction": "long" if signal_sign > 0 else "short",
                "direction_sign": float(signal_sign),
                "entry_close": float(entry_close),
                "regime": regime_label(entry_time),
            }
            for column, values in context_arrays.items():
                base[column] = values[entry_idx]
            h4_sign = market_context["context_4h_cycle_sign"].iloc[entry_idx] if "context_4h_cycle_sign" in market_context.columns else np.nan
            base["h4_alignment"] = bool(h4_sign == signal_sign) if pd.notna(h4_sign) else None
            for tf in CONTEXT_TFS:
                sign_col = f"context_{tf}_cycle_sign"
                align_col = f"context_{tf}_aligned"
                sign_value = market_context[sign_col].iloc[entry_idx] if sign_col in market_context.columns else np.nan
                base[align_col] = bool(sign_value == signal_sign) if pd.notna(sign_value) else None
            for col, values in indicator_arrays.items():
                base[f"entry_{col}"] = values[entry_idx]

            for horizon in horizons:
                exit_idx = entry_idx + horizon
                window_high = highs[entry_idx + 1 : exit_idx + 1]
                window_low = lows[entry_idx + 1 : exit_idx + 1]
                window_close = closes[entry_idx + 1 : exit_idx + 1]
                raw_return = (closes[exit_idx] / entry_close - 1.0) * 100.0
                gross_return = raw_return * signal_sign

                if signal_sign > 0:
                    mfe = (np.nanmax(window_high) / entry_close - 1.0) * 100.0
                    mae = (np.nanmin(window_low) / entry_close - 1.0) * 100.0
                else:
                    mfe = (1.0 - np.nanmin(window_low) / entry_close) * 100.0
                    mae = (1.0 - np.nanmax(window_high) / entry_close) * 100.0

                row = {
                    **base,
                    "exit_rule": "fixed_horizon",
                    "horizon_candles": int(horizon),
                    "exit_time": pd.Timestamp(times[exit_idx]),
                    "exit_close": float(closes[exit_idx]),
                    "gross_return_pct": float(gross_return),
                    "net_return_pct": float(gross_return - total_cost_pct),
                    "mfe_pct": float(mfe),
                    "mae_pct": float(mae),
                    "adverse_excursion_pct": float(abs(min(mae, 0.0))),
                    "favorable_excursion_pct": float(max(mfe, 0.0)),
                    "cycle_exit_available": cycle_exit_idx is not None and cycle_exit_idx > entry_idx,
                }
                rows.append(row)

                if cycle_exit_idx is not None and cycle_exit_idx > entry_idx:
                    managed_idx = min(cycle_exit_idx, exit_idx)
                    managed_raw = (closes[managed_idx] / entry_close - 1.0) * 100.0
                    rows.append({
                        **base,
                        "exit_rule": "opposite_signal_or_horizon",
                        "horizon_candles": int(horizon),
                        "exit_time": pd.Timestamp(times[managed_idx]),
                        "exit_close": float(closes[managed_idx]),
                        "gross_return_pct": float(managed_raw * signal_sign),
                        "net_return_pct": float(managed_raw * signal_sign - total_cost_pct),
                        "mfe_pct": float(mfe),
                        "mae_pct": float(mae),
                        "adverse_excursion_pct": float(abs(min(mae, 0.0))),
                        "favorable_excursion_pct": float(max(mfe, 0.0)),
                        "cycle_exit_available": True,
                    })

    return pd.DataFrame(rows)


def build_tpsl_p95_ledger(
    timeframe: str,
    horizons: tuple[int, ...],
    entry_offsets: tuple[int, ...],
    p95_lookup: dict[tuple[str, str, int, int], float],
    total_cost_pct: float,
    require_confirmation: bool,
    tp_pct: float = 1.0,
) -> pd.DataFrame:
    market = load_market(timeframe)
    directions = market["hist_delta_direction"].to_numpy(dtype="float64")
    signal_indices = find_signal_indices(directions)
    if len(signal_indices) == 0:
        return pd.DataFrame()

    times = market["timestamp"].to_numpy(dtype="datetime64[ns]")
    highs = market["high"].to_numpy(dtype="float64")
    lows = market["low"].to_numpy(dtype="float64")
    closes = market["close"].to_numpy(dtype="float64")

    rows: list[dict[str, Any]] = []
    max_horizon = max(horizons)
    for signal_idx in signal_indices:
        signal_sign = directions[signal_idx]
        direction = "long" if signal_sign > 0 else "short"
        for entry_offset in entry_offsets:
            entry_idx = int(signal_idx + entry_offset)
            if entry_idx + max_horizon >= len(market):
                continue
            if require_confirmation and directions[entry_idx] != signal_sign:
                continue
            entry_close = closes[entry_idx]
            if not np.isfinite(entry_close) or entry_close <= 0:
                continue

            for horizon in horizons:
                sl_pct = p95_lookup.get((timeframe, direction, int(entry_offset), int(horizon)))
                if sl_pct is None or not np.isfinite(sl_pct) or sl_pct <= 0:
                    continue
                exit_idx = entry_idx + horizon
                window_high = highs[entry_idx + 1 : exit_idx + 1]
                window_low = lows[entry_idx + 1 : exit_idx + 1]
                window_close = closes[entry_idx + 1 : exit_idx + 1]
                bars_to_exit, outcome, exit_price = first_hit_exit(
                    window_high,
                    window_low,
                    window_close,
                    entry_close,
                    signal_sign,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                )
                raw_return = (exit_price / entry_close - 1.0) * 100.0
                gross_return = raw_return * signal_sign
                rows.append({
                    "timeframe": timeframe,
                    "direction": direction,
                    "entry_offset_candles": int(entry_offset),
                    "exit_rule": "tp_1_or_sl_p95",
                    "horizon_candles": int(horizon),
                    "signal_time": pd.Timestamp(times[signal_idx]),
                    "entry_time": pd.Timestamp(times[entry_idx]),
                    "exit_time": pd.Timestamp(times[entry_idx + bars_to_exit]),
                    "entry_close": float(entry_close),
                    "exit_close": float(exit_price),
                    "gross_return_pct": float(gross_return),
                    "net_return_pct": float(gross_return - total_cost_pct),
                    "tp_sl_outcome": outcome,
                    "sl_pct": float(sl_pct),
                    "mfe_pct": np.nan,
                    "mae_pct": np.nan,
                    "adverse_excursion_pct": np.nan,
                })
    return pd.DataFrame(rows)


def profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    if losses == 0:
        return np.inf if wins > 0 else np.nan
    return float(wins / losses)


def summarize_group(group: pd.DataFrame) -> pd.Series:
    r = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna()
    gross = pd.to_numeric(group["gross_return_pct"], errors="coerce").dropna()
    if r.empty:
        return pd.Series({"n": 0})
    return pd.Series({
        "n": int(len(r)),
        "win_rate_pct": float((r > 0).mean() * 100.0),
        "gross_avg_return_pct": float(gross.mean()) if not gross.empty else np.nan,
        "net_avg_return_pct": float(r.mean()),
        "net_median_return_pct": float(r.median()),
        "net_profit_factor": profit_factor(r),
        "net_p10_return_pct": float(r.quantile(0.10)),
        "net_p25_return_pct": float(r.quantile(0.25)),
        "net_p75_return_pct": float(r.quantile(0.75)),
        "net_p90_return_pct": float(r.quantile(0.90)),
        "net_worst_return_pct": float(r.min()),
        "net_best_return_pct": float(r.max()),
        "avg_mfe_pct": float(pd.to_numeric(group["mfe_pct"], errors="coerce").mean()),
        "avg_mae_pct": float(pd.to_numeric(group["mae_pct"], errors="coerce").mean()),
        "p95_adverse_excursion_pct": float(pd.to_numeric(group["adverse_excursion_pct"], errors="coerce").quantile(0.95)),
    })


def summarize(
    ledger: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    return (
        ledger.groupby(group_cols, dropna=False, observed=True)
        .apply(summarize_group)
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def best_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    ranked = summary.copy()
    ranked["score"] = (
        ranked["net_avg_return_pct"].fillna(-999.0)
        + ranked["win_rate_pct"].fillna(0.0) / 100.0
        - ranked["p95_adverse_excursion_pct"].fillna(999.0) * 0.20
    )
    sort_cols = [col for col in ["timeframe", "direction", "entry_offset_candles", "exit_rule"] if col in ranked.columns]
    ranked = ranked.sort_values(sort_cols + ["score", "n"], ascending=[True] * len(sort_cols) + [False, False])
    return ranked.groupby(sort_cols, as_index=False, observed=True).head(3).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 80) -> str:
    if df.empty:
        return "No rows."
    show = df.head(limit).copy()
    columns = [col for col in columns if col in show.columns]
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    return show[columns].to_markdown(index=False)


def build_report(
    metadata: dict[str, Any],
    base_summary: pd.DataFrame,
    conditional_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    best: pd.DataFrame,
) -> str:
    lines = [
        "# Tradeable MACD Candle-Signal Backtest",
        "",
        "## Setup",
        "",
        f"- Entry offsets: `{', '.join(str(v) for v in metadata['entry_offsets'])}` candles",
        f"- Horizons: `{', '.join(str(v) for v in metadata['horizons'])}` candles",
        f"- Round-trip cost: `{metadata['total_cost_pct']:.3f}%`",
        f"- Confirmation required at entry: `{metadata['require_confirmation']}`",
        f"- Output dir: `{metadata['output_dir']}`",
        "",
        "## Best Net Candidates",
        "",
        markdown_table(best, [
            "timeframe",
            "direction",
            "entry_offset_candles",
            "exit_rule",
            "horizon_candles",
            "n",
            "win_rate_pct",
            "net_avg_return_pct",
            "net_profit_factor",
            "p95_adverse_excursion_pct",
            "score",
        ], 80),
        "",
        "## Base Summary",
        "",
        markdown_table(base_summary, [
            "timeframe",
            "direction",
            "entry_offset_candles",
            "exit_rule",
            "horizon_candles",
            "n",
            "win_rate_pct",
            "gross_avg_return_pct",
            "net_avg_return_pct",
            "net_profit_factor",
            "p95_adverse_excursion_pct",
        ], 120),
        "",
        "## Direction x n_up x 4h Alignment",
        "",
        markdown_table(conditional_summary, [
            "timeframe",
            "direction",
            "entry_offset_candles",
            "exit_rule",
            "horizon_candles",
            "n_up_4",
            "h4_alignment",
            "n",
            "win_rate_pct",
            "net_avg_return_pct",
            "net_profit_factor",
            "p95_adverse_excursion_pct",
        ], 120),
        "",
        "## Regime Split",
        "",
        markdown_table(regime_summary, [
            "timeframe",
            "direction",
            "entry_offset_candles",
            "exit_rule",
            "horizon_candles",
            "regime",
            "n",
            "win_rate_pct",
            "net_avg_return_pct",
            "net_profit_factor",
        ], 120),
        "",
        "## Notes",
        "",
        "- This version does not use future cycle labels for entry direction.",
        "- `opposite_signal_or_horizon` is the candle-signal version of cycle-end active management.",
        "- TP/SL same-candle ambiguity is conservative: SL first.",
    ]
    return "\n".join(lines) + "\n"


def run(
    timeframes: tuple[str, ...],
    horizons: tuple[int, ...],
    entry_offsets: tuple[int, ...],
    risk_levels: tuple[float, ...],
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
    require_confirmation: bool,
    include_ledger_csv: bool,
) -> dict[str, Any]:
    out = output_dir()
    total_cost_pct = 2.0 * (fee_bps_per_side + slippage_bps_per_side) / 100.0
    context_frames = load_context_frames()
    ledgers: list[pd.DataFrame] = []
    errors: dict[str, str] = {}

    for tf in timeframes:
        try:
            ledgers.append(build_timeframe_ledger(
                timeframe=tf,
                horizons=horizons,
                entry_offsets=entry_offsets,
                risk_levels=risk_levels,
                total_cost_pct=total_cost_pct,
                context_frames=context_frames,
                require_confirmation=require_confirmation,
            ))
        except Exception as exc:
            errors[tf] = str(exc)

    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    fixed_and_managed = ledger[ledger["exit_rule"].isin(["fixed_horizon", "opposite_signal_or_horizon"])].copy() if not ledger.empty else ledger
    base_cols = ["timeframe", "direction", "entry_offset_candles", "exit_rule", "horizon_candles"]
    conditional_cols = base_cols + ["n_up_4", "h4_alignment"]
    regime_cols = base_cols + ["regime"]
    base_summary = summarize(fixed_and_managed, base_cols)
    conditional_summary = summarize(fixed_and_managed, conditional_cols)
    regime_summary = summarize(fixed_and_managed, regime_cols)
    fixed_summary = base_summary[base_summary["exit_rule"].eq("fixed_horizon")].copy() if not base_summary.empty else pd.DataFrame()
    p95_lookup = {
        (
            str(row.timeframe),
            str(row.direction),
            int(row.entry_offset_candles),
            int(row.horizon_candles),
        ): float(row.p95_adverse_excursion_pct)
        for row in fixed_summary.itertuples(index=False)
        if pd.notna(row.p95_adverse_excursion_pct)
    }
    tpsl_ledgers: list[pd.DataFrame] = []
    for tf in timeframes:
        if tf in errors:
            continue
        try:
            tpsl_ledgers.append(build_tpsl_p95_ledger(
                timeframe=tf,
                horizons=horizons,
                entry_offsets=entry_offsets,
                p95_lookup=p95_lookup,
                total_cost_pct=total_cost_pct,
                require_confirmation=require_confirmation,
            ))
        except Exception as exc:
            errors[f"{tf}_tp_sl_p95"] = str(exc)
    tpsl_ledger = pd.concat(tpsl_ledgers, ignore_index=True) if tpsl_ledgers else pd.DataFrame()
    tp_sl_summary = summarize(tpsl_ledger, base_cols)
    best = best_candidates(pd.concat([base_summary, tp_sl_summary], ignore_index=True) if not tp_sl_summary.empty else base_summary)

    files = {
        "ledger_parquet": str(out / "tradeable_signal_ledger.parquet"),
        "base_summary_csv": str(out / "base_horizon_vs_managed_summary.csv"),
        "conditional_summary_csv": str(out / "conditional_nup_h4_summary.csv"),
        "regime_summary_csv": str(out / "regime_summary.csv"),
        "tp_sl_summary_csv": str(out / "tp_sl_rule_summary.csv"),
        "tp_sl_p95_ledger_parquet": str(out / "tp_sl_p95_ledger.parquet"),
        "best_candidates_csv": str(out / "best_net_candidates.csv"),
        "report_md": str(out / "report.md"),
        "metadata_json": str(out / "metadata.json"),
    }
    ledger.to_parquet(files["ledger_parquet"], index=False)
    if include_ledger_csv:
        ledger.to_csv(out / "tradeable_signal_ledger.csv", index=False, encoding="utf-8-sig")
    base_summary.to_csv(files["base_summary_csv"], index=False, encoding="utf-8-sig")
    conditional_summary.to_csv(files["conditional_summary_csv"], index=False, encoding="utf-8-sig")
    regime_summary.to_csv(files["regime_summary_csv"], index=False, encoding="utf-8-sig")
    tp_sl_summary.to_csv(files["tp_sl_summary_csv"], index=False, encoding="utf-8-sig")
    tpsl_ledger.to_parquet(files["tp_sl_p95_ledger_parquet"], index=False)
    best.to_csv(files["best_candidates_csv"], index=False, encoding="utf-8-sig")

    metadata = {
        "timeframes": list(timeframes),
        "horizons": list(horizons),
        "entry_offsets": list(entry_offsets),
        "risk_levels": list(risk_levels),
        "fee_bps_per_side": fee_bps_per_side,
        "slippage_bps_per_side": slippage_bps_per_side,
        "total_cost_pct": total_cost_pct,
        "require_confirmation": require_confirmation,
        "output_dir": str(out),
        "ledger_rows": int(len(ledger)),
        "errors": errors,
        "files": files,
    }
    Path(files["report_md"]).write_text(build_report(metadata, base_summary, conditional_summary, regime_summary, best), encoding="utf-8")
    Path(files["metadata_json"]).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tradeable candle-level MACD histogram signal backtest.")
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), choices=list(TIMEFRAMES))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--entry-offsets", nargs="+", type=int, default=list(DEFAULT_ENTRY_OFFSETS))
    parser.add_argument("--risk-levels", nargs="+", type=float, default=list(DEFAULT_RISK_LEVELS))
    parser.add_argument("--fee-bps-per-side", type=float, default=DEFAULT_FEE_BPS_PER_SIDE)
    parser.add_argument("--slippage-bps-per-side", type=float, default=DEFAULT_SLIPPAGE_BPS_PER_SIDE)
    parser.add_argument("--allow-faded-signals", action="store_true", help="Enter even if hist-delta direction no longer matches at entry offset.")
    parser.add_argument("--include-ledger-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run(
        timeframes=tuple(args.timeframes),
        horizons=tuple(sorted(set(args.horizons))),
        entry_offsets=tuple(sorted(set(args.entry_offsets))),
        risk_levels=tuple(sorted(set(args.risk_levels))),
        fee_bps_per_side=args.fee_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        require_confirmation=not args.allow_faded_signals,
        include_ledger_csv=args.include_ledger_csv,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
