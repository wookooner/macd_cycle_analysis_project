from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.ppo_reversal_candidate_backtest import (  # noqa: E402
    LOW_SAMPLE_N,
    PROJECT_PATHS,
    ROUND_TRIP_FEE_PCT,
    SLIPPAGE_PER_SIDE_PCT,
    TF_SECONDS,
    _add_full_deciles,
    _candidate_sign,
    _direction_return,
    _raw_market_path,
    _read_timestamp,
)


BASE_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_reversal_candidate_backtest"
LEAD_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_15m_leading_1h_reversal_analysis"
OUT_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_successful_countertrend_stop_analysis"
TIMEFRAMES = ("15m", "1h", "4h", "1d")
UPPER_TF = {"15m": "1h", "1h": "4h", "4h": "1d"}
TP_VALUES = (0.8, 1.2, 1.8, 2.5)
SL_VALUES = (0.4, 0.6, 0.8, 1.0, 1.2)
GRID_SL_VALUES = (0.4, 0.6, 0.8, 1.2)
ATR_MULTS = (0.8, 1.0, 1.5, 2.0)
POSITION_COST_PCT = ROUND_TRIP_FEE_PCT + SLIPPAGE_PER_SIDE_PCT * 2


def output_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def is_bottom20(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(["bottom10", "bottom20"])


def is_top20(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(["top10", "top20"])


def pct_bool(series: pd.Series) -> float:
    return float(series.mean() * 100.0) if len(series) else np.nan


def max_drawdown(returns_pct: pd.Series) -> float:
    if returns_pct.empty:
        return np.nan
    equity = (1.0 + returns_pct.fillna(0.0) / 100.0).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def profit_factor(returns_pct: pd.Series) -> float:
    wins = returns_pct[returns_pct > 0].sum()
    losses = -returns_pct[returns_pct < 0].sum()
    if losses == 0:
        return np.inf if wins > 0 else np.nan
    return float(wins / losses)


def load_raw_candles(timeframe: str) -> pd.DataFrame:
    path = _raw_market_path(timeframe)
    wanted = {
        "date",
        "timestamp",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "ppo",
        "ppo_hist",
        "rsi",
        "ma_25",
        "ma_99",
        "cvd",
        "volume",
        "volume_delta",
    }
    available = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[col for col in available if col in wanted])
    ts_col = next(col for col in ("timestamp", "open_time", "date") if col in df.columns)
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _read_timestamp(df["timestamp"])
    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "ppo", "ppo_hist"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df), dtype=np.int64)
    df["ppo_hist_diff"] = df["ppo_hist"].diff()
    df = _add_full_deciles(df, "ppo", "ppo")
    df = _add_full_deciles(df, "ppo_hist", "hist")
    df["distance_from_ma25"] = (df["close"] / df["ma_25"] - 1.0) * 100.0 if "ma_25" in df else np.nan
    df["distance_from_ma99"] = (df["close"] / df["ma_99"] - 1.0) * 100.0 if "ma_99" in df else np.nan
    df["cvd_delta"] = df["cvd"].diff() if "cvd" in df else np.nan
    if "volume_delta" not in df:
        df["volume_delta"] = df["volume"].diff() if "volume" in df else np.nan
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14, min_periods=1).mean()
    df["hist_improving"] = df["ppo_hist_diff"] > 0
    df["long_bias"] = df["ppo_hist"] < 0
    df["short_bias"] = df["ppo_hist"] > 0
    return df


def load_candidates() -> pd.DataFrame:
    path = BASE_DIR / "20_reversal_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    cols = pd.read_csv(path, nrows=0).columns
    wanted = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "ppo",
        "ppo_hist",
        "ppo_hist_diff",
        "bar_index",
        "candidate_tf",
        "candidate_direction",
        "close_at_entry",
        "ppo_bin",
        "hist_bin",
        "cycle_type",
        "cycle_progress",
        "candidate_run_length_afterward",
        "is_noise",
        "is_true_reversal",
        "upper_1h_ppo",
        "upper_1h_hist",
        "upper_1h_hist_diff",
        "upper_1h_ppo_bin",
        "upper_1h_hist_bin",
        "upper_1h_cycle_direction",
        "upper_1h_cycle_progress",
        "upper_1h_align",
        "upper_4h_ppo",
        "upper_4h_hist",
        "upper_4h_hist_diff",
        "upper_4h_ppo_bin",
        "upper_4h_hist_bin",
        "upper_4h_cycle_direction",
        "upper_4h_cycle_progress",
        "upper_4h_align",
        "has_supportive_4h_extreme",
        "has_contra_4h_extreme",
        "upper_1d_ppo",
        "upper_1d_hist",
        "upper_1d_hist_diff",
        "upper_1d_ppo_bin",
        "upper_1d_hist_bin",
        "upper_1d_cycle_direction",
        "upper_1d_cycle_progress",
        "upper_1d_align",
        "align_count",
        "major_align_count",
    ]
    df = pd.read_csv(path, usecols=[col for col in wanted if col in cols], engine="python")
    df["timestamp"] = _read_timestamp(df["timestamp"])
    for col in df.columns:
        if col not in {"timestamp", "candidate_tf", "candidate_direction", "ppo_bin", "hist_bin", "cycle_type"} and not col.endswith("_bin") and not col.endswith("_direction"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [c for c in df.columns if c.endswith("_direction") or c == "cycle_type"]:
        df[col] = df[col].astype(str).str.lower()
    return df.sort_values(["candidate_tf", "timestamp"]).reset_index(drop=True)


def add_own_market_features(candidates: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    keep = [
        "timestamp",
        "bar_index",
        "rsi",
        "distance_from_ma25",
        "distance_from_ma99",
        "cvd_delta",
        "volume_delta",
        "atr_14",
    ]
    for tf, group in candidates.groupby("candidate_tf", sort=False):
        own = candles[tf][[col for col in keep if col in candles[tf].columns]].copy()
        own = own.rename(columns={col: f"own_{col}" for col in own.columns if col != "timestamp"})
        merged = group.merge(own, on="timestamp", how="left")
        if "own_bar_index" in merged:
            merged["bar_index"] = merged["bar_index"].fillna(merged["own_bar_index"])
        frames.append(merged)
    return pd.concat(frames, ignore_index=True).sort_values(["candidate_tf", "timestamp"]).reset_index(drop=True)


def build_event_tables(candidates: pd.DataFrame) -> dict[tuple[str, str, bool], pd.DataFrame]:
    tables = {}
    for tf in TIMEFRAMES:
        frame = candidates[candidates["candidate_tf"].eq(tf)].sort_values("timestamp").reset_index(drop=True)
        for direction in ("long", "short"):
            sub = frame[frame["candidate_direction"].eq(direction)].reset_index(drop=True)
            tables[(tf, direction, False)] = sub
            tables[(tf, direction, True)] = sub[sub["is_true_reversal"].eq(1)].reset_index(drop=True)
    return tables


def next_event(events: pd.DataFrame, when: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    if events.empty:
        return None, None
    times = events["timestamp"].to_numpy(dtype="datetime64[ns]")
    idx = int(np.searchsorted(times, np.datetime64(when), side="right"))
    if idx >= len(events):
        return None, None
    row = events.iloc[idx]
    return row["timestamp"], float(row["close_at_entry"])


def first_tp_sl(
    segment: pd.DataFrame,
    entry_price: float,
    direction: str,
    tp_pct: float | None = None,
    sl_pct: float | None = None,
    invalid_low: float | None = None,
    invalid_high: float | None = None,
) -> tuple[pd.Timestamp | None, float | None, str]:
    sign = _candidate_sign(direction)
    for row in segment.itertuples(index=False):
        if sign > 0:
            sl_hit = sl_pct is not None and (row.low / entry_price - 1.0) * 100.0 <= -sl_pct
            invalid_hit = invalid_low is not None and row.low <= invalid_low
            tp_hit = tp_pct is not None and (row.high / entry_price - 1.0) * 100.0 >= tp_pct
            if sl_hit:
                return row.timestamp, entry_price * (1.0 - sl_pct / 100.0), "stop_loss"
            if invalid_hit:
                return row.timestamp, float(row.close), "invalidation"
            if tp_hit:
                return row.timestamp, entry_price * (1.0 + tp_pct / 100.0), "take_profit"
        else:
            sl_hit = sl_pct is not None and (entry_price / row.high - 1.0) * 100.0 <= -sl_pct
            invalid_hit = invalid_high is not None and row.high >= invalid_high
            tp_hit = tp_pct is not None and (entry_price / row.low - 1.0) * 100.0 >= tp_pct
            if sl_hit:
                return row.timestamp, entry_price * (1.0 + sl_pct / 100.0), "stop_loss"
            if invalid_hit:
                return row.timestamp, float(row.close), "invalidation"
            if tp_hit:
                return row.timestamp, entry_price * (1.0 - tp_pct / 100.0), "take_profit"
    return None, None, ""


def add_forward_metrics(candidates: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for tf, group in candidates.groupby("candidate_tf", sort=False):
        market = candles[tf].reset_index(drop=True)
        high = market["high"].to_numpy()
        low = market["low"].to_numpy()
        close = market["close"].to_numpy()
        rows: list[dict[str, Any]] = []
        max_i = len(market) - 1
        for row in group.itertuples(index=False):
            item = row._asdict()
            idx = int(item.get("bar_index") if not pd.isna(item.get("bar_index")) else item.get("own_bar_index"))
            entry = float(item.get("close_at_entry"))
            sign = _candidate_sign(item.get("candidate_direction"))
            for bars in (4, 8, 16, 32):
                end = min(max_i, idx + bars)
                if end <= idx or not np.isfinite(entry):
                    item[f"forward_return_{bars}bars"] = np.nan
                    continue
                item[f"forward_return_{bars}bars"] = _direction_return(entry, float(close[end]), item["candidate_direction"])
            end32 = min(max_i, idx + 32)
            seg_hi = high[idx + 1 : end32 + 1]
            seg_lo = low[idx + 1 : end32 + 1]
            if len(seg_hi) and np.isfinite(entry):
                if sign > 0:
                    item["mfe_32bars"] = (np.nanmax(seg_hi) / entry - 1.0) * 100.0
                    item["mae_32bars"] = (np.nanmin(seg_lo) / entry - 1.0) * 100.0
                else:
                    item["mfe_32bars"] = (entry / np.nanmin(seg_lo) - 1.0) * 100.0
                    item["mae_32bars"] = (entry / np.nanmax(seg_hi) - 1.0) * 100.0
            else:
                item["mfe_32bars"] = np.nan
                item["mae_32bars"] = np.nan
            item["trap"] = False
            for j in range(idx + 1, end32 + 1):
                if sign > 0:
                    fav = (high[j] / entry - 1.0) * 100.0
                    adv = (low[j] / entry - 1.0) * 100.0
                else:
                    fav = (entry / low[j] - 1.0) * 100.0
                    adv = (entry / high[j] - 1.0) * 100.0
                if adv <= -1.0:
                    item["trap"] = fav < 0.5
                    break
                if fav >= 0.5:
                    break
            rows.append(item)
        frames.append(pd.DataFrame(rows))
    out = pd.concat(frames, ignore_index=True)
    out["small_win"] = out["mfe_32bars"] >= 0.8
    out["mid_win"] = out["mfe_32bars"] >= 1.2
    out["strong_win"] = out["mfe_32bars"] >= 1.8
    out["big_win"] = out["mfe_32bars"] >= 2.5
    out["clean_win"] = (out["mfe_32bars"] >= 1.2) & (out["mae_32bars"] > -0.6)
    return out


def add_event_returns(candidates: pd.DataFrame, events: dict[tuple[str, str, bool], pd.DataFrame]) -> pd.DataFrame:
    out = candidates.copy()
    out["same_tf_opposite_true_time"] = pd.NaT
    out["return_until_same_tf_opposite"] = np.nan
    out["upper_tf_opposite_true_time"] = pd.NaT
    out["return_until_upper_tf_opposite"] = np.nan

    def assign_next(mask: pd.Series, event_tf: str, opposite: str, time_col: str, return_col: str) -> None:
        event_frame = events[(event_tf, opposite, True)]
        idx = out.index[mask]
        if event_frame.empty or len(idx) == 0:
            return
        event_times = event_frame["timestamp"].to_numpy(dtype="datetime64[ns]")
        event_prices = event_frame["close_at_entry"].to_numpy(dtype="float64")
        query_times = out.loc[idx, "timestamp"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(event_times, query_times, side="right")
        valid = positions < len(event_frame)
        if not valid.any():
            return
        valid_idx = idx[valid]
        matched_times = event_times[positions[valid]]
        matched_prices = event_prices[positions[valid]]
        entries = out.loc[valid_idx, "close_at_entry"].to_numpy(dtype="float64")
        directions = out.loc[valid_idx, "candidate_direction"].to_numpy()
        returns = np.array([
            _direction_return(float(entry), float(exit_price), str(direction))
            for entry, exit_price, direction in zip(entries, matched_prices, directions)
        ])
        out.loc[valid_idx, time_col] = pd.to_datetime(matched_times)
        out.loc[valid_idx, return_col] = returns

    for tf in TIMEFRAMES:
        for direction, opposite in (("long", "short"), ("short", "long")):
            base_mask = out["candidate_tf"].eq(tf) & out["candidate_direction"].eq(direction)
            assign_next(base_mask, tf, opposite, "same_tf_opposite_true_time", "return_until_same_tf_opposite")
            upper = UPPER_TF.get(tf)
            if upper:
                assign_next(base_mask, upper, opposite, "upper_tf_opposite_true_time", "return_until_upper_tf_opposite")
    return out


def summarize_success(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in candidates.groupby(["candidate_tf", "candidate_direction"], dropna=False):
        row = {
            "candidate_tf": keys[0],
            "candidate_direction": keys[1],
            "n": len(group),
            "low_sample": len(group) < LOW_SAMPLE_N,
            "small_win_pct": pct_bool(group["small_win"]),
            "mid_win_pct": pct_bool(group["mid_win"]),
            "strong_win_pct": pct_bool(group["strong_win"]),
            "big_win_pct": pct_bool(group["big_win"]),
            "clean_win_pct": pct_bool(group["clean_win"]),
            "trap_pct": pct_bool(group["trap"]),
            "avg_mfe_32bars": group["mfe_32bars"].mean(),
            "avg_mae_32bars": group["mae_32bars"].mean(),
            "avg_forward_return_8bars": group["forward_return_8bars"].mean(),
            "avg_forward_return_32bars": group["forward_return_32bars"].mean(),
            "avg_return_until_same_tf_opposite": group["return_until_same_tf_opposite"].mean(),
            "avg_return_until_upper_tf_opposite": group["return_until_upper_tf_opposite"].mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def feature_profile(candidates: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "ppo",
        "ppo_hist",
        "ppo_hist_diff",
        "own_rsi",
        "own_distance_from_ma25",
        "own_distance_from_ma99",
        "own_cvd_delta",
        "mfe_32bars",
        "mae_32bars",
        "upper_1h_ppo",
        "upper_1h_hist",
        "upper_1h_hist_diff",
        "upper_4h_ppo",
        "upper_4h_hist",
        "upper_4h_hist_diff",
        "upper_1d_ppo",
        "upper_1d_hist",
    ]
    categories = [
        "ppo_bin",
        "hist_bin",
        "upper_1h_ppo_bin",
        "upper_1h_hist_bin",
        "upper_1h_cycle_direction",
        "upper_4h_ppo_bin",
        "upper_4h_hist_bin",
        "upper_4h_cycle_direction",
        "upper_1d_ppo_bin",
        "upper_1d_hist_bin",
        "upper_1d_cycle_direction",
    ]
    masks = {
        "small_win": candidates["small_win"],
        "mid_win": candidates["mid_win"],
        "strong_win": candidates["strong_win"],
        "big_win": candidates["big_win"],
        "clean_win": candidates["clean_win"],
        "trap": candidates["trap"],
    }
    rows = []
    for direction in ("long", "short"):
        direction_mask = candidates["candidate_direction"].eq(direction)
        for label, mask in masks.items():
            group = candidates[direction_mask & mask]
            row: dict[str, Any] = {
                "candidate_direction": direction,
                "profile_group": label,
                "n": len(group),
                "low_sample": len(group) < LOW_SAMPLE_N,
            }
            for col in numeric:
                if col in group:
                    row[f"{col}_mean"] = group[col].mean()
                    row[f"{col}_median"] = group[col].median()
            for col in categories:
                if col in group and not group.empty:
                    vc = group[col].astype(str).value_counts(normalize=True)
                    row[f"{col}_mode"] = vc.index[0] if len(vc) else ""
                    row[f"{col}_mode_pct"] = float(vc.iloc[0] * 100.0) if len(vc) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def return_bucket_profile(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, group in candidates.groupby("candidate_direction"):
        ret = group["forward_return_32bars"]
        q10 = ret.quantile(0.10)
        q20 = ret.quantile(0.20)
        q80 = ret.quantile(0.80)
        q90 = ret.quantile(0.90)
        bucket_masks = {
            "top10_return": ret >= q90,
            "top20_return": ret >= q80,
            "mid_return": (ret > q20) & (ret < q80),
            "bottom20_return": ret <= q20,
            "bottom10_return": ret <= q10,
            "clean_win": group["clean_win"],
            "trap": group["trap"],
        }
        for bucket, mask in bucket_masks.items():
            sub = group[mask]
            row: dict[str, Any] = {
                "candidate_direction": direction,
                "bucket": bucket,
                "n": len(sub),
                "low_sample": len(sub) < LOW_SAMPLE_N,
                "avg_forward_return_32bars": sub["forward_return_32bars"].mean(),
                "median_forward_return_32bars": sub["forward_return_32bars"].median(),
                "avg_mfe_32bars": sub["mfe_32bars"].mean(),
                "avg_mae_32bars": sub["mae_32bars"].mean(),
                "h1_down_pct": pct_bool(sub["upper_1h_cycle_direction"].eq("down")) if "upper_1h_cycle_direction" in sub else np.nan,
                "h4_down_pct": pct_bool(sub["upper_4h_cycle_direction"].eq("down")) if "upper_4h_cycle_direction" in sub else np.nan,
                "d1_down_pct": pct_bool(sub["upper_1d_cycle_direction"].eq("down")) if "upper_1d_cycle_direction" in sub else np.nan,
                "h1_hist_improving_pct": pct_bool(sub["upper_1h_hist_diff"] > 0) if "upper_1h_hist_diff" in sub else np.nan,
                "h4_long_bias_pct": pct_bool(sub["upper_4h_hist"] < 0) if "upper_4h_hist" in sub else np.nan,
                "h4_short_bias_pct": pct_bool(sub["upper_4h_hist"] > 0) if "upper_4h_hist" in sub else np.nan,
                "d1_strong_short_pct": pct_bool((sub["upper_1d_hist"] > 0) & is_top20(sub["upper_1d_ppo_bin"])) if "upper_1d_hist" in sub else np.nan,
            }
            for col in ("ppo_bin", "hist_bin", "upper_1h_ppo_bin", "upper_1h_hist_bin", "upper_4h_ppo_bin", "upper_4h_hist_bin", "upper_1d_ppo_bin", "upper_1d_hist_bin"):
                if col in sub and len(sub):
                    vc = sub[col].astype(str).value_counts(normalize=True)
                    row[f"{col}_mode"] = vc.index[0]
                    row[f"{col}_mode_pct"] = float(vc.iloc[0] * 100.0)
            rows.append(row)
    return pd.DataFrame(rows)


def tp_before_sl_flags(candidates: pd.DataFrame, candles: dict[str, pd.DataFrame], tp: float, sl: float) -> pd.Series:
    candle_arrays = {
        tf: {
            "high": frame["high"].to_numpy(dtype="float64"),
            "low": frame["low"].to_numpy(dtype="float64"),
        }
        for tf, frame in candles.items()
    }
    values = []
    for row in candidates.itertuples(index=False):
        arr = candle_arrays[row.candidate_tf]
        idx = int(row.bar_index)
        entry = float(row.close_at_entry)
        start = idx + 1
        end = min(len(arr["high"]), idx + 33)
        highs = arr["high"][start:end]
        lows = arr["low"][start:end]
        if len(highs) == 0 or not np.isfinite(entry):
            values.append(False)
            continue
        if row.candidate_direction == "long":
            take = (highs / entry - 1.0) * 100.0 >= tp
            stop = (lows / entry - 1.0) * 100.0 <= -sl
        else:
            take = (entry / lows - 1.0) * 100.0 >= tp
            stop = (entry / highs - 1.0) * 100.0 <= -sl
        either = take | stop
        if not either.any():
            values.append(False)
            continue
        hit = int(np.argmax(either))
        values.append(bool(take[hit] and not stop[hit]))
    return pd.Series(values, index=candidates.index)


def countertrend_analysis(candidates: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = {
        "15m_long_while_1h_down": candidates["candidate_tf"].eq("15m") & candidates["candidate_direction"].eq("long") & candidates["upper_1h_cycle_direction"].eq("down"),
        "15m_long_while_4h_down": candidates["candidate_tf"].eq("15m") & candidates["candidate_direction"].eq("long") & candidates["upper_4h_cycle_direction"].eq("down"),
        "1h_long_while_4h_down": candidates["candidate_tf"].eq("1h") & candidates["candidate_direction"].eq("long") & candidates["upper_4h_cycle_direction"].eq("down"),
        "15m_1h_long_while_1d_down": candidates["candidate_tf"].isin(["15m", "1h"]) & candidates["candidate_direction"].eq("long") & candidates["upper_1d_cycle_direction"].eq("down"),
        "15m_short_while_1h_up": candidates["candidate_tf"].eq("15m") & candidates["candidate_direction"].eq("short") & candidates["upper_1h_cycle_direction"].eq("up"),
        "15m_short_while_4h_up": candidates["candidate_tf"].eq("15m") & candidates["candidate_direction"].eq("short") & candidates["upper_4h_cycle_direction"].eq("up"),
        "1h_short_while_4h_up": candidates["candidate_tf"].eq("1h") & candidates["candidate_direction"].eq("short") & candidates["upper_4h_cycle_direction"].eq("up"),
        "15m_1h_short_while_1d_up": candidates["candidate_tf"].isin(["15m", "1h"]) & candidates["candidate_direction"].eq("short") & candidates["upper_1d_cycle_direction"].eq("up"),
    }
    rows = []
    cond_rows = []
    for name, mask in definitions.items():
        group = candidates[mask].copy()
        upper_limit = group["timestamp"] + pd.to_timedelta(
            group["candidate_tf"].map(lambda tf: TF_SECONDS.get(str(tf), 0) * 32),
            unit="s",
        )
        upper_reversed_within_window = group["upper_tf_opposite_true_time"].notna() & group["upper_tf_opposite_true_time"].le(upper_limit)
        row = {
            "countertrend_group": name,
            "n": len(group),
            "low_sample": len(group) < LOW_SAMPLE_N,
            "avg_return_32bars": group["forward_return_32bars"].mean(),
            "median_return_32bars": group["forward_return_32bars"].median(),
            "avg_mfe_32bars": group["mfe_32bars"].mean(),
            "avg_mae_32bars": group["mae_32bars"].mean(),
            "upper_tf_reversal_within_32bars_pct": pct_bool(upper_reversed_within_window),
            "noise_but_profitable_32bars_pct": pct_bool(group["is_noise"].eq(1) & (group["forward_return_32bars"] > 0)),
        }
        for tp in (0.8, 1.2, 1.8):
            row[f"win_rate_tp_{tp}_pct"] = pct_bool(group["mfe_32bars"] >= tp)
            if len(group):
                row[f"tp_{tp}_before_sl_0.6_pct"] = pct_bool(tp_before_sl_flags(group, candles, tp, 0.6))
        rows.append(row)

        for label, submask in {
            "own_extreme": (is_bottom20(group["ppo_bin"]) if "long" in name else is_top20(group["ppo_bin"])) & (is_bottom20(group["hist_bin"]) if "long" in name else is_top20(group["hist_bin"])),
            "h1_hist_improving_or_worsening": (group["upper_1h_hist_diff"] > 0) if "long" in name else (group["upper_1h_hist_diff"] < 0),
            "not_1d_strong_against": ~((group["upper_1d_hist"] > 0) & is_top20(group["upper_1d_ppo_bin"])) if "long" in name else ~((group["upper_1d_hist"] < 0) & is_bottom20(group["upper_1d_ppo_bin"])),
        }.items():
            sub = group[submask]
            cond_rows.append({
                "countertrend_group": name,
                "condition": label,
                "n": len(sub),
                "low_sample": len(sub) < LOW_SAMPLE_N,
                "tp_0.8_pct": pct_bool(sub["mfe_32bars"] >= 0.8),
                "tp_1.2_pct": pct_bool(sub["mfe_32bars"] >= 1.2),
                "avg_return_32bars": sub["forward_return_32bars"].mean(),
                "avg_mfe_32bars": sub["mfe_32bars"].mean(),
                "avg_mae_32bars": sub["mae_32bars"].mean(),
            })
    return pd.DataFrame(rows), pd.DataFrame(cond_rows)


def practical_rule_masks(candidates: pd.DataFrame) -> dict[str, pd.Series]:
    tf15 = candidates["candidate_tf"].eq("15m")
    tf15or1h = candidates["candidate_tf"].isin(["15m", "1h"])
    long = candidates["candidate_direction"].eq("long")
    short = candidates["candidate_direction"].eq("short")
    h1_long_bias = candidates["upper_1h_hist"] < 0
    h1_short_bias = candidates["upper_1h_hist"] > 0
    h4_long_bias = candidates["upper_4h_hist"] < 0
    h4_short_bias = candidates["upper_4h_hist"] > 0
    d1_short_bias = candidates["upper_1d_hist"] > 0
    d1_strong_short = d1_short_bias & is_top20(candidates["upper_1d_ppo_bin"])
    d1_strong_long = (candidates["upper_1d_hist"] < 0) & is_bottom20(candidates["upper_1d_ppo_bin"])
    own_bottom = is_bottom20(candidates["ppo_bin"]) & is_bottom20(candidates["hist_bin"])
    own_top = is_top20(candidates["ppo_bin"]) & is_top20(candidates["hist_bin"])
    h1_bottom = is_bottom20(candidates["upper_1h_ppo_bin"]) & is_bottom20(candidates["upper_1h_hist_bin"])
    h1_top = is_top20(candidates["upper_1h_ppo_bin"]) & is_top20(candidates["upper_1h_hist_bin"])
    h1_improving = candidates["upper_1h_hist_diff"] > 0
    h1_worsening = candidates["upper_1h_hist_diff"] < 0
    return {
        "L1_long_trend_following": tf15 & long & h1_long_bias & h4_long_bias & ~d1_strong_short,
        "L2_long_early_reversal": tf15 & long & own_bottom & h1_bottom & h1_improving & ~h4_short_bias & ~d1_strong_short,
        "L3_long_countertrend_scalp": tf15 & long & candidates["upper_1h_cycle_direction"].eq("down") & own_bottom & h1_improving,
        "L4_long_1h_confirmed_proxy": tf15 & long & candidates["upper_1h_cycle_direction"].eq("up") & ~h4_short_bias,
        "S1_short_trend_following": tf15 & short & h1_short_bias & h4_short_bias & d1_short_bias,
        "S2_short_countertrend_scalp": tf15 & short & candidates["upper_1h_cycle_direction"].eq("up") & own_top & h1_worsening,
        "S3_short_1d_strong_short": tf15or1h & short & d1_strong_short & h4_short_bias & ~d1_strong_long,
    }


def simulate_trade(
    row: pd.Series,
    candles: dict[str, pd.DataFrame],
    method: str,
    events: dict[tuple[str, str, bool], pd.DataFrame],
    tp: float | None = None,
    sl: float | None = None,
    atr_mult: float | None = None,
) -> dict[str, Any]:
    tf = str(row["candidate_tf"])
    direction = str(row["candidate_direction"])
    market = candles[tf]
    idx = int(row["bar_index"])
    entry = float(row["close_at_entry"])
    opposite = "short" if direction == "long" else "long"
    fallback_time, fallback_price = next_event(events[(tf, opposite, True)], row["timestamp"])
    if fallback_time is None:
        end_idx = min(len(market) - 1, idx + 32)
        fallback_time = market.iloc[end_idx]["timestamp"]
        fallback_price = float(market.iloc[end_idx]["close"])
    fallback_pos = int(np.searchsorted(market["timestamp"].to_numpy(dtype="datetime64[ns]"), np.datetime64(fallback_time), side="right"))
    # Stop/TP methods are short-horizon risk controls. Scanning all the way to a
    # distant opposite cycle confirmation makes the grid unnecessarily slow and
    # overstates how long a fixed stop would be monitored without re-evaluation.
    risk_end_pos = min(max(idx + 2, fallback_pos), idx + 33)
    segment = market.iloc[idx + 1 : risk_end_pos]
    exit_time = fallback_time
    exit_price = fallback_price
    reason = "opposite_true_reversal"

    if method.startswith("fixed_sl_"):
        t, p, r = first_tp_sl(segment, entry, direction, sl_pct=sl)
        if t is not None:
            exit_time, exit_price, reason = t, p, r
    elif method.startswith("tp_sl_"):
        t, p, r = first_tp_sl(segment, entry, direction, tp_pct=tp, sl_pct=sl)
        if t is not None:
            exit_time, exit_price, reason = t, p, r
    elif method == "candidate_invalidation":
        t, p, r = first_tp_sl(segment, entry, direction, invalid_low=float(row["low"]) if direction == "long" else None, invalid_high=float(row["high"]) if direction == "short" else None)
        if t is not None:
            exit_time, exit_price, reason = t, p, r
    elif method.startswith("atr_stop_"):
        atr = float(row.get("own_atr_14", np.nan))
        if np.isfinite(atr) and entry > 0:
            sl_atr = atr / entry * 100.0 * float(atr_mult)
            t, p, r = first_tp_sl(segment, entry, direction, sl_pct=sl_atr)
            if t is not None:
                exit_time, exit_price, reason = t, p, "atr_stop"
    elif method == "time_stop_4bar_no_0.4":
        check = market.iloc[idx + 1 : min(len(market), idx + 5)]
        if not check.empty:
            mfe = ((check["high"].max() / entry - 1.0) * 100.0) if direction == "long" else ((entry / check["low"].min() - 1.0) * 100.0)
            if mfe < 0.4:
                last = check.iloc[-1]
                exit_time, exit_price, reason = last["timestamp"], float(last["close"]), "time_stop"
    elif method == "time_stop_8bar_no_0.8":
        check = market.iloc[idx + 1 : min(len(market), idx + 9)]
        if not check.empty:
            mfe = ((check["high"].max() / entry - 1.0) * 100.0) if direction == "long" else ((entry / check["low"].min() - 1.0) * 100.0)
            if mfe < 0.8:
                last = check.iloc[-1]
                exit_time, exit_price, reason = last["timestamp"], float(last["close"]), "time_stop"
    elif method == "partial_tp_0.8_then_opposite":
        t, p, r = first_tp_sl(segment, entry, direction, tp_pct=0.8)
        if t is not None:
            gross_rest = _direction_return(entry, fallback_price, direction)
            gross_tp = _direction_return(entry, p, direction)
            blended = 0.5 * gross_tp + 0.5 * gross_rest
            return {
                "exit_time": fallback_time,
                "exit_price": fallback_price,
                "gross_return": blended,
                "exit_reason": "partial_tp_then_opposite",
            }

    gross = _direction_return(entry, float(exit_price), direction)
    return {"exit_time": exit_time, "exit_price": exit_price, "gross_return": gross, "exit_reason": reason}


def summarize_trades(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in trades.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_trades": len(group),
            "low_sample": len(group) < LOW_SAMPLE_N,
            "win_rate": pct_bool(group["net_return"] > 0),
            "avg_return_net": group["net_return"].mean(),
            "median_return_net": group["net_return"].median(),
            "profit_factor": profit_factor(group["net_return"]),
            "max_drawdown": max_drawdown(group["net_return"]),
            "avg_MFE": group["mfe_32bars"].mean(),
            "avg_MAE": group["mae_32bars"].mean(),
            "avg_holding_hours": group["holding_hours"].mean(),
            "stop_hit_rate": pct_bool(group["exit_reason"].astype(str).str.contains("stop", case=False, na=False)),
            "tp_hit_rate": pct_bool(group["exit_reason"].astype(str).str.contains("take_profit|partial_tp", case=False, regex=True, na=False)),
            "invalidation_rate": pct_bool(group["exit_reason"].eq("invalidation")),
            "time_stop_rate": pct_bool(group["exit_reason"].eq("time_stop")),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def practical_rule_backtest(candidates: pd.DataFrame, candles: dict[str, pd.DataFrame], events: dict[tuple[str, str, bool], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods: list[tuple[str, float | None, float | None, float | None]] = [("opposite_true_reversal_confirmed_close", None, None, None)]
    methods += [(f"fixed_sl_{sl}", None, sl, None) for sl in SL_VALUES]
    methods += [(f"tp_sl_{tp}_{sl}", tp, sl, None) for tp in TP_VALUES for sl in GRID_SL_VALUES]
    methods += [("candidate_invalidation", None, None, None)]
    methods += [(f"atr_stop_{mult}", None, None, mult) for mult in ATR_MULTS]
    methods += [("time_stop_4bar_no_0.4", None, None, None), ("time_stop_8bar_no_0.8", None, None, None), ("partial_tp_0.8_then_opposite", None, None, None)]

    masks = practical_rule_masks(candidates)
    entry_ids = sorted(set().union(*[set(candidates.index[mask]) for mask in masks.values()]))
    entry_frame = candidates.loc[entry_ids].copy()

    # Precompute each candidate/method outcome once, then attach rule names. This
    # avoids re-scanning the same 32-bar window for every rule that includes it.
    outcome_by_candidate: dict[int, list[dict[str, Any]]] = {}
    candle_arrays = {
        tf: {
            "timestamp": frame["timestamp"].to_numpy(),
            "high": frame["high"].to_numpy(dtype="float64"),
            "low": frame["low"].to_numpy(dtype="float64"),
            "close": frame["close"].to_numpy(dtype="float64"),
        }
        for tf, frame in candles.items()
    }

    def first_true(values: np.ndarray) -> int | None:
        if not values.any():
            return None
        return int(np.argmax(values))

    for cid, row in entry_frame.iterrows():
        tf = str(row["candidate_tf"])
        arr = candle_arrays[tf]
        idx = int(row["bar_index"])
        entry = float(row["close_at_entry"])
        direction = str(row["candidate_direction"])
        sign = _candidate_sign(direction)
        start = idx + 1
        end = min(len(arr["close"]), idx + 33)
        highs = arr["high"][start:end]
        lows = arr["low"][start:end]
        closes = arr["close"][start:end]
        times = arr["timestamp"][start:end]
        if len(closes) == 0 or not np.isfinite(entry):
            continue

        fallback_gross = row["return_until_same_tf_opposite"]
        if not np.isfinite(fallback_gross):
            fallback_gross = row["forward_return_32bars"]
        fallback_time = row.get("same_tf_opposite_true_time")
        if pd.isna(fallback_time):
            fallback_time = pd.Timestamp(times[-1])
        fallback_hours = (pd.Timestamp(fallback_time) - row["timestamp"]).total_seconds() / 3600.0

        if sign > 0:
            fav = (highs / entry - 1.0) * 100.0
            adv = (lows / entry - 1.0) * 100.0
            invalid = lows <= float(row["low"])
        else:
            fav = (entry / lows - 1.0) * 100.0
            adv = (entry / highs - 1.0) * 100.0
            invalid = highs >= float(row["high"])

        outcomes = []
        for method, tp, sl, atr_mult in methods:
            gross = float(fallback_gross)
            exit_time = pd.Timestamp(fallback_time)
            exit_reason = "opposite_true_reversal"

            if method.startswith("fixed_sl_"):
                hit = first_true(adv <= -float(sl))
                if hit is not None:
                    gross = -float(sl)
                    exit_time = pd.Timestamp(times[hit])
                    exit_reason = "stop_loss"
            elif method.startswith("tp_sl_"):
                stop = adv <= -float(sl)
                take = fav >= float(tp)
                either = stop | take
                hit = first_true(either)
                if hit is not None:
                    exit_time = pd.Timestamp(times[hit])
                    if stop[hit]:
                        gross = -float(sl)
                        exit_reason = "stop_loss"
                    else:
                        gross = float(tp)
                        exit_reason = "take_profit"
            elif method == "candidate_invalidation":
                hit = first_true(invalid)
                if hit is not None:
                    gross = _direction_return(entry, float(closes[hit]), direction)
                    exit_time = pd.Timestamp(times[hit])
                    exit_reason = "invalidation"
            elif method.startswith("atr_stop_"):
                atr = float(row.get("own_atr_14", np.nan))
                if np.isfinite(atr) and entry > 0:
                    atr_sl = atr / entry * 100.0 * float(atr_mult)
                    hit = first_true(adv <= -atr_sl)
                    if hit is not None:
                        gross = -atr_sl
                        exit_time = pd.Timestamp(times[hit])
                        exit_reason = "atr_stop"
            elif method == "time_stop_4bar_no_0.4":
                horizon = min(4, len(fav))
                if horizon and np.nanmax(fav[:horizon]) < 0.4:
                    gross = _direction_return(entry, float(closes[horizon - 1]), direction)
                    exit_time = pd.Timestamp(times[horizon - 1])
                    exit_reason = "time_stop"
            elif method == "time_stop_8bar_no_0.8":
                horizon = min(8, len(fav))
                if horizon and np.nanmax(fav[:horizon]) < 0.8:
                    gross = _direction_return(entry, float(closes[horizon - 1]), direction)
                    exit_time = pd.Timestamp(times[horizon - 1])
                    exit_reason = "time_stop"
            elif method == "partial_tp_0.8_then_opposite":
                hit = first_true(fav >= 0.8)
                if hit is not None:
                    gross = 0.5 * 0.8 + 0.5 * float(fallback_gross)
                    exit_reason = "partial_tp_then_opposite"

            holding_hours = (exit_time - row["timestamp"]).total_seconds() / 3600.0
            outcomes.append({
                "stop_method": method,
                "entry_time": row["timestamp"],
                "exit_time": exit_time,
                "candidate_tf": tf,
                "direction": direction,
                "entry_price": entry,
                "gross_return": gross,
                "net_return": gross - POSITION_COST_PCT,
                "exit_reason": exit_reason,
                "holding_hours": holding_hours if np.isfinite(holding_hours) else fallback_hours,
                "mfe_32bars": row["mfe_32bars"],
                "mae_32bars": row["mae_32bars"],
                "ppo_bin": row["ppo_bin"],
                "hist_bin": row["hist_bin"],
                "upper_1h_cycle_direction": row.get("upper_1h_cycle_direction"),
                "upper_4h_cycle_direction": row.get("upper_4h_cycle_direction"),
                "upper_1d_cycle_direction": row.get("upper_1d_cycle_direction"),
            })
        outcome_by_candidate[cid] = outcomes

    trade_rows = []
    for rule, mask in masks.items():
        for cid in candidates.index[mask]:
            for outcome in outcome_by_candidate.get(int(cid), []):
                item = {"rule_name": rule, **outcome}
                trade_rows.append(item)
    trades = pd.DataFrame(trade_rows).sort_values(["rule_name", "stop_method", "entry_time"]).reset_index(drop=True)
    summary = summarize_trades(trades, ["rule_name", "stop_method"])
    return summary, trades


def period_label(ts: pd.Timestamp) -> str:
    year = ts.year
    if 2017 <= year <= 2020:
        return "2017-2020"
    if year == 2021:
        return "2021"
    if year == 2022:
        return "2022"
    if 2023 <= year <= 2025:
        return "2023-2025"
    if year == 2026:
        return "2026"
    return "other"


def top_rows(df: pd.DataFrame, sort_col: str, n: int = 8, ascending: bool = False) -> str:
    if df.empty or sort_col not in df:
        return "_No rows._"
    cols = [col for col in ["rule_name", "stop_method", "countertrend_group", "candidate_tf", "candidate_direction", "n", "n_trades", "win_rate", "avg_return_net", "profit_factor", "max_drawdown", sort_col, "low_sample"] if col in df.columns]
    return df.sort_values(sort_col, ascending=ascending).head(n)[cols].to_markdown(index=False)


def write_report(
    out: Path,
    success: pd.DataFrame,
    profiles: pd.DataFrame,
    bucket: pd.DataFrame,
    counter: pd.DataFrame,
    stop_summary: pd.DataFrame,
    period: pd.DataFrame,
) -> None:
    robust = stop_summary[(~stop_summary["low_sample"]) & stop_summary["n_trades"].ge(100)].copy()
    if not robust.empty:
        robust["score"] = robust["avg_return_net"].fillna(0) + robust["profit_factor"].replace(np.inf, 5).fillna(0) * 0.2 + robust["max_drawdown"].fillna(-100) * 0.03
    lines = [
        "# PPO Successful Long/Short, Countertrend Noise, Stop Analysis",
        "",
        "## 1. 분석 목적",
        "성공한 롱/숏 후보의 사전 MTF 상태, counter-trend noise의 단기 수익화 가능성, 그리고 기존 opposite true reversal 청산 대비 다양한 stop/exit 방식을 비교했다.",
        "",
        "## 2. 기존 분석과 차이",
        "이번 분석은 후보 row를 다시 대량 저장하지 않고, 기존 `20_reversal_candidates.csv`에 forward MFE/MAE와 룰별 청산 결과를 붙여 요약 산출물만 만든다. 상위 TF feature는 기존 closed-only join 값을 사용했다.",
        "",
        "## 3. 성공 롱/숏 정의",
        "`small_win`은 32 bars 내 MFE 0.8% 이상, `mid_win`은 1.2%, `strong_win`은 1.8%, `big_win`은 2.5%, `clean_win`은 MFE 1.2% 이상이면서 MAE가 -0.6%보다 양호한 경우, `trap`은 +0.5% 도달 전에 -1.0% MAE가 먼저 발생한 경우다.",
        "",
        "## 4. 성공 거래의 사전 MTF profile",
        success.to_markdown(index=False),
        "",
        "## 5. 실패 거래의 사전 MTF profile",
        top_rows(profiles[profiles["profile_group"].eq("trap")], "n", n=10, ascending=False),
        "",
        "## 6. 가격변화율 bucket별 MTF 상태",
        top_rows(bucket[bucket["bucket"].isin(["top10_return", "bottom10_return", "trap"])], "n", n=12, ascending=False),
        "",
        "## 7. Counter-trend noise 수익화 가능성",
        counter.to_markdown(index=False),
        "",
        "## 8. Long 조건과 Short 조건의 비대칭성",
        "롱/숏은 동일한 반대 조건으로 취급하지 않았다. 룰 L1~L4, S1~S3을 분리했고, 1d strong short는 숏 허용 조건이면서 롱 회피 조건으로 별도 반영했다.",
        "",
        "## 9. 손절/청산 방식별 비교",
        top_rows(stop_summary, "avg_return_net", n=15, ascending=False),
        "",
        "## 10. TP/SL grid 결과",
        top_rows(stop_summary[stop_summary["stop_method"].astype(str).str.startswith("tp_sl_")], "profit_factor", n=15, ascending=False),
        "",
        "## 11. invalidation stop 결과",
        top_rows(stop_summary[stop_summary["stop_method"].eq("candidate_invalidation")], "avg_return_net", n=10, ascending=False),
        "",
        "## 12. ATR/time stop 결과",
        top_rows(stop_summary[stop_summary["stop_method"].astype(str).str.contains("atr_stop|time_stop", regex=True)], "avg_return_net", n=15, ascending=False),
        "",
        "## 13. 실전 rule L1~L4, S1~S3 결과",
        top_rows(stop_summary, "profit_factor", n=20, ascending=False),
        "",
        "## 14. 기간별 안정성",
        top_rows(period, "avg_return_net", n=20, ascending=False),
        "",
        "## 15. 최종 자동매매 후보",
        top_rows(robust if not robust.empty else stop_summary, "score" if not robust.empty else "avg_return_net", n=10, ascending=False),
        "",
        "## 16. 피해야 할 조건",
        "low_sample이거나 trap 비율이 높고, 1d strong-short에 역행하는 롱/1d strong-long에 역행하는 숏은 자동매매 후보에서 제외해야 한다. 특히 counter-trend는 평균 수익보다 TP-before-SL과 MAE를 우선 봐야 한다.",
        "",
        "## 17. 현재 실시간 상황에 적용하는 체크리스트",
        "15m 후보 방향, 1h hist 개선/악화, 4h bias, 1d strong regime 여부, candidate low/high invalidation 위치, ATR 기반 손절폭을 순서대로 확인한다.",
        "",
        "## 18. 한계와 다음 분석",
        "이번 구현의 `L4`는 기존 15m 후보 시점에서 이미 닫힌 1h UP 상태인 proxy다. 실제 '15m 관측 후 1h confirmed entry'는 별도 이벤트 기반 엔트리 재구성이 필요하다. 1h swing low/high invalidation도 현재는 candidate candle high/low invalidation 중심으로 구현했다.",
    ]
    (out / "PPO_successful_long_short_countertrend_stop_report.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Path]:
    out = output_dir()
    candles = {tf: load_raw_candles(tf) for tf in TIMEFRAMES}
    candidates = load_candidates()
    candidates = add_own_market_features(candidates, candles)
    candidates = add_forward_metrics(candidates, candles)
    events = build_event_tables(candidates)
    candidates = add_event_returns(candidates, events)

    success = summarize_success(candidates)
    profiles = feature_profile(candidates)
    counter, counter_conditions = countertrend_analysis(candidates, candles)
    bucket = return_bucket_profile(candidates)
    stop_summary, stop_details = practical_rule_backtest(candidates, candles, events)
    stop_details["period"] = stop_details["entry_time"].map(period_label)
    period = summarize_trades(stop_details, ["rule_name", "stop_method", "period"])

    paths = {
        "46_successful_long_short_precondition_summary.csv": out / "46_successful_long_short_precondition_summary.csv",
        "47_successful_trade_feature_profiles.csv": out / "47_successful_trade_feature_profiles.csv",
        "48_countertrend_noise_profitability.csv": out / "48_countertrend_noise_profitability.csv",
        "49_countertrend_success_conditions.csv": out / "49_countertrend_success_conditions.csv",
        "50_forward_return_bucket_mtf_profile.csv": out / "50_forward_return_bucket_mtf_profile.csv",
        "51_stop_loss_method_comparison.csv": out / "51_stop_loss_method_comparison.csv",
        "52_stop_loss_trade_details.csv": out / "52_stop_loss_trade_details.csv",
        "53_practical_countertrend_rule_backtest.csv": out / "53_practical_countertrend_rule_backtest.csv",
        "54_rule_period_stability.csv": out / "54_rule_period_stability.csv",
    }
    success.to_csv(paths["46_successful_long_short_precondition_summary.csv"], index=False, encoding="utf-8-sig")
    profiles.to_csv(paths["47_successful_trade_feature_profiles.csv"], index=False, encoding="utf-8-sig")
    counter.to_csv(paths["48_countertrend_noise_profitability.csv"], index=False, encoding="utf-8-sig")
    counter_conditions.to_csv(paths["49_countertrend_success_conditions.csv"], index=False, encoding="utf-8-sig")
    bucket.to_csv(paths["50_forward_return_bucket_mtf_profile.csv"], index=False, encoding="utf-8-sig")
    stop_summary.to_csv(paths["51_stop_loss_method_comparison.csv"], index=False, encoding="utf-8-sig")
    # Minimal trade-level file for stop/rule audit only, not a repeated candidate feature dump.
    stop_details.to_csv(paths["52_stop_loss_trade_details.csv"], index=False, encoding="utf-8-sig")
    stop_summary.to_csv(paths["53_practical_countertrend_rule_backtest.csv"], index=False, encoding="utf-8-sig")
    period.to_csv(paths["54_rule_period_stability.csv"], index=False, encoding="utf-8-sig")
    write_report(out, success, profiles, bucket, counter, stop_summary, period)
    paths["PPO_successful_long_short_countertrend_stop_report.md"] = out / "PPO_successful_long_short_countertrend_stop_report.md"
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze successful long/short candidates, countertrend noise, and stop methods.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    paths = run()
    print(f"Saved outputs to: {output_dir()}")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
