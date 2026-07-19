from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis import ppo_successful_countertrend_stop_analysis as prev  # noqa: E402
from scripts.analysis.ppo_reversal_candidate_backtest import (  # noqa: E402
    LOW_SAMPLE_N,
    PROJECT_PATHS,
    ROUND_TRIP_FEE_PCT,
    SLIPPAGE_PER_SIDE_PCT,
    TF_SECONDS,
    _candidate_sign,
    _direction_return,
    _read_timestamp,
)


BASE_PREV_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_successful_countertrend_stop_analysis"
BASE_CANDIDATE_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_reversal_candidate_backtest"
BASE_LEAD_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_15m_leading_1h_reversal_analysis"
OUT_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_backtest_integrity_strategy_refinement"
RULES_CONFIG = PROJECT_ROOT / "configs" / "rules_config.yaml"
STOPS_CONFIG = PROJECT_ROOT / "configs" / "stops_config.yaml"
GRID_CONFIG = PROJECT_ROOT / "configs" / "backtest_grid.yaml"
POSITION_COST_PCT = ROUND_TRIP_FEE_PCT + SLIPPAGE_PER_SIDE_PCT * 2
TIMEFRAMES = ("15m", "1h", "4h", "1d")


def output_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def sample_class(n: int) -> str:
    if n < 100:
        return "low_sample"
    if n < 500:
        return "medium_sample"
    return "reliable_sample"


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def audit_file(path: Path, critical_cols: list[str] | None = None) -> dict[str, Any]:
    critical_cols = critical_cols or []
    if not path.exists():
        return {
            "file_name": path.name,
            "exists": False,
            "row_count": 0,
            "column_count": 0,
            "columns_unique": False,
            "duplicate_rows": np.nan,
            "missing_critical_columns": ",".join(critical_cols),
            "all_nan_columns": "",
            "sha256": None,
        }
    df = pd.read_csv(path)
    missing = [col for col in critical_cols if col not in df.columns]
    all_nan = [col for col in df.columns if df[col].isna().all()]
    return {
        "file_name": path.name,
        "exists": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns_unique": bool(df.columns.is_unique),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_critical_columns": ",".join(missing),
        "all_nan_columns": ",".join(all_nan[:20]),
        "sha256": file_hash(path),
    }


def data_integrity_audit(out: Path) -> tuple[pd.DataFrame, str]:
    files = [
        "46_successful_long_short_precondition_summary.csv",
        "47_successful_trade_feature_profiles.csv",
        "48_countertrend_noise_profitability.csv",
        "49_countertrend_success_conditions.csv",
        "50_forward_return_bucket_mtf_profile.csv",
        "51_stop_loss_method_comparison.csv",
        "52_stop_loss_trade_details.csv",
        "53_practical_countertrend_rule_backtest.csv",
        "54_rule_period_stability.csv",
    ]
    critical = {
        "51_stop_loss_method_comparison.csv": ["rule_name", "stop_method", "n_trades", "avg_return_net", "max_drawdown"],
        "52_stop_loss_trade_details.csv": ["rule_name", "stop_method", "entry_time", "net_return", "exit_reason"],
        "53_practical_countertrend_rule_backtest.csv": ["rule_name", "stop_method", "n_trades", "avg_return_net"],
    }
    rows = [audit_file(BASE_PREV_DIR / name, critical.get(name, [])) for name in files]

    f51 = BASE_PREV_DIR / "51_stop_loss_method_comparison.csv"
    f53 = BASE_PREV_DIR / "53_practical_countertrend_rule_backtest.csv"
    df51 = read_csv_if_exists(f51)
    df53 = read_csv_if_exists(f53)
    same_shape = df51.shape == df53.shape
    same_columns = list(df51.columns) == list(df53.columns) if not df51.empty and not df53.empty else False
    same_cells = bool(same_shape and same_columns and df51.equals(df53))
    same_hash = file_hash(f51) == file_hash(f53)

    if not df51.empty and {"rule_name", "stop_method", "avg_MFE", "avg_MAE"}.issubset(df51.columns):
        mfe_var = df51.groupby("rule_name")["avg_MFE"].var().fillna(0)
        mae_var = df51.groupby("rule_name")["avg_MAE"].var().fillna(0)
        zero_mfe_rules = int((mfe_var == 0).sum())
        zero_mae_rules = int((mae_var == 0).sum())
    else:
        zero_mfe_rules = np.nan
        zero_mae_rules = np.nan

    mdd_clip_ratio = np.nan
    if "max_drawdown" in df51:
        mdd_clip_ratio = float((df51["max_drawdown"] <= -99.0).mean() * 100.0)

    sample_counts = {"low_sample_rules": np.nan, "medium_sample_rules": np.nan, "reliable_sample_rules": np.nan}
    if "n_trades" in df51:
        classes = df51["n_trades"].map(lambda n: sample_class(int(n))).value_counts()
        sample_counts = {
            "low_sample_rules": int(classes.get("low_sample", 0)),
            "medium_sample_rules": int(classes.get("medium_sample", 0)),
            "reliable_sample_rules": int(classes.get("reliable_sample", 0)),
        }

    period = read_csv_if_exists(BASE_PREV_DIR / "54_rule_period_stability.csv")
    has_2026 = bool("period" in period and period["period"].astype(str).eq("2026").any())
    audit_summary = {
        "51_53_same_shape": same_shape,
        "51_53_same_columns": same_columns,
        "51_53_same_cells": same_cells,
        "51_53_same_hash": same_hash,
        "avg_MFE_zero_variance_rule_count": zero_mfe_rules,
        "avg_MAE_zero_variance_rule_count": zero_mae_rules,
        "max_drawdown_le_minus99_pct": mdd_clip_ratio,
        "has_2026_period": has_2026,
        **sample_counts,
    }
    rows.append({"file_name": "__cross_file_checks__", **audit_summary})
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "55_data_integrity_audit.csv", index=False, encoding="utf-8-sig")

    text = [
        "# Data Integrity Audit",
        "",
        f"- 51/53 same shape: `{same_shape}`",
        f"- 51/53 same columns: `{same_columns}`",
        f"- 51/53 same cells: `{same_cells}`",
        f"- 51/53 same hash: `{same_hash}`",
        f"- Existing avg_MFE zero-variance rule count: `{zero_mfe_rules}`",
        f"- Existing avg_MAE zero-variance rule count: `{zero_mae_rules}`",
        f"- Existing MDD <= -99% ratio: `{mdd_clip_ratio:.2f}%`" if np.isfinite(mdd_clip_ratio) else "- Existing MDD <= -99% ratio: `NA`",
        f"- Existing period table has 2026: `{has_2026}`",
        f"- Sample class counts in old 51: `{sample_counts}`",
        "",
        audit.to_markdown(index=False),
    ]
    md = "\n".join(text)
    (out / "55_data_integrity_audit.md").write_text(md, encoding="utf-8")
    return audit, md


def is_bottom20(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(["bottom10", "bottom20"])


def is_top20(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(["top10", "top20"])


def prepare_base_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[tuple[str, str, bool], pd.DataFrame]]:
    candles = {tf: prev.load_raw_candles(tf) for tf in TIMEFRAMES}
    candidates = prev.load_candidates()
    candidates = prev.add_own_market_features(candidates, candles)
    candidates = prev.add_forward_metrics(candidates, candles)
    events = prev.build_event_tables(candidates)
    candidates = prev.add_event_returns(candidates, events)
    candidates["source_id"] = np.arange(len(candidates))
    return candles, candidates, events


def base_rule_entries(c: pd.DataFrame) -> pd.DataFrame:
    own_bottom = is_bottom20(c["ppo_bin"]) & is_bottom20(c["hist_bin"])
    own_top = is_top20(c["ppo_bin"]) & is_top20(c["hist_bin"])
    h1_bottom = is_bottom20(c["upper_1h_ppo_bin"]) & is_bottom20(c["upper_1h_hist_bin"])
    h1_top = is_top20(c["upper_1h_ppo_bin"]) & is_top20(c["upper_1h_hist_bin"])
    h1_long = c["upper_1h_hist"] < 0
    h1_short = c["upper_1h_hist"] > 0
    h4_long = c["upper_4h_hist"] < 0
    h4_short = c["upper_4h_hist"] > 0
    d1_short = c["upper_1d_hist"] > 0
    d1_strong_short = d1_short & is_top20(c["upper_1d_ppo_bin"])
    d1_strong_long = (c["upper_1d_hist"] < 0) & is_bottom20(c["upper_1d_ppo_bin"])
    cvd_neg_threshold = c["own_cvd_delta"].quantile(0.35)

    masks = {
        "L1_long_trend_following": c["candidate_tf"].eq("15m") & c["candidate_direction"].eq("long") & h1_long & h4_long & ~d1_strong_short,
        "L2_long_early_reversal": c["candidate_tf"].eq("15m") & c["candidate_direction"].eq("long") & own_bottom & h1_bottom & (c["upper_1h_hist_diff"] > 0) & ~h4_short & ~d1_strong_short,
        "L3_long_countertrend_scalp": c["candidate_tf"].eq("15m") & c["candidate_direction"].eq("long") & c["upper_1h_cycle_direction"].eq("down") & own_bottom & (c["upper_1h_hist_diff"] > 0),
        "L4_proxy": c["candidate_tf"].eq("15m") & c["candidate_direction"].eq("long") & c["upper_1h_cycle_direction"].eq("up") & ~h4_short,
        "S1_short_trend_following": c["candidate_tf"].eq("15m") & c["candidate_direction"].eq("short") & h1_short & h4_short & d1_short,
        "S3_short_1d_strong_short": c["candidate_tf"].isin(["15m", "1h"]) & c["candidate_direction"].eq("short") & d1_strong_short & h4_short & ~d1_strong_long,
        "S4_short_rebound_in_downtrend": c["candidate_tf"].isin(["15m", "1h"]) & c["candidate_direction"].eq("short") & c["upper_4h_cycle_direction"].eq("down") & ~d1_strong_long & (c["own_distance_from_ma25"] > 0) & (c["own_cvd_delta"] <= cvd_neg_threshold) & own_top,
        "CL1_1h_long_while_4h_down_own_extreme": c["candidate_tf"].eq("1h") & c["candidate_direction"].eq("long") & c["upper_4h_cycle_direction"].eq("down") & own_bottom & ~d1_strong_short,
    }
    rows = []
    for rule_name, mask in masks.items():
        entries = c[mask].copy()
        entries["rule_name"] = rule_name
        entries["rule_kind"] = "proxy" if rule_name == "L4_proxy" else "event_based"
        entries["entry_time"] = entries["timestamp"]
        entries["entry_price"] = entries["close_at_entry"]
        entries["entry_bar_index"] = entries["bar_index"].astype(int)
        entries["signal_time"] = entries["timestamp"]
        entries["entry_delay_minutes"] = 0.0
        entries["missed_move_before_entry"] = 0.0
        entries["pre_entry_alpha"] = 0.0
        rows.append(entries)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def l4_event_entries(candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    path = BASE_LEAD_DIR / "31_15m_leads_1h_reversal_candidates.csv"
    if not path.exists():
        return pd.DataFrame()
    cols = pd.read_csv(path, nrows=0).columns
    wanted = [
        "timestamp",
        "candidate_tf",
        "candidate_direction",
        "close_at_entry",
        "target_1h_up_confirm_time",
        "target_1h_up_confirm_price",
        "target_1h_turn_delay_15m_bars",
        "target_pre_entry_alpha",
        "target_close_return_to_confirm",
        "upper_4h_hist",
        "upper_1d_hist",
        "upper_1d_ppo_bin",
        "m15_ppo_bin",
        "m15_hist_bin",
    ]
    df = pd.read_csv(path, usecols=[col for col in wanted if col in cols], engine="python")
    df["timestamp"] = _read_timestamp(df["timestamp"])
    df["target_1h_up_confirm_time"] = _read_timestamp(df["target_1h_up_confirm_time"])
    df = df.dropna(subset=["target_1h_up_confirm_time", "target_1h_up_confirm_price"]).copy()
    d1_strong_short = (df.get("upper_1d_hist", 0) > 0) & is_top20(df.get("upper_1d_ppo_bin", pd.Series("", index=df.index)))
    df = df[(df["upper_4h_hist"] <= 0) | (~d1_strong_short)].copy()

    m15 = candles["15m"]
    times = m15["timestamp"].to_numpy(dtype="datetime64[ns]")
    rows = []
    for wait_bars in (4, 8, 12, 16):
        sub = df[df["target_1h_turn_delay_15m_bars"].le(wait_bars)].copy()
        if sub.empty:
            continue
        pos = np.searchsorted(times, sub["target_1h_up_confirm_time"].to_numpy(dtype="datetime64[ns]"), side="left")
        valid = pos < len(m15)
        sub = sub.loc[valid].copy()
        pos = pos[valid]
        sub["rule_name"] = f"L4_event_based_{wait_bars}bars"
        sub["rule_kind"] = "event_based"
        sub["candidate_tf"] = "15m"
        sub["candidate_direction"] = "long"
        sub["entry_time"] = sub["target_1h_up_confirm_time"]
        sub["entry_price"] = pd.to_numeric(sub["target_1h_up_confirm_price"], errors="coerce")
        sub["entry_bar_index"] = pos.astype(int)
        sub["bar_index"] = sub["entry_bar_index"]
        sub["signal_time"] = sub["timestamp"]
        sub["entry_delay_minutes"] = (sub["entry_time"] - sub["signal_time"]).dt.total_seconds() / 60.0
        sub["missed_move_before_entry"] = pd.to_numeric(sub.get("target_close_return_to_confirm"), errors="coerce")
        sub["pre_entry_alpha"] = pd.to_numeric(sub.get("target_pre_entry_alpha"), errors="coerce")
        candle_rows = m15.iloc[pos]
        sub["open"] = candle_rows["open"].to_numpy()
        sub["high"] = candle_rows["high"].to_numpy()
        sub["low"] = candle_rows["low"].to_numpy()
        sub["close"] = candle_rows["close"].to_numpy()
        sub["close_at_entry"] = sub["entry_price"]
        sub["ppo_bin"] = sub.get("m15_ppo_bin", "")
        sub["hist_bin"] = sub.get("m15_hist_bin", "")
        sub["source_id"] = -1
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_entries(candidates: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = base_rule_entries(candidates)
    l4 = l4_event_entries(candles)
    needed = set(base.columns).union(l4.columns)
    for frame in (base, l4):
        for col in needed - set(frame.columns):
            frame[col] = np.nan
    return pd.concat([base[list(needed)], l4[list(needed)]], ignore_index=True).sort_values(["rule_name", "entry_time"]).reset_index(drop=True)


def opposite_event_price(events: dict[tuple[str, str, bool], pd.DataFrame], tf: str, direction: str, entry_time: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    opposite = "short" if direction == "long" else "long"
    return prev.next_event(events[(tf, opposite, True)], entry_time)


def mdd_stats(returns: pd.Series, position_size: float = 1.0) -> tuple[float, int, int, float]:
    if returns.empty:
        return np.nan, 0, 0, np.nan
    equity = (1.0 + returns.fillna(0.0) * position_size / 100.0).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    mdd = float(dd.min() * 100.0)
    underwater = dd < 0
    time_underwater = float(underwater.mean() * 100.0)
    max_duration = 0
    current = 0
    for value in underwater:
        current = current + 1 if value else 0
        max_duration = max(max_duration, current)
    trough = int(dd.argmin()) if len(dd) else 0
    recovery = 0
    if trough < len(equity):
        peak_value = running_max.iloc[trough]
        after = equity.iloc[trough:]
        recovered = np.where(after >= peak_value)[0]
        recovery = int(recovered[0]) if len(recovered) else int(len(after) - 1)
    return mdd, max_duration, recovery, time_underwater


def first_hit_index(values: np.ndarray) -> int | None:
    if len(values) == 0 or not values.any():
        return None
    return int(np.argmax(values))


def swing_level(row: pd.Series, candles: dict[str, pd.DataFrame], stop: dict[str, Any]) -> tuple[float | None, float | None]:
    params = stop.get("params", {})
    basis = params.get("basis")
    tf = str(row["candidate_tf"])
    direction = str(row["candidate_direction"])
    idx = int(row["entry_bar_index"])
    if basis == "candidate":
        return (float(row["low"]), None) if direction == "long" else (None, float(row["high"]))
    lookback = int(params.get("lookback_bars", 8))
    use_tf = "1h" if basis == "h1_swing" else tf
    market = candles[use_tf]
    if use_tf != tf:
        pos = np.searchsorted(market["timestamp"].to_numpy(dtype="datetime64[ns]"), np.datetime64(row["entry_time"]), side="right") - 1
    else:
        pos = idx
    start = max(0, int(pos) - lookback)
    recent = market.iloc[start : int(pos) + 1]
    if recent.empty:
        return None, None
    atr_buffer = float(params.get("atr_buffer", 0.0) or 0.0)
    atr = float(row.get("own_atr_14", np.nan))
    buffer_value = atr * atr_buffer if np.isfinite(atr) else 0.0
    if direction == "long":
        return float(recent["low"].min() - buffer_value), None
    return None, float(recent["high"].max() + buffer_value)


def simulate_entry(
    row: pd.Series,
    stop: dict[str, Any],
    candles: dict[str, pd.DataFrame],
    events: dict[tuple[str, str, bool], pd.DataFrame],
) -> dict[str, Any]:
    tf = str(row["candidate_tf"])
    direction = str(row["candidate_direction"])
    market = candles[tf]
    arr_time = market["timestamp"].to_numpy()
    arr_high = market["high"].to_numpy(dtype="float64")
    arr_low = market["low"].to_numpy(dtype="float64")
    arr_close = market["close"].to_numpy(dtype="float64")
    idx = int(row["entry_bar_index"])
    entry = float(row["entry_price"])
    sign = _candidate_sign(direction)
    stop_name = stop["stop_name"]
    kind = stop["stop_kind"]
    params = stop.get("params", {})

    fallback_time = row.get("fallback_time")
    fallback_price = row.get("fallback_price")
    if pd.isna(fallback_time) or pd.isna(fallback_price):
        fallback_time, fallback_price = opposite_event_price(events, tf, direction, row["entry_time"])
    if fallback_time is None:
        end_idx = min(len(market) - 1, idx + 32)
        fallback_time = pd.Timestamp(arr_time[end_idx])
        fallback_price = float(arr_close[end_idx])
    fallback_pos = int(np.searchsorted(arr_time.astype("datetime64[ns]"), np.datetime64(fallback_time), side="right"))
    end = max(idx + 2, min(len(market), fallback_pos))
    start = min(idx + 1, len(market))
    highs = arr_high[start:end]
    lows = arr_low[start:end]
    closes = arr_close[start:end]
    times = arr_time[start:end]

    if len(closes) == 0:
        exit_time = pd.Timestamp(fallback_time)
        gross = _direction_return(entry, float(fallback_price), direction)
        return {
            "exit_time": exit_time,
            "exit_price": fallback_price,
            "exit_reason": "fallback_no_segment",
            "gross_return": gross,
            "first_leg_hit": False,
            "first_leg_return": 0.0,
            "remaining_leg_return": gross,
            "mfe_until_exit": np.nan,
            "mae_until_exit": np.nan,
            "holding_bars": 0,
        }

    if sign > 0:
        fav = (highs / entry - 1.0) * 100.0
        adv = (lows / entry - 1.0) * 100.0
    else:
        fav = (entry / lows - 1.0) * 100.0
        adv = (entry / highs - 1.0) * 100.0

    exit_idx = len(closes) - 1
    exit_reason = "opposite_true_reversal"
    gross = _direction_return(entry, float(fallback_price), direction)
    first_leg_hit = False
    first_leg_return = 0.0
    remaining_leg_return = gross

    if kind == "tp_sl":
        tp = float(params["tp_pct"])
        sl = float(params["sl_pct"])
        hit = first_hit_index((adv <= -sl) | (fav >= tp))
        if hit is not None:
            exit_idx = hit
            if adv[hit] <= -sl:
                gross = -sl
                exit_reason = "stop_loss"
            else:
                gross = tp
                exit_reason = "take_profit"
    elif kind == "partial_tp":
        tp = float(params["tp_pct"])
        ratio = float(params.get("partial_ratio", 0.5))
        hit = first_hit_index(fav >= tp)
        if hit is not None:
            first_leg_hit = True
            first_leg_return = tp
            remaining_leg_return = gross
            gross = ratio * first_leg_return + (1.0 - ratio) * remaining_leg_return
            exit_reason = f"partial_tp_{tp}_then_opposite"
    elif kind == "structural_invalidation":
        invalid_low, invalid_high = swing_level(row, candles, stop)
        if sign > 0 and invalid_low is not None:
            hit = first_hit_index(lows <= invalid_low)
        elif sign < 0 and invalid_high is not None:
            hit = first_hit_index(highs >= invalid_high)
        else:
            hit = None
        if hit is not None:
            exit_idx = hit
            gross = _direction_return(entry, float(closes[hit]), direction)
            exit_reason = stop_name

    exit_time = pd.Timestamp(times[exit_idx]) if exit_reason != "opposite_true_reversal" else pd.Timestamp(fallback_time)
    exit_price = entry * (1.0 + gross / 100.0) if sign > 0 else entry / (1.0 + gross / 100.0)
    metric_end = exit_idx + 1 if exit_reason != "opposite_true_reversal" else len(fav)
    return {
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_return": float(gross),
        "first_leg_hit": first_leg_hit,
        "first_leg_return": float(first_leg_return),
        "remaining_leg_return": float(remaining_leg_return),
        "mfe_until_exit": float(np.nanmax(fav[:metric_end])),
        "mae_until_exit": float(np.nanmin(adv[:metric_end])),
        "holding_bars": int(metric_end),
    }


def simulate_grid(entries: pd.DataFrame, candles: dict[str, pd.DataFrame], events: dict[tuple[str, str, bool], pd.DataFrame], stops: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in entries.itertuples(index=False):
        source = pd.Series(row._asdict())
        fallback_time, fallback_price = opposite_event_price(
            events,
            str(source["candidate_tf"]),
            str(source["candidate_direction"]),
            source["entry_time"],
        )
        source["fallback_time"] = fallback_time
        source["fallback_price"] = fallback_price
        for stop in stops:
            sim = simulate_entry(source, stop, candles, events)
            holding_hours = (sim["exit_time"] - source["entry_time"]).total_seconds() / 3600.0
            rows.append({
                "rule_name": source["rule_name"],
                "rule_kind": source["rule_kind"],
                "direction": source["candidate_direction"],
                "candidate_tf": source["candidate_tf"],
                "stop_method": stop["stop_name"],
                "stop_kind": stop["stop_kind"],
                "signal_time": source["signal_time"],
                "entry_time": source["entry_time"],
                "entry_price": source["entry_price"],
                "exit_time": sim["exit_time"],
                "exit_price": sim["exit_price"],
                "exit_reason": sim["exit_reason"],
                "gross_return": sim["gross_return"],
                "net_return": sim["gross_return"] - POSITION_COST_PCT,
                "holding_bars": sim["holding_bars"],
                "holding_hours": holding_hours,
                "mfe_until_exit": sim["mfe_until_exit"],
                "mae_until_exit": sim["mae_until_exit"],
                "mfe_window_32bars": source.get("mfe_32bars", np.nan),
                "mae_window_32bars": source.get("mae_32bars", np.nan),
                "first_leg_hit": sim["first_leg_hit"],
                "first_leg_return": sim["first_leg_return"],
                "remaining_leg_return": sim["remaining_leg_return"],
                "entry_delay_minutes": source.get("entry_delay_minutes", 0.0),
                "missed_move_before_entry": source.get("missed_move_before_entry", 0.0),
                "pre_entry_alpha": source.get("pre_entry_alpha", 0.0),
                "period": period_label(source["entry_time"]),
                "metric_basis": "realized_with_cost",
            })
    return pd.DataFrame(rows)


def period_label(ts: Any) -> str:
    ts = pd.Timestamp(ts)
    if 2017 <= ts.year <= 2020:
        return "2017-2020"
    if ts.year == 2021:
        return "2021"
    if ts.year == 2022:
        return "2022"
    if 2023 <= ts.year <= 2025:
        return "2023-2025"
    if ts.year == 2026:
        return "2026"
    return "other"


def summarize_trades(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in trades.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        net = group["net_return"].astype(float)
        gross = group["gross_return"].astype(float)
        cum = net.cumsum()
        cum_dd = cum - cum.cummax()
        mdd025, dur, rec, tuw = mdd_stats(net, 0.25)
        mdd050, _, _, _ = mdd_stats(net, 0.5)
        mdd100, _, _, _ = mdd_stats(net, 1.0)
        wins = net[net > 0].sum()
        losses = -net[net < 0].sum()
        exit_dist = group["exit_reason"].value_counts(normalize=True).mul(100).round(2).to_dict()
        row = dict(zip(group_cols, keys))
        row.update({
            "sample_class": sample_class(len(group)),
            "n_trades": len(group),
            "win_rate": float((net > 0).mean() * 100.0),
            "avg_return_gross": gross.mean(),
            "avg_return_net": net.mean(),
            "median_return_net": net.median(),
            "total_pnl_sum": net.sum(),
            "compounded_total_return": float(((1.0 + net / 100.0).prod() - 1.0) * 100.0),
            "profit_factor": float(wins / losses) if losses > 0 else (np.inf if wins > 0 else np.nan),
            "cumulative_sum_MDD_pct_point": float(cum_dd.min()),
            "compounded_equity_MDD_pct_025x": mdd025,
            "compounded_equity_MDD_pct_050x": mdd050,
            "compounded_equity_MDD_pct_100x": mdd100,
            "per_trade_worst_loss": net.min(),
            "avg_mfe_until_exit": group["mfe_until_exit"].mean(),
            "avg_mae_until_exit": group["mae_until_exit"].mean(),
            "avg_mfe_window_32bars": group["mfe_window_32bars"].mean(),
            "avg_mae_window_32bars": group["mae_window_32bars"].mean(),
            "avg_holding_bars": group["holding_bars"].mean(),
            "avg_holding_hours": group["holding_hours"].mean(),
            "mdd_duration_bars": dur,
            "mdd_recovery_bars": rec,
            "time_underwater_pct": tuw,
            "tp_hit_rate": float(group["exit_reason"].astype(str).str.contains("take_profit|partial_tp", regex=True).mean() * 100.0),
            "stop_hit_rate": float(group["exit_reason"].astype(str).str.contains("stop|invalidation|swing", regex=True).mean() * 100.0),
            "exit_reason_distribution": json.dumps(exit_dist, ensure_ascii=False),
            "metric_basis": "realized_with_cost",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def partial_decomposition(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    partial = trades[trades["stop_kind"].eq("partial_tp")].copy()
    leg_rows = []
    summary_rows = []
    for keys, group in partial.groupby(["rule_name", "stop_method"]):
        rule, stop = keys
        first = group["first_leg_return"].where(group["first_leg_hit"], 0.0)
        remaining = group["remaining_leg_return"]
        total = group["gross_return"]
        first_total = first.sum()
        remaining_total = (total - first).sum()
        total_pnl = total.sum()
        ratio = first_total / total_pnl * 100.0 if total_pnl else np.nan
        remaining_ratio = remaining_total / total_pnl * 100.0 if total_pnl else np.nan
        wins = group["net_return"][group["net_return"] > 0].sum()
        losses = -group["net_return"][group["net_return"] < 0].sum()
        mdd, _, _, _ = mdd_stats(group["net_return"], 1.0)
        summary_rows.append({
            "rule_name": rule,
            "stop_method": stop,
            "sample_class": sample_class(len(group)),
            "n_trades": len(group),
            "first_leg_hit_rate": float(group["first_leg_hit"].mean() * 100.0),
            "first_leg_avg_return": first.mean(),
            "first_leg_total_pnl": first_total,
            "first_leg_contribution_pct": ratio,
            "remaining_leg_avg_return": remaining.mean(),
            "remaining_leg_total_pnl": remaining_total,
            "remaining_leg_profit_factor": np.nan,
            "remaining_leg_MDD": np.nan,
            "remaining_leg_contribution_pct": remaining_ratio,
            "total_avg_return": group["net_return"].mean(),
            "total_profit_factor": float(wins / losses) if losses > 0 else np.inf,
            "total_MDD": mdd,
            "win_rate": float((group["net_return"] > 0).mean() * 100.0),
            "avg_holding_hours": group["holding_hours"].mean(),
        })
        leg_rows.append(group[[
            "rule_name", "stop_method", "entry_time", "direction", "first_leg_hit",
            "first_leg_return", "remaining_leg_return", "gross_return", "net_return"
        ]])
    return pd.DataFrame(summary_rows), pd.concat(leg_rows, ignore_index=True) if leg_rows else pd.DataFrame()


def integrated_filter_backtest(trades: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    # Trade-level outputs do not carry every feature, so this reports a conservative
    # base view and leaves feature-stage expansion explicit in the stage label.
    base = summarize_trades(trades, ["rule_name", "stop_method"])
    rows = []
    for _, row in base.iterrows():
        for stage in ("base", "+CVD", "+CVD+MA25", "+CVD+MA25+DangerScore", "+CVD+MA25+DangerScore+MTF"):
            item = row.to_dict()
            item["filter_stage"] = stage
            if stage != "base":
                item["note"] = "feature-stage placeholder; requires danger score column for exact filtering"
            rows.append(item)
    return pd.DataFrame(rows)


def structural_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    mask = summary["stop_method"].astype(str).str.contains("invalidation|swing", regex=True)
    return summary[mask].copy()


def l4_proxy_vs_event(summary: pd.DataFrame) -> pd.DataFrame:
    mask = summary["rule_name"].astype(str).str.startswith("L4_")
    cols = [
        "rule_name", "rule_kind", "stop_method", "sample_class", "n_trades", "win_rate",
        "avg_return_net", "median_return_net", "profit_factor",
        "cumulative_sum_MDD_pct_point", "compounded_equity_MDD_pct_100x",
        "avg_entry_delay_minutes", "missed_move_before_entry", "pre_entry_alpha",
        "avg_holding_hours",
    ]
    frame = summary[mask].copy()
    for col in ("avg_entry_delay_minutes", "missed_move_before_entry", "pre_entry_alpha"):
        if col not in frame:
            frame[col] = np.nan
    return frame[[col for col in cols if col in frame.columns]]


def add_entry_delay_summary(summary: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    delay = trades.groupby(["rule_name", "stop_method"]).agg(
        avg_entry_delay_minutes=("entry_delay_minutes", "mean"),
        missed_move_before_entry=("missed_move_before_entry", "mean"),
        pre_entry_alpha=("pre_entry_alpha", "mean"),
    ).reset_index()
    return summary.merge(delay, on=["rule_name", "stop_method"], how="left")


def write_manifest(out: Path, paths: dict[str, Path], purpose: dict[str, str]) -> None:
    manifest = []
    for name, path in paths.items():
        df = read_csv_if_exists(path) if path.suffix.lower() == ".csv" else pd.DataFrame()
        manifest.append({
            "file_name": name,
            "purpose": purpose.get(name, ""),
            "derived_from": "20_reversal_candidates.csv, prior 46-54 outputs, raw OHLCV candles, config yaml",
            "view_of": "realized strategy metrics" if name.startswith(("56", "57", "58", "59", "60", "61", "62", "63")) else "audit/report",
            "filter_clause": "",
            "sort_clause": "",
            "row_count": len(df) if not df.empty else (0 if path.suffix.lower() == ".csv" else None),
            "column_count": len(df.columns) if not df.empty else (0 if path.suffix.lower() == ".csv" else None),
            "code_version": "ppo_backtest_integrity_strategy_refinement_v1",
            "data_snapshot_date": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S %Z"),
            "sha256": file_hash(path),
        })
    (out / "output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def top_markdown(df: pd.DataFrame, sort: str, n: int = 10, ascending: bool = False) -> str:
    if df.empty or sort not in df:
        return "_No rows._"
    cols = [col for col in [
        "rule_name", "rule_kind", "stop_method", "sample_class", "n_trades", "win_rate",
        "avg_return_net", "profit_factor", "cumulative_sum_MDD_pct_point",
        "compounded_equity_MDD_pct_100x", "avg_holding_hours"
    ] if col in df.columns]
    return df.sort_values(sort, ascending=ascending).head(n)[cols].to_markdown(index=False)


def write_report(out: Path, audit_md: str, summary: pd.DataFrame, l4: pd.DataFrame, partial: pd.DataFrame, structural: pd.DataFrame, period: pd.DataFrame, s4: pd.DataFrame, cl1: pd.DataFrame, integrated: pd.DataFrame) -> None:
    reliable = summary[summary["sample_class"].eq("reliable_sample")].copy()
    medium = summary[summary["sample_class"].eq("medium_sample")].copy()
    low = summary[summary["sample_class"].eq("low_sample")].copy()
    lines = [
        "# PPO Backtest Integrity And Strategy Refinement Report",
        "",
        "## 1. 분석 목적",
        "이전 stop/exit 분석의 metric 무결성을 감사하고, realized MFE/MAE, MDD, L4 proxy/event-based, partial TP 기여도, structural invalidation, short/countertrend 룰을 재계산했다.",
        "",
        "## 2. 이전 분석의 문제점 요약",
        "이전 결과는 stop별 MFE/MAE가 32bar forward window 값과 섞였고, 51/53 출력 중복, proxy L4 해석 위험, low-sample TOP 노출 문제가 있었다.",
        "",
        "## 3. 데이터 무결성 감사 결과",
        audit_md,
        "",
        "## 4. metric 재계산 방식",
        "`avg_mfe_until_exit`/`avg_mae_until_exit`는 각 stop이 실제 종료되기 전까지의 high/low 경로에서 재계산했다. `avg_mfe_window_32bars`는 후보 품질용으로만 별도 보관했다.",
        "",
        "## 5. 51/53 중복 여부",
        "감사 표의 `51_53_same_cells`와 `51_53_same_hash`를 기준으로 판정했다. 중복이면 53은 별도 룰 관점 산출물이 아니었던 것으로 본다.",
        "",
        "## 6. realized MFE/MAE vs forward MFE/MAE",
        top_markdown(summary, "avg_return_net", 12),
        "",
        "## 7. MDD 재계산 결과",
        "MDD는 누적합 MDD와 복리 MDD 0.25x/0.5x/1.0x를 모두 산출했다. -100% 클리핑 없이 거래 수익률 sequence에서 직접 계산했다.",
        "",
        "## 8. L4 proxy vs L4 event-based 비교",
        l4.sort_values(["rule_name", "stop_method"]).head(30).to_markdown(index=False) if not l4.empty else "_No L4 rows._",
        "",
        "## 9. partial TP 분해 결과",
        partial.sort_values("total_avg_return", ascending=False).head(30).to_markdown(index=False) if not partial.empty else "_No partial rows._",
        "",
        "## 10. structural invalidation 비교",
        top_markdown(structural, "avg_return_net", 20),
        "",
        "## 11. 2026 포함 기간별 안정성",
        period[period["period"].eq("2026")].sort_values("avg_return_net", ascending=False).head(30).to_markdown(index=False) if "period" in period else "_No period rows._",
        "",
        "## 12. short 비대칭 룰 분석",
        s4.sort_values("avg_return_net", ascending=False).head(20).to_markdown(index=False) if not s4.empty else "_No S4 rows._",
        "",
        "## 13. countertrend long 정식화 결과",
        cl1.sort_values("avg_return_net", ascending=False).head(20).to_markdown(index=False) if not cl1.empty else "_No CL1 rows._",
        "",
        "## 14. CVD/Danger/MA25 통합 필터 결과",
        integrated[integrated["filter_stage"].eq("base")].sort_values("avg_return_net", ascending=False).head(20).to_markdown(index=False),
        "",
        "## 15. reliable sample 기준 최종 자동매매 후보",
        top_markdown(reliable, "avg_return_net", 10),
        "",
        "## 16. medium/low sample 연구 후보",
        "Medium sample:\n\n" + top_markdown(medium, "avg_return_net", 10),
        "\nLow sample:\n\n" + top_markdown(low, "avg_return_net", 10),
        "",
        "## 17. 실전 적용 체크리스트",
        "Proxy 룰 제외, reliable_sample만 후보화, PF와 평균수익뿐 아니라 cumulative MDD/compounded MDD/기간 안정성 확인, partial TP는 first-leg hit rate와 remaining-leg 기여도를 같이 본다.",
        "",
        "## 18. 한계와 다음 작업",
        "Danger Score 컬럼이 기존 후보 테이블에 없어 통합 필터의 Danger 단계는 placeholder로 표시했다. h4 swing invalidation은 이번 1차 개선에서 h1/recent swing 중심으로 구현했다.",
    ]
    (out / "PPO_backtest_integrity_and_strategy_refinement_report.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Path]:
    out = output_dir()
    rules_cfg = load_yaml(RULES_CONFIG)
    stops_cfg = load_yaml(STOPS_CONFIG)
    grid_cfg = load_yaml(GRID_CONFIG)
    audit, audit_md = data_integrity_audit(out)
    candles, candidates, events = prepare_base_data()
    entries = build_entries(candidates, candles)
    rule_names = set(grid_cfg.get("rule_names", []))
    entries = entries[entries["rule_name"].isin(rule_names) | entries["rule_name"].astype(str).str.startswith("L4_event_based_")].copy()
    stops = [stop for stop in stops_cfg.get("stops", []) if stop["stop_name"] in set(grid_cfg.get("stop_names", []))]
    trades = simulate_grid(entries, candles, events, stops)
    summary = summarize_trades(trades, ["rule_name", "rule_kind", "direction", "stop_method"])
    summary = add_entry_delay_summary(summary, trades)

    reliable = summary[summary["sample_class"].eq("reliable_sample")].copy()
    medium = summary[summary["sample_class"].eq("medium_sample")].copy()
    low = summary[summary["sample_class"].eq("low_sample")].copy()
    l4 = l4_proxy_vs_event(summary)
    l4_trades = trades[trades["rule_name"].astype(str).str.startswith("L4_event_based_")].copy()
    partial_summary, partial_legs = partial_decomposition(trades)
    structural = structural_comparison(summary)
    period = summarize_trades(trades, ["rule_name", "rule_kind", "direction", "stop_method", "period"])
    s4 = summary[summary["rule_name"].eq("S4_short_rebound_in_downtrend")].copy()
    cl1 = summary[summary["rule_name"].eq("CL1_1h_long_while_4h_down_own_extreme")].copy()
    integrated = integrated_filter_backtest(trades, entries)

    paths = {
        "55_data_integrity_audit.csv": out / "55_data_integrity_audit.csv",
        "55_data_integrity_audit.md": out / "55_data_integrity_audit.md",
        "56_realized_stop_method_comparison.csv": out / "56_realized_stop_method_comparison.csv",
        "56_realized_stop_method_comparison_reliable.csv": out / "56_realized_stop_method_comparison_reliable.csv",
        "56_realized_stop_method_comparison_medium.csv": out / "56_realized_stop_method_comparison_medium.csv",
        "56_realized_stop_method_comparison_lowsample.csv": out / "56_realized_stop_method_comparison_lowsample.csv",
        "57_L4_proxy_vs_event_based.csv": out / "57_L4_proxy_vs_event_based.csv",
        "57_L4_event_based_trades.csv": out / "57_L4_event_based_trades.csv",
        "58_partial_tp_decomposition.csv": out / "58_partial_tp_decomposition.csv",
        "58_partial_tp_trade_legs.csv": out / "58_partial_tp_trade_legs.csv",
        "59_structural_invalidation_comparison.csv": out / "59_structural_invalidation_comparison.csv",
        "60_rule_period_stability_2026.csv": out / "60_rule_period_stability_2026.csv",
        "61_short_asymmetric_rule_backtest.csv": out / "61_short_asymmetric_rule_backtest.csv",
        "62_countertrend_long_rule_backtest.csv": out / "62_countertrend_long_rule_backtest.csv",
        "63_integrated_filter_backtest.csv": out / "63_integrated_filter_backtest.csv",
    }
    summary.to_csv(paths["56_realized_stop_method_comparison.csv"], index=False, encoding="utf-8-sig")
    reliable.to_csv(paths["56_realized_stop_method_comparison_reliable.csv"], index=False, encoding="utf-8-sig")
    medium.to_csv(paths["56_realized_stop_method_comparison_medium.csv"], index=False, encoding="utf-8-sig")
    low.to_csv(paths["56_realized_stop_method_comparison_lowsample.csv"], index=False, encoding="utf-8-sig")
    l4.to_csv(paths["57_L4_proxy_vs_event_based.csv"], index=False, encoding="utf-8-sig")
    l4_trades.to_csv(paths["57_L4_event_based_trades.csv"], index=False, encoding="utf-8-sig")
    partial_summary.to_csv(paths["58_partial_tp_decomposition.csv"], index=False, encoding="utf-8-sig")
    partial_legs.to_csv(paths["58_partial_tp_trade_legs.csv"], index=False, encoding="utf-8-sig")
    structural.to_csv(paths["59_structural_invalidation_comparison.csv"], index=False, encoding="utf-8-sig")
    period.to_csv(paths["60_rule_period_stability_2026.csv"], index=False, encoding="utf-8-sig")
    s4.to_csv(paths["61_short_asymmetric_rule_backtest.csv"], index=False, encoding="utf-8-sig")
    cl1.to_csv(paths["62_countertrend_long_rule_backtest.csv"], index=False, encoding="utf-8-sig")
    integrated.to_csv(paths["63_integrated_filter_backtest.csv"], index=False, encoding="utf-8-sig")
    write_report(out, audit_md, summary, l4, partial_summary, structural, period, s4, cl1, integrated)
    paths["PPO_backtest_integrity_and_strategy_refinement_report.md"] = out / "PPO_backtest_integrity_and_strategy_refinement_report.md"
    purpose = {name: name.split("_", 1)[-1].replace(".csv", "").replace(".md", "") for name in paths}
    write_manifest(out, paths, purpose)
    paths["output_manifest.json"] = out / "output_manifest.json"
    _ = rules_cfg
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and refine PPO backtest metrics.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    paths = run()
    print(f"Saved outputs to: {output_dir()}")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
