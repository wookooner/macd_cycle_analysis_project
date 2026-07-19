from __future__ import annotations

import argparse
import json
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
    _bin_from_decile,
    _candidate_sign,
    _cycle_path,
    _direction_return,
    _raw_market_path,
    _read_timestamp,
    _tf_delta,
    _zone4,
)


BASE_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_reversal_candidate_backtest"
OUT_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_15m_leading_1h_reversal_analysis"
TIMEFRAMES = ("15m", "1h", "4h", "1d")
POSITION_COST_PCT = ROUND_TRIP_FEE_PCT + SLIPPAGE_PER_SIDE_PCT * 2


def output_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


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
        "ma_7",
        "ma_25",
        "ma_99",
        "cvd",
        "volume_delta",
        "volume",
    }
    available = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[col for col in available if col in wanted])
    ts_col = next(col for col in ("timestamp", "open_time", "date") if col in df.columns)
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _read_timestamp(df["timestamp"])
    for col in [c for c in df.columns if c != "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "ppo", "ppo_hist"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df), dtype=np.int64)
    df["ppo_hist_diff"] = df["ppo_hist"].diff()
    df["ppo_hist_delta_1"] = df["ppo_hist_diff"]
    df["ppo_hist_delta_2"] = df["ppo_hist"].diff(2)
    df["ppo_hist_acceleration"] = df["ppo_hist_diff"].diff()
    df["zone4"] = _zone4(df["ppo"], df["ppo_hist"])
    df = _add_full_deciles(df, "ppo", "ppo")
    df = _add_full_deciles(df, "ppo_hist", "hist")
    df["distance_from_ma25"] = (df["close"] / df.get("ma_25") - 1.0) * 100.0 if "ma_25" in df else np.nan
    df["distance_from_ma99"] = (df["close"] / df.get("ma_99") - 1.0) * 100.0 if "ma_99" in df else np.nan
    if "cvd" in df:
        df["cvd_delta"] = df["cvd"].diff()
    else:
        df["cvd_delta"] = np.nan
    df["recent_low_20"] = df["low"].rolling(20, min_periods=1).min()
    df["recent_low_distance"] = (df["close"] / df["recent_low_20"] - 1.0) * 100.0
    df["hist_improving"] = df["ppo_hist_diff"] > 0
    df["long_bias"] = df["ppo_hist"] < 0
    df["short_bias"] = df["ppo_hist"] > 0
    return df


def load_cycles(timeframe: str) -> pd.DataFrame:
    df = pd.read_parquet(_cycle_path(timeframe)).copy()
    df["start_date"] = _read_timestamp(df["start_date"])
    df["end_date"] = _read_timestamp(df["end_date"])
    df = df.dropna(subset=["start_date", "end_date"]).sort_values("start_date").reset_index(drop=True)
    df["cycle_sign"] = np.where(df["cycle_type"].astype(str).str.lower().eq("up"), 1, -1)
    df["end_exclusive"] = df["end_date"] + _tf_delta(timeframe)
    return df


def add_cycle_state(candles: pd.DataFrame, cycles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    keep = ["cycle_id", "cycle_type", "cycle_sign", "start_date", "end_exclusive", "duration_candles", "cycle_features"]
    merged = pd.merge_asof(
        candles.sort_values("timestamp"),
        cycles[keep].sort_values("start_date"),
        left_on="timestamp",
        right_on="start_date",
        direction="backward",
    )
    in_cycle = merged["timestamp"].lt(merged["end_exclusive"])
    merged.loc[~in_cycle, ["cycle_id", "cycle_type", "cycle_sign", "start_date", "end_exclusive", "duration_candles"]] = np.nan
    elapsed = (merged["timestamp"] - merged["start_date"]) / _tf_delta(timeframe)
    # In live trading the final cycle duration/progress/noise_count are unknown.
    # Use only the age observed up to the already-closed candle.
    merged["cycle_age_bars"] = elapsed + 1
    merged["cycle_duration"] = merged["cycle_age_bars"]
    merged["cycle_progress"] = np.nan
    merged["cycle_noise_ratio"] = np.nan
    merged = merged.rename(columns={"start_date": "cycle_start", "end_exclusive": "cycle_end"})
    return merged.drop(columns=["cycle_features"], errors="ignore")


def _noise_ratio_from_features(value: Any) -> float:
    if not isinstance(value, dict):
        return np.nan
    try:
        shape = value.get("shape", {})
        duration = float(shape.get("duration_candles") or 0)
        noise = float(shape.get("noise_count") or 0)
        return noise / duration if duration else np.nan
    except Exception:
        return np.nan


def load_candidates() -> pd.DataFrame:
    path = BASE_DIR / "20_reversal_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing prior candidate output: {path}")
    wanted = {
        "timestamp",
        "candidate_tf",
        "candidate_direction",
        "close_at_entry",
        "bar_index",
        "is_noise",
        "is_true_reversal",
        "upper_1h_cycle_direction",
        "upper_1h_cycle_progress",
        "upper_1h_ppo",
        "upper_1h_hist",
        "upper_1h_hist_diff",
        "upper_1h_ppo_bin",
        "upper_1h_hist_bin",
        "upper_4h_cycle_direction",
        "upper_4h_ppo",
        "upper_4h_hist",
        "upper_4h_hist_diff",
        "upper_4h_ppo_bin",
        "upper_4h_hist_bin",
        "upper_1d_cycle_direction",
        "upper_1d_ppo",
        "upper_1d_hist",
        "upper_1d_ppo_bin",
        "upper_1d_hist_bin",
    }
    available = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[col for col in wanted if col in available], engine="python")
    df["timestamp"] = _read_timestamp(df["timestamp"])
    return df[
        df["candidate_tf"].eq("15m")
        & df["candidate_direction"].eq("long")
        & df.get("upper_1h_cycle_direction").astype(str).str.lower().eq("down")
    ].sort_values("timestamp").reset_index(drop=True)


def closed_join(base: pd.DataFrame, upper: pd.DataFrame, upper_tf: str, prefix: str) -> pd.DataFrame:
    frame = upper.copy()
    frame["available_at"] = frame["timestamp"] + _tf_delta(upper_tf)
    keep = [
        "available_at",
        "timestamp",
        "cycle_id",
        "cycle_type",
        "cycle_start",
        "cycle_end",
        "cycle_duration",
        "cycle_progress",
        "cycle_noise_ratio",
        "open",
        "high",
        "low",
        "close",
        "ppo",
        "ppo_hist",
        "ppo_hist_diff",
        "ppo_hist_delta_1",
        "ppo_hist_delta_2",
        "ppo_hist_acceleration",
        "rsi",
        "ppo_bin",
        "hist_bin",
        "hist_improving",
        "long_bias",
        "short_bias",
    ]
    existing = [col for col in keep if col in frame.columns]
    renamed = frame[existing].rename(columns={col: f"{prefix}_{col}" for col in existing if col != "available_at"})
    return pd.merge_asof(
        base.sort_values("timestamp"),
        renamed.sort_values("available_at"),
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
    ).drop(columns=["available_at"], errors="ignore")


def enrich_candidates(candidates: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = candidates.copy()
    fifteen = candles["15m"].add_prefix("m15_").rename(columns={"m15_timestamp": "timestamp"})
    keep_15m = [
        "timestamp",
        "m15_open",
        "m15_high",
        "m15_low",
        "m15_close",
        "m15_ppo",
        "m15_ppo_hist",
        "m15_ppo_hist_diff",
        "m15_ppo_bin",
        "m15_hist_bin",
        "m15_zone4",
        "m15_rsi",
        "m15_distance_from_ma25",
        "m15_distance_from_ma99",
        "m15_cvd_delta",
        "m15_volume_delta",
        "m15_recent_low_distance",
        "m15_bar_index",
    ]
    base = base.merge(fifteen[[col for col in keep_15m if col in fifteen.columns]], on="timestamp", how="left")
    base = closed_join(base, candles["1h"], "1h", "h1")
    base = closed_join(base, candles["4h"], "4h", "h4")
    base = closed_join(base, candles["1d"], "1d", "d1")
    base["h1_current_drawdown_from_cycle_start"] = current_cycle_drawdown(base, candles["1h"])
    base["h1_is_hist_improving"] = base["h1_ppo_hist_diff"] > 0
    base["h1_is_down_accelerating"] = (base["h1_ppo_hist"] < 0) & (base["h1_ppo_hist_diff"] < 0)
    base["h4_extreme_long"] = base["h4_ppo_bin"].isin(["bottom10", "bottom20"]) & (base["h4_ppo_hist"] < 0)
    base["h4_extreme_short"] = base["h4_ppo_bin"].isin(["top10", "top20"]) & (base["h4_ppo_hist"] > 0)
    base["d1_long_regime"] = base["d1_ppo_bin"].isin(["bottom10", "bottom20"]) | (base["d1_ppo_hist"] < 0)
    base["d1_short_regime"] = base["d1_ppo_bin"].isin(["top10", "top20"]) | (base["d1_ppo_hist"] > 0)
    return base


def current_cycle_drawdown(rows: pd.DataFrame, h1: pd.DataFrame) -> pd.Series:
    lows = []
    h1_indexed = h1.set_index("timestamp")
    for row in rows.itertuples(index=False):
        start = getattr(row, "h1_cycle_start", pd.NaT)
        ts = getattr(row, "h1_timestamp", pd.NaT)
        close = getattr(row, "h1_close", np.nan)
        if pd.isna(start) or pd.isna(ts) or pd.isna(close):
            lows.append(np.nan)
            continue
        segment = h1[(h1["timestamp"] >= start) & (h1["timestamp"] <= ts)]
        high = segment["high"].max()
        lows.append((close / high - 1.0) * 100.0 if high and not pd.isna(high) else np.nan)
    return pd.Series(lows, index=rows.index)


def next_1h_up_events(candidates: pd.DataFrame) -> pd.DataFrame:
    one_h = candidates[
        candidates["candidate_tf"].eq("1h")
        & candidates["candidate_direction"].eq("long")
        & pd.to_numeric(candidates["is_true_reversal"], errors="coerce").eq(1)
    ].sort_values("timestamp")
    return one_h


def next_1h_down_events(candidates: pd.DataFrame) -> pd.DataFrame:
    one_h = candidates[
        candidates["candidate_tf"].eq("1h")
        & candidates["candidate_direction"].eq("short")
        & pd.to_numeric(candidates["is_true_reversal"], errors="coerce").eq(1)
    ].sort_values("timestamp")
    return one_h


def add_targets(rows: pd.DataFrame, all_candidates: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = rows.copy()
    up_events = next_1h_up_events(all_candidates)
    event_times = up_events["timestamp"].to_numpy(dtype="datetime64[ns]")
    event_prices = up_events["close_at_entry"].to_numpy(dtype="float64")
    down_events = next_1h_down_events(all_candidates)
    down_times = down_events["timestamp"].to_numpy(dtype="datetime64[ns]")
    down_prices = down_events["close_at_entry"].to_numpy(dtype="float64")
    h15 = candles["15m"]
    h1 = candles["1h"]
    target_rows = []
    for row in rows.itertuples(index=False):
        ts = row.timestamp
        entry_price = float(row.close_at_entry)
        idx = int(np.searchsorted(event_times, np.datetime64(ts), side="right"))
        confirm_time = pd.NaT
        confirm_price = np.nan
        if idx < len(event_times):
            confirm_time = pd.Timestamp(event_times[idx])
            confirm_price = float(event_prices[idx])
        common_exit_time = pd.NaT
        common_exit_price = np.nan
        if not pd.isna(confirm_time):
            down_idx = int(np.searchsorted(down_times, np.datetime64(confirm_time), side="right"))
            if down_idx < len(down_times):
                common_exit_time = pd.Timestamp(down_times[down_idx])
                common_exit_price = float(down_prices[down_idx])
        future_15 = h15[(h15["timestamp"] > ts) & ((h15["timestamp"] <= confirm_time) if not pd.isna(confirm_time) else (h15["timestamp"] <= ts + pd.Timedelta(hours=3)))]
        if future_15.empty:
            mfe = mae = close_ret_to_confirm = np.nan
        else:
            mfe = (future_15["high"].max() / entry_price - 1.0) * 100.0
            mae = (future_15["low"].min() / entry_price - 1.0) * 100.0
            close_ret_to_confirm = (future_15.iloc[-1]["close"] / entry_price - 1.0) * 100.0
        h1_start_low = getattr(row, "h1_low", np.nan)
        h1_future = h1[(h1["timestamp"] > getattr(row, "h1_timestamp", ts)) & (h1["timestamp"] < confirm_time)] if not pd.isna(confirm_time) else pd.DataFrame()
        made_new_low = False
        if not h1_future.empty and not pd.isna(h1_start_low):
            made_new_low = bool(h1_future["low"].min() < h1_start_low)
        delay_bars = (confirm_time - ts) / _tf_delta("15m") if not pd.isna(confirm_time) else np.nan
        early_common_return = (
            (common_exit_price / entry_price - 1.0) * 100.0
            if not pd.isna(common_exit_price)
            else np.nan
        )
        confirmed_common_return = (
            (common_exit_price / confirm_price - 1.0) * 100.0
            if not pd.isna(common_exit_price) and not pd.isna(confirm_price) and confirm_price
            else np.nan
        )
        target_rows.append(
            {
                "target_1h_up_confirm_time": confirm_time,
                "target_1h_up_confirm_price": confirm_price,
                "target_common_exit_1h_down_time": common_exit_time,
                "target_common_exit_1h_down_price": common_exit_price,
                "target_1h_turns_up_within_4bars": bool(not pd.isna(delay_bars) and delay_bars <= 4),
                "target_1h_turns_up_within_8bars": bool(not pd.isna(delay_bars) and delay_bars <= 8),
                "target_1h_turns_up_within_12bars": bool(not pd.isna(delay_bars) and delay_bars <= 12),
                "target_1h_turn_delay_15m_bars": float(delay_bars) if not pd.isna(delay_bars) else np.nan,
                "target_1h_turns_up_before_new_low": bool(not pd.isna(confirm_time) and not made_new_low),
                "target_max_favorable_excursion": mfe,
                "target_max_adverse_excursion": mae,
                "target_profitable_before_1h_confirm": bool((mfe > 0) if not pd.isna(mfe) else False),
                "target_pre_confirm_return": (confirm_price / entry_price - 1.0) * 100.0 if not pd.isna(confirm_price) else np.nan,
                "target_pre_entry_alpha": early_common_return - confirmed_common_return,
                "target_early_common_return": early_common_return,
                "target_confirmed_common_return": confirmed_common_return,
                "target_close_return_to_confirm": close_ret_to_confirm,
            }
        )
    return pd.concat([rows.reset_index(drop=True), pd.DataFrame(target_rows)], axis=1)


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        n = len(group)
        row.update(
            {
                "n": n,
                "1h_turn_up_within_4bars_pct": float(group["target_1h_turns_up_within_4bars"].mean() * 100),
                "1h_turn_up_within_8bars_pct": float(group["target_1h_turns_up_within_8bars"].mean() * 100),
                "1h_turn_up_within_12bars_pct": float(group["target_1h_turns_up_within_12bars"].mean() * 100),
                "before_new_low_pct": float(group["target_1h_turns_up_before_new_low"].mean() * 100),
                "avg_pre_entry_alpha": float(group["target_pre_entry_alpha"].mean()),
                "avg_max_favorable_excursion": float(group["target_max_favorable_excursion"].mean()),
                "avg_max_adverse_excursion": float(group["target_max_adverse_excursion"].mean()),
                "low_sample": n < LOW_SAMPLE_N,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["1h_turn_up_within_8bars_pct", "n"], ascending=[False, False])


def max_drawdown(returns: pd.Series) -> float:
    net = pd.to_numeric(returns, errors="coerce").fillna(0) / 100.0
    equity = (1 + net).cumprod()
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1).min() * 100.0)


def compare_entry_timing(df: pd.DataFrame, candles: dict[str, pd.DataFrame], all_candidates: pd.DataFrame) -> pd.DataFrame:
    h15 = candles["15m"]
    down_events = next_1h_down_events(all_candidates)
    down_times = down_events["timestamp"].to_numpy(dtype="datetime64[ns]")
    down_prices = down_events["close_at_entry"].to_numpy(dtype="float64")
    rows = []
    for mode, delay in [("15m_immediate", 0), ("wait_1_15m_bar", 1), ("wait_2_15m_bars", 2), ("1h_up_confirmed", None)]:
        rets = []
        delays = []
        alphas = []
        for item in df.itertuples(index=False):
            if pd.isna(item.target_1h_up_confirm_time):
                continue
            if pd.isna(getattr(item, "m15_bar_index", np.nan)):
                continue
            down_idx = int(np.searchsorted(down_times, np.datetime64(item.target_1h_up_confirm_time), side="right"))
            if down_idx >= len(down_times):
                continue
            exit_time = pd.Timestamp(down_times[down_idx])
            exit_price = float(down_prices[down_idx])
            if delay is None:
                entry_price = float(item.target_1h_up_confirm_price)
                entry_time = item.target_1h_up_confirm_time
                entry_delay = item.target_1h_turn_delay_15m_bars
            else:
                entry_idx = int(item.m15_bar_index) + delay
                if entry_idx >= len(h15):
                    continue
                entry = h15.iloc[entry_idx]
                required_sign = 1
                if delay and not (h15.iloc[int(item.m15_bar_index) : entry_idx + 1]["ppo_hist_diff"].fillna(0) > 0).all():
                    continue
                entry_price = float(entry["close"])
                entry_time = entry["timestamp"]
                entry_delay = delay
            if entry_time >= item.target_1h_up_confirm_time and delay is not None:
                continue
            if entry_time >= exit_time:
                continue
            ret = (exit_price / entry_price - 1.0) * 100.0 - POSITION_COST_PCT
            rets.append(ret)
            delays.append(entry_delay)
            confirmed_ret = (exit_price / float(item.target_1h_up_confirm_price) - 1.0) * 100.0 - POSITION_COST_PCT
            alphas.append(ret - confirmed_ret)
        s = pd.Series(rets, dtype="float64")
        rows.append(
            {
                "entry_mode": mode,
                "n": len(s),
                "win_rate": float((s > 0).mean() * 100) if len(s) else np.nan,
                "avg_return": float(s.mean()) if len(s) else np.nan,
                "median_return": float(s.median()) if len(s) else np.nan,
                "mdd": max_drawdown(s),
                "avg_entry_delay": float(np.nanmean(delays)) if delays else np.nan,
                "avg_pre_entry_alpha": float(np.nanmean(alphas)) if alphas else np.nan,
                "low_sample": len(s) < LOW_SAMPLE_N,
            }
        )
    return pd.DataFrame(rows)


def train_model(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:
        result = pd.DataFrame([{"model": "unavailable", "error": str(exc)}])
        result.to_csv(out_dir / "36_15m_lead_prediction_model.csv", index=False, encoding="utf-8-sig")
        return result
    feature_cols = [
        "m15_ppo",
        "m15_ppo_hist",
        "m15_ppo_hist_diff",
        "m15_ppo_bin",
        "m15_hist_bin",
        "m15_zone4",
        "m15_rsi",
        "m15_distance_from_ma25",
        "m15_distance_from_ma99",
        "m15_cvd_delta",
        "m15_volume_delta",
        "m15_recent_low_distance",
        "h1_ppo",
        "h1_ppo_hist",
        "h1_ppo_hist_diff",
        "h1_ppo_hist_delta_2",
        "h1_ppo_hist_acceleration",
        "h1_rsi",
        "h1_cycle_duration",
        "h1_cycle_progress",
        "h1_cycle_noise_ratio",
        "h1_current_drawdown_from_cycle_start",
        "h1_is_hist_improving",
        "h1_is_down_accelerating",
        "h4_ppo",
        "h4_ppo_hist",
        "h4_ppo_hist_diff",
        "h4_hist_improving",
        "h4_long_bias",
        "h4_ppo_bin",
        "h4_hist_bin",
        "h4_extreme_long",
        "h4_extreme_short",
        "d1_ppo",
        "d1_ppo_hist",
        "d1_long_regime",
        "d1_short_regime",
    ]
    feature_cols = [col for col in feature_cols if col in df.columns and df[col].notna().any()]
    data = df.dropna(subset=["target_1h_turns_up_within_8bars"]).sort_values("timestamp").copy()
    y = data["target_1h_turns_up_within_8bars"].astype(int)
    split = int(len(data) * 0.7)
    if y.nunique() < 2 or split <= 0:
        result = pd.DataFrame([{"model": "skipped", "reason": "not enough class diversity"}])
        result.to_csv(out_dir / "36_15m_lead_prediction_model.csv", index=False, encoding="utf-8-sig")
        return result
    train, test = data.iloc[:split], data.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    numeric = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]
    categorical = [col for col in feature_cols if col not in numeric]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=25, random_state=42, class_weight="balanced_subsample", n_jobs=-1),
    }
    rows = []
    for model_name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(train[feature_cols], y_train)
        probs = pipe.predict_proba(test[feature_cols])[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probs) if y_test.nunique() == 2 else np.nan
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        names = pipe.named_steps["pre"].get_feature_names_out()
        if model_name == "random_forest":
            importances = pipe.named_steps["model"].feature_importances_
        else:
            importances = np.abs(pipe.named_steps["model"].coef_[0])
        for rank, idx in enumerate(np.argsort(importances)[::-1][:40], start=1):
            rows.append(
                {
                    "model": model_name,
                    "rank": rank,
                    "feature": names[idx],
                    "importance": float(importances[idx]),
                    "auc": float(auc),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "train_n": len(train),
                    "test_n": len(test),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "36_15m_lead_prediction_model.csv", index=False, encoding="utf-8-sig")
    return result


def table(df: pd.DataFrame, columns: list[str], n: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    return df[columns].head(n).to_markdown(index=False)


def write_report(out_dir: Path, candidates: pd.DataFrame, by15: pd.DataFrame, by1h: pd.DataFrame, by4h: pd.DataFrame, timing: pd.DataFrame, model: pd.DataFrame) -> None:
    overall = {
        "n": len(candidates),
        "within_4": candidates["target_1h_turns_up_within_4bars"].mean() * 100,
        "within_8": candidates["target_1h_turns_up_within_8bars"].mean() * 100,
        "within_12": candidates["target_1h_turns_up_within_12bars"].mean() * 100,
        "before_low": candidates["target_1h_turns_up_before_new_low"].mean() * 100,
        "alpha": candidates["target_pre_entry_alpha"].mean(),
    }
    improving = candidates[candidates["h1_is_hist_improving"].eq(True)]
    worsening = candidates[candidates["h1_is_hist_improving"].eq(False)]
    improve_line = (
        f"1h hist improving success within 8 bars: {improving['target_1h_turns_up_within_8bars'].mean() * 100:.2f}% "
        f"vs not improving: {worsening['target_1h_turns_up_within_8bars'].mean() * 100:.2f}%."
        if len(improving) and len(worsening)
        else "Not enough improving/worsening split rows."
    )
    report = f"""# PPO 15m Leading 1h Reversal Report

## 1. 분석 목적

15m long candidate가 1h cycle이 아직 DOWN인 상황에서 1h UP 전환을 선행 예측할 수 있는지 검증했다.

## 2. 왜 기존 noise 분석과 다른가

기존 분석은 후보 자체가 noise인지 true reversal인지 봤다. 이번 분석은 15m 후보가 상위 1h 전환보다 먼저 발생하는 early signal인지 본다.

## 3. 15m 선행 롱 후보 정의

기존 `20_reversal_candidates.csv`에서 `candidate_tf=15m`, `candidate_direction=long`, 닫힌 1h cycle direction이 `DOWN`인 후보만 사용했다.

## 4. 1h 전환 성공 label 정의

후보 이후 다음 1h long true reversal candidate가 4/8/12개 15m bar 안에 발생하는지, 그리고 확인 전 새 저점 여부와 pre-entry alpha를 계산했다.

## 5. 전체 성공률

- 대상 후보 수: {overall['n']:,}
- 1시간 이내 1h UP 전환: {overall['within_4']:.2f}%
- 2시간 이내 1h UP 전환: {overall['within_8']:.2f}%
- 3시간 이내 1h UP 전환: {overall['within_12']:.2f}%
- 새 저점 전 1h UP 전환: {overall['before_low']:.2f}%
- 평균 pre-entry alpha: {overall['alpha']:.4f}%

## 6. 15m PPO/HIST 구간별 성공률

{table(by15[~by15['low_sample']], ['m15_ppo_bin', 'm15_hist_bin', 'n', '1h_turn_up_within_8bars_pct', 'avg_pre_entry_alpha', 'avg_max_favorable_excursion', 'avg_max_adverse_excursion'])}

## 7. 1h 전환 직전 상태 분석

{improve_line}

{table(by1h[~by1h['low_sample']], ['h1_ppo_bin', 'h1_hist_bin', 'h1_is_hist_improving', 'n', '1h_turn_up_within_8bars_pct', 'avg_pre_entry_alpha'])}

## 8. 4h/1d 조건별 성공률

{table(by4h[~by4h['low_sample']], ['h4_cycle_type', 'h4_ppo_bin', 'h4_hist_bin', 'h4_hist_improving', 'n', '1h_turn_up_within_8bars_pct', 'avg_pre_entry_alpha'])}

## 9. 15m 선행 진입 vs 1h 확인 진입 비교

{timing.to_markdown(index=False)}

## 10. 기다려보기 전략 비교

Wait 1/2 bar는 같은 방향 PPO hist diff가 이어질 때만 진입한다. 성공률과 평균 수익은 `35_15m_early_entry_vs_1h_confirmed_entry.csv`에서 확인한다.

## 11. 실전 자동매매 룰 후보

15m PPO/HIST bottom 계열, 1h hist improving, 4h long-bias/extreme_long 조합이 높은 성공률과 양수 alpha를 동시에 보이는지 우선 확인해야 한다.

## 12. 현재 상황에 대한 적용 방법

실시간 적용 시 15m long candidate가 뜨더라도 닫힌 1h가 DOWN인지 먼저 확인하고, 닫힌 1h hist가 개선 중인지와 4h 상태를 함께 체크한다.

## 13. 한계와 다음 분석

feature join은 닫힌 상위 TF candle만 사용했다. target은 미래 1h UP event를 사용한다. 다음 단계는 이 조건을 now-cycle 상태와 직접 연결하는 live rule checker다.
"""
    (out_dir / "PPO_15m_leading_1h_reversal_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze whether 15m long candidates lead 1h UP reversals.")
    _ = parser.parse_args()
    out_dir = output_dir()
    all_candidates = load_candidates_all()
    lead_candidates = load_candidates()
    cycles = {tf: load_cycles(tf) for tf in TIMEFRAMES}
    candles = {tf: add_cycle_state(load_raw_candles(tf), cycles[tf], tf) for tf in TIMEFRAMES}
    enriched = enrich_candidates(lead_candidates, candles)
    enriched = add_targets(enriched, all_candidates, candles)

    enriched.to_csv(out_dir / "31_15m_leads_1h_reversal_candidates.csv", index=False, encoding="utf-8-sig")
    by15 = summarize_group(enriched, ["m15_ppo_bin", "m15_hist_bin"])
    by15.to_csv(out_dir / "32_15m_lead_success_by_15m_ppo_hist.csv", index=False, encoding="utf-8-sig")
    by1h = summarize_group(enriched, ["h1_ppo_bin", "h1_hist_bin", "h1_is_hist_improving"])
    by1h.to_csv(out_dir / "33_15m_lead_success_by_1h_state.csv", index=False, encoding="utf-8-sig")
    by4h = summarize_group(enriched, ["h4_cycle_type", "h4_ppo_bin", "h4_hist_bin", "h4_hist_improving"])
    by4h.to_csv(out_dir / "34_15m_lead_success_by_4h_state.csv", index=False, encoding="utf-8-sig")
    timing = compare_entry_timing(enriched, candles, all_candidates)
    timing.to_csv(out_dir / "35_15m_early_entry_vs_1h_confirmed_entry.csv", index=False, encoding="utf-8-sig")
    model = train_model(enriched, out_dir)
    write_report(out_dir, enriched, by15, by1h, by4h, timing, model)
    print(f"Wrote 15m leading 1h reversal analysis to {out_dir}")
    print(f"Candidates: {len(enriched):,}; within 8 bars: {enriched['target_1h_turns_up_within_8bars'].mean() * 100:.2f}%")
    return 0


def load_candidates_all() -> pd.DataFrame:
    path = BASE_DIR / "20_reversal_candidates.csv"
    wanted = {"timestamp", "candidate_tf", "candidate_direction", "close_at_entry", "is_true_reversal"}
    available = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[col for col in wanted if col in available], engine="python")
    df["timestamp"] = _read_timestamp(df["timestamp"])
    return df.sort_values(["candidate_tf", "timestamp"]).reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
