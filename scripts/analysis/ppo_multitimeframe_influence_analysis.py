from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import PROJECT_PATHS


TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d", "1w")
TF_SECONDS = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}
PROGRESS_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.000001)
PROGRESS_LABELS = ("p00_20", "p20_40", "p40_60", "p60_80", "p80_100")
DEFAULT_MIN_CASES = 80


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_multitimeframe_influence_analysis"


def cycle_dir() -> Path:
    return PROJECT_PATHS.asset_cycle_dir("btc")


def cycle_pairs(timeframes: tuple[str, ...]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for lower_idx, lower_tf in enumerate(timeframes):
        for upper_tf in timeframes[lower_idx + 1 :]:
            pairs.append((lower_tf, upper_tf))
    return pairs


def to_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def to_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    try:
        return pd.Timestamp(value)
    except Exception:
        return pd.NaT


def cycle_candles(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    rows: list[dict[str, Any]] = []
    for candle in value:
        if isinstance(candle, dict):
            rows.append(candle)
        else:
            try:
                rows.append(dict(candle))
            except Exception:
                continue
    return rows


def cycle_sign(value: Any) -> float:
    label = str(value or "").strip().lower()
    if label == "up":
        return 1.0
    if label == "down":
        return -1.0
    return np.nan


def signed_move(start_close: float, end_close: float, sign: float) -> float:
    if pd.isna(start_close) or pd.isna(end_close) or start_close == 0 or pd.isna(sign):
        return np.nan
    return (end_close / start_close - 1.0) * 100.0 * sign


def progress_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=1.0),
        bins=PROGRESS_BINS,
        labels=PROGRESS_LABELS,
        include_lowest=True,
    )


def quantile_bucket(series: pd.Series, prefix: str, bins: int = 5) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ranked = clean.rank(method="first")
    try:
        return pd.qcut(ranked, q=bins, labels=[f"{prefix}_q{i + 1}" for i in range(bins)])
    except ValueError:
        return pd.Series(pd.NA, index=series.index, dtype="object")


def regime(ppo: pd.Series, ppo_hist: pd.Series) -> pd.Series:
    ppo_sign = np.where(ppo >= 0, "ppo_pos", "ppo_neg")
    hist_sign = np.where(ppo_hist >= 0, "hist_pos", "hist_neg")
    return pd.Series(ppo_sign + "__" + hist_sign, index=ppo.index)


def load_cycles(timeframe: str) -> pd.DataFrame:
    path = cycle_dir() / f"cycles_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing cycle file: {path}")

    df = pd.read_parquet(path).copy()
    if df.empty:
        return df

    df["timeframe"] = timeframe
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df = df.dropna(subset=["start_date", "end_date"]).sort_values("start_date").reset_index(drop=True)

    if "cycle_key" not in df.columns:
        df["cycle_key"] = pd.Series(np.arange(1, len(df) + 1), dtype="Int32")
    else:
        df["cycle_key"] = pd.to_numeric(df["cycle_key"], errors="coerce").astype("Int32")
        missing = df["cycle_key"].isna()
        if missing.any():
            synthetic = np.arange(1, len(df) + 1, dtype=np.int64) + 1_000_000
            df.loc[missing, "cycle_key"] = synthetic[missing.to_numpy()]
            df["cycle_key"] = df["cycle_key"].astype("Int32")

    df["cycle_uid"] = df["timeframe"].astype(str) + ":" + df["cycle_id"].astype(str)
    df["cycle_sign"] = df["cycle_type"].map(cycle_sign).astype("float32")
    df["end_exclusive"] = df["end_date"] + pd.to_timedelta(TF_SECONDS[timeframe], unit="s")
    df["duration_candles"] = pd.to_numeric(df.get("duration_candles"), errors="coerce")

    start_closes: list[float] = []
    end_closes: list[float] = []
    cycle_moves: list[float] = []
    for _, row in df.iterrows():
        candles = cycle_candles(row.get("candle_data"))
        first_close = to_float(candles[0].get("close")) if candles else np.nan
        last_close = to_float(candles[-1].get("close")) if candles else np.nan
        start_closes.append(first_close)
        end_closes.append(last_close)
        cycle_moves.append(signed_move(first_close, last_close, row["cycle_sign"]))
    df["cycle_start_close"] = start_closes
    df["cycle_end_close"] = end_closes
    df["cycle_signed_move_pct"] = cycle_moves
    return df


def build_candle_frame(cycles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rows: list[tuple[Any, ...]] = []
    tf_seconds = TF_SECONDS[timeframe]

    for cycle in cycles.itertuples(index=False):
        candles = cycle_candles(getattr(cycle, "candle_data", None))
        if not candles:
            continue

        timestamps = pd.to_datetime(
            [candle.get("timestamp", candle.get("date")) for candle in candles],
            errors="coerce",
        )
        closes = [to_float(candle.get("close")) for candle in candles]
        ppos = [to_float(candle.get("ppo")) for candle in candles]
        ppo_hists = [to_float(candle.get("ppo_hist")) for candle in candles]

        valid_indices = [idx for idx, ts in enumerate(timestamps) if not pd.isna(ts)]
        if not valid_indices:
            continue

        cycle_len = len(valid_indices)
        valid_timestamps = [timestamps[idx] for idx in valid_indices]
        valid_closes = [closes[idx] for idx in valid_indices]
        valid_ppos = [ppos[idx] for idx in valid_indices]
        valid_ppo_hists = [ppo_hists[idx] for idx in valid_indices]

        ppo_deltas: list[float] = []
        ppo_hist_deltas: list[float] = []
        price_deltas: list[float] = []
        states: list[str] = []
        sign = float(getattr(cycle, "cycle_sign"))
        prev_close = np.nan
        prev_ppo = np.nan
        prev_ppo_hist = np.nan
        for idx in range(cycle_len):
            close = valid_closes[idx]
            ppo = valid_ppos[idx]
            ppo_hist = valid_ppo_hists[idx]

            ppo_delta = ppo - prev_ppo if idx > 0 and not pd.isna(ppo) and not pd.isna(prev_ppo) else np.nan
            ppo_hist_delta = (
                ppo_hist - prev_ppo_hist if idx > 0 and not pd.isna(ppo_hist) and not pd.isna(prev_ppo_hist) else np.nan
            )
            price_delta = (
                (close / prev_close - 1.0) * 100.0
                if idx > 0 and not pd.isna(close) and not pd.isna(prev_close) and prev_close != 0
                else np.nan
            )
            delta_sign = np.sign(ppo_hist_delta) if not pd.isna(ppo_hist_delta) else 0.0
            if idx == 0:
                state = "start"
            elif delta_sign == sign:
                state = "trend"
            elif delta_sign == -sign:
                state = "noise"
            else:
                state = "flat"

            ppo_deltas.append(ppo_delta)
            ppo_hist_deltas.append(ppo_hist_delta)
            price_deltas.append(price_delta)
            states.append(state)
            prev_close = close
            prev_ppo = ppo
            prev_ppo_hist = ppo_hist

        prev_noise_streak: list[int] = []
        streak = 0
        for value in states:
            prev_noise_streak.append(streak)
            streak = streak + 1 if value == "noise" else 0
        next_states = states[1:] + [None]
        cycle_denom = max(cycle_len - 1, 1)
        cycle_start = getattr(cycle, "start_date")
        cycle_end = getattr(cycle, "end_date")

        for idx in range(cycle_len):
            rows.append(
                (
                    timeframe,
                    getattr(cycle, "cycle_uid"),
                    getattr(cycle, "cycle_id"),
                    getattr(cycle, "cycle_key"),
                    getattr(cycle, "cycle_type"),
                    sign,
                    cycle_start,
                    cycle_end,
                    valid_timestamps[idx],
                    idx + 1,
                    valid_closes[idx],
                    valid_ppos[idx],
                    valid_ppo_hists[idx],
                    ppo_deltas[idx],
                    ppo_hist_deltas[idx],
                    price_deltas[idx],
                    idx / cycle_denom,
                    states[idx],
                    prev_noise_streak[idx],
                    next_states[idx],
                    states[idx] == "trend" and prev_noise_streak[idx] > 0,
                    states[idx] == "noise" and next_states[idx] == "trend",
                    valid_timestamps[idx] + pd.to_timedelta(tf_seconds, unit="s"),
                )
            )

    frame = pd.DataFrame(
        rows,
        columns=[
            "timeframe",
            "cycle_uid",
            "cycle_id",
            "cycle_key",
            "cycle_type",
            "cycle_sign",
            "cycle_start",
            "cycle_end",
            "timestamp",
            "candle_index",
            "close",
            "ppo",
            "ppo_hist",
            "ppo_delta",
            "ppo_hist_delta",
            "price_delta_pct",
            "cycle_progress",
            "parent_state",
            "prev_noise_streak",
            "next_parent_state",
            "is_rebound_candle",
            "noise_then_rebound_next",
            "end_exclusive",
        ],
    )
    if frame.empty:
        return frame
    return frame.sort_values("timestamp").reset_index(drop=True)


def build_cycle_interval_frame(cycles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    frame = cycles[
        [
            "timeframe",
            "cycle_uid",
            "cycle_id",
            "cycle_key",
            "cycle_type",
            "cycle_sign",
            "start_date",
            "end_date",
            "end_exclusive",
            "duration_candles",
            "cycle_signed_move_pct",
        ]
    ].copy()
    frame = frame.rename(columns={"start_date": "start_ts", "end_date": "end_ts"})
    frame["pair_timeframe"] = timeframe
    return frame


def assign_parent_interval(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    *,
    child_ts_col: str,
    parent_start_col: str,
    parent_end_col: str,
    parent_cols: list[str],
    parent_prefix: str,
) -> pd.DataFrame:
    if child.empty or parent.empty:
        return pd.DataFrame()

    child = child.copy().reset_index(drop=True)
    parent = parent.copy().sort_values(parent_start_col).reset_index(drop=True)
    parent = parent.reset_index(names="parent_row")

    child_ns = child[child_ts_col].astype("int64").to_numpy()
    parent_start_ns = parent[parent_start_col].astype("int64").to_numpy()
    parent_end_ns = parent[parent_end_col].astype("int64").to_numpy()

    idx = np.searchsorted(parent_start_ns, child_ns, side="right") - 1
    valid = idx >= 0
    parent_idx = np.full(len(child), -1, dtype=np.int64)
    parent_idx[valid] = idx[valid]

    covered = np.zeros(len(child), dtype=bool)
    covered[valid] = child_ns[valid] < parent_end_ns[idx[valid]]
    mapped = child.loc[covered].copy()
    mapped["parent_row"] = parent_idx[covered]

    cols = ["parent_row"] + parent_cols
    parent_view = parent[cols].rename(
        columns={col: f"{parent_prefix}{col}" for col in parent_cols}
    )
    return mapped.merge(parent_view, on="parent_row", how="left")


def _sign_ratio(series: pd.Series, target_sign: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    sign = np.sign(clean)
    non_zero = sign[sign != 0]
    if non_zero.empty:
        return np.nan
    return float((non_zero == target_sign).mean())


def _max_streak(signs: np.ndarray, target_sign: float) -> int:
    max_len = 0
    cur = 0
    for value in signs:
        if value == target_sign:
            cur += 1
            if cur > max_len:
                max_len = cur
        else:
            cur = 0
    return int(max_len)


def lower_candle_features(group: pd.DataFrame, parent_sign: float) -> dict[str, Any]:
    ppo_hist_delta_sign = np.sign(pd.to_numeric(group["ppo_hist_delta"], errors="coerce").fillna(0.0).to_numpy())
    price_delta_sign = np.sign(pd.to_numeric(group["price_delta_pct"], errors="coerce").fillna(0.0).to_numpy())
    final_hist_delta = pd.to_numeric(group["ppo_hist_delta"], errors="coerce").iloc[-1]

    rows = {
        "lower_candle_count": int(len(group)),
        "lower_distinct_cycle_count": int(group["cycle_uid"].nunique()),
        "lower_cycle_transition_count": int(max(group["cycle_uid"].nunique() - 1, 0)),
        "lower_ppo_mean": round(float(pd.to_numeric(group["ppo"], errors="coerce").mean()), 6),
        "lower_ppo_abs_mean": round(float(pd.to_numeric(group["ppo"], errors="coerce").abs().mean()), 6),
        "lower_ppo_hist_mean": round(float(pd.to_numeric(group["ppo_hist"], errors="coerce").mean()), 6),
        "lower_ppo_hist_abs_mean": round(float(pd.to_numeric(group["ppo_hist"], errors="coerce").abs().mean()), 6),
        "lower_ppo_delta_mean": round(float(pd.to_numeric(group["ppo_delta"], errors="coerce").mean()), 6),
        "lower_ppo_hist_delta_mean": round(float(pd.to_numeric(group["ppo_hist_delta"], errors="coerce").mean()), 6),
        "lower_price_delta_mean": round(float(pd.to_numeric(group["price_delta_pct"], errors="coerce").mean()), 6),
        "lower_same_direction_ratio": round(_sign_ratio(group["ppo_hist_delta"], parent_sign), 6),
        "lower_opposite_direction_ratio": round(_sign_ratio(group["ppo_hist_delta"], -parent_sign), 6),
        "lower_same_price_ratio": round(_sign_ratio(group["price_delta_pct"], parent_sign), 6),
        "lower_opposite_price_ratio": round(_sign_ratio(group["price_delta_pct"], -parent_sign), 6),
        "lower_max_opposite_hist_streak": _max_streak(ppo_hist_delta_sign, -parent_sign),
        "lower_max_same_hist_streak": _max_streak(ppo_hist_delta_sign, parent_sign),
        "lower_final_hist_delta_sign": float(np.sign(final_hist_delta)) if not pd.isna(final_hist_delta) else np.nan,
        "lower_final_alignment": float(np.sign(final_hist_delta) == parent_sign) if not pd.isna(final_hist_delta) and final_hist_delta != 0 else np.nan,
        "lower_price_flip_count": int(np.sum(np.diff(price_delta_sign[price_delta_sign != 0]) != 0)) if len(price_delta_sign[price_delta_sign != 0]) > 1 else 0,
        "lower_hist_flip_count": int(np.sum(np.diff(ppo_hist_delta_sign[ppo_hist_delta_sign != 0]) != 0)) if len(ppo_hist_delta_sign[ppo_hist_delta_sign != 0]) > 1 else 0,
    }

    first_half = group.iloc[: max(len(group) // 2, 1)]
    second_half = group.iloc[len(group) // 2 :]
    rows["lower_recovery_ratio_shift"] = round(
        _sign_ratio(second_half["ppo_hist_delta"], parent_sign) - _sign_ratio(first_half["ppo_hist_delta"], parent_sign),
        6,
    )
    return rows


def summarize_numeric(frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()

    for keys, group in frame.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["count"] = int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_avg"] = None
                row[f"{metric}_median"] = None
                row[f"{metric}_p25"] = None
                row[f"{metric}_p75"] = None
                continue
            row[f"{metric}_avg"] = round(float(values.mean()), 6)
            row[f"{metric}_median"] = round(float(values.median()), 6)
            row[f"{metric}_p25"] = round(float(values.quantile(0.25)), 6)
            row[f"{metric}_p75"] = round(float(values.quantile(0.75)), 6)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols + ["count"], ascending=[True] * len(group_cols) + [False])


def feature_contrast(
    frame: pd.DataFrame,
    *,
    pair_label: str,
    analysis_name: str,
    mask_a: pd.Series,
    mask_b: pd.Series,
    features: list[str],
    min_cases: int,
) -> pd.DataFrame:
    a = frame.loc[mask_a].copy()
    b = frame.loc[mask_b].copy()
    if len(a) < min_cases or len(b) < min_cases:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for feature in features:
        va = pd.to_numeric(a[feature], errors="coerce").dropna()
        vb = pd.to_numeric(b[feature], errors="coerce").dropna()
        if len(va) < min_cases or len(vb) < min_cases:
            continue
        mean_a = float(va.mean())
        mean_b = float(vb.mean())
        pooled = np.sqrt((float(va.var(ddof=1)) + float(vb.var(ddof=1))) / 2.0)
        effect_size = (mean_a - mean_b) / pooled if pooled and not np.isnan(pooled) else np.nan
        rows.append(
            {
                "pair": pair_label,
                "analysis": analysis_name,
                "feature": feature,
                "group_a_count": int(len(va)),
                "group_b_count": int(len(vb)),
                "group_a_mean": round(mean_a, 6),
                "group_b_mean": round(mean_b, 6),
                "mean_diff": round(mean_a - mean_b, 6),
                "effect_size": round(float(effect_size), 6) if not pd.isna(effect_size) else None,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("effect_size", key=lambda s: s.abs(), ascending=False)


def analyze_pair(
    lower_tf: str,
    upper_tf: str,
    candle_frames: dict[str, pd.DataFrame],
    cycle_frames: dict[str, pd.DataFrame],
    min_cases: int,
) -> dict[str, pd.DataFrame]:
    pair_label = f"{lower_tf}->{upper_tf}"

    lower_candles = candle_frames[lower_tf]
    upper_candles = candle_frames[upper_tf]
    lower_cycles = cycle_frames[lower_tf]
    upper_cycles = cycle_frames[upper_tf]

    candle_mapping = assign_parent_interval(
        lower_candles,
        upper_candles,
        child_ts_col="timestamp",
        parent_start_col="timestamp",
        parent_end_col="end_exclusive",
        parent_prefix="upper_",
        parent_cols=[
            "timeframe",
            "cycle_uid",
            "cycle_id",
            "cycle_key",
            "cycle_type",
            "cycle_sign",
            "timestamp",
            "parent_state",
            "next_parent_state",
            "is_rebound_candle",
            "noise_then_rebound_next",
            "cycle_progress",
        ],
    )

    parent_candle_cases: list[dict[str, Any]] = []
    if not candle_mapping.empty:
        for parent_row, group in candle_mapping.groupby("parent_row", sort=False):
            parent_info = group.iloc[0]
            features = lower_candle_features(group, float(parent_info["upper_cycle_sign"]))
            parent_candle_cases.append(
                {
                    "pair": pair_label,
                    "lower_tf": lower_tf,
                    "upper_tf": upper_tf,
                    "upper_parent_row": int(parent_row),
                    "upper_cycle_uid": parent_info["upper_cycle_uid"],
                    "upper_cycle_id": parent_info["upper_cycle_id"],
                    "upper_cycle_key": parent_info["upper_cycle_key"],
                    "upper_cycle_type": parent_info["upper_cycle_type"],
                    "upper_cycle_sign": parent_info["upper_cycle_sign"],
                    "upper_candle_timestamp": parent_info["upper_timestamp"],
                    "upper_candle_state": parent_info["upper_parent_state"],
                    "upper_next_state": parent_info["upper_next_parent_state"],
                    "upper_is_rebound_candle": bool(parent_info["upper_is_rebound_candle"]),
                    "upper_noise_then_rebound_next": bool(parent_info["upper_noise_then_rebound_next"]),
                    "upper_cycle_progress": float(parent_info["upper_cycle_progress"]),
                    **features,
                }
            )

    parent_candle_cases_df = pd.DataFrame(parent_candle_cases)
    if not parent_candle_cases_df.empty:
        parent_candle_cases_df["upper_progress_bucket"] = progress_bucket(parent_candle_cases_df["upper_cycle_progress"])
        parent_candle_cases_df["lower_ppo_regime"] = regime(
            parent_candle_cases_df["lower_ppo_mean"],
            parent_candle_cases_df["lower_ppo_hist_mean"],
        )

    cycle_mapping = assign_parent_interval(
        lower_cycles,
        upper_cycles,
        child_ts_col="start_ts",
        parent_start_col="start_ts",
        parent_end_col="end_exclusive",
        parent_prefix="upper_",
        parent_cols=[
            "timeframe",
            "cycle_uid",
            "cycle_id",
            "cycle_key",
            "cycle_type",
            "cycle_sign",
            "start_ts",
            "end_exclusive",
            "duration_candles",
        ],
    )

    lower_cycle_cases = pd.DataFrame()
    if not cycle_mapping.empty:
        lower_cycle_cases = cycle_mapping.rename(
            columns={
                "timeframe": "lower_tf",
                "cycle_uid": "lower_cycle_uid",
                "cycle_id": "lower_cycle_id",
                "cycle_key": "lower_cycle_key",
                "cycle_type": "lower_cycle_type",
                "cycle_sign": "lower_cycle_sign",
                "duration_candles": "lower_duration_candles",
                "cycle_signed_move_pct": "lower_cycle_signed_move_pct",
                "start_ts": "lower_cycle_start",
                "end_ts": "lower_cycle_end",
                "upper_timeframe": "upper_tf",
                "upper_cycle_uid": "upper_cycle_uid",
                "upper_cycle_id": "upper_cycle_id",
                "upper_cycle_key": "upper_cycle_key",
                "upper_cycle_type": "upper_cycle_type",
                "upper_cycle_sign": "upper_cycle_sign",
                "upper_start_ts": "upper_cycle_start",
                "upper_end_exclusive": "upper_cycle_end_exclusive",
                "upper_duration_candles": "upper_duration_candles",
            }
        )
        upper_span = (
            lower_cycle_cases["upper_cycle_end_exclusive"] - lower_cycle_cases["upper_cycle_start"]
        ).dt.total_seconds().replace(0, np.nan)
        lower_cycle_cases["upper_cycle_progress"] = (
            (lower_cycle_cases["lower_cycle_start"] - lower_cycle_cases["upper_cycle_start"]).dt.total_seconds() / upper_span
        )
        lower_cycle_cases["upper_progress_bucket"] = progress_bucket(lower_cycle_cases["upper_cycle_progress"])
        lower_cycle_cases["same_direction_as_upper"] = (
            lower_cycle_cases["lower_cycle_sign"] == lower_cycle_cases["upper_cycle_sign"]
        )
        lower_cycle_cases["pair"] = pair_label
        lower_cycle_cases = lower_cycle_cases.sort_values(["upper_cycle_uid", "lower_cycle_start"]).reset_index(drop=True)
        lower_cycle_cases["order_in_upper_cycle"] = lower_cycle_cases.groupby("upper_cycle_uid").cumcount() + 1
        lower_cycle_cases["total_lower_cycles_in_upper"] = lower_cycle_cases.groupby("upper_cycle_uid")["lower_cycle_uid"].transform("count")

    numeric_features = [
        "lower_candle_count",
        "lower_distinct_cycle_count",
        "lower_cycle_transition_count",
        "lower_ppo_mean",
        "lower_ppo_abs_mean",
        "lower_ppo_hist_mean",
        "lower_ppo_hist_abs_mean",
        "lower_ppo_delta_mean",
        "lower_ppo_hist_delta_mean",
        "lower_price_delta_mean",
        "lower_same_direction_ratio",
        "lower_opposite_direction_ratio",
        "lower_same_price_ratio",
        "lower_opposite_price_ratio",
        "lower_max_opposite_hist_streak",
        "lower_max_same_hist_streak",
        "lower_final_alignment",
        "lower_price_flip_count",
        "lower_hist_flip_count",
        "lower_recovery_ratio_shift",
    ]

    state_summary = summarize_numeric(
        parent_candle_cases_df,
        ["pair", "lower_tf", "upper_tf", "upper_candle_state", "upper_progress_bucket"],
        numeric_features,
    )
    rebound_summary = summarize_numeric(
        parent_candle_cases_df[parent_candle_cases_df["upper_candle_state"] == "noise"].copy(),
        ["pair", "lower_tf", "upper_tf", "upper_noise_then_rebound_next", "upper_progress_bucket"],
        numeric_features,
    )
    cycle_progress_summary = summarize_numeric(
        lower_cycle_cases,
        ["pair", "lower_tf", "upper_tf", "upper_cycle_type", "upper_progress_bucket", "same_direction_as_upper"],
        ["lower_cycle_signed_move_pct", "lower_duration_candles", "order_in_upper_cycle", "total_lower_cycles_in_upper"],
    )

    noise_cases = parent_candle_cases_df[parent_candle_cases_df["upper_candle_state"] == "noise"].copy()
    contrasts = [
        feature_contrast(
            parent_candle_cases_df,
            pair_label=pair_label,
            analysis_name="upper_noise_vs_trend",
            mask_a=parent_candle_cases_df["upper_candle_state"] == "noise",
            mask_b=parent_candle_cases_df["upper_candle_state"] == "trend",
            features=numeric_features,
            min_cases=min_cases,
        ),
        feature_contrast(
            noise_cases,
            pair_label=pair_label,
            analysis_name="noise_then_rebound_vs_stall",
            mask_a=noise_cases["upper_noise_then_rebound_next"] == True,
            mask_b=noise_cases["upper_noise_then_rebound_next"] == False,
            features=numeric_features,
            min_cases=min_cases,
        ),
    ]
    contrast_df = pd.concat([df for df in contrasts if not df.empty], ignore_index=True) if any(not df.empty for df in contrasts) else pd.DataFrame()

    return {
        "parent_candle_cases": parent_candle_cases_df,
        "state_summary": state_summary,
        "rebound_summary": rebound_summary,
        "lower_cycle_cases": lower_cycle_cases,
        "cycle_progress_summary": cycle_progress_summary,
        "feature_contrasts": contrast_df,
    }


def build_report(
    *,
    timeframes: tuple[str, ...],
    state_summary: pd.DataFrame,
    rebound_summary: pd.DataFrame,
    cycle_progress_summary: pd.DataFrame,
    feature_contrasts: pd.DataFrame,
    meta: dict[str, Any],
) -> str:
    lines = [
        "# PPO Multitimeframe Influence Analysis",
        "",
        "## Focus",
        "",
        "- Low timeframe PPO structure inside higher timeframe candles and cycles.",
        "- Higher-candle states are labeled with PPO histogram slope relative to the higher-cycle direction.",
        "- `noise`: higher-candle `ppo_hist_delta` opposes the higher-cycle direction.",
        "- `trend`: higher-candle `ppo_hist_delta` aligns with the higher-cycle direction.",
        "- `rebound`: a higher-candle trend candle that appears right after one or more higher noise candles.",
        "",
        "## Metadata",
        "",
        f"- timeframes: `{', '.join(timeframes)}`",
        f"- pairs: `{meta['pair_count']}`",
        f"- parent_candle_case_count: `{meta['parent_candle_case_count']}`",
        f"- lower_cycle_case_count: `{meta['lower_cycle_case_count']}`",
        f"- output_dir: `{meta['output_dir']}`",
        "",
    ]

    if not feature_contrasts.empty:
        lines.extend(["## Strongest Contrasts", ""])
        top = feature_contrasts.groupby(["pair", "analysis"], group_keys=False).head(5)
        lines.append(top.to_markdown(index=False))
        lines.append("")

    if not rebound_summary.empty:
        lines.extend(["## Noise To Rebound Snapshot", ""])
        cols = [
            "pair",
            "upper_noise_then_rebound_next",
            "upper_progress_bucket",
            "count",
            "lower_same_direction_ratio_avg",
            "lower_opposite_direction_ratio_avg",
            "lower_recovery_ratio_shift_avg",
            "lower_cycle_transition_count_avg",
            "lower_max_opposite_hist_streak_avg",
        ]
        view = rebound_summary[[col for col in cols if col in rebound_summary.columns]].copy()
        lines.append(view.head(30).to_markdown(index=False))
        lines.append("")

    if not cycle_progress_summary.empty:
        lines.extend(["## Big-Cycle Progress Snapshot", ""])
        cols = [
            "pair",
            "upper_cycle_type",
            "upper_progress_bucket",
            "same_direction_as_upper",
            "count",
            "lower_cycle_signed_move_pct_avg",
            "lower_duration_candles_avg",
            "order_in_upper_cycle_avg",
            "total_lower_cycles_in_upper_avg",
        ]
        view = cycle_progress_summary[[col for col in cols if col in cycle_progress_summary.columns]].copy()
        lines.append(view.head(40).to_markdown(index=False))
        lines.append("")

    if not state_summary.empty:
        lines.extend(["## Parent Candle State Snapshot", ""])
        cols = [
            "pair",
            "upper_candle_state",
            "upper_progress_bucket",
            "count",
            "lower_same_direction_ratio_avg",
            "lower_opposite_direction_ratio_avg",
            "lower_hist_flip_count_avg",
            "lower_cycle_transition_count_avg",
        ]
        view = state_summary[[col for col in cols if col in state_summary.columns]].copy()
        lines.append(view.head(40).to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def run(timeframes: tuple[str, ...] = TIMEFRAMES, min_cases: int = DEFAULT_MIN_CASES) -> dict[str, Any]:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    loaded_cycles = {tf: load_cycles(tf) for tf in timeframes}
    candle_frames = {tf: build_candle_frame(loaded_cycles[tf], tf) for tf in timeframes}
    cycle_frames = {tf: build_cycle_interval_frame(loaded_cycles[tf], tf) for tf in timeframes}

    all_parent_candle_cases: list[pd.DataFrame] = []
    all_state_summary: list[pd.DataFrame] = []
    all_rebound_summary: list[pd.DataFrame] = []
    all_lower_cycle_cases: list[pd.DataFrame] = []
    all_cycle_progress_summary: list[pd.DataFrame] = []
    all_feature_contrasts: list[pd.DataFrame] = []

    for lower_tf, upper_tf in cycle_pairs(timeframes):
        result = analyze_pair(lower_tf, upper_tf, candle_frames, cycle_frames, min_cases=min_cases)
        if not result["parent_candle_cases"].empty:
            all_parent_candle_cases.append(result["parent_candle_cases"])
        if not result["state_summary"].empty:
            all_state_summary.append(result["state_summary"])
        if not result["rebound_summary"].empty:
            all_rebound_summary.append(result["rebound_summary"])
        if not result["lower_cycle_cases"].empty:
            all_lower_cycle_cases.append(result["lower_cycle_cases"])
        if not result["cycle_progress_summary"].empty:
            all_cycle_progress_summary.append(result["cycle_progress_summary"])
        if not result["feature_contrasts"].empty:
            all_feature_contrasts.append(result["feature_contrasts"])

    parent_candle_cases = pd.concat(all_parent_candle_cases, ignore_index=True) if all_parent_candle_cases else pd.DataFrame()
    state_summary = pd.concat(all_state_summary, ignore_index=True) if all_state_summary else pd.DataFrame()
    rebound_summary = pd.concat(all_rebound_summary, ignore_index=True) if all_rebound_summary else pd.DataFrame()
    lower_cycle_cases = pd.concat(all_lower_cycle_cases, ignore_index=True) if all_lower_cycle_cases else pd.DataFrame()
    cycle_progress_summary = pd.concat(all_cycle_progress_summary, ignore_index=True) if all_cycle_progress_summary else pd.DataFrame()
    feature_contrasts = pd.concat(all_feature_contrasts, ignore_index=True) if all_feature_contrasts else pd.DataFrame()

    if not parent_candle_cases.empty:
        parent_candle_cases.to_csv(out / "parent_candle_cases.csv", index=False, encoding="utf-8-sig")
    if not state_summary.empty:
        state_summary.to_csv(out / "parent_candle_state_summary.csv", index=False, encoding="utf-8-sig")
    if not rebound_summary.empty:
        rebound_summary.to_csv(out / "noise_to_rebound_summary.csv", index=False, encoding="utf-8-sig")
    if not lower_cycle_cases.empty:
        lower_cycle_cases.to_csv(out / "lower_cycle_in_upper_cycle_cases.csv", index=False, encoding="utf-8-sig")
    if not cycle_progress_summary.empty:
        cycle_progress_summary.to_csv(out / "lower_cycle_progress_summary.csv", index=False, encoding="utf-8-sig")
    if not feature_contrasts.empty:
        feature_contrasts.to_csv(out / "feature_contrasts.csv", index=False, encoding="utf-8-sig")

    meta = {
        "timeframes": list(timeframes),
        "pair_count": len(cycle_pairs(timeframes)),
        "parent_candle_case_count": int(len(parent_candle_cases)),
        "lower_cycle_case_count": int(len(lower_cycle_cases)),
        "output_dir": str(out),
        "min_cases_for_contrast": int(min_cases),
    }
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(
        build_report(
            timeframes=timeframes,
            state_summary=state_summary,
            rebound_summary=rebound_summary,
            cycle_progress_summary=cycle_progress_summary,
            feature_contrasts=feature_contrasts,
            meta=meta,
        ),
        encoding="utf-8",
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze how lower-timeframe PPO structure influences higher-timeframe cycles.")
    parser.add_argument("--timeframes", nargs="+", default=list(TIMEFRAMES), choices=list(TIMEFRAMES))
    parser.add_argument("--min-cases", type=int, default=DEFAULT_MIN_CASES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(timeframes=tuple(args.timeframes), min_cases=args.min_cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))
