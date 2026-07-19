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

from scripts.analysis.ppo_multitimeframe_influence_analysis import (
    TF_SECONDS,
    build_candle_frame,
    build_cycle_interval_frame,
    cycle_sign,
    load_cycles,
    progress_bucket,
    quantile_bucket,
    regime,
    summarize_numeric,
)
from src.common.paths import PROJECT_PATHS


TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d", "1w")
LOWER_TF = {"15m": "5m", "1h": "15m", "4h": "1h", "1d": "4h", "1w": "1d"}
UPPER_TF = {"5m": "15m", "15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}
DEFAULT_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")
DEFAULT_HORIZONS = (1, 3, 5, 10)
DEFAULT_START_WINDOW = 2
DEFAULT_MIN_CASES = 40


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_candle_reversal_noise_analysis"


def to_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def signed_return(entry: float, exit_price: float, sign: float) -> float:
    if pd.isna(entry) or pd.isna(exit_price) or entry == 0 or pd.isna(sign):
        return np.nan
    return (exit_price / entry - 1.0) * 100.0 * sign


def add_position_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["ppo_regime"] = regime(frame["ppo"], frame["ppo_hist"])
    frame["ppo_quantile"] = quantile_bucket(frame["ppo"], "ppo")
    frame["ppo_hist_quantile"] = quantile_bucket(frame["ppo_hist"], "hist")
    frame["ppo_abs_quantile"] = quantile_bucket(frame["ppo"].abs(), "ppo_abs")
    frame["ppo_hist_abs_quantile"] = quantile_bucket(frame["ppo_hist"].abs(), "hist_abs")
    frame["cycle_progress_bucket"] = progress_bucket(frame["cycle_progress"])
    frame["candle_age_bucket"] = pd.cut(
        pd.to_numeric(frame["candle_index"], errors="coerce"),
        bins=[0, 2, 5, 10, 20, 50, 10**9],
        labels=["age_1_2", "age_3_5", "age_6_10", "age_11_20", "age_21_50", "age_51_plus"],
        include_lowest=True,
    )
    return frame


def build_reversal_candidates(
    candles: pd.DataFrame,
    cycles: pd.DataFrame,
    *,
    timeframe: str,
    horizons: tuple[int, ...],
    start_window: int,
) -> pd.DataFrame:
    cycles = cycles.sort_values("start_date").reset_index(drop=True)
    next_cycle = cycles[["cycle_uid", "cycle_type", "cycle_sign", "start_date", "duration_candles", "cycle_signed_move_pct"]].shift(-1)
    next_cycle = next_cycle.rename(
        columns={
            "cycle_uid": "next_cycle_uid",
            "cycle_type": "next_cycle_type",
            "cycle_sign": "next_cycle_sign",
            "start_date": "next_cycle_start",
            "duration_candles": "next_cycle_duration_candles",
            "cycle_signed_move_pct": "next_cycle_signed_move_pct",
        }
    )
    cycle_lookup = pd.concat([cycles[["cycle_uid"]].reset_index(drop=True), next_cycle], axis=1)
    frame = candles.merge(cycle_lookup, on="cycle_uid", how="left").sort_values(["cycle_uid", "candle_index"]).reset_index(drop=True)
    if frame.empty:
        return frame

    sign = pd.to_numeric(frame["cycle_sign"], errors="coerce")
    frame["cycle_candle_count"] = frame.groupby("cycle_uid")["cycle_uid"].transform("size")
    frame["candles_to_cycle_end"] = frame["cycle_candle_count"] - pd.to_numeric(frame["candle_index"], errors="coerce")
    frame["hist_opposite"] = np.sign(pd.to_numeric(frame["ppo_hist_delta"], errors="coerce")) == -sign
    frame["ppo_slope_opposite"] = np.sign(pd.to_numeric(frame["ppo_delta"], errors="coerce")) == -sign
    frame["price_opposite"] = np.sign(pd.to_numeric(frame["price_delta_pct"], errors="coerce")) == -sign
    frame[["hist_opposite", "ppo_slope_opposite", "price_opposite"]] = frame[
        ["hist_opposite", "ppo_slope_opposite", "price_opposite"]
    ].fillna(False)
    frame["opposite_signal_count"] = (
        frame["hist_opposite"].astype(int) + frame["ppo_slope_opposite"].astype(int) + frame["price_opposite"].astype(int)
    )
    frame = frame[(frame["candle_index"] > 1) & (frame["opposite_signal_count"] > 0)].copy()
    if frame.empty:
        return add_position_features(frame)

    next_sign = pd.to_numeric(frame["next_cycle_sign"], errors="coerce")
    next_is_opposite = next_sign == -pd.to_numeric(frame["cycle_sign"], errors="coerce")
    frame["is_true_new_cycle_start"] = next_is_opposite & (pd.to_numeric(frame["candles_to_cycle_end"], errors="coerce") <= start_window)
    frame["reversal_label"] = np.where(frame["is_true_new_cycle_start"], "true_new_cycle_start", "continuation_noise")
    ambiguous = (~frame["is_true_new_cycle_start"]) & (frame["candles_to_cycle_end"] <= start_window) & (~next_is_opposite.fillna(False))
    frame.loc[ambiguous, "reversal_label"] = "ambiguous_tail_reversal"
    frame["is_continuation_noise"] = frame["reversal_label"].eq("continuation_noise")

    base = candles[["cycle_uid", "candle_index", "close", "cycle_sign", "ppo_hist_delta", "price_delta_pct"]].copy()
    base = base.sort_values(["cycle_uid", "candle_index"]).reset_index(drop=True)
    max_horizon = max(horizons) if horizons else 1
    base_sign = pd.to_numeric(base["cycle_sign"], errors="coerce")
    base["hist_opposite_bool"] = np.sign(pd.to_numeric(base["ppo_hist_delta"], errors="coerce")) == -base_sign
    base["hist_cycle_bool"] = np.sign(pd.to_numeric(base["ppo_hist_delta"], errors="coerce")) == base_sign
    base["price_opposite_bool"] = np.sign(pd.to_numeric(base["price_delta_pct"], errors="coerce")) == -base_sign
    base["price_cycle_bool"] = np.sign(pd.to_numeric(base["price_delta_pct"], errors="coerce")) == base_sign

    def future_count(values: pd.Series) -> pd.Series:
        arr = values.astype(int).to_numpy()
        counts = np.zeros(len(arr), dtype=int)
        csum = np.concatenate([[0], np.cumsum(arr)])
        for idx in range(len(arr)):
            counts[idx] = csum[min(len(arr), idx + 1 + max_horizon)] - csum[idx + 1]
        return pd.Series(counts, index=values.index)

    for source_col, out_col in [
        ("hist_opposite_bool", "opposite_followthrough_count"),
        ("hist_cycle_bool", "cycle_recovery_count"),
        ("price_opposite_bool", "opposite_price_followthrough_count"),
        ("price_cycle_bool", "cycle_price_recovery_count"),
    ]:
        base[out_col] = base.groupby("cycle_uid", group_keys=False)[source_col].apply(future_count)

    count_cols = [
        "opposite_followthrough_count",
        "cycle_recovery_count",
        "opposite_price_followthrough_count",
        "cycle_price_recovery_count",
    ]
    frame = frame.merge(base[["cycle_uid", "candle_index", *count_cols]], on=["cycle_uid", "candle_index"], how="left")

    for horizon in horizons:
        target = candles[["cycle_uid", "candle_index", "close"]].copy()
        target["candle_index"] = target["candle_index"] - horizon
        target = target.rename(columns={"close": f"target_close_{horizon}"})
        frame = frame.merge(target, on=["cycle_uid", "candle_index"], how="left")
        frame[f"ret_fwd_{horizon}_opposite_pct"] = (
            frame[f"target_close_{horizon}"] / frame["close"] - 1.0
        ) * 100.0 * -pd.to_numeric(frame["cycle_sign"], errors="coerce")
        frame[f"ret_fwd_{horizon}_cycle_pct"] = (
            frame[f"target_close_{horizon}"] / frame["close"] - 1.0
        ) * 100.0 * pd.to_numeric(frame["cycle_sign"], errors="coerce")
        frame = frame.drop(columns=[f"target_close_{horizon}"])

    end_close = candles.sort_values(["cycle_uid", "candle_index"]).groupby("cycle_uid").tail(1)[["cycle_uid", "close"]]
    end_close = end_close.rename(columns={"close": "cycle_end_close"})
    frame = frame.merge(end_close, on="cycle_uid", how="left")
    frame["ret_to_cycle_end_opposite_pct"] = (
        frame["cycle_end_close"] / frame["close"] - 1.0
    ) * 100.0 * -pd.to_numeric(frame["cycle_sign"], errors="coerce")
    frame["ret_to_cycle_end_cycle_pct"] = (
        frame["cycle_end_close"] / frame["close"] - 1.0
    ) * 100.0 * pd.to_numeric(frame["cycle_sign"], errors="coerce")

    frame = frame.rename(columns={"end_exclusive": "event_end_exclusive"})
    keep_cols = [
        "timeframe",
        "cycle_uid",
        "cycle_id",
        "cycle_type",
        "cycle_sign",
        "cycle_start",
        "cycle_end",
        "timestamp",
        "event_end_exclusive",
        "candle_index",
        "cycle_candle_count",
        "candles_to_cycle_end",
        "cycle_progress",
        "close",
        "ppo",
        "ppo_hist",
        "ppo_delta",
        "ppo_hist_delta",
        "price_delta_pct",
        "hist_opposite",
        "ppo_slope_opposite",
        "price_opposite",
        "opposite_signal_count",
        "reversal_label",
        "is_true_new_cycle_start",
        "is_continuation_noise",
        "next_cycle_uid",
        "next_cycle_type",
        "next_cycle_start",
        "next_cycle_duration_candles",
        "next_cycle_signed_move_pct",
        *count_cols,
        *[f"ret_fwd_{horizon}_opposite_pct" for horizon in horizons],
        *[f"ret_fwd_{horizon}_cycle_pct" for horizon in horizons],
        "ret_to_cycle_end_opposite_pct",
        "ret_to_cycle_end_cycle_pct",
    ]
    return add_position_features(frame[[col for col in keep_cols if col in frame.columns]])


def assign_context_interval(
    events: pd.DataFrame,
    context: pd.DataFrame,
    *,
    prefix: str,
    event_ts_col: str = "timestamp",
    context_start_col: str = "timestamp",
    context_end_col: str = "end_exclusive",
    context_cols: list[str],
) -> pd.DataFrame:
    if events.empty or context.empty:
        return events.copy()

    events = events.copy().reset_index(drop=True)
    context = context.copy().sort_values(context_start_col).reset_index(drop=True)
    event_ns = events[event_ts_col].astype("int64").to_numpy()
    start_ns = context[context_start_col].astype("int64").to_numpy()
    end_ns = context[context_end_col].astype("int64").to_numpy()
    idx = np.searchsorted(start_ns, event_ns, side="right") - 1
    valid = idx >= 0
    covered = np.zeros(len(events), dtype=bool)
    covered[valid] = event_ns[valid] < end_ns[idx[valid]]

    source = pd.DataFrame(index=events.index)
    for col in context_cols:
        source[f"{prefix}{col}"] = pd.Series(pd.NA, index=events.index, dtype="object")
    if covered.any():
        mapped = context.iloc[idx[covered]][context_cols].reset_index(drop=True)
        mapped.index = events.index[covered]
        mapped = mapped.rename(columns={col: f"{prefix}{col}" for col in mapped.columns})
        source.loc[mapped.index, mapped.columns] = mapped
    events = events.drop(columns=[col for col in source.columns if col in events.columns], errors="ignore")
    return pd.concat([events, source], axis=1)


def aggregate_lower_context(events: pd.DataFrame, lower_candles: pd.DataFrame, lower_tf: str) -> pd.DataFrame:
    if events.empty or lower_candles.empty:
        return events.copy()

    events = events.copy().reset_index(drop=True)
    lower = lower_candles.sort_values("timestamp").reset_index(drop=True)
    lower_ts = lower["timestamp"].to_numpy()
    starts = pd.to_datetime(events["timestamp"], errors="coerce").to_numpy()
    ends = pd.to_datetime(events["event_end_exclusive"], errors="coerce").to_numpy()
    left = np.searchsorted(lower_ts, starts, side="left")
    right = np.searchsorted(lower_ts, ends, side="left")
    counts = right - left
    valid = counts > 0

    def prefix_sum(values: pd.Series) -> np.ndarray:
        arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        return np.concatenate([[0.0], np.cumsum(arr)])

    def range_mean(values: pd.Series) -> np.ndarray:
        pref = prefix_sum(values)
        sums = pref[right] - pref[left]
        out = np.full(len(events), np.nan, dtype="float64")
        np.divide(sums, counts, out=out, where=valid)
        return np.round(out, 6)

    hist_sign = np.sign(pd.to_numeric(lower["ppo_hist_delta"], errors="coerce").fillna(0.0)).to_numpy()
    price_sign = np.sign(pd.to_numeric(lower["price_delta_pct"], errors="coerce").fillna(0.0)).to_numpy()
    hist_pos_pref = np.concatenate([[0], np.cumsum(hist_sign == 1)])
    hist_neg_pref = np.concatenate([[0], np.cumsum(hist_sign == -1)])
    price_pos_pref = np.concatenate([[0], np.cumsum(price_sign == 1)])
    price_neg_pref = np.concatenate([[0], np.cumsum(price_sign == -1)])
    event_sign = pd.to_numeric(events["cycle_sign"], errors="coerce").to_numpy()

    def signed_ratio(pos_pref: np.ndarray, neg_pref: np.ndarray, target: np.ndarray) -> np.ndarray:
        pos_counts = pos_pref[right] - pos_pref[left]
        neg_counts = neg_pref[right] - neg_pref[left]
        selected = np.where(target == 1, pos_counts, np.where(target == -1, neg_counts, np.nan))
        out = np.full(len(events), np.nan, dtype="float64")
        np.divide(selected, counts, out=out, where=valid)
        return np.round(out, 6)

    cycle_codes = pd.Categorical(lower["cycle_uid"]).codes
    transitions = np.zeros(len(lower), dtype=int)
    transitions[1:] = cycle_codes[1:] != cycle_codes[:-1]
    transition_pref = np.concatenate([[0], np.cumsum(transitions)])
    distinct_counts = np.where(valid, (transition_pref[right] - transition_pref[left]) + 1, np.nan)

    final_idx = np.clip(right - 1, 0, len(lower) - 1)
    final_type = np.full(len(events), pd.NA, dtype="object")
    final_sign = np.full(len(events), np.nan, dtype="float64")
    if valid.any():
        final_type[valid] = lower.iloc[final_idx[valid]]["cycle_type"].astype(str).to_numpy()
        final_sign[valid] = pd.to_numeric(lower.iloc[final_idx[valid]]["cycle_sign"], errors="coerce").to_numpy()

    context = pd.DataFrame(
        {
            "lower_tf": np.where(valid, lower_tf, pd.NA),
            "lower_candle_count": np.where(valid, counts, np.nan),
            "lower_distinct_cycle_count": distinct_counts,
            "lower_ppo_mean": range_mean(lower["ppo"]),
            "lower_ppo_hist_mean": range_mean(lower["ppo_hist"]),
            "lower_ppo_delta_mean": range_mean(lower["ppo_delta"]),
            "lower_ppo_hist_delta_mean": range_mean(lower["ppo_hist_delta"]),
            "lower_opposite_hist_ratio": signed_ratio(hist_pos_pref, hist_neg_pref, -event_sign),
            "lower_cycle_hist_ratio": signed_ratio(hist_pos_pref, hist_neg_pref, event_sign),
            "lower_opposite_price_ratio": signed_ratio(price_pos_pref, price_neg_pref, -event_sign),
            "lower_cycle_price_ratio": signed_ratio(price_pos_pref, price_neg_pref, event_sign),
            "lower_final_cycle_type": final_type,
            "lower_final_cycle_same_as_event": final_sign == event_sign,
        }
    )
    return pd.concat([events, context], axis=1)


def add_higher_context(events: pd.DataFrame, candle_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = events.copy()
    for tf in TIMEFRAMES:
        if tf not in UPPER_TF:
            continue
        upper_tf = UPPER_TF[tf]
        mask = result["timeframe"].eq(tf)
        if not mask.any() or upper_tf not in candle_frames:
            continue
        context_cols = [
            "timeframe",
            "cycle_uid",
            "cycle_type",
            "cycle_sign",
            "timestamp",
            "cycle_progress",
            "parent_state",
            "ppo",
            "ppo_hist",
            "ppo_delta",
            "ppo_hist_delta",
            "price_delta_pct",
            "prev_noise_streak",
        ]
        subset = assign_context_interval(
            result.loc[mask],
            candle_frames[upper_tf],
            prefix="upper_",
            context_cols=context_cols,
        )
        for col in subset.columns:
            if col.startswith("upper_"):
                result.loc[mask, col] = subset[col].to_numpy()

    if "upper_ppo" in result:
        result["upper_ppo_quantile"] = result.groupby("upper_timeframe", dropna=False)["upper_ppo"].transform(
            lambda s: quantile_bucket(s, "upper_ppo")
        )
        result["upper_ppo_hist_quantile"] = result.groupby("upper_timeframe", dropna=False)["upper_ppo_hist"].transform(
            lambda s: quantile_bucket(s, "upper_hist")
        )
        result["upper_progress_bucket"] = progress_bucket(result["upper_cycle_progress"])
        upper_sign = pd.to_numeric(result["upper_cycle_sign"], errors="coerce")
        event_sign = pd.to_numeric(result["cycle_sign"], errors="coerce")
        result["upper_same_cycle_direction"] = upper_sign == event_sign
        result["upper_hist_opposes_event_cycle"] = (
            np.sign(pd.to_numeric(result["upper_ppo_hist_delta"], errors="coerce")) == -pd.to_numeric(result["cycle_sign"], errors="coerce")
        )
    return result


def add_lower_context(events: pd.DataFrame, candle_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for tf, group in events.groupby("timeframe", sort=False):
        lower_tf = LOWER_TF.get(str(tf))
        if lower_tf is None or lower_tf not in candle_frames:
            frames.append(group.copy())
            continue
        frames.append(aggregate_lower_context(group.copy(), candle_frames[lower_tf], lower_tf))
    return pd.concat(frames, ignore_index=True) if frames else events


def contrast_true_vs_noise(cases: pd.DataFrame, min_cases: int) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()

    features = [
        "cycle_progress",
        "candles_to_cycle_end",
        "candle_index",
        "ppo",
        "ppo_hist",
        "ppo_delta",
        "ppo_hist_delta",
        "price_delta_pct",
        "opposite_signal_count",
        "opposite_followthrough_count",
        "cycle_recovery_count",
        "opposite_price_followthrough_count",
        "cycle_price_recovery_count",
        "upper_cycle_progress",
        "upper_ppo",
        "upper_ppo_hist",
        "upper_ppo_delta",
        "upper_ppo_hist_delta",
        "upper_prev_noise_streak",
        "lower_candle_count",
        "lower_distinct_cycle_count",
        "lower_ppo_mean",
        "lower_ppo_hist_mean",
        "lower_ppo_delta_mean",
        "lower_ppo_hist_delta_mean",
        "lower_opposite_hist_ratio",
        "lower_cycle_hist_ratio",
        "lower_opposite_price_ratio",
        "lower_cycle_price_ratio",
    ]
    rows: list[dict[str, Any]] = []
    for tf, group in cases.groupby("timeframe", dropna=False):
        true_group = group[group["reversal_label"].eq("true_new_cycle_start")]
        noise_group = group[group["reversal_label"].eq("continuation_noise")]
        if len(true_group) < min_cases or len(noise_group) < min_cases:
            continue
        for feature in features:
            if feature not in group.columns:
                continue
            a = pd.to_numeric(true_group[feature], errors="coerce").dropna()
            b = pd.to_numeric(noise_group[feature], errors="coerce").dropna()
            if len(a) < min_cases or len(b) < min_cases:
                continue
            pooled = np.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2.0)
            effect = (float(a.mean()) - float(b.mean())) / pooled if pooled and not np.isnan(pooled) else np.nan
            rows.append(
                {
                    "timeframe": tf,
                    "feature": feature,
                    "true_count": int(len(a)),
                    "noise_count": int(len(b)),
                    "true_mean": round(float(a.mean()), 6),
                    "noise_mean": round(float(b.mean()), 6),
                    "mean_diff_true_minus_noise": round(float(a.mean() - b.mean()), 6),
                    "effect_size": round(float(effect), 6) if not pd.isna(effect) else None,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["timeframe", "effect_size"], key=lambda s: s.abs() if s.name == "effect_size" else s, ascending=False)


def ratio_summary(cases: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if cases.empty:
        return pd.DataFrame()
    for keys, group in cases.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["count"] = int(len(group))
        row["true_start_count"] = int(group["reversal_label"].eq("true_new_cycle_start").sum())
        row["noise_count"] = int(group["reversal_label"].eq("continuation_noise").sum())
        row["ambiguous_count"] = int(group["reversal_label"].eq("ambiguous_tail_reversal").sum())
        row["true_start_rate_pct"] = round(float(row["true_start_count"] / row["count"] * 100.0), 6) if row["count"] else np.nan
        row["noise_rate_pct"] = round(float(row["noise_count"] / row["count"] * 100.0), 6) if row["count"] else np.nan
        row["avg_next_cycle_signed_move_pct"] = round(float(pd.to_numeric(group["next_cycle_signed_move_pct"], errors="coerce").mean()), 6)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols + ["count"], ascending=[True] * len(group_cols) + [False])


def build_report(
    *,
    cases: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    contrasts: pd.DataFrame,
    meta: dict[str, Any],
) -> str:
    lines = [
        "# PPO Candle Reversal Noise Analysis",
        "",
        "## Label Rule",
        "",
        "- Candidate candle: price, PPO slope, or PPO histogram slope moves opposite to the current cycle direction.",
        f"- `true_new_cycle_start`: candidate is within `{meta['start_window_candles']}` candles of the current cycle end and the next cycle is opposite.",
        "- `continuation_noise`: candidate appears inside the cycle but does not become the formal next-cycle start zone.",
        "- Future cycle boundaries are labels only; feature columns use the candidate candle close and aligned upper/lower timeframe state.",
        "",
        "## Metadata",
        "",
        f"- timeframes: `{', '.join(meta['timeframes'])}`",
        f"- horizons: `{', '.join(str(v) for v in meta['horizons'])}`",
        f"- candidate_count: `{meta['candidate_count']}`",
        f"- output_dir: `{meta['output_dir']}`",
        "",
    ]

    if not cases.empty:
        label_counts = cases["reversal_label"].value_counts().rename_axis("label").reset_index(name="count")
        label_counts["rate_pct"] = (label_counts["count"] / len(cases) * 100.0).round(4)
        lines.extend(["## Overall Label Mix", "", label_counts.to_markdown(index=False), ""])

    for title, key, count_col in [
        ("PPO Position Summary", "ppo_position_summary", "count"),
        ("Upper Context Summary", "upper_context_summary", "count"),
        ("Lower Context Summary", "lower_context_summary", "count"),
    ]:
        table = summaries.get(key, pd.DataFrame())
        if table.empty:
            continue
        view = table.sort_values(["timeframe", "true_start_rate_pct", count_col], ascending=[True, False, False]).head(40)
        lines.extend([f"## {title}", "", view.to_markdown(index=False), ""])

    if not contrasts.empty:
        lines.extend(["## Strongest True Vs Noise Feature Gaps", ""])
        lines.append(contrasts.groupby("timeframe", group_keys=False).head(12).to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def run(
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    start_window: int = DEFAULT_START_WINDOW,
    min_cases: int = DEFAULT_MIN_CASES,
    include_cases: bool = True,
) -> dict[str, Any]:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    required_tfs = set(timeframes)
    required_tfs.update(LOWER_TF[tf] for tf in timeframes if tf in LOWER_TF)
    required_tfs.update(UPPER_TF[tf] for tf in timeframes if tf in UPPER_TF)
    loaded_cycles = {tf: load_cycles(tf) for tf in TIMEFRAMES if tf in required_tfs}
    candle_frames = {tf: build_candle_frame(loaded_cycles[tf], tf) for tf in loaded_cycles}
    cycle_frames = {tf: build_cycle_interval_frame(loaded_cycles[tf], tf) for tf in loaded_cycles}

    candidate_frames = [
        build_reversal_candidates(
            candle_frames[tf],
            loaded_cycles[tf],
            timeframe=tf,
            horizons=horizons,
            start_window=start_window,
        )
        for tf in timeframes
        if tf in candle_frames and tf in cycle_frames
    ]
    cases = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    if not cases.empty:
        cases = add_higher_context(cases, candle_frames)
        cases = add_lower_context(cases, candle_frames)

    summaries = {
        "ppo_position_summary": ratio_summary(
            cases,
            ["timeframe", "cycle_type", "ppo_regime", "ppo_quantile", "ppo_hist_quantile", "cycle_progress_bucket"],
        ),
        "upper_context_summary": ratio_summary(
            cases.dropna(subset=["upper_timeframe"]) if "upper_timeframe" in cases else pd.DataFrame(),
            ["timeframe", "upper_timeframe", "upper_cycle_type", "upper_parent_state", "upper_progress_bucket", "upper_ppo_quantile", "upper_ppo_hist_quantile"],
        ),
        "lower_context_summary": ratio_summary(
            cases.dropna(subset=["lower_tf"]) if "lower_tf" in cases else pd.DataFrame(),
            ["timeframe", "lower_tf", "lower_final_cycle_type", "lower_final_cycle_same_as_event"],
        ),
    }
    contrasts = contrast_true_vs_noise(cases, min_cases=min_cases)

    if include_cases and not cases.empty:
        cases.to_csv(out / "candle_reversal_cases.csv", index=False, encoding="utf-8-sig")
    for name, frame in summaries.items():
        if not frame.empty:
            frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    if not contrasts.empty:
        contrasts.to_csv(out / "true_vs_noise_feature_contrasts.csv", index=False, encoding="utf-8-sig")

    meta = {
        "timeframes": list(timeframes),
        "horizons": list(horizons),
        "start_window_candles": int(start_window),
        "min_cases_for_contrast": int(min_cases),
        "candidate_count": int(len(cases)),
        "output_dir": str(out),
        "label_rule": "candidate is true_new_cycle_start when it occurs within start_window candles before an opposite next cycle",
    }
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(build_report(cases=cases, summaries=summaries, contrasts=contrasts, meta=meta), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify opposite-direction candles by PPO position as continuation noise or true new cycle starts.")
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), choices=list(TIMEFRAMES))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--start-window", type=int, default=DEFAULT_START_WINDOW)
    parser.add_argument("--min-cases", type=int, default=DEFAULT_MIN_CASES)
    parser.add_argument("--no-cases", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            run(
                timeframes=tuple(args.timeframes),
                horizons=tuple(args.horizons),
                start_window=args.start_window,
                min_cases=args.min_cases,
                include_cases=not args.no_cases,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
