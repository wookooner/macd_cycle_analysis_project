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
    TIMEFRAMES,
    build_candle_frame,
    feature_contrast,
    load_cycles,
    progress_bucket,
    summarize_numeric,
)
from src.common.paths import PROJECT_PATHS


DEFAULT_PAIRS = ("5m->15m", "15m->1h", "1h->4h", "4h->1d", "1d->1w")
DEFAULT_LOOKAHEAD = 3
DEFAULT_MIN_CASES = 40
REBOUND_THRESHOLD_MULTIPLIER = 0.75
MIN_REBOUND_THRESHOLD_PCT = 0.05


def source_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_multitimeframe_influence_analysis"


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_noise_vs_true_rebound_analysis"


def _round(value: Any, digits: int = 6) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return value


def load_parent_cases(pairs: tuple[str, ...]) -> pd.DataFrame:
    path = source_dir() / "parent_candle_cases.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing parent candle cases: {path}")

    usecols = [
        "pair",
        "lower_tf",
        "upper_tf",
        "upper_cycle_uid",
        "upper_cycle_id",
        "upper_cycle_type",
        "upper_cycle_sign",
        "upper_candle_timestamp",
        "upper_candle_state",
        "upper_next_state",
        "upper_cycle_progress",
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
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["pair"].isin(pairs)].copy()
    try:
        df["upper_candle_timestamp"] = pd.to_datetime(df["upper_candle_timestamp"], errors="coerce", format="mixed")
    except TypeError:
        df["upper_candle_timestamp"] = pd.to_datetime(df["upper_candle_timestamp"], errors="coerce")
    df["upper_cycle_sign"] = pd.to_numeric(df["upper_cycle_sign"], errors="coerce")
    df = df.dropna(subset=["upper_candle_timestamp", "upper_cycle_sign"])
    return df.sort_values(["pair", "upper_cycle_uid", "upper_candle_timestamp"]).reset_index(drop=True)


def build_upper_candle_lookup(timeframes: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for timeframe in timeframes:
        candles = build_candle_frame(load_cycles(timeframe), timeframe)
        if candles.empty:
            continue
        keep = [
            "timeframe",
            "cycle_uid",
            "timestamp",
            "close",
            "ppo",
            "ppo_hist",
            "ppo_delta",
            "ppo_hist_delta",
            "price_delta_pct",
            "prev_noise_streak",
        ]
        frames.append(candles[keep].rename(columns={"timeframe": "upper_tf", "cycle_uid": "upper_cycle_uid", "timestamp": "upper_candle_timestamp"}))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_upper_current_values(cases: pd.DataFrame) -> pd.DataFrame:
    upper_tfs = tuple(sorted(set(cases["upper_tf"]).intersection(TIMEFRAMES), key=TIMEFRAMES.index))
    lookup = build_upper_candle_lookup(upper_tfs)
    if lookup.empty:
        raise ValueError("failed to build upper candle lookup")

    merged = cases.merge(
        lookup,
        on=["upper_tf", "upper_cycle_uid", "upper_candle_timestamp"],
        how="left",
    )
    return merged.dropna(subset=["close"]).reset_index(drop=True)


def threshold_by_upper_tf(cases: pd.DataFrame) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for upper_tf, group in cases.groupby("upper_tf"):
        abs_move = pd.to_numeric(group["price_delta_pct"], errors="coerce").abs().dropna()
        if abs_move.empty:
            thresholds[str(upper_tf)] = MIN_REBOUND_THRESHOLD_PCT
            continue
        thresholds[str(upper_tf)] = max(
            MIN_REBOUND_THRESHOLD_PCT,
            float(abs_move.median()) * REBOUND_THRESHOLD_MULTIPLIER,
        )
    return thresholds


def add_noise_labels(cases: pd.DataFrame, lookahead: int) -> pd.DataFrame:
    cases = cases.sort_values(["pair", "upper_cycle_uid", "upper_candle_timestamp"]).reset_index(drop=True)
    thresholds = threshold_by_upper_tf(cases)
    rows: list[dict[str, Any]] = []

    for _, group in cases.groupby(["pair", "upper_cycle_uid"], sort=False):
        group = group.reset_index(drop=True)
        closes = pd.to_numeric(group["close"], errors="coerce")
        signs = pd.to_numeric(group["upper_cycle_sign"], errors="coerce")
        ppo_hist_delta_sign = np.sign(pd.to_numeric(group["ppo_hist_delta"], errors="coerce").fillna(0.0).to_numpy())
        states = group["upper_candle_state"].astype(str).to_numpy()

        for idx, item in group.iterrows():
            if item["upper_candle_state"] != "noise":
                continue

            entry_close = closes.iloc[idx]
            parent_sign = signs.iloc[idx]
            if pd.isna(entry_close) or entry_close == 0 or pd.isna(parent_sign):
                continue

            future = group.iloc[idx + 1 : idx + 1 + lookahead].copy()
            if future.empty:
                continue

            future_closes = pd.to_numeric(future["close"], errors="coerce")
            opposite_returns = (future_closes / entry_close - 1.0) * 100.0 * (-parent_sign)
            parent_returns = (future_closes / entry_close - 1.0) * 100.0 * parent_sign
            max_opposite_return = float(opposite_returns.max(skipna=True))
            max_parent_return = float(parent_returns.max(skipna=True))

            future_delta_signs = ppo_hist_delta_sign[idx + 1 : idx + 1 + lookahead]
            future_states = states[idx + 1 : idx + 1 + lookahead]
            opposite_followthrough_count = int(np.sum(future_delta_signs == -parent_sign))
            parent_recovery_count = int(np.sum(future_delta_signs == parent_sign))
            next_state = future_states[0] if len(future_states) else None
            threshold = thresholds.get(str(item["upper_tf"]), MIN_REBOUND_THRESHOLD_PCT)

            true_rebound = (
                max_opposite_return >= threshold
                and opposite_followthrough_count >= 1
                and max_opposite_return >= max_parent_return
            )
            continuation_noise = (
                max_opposite_return < threshold
                or (parent_recovery_count > opposite_followthrough_count and max_parent_return > max_opposite_return)
            )
            ambiguous = not true_rebound and not continuation_noise

            row = item.to_dict()
            row.update(
                {
                    "lookahead_candles": lookahead,
                    "rebound_threshold_pct": threshold,
                    "max_opposite_return_fwd_pct": max_opposite_return,
                    "max_parent_return_fwd_pct": max_parent_return,
                    "opposite_followthrough_count": opposite_followthrough_count,
                    "parent_recovery_count": parent_recovery_count,
                    "next_state_after_noise": next_state,
                    "true_rebound": bool(true_rebound),
                    "continuation_noise": bool(continuation_noise),
                    "ambiguous_noise": bool(ambiguous),
                }
            )
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["upper_progress_bucket"] = progress_bucket(result["upper_cycle_progress"])
    result["upper_direction_label"] = np.where(result["upper_cycle_sign"] > 0, "parent_up", "parent_down")
    result["rebound_label"] = np.select(
        [result["true_rebound"], result["continuation_noise"]],
        ["true_rebound", "continuation_noise"],
        default="ambiguous",
    )
    return result


def add_lag_features(cases: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    cases = cases.sort_values(["pair", "upper_cycle_uid", "upper_candle_timestamp"]).reset_index(drop=True)
    lag_cols = [
        "lower_same_direction_ratio",
        "lower_opposite_direction_ratio",
        "lower_same_price_ratio",
        "lower_opposite_price_ratio",
        "lower_recovery_ratio_shift",
        "lower_final_alignment",
        "lower_max_opposite_hist_streak",
        "ppo",
        "ppo_hist",
        "ppo_delta",
        "price_delta_pct",
        "prev_noise_streak",
    ]
    for lag in range(1, max_lag + 1):
        shifted = cases.groupby(["pair", "upper_cycle_uid"], sort=False)[lag_cols].shift(lag)
        shifted = shifted.rename(columns={col: f"lag{lag}_{col}" for col in shifted.columns})
        cases = pd.concat([cases, shifted], axis=1)
    return cases


def contrast_features() -> list[str]:
    return [
        "upper_cycle_progress",
        "prev_noise_streak",
        "ppo",
        "ppo_hist",
        "ppo_delta",
        "ppo_hist_delta",
        "price_delta_pct",
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
        "lag1_lower_same_direction_ratio",
        "lag1_lower_opposite_direction_ratio",
        "lag1_lower_same_price_ratio",
        "lag1_lower_opposite_price_ratio",
        "lag1_lower_recovery_ratio_shift",
        "lag1_lower_final_alignment",
        "lag1_lower_max_opposite_hist_streak",
        "lag1_ppo_hist",
        "lag1_ppo_delta",
        "lag1_price_delta_pct",
        "lag1_prev_noise_streak",
        "lag2_lower_same_direction_ratio",
        "lag2_lower_opposite_direction_ratio",
        "lag2_lower_recovery_ratio_shift",
        "lag2_ppo_hist",
        "lag2_price_delta_pct",
    ]


def build_contrasts(cases: pd.DataFrame, min_cases: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    features = contrast_features()
    clean = cases[cases["rebound_label"].isin(["true_rebound", "continuation_noise"])].copy()

    for pair, group in clean.groupby("pair", sort=False):
        contrast = feature_contrast(
            group,
            pair_label=str(pair),
            analysis_name="true_rebound_vs_continuation_noise",
            mask_a=group["rebound_label"] == "true_rebound",
            mask_b=group["rebound_label"] == "continuation_noise",
            features=[feature for feature in features if feature in group.columns],
            min_cases=min_cases,
        )
        if not contrast.empty:
            frames.append(contrast)

    for (pair, direction), group in clean.groupby(["pair", "upper_direction_label"], sort=False):
        contrast = feature_contrast(
            group,
            pair_label=f"{pair}|{direction}",
            analysis_name="direction_split_true_rebound_vs_noise",
            mask_a=group["rebound_label"] == "true_rebound",
            mask_b=group["rebound_label"] == "continuation_noise",
            features=[feature for feature in features if feature in group.columns],
            min_cases=min_cases,
        )
        if not contrast.empty:
            frames.append(contrast)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_summary(cases: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "max_opposite_return_fwd_pct",
        "max_parent_return_fwd_pct",
        "opposite_followthrough_count",
        "parent_recovery_count",
        "upper_cycle_progress",
        "prev_noise_streak",
        "ppo_hist",
        "ppo_hist_delta",
        "lower_same_direction_ratio",
        "lower_opposite_direction_ratio",
        "lower_final_alignment",
        "lower_recovery_ratio_shift",
        "lower_max_opposite_hist_streak",
        "lag1_lower_same_direction_ratio",
        "lag1_lower_opposite_direction_ratio",
        "lag1_lower_final_alignment",
    ]
    return summarize_numeric(
        cases,
        ["pair", "upper_direction_label", "rebound_label", "upper_progress_bucket"],
        [metric for metric in metrics if metric in cases.columns],
    )


def build_report(meta: dict[str, Any], summary: pd.DataFrame, contrasts: pd.DataFrame) -> str:
    lines = [
        "# PPO Noise vs True Rebound Analysis",
        "",
        "## Rule",
        "",
        "- Candidate events are higher-timeframe noise candles only.",
        "- `true_rebound` means the next candles move meaningfully against the parent-cycle direction and PPO histogram keeps at least one follow-through move against the parent direction.",
        "- `continuation_noise` means the noise does not clear the adaptive price threshold or the parent direction recovers more strongly.",
        "- Thresholds are adaptive by upper timeframe: max(0.05%, 0.75 x median absolute upper candle move).",
        "- Current parent `ppo_hist_delta` is kept only as event magnitude, not as a noise classifier, because all rows are already noise candidates.",
        "",
        "## Metadata",
        "",
        f"- pairs: `{', '.join(meta['pairs'])}`",
        f"- lookahead_candles: `{meta['lookahead_candles']}`",
        f"- case_count: `{meta['case_count']}`",
        f"- true_rebound_count: `{meta['true_rebound_count']}`",
        f"- continuation_noise_count: `{meta['continuation_noise_count']}`",
        f"- ambiguous_count: `{meta['ambiguous_count']}`",
        f"- output_dir: `{meta['output_dir']}`",
        "",
    ]

    if not contrasts.empty:
        lines.extend(["## Strongest Differences", ""])
        top = contrasts.groupby(["pair", "analysis"], group_keys=False).head(8)
        lines.append(top.to_markdown(index=False))
        lines.append("")

    if not summary.empty:
        lines.extend(["## Label Summary", ""])
        view_cols = [
            "pair",
            "upper_direction_label",
            "rebound_label",
            "upper_progress_bucket",
            "count",
            "max_opposite_return_fwd_pct_avg",
            "opposite_followthrough_count_avg",
            "parent_recovery_count_avg",
            "lower_same_direction_ratio_avg",
            "lower_opposite_direction_ratio_avg",
            "lag1_lower_same_direction_ratio_avg",
            "lag1_lower_opposite_direction_ratio_avg",
            "lower_final_alignment_avg",
        ]
        view = summary[[col for col in view_cols if col in summary.columns]].head(60)
        lines.append(view.to_markdown(index=False))
        lines.append("")

    lines.extend(
        [
            "## Read Me",
            "",
            "This analysis deliberately separates a parent-cycle noise candle from a meaningful rebound/pullback.",
            "The older `noise_then_rebound_next` field in the first report should be read as a return-to-parent-trend marker, not as a tradable rebound marker.",
        ]
    )
    return "\n".join(lines)


def run(
    pairs: tuple[str, ...] = DEFAULT_PAIRS,
    lookahead: int = DEFAULT_LOOKAHEAD,
    min_cases: int = DEFAULT_MIN_CASES,
) -> dict[str, Any]:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    base_cases = load_parent_cases(pairs)
    enriched = add_upper_current_values(base_cases)
    noise_cases = add_noise_labels(enriched, lookahead=lookahead)
    noise_cases = add_lag_features(noise_cases, max_lag=3)

    summary = build_summary(noise_cases)
    contrasts = build_contrasts(noise_cases, min_cases=min_cases)

    noise_cases.to_csv(out / "noise_rebound_cases.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "noise_rebound_summary.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(out / "noise_rebound_feature_contrasts.csv", index=False, encoding="utf-8-sig")

    meta = {
        "pairs": list(pairs),
        "lookahead_candles": int(lookahead),
        "case_count": int(len(noise_cases)),
        "true_rebound_count": int((noise_cases["rebound_label"] == "true_rebound").sum()),
        "continuation_noise_count": int((noise_cases["rebound_label"] == "continuation_noise").sum()),
        "ambiguous_count": int((noise_cases["rebound_label"] == "ambiguous").sum()),
        "output_dir": str(out),
        "min_cases_for_contrast": int(min_cases),
    }
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(build_report(meta, summary, contrasts), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate temporary PPO noise from meaningful rebound/pullback follow-through.")
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD)
    parser.add_argument("--min-cases", type=int, default=DEFAULT_MIN_CASES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            run(pairs=tuple(args.pairs), lookahead=args.lookahead, min_cases=args.min_cases),
            ensure_ascii=False,
            indent=2,
        )
    )
