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
    cycle_candles,
    feature_contrast,
    load_cycles,
    progress_bucket,
    quantile_bucket,
    summarize_numeric,
    to_float,
    to_timestamp,
)
from src.common.paths import PROJECT_PATHS


DEFAULT_PAIRS = ("5m->15m", "15m->1h", "1h->4h", "4h->1d", "1d->1w")
DEFAULT_TRANSITION_WINDOW = 5
DEFAULT_HOLD_CANDLES = (1, 2, 3)
DEFAULT_MIN_CASES = 30


def prior_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_noise_vs_true_rebound_analysis"


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_rebound_entry_risk_analysis"


def _round(value: Any, digits: int = 6) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return value


def signed_return(entry: float, exit_: float, sign: float) -> float:
    if pd.isna(entry) or pd.isna(exit_) or entry == 0 or pd.isna(sign):
        return np.nan
    return (exit_ / entry - 1.0) * 100.0 * sign


def load_noise_rebound_cases(pairs: tuple[str, ...]) -> pd.DataFrame:
    path = prior_dir() / "noise_rebound_cases.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing noise rebound cases: {path}")
    df = pd.read_csv(path)
    df = df[df["pair"].isin(pairs)].copy()
    try:
        df["upper_candle_timestamp"] = pd.to_datetime(df["upper_candle_timestamp"], errors="coerce", format="mixed")
    except TypeError:
        df["upper_candle_timestamp"] = pd.to_datetime(df["upper_candle_timestamp"], errors="coerce")
    return df.dropna(subset=["upper_candle_timestamp"]).reset_index(drop=True)


def build_upper_lookup(timeframes: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for timeframe in timeframes:
        candles = build_candle_frame(load_cycles(timeframe), timeframe)
        if candles.empty:
            continue
        keep = [
            "timeframe",
            "cycle_uid",
            "timestamp",
            "candle_index",
            "close",
            "ppo",
            "ppo_hist",
            "ppo_delta",
            "ppo_hist_delta",
            "price_delta_pct",
            "parent_state",
        ]
        frame = candles[keep].copy()
        frame["cycle_candle_count"] = frame.groupby("cycle_uid")["candle_index"].transform("max")
        frame["remaining_candles_to_cycle_end"] = frame["cycle_candle_count"] - frame["candle_index"]
        frames.append(
            frame.rename(
                columns={
                    "timeframe": "upper_tf",
                    "cycle_uid": "upper_cycle_uid",
                    "timestamp": "upper_candle_timestamp",
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_extended_forward_metrics(cases: pd.DataFrame, transition_window: int) -> pd.DataFrame:
    upper_tfs = tuple(sorted(set(cases["upper_tf"]).intersection(TIMEFRAMES), key=TIMEFRAMES.index))
    lookup = build_upper_lookup(upper_tfs)
    if lookup.empty:
        return cases

    merged = cases.merge(
        lookup,
        on=["upper_tf", "upper_cycle_uid", "upper_candle_timestamp"],
        how="left",
        suffixes=("", "_upper_lookup"),
    )
    rows: list[dict[str, Any]] = []
    lookup = lookup.sort_values(["upper_tf", "upper_cycle_uid", "candle_index"]).reset_index(drop=True)
    lookup_groups = {(tf, uid): group.reset_index(drop=True) for (tf, uid), group in lookup.groupby(["upper_tf", "upper_cycle_uid"], sort=False)}

    for _, row in merged.iterrows():
        group = lookup_groups.get((row["upper_tf"], row["upper_cycle_uid"]))
        item = row.to_dict()
        if group is None or pd.isna(row.get("candle_index")):
            rows.append(item)
            continue

        idx_matches = group.index[group["candle_index"] == row["candle_index"]].tolist()
        if not idx_matches:
            rows.append(item)
            continue

        idx = idx_matches[0]
        future = group.iloc[idx + 1 : idx + 1 + transition_window].copy()
        entry = to_float(row.get("close"))
        sign = to_float(row.get("upper_cycle_sign"))
        if future.empty or pd.isna(entry) or entry == 0 or pd.isna(sign):
            rows.append(item)
            continue

        future_closes = pd.to_numeric(future["close"], errors="coerce")
        opposite_returns = (future_closes / entry - 1.0) * 100.0 * (-sign)
        parent_returns = (future_closes / entry - 1.0) * 100.0 * sign
        future_hist_signs = np.sign(pd.to_numeric(future["ppo_hist_delta"], errors="coerce").fillna(0.0).to_numpy())

        item["transition_window_candles"] = transition_window
        item["transition_within_window"] = bool(row.get("remaining_candles_to_cycle_end", np.nan) <= transition_window)
        item["max_opposite_return_ext_pct"] = float(opposite_returns.max(skipna=True))
        item["max_parent_return_ext_pct"] = float(parent_returns.max(skipna=True))
        item["min_countertrend_return_ext_pct"] = float(opposite_returns.min(skipna=True))
        item["opposite_followthrough_ext_count"] = int(np.sum(future_hist_signs == -sign))
        item["parent_recovery_ext_count"] = int(np.sum(future_hist_signs == sign))
        rows.append(item)

    return pd.DataFrame(rows)


def refined_rebound_label(row: pd.Series) -> str:
    base = str(row.get("rebound_label", ""))
    if base != "true_rebound":
        return base

    threshold = to_float(row.get("rebound_threshold_pct"))
    max_opp = to_float(row.get("max_opposite_return_ext_pct", row.get("max_opposite_return_fwd_pct")))
    max_parent = to_float(row.get("max_parent_return_ext_pct", row.get("max_parent_return_fwd_pct")))
    opp_count = to_float(row.get("opposite_followthrough_ext_count", row.get("opposite_followthrough_count")))
    parent_count = to_float(row.get("parent_recovery_ext_count", row.get("parent_recovery_count")))

    if bool(row.get("transition_within_window")):
        return "reversal_candidate"
    if not pd.isna(max_parent) and not pd.isna(max_opp) and max_parent > max_opp * 1.15 and parent_count >= opp_count:
        return "trap_rebound"
    if not pd.isna(threshold) and not pd.isna(max_opp) and (max_opp < threshold * 1.5 or opp_count <= 1):
        return "weak_rebound"
    return "clean_rebound"


def refine_rebounds(cases: pd.DataFrame, transition_window: int, min_cases: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    refined = add_extended_forward_metrics(cases, transition_window=transition_window)
    refined["refined_rebound_label"] = refined.apply(refined_rebound_label, axis=1)

    metrics = [
        "max_opposite_return_fwd_pct",
        "max_parent_return_fwd_pct",
        "max_opposite_return_ext_pct",
        "max_parent_return_ext_pct",
        "min_countertrend_return_ext_pct",
        "opposite_followthrough_ext_count",
        "parent_recovery_ext_count",
        "upper_cycle_progress",
        "remaining_candles_to_cycle_end",
        "prev_noise_streak",
        "ppo",
        "ppo_hist",
        "ppo_delta",
        "ppo_hist_delta",
        "lower_same_direction_ratio",
        "lower_opposite_direction_ratio",
        "lower_same_price_ratio",
        "lower_opposite_price_ratio",
        "lower_final_alignment",
        "lower_recovery_ratio_shift",
        "lag1_ppo_hist",
        "lag1_ppo_delta",
        "lag1_lower_final_alignment",
    ]
    summary = summarize_numeric(
        refined,
        ["pair", "upper_direction_label", "refined_rebound_label", "upper_progress_bucket"],
        [metric for metric in metrics if metric in refined.columns],
    )

    contrast_frames: list[pd.DataFrame] = []
    clean = refined[refined["refined_rebound_label"].isin(["clean_rebound", "weak_rebound", "trap_rebound", "reversal_candidate", "continuation_noise"])].copy()
    for pair, group in clean.groupby("pair", sort=False):
        for target in ["clean_rebound", "reversal_candidate", "trap_rebound"]:
            if target not in set(group["refined_rebound_label"]):
                continue
            contrast = feature_contrast(
                group,
                pair_label=str(pair),
                analysis_name=f"{target}_vs_continuation_noise",
                mask_a=group["refined_rebound_label"] == target,
                mask_b=group["refined_rebound_label"] == "continuation_noise",
                features=[metric for metric in metrics if metric in group.columns],
                min_cases=min_cases,
            )
            if not contrast.empty:
                contrast_frames.append(contrast)
    contrasts = pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame()
    return refined, summary, contrasts


def cycle_entry_cases(timeframes: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timeframe in timeframes:
        cycles = load_cycles(timeframe)
        previous_return = np.nan
        previous_duration = np.nan
        for _, cycle in cycles.iterrows():
            candles = cycle_candles(cycle.get("candle_data"))
            if len(candles) < 2:
                continue
            sign = to_float(cycle.get("cycle_sign"))
            entry = to_float(candles[0].get("close"))
            exit_ = to_float(candles[-1].get("close"))
            if pd.isna(sign) or pd.isna(entry) or entry == 0 or pd.isna(exit_):
                continue

            closes = pd.Series([to_float(candle.get("close")) for candle in candles], dtype="float64")
            ppo = pd.Series([to_float(candle.get("ppo")) for candle in candles], dtype="float64")
            ppo_hist = pd.Series([to_float(candle.get("ppo_hist")) for candle in candles], dtype="float64")
            ppo_hist_delta = ppo_hist.diff()
            signed_path = (closes / entry - 1.0) * 100.0 * sign
            noise_mask = np.sign(ppo_hist_delta.fillna(0.0)) == -sign
            noise_positions = np.where(noise_mask.to_numpy())[0]
            first_noise_idx = int(noise_positions[0]) if len(noise_positions) else -1
            first_noise_progress = first_noise_idx / max(len(candles) - 1, 1) if first_noise_idx >= 0 else np.nan

            final_return = signed_return(entry, exit_, sign)
            row = {
                "timeframe": timeframe,
                "cycle_uid": cycle.get("cycle_uid"),
                "cycle_id": cycle.get("cycle_id"),
                "cycle_type": cycle.get("cycle_type"),
                "cycle_sign": sign,
                "start_date": cycle.get("start_date"),
                "end_date": cycle.get("end_date"),
                "duration_candles": len(candles),
                "entry_after_first_close": entry,
                "exit_cycle_end_close": exit_,
                "entry_to_cycle_end_signed_pct": final_return,
                "cycle_entry_loser": bool(final_return < 0),
                "mfe_signed_pct": float(signed_path.max(skipna=True)),
                "mae_signed_pct": float(signed_path.min(skipna=True)),
                "entry_ppo": float(ppo.iloc[0]) if not pd.isna(ppo.iloc[0]) else np.nan,
                "entry_ppo_hist": float(ppo_hist.iloc[0]) if not pd.isna(ppo_hist.iloc[0]) else np.nan,
                "entry_ppo_hist_aligned": bool(np.sign(ppo_hist.iloc[0]) == sign) if not pd.isna(ppo_hist.iloc[0]) and ppo_hist.iloc[0] != 0 else np.nan,
                "second_candle_ppo_hist_delta": float(ppo_hist_delta.iloc[1]) if len(ppo_hist_delta) > 1 and not pd.isna(ppo_hist_delta.iloc[1]) else np.nan,
                "second_candle_aligned": bool(np.sign(ppo_hist_delta.iloc[1]) == sign) if len(ppo_hist_delta) > 1 and not pd.isna(ppo_hist_delta.iloc[1]) and ppo_hist_delta.iloc[1] != 0 else np.nan,
                "noise_candle_count": int(noise_mask.sum()),
                "noise_ratio": float(noise_mask.sum() / max(len(candles) - 1, 1)),
                "first_noise_candle_index": first_noise_idx + 1 if first_noise_idx >= 0 else np.nan,
                "first_noise_progress": first_noise_progress,
                "early_noise_within_3": bool(first_noise_idx in (1, 2, 3)),
                "prev_cycle_return_signed_pct": previous_return,
                "prev_cycle_duration_candles": previous_duration,
            }
            rows.append(row)
            previous_return = final_return
            previous_duration = len(candles)

    cases = pd.DataFrame(rows)
    if cases.empty:
        return cases
    cases["entry_ppo_quantile"] = quantile_bucket(cases["entry_ppo"], "entry_ppo")
    cases["entry_ppo_hist_quantile"] = quantile_bucket(cases["entry_ppo_hist"], "entry_hist")
    cases["first_noise_progress_bucket"] = progress_bucket(cases["first_noise_progress"])
    cases["duration_bucket"] = pd.cut(
        pd.to_numeric(cases["duration_candles"], errors="coerce"),
        bins=[0, 3, 5, 10, 20, 50, 10**9],
        labels=["dur_2_3", "dur_4_5", "dur_6_10", "dur_11_20", "dur_21_50", "dur_51_plus"],
        include_lowest=True,
    )
    return cases


def build_cycle_entry_summary(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "entry_to_cycle_end_signed_pct",
        "mfe_signed_pct",
        "mae_signed_pct",
        "duration_candles",
        "entry_ppo",
        "entry_ppo_hist",
        "second_candle_ppo_hist_delta",
        "noise_candle_count",
        "noise_ratio",
        "first_noise_progress",
        "prev_cycle_return_signed_pct",
    ]
    summary = summarize_numeric(
        cases,
        ["timeframe", "cycle_type", "cycle_entry_loser", "early_noise_within_3", "entry_ppo_hist_aligned"],
        metrics,
    )

    contrast_frames: list[pd.DataFrame] = []
    for timeframe, group in cases.groupby("timeframe", sort=False):
        contrast = feature_contrast(
            group,
            pair_label=str(timeframe),
            analysis_name="losing_cycle_after_first_candle_entry",
            mask_a=group["cycle_entry_loser"] == True,
            mask_b=group["cycle_entry_loser"] == False,
            features=[metric for metric in metrics if metric in group.columns],
            min_cases=DEFAULT_MIN_CASES,
        )
        if not contrast.empty:
            contrast_frames.append(contrast)
    contrasts = pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame()
    return summary, contrasts


def build_noise_trade_summary(refined: pd.DataFrame) -> pd.DataFrame:
    tradeable = refined[refined["refined_rebound_label"].isin(["weak_rebound", "clean_rebound", "reversal_candidate", "trap_rebound", "continuation_noise"])].copy()
    metrics = [
        "max_opposite_return_fwd_pct",
        "max_parent_return_fwd_pct",
        "max_opposite_return_ext_pct",
        "max_parent_return_ext_pct",
        "min_countertrend_return_ext_pct",
        "opposite_followthrough_ext_count",
        "parent_recovery_ext_count",
        "upper_cycle_progress",
        "remaining_candles_to_cycle_end",
    ]
    return summarize_numeric(
        tradeable,
        ["pair", "upper_direction_label", "refined_rebound_label", "upper_progress_bucket"],
        [metric for metric in metrics if metric in tradeable.columns],
    )


def build_report(meta: dict[str, Any], refined_summary: pd.DataFrame, entry_summary: pd.DataFrame, entry_contrasts: pd.DataFrame) -> str:
    lines = [
        "# PPO Rebound And Cycle Entry Risk Analysis",
        "",
        "## Metadata",
        "",
        f"- pairs: `{', '.join(meta['pairs'])}`",
        f"- transition_window_candles: `{meta['transition_window_candles']}`",
        f"- refined_noise_case_count: `{meta['refined_noise_case_count']}`",
        f"- cycle_entry_case_count: `{meta['cycle_entry_case_count']}`",
        f"- output_dir: `{meta['output_dir']}`",
        "",
        "## Rebound Label Counts",
        "",
        pd.DataFrame(meta["refined_label_counts"]).to_markdown(index=False),
        "",
        "## Entry Loss Snapshot",
        "",
    ]
    view_cols = [
        "timeframe",
        "cycle_type",
        "cycle_entry_loser",
        "early_noise_within_3",
        "entry_ppo_hist_aligned",
        "count",
        "entry_to_cycle_end_signed_pct_avg",
        "mfe_signed_pct_avg",
        "mae_signed_pct_avg",
        "noise_ratio_avg",
        "first_noise_progress_avg",
    ]
    if not entry_summary.empty:
        lines.append(entry_summary[[col for col in view_cols if col in entry_summary.columns]].head(50).to_markdown(index=False))
        lines.append("")

    if not entry_contrasts.empty:
        lines.extend(["## Losing Entry Contrasts", ""])
        lines.append(entry_contrasts.groupby(["pair", "analysis"], group_keys=False).head(8).to_markdown(index=False))
        lines.append("")

    if not refined_summary.empty:
        lines.extend(["## Refined Noise/Trade Snapshot", ""])
        cols = [
            "pair",
            "upper_direction_label",
            "refined_rebound_label",
            "upper_progress_bucket",
            "count",
            "max_opposite_return_ext_pct_avg",
            "max_parent_return_ext_pct_avg",
            "min_countertrend_return_ext_pct_avg",
            "remaining_candles_to_cycle_end_avg",
        ]
        lines.append(refined_summary[[col for col in cols if col in refined_summary.columns]].head(60).to_markdown(index=False))
        lines.append("")

    lines.extend(
        [
            "## Interpretation Guardrails",
            "",
            "- `reversal_candidate` means the current parent cycle transition is near in candle-count terms; it is not proof of a profitable reversal entry by itself.",
            "- `trap_rebound` is the case to avoid for counter-trend trades because parent-direction recovery dominates the rebound window.",
            "- Cycle-entry results assume entry at the first cycle candle close and exit at cycle end close.",
        ]
    )
    return "\n".join(lines)


def run(
    pairs: tuple[str, ...] = DEFAULT_PAIRS,
    timeframes: tuple[str, ...] = TIMEFRAMES,
    transition_window: int = DEFAULT_TRANSITION_WINDOW,
    min_cases: int = DEFAULT_MIN_CASES,
) -> dict[str, Any]:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    noise_cases = load_noise_rebound_cases(pairs)
    refined, refined_summary, refined_contrasts = refine_rebounds(
        noise_cases,
        transition_window=transition_window,
        min_cases=min_cases,
    )
    entry_cases = cycle_entry_cases(timeframes)
    entry_summary, entry_contrasts = build_cycle_entry_summary(entry_cases)
    noise_trade_summary = build_noise_trade_summary(refined)

    refined.to_csv(out / "refined_rebound_cases.csv", index=False, encoding="utf-8-sig")
    refined_summary.to_csv(out / "refined_rebound_summary.csv", index=False, encoding="utf-8-sig")
    refined_contrasts.to_csv(out / "refined_rebound_feature_contrasts.csv", index=False, encoding="utf-8-sig")
    entry_cases.to_csv(out / "cycle_first_candle_entry_cases.csv", index=False, encoding="utf-8-sig")
    entry_summary.to_csv(out / "cycle_first_candle_entry_summary.csv", index=False, encoding="utf-8-sig")
    entry_contrasts.to_csv(out / "cycle_first_candle_entry_loss_contrasts.csv", index=False, encoding="utf-8-sig")
    noise_trade_summary.to_csv(out / "noise_countertrend_trade_summary.csv", index=False, encoding="utf-8-sig")

    label_counts = (
        refined.groupby(["pair", "upper_direction_label", "refined_rebound_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )
    meta = {
        "pairs": list(pairs),
        "timeframes": list(timeframes),
        "transition_window_candles": int(transition_window),
        "refined_noise_case_count": int(len(refined)),
        "cycle_entry_case_count": int(len(entry_cases)),
        "output_dir": str(out),
        "refined_label_counts": label_counts,
    }
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(
        build_report(meta, noise_trade_summary, entry_summary, entry_contrasts),
        encoding="utf-8",
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine PPO rebound labels and analyze first-candle cycle entry risk.")
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--timeframes", nargs="+", default=list(TIMEFRAMES), choices=list(TIMEFRAMES))
    parser.add_argument("--transition-window", type=int, default=DEFAULT_TRANSITION_WINDOW)
    parser.add_argument("--min-cases", type=int, default=DEFAULT_MIN_CASES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            run(
                pairs=tuple(args.pairs),
                timeframes=tuple(args.timeframes),
                transition_window=args.transition_window,
                min_cases=args.min_cases,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
