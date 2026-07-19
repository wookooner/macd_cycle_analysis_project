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
    build_cycle_interval_frame,
    cycle_candles,
    load_cycles,
    progress_bucket,
    quantile_bucket,
    regime,
    summarize_numeric,
    to_float,
)
from src.common.paths import PROJECT_PATHS


PARENT_TF = "1d"
CHILD_TFS = ("4h", "1h", "15m")
CHAIN_TFS = ("15m", "1h", "4h", "1d")
PROGRESS_LABELS = ("p00_20", "p20_40", "p40_60", "p60_80", "p80_100")


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_1d_hierarchy_path_analysis"


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


def add_cycle_start_end_indicators(cycles: pd.DataFrame) -> pd.DataFrame:
    cycles = cycles.copy()
    start_ppo: list[float] = []
    start_ppo_hist: list[float] = []
    end_ppo: list[float] = []
    end_ppo_hist: list[float] = []
    raw_return: list[float] = []
    noise_count: list[int] = []
    direction_change_count: list[int] = []
    area_ppo_hist: list[float] = []

    for _, cycle in cycles.iterrows():
        candles = cycle_candles(cycle.get("candle_data"))
        if not candles:
            start_ppo.append(np.nan)
            start_ppo_hist.append(np.nan)
            end_ppo.append(np.nan)
            end_ppo_hist.append(np.nan)
            raw_return.append(np.nan)
            noise_count.append(0)
            direction_change_count.append(0)
            area_ppo_hist.append(np.nan)
            continue

        first = candles[0]
        last = candles[-1]
        first_close = to_float(first.get("close"))
        last_close = to_float(last.get("close"))
        ppo_hist = pd.Series([to_float(candle.get("ppo_hist")) for candle in candles], dtype="float64")
        ppo_hist_delta = ppo_hist.diff()
        delta_sign = np.sign(ppo_hist_delta.fillna(0.0))
        non_zero_delta = delta_sign[delta_sign != 0].to_numpy()
        cycle_sign = to_float(cycle.get("cycle_sign"))

        start_ppo.append(to_float(first.get("ppo")))
        start_ppo_hist.append(to_float(first.get("ppo_hist")))
        end_ppo.append(to_float(last.get("ppo")))
        end_ppo_hist.append(to_float(last.get("ppo_hist")))
        raw_return.append((last_close / first_close - 1.0) * 100.0 if first_close and not pd.isna(last_close) else np.nan)
        noise_count.append(int(np.sum(delta_sign.to_numpy() == -cycle_sign)))
        direction_change_count.append(int(np.sum(np.diff(non_zero_delta) != 0)) if len(non_zero_delta) > 1 else 0)
        area_ppo_hist.append(float(ppo_hist.abs().sum(skipna=True)))

    cycles["start_ppo"] = start_ppo
    cycles["start_ppo_hist"] = start_ppo_hist
    cycles["end_ppo"] = end_ppo
    cycles["end_ppo_hist"] = end_ppo_hist
    cycles["raw_return_pct"] = raw_return
    cycles["noise_count_asof_cycle"] = noise_count
    cycles["direction_change_count"] = direction_change_count
    cycles["area_abs_ppo_hist"] = area_ppo_hist
    cycles["ppo_change"] = cycles["end_ppo"] - cycles["start_ppo"]
    cycles["ppo_hist_change"] = cycles["end_ppo_hist"] - cycles["start_ppo_hist"]
    cycles["start_ppo_regime"] = regime(cycles["start_ppo"], cycles["start_ppo_hist"])
    return cycles


def load_enriched_cycles(timeframes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return {timeframe: add_cycle_start_end_indicators(load_cycles(timeframe)) for timeframe in timeframes}


def interval_frame(cycles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    frame = build_cycle_interval_frame(cycles, timeframe)
    add_cols = [
        "start_ppo",
        "start_ppo_hist",
        "end_ppo",
        "end_ppo_hist",
        "raw_return_pct",
        "noise_count_asof_cycle",
        "direction_change_count",
        "area_abs_ppo_hist",
        "ppo_change",
        "ppo_hist_change",
        "start_ppo_regime",
    ]
    for column in add_cols:
        frame[column] = cycles[column].to_numpy()
    return frame


def assign_parent_by_start(child: pd.DataFrame, parent: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if child.empty or parent.empty:
        return pd.DataFrame()

    child = child.copy().reset_index(drop=True)
    parent = parent.copy().sort_values("start_ts").reset_index(drop=True)
    parent = parent.reset_index(names=f"{prefix}row")
    child_ns = child["start_ts"].astype("int64").to_numpy()
    parent_start_ns = parent["start_ts"].astype("int64").to_numpy()
    parent_end_ns = parent["end_exclusive"].astype("int64").to_numpy()
    idx = np.searchsorted(parent_start_ns, child_ns, side="right") - 1
    valid = idx >= 0
    covered = np.zeros(len(child), dtype=bool)
    covered[valid] = child_ns[valid] < parent_end_ns[idx[valid]]
    mapped = child.loc[covered].copy()
    mapped[f"{prefix}row"] = idx[covered]

    parent_cols = [
        f"{prefix}row",
        "cycle_uid",
        "cycle_id",
        "cycle_key",
        "cycle_type",
        "cycle_sign",
        "start_ts",
        "end_ts",
        "end_exclusive",
        "duration_candles",
        "cycle_signed_move_pct",
        "raw_return_pct",
        "start_ppo",
        "start_ppo_hist",
        "end_ppo",
        "end_ppo_hist",
        "start_ppo_regime",
    ]
    parent_view = parent[[column for column in parent_cols if column in parent.columns]].rename(
        columns={column: f"{prefix}{column}" for column in parent_cols if column != f"{prefix}row"}
    )
    return mapped.merge(parent_view, on=f"{prefix}row", how="left")


def child_cycles_inside_1d(intervals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    parent = intervals[PARENT_TF]
    for child_tf in CHILD_TFS:
        child = intervals[child_tf].copy()
        mapped = assign_parent_by_start(child, parent, prefix="parent_")
        if mapped.empty:
            continue

        parent_span = (mapped["parent_end_exclusive"] - mapped["parent_start_ts"]).dt.total_seconds().replace(0, np.nan)
        mapped["child_tf"] = child_tf
        mapped["parent_tf"] = PARENT_TF
        mapped["parent_cycle_progress"] = (
            (mapped["start_ts"] - mapped["parent_start_ts"]).dt.total_seconds() / parent_span
        ).clip(lower=0.0, upper=1.0)
        mapped["parent_progress_bucket"] = progress_bucket(mapped["parent_cycle_progress"])
        mapped["same_direction_as_1d"] = mapped["cycle_sign"] == mapped["parent_cycle_sign"]
        mapped["child_raw_return_vs_1d_pct"] = mapped["raw_return_pct"] * mapped["parent_cycle_sign"]
        mapped["child_start_ppo_vs_1d_sign"] = np.sign(pd.to_numeric(mapped["start_ppo"], errors="coerce")) * mapped["parent_cycle_sign"]
        mapped["child_start_hist_vs_1d_sign"] = np.sign(pd.to_numeric(mapped["start_ppo_hist"], errors="coerce")) * mapped["parent_cycle_sign"]
        mapped["child_ppo_change_vs_1d"] = mapped["ppo_change"] * mapped["parent_cycle_sign"]
        mapped["child_hist_change_vs_1d"] = mapped["ppo_hist_change"] * mapped["parent_cycle_sign"]
        rows.append(mapped)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def daily_cycle_profiles(child_map: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (parent_uid, child_tf), group in child_map.groupby(["parent_cycle_uid", "child_tf"], sort=False):
        group = group.sort_values("start_ts")
        parent_sign = float(group["parent_cycle_sign"].iloc[0])
        same = group[group["same_direction_as_1d"]]
        opposite = group[~group["same_direction_as_1d"]]
        final = group.iloc[-1]
        first_opposite = opposite.iloc[0] if not opposite.empty else None
        first_same = same.iloc[0] if not same.empty else None
        rows.append(
            {
                "parent_cycle_uid": parent_uid,
                "parent_cycle_id": group["parent_cycle_id"].iloc[0],
                "parent_cycle_type": group["parent_cycle_type"].iloc[0],
                "parent_cycle_sign": parent_sign,
                "parent_start_ts": group["parent_start_ts"].iloc[0],
                "parent_end_ts": group["parent_end_ts"].iloc[0],
                "parent_duration_candles": group["parent_duration_candles"].iloc[0],
                "parent_raw_return_pct": group["parent_raw_return_pct"].iloc[0],
                "parent_signed_return_pct": group["parent_cycle_signed_move_pct"].iloc[0],
                "child_tf": child_tf,
                "child_cycle_count": int(len(group)),
                "child_same_dir_count": int(len(same)),
                "child_opposite_dir_count": int(len(opposite)),
                "child_same_dir_ratio": _round(len(same) / len(group)),
                "child_opposite_dir_ratio": _round(len(opposite) / len(group)),
                "child_move_vs_1d_sum_pct": _round(group["child_raw_return_vs_1d_pct"].sum()),
                "child_move_vs_1d_avg_pct": _round(group["child_raw_return_vs_1d_pct"].mean()),
                "child_same_move_vs_1d_sum_pct": _round(same["child_raw_return_vs_1d_pct"].sum()) if not same.empty else 0.0,
                "child_opposite_move_vs_1d_sum_pct": _round(opposite["child_raw_return_vs_1d_pct"].sum()) if not opposite.empty else 0.0,
                "child_first_opposite_progress": _round(first_opposite["parent_cycle_progress"]) if first_opposite is not None else None,
                "child_first_same_progress": _round(first_same["parent_cycle_progress"]) if first_same is not None else None,
                "child_final_same_direction": float(bool(final["same_direction_as_1d"])),
                "child_final_start_ppo": _round(final["start_ppo"]),
                "child_final_start_ppo_hist": _round(final["start_ppo_hist"]),
                "child_final_progress": _round(final["parent_cycle_progress"]),
                "child_avg_start_ppo": _round(group["start_ppo"].mean()),
                "child_avg_start_ppo_hist": _round(group["start_ppo_hist"].mean()),
                "child_avg_abs_area_ppo_hist": _round(group["area_abs_ppo_hist"].mean()),
                "child_noise_count_avg": _round(group["noise_count_asof_cycle"].mean()),
                "child_direction_change_avg": _round(group["direction_change_count"].mean()),
            }
        )
    profiles = pd.DataFrame(rows)
    if profiles.empty:
        return profiles
    profiles["parent_strength_bucket"] = pd.qcut(
        profiles["parent_signed_return_pct"].rank(method="first"),
        q=4,
        labels=["q1_weak", "q2_mid", "q3_strong", "q4_extreme"],
    )
    return profiles


def segment_zone_cases(child_map: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in child_map.groupby(
        ["parent_cycle_uid", "parent_cycle_type", "child_tf", "parent_progress_bucket"],
        dropna=False,
        observed=True,
    ):
        parent_uid, parent_type, child_tf, bucket = keys
        same_ratio = float(group["same_direction_as_1d"].mean())
        move_avg = float(group["child_raw_return_vs_1d_pct"].mean())
        hist_change_avg = float(group["child_hist_change_vs_1d"].mean())
        if same_ratio >= 0.65 and move_avg > 0 and hist_change_avg >= 0:
            zone = "strong_parent_trend"
        elif same_ratio <= 0.35 and move_avg < 0:
            zone = "countertrend_rebound_or_pullback"
        elif hist_change_avg < 0 and move_avg < 0:
            zone = "weakening_retracement"
        else:
            zone = "mixed_transition"
        rows.append(
            {
                "parent_cycle_uid": parent_uid,
                "parent_cycle_type": parent_type,
                "child_tf": child_tf,
                "parent_progress_bucket": bucket,
                "zone_label": zone,
                "child_count": int(len(group)),
                "same_direction_ratio": _round(same_ratio),
                "opposite_direction_ratio": _round(1.0 - same_ratio),
                "child_move_vs_1d_avg_pct": _round(move_avg),
                "child_move_vs_1d_sum_pct": _round(group["child_raw_return_vs_1d_pct"].sum()),
                "child_start_ppo_avg": _round(group["start_ppo"].mean()),
                "child_start_ppo_hist_avg": _round(group["start_ppo_hist"].mean()),
                "child_ppo_change_vs_1d_avg": _round(group["child_ppo_change_vs_1d"].mean()),
                "child_hist_change_vs_1d_avg": _round(hist_change_avg),
                "child_noise_count_avg": _round(group["noise_count_asof_cycle"].mean()),
                "child_direction_change_avg": _round(group["direction_change_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_15m_chain(intervals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = intervals["15m"].copy()
    chain = assign_parent_by_start(base, intervals["1h"], prefix="p1h_")
    chain = assign_parent_by_start(chain, intervals["4h"], prefix="p4h_")
    chain = assign_parent_by_start(chain, intervals["1d"], prefix="p1d_")
    if chain.empty:
        return chain

    span_1d = (chain["p1d_end_exclusive"] - chain["p1d_start_ts"]).dt.total_seconds().replace(0, np.nan)
    chain["p1d_progress"] = ((chain["start_ts"] - chain["p1d_start_ts"]).dt.total_seconds() / span_1d).clip(0.0, 1.0)
    chain["p1d_progress_bucket"] = progress_bucket(chain["p1d_progress"])
    chain["align_15m_to_1d"] = chain["cycle_sign"] == chain["p1d_cycle_sign"]
    chain["align_1h_to_1d"] = chain["p1h_cycle_sign"] == chain["p1d_cycle_sign"]
    chain["align_4h_to_1d"] = chain["p4h_cycle_sign"] == chain["p1d_cycle_sign"]
    chain["chain_alignment_pattern"] = (
        np.where(chain["align_15m_to_1d"], "15S", "15O")
        + "|"
        + np.where(chain["align_1h_to_1d"], "1hS", "1hO")
        + "|"
        + np.where(chain["align_4h_to_1d"], "4hS", "4hO")
    )
    chain["return_vs_1d_pct"] = chain["raw_return_pct"] * chain["p1d_cycle_sign"]
    chain["start_ppo_vs_1d"] = chain["start_ppo"] * chain["p1d_cycle_sign"]
    chain["start_hist_vs_1d"] = chain["start_ppo_hist"] * chain["p1d_cycle_sign"]
    chain["ppo_change_vs_1d"] = chain["ppo_change"] * chain["p1d_cycle_sign"]
    chain["hist_change_vs_1d"] = chain["ppo_hist_change"] * chain["p1d_cycle_sign"]
    chain["start_ppo_quantile"] = quantile_bucket(chain["start_ppo"], "ppo")
    chain["start_hist_quantile"] = quantile_bucket(chain["start_ppo_hist"], "hist")
    return chain


def build_summaries(
    profiles: pd.DataFrame,
    zones: pd.DataFrame,
    chain: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    profile_summary = summarize_numeric(
        profiles,
        ["parent_cycle_type", "parent_strength_bucket", "child_tf"],
        [
            "child_cycle_count",
            "child_same_dir_ratio",
            "child_opposite_dir_ratio",
            "child_move_vs_1d_avg_pct",
            "child_first_opposite_progress",
            "child_final_same_direction",
            "child_avg_start_ppo",
            "child_avg_start_ppo_hist",
        ],
    )
    zone_summary = summarize_numeric(
        zones,
        ["parent_cycle_type", "child_tf", "parent_progress_bucket", "zone_label"],
        [
            "child_count",
            "same_direction_ratio",
            "opposite_direction_ratio",
            "child_move_vs_1d_avg_pct",
            "child_start_ppo_avg",
            "child_start_ppo_hist_avg",
            "child_hist_change_vs_1d_avg",
        ],
    )
    chain_summary = summarize_numeric(
        chain,
        ["p1d_cycle_type", "p1d_progress_bucket", "chain_alignment_pattern", "start_ppo_regime"],
        [
            "return_vs_1d_pct",
            "start_ppo_vs_1d",
            "start_hist_vs_1d",
            "ppo_change_vs_1d",
            "hist_change_vs_1d",
            "noise_count_asof_cycle",
            "direction_change_count",
        ],
    )
    return {
        "daily_profile_summary": profile_summary,
        "segment_zone_summary": zone_summary,
        "chain_alignment_summary": chain_summary,
    }


def build_report(meta: dict[str, Any], summaries: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# PPO 1D Hierarchy Path Analysis",
        "",
        "## Focus",
        "",
        "- Parent container: complete `1d` cycles.",
        "- Child paths: `4h`, `1h`, `15m` cycles inside each `1d` cycle.",
        "- Propagation chain: each `15m` cycle mapped to its containing `1h`, `4h`, and `1d` cycles.",
        "- Returns are also measured in the parent `1d` direction so positive means it helps the daily cycle.",
        "",
        "## Metadata",
        "",
        f"- daily_cycle_count: `{meta['daily_cycle_count']}`",
        f"- child_cycle_case_count: `{meta['child_cycle_case_count']}`",
        f"- chain_15m_case_count: `{meta['chain_15m_case_count']}`",
        f"- output_dir: `{meta['output_dir']}`",
        "",
    ]

    profile = summaries["daily_profile_summary"]
    if not profile.empty:
        lines.extend(["## Daily Profile Summary", ""])
        cols = [
            "parent_cycle_type",
            "parent_strength_bucket",
            "child_tf",
            "count",
            "child_cycle_count_avg",
            "child_same_dir_ratio_avg",
            "child_opposite_dir_ratio_avg",
            "child_move_vs_1d_avg_pct_avg",
            "child_first_opposite_progress_avg",
            "child_final_same_direction_avg",
        ]
        lines.append(profile[[col for col in cols if col in profile.columns]].head(50).to_markdown(index=False))
        lines.append("")

    zones = summaries["segment_zone_summary"]
    if not zones.empty:
        lines.extend(["## Segment Zone Summary", ""])
        cols = [
            "parent_cycle_type",
            "child_tf",
            "parent_progress_bucket",
            "zone_label",
            "count",
            "same_direction_ratio_avg",
            "child_move_vs_1d_avg_pct_avg",
            "child_start_ppo_avg_avg",
            "child_start_ppo_hist_avg_avg",
            "child_hist_change_vs_1d_avg_avg",
        ]
        lines.append(zones[[col for col in cols if col in zones.columns]].head(80).to_markdown(index=False))
        lines.append("")

    chain = summaries["chain_alignment_summary"]
    if not chain.empty:
        lines.extend(["## 15m To 1D Chain Summary", ""])
        cols = [
            "p1d_cycle_type",
            "p1d_progress_bucket",
            "chain_alignment_pattern",
            "start_ppo_regime",
            "count",
            "return_vs_1d_pct_avg",
            "start_ppo_vs_1d_avg",
            "start_hist_vs_1d_avg",
            "ppo_change_vs_1d_avg",
            "hist_change_vs_1d_avg",
        ]
        view = chain[[col for col in cols if col in chain.columns]].sort_values(
            ["p1d_cycle_type", "p1d_progress_bucket", "count"],
            ascending=[True, True, False],
        )
        lines.append(view.head(100).to_markdown(index=False))
        lines.append("")

    lines.extend(
        [
            "## Reading Notes",
            "",
            "- `S` means aligned with the 1D parent direction; `O` means opposite.",
            "- `strong_parent_trend` is a segment where lower cycles mostly align and their PPO/hist movement supports the 1D direction.",
            "- `countertrend_rebound_or_pullback` is the zone where lower cycles mostly oppose the 1D direction and returns also move against it.",
            "- `weakening_retracement` is a mixed zone with negative movement and weakening PPO histogram versus the 1D direction.",
        ]
    )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    cycles = load_enriched_cycles(CHAIN_TFS)
    intervals = {timeframe: interval_frame(cycles[timeframe], timeframe) for timeframe in CHAIN_TFS}
    child_map = child_cycles_inside_1d(intervals)
    profiles = daily_cycle_profiles(child_map)
    zones = segment_zone_cases(child_map)
    chain = build_15m_chain(intervals)
    summaries = build_summaries(profiles, zones, chain)

    child_map.to_csv(out / "child_cycles_inside_1d_cases.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(out / "daily_cycle_profiles.csv", index=False, encoding="utf-8-sig")
    zones.to_csv(out / "daily_cycle_segment_zones.csv", index=False, encoding="utf-8-sig")
    chain.to_csv(out / "chain_15m_1h_4h_1d_cases.csv", index=False, encoding="utf-8-sig")
    for name, frame in summaries.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    meta = {
        "daily_cycle_count": int(len(cycles[PARENT_TF])),
        "child_cycle_case_count": int(len(child_map)),
        "daily_profile_count": int(len(profiles)),
        "segment_zone_count": int(len(zones)),
        "chain_15m_case_count": int(len(chain)),
        "output_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(build_report(meta, summaries), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze complete 1D cycles through 15m-1h-4h hierarchy paths.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))
