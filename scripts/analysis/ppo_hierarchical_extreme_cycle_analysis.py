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

from scripts.analysis.ppo_1d_hierarchy_path_analysis import (  # noqa: E402
    add_cycle_start_end_indicators,
    assign_parent_by_start,
)
from scripts.analysis.ppo_multitimeframe_influence_analysis import (  # noqa: E402
    build_cycle_interval_frame,
    cycle_candles,
    load_cycles,
    progress_bucket,
    regime,
    summarize_numeric,
    to_float,
)
from src.common.paths import PROJECT_PATHS  # noqa: E402


TIMEFRAMES = ("15m", "1h", "4h", "1d")
PARENT_CHILDREN = {
    "1d": ("4h", "1h", "15m"),
    "4h": ("1h", "15m"),
    "1h": ("15m",),
}
EVENT_SPECS = (
    ("max_start_ppo", "start_ppo", "idxmax"),
    ("min_start_ppo", "start_ppo", "idxmin"),
    ("max_start_ppo_hist", "start_ppo_hist", "idxmax"),
    ("min_start_ppo_hist", "start_ppo_hist", "idxmin"),
    ("lowest_price_cycle", "cycle_low", "idxmin"),
    ("highest_price_cycle", "cycle_high", "idxmax"),
)


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_hierarchical_extreme_cycle_analysis"


def _round(value: Any, digits: int = 6) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return value


def _safe_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    try:
        return pd.Timestamp(value)
    except Exception:
        return pd.NaT


def _extreme_candle(candles: list[dict[str, Any]], price_field: str, chooser: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_value = np.inf if chooser == "min" else -np.inf

    for idx, candle in enumerate(candles):
        price = to_float(candle.get(price_field))
        if pd.isna(price):
            continue
        if (chooser == "min" and price < best_value) or (chooser == "max" and price > best_value):
            best_value = price
            best = {
                "idx": idx + 1,
                "timestamp": _safe_timestamp(candle.get("timestamp", candle.get("date"))),
                "price": price,
                "ppo": to_float(candle.get("ppo")),
                "ppo_hist": to_float(candle.get("ppo_hist")),
                "close": to_float(candle.get("close")),
            }

    if best is None:
        return {
            "idx": np.nan,
            "timestamp": pd.NaT,
            "price": np.nan,
            "ppo": np.nan,
            "ppo_hist": np.nan,
            "close": np.nan,
        }
    return best


def add_cycle_price_extremes(cycles: pd.DataFrame) -> pd.DataFrame:
    cycles = cycles.copy()
    rows: list[dict[str, Any]] = []

    for _, cycle in cycles.iterrows():
        candles = cycle_candles(cycle.get("candle_data"))
        low = _extreme_candle(candles, "low", "min")
        high = _extreme_candle(candles, "high", "max")
        rows.append(
            {
                "cycle_low": low["price"],
                "cycle_low_ts": low["timestamp"],
                "cycle_low_candle_index": low["idx"],
                "cycle_low_ppo": low["ppo"],
                "cycle_low_ppo_hist": low["ppo_hist"],
                "cycle_low_close": low["close"],
                "cycle_high": high["price"],
                "cycle_high_ts": high["timestamp"],
                "cycle_high_candle_index": high["idx"],
                "cycle_high_ppo": high["ppo"],
                "cycle_high_ppo_hist": high["ppo_hist"],
                "cycle_high_close": high["close"],
            }
        )

    return pd.concat([cycles.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def load_featured_cycles(timeframes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return {
        timeframe: add_cycle_price_extremes(add_cycle_start_end_indicators(load_cycles(timeframe)))
        for timeframe in timeframes
    }


def interval_frame(cycles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    frame = build_cycle_interval_frame(cycles, timeframe)
    add_cols = [
        "start_ppo",
        "start_ppo_hist",
        "end_ppo",
        "end_ppo_hist",
        "raw_return_pct",
        "ppo_change",
        "ppo_hist_change",
        "start_ppo_regime",
        "cycle_low",
        "cycle_low_ts",
        "cycle_low_candle_index",
        "cycle_low_ppo",
        "cycle_low_ppo_hist",
        "cycle_low_close",
        "cycle_high",
        "cycle_high_ts",
        "cycle_high_candle_index",
        "cycle_high_ppo",
        "cycle_high_ppo_hist",
        "cycle_high_close",
    ]
    for column in add_cols:
        frame[column] = cycles[column].to_numpy()
    return frame


def _parent_progress(selected: pd.Series) -> float:
    span = (selected["parent_end_exclusive"] - selected["parent_start_ts"]).total_seconds()
    if not span:
        return np.nan
    return float(((selected["start_ts"] - selected["parent_start_ts"]).total_seconds() / span))


def _price_event_fields(selected: pd.Series, event_name: str) -> dict[str, Any]:
    if event_name == "lowest_price_cycle":
        prefix = "cycle_low"
    elif event_name == "highest_price_cycle":
        prefix = "cycle_high"
    else:
        return {
            "event_price": np.nan,
            "event_price_ts": pd.NaT,
            "event_price_candle_index": np.nan,
            "event_price_ppo": np.nan,
            "event_price_ppo_hist": np.nan,
            "event_price_close": np.nan,
        }

    return {
        "event_price": selected[prefix],
        "event_price_ts": selected[f"{prefix}_ts"],
        "event_price_candle_index": selected[f"{prefix}_candle_index"],
        "event_price_ppo": selected[f"{prefix}_ppo"],
        "event_price_ppo_hist": selected[f"{prefix}_ppo_hist"],
        "event_price_close": selected[f"{prefix}_close"],
    }


def _select_event(group: pd.DataFrame, metric: str, selector: str) -> pd.Series | None:
    values = pd.to_numeric(group[metric], errors="coerce")
    if values.dropna().empty:
        return None
    idx = values.idxmax() if selector == "idxmax" else values.idxmin()
    return group.loc[idx]


def event_row(group: pd.DataFrame, child_tf: str, parent_tf: str, event_name: str, selected: pd.Series) -> dict[str, Any]:
    progress = _parent_progress(selected)
    same_direction = bool(selected["cycle_sign"] == selected["parent_cycle_sign"])
    parent_sign = to_float(selected["parent_cycle_sign"])
    child_signed_vs_parent = to_float(selected["raw_return_pct"]) * parent_sign

    return {
        "pair": f"{child_tf}->{parent_tf}",
        "parent_tf": parent_tf,
        "child_tf": child_tf,
        "event_name": event_name,
        "parent_cycle_uid": selected["parent_cycle_uid"],
        "parent_cycle_id": selected["parent_cycle_id"],
        "parent_cycle_type": selected["parent_cycle_type"],
        "parent_cycle_sign": parent_sign,
        "parent_start_ts": selected["parent_start_ts"],
        "parent_end_ts": selected["parent_end_ts"],
        "parent_duration_candles": selected["parent_duration_candles"],
        "parent_start_ppo": selected.get("parent_start_ppo"),
        "parent_start_ppo_hist": selected.get("parent_start_ppo_hist"),
        "parent_end_ppo": selected.get("parent_end_ppo"),
        "parent_end_ppo_hist": selected.get("parent_end_ppo_hist"),
        "child_cycle_count_in_parent": int(len(group)),
        "selected_child_order": int((group["start_ts"] < selected["start_ts"]).sum() + 1),
        "selected_child_progress": progress,
        "selected_child_progress_bucket": progress_bucket(pd.Series([progress])).iloc[0],
        "selected_same_direction_as_parent": float(same_direction),
        "selected_child_uid": selected["cycle_uid"],
        "selected_child_id": selected["cycle_id"],
        "selected_child_type": selected["cycle_type"],
        "selected_child_sign": selected["cycle_sign"],
        "selected_child_start_ts": selected["start_ts"],
        "selected_child_end_ts": selected["end_ts"],
        "selected_child_duration_candles": selected["duration_candles"],
        "selected_child_start_ppo": selected["start_ppo"],
        "selected_child_start_ppo_hist": selected["start_ppo_hist"],
        "selected_child_end_ppo": selected["end_ppo"],
        "selected_child_end_ppo_hist": selected["end_ppo_hist"],
        "selected_child_ppo_change": selected["ppo_change"],
        "selected_child_ppo_hist_change": selected["ppo_hist_change"],
        "selected_child_raw_return_pct": selected["raw_return_pct"],
        "selected_child_return_vs_parent_pct": child_signed_vs_parent,
        "selected_child_low": selected["cycle_low"],
        "selected_child_low_ts": selected["cycle_low_ts"],
        "selected_child_low_ppo": selected["cycle_low_ppo"],
        "selected_child_low_ppo_hist": selected["cycle_low_ppo_hist"],
        "selected_child_high": selected["cycle_high"],
        "selected_child_high_ts": selected["cycle_high_ts"],
        "selected_child_high_ppo": selected["cycle_high_ppo"],
        "selected_child_high_ppo_hist": selected["cycle_high_ppo_hist"],
        **_price_event_fields(selected, event_name),
    }


def build_pair_cases(child: pd.DataFrame, parent: pd.DataFrame, child_tf: str, parent_tf: str) -> pd.DataFrame:
    mapped = assign_parent_by_start(child, parent, prefix="parent_")
    if mapped.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, group in mapped.groupby("parent_cycle_uid", sort=False):
        group = group.sort_values("start_ts").copy()
        for event_name, metric, selector in EVENT_SPECS:
            selected = _select_event(group, metric, selector)
            if selected is not None:
                rows.append(event_row(group, child_tf, parent_tf, event_name, selected))

    return pd.DataFrame(rows)


def build_cases(intervals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for parent_tf, child_tfs in PARENT_CHILDREN.items():
        for child_tf in child_tfs:
            cases = build_pair_cases(intervals[child_tf], intervals[parent_tf], child_tf, parent_tf)
            if not cases.empty:
                frames.append(cases)
    if not frames:
        return pd.DataFrame()

    cases = pd.concat(frames, ignore_index=True)
    cases["selected_child_start_regime"] = regime(
        cases["selected_child_start_ppo"],
        cases["selected_child_start_ppo_hist"],
    )
    cases["event_price_regime"] = regime(cases["event_price_ppo"], cases["event_price_ppo_hist"])
    return cases


def build_wide_cases(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()

    id_cols = [
        "pair",
        "parent_tf",
        "child_tf",
        "parent_cycle_uid",
        "parent_cycle_id",
        "parent_cycle_type",
        "parent_start_ts",
        "parent_end_ts",
        "child_cycle_count_in_parent",
    ]
    value_cols = [
        "selected_child_uid",
        "selected_child_type",
        "selected_child_progress",
        "selected_child_start_ppo",
        "selected_child_start_ppo_hist",
        "event_price",
        "event_price_ppo",
        "event_price_ppo_hist",
    ]
    wide = cases[id_cols + ["event_name"] + value_cols].pivot_table(
        index=id_cols,
        columns="event_name",
        values=value_cols,
        aggfunc="first",
        observed=True,
    )
    wide.columns = [f"{event}_{metric}" for metric, event in wide.columns]
    return wide.reset_index()


def build_summaries(cases: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metric_cols = [
        "child_cycle_count_in_parent",
        "selected_child_order",
        "selected_child_progress",
        "selected_same_direction_as_parent",
        "selected_child_start_ppo",
        "selected_child_start_ppo_hist",
        "selected_child_ppo_change",
        "selected_child_ppo_hist_change",
        "selected_child_return_vs_parent_pct",
        "event_price_ppo",
        "event_price_ppo_hist",
    ]
    summary = summarize_numeric(
        cases,
        ["pair", "event_name", "parent_cycle_type", "selected_child_type", "selected_child_progress_bucket"],
        metric_cols,
    )
    regime_summary = summarize_numeric(
        cases,
        ["pair", "event_name", "selected_child_start_regime", "event_price_regime"],
        ["selected_child_progress", "selected_child_return_vs_parent_pct"],
    )
    return {
        "extreme_cycle_summary": summary,
        "extreme_cycle_regime_summary": regime_summary,
    }


def build_report(meta: dict[str, Any], summaries: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# PPO Hierarchical Extreme Cycle Analysis",
        "",
        "## Focus",
        "",
        "- Parent-child cycle containers: `4h/1h/15m` inside `1d`, `1h/15m` inside `4h`, and `15m` inside `1h`.",
        "- For each parent cycle, select child cycles with max/min start PPO, max/min start PPO hist, lowest price, and highest price.",
        "- Price extreme rows include the PPO/PPO hist at the exact low/high candle inside the selected child cycle.",
        "",
        "## Metadata",
        "",
        f"- parent_child_pairs: `{meta['pair_count']}`",
        f"- extreme_case_count: `{meta['extreme_case_count']}`",
        f"- wide_parent_case_count: `{meta['wide_parent_case_count']}`",
        f"- output_dir: `{meta['output_dir']}`",
        "",
    ]

    summary = summaries["extreme_cycle_summary"]
    if not summary.empty:
        lines.extend(["## Extreme Cycle Summary", ""])
        cols = [
            "pair",
            "event_name",
            "parent_cycle_type",
            "selected_child_type",
            "selected_child_progress_bucket",
            "count",
            "selected_same_direction_as_parent_avg",
            "selected_child_start_ppo_avg",
            "selected_child_start_ppo_hist_avg",
            "event_price_ppo_avg",
            "event_price_ppo_hist_avg",
        ]
        lines.append(summary[[col for col in cols if col in summary.columns]].head(100).to_markdown(index=False))
        lines.append("")

    regime_summary = summaries["extreme_cycle_regime_summary"]
    if not regime_summary.empty:
        lines.extend(["## Regime Snapshot", ""])
        cols = [
            "pair",
            "event_name",
            "selected_child_start_regime",
            "event_price_regime",
            "count",
            "selected_child_progress_avg",
            "selected_child_return_vs_parent_pct_avg",
        ]
        lines.append(regime_summary[[col for col in cols if col in regime_summary.columns]].head(80).to_markdown(index=False))
        lines.append("")

    lines.extend(
        [
            "## Reading Notes",
            "",
            "- `selected_child_start_ppo/hist` is the selected child cycle's first candle PPO state.",
            "- `event_price_ppo/hist` is only populated for `lowest_price_cycle` and `highest_price_cycle`; it is the PPO state of the low/high candle itself.",
            "- `selected_child_progress` is where the selected child cycle starts inside the parent cycle, from 0 to 1.",
            "- `selected_child_return_vs_parent_pct` is positive when the selected child moved in the parent cycle direction.",
        ]
    )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    cycles = load_featured_cycles(TIMEFRAMES)
    intervals = {timeframe: interval_frame(cycles[timeframe], timeframe) for timeframe in TIMEFRAMES}
    cases = build_cases(intervals)
    wide = build_wide_cases(cases)
    summaries = build_summaries(cases)

    cases.to_csv(out / "extreme_cycle_cases.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(out / "extreme_cycle_wide_cases.csv", index=False, encoding="utf-8-sig")
    for name, frame in summaries.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    meta = {
        "timeframes": list(TIMEFRAMES),
        "parent_child_pairs": {parent: list(children) for parent, children in PARENT_CHILDREN.items()},
        "pair_count": int(sum(len(children) for children in PARENT_CHILDREN.values())),
        "extreme_case_count": int(len(cases)),
        "wide_parent_case_count": int(len(wide)),
        "output_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(build_report(meta, summaries), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find lower-timeframe PPO and price-extreme cycles inside 1d/4h/1h parent cycles."
    )
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))
