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
    cycle_sign,
    load_cycles,
    progress_bucket,
    quantile_bucket,
    regime,
)
from src.common.paths import PROJECT_PATHS


DEFAULT_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")
DEFAULT_HORIZONS = (1, 3, 5, 10)
DEFAULT_MIN_CASES = 50
HIGHER_TF = {"5m": "15m", "15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}
RELATION_COLS = [
    "cycle_uid",
    "parent_type",
    "parent_progress_at_start",
    "parent_progress_at_end",
    "order_in_parent",
    "total_siblings",
    "boundary_type",
    "n_up_4",
    "combo_4",
    "prev_type",
    "prev_dur",
    "opposite_child_ratio",
    "max_opposite_child_streak",
]


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "hist_candle_reaction_matrix"


def _round(value: Any, digits: int = 6) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return value


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _type_label(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "unknown"
    except Exception:
        pass
    if value in (1, "1", "up", "UP"):
        return "up"
    if value in (-1, "-1", "down", "DOWN"):
        return "down"
    return "unknown"


def _sign_label(sign: pd.Series, prefix: str = "") -> pd.Series:
    values = pd.to_numeric(sign, errors="coerce")
    labels = pd.Series("flat", index=values.index, dtype="object")
    labels[values > 0] = f"{prefix}up" if prefix else "up"
    labels[values < 0] = f"{prefix}down" if prefix else "down"
    labels[values.isna()] = "unknown"
    return labels


def _sibling_bucket(order: pd.Series, total: pd.Series) -> pd.Series:
    order_num = pd.to_numeric(order, errors="coerce")
    total_num = pd.to_numeric(total, errors="coerce")
    denom = (total_num - 1).replace(0, np.nan)
    ratio = ((order_num - 1) / denom).fillna(0.0).clip(lower=0.0, upper=1.0)
    return progress_bucket(ratio)


def _add_forward_returns(frame: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    frame = frame.sort_values(["timeframe", "timestamp"]).reset_index(drop=True).copy()
    for horizon in horizons:
        future_close = frame.groupby("timeframe", observed=True)["close"].shift(-horizon)
        raw = (future_close / frame["close"] - 1.0) * 100.0
        sign = pd.to_numeric(frame["cycle_sign"], errors="coerce")
        frame[f"ret_fwd_{horizon}_cycle_pct"] = raw * sign
        frame[f"ret_fwd_{horizon}_opposite_pct"] = raw * -sign
        frame[f"ret_fwd_{horizon}_long_pct"] = raw
        frame[f"ret_fwd_{horizon}_short_pct"] = -raw
    return frame


def _add_cycle_end_returns(frame: pd.DataFrame) -> pd.DataFrame:
    end_close = (
        frame.sort_values(["cycle_uid", "candle_index"])
        .groupby("cycle_uid", observed=True)
        .tail(1)[["cycle_uid", "close"]]
        .rename(columns={"close": "cycle_end_close"})
    )
    frame = frame.merge(end_close, on="cycle_uid", how="left")
    raw = (frame["cycle_end_close"] / frame["close"] - 1.0) * 100.0
    sign = pd.to_numeric(frame["cycle_sign"], errors="coerce")
    frame["ret_to_cycle_end_cycle_pct"] = raw * sign
    frame["ret_to_cycle_end_opposite_pct"] = raw * -sign
    return frame


def assign_upper_context(events: pd.DataFrame, cycles_by_tf: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if events.empty:
        return events

    events = events.copy()
    for column in ["upper_timeframe", "upper_cycle_id", "upper_cycle_type"]:
        events[column] = pd.NA
    for column in ["upper_cycle_sign", "upper_cycle_progress"]:
        events[column] = np.nan

    for timeframe, upper_tf in HIGHER_TF.items():
        mask = events["timeframe"].eq(timeframe)
        upper = cycles_by_tf.get(upper_tf)
        if not mask.any() or upper is None or upper.empty:
            continue

        upper = upper.sort_values("start_date").reset_index(drop=True)
        event_ns = events.loc[mask, "timestamp"].astype("int64").to_numpy()
        start_ns = upper["start_date"].astype("int64").to_numpy()
        end_ns = upper["end_exclusive"].astype("int64").to_numpy()
        idx = np.searchsorted(start_ns, event_ns, side="right") - 1
        valid = idx >= 0
        covered = np.zeros(len(event_ns), dtype=bool)
        covered[valid] = event_ns[valid] < end_ns[idx[valid]]
        target_index = events.index[mask][covered]
        if len(target_index) == 0:
            continue

        mapped = upper.iloc[idx[covered]].reset_index(drop=True)
        denom = (end_ns[idx[covered]] - start_ns[idx[covered]]).astype("float64")
        progress = np.divide(
            event_ns[covered] - start_ns[idx[covered]],
            denom,
            out=np.full(len(target_index), np.nan, dtype="float64"),
            where=denom != 0,
        )
        events.loc[target_index, "upper_timeframe"] = upper_tf
        events.loc[target_index, "upper_cycle_id"] = mapped["cycle_id"].astype(str).to_numpy()
        events.loc[target_index, "upper_cycle_type"] = mapped["cycle_type"].astype(str).to_numpy()
        events.loc[target_index, "upper_cycle_sign"] = pd.to_numeric(mapped["cycle_sign"], errors="coerce").to_numpy()
        events.loc[target_index, "upper_cycle_progress"] = np.clip(progress, 0.0, 1.0)

    return events


def build_cases(timeframes: tuple[str, ...], horizons: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required_tfs = set(timeframes)
    required_tfs.update(HIGHER_TF[tf] for tf in timeframes if tf in HIGHER_TF)
    cycles_by_tf = {timeframe: load_cycles(timeframe) for timeframe in required_tfs}
    for timeframe in timeframes:
        cycles = cycles_by_tf[timeframe]
        candles = build_candle_frame(cycles, timeframe)
        if candles.empty:
            continue

        relation_cols = [col for col in RELATION_COLS if col in cycles.columns or col == "cycle_uid"]
        relations = cycles[relation_cols].copy()
        candles = candles.merge(relations, on="cycle_uid", how="left")
        candles["timeframe"] = timeframe
        frames.append(candles)

    cases = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if cases.empty:
        return cases

    cases = assign_upper_context(cases, cycles_by_tf)
    cases = cases[cases["candle_index"] > 1].copy()
    cases = _add_forward_returns(cases, horizons)
    cases = _add_cycle_end_returns(cases)

    hist_delta = pd.to_numeric(cases["ppo_hist_delta"], errors="coerce")
    hist_delta_sign = np.sign(hist_delta)
    cycle_signs = pd.to_numeric(cases["cycle_sign"], errors="coerce")
    cases["hist_delta_direction"] = _sign_label(pd.Series(hist_delta_sign, index=cases.index))
    cases["hist_delta_vs_cycle"] = np.select(
        [
            hist_delta_sign == cycle_signs,
            hist_delta_sign == -cycle_signs,
            hist_delta_sign == 0,
        ],
        ["with_cycle", "against_cycle", "flat"],
        default="unknown",
    )
    cases["hist_delta_abs"] = hist_delta.abs()
    cases["hist_delta_abs_bucket"] = cases.groupby("timeframe", observed=True)["hist_delta_abs"].transform(
        lambda series: quantile_bucket(series, "hist_abs")
    )
    cases["ppo_regime"] = regime(cases["ppo"], cases["ppo_hist"])
    cases["ppo_bucket"] = cases.groupby("timeframe", observed=True)["ppo"].transform(lambda series: quantile_bucket(series, "ppo"))
    cases["ppo_hist_bucket"] = cases.groupby("timeframe", observed=True)["ppo_hist"].transform(
        lambda series: quantile_bucket(series, "hist")
    )
    cases["cycle_progress_bucket"] = progress_bucket(cases["cycle_progress"])
    cases["candle_age_bucket"] = pd.cut(
        pd.to_numeric(cases["candle_index"], errors="coerce"),
        bins=[1, 2, 5, 10, 20, 50, 10**9],
        labels=["age_2", "age_3_5", "age_6_10", "age_11_20", "age_21_50", "age_51_plus"],
        include_lowest=True,
    )
    parent_source = cases.get("parent_type", pd.Series(index=cases.index, dtype="object"))
    cases["parent_type_label"] = parent_source.map(_type_label)
    upper_labels = cases["upper_cycle_type"].map(_type_label)
    cases.loc[cases["parent_type_label"].eq("unknown") & upper_labels.ne("unknown"), "parent_type_label"] = upper_labels
    parent_sign = cases["parent_type_label"].map(cycle_sign)
    cases["parent_aligned"] = parent_sign == cycle_signs
    parent_progress = cases.get("parent_progress_at_start", pd.Series(index=cases.index, dtype="float64"))
    parent_progress = pd.to_numeric(parent_progress, errors="coerce").fillna(pd.to_numeric(cases["upper_cycle_progress"], errors="coerce"))
    cases["parent_progress_bucket"] = progress_bucket(parent_progress)
    cases["upper_progress_bucket"] = progress_bucket(cases["upper_cycle_progress"])
    cases["sibling_position_bucket"] = _sibling_bucket(
        cases.get("order_in_parent", pd.Series(index=cases.index)),
        cases.get("total_siblings", pd.Series(index=cases.index)),
    )
    for column in ["n_up_4", "combo_4", "boundary_type", "opposite_child_ratio", "max_opposite_child_streak"]:
        if column not in cases.columns:
            cases[column] = pd.NA
    cases["position_context"] = (
        cases["timeframe"].astype(str)
        + "|"
        + cases["cycle_type"].astype(str)
        + "|"
        + cases["hist_delta_vs_cycle"].astype(str)
        + "|"
        + cases["cycle_progress_bucket"].astype(str)
        + "|parent="
        + cases["parent_type_label"].astype(str)
        + "|p_prog="
        + cases["parent_progress_bucket"].astype(str)
    )
    return cases


def summarize(cases: pd.DataFrame, group_cols: list[str], horizons: tuple[int, ...], min_cases: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if cases.empty:
        return pd.DataFrame()

    metrics = []
    for horizon in horizons:
        metrics.extend([f"ret_fwd_{horizon}_cycle_pct", f"ret_fwd_{horizon}_opposite_pct"])
    metrics.extend(["ret_to_cycle_end_cycle_pct", "ret_to_cycle_end_opposite_pct"])

    first_horizon = horizons[0]
    for keys, group in cases.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(group) < min_cases:
            continue
        row = {column: value for column, value in zip(group_cols, keys)}
        row["count"] = int(len(group))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_avg"] = _round(values.mean()) if not values.empty else None
            row[f"{metric}_wr_pct"] = _round((values > 0).mean() * 100.0) if not values.empty else None
            row[f"{metric}_p25"] = _round(values.quantile(0.25)) if not values.empty else None
            row[f"{metric}_p75"] = _round(values.quantile(0.75)) if not values.empty else None

        cycle_avg = row.get(f"ret_fwd_{first_horizon}_cycle_pct_avg")
        opposite_avg = row.get(f"ret_fwd_{first_horizon}_opposite_pct_avg")
        cycle_wr = row.get(f"ret_fwd_{first_horizon}_cycle_pct_wr_pct")
        opposite_wr = row.get(f"ret_fwd_{first_horizon}_opposite_pct_wr_pct")
        edge = None if cycle_avg is None or opposite_avg is None else cycle_avg - opposite_avg
        row["cycle_vs_opposite_edge_pct"] = _round(edge)
        if edge is None:
            action = "watch"
        elif edge > 0 and (cycle_wr or 0) >= 52:
            action = "hold_or_add_cycle_side"
        elif edge < 0 and (opposite_wr or 0) >= 52:
            action = "reduce_or_flip_opposite_side"
        else:
            action = "wait_or_reduce_size"
        row["empirical_action"] = action
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["timeframe", "count"], ascending=[True, False] if "timeframe" in group_cols else [False])


def build_report(summaries: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> str:
    lines = [
        "# Hist Candle Reaction Matrix",
        "",
        "## 목적",
        "",
        "한 캔들의 `ppo_hist_delta`가 현재 사이클 방향과 같거나 반대로 움직였을 때, 현재 위치와 상위 사이클 맥락별로 포지션 반응을 집계한다.",
        "",
        "## 읽는 법",
        "",
        "- `hold_or_add_cycle_side`: 같은 사이클 방향 포지션 유지/추가가 과거 평균과 승률에서 우위.",
        "- `reduce_or_flip_opposite_side`: 현재 사이클 반대 방향 대응이 우위.",
        "- `wait_or_reduce_size`: 우위가 약하거나 승률 조건이 부족.",
        "- 결과는 규칙 확정이 아니라 다음 백테스트 후보를 좁히는 조건부 매트릭스.",
        "",
        "## Metadata",
        "",
        f"- timeframes: `{', '.join(metadata['timeframes'])}`",
        f"- horizons: `{', '.join(str(v) for v in metadata['horizons'])}`",
        f"- min_cases: `{metadata['min_cases']}`",
        f"- case_count: `{metadata['case_count']}`",
        f"- output_dir: `{metadata['output_dir']}`",
        "",
    ]

    for title, key in [
        ("Core Reaction By Current Cycle Position", "core"),
        ("Upper Cycle Context", "upper_context"),
        ("Chain And Sibling Context", "chain_sibling"),
        ("Detailed PPO Zone Context", "ppo_zone"),
    ]:
        table = summaries.get(key, pd.DataFrame())
        lines.extend([f"## {title}", ""])
        if table.empty:
            lines.extend(["결과 없음", ""])
            continue
        edge_col = "cycle_vs_opposite_edge_pct"
        view = table.copy()
        view["_abs_edge"] = pd.to_numeric(view[edge_col], errors="coerce").abs()
        view = view.sort_values(["timeframe", "_abs_edge", "count"], ascending=[True, False, False]).drop(columns=["_abs_edge"]).head(40)
        show_cols = [
            col
            for col in view.columns
            if col
            in {
                "timeframe",
                "cycle_type",
                "hist_delta_vs_cycle",
                "hist_delta_abs_bucket",
                "cycle_progress_bucket",
                "candle_age_bucket",
                "parent_type_label",
                "parent_aligned",
                "parent_progress_bucket",
                "sibling_position_bucket",
                "n_up_4",
                "combo_4",
                "ppo_regime",
                "ppo_bucket",
                "ppo_hist_bucket",
                "count",
                "cycle_vs_opposite_edge_pct",
                "empirical_action",
            }
        ]
        lines.append(view[show_cols].to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def run(
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    min_cases: int = DEFAULT_MIN_CASES,
    include_cases: bool = True,
) -> dict[str, Any]:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    cases = build_cases(timeframes=timeframes, horizons=horizons)

    summaries = {
        "core": summarize(
            cases,
            ["timeframe", "cycle_type", "hist_delta_vs_cycle", "hist_delta_abs_bucket", "cycle_progress_bucket", "candle_age_bucket"],
            horizons,
            min_cases,
        ),
        "upper_context": summarize(
            cases,
            ["timeframe", "cycle_type", "hist_delta_vs_cycle", "parent_type_label", "parent_aligned", "parent_progress_bucket"],
            horizons,
            min_cases,
        ),
        "chain_sibling": summarize(
            cases,
            ["timeframe", "cycle_type", "hist_delta_vs_cycle", "n_up_4", "combo_4", "sibling_position_bucket"],
            horizons,
            min_cases,
        ),
        "ppo_zone": summarize(
            cases,
            ["timeframe", "cycle_type", "hist_delta_vs_cycle", "ppo_regime", "ppo_bucket", "ppo_hist_bucket", "cycle_progress_bucket"],
            horizons,
            min_cases,
        ),
    }

    if include_cases and not cases.empty:
        cases.to_csv(out / "hist_candle_reaction_cases.csv", index=False, encoding="utf-8-sig")
    for name, frame in summaries.items():
        if not frame.empty:
            frame.to_csv(out / f"{name}_summary.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "timeframes": list(timeframes),
        "horizons": list(horizons),
        "min_cases": int(min_cases),
        "case_count": int(len(cases)),
        "output_dir": str(out),
        "files": {
            "cases": "hist_candle_reaction_cases.csv" if include_cases else None,
            **{f"{name}_summary": f"{name}_summary.csv" for name, frame in summaries.items() if not frame.empty},
            "report": "report.md",
            "summary": "summary.json",
        },
    }
    (out / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(build_report(summaries, metadata), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze position reactions to one-candle PPO hist changes by cycle context.")
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), choices=list(TIMEFRAMES))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
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
                min_cases=args.min_cases,
                include_cases=not args.no_cases,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
