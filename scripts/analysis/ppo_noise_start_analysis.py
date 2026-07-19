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


TIMEFRAMES = ("1M", "1w", "1d", "4h", "1h", "30m", "15m", "5m", "1min")
DEFAULT_TIMEFRAMES = ("1d", "4h", "1h", "30m", "15m", "5m")
PROGRESS_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.000001)
PROGRESS_LABELS = ("p00_20", "p20_40", "p40_60", "p60_80", "p80_100")


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_noise_start_analysis"


def cycle_dir() -> Path:
    return PROJECT_PATHS.asset_cycle_dir("btc")


def load_cycles(timeframe: str) -> pd.DataFrame:
    path = cycle_dir() / f"cycles_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing cycle file: {path}")
    return pd.read_parquet(path)


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


def to_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def type_label(value: Any) -> str:
    try:
        if pd.isna(value):
            return "unknown"
    except Exception:
        pass
    if value in (1, "1", "up", "UP"):
        return "up"
    if value in (-1, "-1", "down", "DOWN"):
        return "down"
    return "unknown"


def regime(ppo: pd.Series, ppo_hist: pd.Series) -> pd.Series:
    ppo_sign = np.where(ppo >= 0, "ppo_pos", "ppo_neg")
    hist_sign = np.where(ppo_hist >= 0, "hist_pos", "hist_neg")
    return pd.Series(ppo_sign + "__" + hist_sign, index=ppo.index)


def quantile_bucket(series: pd.Series, prefix: str, bins: int = 5) -> pd.Series:
    ranked = series.replace([np.inf, -np.inf], np.nan).rank(method="first")
    try:
        return pd.qcut(ranked, q=bins, labels=[f"{prefix}_q{i + 1}" for i in range(bins)])
    except ValueError:
        return pd.Series(pd.NA, index=series.index, dtype="object")


def progress_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=1.0),
        bins=PROGRESS_BINS,
        labels=PROGRESS_LABELS,
        include_lowest=True,
    )


def sibling_bucket(order: pd.Series, total: pd.Series) -> pd.Series:
    order_num = pd.to_numeric(order, errors="coerce")
    total_num = pd.to_numeric(total, errors="coerce")
    ratio = (order_num - 1) / (total_num - 1).replace(0, np.nan)
    ratio = ratio.fillna(0.0).clip(lower=0.0, upper=1.0)
    return progress_bucket(ratio)


def summarize(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    for keys, group in frame.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["count"] = int(len(group))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_avg"] = None
                row[f"{metric}_median"] = None
                row[f"{metric}_win_rate_pct"] = None
                continue
            row[f"{metric}_avg"] = round(float(values.mean()), 6)
            row[f"{metric}_median"] = round(float(values.median()), 6)
            row[f"{metric}_win_rate_pct"] = round(float((values > 0).mean() * 100), 6)
            row[f"{metric}_p25"] = round(float(values.quantile(0.25)), 6)
            row[f"{metric}_p75"] = round(float(values.quantile(0.75)), 6)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols + ["count"], ascending=[True] * len(group_cols) + [False])


def candle_frame(candles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, candle in enumerate(candles):
        rows.append(
            {
                "idx": idx,
                "timestamp": candle.get("timestamp", candle.get("date")),
                "close": to_float(candle.get("close")),
                "ppo": to_float(candle.get("ppo")),
                "ppo_hist": to_float(candle.get("ppo_hist")),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ppo_delta"] = df["ppo"].diff()
    df["ppo_hist_delta"] = df["ppo_hist"].diff()
    df["price_delta_pct"] = (df["close"] / df["close"].shift(1) - 1.0) * 100.0
    return df


def analyze_timeframe(timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = load_cycles(timeframe)
    start_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []

    for _, row in cycles.iterrows():
        cycle_type = str(row.get("cycle_type", "unknown")).lower()
        sign = 1.0 if cycle_type == "up" else -1.0 if cycle_type == "down" else np.nan
        if pd.isna(sign):
            continue

        candles = cycle_candles(row.get("candle_data"))
        if len(candles) < 2:
            continue
        cdf = candle_frame(candles).dropna(subset=["close", "ppo", "ppo_hist"]).reset_index(drop=True)
        if len(cdf) < 2:
            continue

        start = cdf.iloc[0]
        end = cdf.iloc[-1]
        start_close = start["close"]
        end_close = end["close"]
        if start_close == 0 or pd.isna(start_close) or pd.isna(end_close):
            continue

        common = {
            "timeframe": timeframe,
            "cycle_id": row.get("cycle_id"),
            "cycle_key": row.get("cycle_key"),
            "cycle_type": cycle_type,
            "parent_key": row.get("parent_key"),
            "parent_type": type_label(row.get("parent_type")),
            "parent_progress_at_start": row.get("parent_progress_at_start"),
            "parent_progress_at_end": row.get("parent_progress_at_end"),
            "order_in_parent": row.get("order_in_parent"),
            "total_siblings": row.get("total_siblings"),
            "boundary_type": row.get("boundary_type"),
            "n_up_4": row.get("n_up_4"),
            "combo_4": row.get("combo_4"),
            "duration_candles": row.get("duration_candles"),
        }
        cycle_raw_move = (end_close / start_close - 1.0) * 100.0
        start_rows.append(
            {
                **common,
                "start_ppo": float(start["ppo"]),
                "start_ppo_hist": float(start["ppo_hist"]),
                "cycle_price_change_pct": cycle_raw_move,
                "cycle_price_change_signed_pct": cycle_raw_move * sign,
            }
        )

        for _, candle in cdf.iloc[1:].iterrows():
            hist_noise = np.sign(candle["ppo_hist_delta"]) == -sign if candle["ppo_hist_delta"] != 0 else False
            ppo_noise = np.sign(candle["ppo_delta"]) == -sign if candle["ppo_delta"] != 0 else False
            price_noise = np.sign(candle["price_delta_pct"]) == -sign if candle["price_delta_pct"] != 0 else False
            if not hist_noise:
                continue

            close = candle["close"]
            if close == 0 or pd.isna(close):
                continue
            raw_move = (end_close / close - 1.0) * 100.0
            progress = float(candle["idx"]) / float(max(len(cdf) - 1, 1))
            noise_rows.append(
                {
                    **common,
                    "noise_idx": int(candle["idx"]) + 1,
                    "noise_progress": progress,
                    "noise_remaining_candles": int(len(cdf) - int(candle["idx"]) - 1),
                    "noise_ppo": float(candle["ppo"]),
                    "noise_ppo_hist": float(candle["ppo_hist"]),
                    "noise_ppo_delta": float(candle["ppo_delta"]),
                    "noise_ppo_hist_delta": float(candle["ppo_hist_delta"]),
                    "noise_price_delta_pct": float(candle["price_delta_pct"]),
                    "ppo_slope_noise": bool(ppo_noise),
                    "price_noise": bool(price_noise),
                    "move_from_noise_to_end_pct": raw_move,
                    "move_from_noise_to_end_signed_pct": raw_move * sign,
                }
            )

    starts = pd.DataFrame(start_rows)
    noises = pd.DataFrame(noise_rows)

    if not starts.empty:
        starts["start_ppo_regime"] = regime(starts["start_ppo"], starts["start_ppo_hist"])
        starts["parent_progress_bucket"] = progress_bucket(starts["parent_progress_at_start"])
        starts["sibling_position_bucket"] = sibling_bucket(starts["order_in_parent"], starts["total_siblings"])
        starts["start_ppo_quantile"] = quantile_bucket(starts["start_ppo"], "ppo")
        starts["start_ppo_hist_quantile"] = quantile_bucket(starts["start_ppo_hist"], "hist")
        starts["parent_aligned"] = starts["cycle_type"] == starts["parent_type"]

    if not noises.empty:
        noises["noise_ppo_regime"] = regime(noises["noise_ppo"], noises["noise_ppo_hist"])
        noises["noise_progress_bucket"] = progress_bucket(noises["noise_progress"])
        noises["parent_progress_bucket"] = progress_bucket(noises["parent_progress_at_start"])
        noises["sibling_position_bucket"] = sibling_bucket(noises["order_in_parent"], noises["total_siblings"])
        noises["noise_ppo_quantile"] = quantile_bucket(noises["noise_ppo"], "ppo")
        noises["noise_ppo_hist_quantile"] = quantile_bucket(noises["noise_ppo_hist"], "hist")
        noises["parent_aligned"] = noises["cycle_type"] == noises["parent_type"]

    return starts, noises


def build_report(start_summary: pd.DataFrame, noise_summary: pd.DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# PPO Noise And Start Position Analysis",
        "",
        "## Scenarios",
        "",
        "1. Cycle start: group cycles by start PPO/PPO hist quantile and sign regime.",
        "2. Noise candles: primary noise is a candle where PPO hist slope moves opposite to the cycle direction.",
        "3. Parent context: parent direction, parent progress bucket, and sibling position are included.",
        "",
        "## Metadata",
        "",
        f"- timeframes: `{', '.join(meta['timeframes'])}`",
        f"- start cases: `{meta['start_case_count']}`",
        f"- noise cases: `{meta['noise_case_count']}`",
        f"- output_dir: `{meta['output_dir']}`",
        "",
    ]

    def add_table(title: str, df: pd.DataFrame, metric: str, cols: list[str]) -> None:
        lines.extend([f"## {title}", ""])
        if df.empty:
            lines.extend(["No rows.", ""])
            return
        view = df[df["count"] >= 1000].sort_values(metric, ascending=False).head(20)
        if view.empty:
            view = df.sort_values(metric, ascending=False).head(20)
        lines.append(view[[col for col in cols if col in view.columns]].to_markdown(index=False))
        lines.append("")

    add_table(
        "Cycle Start PPO Position Top",
        start_summary,
        "cycle_price_change_signed_pct_avg",
        [
            "timeframe",
            "cycle_type",
            "parent_type",
            "parent_aligned",
            "parent_progress_bucket",
            "start_ppo_regime",
            "start_ppo_quantile",
            "start_ppo_hist_quantile",
            "count",
            "cycle_price_change_signed_pct_avg",
            "cycle_price_change_signed_pct_win_rate_pct",
            "duration_candles_avg",
        ],
    )
    add_table(
        "Noise Candle Followthrough Top",
        noise_summary,
        "move_from_noise_to_end_signed_pct_avg",
        [
            "timeframe",
            "cycle_type",
            "parent_type",
            "parent_aligned",
            "parent_progress_bucket",
            "noise_progress_bucket",
            "noise_ppo_regime",
            "noise_ppo_quantile",
            "noise_ppo_hist_quantile",
            "price_noise",
            "ppo_slope_noise",
            "count",
            "move_from_noise_to_end_signed_pct_avg",
            "move_from_noise_to_end_signed_pct_win_rate_pct",
            "noise_remaining_candles_avg",
        ],
    )
    return "\n".join(lines)


def run(timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES, include_cases: bool = False) -> dict[str, Any]:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    start_frames: list[pd.DataFrame] = []
    noise_frames: list[pd.DataFrame] = []
    for timeframe in timeframes:
        starts, noises = analyze_timeframe(timeframe)
        start_frames.append(starts)
        noise_frames.append(noises)

    starts_all = pd.concat(start_frames, ignore_index=True) if start_frames else pd.DataFrame()
    noises_all = pd.concat(noise_frames, ignore_index=True) if noise_frames else pd.DataFrame()

    start_summary = summarize(
        starts_all,
        [
            "timeframe",
            "cycle_type",
            "parent_type",
            "parent_aligned",
            "parent_progress_bucket",
            "sibling_position_bucket",
            "start_ppo_regime",
            "start_ppo_quantile",
            "start_ppo_hist_quantile",
        ],
        ["cycle_price_change_signed_pct", "cycle_price_change_pct", "duration_candles"],
    )
    noise_summary = summarize(
        noises_all,
        [
            "timeframe",
            "cycle_type",
            "parent_type",
            "parent_aligned",
            "parent_progress_bucket",
            "noise_progress_bucket",
            "noise_ppo_regime",
            "noise_ppo_quantile",
            "noise_ppo_hist_quantile",
            "price_noise",
            "ppo_slope_noise",
        ],
        ["move_from_noise_to_end_signed_pct", "move_from_noise_to_end_pct", "noise_remaining_candles"],
    )

    start_summary.to_csv(out / "cycle_start_ppo_position_summary.csv", index=False, encoding="utf-8-sig")
    noise_summary.to_csv(out / "ppo_hist_noise_candle_summary.csv", index=False, encoding="utf-8-sig")
    if include_cases:
        starts_all.to_csv(out / "cycle_start_ppo_position_cases.csv", index=False, encoding="utf-8-sig")
        noises_all.to_csv(out / "ppo_hist_noise_candle_cases.csv", index=False, encoding="utf-8-sig")

    meta = {
        "timeframes": list(timeframes),
        "output_dir": str(out),
        "start_case_count": int(len(starts_all)),
        "noise_case_count": int(len(noises_all)),
        "noise_definition": "ppo_hist_delta sign opposite to cycle direction",
    }
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(build_report(start_summary, noise_summary, meta), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PPO cycle start position and PPO-hist noise candles.")
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), choices=list(TIMEFRAMES))
    parser.add_argument("--include-cases", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(tuple(args.timeframes), include_cases=args.include_cases), ensure_ascii=False, indent=2))
