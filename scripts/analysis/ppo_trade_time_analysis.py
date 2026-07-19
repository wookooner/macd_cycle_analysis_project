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
DEFAULT_HORIZONS = (1, 3, 5, 10)
TF_SECONDS = {
    "1min": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
    "1M": 2592000,
}


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_trade_time_analysis"


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


def to_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    try:
        return pd.Timestamp(value)
    except Exception:
        return pd.NaT


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
    ranked = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).rank(method="first")
    try:
        return pd.qcut(ranked, q=bins, labels=[f"{prefix}_q{i + 1}" for i in range(bins)])
    except ValueError:
        return pd.Series(pd.NA, index=series.index, dtype="object")


def age_bucket(series: pd.Series) -> pd.Series:
    bins = [0, 2, 5, 10, 20, 50, 10**9]
    labels = ["age_1_2", "age_3_5", "age_6_10", "age_11_20", "age_21_50", "age_51_plus"]
    return pd.cut(pd.to_numeric(series, errors="coerce"), bins=bins, labels=labels, include_lowest=True)


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
                "timestamp": to_timestamp(candle.get("timestamp", candle.get("date"))),
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
    return df.dropna(subset=["timestamp", "close", "ppo", "ppo_hist"]).reset_index(drop=True)


def build_parent_lookup(timeframes: tuple[str, ...]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    parent_tfs = set(TIMEFRAMES)
    for timeframe in parent_tfs.intersection(set(timeframes).union({"1M", "1w", "1d", "4h", "1h", "30m", "15m", "5m"})):
        path = cycle_dir() / f"cycles_{timeframe}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["cycle_key", "timeframe", "cycle_type", "start_date"])
        for _, row in df.dropna(subset=["cycle_key"]).iterrows():
            lookup[int(row["cycle_key"])] = {
                "timeframe": str(row["timeframe"]),
                "cycle_type": str(row["cycle_type"]).lower(),
                "start_date": to_timestamp(row["start_date"]),
            }
    return lookup


def parent_age_candles(parent_info: dict[str, Any] | None, signal_ts: pd.Timestamp) -> float:
    if not parent_info:
        return np.nan
    parent_start = parent_info.get("start_date")
    parent_tf = parent_info.get("timeframe")
    if pd.isna(parent_start) or parent_tf not in TF_SECONDS:
        return np.nan
    elapsed_seconds = (signal_ts - parent_start).total_seconds()
    if elapsed_seconds < 0:
        return np.nan
    return float(np.floor(elapsed_seconds / TF_SECONDS[parent_tf]) + 1)


def future_return(cdf: pd.DataFrame, signal_idx: int, horizon: int, sign: float) -> float:
    target_idx = signal_idx + horizon
    if target_idx >= len(cdf):
        return np.nan
    entry = cdf.loc[signal_idx, "close"]
    target = cdf.loc[target_idx, "close"]
    if pd.isna(entry) or entry == 0 or pd.isna(target):
        return np.nan
    return (target / entry - 1.0) * 100.0 * sign


def event_row(
    *,
    timeframe: str,
    cycle: pd.Series,
    cdf: pd.DataFrame,
    signal_idx: int,
    signal_kind: str,
    sign: float,
    parent_lookup: dict[int, dict[str, Any]],
    horizons: tuple[int, ...],
) -> dict[str, Any] | None:
    signal = cdf.loc[signal_idx]
    parent_key = cycle.get("parent_key")
    parent_info = None
    try:
        if not pd.isna(parent_key):
            parent_info = parent_lookup.get(int(parent_key))
    except Exception:
        parent_info = None

    parent_type = type_label(cycle.get("parent_type"))
    cycle_type = str(cycle.get("cycle_type", "unknown")).lower()
    row = {
        "timeframe": timeframe,
        "signal_kind": signal_kind,
        "cycle_id": cycle.get("cycle_id"),
        "cycle_key": cycle.get("cycle_key"),
        "cycle_type": cycle_type,
        "parent_key": parent_key,
        "parent_type": parent_type,
        "parent_aligned": cycle_type == parent_type,
        "signal_timestamp": signal["timestamp"],
        "child_age_candles": int(signal_idx + 1),
        "remaining_candles_label": int(len(cdf) - signal_idx - 1),
        "parent_age_candles": parent_age_candles(parent_info, signal["timestamp"]),
        "ppo": float(signal["ppo"]),
        "ppo_hist": float(signal["ppo_hist"]),
        "ppo_delta": float(signal["ppo_delta"]) if not pd.isna(signal["ppo_delta"]) else np.nan,
        "ppo_hist_delta": float(signal["ppo_hist_delta"]) if not pd.isna(signal["ppo_hist_delta"]) else np.nan,
        "price_delta_pct": float(signal["price_delta_pct"]) if not pd.isna(signal["price_delta_pct"]) else np.nan,
        "n_up_4": cycle.get("n_up_4"),
        "combo_4": cycle.get("combo_4"),
    }
    end_close = cdf.iloc[-1]["close"]
    entry = signal["close"]
    row["ret_to_cycle_end_signed_pct_label"] = (end_close / entry - 1.0) * 100.0 * sign if entry else np.nan
    for horizon in horizons:
        row[f"ret_fwd_{horizon}_signed_pct_label"] = future_return(cdf, signal_idx, horizon, sign)
    return row


def analyze_timeframe(
    timeframe: str,
    confirm_candles: int,
    parent_lookup: dict[int, dict[str, Any]],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    cycles = load_cycles(timeframe)
    rows: list[dict[str, Any]] = []

    for _, cycle in cycles.iterrows():
        cycle_type = str(cycle.get("cycle_type", "unknown")).lower()
        sign = 1.0 if cycle_type == "up" else -1.0 if cycle_type == "down" else np.nan
        if pd.isna(sign):
            continue

        cdf = candle_frame(cycle_candles(cycle.get("candle_data")))
        if len(cdf) < confirm_candles + 1:
            continue

        confirm_idx = confirm_candles - 1
        confirm = event_row(
            timeframe=timeframe,
            cycle=cycle,
            cdf=cdf,
            signal_idx=confirm_idx,
            signal_kind=f"confirm_{confirm_candles}",
            sign=sign,
            parent_lookup=parent_lookup,
            horizons=horizons,
        )
        if confirm:
            rows.append(confirm)

        for signal_idx in range(confirm_candles, len(cdf)):
            signal = cdf.loc[signal_idx]
            hist_noise = np.sign(signal["ppo_hist_delta"]) == -sign if signal["ppo_hist_delta"] != 0 else False
            if not hist_noise:
                continue
            noise = event_row(
                timeframe=timeframe,
                cycle=cycle,
                cdf=cdf,
                signal_idx=signal_idx,
                signal_kind="ppo_hist_noise",
                sign=sign,
                parent_lookup=parent_lookup,
                horizons=horizons,
            )
            if noise:
                noise["price_noise"] = bool(np.sign(signal["price_delta_pct"]) == -sign) if signal["price_delta_pct"] != 0 else False
                noise["ppo_slope_noise"] = bool(np.sign(signal["ppo_delta"]) == -sign) if signal["ppo_delta"] != 0 else False
                rows.append(noise)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["ppo_regime"] = regime(df["ppo"], df["ppo_hist"])
    df["ppo_quantile"] = quantile_bucket(df["ppo"], "ppo")
    df["ppo_hist_quantile"] = quantile_bucket(df["ppo_hist"], "hist")
    df["child_age_bucket"] = age_bucket(df["child_age_candles"])
    df["parent_age_bucket"] = age_bucket(df["parent_age_candles"])
    return df


def build_report(summary: pd.DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# PPO Trade-Time Analysis",
        "",
        "## Rule",
        "",
        "Features use only data observable at the signal candle close.",
        "Future returns and remaining cycle candles are labels only.",
        "",
        "Excluded as features: final parent progress, total siblings, cycle end position, future child count.",
        "",
        "## Metadata",
        "",
        f"- timeframes: `{', '.join(meta['timeframes'])}`",
        f"- confirm_candles: `{meta['confirm_candles']}`",
        f"- horizons: `{', '.join(str(v) for v in meta['horizons'])}`",
        f"- case_count: `{meta['case_count']}`",
        f"- output_dir: `{meta['output_dir']}`",
        "",
    ]
    metric = f"ret_fwd_{meta['horizons'][0]}_signed_pct_label_avg"
    if not summary.empty and metric in summary.columns:
        view = summary[summary["count"] >= 300].sort_values(metric, ascending=False).head(30)
        lines.extend(["## Top Signal Groups", ""])
        lines.append(view[[col for col in [
            "timeframe",
            "signal_kind",
            "cycle_type",
            "parent_type",
            "parent_aligned",
            "parent_age_bucket",
            "child_age_bucket",
            "ppo_regime",
            "ppo_quantile",
            "ppo_hist_quantile",
            "count",
            metric,
            f"ret_fwd_{meta['horizons'][0]}_signed_pct_label_win_rate_pct",
            "ret_to_cycle_end_signed_pct_label_avg",
        ] if col in view.columns]].to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def run(
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
    confirm_candles: int = 3,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    include_cases: bool = False,
) -> dict[str, Any]:
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    parent_lookup = build_parent_lookup(timeframes)

    frames = [
        analyze_timeframe(timeframe, confirm_candles, parent_lookup, horizons)
        for timeframe in timeframes
    ]
    cases = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    metrics = ["ret_to_cycle_end_signed_pct_label"] + [
        f"ret_fwd_{horizon}_signed_pct_label" for horizon in horizons
    ]
    summary = summarize(
        cases,
        [
            "timeframe",
            "signal_kind",
            "cycle_type",
            "parent_type",
            "parent_aligned",
            "parent_age_bucket",
            "child_age_bucket",
            "ppo_regime",
            "ppo_quantile",
            "ppo_hist_quantile",
        ],
        metrics,
    )
    summary.to_csv(out / "trade_time_signal_summary.csv", index=False, encoding="utf-8-sig")
    if include_cases:
        cases.to_csv(out / "trade_time_signal_cases.csv", index=False, encoding="utf-8-sig")

    meta = {
        "timeframes": list(timeframes),
        "confirm_candles": confirm_candles,
        "horizons": list(horizons),
        "case_count": int(len(cases)),
        "output_dir": str(out),
        "feature_rule": "observable_at_signal_close_only",
    }
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(build_report(summary, meta), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lookahead-safe PPO analysis from a trading decision perspective.")
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), choices=list(TIMEFRAMES))
    parser.add_argument("--confirm-candles", type=int, default=3)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--include-cases", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            run(
                timeframes=tuple(args.timeframes),
                confirm_candles=args.confirm_candles,
                horizons=tuple(args.horizons),
                include_cases=args.include_cases,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
