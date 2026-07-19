from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
PROGRESS_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.000001)
PROGRESS_LABELS = ("p00_20", "p20_40", "p40_60", "p60_80", "p80_100")
POSITION_LABELS = ("start", "q25", "mid", "q75", "end")
POSITION_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Paths:
    output_dir: Path
    raw_market_dir: Path
    cycle_dir: Path


def _paths() -> Paths:
    return Paths(
        output_dir=PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_position_movement_analysis",
        raw_market_dir=PROJECT_PATHS.base_data_dir,
        cycle_dir=PROJECT_PATHS.asset_cycle_dir("btc"),
    )


def _round(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return value


def _safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def ppo_regime(ppo: pd.Series, ppo_hist: pd.Series) -> pd.Series:
    ppo_sign = np.where(ppo >= 0, "ppo_pos", "ppo_neg")
    hist_sign = np.where(ppo_hist >= 0, "hist_pos", "hist_neg")
    return pd.Series(ppo_sign + "__" + hist_sign, index=ppo.index)


def _bucket_quantile(series: pd.Series, bins: int, prefix: str) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    ranked = clean.rank(method="first")
    try:
        return pd.qcut(ranked, q=bins, labels=[f"{prefix}_q{i + 1}" for i in range(bins)])
    except ValueError:
        return pd.Series(pd.NA, index=series.index, dtype="object")


def summarize_numeric(frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()

    for keys, group in frame.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_cols, keys)}
        row["count"] = int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_avg"] = _round(values.mean()) if not values.empty else None
            row[f"{metric}_median"] = _round(values.median()) if not values.empty else None
            row[f"{metric}_win_rate_pct"] = _round((values > 0).mean() * 100) if not values.empty else None
            row[f"{metric}_p25"] = _round(values.quantile(0.25)) if not values.empty else None
            row[f"{metric}_p75"] = _round(values.quantile(0.75)) if not values.empty else None
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols + ["count"], ascending=[True] * len(group_cols) + [False])


def load_market(timeframe: str) -> pd.DataFrame:
    path = _paths().raw_market_dir / f"BTCUSD_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing market file: {path}")

    usecols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "ppo",
        "ppo_signal",
        "ppo_hist",
        "macd_hist",
    ]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    df = pd.read_csv(path, usecols=[column for column in usecols if column in header])
    required = {"date", "close", "ppo", "ppo_hist"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _safe_numeric(df, [column for column in df.columns if column != "date"])
    return df.sort_values("date").dropna(subset=["date", "close", "ppo", "ppo_hist"]).reset_index(drop=True)


def analyze_market_timeframe(timeframe: str, horizons: tuple[int, ...], bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_market(timeframe)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    cases = pd.DataFrame(
        {
            "timeframe": timeframe,
            "date": df["date"],
            "close": df["close"],
            "ppo": df["ppo"],
            "ppo_hist": df["ppo_hist"],
            "ppo_regime": ppo_regime(df["ppo"], df["ppo_hist"]),
            "ppo_quantile": _bucket_quantile(df["ppo"], bins, "ppo"),
            "ppo_hist_quantile": _bucket_quantile(df["ppo_hist"], bins, "hist"),
        }
    )
    if "macd_hist" in df.columns:
        cases["macd_hist_direction"] = np.sign(df["macd_hist"].diff()).astype("float")

    for horizon in horizons:
        future_close = df["close"].shift(-horizon)
        cases[f"ret_fwd_{horizon}"] = (future_close / df["close"] - 1.0) * 100.0

    cases = cases.dropna(subset=[f"ret_fwd_{horizons[0]}"]).reset_index(drop=True)
    metric_cols = [f"ret_fwd_{horizon}" for horizon in horizons]
    summary = summarize_numeric(
        cases,
        group_cols=["timeframe", "ppo_regime", "ppo_quantile", "ppo_hist_quantile"],
        metric_cols=metric_cols,
    )
    return cases, summary


def _cycle_path(timeframe: str) -> Path:
    return _paths().cycle_dir / f"cycles_{timeframe}.parquet"


def load_cycles(timeframe: str) -> pd.DataFrame:
    path = _cycle_path(timeframe)
    if not path.exists():
        raise FileNotFoundError(f"missing cycle file: {path}")
    return pd.read_parquet(path)


def _cycle_candles(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    candles: list[dict[str, Any]] = []
    for candle in value:
        if isinstance(candle, dict):
            candles.append(candle)
        else:
            try:
                candles.append(dict(candle))
            except Exception:
                continue
    return candles


def _type_to_label(value: Any) -> str:
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


def _progress_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=1.0),
        bins=PROGRESS_BINS,
        labels=PROGRESS_LABELS,
        include_lowest=True,
    )


def _sibling_bucket(order: pd.Series, total: pd.Series) -> pd.Series:
    order_num = pd.to_numeric(order, errors="coerce")
    total_num = pd.to_numeric(total, errors="coerce")
    ratio = (order_num - 1) / (total_num - 1).replace(0, np.nan)
    ratio = ratio.fillna(0.0).clip(lower=0.0, upper=1.0)
    return pd.cut(ratio, bins=PROGRESS_BINS, labels=PROGRESS_LABELS, include_lowest=True)


def analyze_cycle_positions(timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = load_cycles(timeframe)
    rows: list[dict[str, Any]] = []

    for _, cycle in cycles.iterrows():
        candles = _cycle_candles(cycle.get("candle_data"))
        if not candles:
            continue
        last_close = pd.to_numeric(pd.Series([candles[-1].get("close")]), errors="coerce").iloc[0]
        if pd.isna(last_close):
            continue

        cycle_type = str(cycle.get("cycle_type", "unknown")).lower()
        sign = 1.0 if cycle_type == "up" else -1.0 if cycle_type == "down" else np.nan
        last_index = max(len(candles) - 1, 0)
        for label, fraction in zip(POSITION_LABELS, POSITION_FRACTIONS):
            candle = candles[int(round(last_index * fraction))]
            close = pd.to_numeric(pd.Series([candle.get("close")]), errors="coerce").iloc[0]
            ppo = pd.to_numeric(pd.Series([candle.get("ppo")]), errors="coerce").iloc[0]
            ppo_hist = pd.to_numeric(pd.Series([candle.get("ppo_hist")]), errors="coerce").iloc[0]
            if pd.isna(close) or close == 0 or pd.isna(ppo) or pd.isna(ppo_hist):
                continue
            raw_move = (last_close / close - 1.0) * 100.0
            rows.append(
                {
                    "timeframe": timeframe,
                    "cycle_id": cycle.get("cycle_id"),
                    "cycle_key": cycle.get("cycle_key"),
                    "cycle_type": cycle_type,
                    "sample_position": label,
                    "sample_progress": fraction,
                    "duration_candles": cycle.get("duration_candles"),
                    "parent_key": cycle.get("parent_key"),
                    "parent_type": _type_to_label(cycle.get("parent_type")),
                    "parent_progress_at_start": cycle.get("parent_progress_at_start"),
                    "parent_progress_at_end": cycle.get("parent_progress_at_end"),
                    "order_in_parent": cycle.get("order_in_parent"),
                    "total_siblings": cycle.get("total_siblings"),
                    "boundary_type": cycle.get("boundary_type"),
                    "n_up_4": cycle.get("n_up_4"),
                    "combo_4": cycle.get("combo_4"),
                    "ppo": float(ppo),
                    "ppo_hist": float(ppo_hist),
                    "move_to_cycle_end_pct": raw_move,
                    "move_to_cycle_end_signed_pct": raw_move * sign,
                }
            )

    cases = pd.DataFrame(rows)
    if cases.empty:
        return cases, pd.DataFrame()

    cases["ppo_regime"] = ppo_regime(cases["ppo"], cases["ppo_hist"])
    cases["ppo_quantile"] = _bucket_quantile(cases["ppo"], 5, "ppo")
    cases["ppo_hist_quantile"] = _bucket_quantile(cases["ppo_hist"], 5, "hist")
    cases["parent_progress_bucket"] = _progress_bucket(cases["parent_progress_at_start"])
    cases["sibling_position_bucket"] = _sibling_bucket(cases["order_in_parent"], cases["total_siblings"])

    summary = summarize_numeric(
        cases,
        group_cols=[
            "timeframe",
            "cycle_type",
            "sample_position",
            "ppo_regime",
            "parent_type",
            "parent_progress_bucket",
        ],
        metric_cols=["move_to_cycle_end_signed_pct", "move_to_cycle_end_pct"],
    )
    return cases, summary


def analyze_parent_child_cycles(timeframes: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for timeframe in timeframes:
        if timeframe == "1M":
            continue
        cycles = load_cycles(timeframe)
        keep = [
            "cycle_id",
            "cycle_key",
            "timeframe",
            "cycle_type",
            "duration_candles",
            "parent_key",
            "parent_type",
            "order_in_parent",
            "total_siblings",
            "parent_progress_at_start",
            "parent_progress_at_end",
            "boundary_type",
            "n_up_4",
            "combo_4",
            "opposite_child_ratio",
            "max_opposite_child_streak",
            "candle_data",
        ]
        subset = cycles[[column for column in keep if column in cycles.columns]].copy()
        subset["timeframe"] = timeframe
        rows.append(subset)

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    case_rows: list[dict[str, Any]] = []
    for _, row in combined.iterrows():
        candles = _cycle_candles(row.get("candle_data"))
        if not candles:
            continue
        first, last = candles[0], candles[-1]
        start_close = pd.to_numeric(pd.Series([first.get("close")]), errors="coerce").iloc[0]
        end_close = pd.to_numeric(pd.Series([last.get("close")]), errors="coerce").iloc[0]
        start_ppo = pd.to_numeric(pd.Series([first.get("ppo")]), errors="coerce").iloc[0]
        start_hist = pd.to_numeric(pd.Series([first.get("ppo_hist")]), errors="coerce").iloc[0]
        if pd.isna(start_close) or start_close == 0 or pd.isna(end_close) or pd.isna(start_ppo) or pd.isna(start_hist):
            continue
        cycle_type = str(row.get("cycle_type", "unknown")).lower()
        parent_type = _type_to_label(row.get("parent_type"))
        sign = 1.0 if cycle_type == "up" else -1.0 if cycle_type == "down" else np.nan
        parent_aligned = cycle_type == parent_type if parent_type != "unknown" else False
        case_rows.append(
            {
                "timeframe": row.get("timeframe"),
                "cycle_id": row.get("cycle_id"),
                "cycle_key": row.get("cycle_key"),
                "cycle_type": cycle_type,
                "parent_key": row.get("parent_key"),
                "parent_type": parent_type,
                "parent_aligned": parent_aligned,
                "duration_candles": row.get("duration_candles"),
                "order_in_parent": row.get("order_in_parent"),
                "total_siblings": row.get("total_siblings"),
                "parent_progress_at_start": row.get("parent_progress_at_start"),
                "parent_progress_at_end": row.get("parent_progress_at_end"),
                "boundary_type": row.get("boundary_type"),
                "n_up_4": row.get("n_up_4"),
                "combo_4": row.get("combo_4"),
                "opposite_child_ratio": row.get("opposite_child_ratio"),
                "max_opposite_child_streak": row.get("max_opposite_child_streak"),
                "start_ppo": float(start_ppo),
                "start_ppo_hist": float(start_hist),
                "cycle_price_change_pct": (end_close / start_close - 1.0) * 100.0,
                "cycle_price_change_signed_pct": (end_close / start_close - 1.0) * 100.0 * sign,
            }
        )

    cases = pd.DataFrame(case_rows)
    if cases.empty:
        return cases, pd.DataFrame()

    cases["ppo_regime"] = ppo_regime(cases["start_ppo"], cases["start_ppo_hist"])
    cases["ppo_quantile"] = _bucket_quantile(cases["start_ppo"], 5, "ppo")
    cases["ppo_hist_quantile"] = _bucket_quantile(cases["start_ppo_hist"], 5, "hist")
    cases["parent_progress_bucket"] = _progress_bucket(cases["parent_progress_at_start"])
    cases["sibling_position_bucket"] = _sibling_bucket(cases["order_in_parent"], cases["total_siblings"])

    summary = summarize_numeric(
        cases,
        group_cols=[
            "timeframe",
            "parent_type",
            "cycle_type",
            "parent_aligned",
            "parent_progress_bucket",
            "sibling_position_bucket",
            "ppo_regime",
        ],
        metric_cols=["cycle_price_change_signed_pct", "cycle_price_change_pct", "duration_candles"],
    )
    return cases, summary


def hierarchy_map_coverage() -> dict[str, Any]:
    path = _paths().cycle_dir / "cycle_hierarchy_map.json"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return {
        "exists": True,
        "path": str(path),
        "timeframes": {
            timeframe: len(nodes) if isinstance(nodes, dict) else None
            for timeframe, nodes in payload.items()
        },
    }


def build_report(
    market_summary: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    parent_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    lines = [
        "# PPO Position Movement Analysis",
        "",
        "## 분석 시나리오",
        "",
        "1. 원천 캔들 기준: PPO 부호, PPO hist 부호, 각 값의 분위 위치별 다음 N캔들 수익률을 시간대별로 집계.",
        "2. 사이클 내부 위치 기준: start/q25/mid/q75/end 지점의 PPO 상태에서 해당 사이클 종료까지 추가 움직임을 집계.",
        "3. parent 관계 기준: enriched cycle의 parent_key, parent_type, parent_progress, order_in_parent를 이용해 하위 사이클의 움직임을 집계.",
        "",
        "## 데이터",
        "",
        f"- timeframes: `{', '.join(metadata['timeframes'])}`",
        f"- horizons: `{', '.join(str(h) for h in metadata['horizons'])}`",
        f"- output_dir: `{metadata['output_dir']}`",
        "",
    ]

    def add_top_table(title: str, df: pd.DataFrame, metric: str, limit: int = 20) -> None:
        lines.extend([f"## {title}", ""])
        if df.empty or metric not in df.columns:
            lines.extend(["결과 없음", ""])
            return
        top = df.sort_values(["count", metric], ascending=[False, False]).head(limit)
        show_cols = [col for col in top.columns if col in {
            "timeframe",
            "ppo_regime",
            "ppo_quantile",
            "ppo_hist_quantile",
            "cycle_type",
            "parent_type",
            "parent_aligned",
            "sample_position",
            "parent_progress_bucket",
            "sibling_position_bucket",
            "count",
            metric,
        }]
        lines.append(top[show_cols].to_markdown(index=False))
        lines.append("")

    first_horizon = metadata["horizons"][0]
    add_top_table("캔들 PPO 위치별 다음 움직임 Top", market_summary, f"ret_fwd_{first_horizon}_avg")
    add_top_table("사이클 내부 위치별 종료까지 움직임 Top", cycle_summary, "move_to_cycle_end_signed_pct_avg")
    add_top_table("상하위 사이클 관계별 움직임 Top", parent_summary, "cycle_price_change_signed_pct_avg")

    if metadata.get("hierarchy_map"):
        lines.extend(["## Hierarchy Map", "", "```json"])
        lines.append(json.dumps(metadata["hierarchy_map"], ensure_ascii=False, indent=2))
        lines.extend(["```", ""])

    return "\n".join(lines)


def save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run(
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    bins: int = 5,
    include_cases: bool = False,
    include_hierarchy_map: bool = False,
) -> dict[str, Any]:
    paths = _paths()
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    market_cases: list[pd.DataFrame] = []
    market_summaries: list[pd.DataFrame] = []
    cycle_cases: list[pd.DataFrame] = []
    cycle_summaries: list[pd.DataFrame] = []

    for timeframe in timeframes:
        cases, summary = analyze_market_timeframe(timeframe, horizons=horizons, bins=bins)
        market_cases.append(cases)
        market_summaries.append(summary)

        cycle_case, cycle_summary = analyze_cycle_positions(timeframe)
        cycle_cases.append(cycle_case)
        cycle_summaries.append(cycle_summary)

    parent_cases, parent_summary = analyze_parent_child_cycles(timeframes)

    market_summary = pd.concat(market_summaries, ignore_index=True) if market_summaries else pd.DataFrame()
    cycle_summary = pd.concat(cycle_summaries, ignore_index=True) if cycle_summaries else pd.DataFrame()
    market_case_df = pd.concat(market_cases, ignore_index=True) if market_cases else pd.DataFrame()
    cycle_case_df = pd.concat(cycle_cases, ignore_index=True) if cycle_cases else pd.DataFrame()

    save_frame(market_summary, paths.output_dir / "market_ppo_position_summary.csv")
    save_frame(cycle_summary, paths.output_dir / "cycle_internal_position_summary.csv")
    save_frame(parent_summary, paths.output_dir / "parent_child_ppo_relation_summary.csv")

    if include_cases:
        save_frame(market_case_df, paths.output_dir / "market_ppo_position_cases.csv")
        save_frame(cycle_case_df, paths.output_dir / "cycle_internal_position_cases.csv")
        save_frame(parent_cases, paths.output_dir / "parent_child_ppo_relation_cases.csv")

    metadata = {
        "timeframes": list(timeframes),
        "horizons": list(horizons),
        "bins": bins,
        "output_dir": str(paths.output_dir),
        "market_case_count": int(len(market_case_df)),
        "cycle_position_case_count": int(len(cycle_case_df)),
        "parent_child_case_count": int(len(parent_cases)),
        "hierarchy_map": hierarchy_map_coverage() if include_hierarchy_map else None,
    }
    (paths.output_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (paths.output_dir / "report.md").write_text(
        build_report(market_summary, cycle_summary, parent_summary, metadata),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BTC movement by PPO/PPO hist value position and cycle hierarchy.")
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), choices=list(TIMEFRAMES))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--include-cases", action="store_true", help="Save row-level case CSV files.")
    parser.add_argument("--include-hierarchy-map", action="store_true", help="Load cycle_hierarchy_map.json and include coverage metadata.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(
        timeframes=tuple(args.timeframes),
        horizons=tuple(args.horizons),
        bins=args.bins,
        include_cases=args.include_cases,
        include_hierarchy_map=args.include_hierarchy_map,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
