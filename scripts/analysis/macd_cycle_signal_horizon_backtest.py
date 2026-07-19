"""MACD cycle signal horizon backtest.

Treats each MACD cycle start as a directional signal:

  - cycle_type=up   -> long
  - cycle_type=down -> short

For each signal, enter at the signal candle close and exit after N same-timeframe
candles. The script reports win probability, return distribution, MFE/MAE, and
simple TP/SL hit statistics by timeframe, direction, and holding horizon.
"""

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

from src.common.paths import PROJECT_PATHS  # noqa: E402


TIMEFRAMES = ("5m", "15m", "30m", "1h", "4h", "1d", "1w")
DEFAULT_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d", "1w")
DEFAULT_HORIZONS = (1, 2, 3, 5, 8, 13, 21, 34)
DEFAULT_RISK_LEVELS = (0.5, 1.0, 2.0, 3.0)


def output_dir() -> Path:
    path = PROJECT_PATHS.outputs_root / "analysis_results" / "macd_cycle_signal_horizon_backtest"
    path.mkdir(parents=True, exist_ok=True)
    return path


def market_path(timeframe: str) -> Path:
    candidates = [
        PROJECT_PATHS.base_data_dir / f"BTCUSD_{timeframe}.csv",
        PROJECT_PATHS.base_data_dir / f"BTCUSDT_{timeframe}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing market CSV for {timeframe}: tried {candidates}")


def cycle_path(timeframe: str) -> Path:
    candidates = [
        PROJECT_PATHS.asset_cycle_dir("btc") / f"cycles_{timeframe}.parquet",
        PROJECT_PATHS.cycle_structured_dir / "btc" / f"cycles_{timeframe}.parquet",
        PROJECT_PATHS.cycle_structured_dir / f"cycles_{timeframe}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing cycle parquet for {timeframe}: tried {candidates}")


def read_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def load_market(timeframe: str) -> pd.DataFrame:
    path = market_path(timeframe)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    ts_col = next((col for col in ("date", "timestamp", "open_time") if col in header), None)
    if ts_col is None:
        raise ValueError(f"{path} has no timestamp column")

    wanted = {
        ts_col,
        "open",
        "high",
        "low",
        "close",
        "macd",
        "macd_signal",
        "macd_hist",
        "ppo",
        "ppo_hist",
        "rsi",
    }
    df = pd.read_csv(path, usecols=[col for col in header if col in wanted]).rename(columns={ts_col: "timestamp"})
    df["timestamp"] = read_timestamp(df["timestamp"])
    for col in ("open", "high", "low", "close", "macd", "macd_signal", "macd_hist", "ppo", "ppo_hist", "rsi"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def load_cycles(timeframe: str) -> pd.DataFrame:
    df = pd.read_parquet(cycle_path(timeframe), columns=["cycle_id", "start_date", "end_date", "cycle_type", "duration_candles"]).copy()
    df["start_date"] = read_timestamp(df["start_date"])
    df["end_date"] = read_timestamp(df["end_date"])
    df["cycle_type"] = df["cycle_type"].astype(str).str.lower()
    df["direction"] = df["cycle_type"].map({"up": "long", "down": "short"})
    df["direction_sign"] = df["cycle_type"].map({"up": 1.0, "down": -1.0})
    df["duration_candles"] = pd.to_numeric(df["duration_candles"], errors="coerce")
    return df.dropna(subset=["start_date", "direction", "direction_sign"]).sort_values("start_date").reset_index(drop=True)


def first_hit_outcome(
    highs: np.ndarray,
    lows: np.ndarray,
    entry: float,
    sign: float,
    tp_pct: float,
    sl_pct: float,
) -> str:
    if entry <= 0 or len(highs) == 0:
        return "none"
    if sign > 0:
        tp_price = entry * (1.0 + tp_pct / 100.0)
        sl_price = entry * (1.0 - sl_pct / 100.0)
        for high, low in zip(highs, lows):
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
            if hit_tp and hit_sl:
                return "both_same_candle"
            if hit_tp:
                return "tp_first"
            if hit_sl:
                return "sl_first"
    else:
        tp_price = entry * (1.0 - tp_pct / 100.0)
        sl_price = entry * (1.0 + sl_pct / 100.0)
        for high, low in zip(highs, lows):
            hit_tp = low <= tp_price
            hit_sl = high >= sl_price
            if hit_tp and hit_sl:
                return "both_same_candle"
            if hit_tp:
                return "tp_first"
            if hit_sl:
                return "sl_first"
    return "none"


def build_timeframe_ledger(timeframe: str, horizons: tuple[int, ...], risk_levels: tuple[float, ...]) -> pd.DataFrame:
    market = load_market(timeframe)
    cycles = load_cycles(timeframe)
    if market.empty or cycles.empty:
        return pd.DataFrame()

    times = market["timestamp"].to_numpy(dtype="datetime64[ns]")
    opens = market["open"].to_numpy(dtype="float64")
    highs = market["high"].to_numpy(dtype="float64")
    lows = market["low"].to_numpy(dtype="float64")
    closes = market["close"].to_numpy(dtype="float64")

    indicator_cols = [col for col in ("macd", "macd_signal", "macd_hist", "ppo", "ppo_hist", "rsi") if col in market.columns]
    indicator_arrays = {col: market[col].to_numpy(dtype="float64") for col in indicator_cols}

    rows: list[dict[str, Any]] = []
    max_horizon = max(horizons)
    for cycle in cycles.itertuples(index=False):
        start = np.datetime64(cycle.start_date.to_datetime64())
        entry_idx = int(np.searchsorted(times, start, side="left"))
        if entry_idx >= len(market):
            continue
        if times[entry_idx] != start:
            nearest = int(np.searchsorted(times, start, side="right") - 1)
            if nearest < 0:
                continue
            entry_idx = nearest
        if entry_idx + max_horizon >= len(market):
            continue

        entry_close = closes[entry_idx]
        if not np.isfinite(entry_close) or entry_close <= 0:
            continue

        sign = float(cycle.direction_sign)
        base = {
            "timeframe": timeframe,
            "cycle_id": cycle.cycle_id,
            "cycle_type": cycle.cycle_type,
            "direction": cycle.direction,
            "direction_sign": sign,
            "signal_time": pd.Timestamp(times[entry_idx]),
            "cycle_start_date": cycle.start_date,
            "cycle_end_date": cycle.end_date,
            "cycle_duration_candles": cycle.duration_candles,
            "entry_index": entry_idx,
            "entry_open": opens[entry_idx],
            "entry_close": entry_close,
        }
        for col, values in indicator_arrays.items():
            base[f"entry_{col}"] = values[entry_idx]

        for horizon in horizons:
            exit_idx = entry_idx + horizon
            window_high = highs[entry_idx + 1 : exit_idx + 1]
            window_low = lows[entry_idx + 1 : exit_idx + 1]
            if len(window_high) == 0 or len(window_low) == 0:
                continue

            exit_close = closes[exit_idx]
            raw_return = (exit_close / entry_close - 1.0) * 100.0
            signed_return = raw_return * sign
            if sign > 0:
                mfe = (np.nanmax(window_high) / entry_close - 1.0) * 100.0
                mae = (np.nanmin(window_low) / entry_close - 1.0) * 100.0
            else:
                mfe = (1.0 - np.nanmin(window_low) / entry_close) * 100.0
                mae = (1.0 - np.nanmax(window_high) / entry_close) * 100.0

            row = {
                **base,
                "horizon_candles": horizon,
                "exit_time": pd.Timestamp(times[exit_idx]),
                "exit_close": exit_close,
                "raw_return_pct": raw_return,
                "strategy_return_pct": signed_return,
                "is_win": signed_return > 0,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "adverse_excursion_pct": abs(min(mae, 0.0)),
                "favorable_excursion_pct": max(mfe, 0.0),
            }
            for level in risk_levels:
                level_key = f"{level:g}".replace(".", "p")
                row[f"hit_tp_{level_key}_pct"] = row["favorable_excursion_pct"] >= level
                row[f"hit_sl_{level_key}_pct"] = row["adverse_excursion_pct"] >= level
                row[f"first_hit_{level_key}_pct"] = first_hit_outcome(window_high, window_low, entry_close, sign, level, level)
            rows.append(row)

    return pd.DataFrame(rows)


def profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    if losses == 0:
        return np.inf if wins > 0 else np.nan
    return float(wins / losses)


def summarize_group(group: pd.DataFrame, risk_levels: tuple[float, ...]) -> pd.Series:
    r = pd.to_numeric(group["strategy_return_pct"], errors="coerce").dropna()
    if r.empty:
        return pd.Series({"n": 0})
    row: dict[str, Any] = {
        "n": int(len(r)),
        "win_rate_pct": float((r > 0).mean() * 100.0),
        "loss_rate_pct": float((r < 0).mean() * 100.0),
        "avg_return_pct": float(r.mean()),
        "median_return_pct": float(r.median()),
        "profit_factor": profit_factor(r),
        "expectancy_pct": float(r.mean()),
        "std_return_pct": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        "p05_return_pct": float(r.quantile(0.05)),
        "p10_return_pct": float(r.quantile(0.10)),
        "p25_return_pct": float(r.quantile(0.25)),
        "p75_return_pct": float(r.quantile(0.75)),
        "p90_return_pct": float(r.quantile(0.90)),
        "p95_return_pct": float(r.quantile(0.95)),
        "worst_return_pct": float(r.min()),
        "best_return_pct": float(r.max()),
        "avg_mfe_pct": float(pd.to_numeric(group["mfe_pct"], errors="coerce").mean()),
        "avg_mae_pct": float(pd.to_numeric(group["mae_pct"], errors="coerce").mean()),
        "p90_adverse_excursion_pct": float(pd.to_numeric(group["adverse_excursion_pct"], errors="coerce").quantile(0.90)),
        "p95_adverse_excursion_pct": float(pd.to_numeric(group["adverse_excursion_pct"], errors="coerce").quantile(0.95)),
    }
    for level in risk_levels:
        level_key = f"{level:g}".replace(".", "p")
        row[f"tp_{level_key}_hit_rate_pct"] = float(group[f"hit_tp_{level_key}_pct"].mean() * 100.0)
        row[f"sl_{level_key}_hit_rate_pct"] = float(group[f"hit_sl_{level_key}_pct"].mean() * 100.0)
        first = group[f"first_hit_{level_key}_pct"].value_counts(normalize=True) * 100.0
        row[f"tp_{level_key}_first_rate_pct"] = float(first.get("tp_first", 0.0))
        row[f"sl_{level_key}_first_rate_pct"] = float(first.get("sl_first", 0.0))
        row[f"both_{level_key}_same_candle_rate_pct"] = float(first.get("both_same_candle", 0.0))
    return pd.Series(row)


def summarize_ledger(ledger: pd.DataFrame, risk_levels: tuple[float, ...]) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    summary = (
        ledger.groupby(["timeframe", "direction", "cycle_type", "horizon_candles"], dropna=False, observed=True)
        .apply(lambda group: summarize_group(group, risk_levels))
        .reset_index()
    )
    return summary.sort_values(["timeframe", "direction", "horizon_candles"]).reset_index(drop=True)


def build_best_horizons(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    ranked = summary.copy()
    ranked["score"] = (
        ranked["avg_return_pct"].fillna(-999.0)
        + ranked["win_rate_pct"].fillna(0.0) / 100.0
        - ranked["p95_adverse_excursion_pct"].fillna(999.0) * 0.25
    )
    ranked = ranked.sort_values(["timeframe", "direction", "score", "n"], ascending=[True, True, False, False])
    return ranked.groupby(["timeframe", "direction"], as_index=False, observed=True).head(3).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df.head(limit).copy() if limit else df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    return show[columns].to_markdown(index=False)


def build_report(summary: pd.DataFrame, best: pd.DataFrame, metadata: dict[str, Any]) -> str:
    lines = [
        "# MACD Cycle Signal Horizon Backtest",
        "",
        "## Setup",
        "",
        f"- Timeframes: `{', '.join(metadata['timeframes'])}`",
        f"- Horizons: `{', '.join(str(item) for item in metadata['horizons'])}` candles",
        f"- Risk levels: `{', '.join(str(item) for item in metadata['risk_levels_pct'])}`%",
        f"- Entry: signal candle close at MACD cycle start",
        f"- Exit: close after N candles",
        f"- Output dir: `{metadata['output_dir']}`",
        "",
        "## Best Horizon Candidates",
        "",
    ]
    best_cols = [
        "timeframe",
        "direction",
        "horizon_candles",
        "n",
        "win_rate_pct",
        "avg_return_pct",
        "profit_factor",
        "p95_adverse_excursion_pct",
        "score",
    ]
    lines.append(markdown_table(best, [col for col in best_cols if col in best.columns], limit=50))
    lines.extend(["", "## Full Direction Summary", ""])
    summary_cols = [
        "timeframe",
        "direction",
        "horizon_candles",
        "n",
        "win_rate_pct",
        "avg_return_pct",
        "median_return_pct",
        "profit_factor",
        "worst_return_pct",
        "p95_adverse_excursion_pct",
        "avg_mfe_pct",
        "avg_mae_pct",
    ]
    lines.append(markdown_table(summary, [col for col in summary_cols if col in summary.columns], limit=120))
    lines.extend([
        "",
        "## Notes",
        "",
        "- `strategy_return_pct` is signed: long returns use normal price change, short returns are inverted.",
        "- `mae_pct` is adverse and usually negative; `p95_adverse_excursion_pct` is easier for stop sizing.",
        "- `first_hit_*` assumes candle OHLC only; if TP and SL touch in the same candle, order is marked ambiguous.",
        "",
    ])
    return "\n".join(lines)


def run(
    timeframes: tuple[str, ...],
    horizons: tuple[int, ...],
    risk_levels: tuple[float, ...],
    include_ledger_csv: bool,
) -> dict[str, Any]:
    out = output_dir()
    ledgers: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for timeframe in timeframes:
        try:
            ledger = build_timeframe_ledger(timeframe, horizons, risk_levels)
        except Exception as exc:  # keep other TFs useful even if one file is missing
            errors[timeframe] = str(exc)
            continue
        ledgers.append(ledger)

    ledger_all = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    summary = summarize_ledger(ledger_all, risk_levels)
    best = build_best_horizons(summary)

    ledger_path = out / "signal_horizon_trade_ledger.parquet"
    summary_path = out / "signal_horizon_summary.csv"
    best_path = out / "best_horizon_candidates.csv"
    report_path = out / "report.md"
    metadata_path = out / "metadata.json"

    ledger_all.to_parquet(ledger_path, index=False)
    if include_ledger_csv:
        ledger_all.to_csv(out / "signal_horizon_trade_ledger.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    best.to_csv(best_path, index=False, encoding="utf-8-sig")

    metadata = {
        "timeframes": list(timeframes),
        "horizons": list(horizons),
        "risk_levels_pct": list(risk_levels),
        "output_dir": str(out),
        "ledger_rows": int(len(ledger_all)),
        "summary_rows": int(len(summary)),
        "errors": errors,
        "files": {
            "ledger_parquet": str(ledger_path),
            "summary_csv": str(summary_path),
            "best_horizon_candidates_csv": str(best_path),
            "report_md": str(report_path),
        },
    }
    report_path.write_text(build_report(summary, best, metadata), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest MACD cycle up/down signals by fixed candle holding horizons.")
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), choices=list(TIMEFRAMES))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--risk-levels", nargs="+", type=float, default=list(DEFAULT_RISK_LEVELS), help="Percent levels for TP/SL hit stats.")
    parser.add_argument("--include-ledger-csv", action="store_true", help="Also save the full ledger as CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run(
        timeframes=tuple(args.timeframes),
        horizons=tuple(sorted(set(args.horizons))),
        risk_levels=tuple(sorted(set(args.risk_levels))),
        include_ledger_csv=args.include_ledger_csv,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
