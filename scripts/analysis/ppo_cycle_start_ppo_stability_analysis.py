"""Cycle starting-PPO stability analysis.

For every cycle in each TF, treats the cycle as a single round-trip trade:

  entry = close of cycle's first candle (long if cycle_type=UP, short if DOWN)
  exit  = close of NEXT same-TF cycle's first candle
  return_pct = (exit/entry - 1) * 100 * sign(cycle_type)

Aggregates trade quality by TF x direction x start_ppo bin so we can verify
two intuitions:

  - UP cycles starting at low PPO are reliably long-profitable
  - DOWN cycles starting at high PPO are reliably short-profitable

Three binning schemes are produced:

  - decile (1..10) computed across all cycles within the TF
  - coarse bin (bottom10/bottom20/mid/top20/top10)
  - threshold bin using fixed PPO levels per TF (deep_negative/negative/positive/deep_positive)

In addition, a parent->child roll-up uses cycle_hierarchy_map.json to ask:
"when the upper TF cycle started at PPO bin X, how do the inner cycles of
each lower TF behave?".

Outputs (all under outputs/analysis_results/ppo_cycle_start_ppo_stability_analysis/):

  74_per_tf_cycle_start_ppo_stability.csv          decile + coarse-bin grouping
  75_per_tf_cycle_start_ppo_threshold_bins.csv     fixed-threshold grouping
  76_hierarchy_parent_start_ppo_child_stability.csv parent_tf x parent_bin x child_tf x child_dir x child_bin
  77_cycle_trade_ledger.parquet                    full per-cycle ledger
  PPO_cycle_start_ppo_stability_report.md          findings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import PROJECT_PATHS  # noqa: E402


TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d", "1w")
TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
LOW_SAMPLE_N = 30
MEDIUM_SAMPLE_N = 100
RELIABLE_SAMPLE_N = 1000

# Same conventions as ppo_reversal_candidate_backtest.EXTREME_THRESHOLDS
# (long = deep negative threshold, short = deep positive threshold)
EXTREME_THRESHOLDS = {
    "5m": {"long": -0.20, "short": 0.20},
    "15m": {"long": -0.42, "short": 0.42},
    "1h": {"long": -0.90, "short": 0.93},
    "4h": {"long": -2.03, "short": 2.06},
    "1d": {"long": -4.07, "short": 5.43},
    "1w": {"long": -8.50, "short": 9.00},
}


# ---------------------------------------------------------------------------
# Path resolution and IO
# ---------------------------------------------------------------------------


def _market_path(tf: str) -> Path:
    candidates = [
        PROJECT_PATHS.raw_market_dir / f"BTCUSD_{tf}.csv",
        PROJECT_PATHS.raw_market_dir / f"BTCUSDT_{tf}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing raw candle file for {tf}: tried {candidates}")


def _cycle_path(tf: str) -> Path:
    candidates = [
        PROJECT_PATHS.asset_cycle_dir("btc") / f"cycles_{tf}.parquet",
        PROJECT_PATHS.cycle_structured_dir / "btc" / f"cycles_{tf}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing cycle parquet for {tf}: tried {candidates}")


def _hierarchy_path() -> Path:
    return PROJECT_PATHS.asset_cycle_dir("btc") / "cycle_hierarchy_map.json"


def output_dir() -> Path:
    target = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_cycle_start_ppo_stability_analysis"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _read_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def load_candles(tf: str) -> pd.DataFrame:
    path = _market_path(tf)
    cols_avail = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in cols_avail if c in {"date", "timestamp", "open_time", "open", "high", "low", "close", "ppo", "ppo_hist"}]
    df = pd.read_csv(path, usecols=usecols).copy()
    ts_col = next((c for c in ("timestamp", "open_time", "date") if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"{path} has no timestamp column")
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _read_ts(df["timestamp"])
    for col in ("open", "high", "low", "close", "ppo", "ppo_hist"):
        if col not in df.columns:
            raise ValueError(f"{path} missing column {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "ppo"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def load_cycles(tf: str) -> pd.DataFrame:
    df = pd.read_parquet(_cycle_path(tf)).copy()
    df["start_date"] = _read_ts(df["start_date"])
    df["end_date"] = _read_ts(df["end_date"])
    df = df.dropna(subset=["start_date", "end_date"]).sort_values("start_date").reset_index(drop=True)
    df["cycle_type_norm"] = df["cycle_type"].astype(str).str.lower().map({"up": "UP", "down": "DOWN"}).fillna("UNKNOWN")
    df["direction_sign"] = df["cycle_type_norm"].map({"UP": 1, "DOWN": -1}).fillna(0).astype(int)
    df["duration_candles"] = pd.to_numeric(df.get("duration_candles"), errors="coerce")
    return df[["cycle_id", "cycle_type_norm", "direction_sign", "start_date", "end_date", "duration_candles"]].rename(
        columns={"cycle_type_norm": "cycle_type"}
    )


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def _bin_from_decile(decile: pd.Series) -> pd.Series:
    dec = pd.to_numeric(decile, errors="coerce")
    return pd.Series(
        np.select(
            [dec <= 1, dec <= 2, dec >= 10, dec >= 9],
            ["bottom10", "bottom20", "top10", "top20"],
            default="mid",
        ),
        index=decile.index,
    )


def _threshold_bin(ppo: pd.Series, tf: str) -> pd.Series:
    thr = EXTREME_THRESHOLDS.get(tf)
    if thr is None:
        return pd.Series(["unknown"] * len(ppo), index=ppo.index)
    bins = [-np.inf, thr["long"], 0.0, thr["short"], np.inf]
    labels = ["deep_negative", "negative", "positive", "deep_positive"]
    return pd.cut(ppo, bins=bins, labels=labels, include_lowest=True).astype(str)


# ---------------------------------------------------------------------------
# Cycle trade ledger
# ---------------------------------------------------------------------------


def build_cycle_ledger(tf: str) -> pd.DataFrame:
    cycles = load_cycles(tf)
    candles = load_candles(tf)
    if cycles.empty or candles.empty:
        return pd.DataFrame()
    cycles = cycles.sort_values("start_date").reset_index(drop=True)
    cycles["next_start_date"] = cycles["start_date"].shift(-1)

    cand_keep = candles[["timestamp", "open", "high", "low", "close", "ppo", "ppo_hist"]].copy()
    tol = pd.Timedelta(seconds=TF_SECONDS[tf])

    entry = pd.merge_asof(
        cycles.sort_values("start_date"),
        cand_keep.rename(columns={
            "open": "entry_open",
            "high": "entry_high",
            "low": "entry_low",
            "close": "entry_close",
            "ppo": "entry_ppo",
            "ppo_hist": "entry_hist",
        }).sort_values("timestamp"),
        left_on="start_date",
        right_on="timestamp",
        direction="nearest",
        tolerance=tol,
    ).drop(columns=["timestamp"])

    # The last cycle has no next_start_date; split, merge only the rest, then concat.
    has_next = entry["next_start_date"].notna()
    with_next = entry.loc[has_next].copy()
    no_next = entry.loc[~has_next].copy()
    if not with_next.empty:
        merged = pd.merge_asof(
            with_next.sort_values("next_start_date"),
            cand_keep[["timestamp", "open", "close", "ppo"]].rename(
                columns={"open": "exit_open", "close": "exit_close", "ppo": "exit_ppo"}
            ).sort_values("timestamp"),
            left_on="next_start_date",
            right_on="timestamp",
            direction="nearest",
            tolerance=tol,
        ).drop(columns=["timestamp"])
    else:
        merged = with_next
    for col in ("exit_open", "exit_close", "exit_ppo"):
        if col not in no_next.columns:
            no_next[col] = np.nan
    out = pd.concat([merged, no_next], ignore_index=True).sort_values("start_date").reset_index(drop=True)

    # MFE / MAE within the cycle's hold window [start_date, next_start_date)
    times = candles["timestamp"].to_numpy(dtype="datetime64[ns]")
    highs = candles["high"].to_numpy(dtype="float64")
    lows = candles["low"].to_numpy(dtype="float64")
    closes = candles["close"].to_numpy(dtype="float64")
    starts = out["start_date"].to_numpy(dtype="datetime64[ns]")
    nexts = out["next_start_date"].to_numpy(dtype="datetime64[ns]")
    win_high = np.full(len(out), np.nan)
    win_low = np.full(len(out), np.nan)
    bars_in_window = np.zeros(len(out), dtype=np.int64)
    for i in range(len(out)):
        n = nexts[i]
        if pd.isna(n):
            continue
        i_lo = int(np.searchsorted(times, starts[i], side="left"))
        i_hi = int(np.searchsorted(times, n, side="left"))
        if i_lo >= i_hi:
            continue
        win_high[i] = highs[i_lo:i_hi].max()
        win_low[i] = lows[i_lo:i_hi].min()
        bars_in_window[i] = i_hi - i_lo
    out["cycle_window_high"] = win_high
    out["cycle_window_low"] = win_low
    out["bars_in_window"] = bars_in_window

    sign = out["direction_sign"].astype(float)
    entry_close = out["entry_close"].astype(float)
    out["direction"] = np.where(sign == 1, "long", np.where(sign == -1, "short", "unknown"))
    out["trade_return_pct"] = (out["exit_close"] / entry_close - 1.0) * 100.0 * sign
    # MFE / MAE relative to entry, signed by direction
    long_mfe = (out["cycle_window_high"] / entry_close - 1.0) * 100.0
    long_mae = (out["cycle_window_low"] / entry_close - 1.0) * 100.0
    short_mfe = (1.0 - out["cycle_window_low"] / entry_close) * 100.0
    short_mae = (1.0 - out["cycle_window_high"] / entry_close) * 100.0
    out["mfe_pct"] = np.where(sign == 1, long_mfe, np.where(sign == -1, short_mfe, np.nan))
    out["mae_pct"] = np.where(sign == 1, long_mae, np.where(sign == -1, short_mae, np.nan))

    out["timeframe"] = tf

    # Bins computed within this TF's cycle population
    ranked = pd.to_numeric(out["entry_ppo"], errors="coerce").rank(method="first")
    try:
        out["entry_ppo_decile"] = pd.qcut(ranked, 10, labels=False).astype("float") + 1
    except ValueError:
        out["entry_ppo_decile"] = np.nan
    out["entry_ppo_bin"] = _bin_from_decile(out["entry_ppo_decile"])
    out["entry_ppo_threshold_bin"] = _threshold_bin(pd.to_numeric(out["entry_ppo"], errors="coerce"), tf)

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _sample_class(n: int) -> str:
    if n < LOW_SAMPLE_N:
        return "very_low"
    if n < MEDIUM_SAMPLE_N:
        return "low_sample"
    if n < RELIABLE_SAMPLE_N:
        return "medium_sample"
    return "reliable_sample"


def _stability_metrics(g: pd.DataFrame) -> pd.Series:
    r = pd.to_numeric(g["trade_return_pct"], errors="coerce").dropna()
    n = len(r)
    if n == 0:
        return pd.Series({"n": 0, "sample_class": "empty"})
    pos = r[r > 0]
    neg = r[r < 0]
    pos_sum = pos.sum()
    neg_abs_sum = neg.abs().sum()
    pf = (pos_sum / neg_abs_sum) if neg_abs_sum > 0 else np.inf
    avg = r.mean()
    std = r.std(ddof=1) if n > 1 else 0.0
    cv = (std / abs(avg)) if avg != 0 else np.nan
    sharpe_like = (avg / std) if std > 0 else np.nan
    return pd.Series({
        "n": n,
        "sample_class": _sample_class(n),
        "win_rate": (r > 0).mean() * 100.0,
        "avg_return_pct": avg,
        "median_return_pct": r.median(),
        "std_return_pct": std,
        "cv_volatility": cv,
        "sharpe_like": sharpe_like,
        "profit_factor": pf,
        "p10_return_pct": r.quantile(0.10),
        "p25_return_pct": r.quantile(0.25),
        "p75_return_pct": r.quantile(0.75),
        "p90_return_pct": r.quantile(0.90),
        "worst_return_pct": r.min(),
        "best_return_pct": r.max(),
        "avg_mfe_pct": pd.to_numeric(g["mfe_pct"], errors="coerce").mean(),
        "avg_mae_pct": pd.to_numeric(g["mae_pct"], errors="coerce").mean(),
        "avg_holding_bars": pd.to_numeric(g["bars_in_window"], errors="coerce").mean(),
    })


def write_74(ledger: pd.DataFrame, out: Path) -> Path:
    df = ledger.dropna(subset=["entry_ppo", "trade_return_pct"]).copy()

    # decile-level grouping (more granular)
    decile_summary = (
        df.groupby(["timeframe", "direction", "entry_ppo_decile"], observed=True, dropna=False)
        .apply(_stability_metrics)
        .reset_index()
    )
    decile_summary["bin_scheme"] = "decile"
    decile_summary = decile_summary.rename(columns={"entry_ppo_decile": "ppo_bin_label"})
    decile_summary["ppo_bin_label"] = decile_summary["ppo_bin_label"].astype(str)

    # coarse bin grouping
    coarse_summary = (
        df.groupby(["timeframe", "direction", "entry_ppo_bin"], observed=True, dropna=False)
        .apply(_stability_metrics)
        .reset_index()
        .rename(columns={"entry_ppo_bin": "ppo_bin_label"})
    )
    coarse_summary["bin_scheme"] = "coarse"

    # combined output
    result = pd.concat([coarse_summary, decile_summary], ignore_index=True)
    cols = ["timeframe", "direction", "bin_scheme", "ppo_bin_label"] + [
        c for c in result.columns if c not in {"timeframe", "direction", "bin_scheme", "ppo_bin_label"}
    ]
    result = result[cols].sort_values(["timeframe", "direction", "bin_scheme", "ppo_bin_label"])
    path = out / "74_per_tf_cycle_start_ppo_stability.csv"
    result.to_csv(path, index=False)
    return path


def write_75(ledger: pd.DataFrame, out: Path) -> Path:
    df = ledger.dropna(subset=["entry_ppo", "trade_return_pct"]).copy()
    summary = (
        df.groupby(["timeframe", "direction", "entry_ppo_threshold_bin"], observed=True, dropna=False)
        .apply(_stability_metrics)
        .reset_index()
        .rename(columns={"entry_ppo_threshold_bin": "ppo_threshold_bin"})
    )
    # Add the actual threshold values for clarity
    summary["long_threshold_pct"] = summary["timeframe"].map(lambda t: EXTREME_THRESHOLDS.get(t, {}).get("long"))
    summary["short_threshold_pct"] = summary["timeframe"].map(lambda t: EXTREME_THRESHOLDS.get(t, {}).get("short"))
    summary = summary.sort_values(["timeframe", "direction", "ppo_threshold_bin"])
    path = out / "75_per_tf_cycle_start_ppo_threshold_bins.csv"
    summary.to_csv(path, index=False)
    return path


def write_76_hierarchy(ledger: pd.DataFrame, out: Path) -> Path:
    """For each parent TF cycle, look up child cycle IDs from the hierarchy
    JSON, join to the child ledger, and aggregate by parent_threshold_bin x
    child_tf x child_direction x child_threshold_bin."""
    hierarchy_path = _hierarchy_path()
    if not hierarchy_path.exists():
        path = out / "76_hierarchy_parent_start_ppo_child_stability.csv"
        pd.DataFrame().to_csv(path, index=False)
        return path
    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))

    # ledger indexed for fast lookup
    by_tf_id: dict[tuple[str, str], pd.Series] = {}
    for row in ledger.itertuples(index=False):
        by_tf_id[(row.timeframe, str(row.cycle_id))] = row

    rows: list[dict] = []
    parent_tfs = ("1w", "1d", "4h", "1h")
    child_map = {
        "1w": ("1d", "4h", "1h"),
        "1d": ("4h", "1h", "15m"),
        "4h": ("1h", "15m", "5m"),
        "1h": ("15m", "5m"),
    }
    for parent_tf in parent_tfs:
        parent_dict = hierarchy.get(parent_tf, {})
        for parent_id, parent_meta in parent_dict.items():
            parent_row = by_tf_id.get((parent_tf, str(parent_id)))
            if parent_row is None:
                continue
            parent_dir = parent_row.direction
            parent_bin = parent_row.entry_ppo_threshold_bin
            parent_ppo = parent_row.entry_ppo
            child_ids_by_tf = parent_meta.get("child_cycle_ids", {}) or {}
            for child_tf in child_map[parent_tf]:
                ids = child_ids_by_tf.get(child_tf, []) or []
                for cid in ids:
                    child_row = by_tf_id.get((child_tf, str(cid)))
                    if child_row is None:
                        continue
                    rows.append({
                        "parent_tf": parent_tf,
                        "parent_direction": parent_dir,
                        "parent_threshold_bin": parent_bin,
                        "parent_ppo": parent_ppo,
                        "child_tf": child_tf,
                        "child_id": str(cid),
                        "child_direction": child_row.direction,
                        "child_threshold_bin": child_row.entry_ppo_threshold_bin,
                        "child_entry_ppo": child_row.entry_ppo,
                        "child_trade_return_pct": child_row.trade_return_pct,
                        "child_mfe_pct": child_row.mfe_pct,
                        "child_mae_pct": child_row.mae_pct,
                        "child_bars_in_window": child_row.bars_in_window,
                    })
    if not rows:
        path = out / "76_hierarchy_parent_start_ppo_child_stability.csv"
        pd.DataFrame().to_csv(path, index=False)
        return path
    long_df = pd.DataFrame(rows)

    summary = (
        long_df.groupby(
            ["parent_tf", "parent_direction", "parent_threshold_bin", "child_tf", "child_direction", "child_threshold_bin"],
            observed=True,
            dropna=False,
        )
        .apply(lambda g: _stability_metrics(g.rename(columns={
            "child_trade_return_pct": "trade_return_pct",
            "child_mfe_pct": "mfe_pct",
            "child_mae_pct": "mae_pct",
            "child_bars_in_window": "bars_in_window",
        })))
        .reset_index()
        .sort_values(["parent_tf", "child_tf", "parent_threshold_bin", "child_direction", "child_threshold_bin"])
    )
    path = out / "76_hierarchy_parent_start_ppo_child_stability.csv"
    summary.to_csv(path, index=False)

    # also dump the long-form join for debugging / future analyses
    long_df.to_parquet(out / "76_hierarchy_parent_child_long.parquet", index=False)
    return path


def write_77_ledger(ledger: pd.DataFrame, out: Path) -> Path:
    keep = [
        "timeframe", "cycle_id", "cycle_type", "direction", "direction_sign",
        "start_date", "end_date", "next_start_date", "duration_candles",
        "bars_in_window",
        "entry_open", "entry_high", "entry_low", "entry_close", "entry_ppo", "entry_hist",
        "exit_open", "exit_close", "exit_ppo",
        "cycle_window_high", "cycle_window_low",
        "trade_return_pct", "mfe_pct", "mae_pct",
        "entry_ppo_decile", "entry_ppo_bin", "entry_ppo_threshold_bin",
    ]
    keep = [c for c in keep if c in ledger.columns]
    path = out / "77_cycle_trade_ledger.parquet"
    ledger[keep].to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframes", nargs="*", default=list(TIMEFRAMES))
    args = parser.parse_args()

    out = output_dir()
    print(f"[cycle-start-ppo] output_dir = {out}", flush=True)

    frames: list[pd.DataFrame] = []
    for tf in args.timeframes:
        if tf not in TIMEFRAMES:
            continue
        print(f"[cycle-start-ppo] building ledger for {tf}...", flush=True)
        df = build_cycle_ledger(tf)
        print(f"[cycle-start-ppo]   {tf}: {len(df)} cycles", flush=True)
        frames.append(df)
    ledger = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"[cycle-start-ppo] total cycles: {len(ledger)}", flush=True)

    p77 = write_77_ledger(ledger, out)
    print(f"[cycle-start-ppo] wrote {p77}", flush=True)
    p74 = write_74(ledger, out)
    print(f"[cycle-start-ppo] wrote {p74}", flush=True)
    p75 = write_75(ledger, out)
    print(f"[cycle-start-ppo] wrote {p75}", flush=True)
    p76 = write_76_hierarchy(ledger, out)
    print(f"[cycle-start-ppo] wrote {p76}", flush=True)
    print("[cycle-start-ppo] done.", flush=True)


if __name__ == "__main__":
    main()
