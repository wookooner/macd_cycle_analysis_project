"""PPO flip × multi-timeframe cycle hierarchy filter analysis.

Same always-in-market flip ledger as ``ppo_zone_long_short_flip_backtest.py``,
but each signal is enriched with cycle-context features at:
- own TF cycle (direction, progress, bars_since_start)
- parent TF cycle (direction, progress, bars_since_start)
- grandparent TF cycle (direction, progress)
- great-grandparent TF cycle (only for 15m signals)

Outputs surface (a) which combinations of (parent_direction × own_direction ×
position-in-parent-cycle × signal_direction) are clearly +EV (must-enter) and
(b) which are clearly -EV (avoid).

Cycle progress is derived from ``duration_candles`` (the retrospective full
length of the cycle).  This is post-hoc analysis; ``bars_since_cycle_start``
is also reported as a real-time-knowable proxy.

Output dir: ``outputs/analysis_results/ppo_flip_cycle_hierarchy_analysis/``.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import PROJECT_PATHS  # noqa: E402

TIMEFRAMES = ("15m", "1h", "4h", "1d")
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
ROUND_TRIP_FEE_PCT = 0.08
SLIPPAGE_PER_SIDE_PCT = 0.02
COST_PER_FLIP_PCT = ROUND_TRIP_FEE_PCT + 2 * SLIPPAGE_PER_SIDE_PCT
LOW_SAMPLE_N = 50

# (own_tf, parent_tf, grandparent_tf, great_grandparent_tf)
HIERARCHY = {
    "15m": ("1h", "4h", "1d"),
    "1h": ("4h", "1d", None),
    "4h": ("1d", None, None),
    "1d": (None, None, None),
}


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_flip_cycle_hierarchy_analysis"


def _read_timestamp(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def _raw_market_path(timeframe: str) -> Path:
    candidates = [
        PROJECT_PATHS.raw_market_dir / f"BTCUSD_{timeframe}.csv",
        PROJECT_PATHS.raw_market_dir / f"BTCUSDT_{timeframe}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing raw candle file for {timeframe}: tried {candidates}")


def _cycle_path(timeframe: str) -> Path:
    candidates = [
        PROJECT_PATHS.cycle_structured_dir / "btc" / f"cycles_{timeframe}.parquet",
        PROJECT_PATHS.asset_cycle_dir("btc") / f"cycles_{timeframe}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing cycle parquet for {timeframe}: tried {candidates}")


def _tf_delta(tf: str) -> pd.Timedelta:
    return pd.to_timedelta(TF_SECONDS[tf], unit="s")


def load_candles(timeframe: str, zero_mode: str = "ffill") -> pd.DataFrame:
    path = _raw_market_path(timeframe)
    cols_present = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in ("date", "timestamp", "open_time", "open", "high", "low", "close", "ppo", "ppo_hist") if c in cols_present]
    df = pd.read_csv(path, usecols=usecols).copy()
    ts_col = next((c for c in ("timestamp", "open_time", "date") if c in df.columns), None)
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _read_timestamp(df["timestamp"])
    for col in ("open", "high", "low", "close", "ppo", "ppo_hist"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "ppo", "ppo_hist"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["timeframe"] = timeframe
    df["bar_index"] = np.arange(len(df), dtype=np.int64)
    df["ppo_hist_diff"] = df["ppo_hist"].diff()
    raw = np.sign(df["ppo_hist_diff"]).astype("float")
    if zero_mode == "ffill":
        raw = raw.replace(0, np.nan).ffill().fillna(0)
    else:
        raw = raw.fillna(0)
    df["raw_direction"] = raw.astype("int8")
    return df


def load_cycles(timeframe: str) -> pd.DataFrame:
    df = pd.read_parquet(_cycle_path(timeframe), columns=["cycle_id", "start_date", "end_date", "cycle_type", "duration_candles", "category"]).copy()
    df["start_date"] = _read_timestamp(df["start_date"])
    df["end_date"] = _read_timestamp(df["end_date"])
    df = df.dropna(subset=["start_date", "end_date"]).sort_values("start_date").reset_index(drop=True)
    df["cycle_sign"] = np.where(df["cycle_type"].astype(str).str.lower().isin({"up", "long", "1", "+1"}), 1, -1).astype(np.int8)
    df["end_exclusive"] = df["end_date"] + _tf_delta(timeframe)
    df["duration_candles"] = pd.to_numeric(df["duration_candles"], errors="coerce")
    return df


def attach_cycle_context(target_times: pd.Series, cycles: pd.DataFrame, tf: str, prefix: str) -> pd.DataFrame:
    """For each timestamp, return cycle context columns.  No look-ahead beyond
    the cycle's known end_date (ie. retrospective) — labelled clearly.
    """
    if cycles.empty:
        return pd.DataFrame(
            {
                f"{prefix}_cycle_id": pd.Series([np.nan] * len(target_times)),
                f"{prefix}_cycle_sign": pd.Series([np.nan] * len(target_times)),
                f"{prefix}_cycle_progress": pd.Series([np.nan] * len(target_times)),
                f"{prefix}_bars_since_start": pd.Series([np.nan] * len(target_times)),
                f"{prefix}_cycle_category": pd.Series([np.nan] * len(target_times)),
            }
        )
    keep = cycles[["cycle_id", "cycle_sign", "start_date", "end_exclusive", "duration_candles", "category"]].copy()
    keep = keep.rename(columns={
        "cycle_id": f"{prefix}_cycle_id",
        "cycle_sign": f"{prefix}_cycle_sign",
        "start_date": f"{prefix}_cycle_start",
        "end_exclusive": f"{prefix}_cycle_end_excl",
        "duration_candles": f"{prefix}_duration_candles",
        "category": f"{prefix}_cycle_category",
    })
    target_df = pd.DataFrame({"timestamp": pd.to_datetime(target_times.values)}).sort_values("timestamp").reset_index().rename(columns={"index": "_orig_idx"})
    merged = pd.merge_asof(
        target_df,
        keep,
        left_on="timestamp",
        right_on=f"{prefix}_cycle_start",
        direction="backward",
    )
    in_cycle = merged["timestamp"].lt(merged[f"{prefix}_cycle_end_excl"])
    for col in [f"{prefix}_cycle_id", f"{prefix}_cycle_sign", f"{prefix}_cycle_start", f"{prefix}_cycle_end_excl", f"{prefix}_duration_candles", f"{prefix}_cycle_category"]:
        merged.loc[~in_cycle, col] = np.nan
    elapsed = (merged["timestamp"] - merged[f"{prefix}_cycle_start"]) / _tf_delta(tf)
    merged[f"{prefix}_bars_since_start"] = elapsed
    merged[f"{prefix}_cycle_progress"] = (elapsed / merged[f"{prefix}_duration_candles"].replace(0, np.nan)).clip(lower=0, upper=1)
    out_cols = [
        f"{prefix}_cycle_id",
        f"{prefix}_cycle_sign",
        f"{prefix}_cycle_progress",
        f"{prefix}_bars_since_start",
        f"{prefix}_cycle_category",
    ]
    return merged.set_index("_orig_idx").sort_index()[out_cols].reset_index(drop=True)


def build_signals(candles: pd.DataFrame) -> list[int]:
    raw = candles["raw_direction"].to_numpy(dtype=np.int64)
    out: list[int] = []
    for i in range(1, len(raw)):
        if raw[i] == 0 or raw[i - 1] == 0:
            continue
        if raw[i] != raw[i - 1]:
            out.append(i)
    return out


def progress_bucket(value: float) -> str:
    if pd.isna(value):
        return "na"
    if value < 0.33:
        return "early"
    if value < 0.66:
        return "mid"
    return "late"


def direction_label(sign: float) -> str:
    if pd.isna(sign):
        return "na"
    if sign > 0:
        return "up"
    if sign < 0:
        return "down"
    return "flat"


def build_flip_trades_with_cycles(tf: str, candles_by_tf: dict[str, pd.DataFrame], cycles_by_tf: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candles = candles_by_tf[tf]
    raw = candles["raw_direction"].to_numpy(dtype=np.int64)
    signals = build_signals(candles)
    if len(signals) < 2:
        return pd.DataFrame()

    parent_tf, gp_tf, ggp_tf = HIERARCHY[tf]
    rows: list[dict[str, Any]] = []
    for k in range(len(signals) - 1):
        sig_i = signals[k]
        sig_next = signals[k + 1]
        entry_idx = sig_i + 1
        exit_idx = sig_next + 1
        if entry_idx >= len(candles) or exit_idx >= len(candles) or exit_idx <= entry_idx:
            continue
        sig_row = candles.iloc[sig_i]
        entry_row = candles.iloc[entry_idx]
        exit_row = candles.iloc[exit_idx]
        new_dir_sign = int(raw[sig_i])
        direction = "long" if new_dir_sign > 0 else "short"
        gross = (float(exit_row["close"]) / float(entry_row["close"]) - 1.0) * 100.0 * new_dir_sign
        net = gross - COST_PER_FLIP_PCT
        rows.append(
            {
                "timeframe": tf,
                "signal_time": sig_row["timestamp"],
                "entry_time": entry_row["timestamp"],
                "exit_time": exit_row["timestamp"],
                "direction": direction,
                "direction_sign": new_dir_sign,
                "entry_price": float(entry_row["close"]),
                "exit_price": float(exit_row["close"]),
                "gross_return": gross,
                "net_return": net,
                "holding_bars": int(exit_idx - entry_idx),
            }
        )
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades

    own_ctx = attach_cycle_context(trades["signal_time"], cycles_by_tf[tf], tf, "own")
    trades = pd.concat([trades.reset_index(drop=True), own_ctx], axis=1)
    if parent_tf is not None:
        parent_ctx = attach_cycle_context(trades["signal_time"], cycles_by_tf[parent_tf], parent_tf, "parent")
        trades = pd.concat([trades.reset_index(drop=True), parent_ctx], axis=1)
    if gp_tf is not None:
        gp_ctx = attach_cycle_context(trades["signal_time"], cycles_by_tf[gp_tf], gp_tf, "gp")
        trades = pd.concat([trades.reset_index(drop=True), gp_ctx], axis=1)
    if ggp_tf is not None:
        ggp_ctx = attach_cycle_context(trades["signal_time"], cycles_by_tf[ggp_tf], ggp_tf, "ggp")
        trades = pd.concat([trades.reset_index(drop=True), ggp_ctx], axis=1)

    trades["own_cycle_dir"] = trades["own_cycle_sign"].map(direction_label)
    trades["own_cycle_progress_bucket"] = trades["own_cycle_progress"].apply(progress_bucket)
    if "parent_cycle_sign" in trades.columns:
        trades["parent_cycle_dir"] = trades["parent_cycle_sign"].map(direction_label)
        trades["parent_cycle_progress_bucket"] = trades["parent_cycle_progress"].apply(progress_bucket)
    if "gp_cycle_sign" in trades.columns:
        trades["gp_cycle_dir"] = trades["gp_cycle_sign"].map(direction_label)
        trades["gp_cycle_progress_bucket"] = trades["gp_cycle_progress"].apply(progress_bucket)
    if "ggp_cycle_sign" in trades.columns:
        trades["ggp_cycle_dir"] = trades["ggp_cycle_sign"].map(direction_label)
    return trades


def _max_drawdown_pct(returns: pd.Series) -> float:
    rets = pd.to_numeric(returns, errors="coerce").fillna(0) / 100.0
    if rets.empty:
        return np.nan
    equity = (1.0 + rets).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min() * 100)


def _summary(group: pd.DataFrame) -> dict[str, Any]:
    net = pd.to_numeric(group["net_return"], errors="coerce").dropna()
    wins = net[net > 0]
    losses = net[net < 0]
    compounded = (np.prod(1 + net / 100.0) - 1.0) * 100.0 if not net.empty else np.nan
    if abs(losses.sum()) > 0:
        pf = float(wins.sum() / abs(losses.sum()))
    elif wins.sum() > 0:
        pf = math.inf
    else:
        pf = np.nan
    return {
        "n_trades": int(len(group)),
        "win_rate_pct": float((net > 0).mean() * 100) if len(net) else np.nan,
        "avg_net_pct": float(net.mean()) if len(net) else np.nan,
        "median_net_pct": float(net.median()) if len(net) else np.nan,
        "total_compounded_pct": float(compounded) if pd.notna(compounded) else np.nan,
        "profit_factor": pf,
        "max_dd_pct": _max_drawdown_pct(group.sort_values("entry_time")["net_return"]),
        "low_sample": len(group) < LOW_SAMPLE_N,
    }


def summarize(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in trades.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update(_summary(group))
        tf = row.get("timeframe")
        baseline = trades[trades["timeframe"].eq(tf)]["net_return"].mean()
        row["baseline_avg_net_pct"] = float(baseline)
        row["lift_vs_baseline_pct_pt"] = row["avg_net_pct"] - float(baseline)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def find_extremes(summary: pd.DataFrame, n_top: int = 15, min_n: int = LOW_SAMPLE_N) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    eligible = summary[summary["n_trades"] >= min_n].copy()
    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame()
    must_enter = eligible.sort_values("avg_net_pct", ascending=False).head(n_top).reset_index(drop=True)
    avoid = eligible.sort_values("avg_net_pct", ascending=True).head(n_top).reset_index(drop=True)
    return must_enter, avoid


def write_report(out_dir: Path, summaries: dict[str, pd.DataFrame], extremes: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    def fmt(df: pd.DataFrame, cols: list[str], n: int = 30) -> str:
        if df.empty:
            return "_no rows_"
        cols = [c for c in cols if c in df.columns]
        return df[cols].head(n).to_markdown(index=False, floatfmt=".2f")

    parts: list[str] = ["# PPO Flip × Cycle Hierarchy Filter Analysis\n"]
    parts.append(
        "## 0. 분석 개요\n"
        "각 전환 시그널의 시점에 대해 자기 시간대 + 부모 + 조부모 + (15m의 경우) 증조부 시간대의 사이클 방향(up/down)과 진행도(early 0-33%, mid 33-66%, late 66-100%)를 함께 기록했다.\n"
        "각 조합별 트레이드의 평균 net 손익이 같은 시간대 baseline보다 얼마나 좋은지(`lift_vs_baseline_pct_pt`)로 +EV / -EV를 판정한다. 표본 50건 미만은 제외.\n"
        "\n사이클 진행도는 `duration_candles`(사이클 종료 후에야 알려지는 값)에 기반한 사후적 측정이다. `bars_since_start`(실시간 가능)도 같이 저장.\n"
    )

    for tf in TIMEFRAMES:
        if tf not in extremes:
            continue
        must_enter, avoid = extremes[tf]
        parts.append(f"\n## {tf} 결과 — 무조건 진입 후보 (avg_net 상위, n>=50)\n")
        parts.append(fmt(must_enter, list(must_enter.columns), n=20))
        parts.append(f"\n## {tf} 결과 — 절대 회피 후보 (avg_net 하위, n>=50)\n")
        parts.append(fmt(avoid, list(avoid.columns), n=20))

    name_map = {
        "by_tf_signal_x_parent_dir.csv": "## 부록 A. 시그널 방향 × 부모 사이클 방향",
        "by_tf_signal_x_parent_progress.csv": "## 부록 B. 시그널 방향 × 부모 사이클 진행도",
        "by_tf_signal_x_parent_dir_x_progress.csv": "## 부록 C. 시그널 방향 × 부모 방향 × 부모 진행도",
        "by_tf_own_dir_x_parent_dir.csv": "## 부록 D. 자기 사이클 방향 × 부모 사이클 방향",
        "by_tf_signal_x_parent_dir_x_gp_dir.csv": "## 부록 E. 시그널 × 부모 × 조부모 사이클 방향 (모든 TF)",
        "by_15m_signal_x_full_hierarchy.csv": "## 부록 F. 15m 시그널 × 1h × 4h × 1d 사이클 방향",
    }
    for name, title in name_map.items():
        df = summaries.get(name)
        if df is None or df.empty:
            continue
        parts.append(f"\n{title}\n")
        cols = [c for c in df.columns if c not in ("baseline_avg_net_pct",)]
        parts.append(fmt(df.sort_values(["timeframe"] + [c for c in cols if c not in ("timeframe", "n_trades", "win_rate_pct", "avg_net_pct", "median_net_pct", "total_compounded_pct", "profit_factor", "max_dd_pct", "low_sample", "lift_vs_baseline_pct_pt")]), cols, n=80))

    parts.append(
        "\n## 결론 가이드\n"
        "- **무조건 진입 후보**: `n_trades >= 50`이고 `avg_net_pct`가 큰 양수, `lift_vs_baseline_pct_pt`도 큰 양수.\n"
        "- **절대 회피 후보**: 같은 표본 조건에서 `avg_net_pct`가 가장 큰 음수.\n"
        "- 실시간 적용 시에는 사이클의 `end`가 사후적으로만 확정된다는 점을 유의. 그래서 `bars_since_start`도 같이 활용.\n"
    )
    (out_dir / "PPO_flip_cycle_hierarchy_report.md").write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-mode", choices=("ffill", "zero"), default="ffill")
    parser.add_argument("--timeframes", nargs="*", default=list(TIMEFRAMES), choices=TIMEFRAMES)
    args = parser.parse_args()

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    candles_by_tf = {tf: load_candles(tf, args.zero_mode) for tf in TIMEFRAMES}
    cycles_by_tf = {tf: load_cycles(tf) for tf in TIMEFRAMES}

    all_trades: list[pd.DataFrame] = []
    for tf in args.timeframes:
        trades = build_flip_trades_with_cycles(tf, candles_by_tf, cycles_by_tf)
        if not trades.empty:
            all_trades.append(trades)
        print(f"{tf}: trades={len(trades):,}")
    if not all_trades:
        return 1

    trades_df = pd.concat(all_trades, ignore_index=True).sort_values(["timeframe", "entry_time"]).reset_index(drop=True)
    trades_df.to_csv(out_dir / "00_trades_with_cycle_context.csv", index=False, encoding="utf-8-sig")

    summaries: dict[str, pd.DataFrame] = {}
    summaries["by_tf_signal_dir.csv"] = summarize(trades_df, ["timeframe", "direction"])
    summaries["by_tf_own_dir_x_parent_dir.csv"] = summarize(
        trades_df[trades_df.get("parent_cycle_dir", pd.Series(dtype=object)).notna()] if "parent_cycle_dir" in trades_df.columns else pd.DataFrame(),
        ["timeframe", "own_cycle_dir", "parent_cycle_dir"],
    )
    if "parent_cycle_dir" in trades_df.columns:
        summaries["by_tf_signal_x_parent_dir.csv"] = summarize(trades_df, ["timeframe", "direction", "parent_cycle_dir"])
        summaries["by_tf_signal_x_parent_progress.csv"] = summarize(trades_df, ["timeframe", "direction", "parent_cycle_progress_bucket"])
        summaries["by_tf_signal_x_parent_dir_x_progress.csv"] = summarize(trades_df, ["timeframe", "direction", "parent_cycle_dir", "parent_cycle_progress_bucket"])
    if "gp_cycle_dir" in trades_df.columns:
        summaries["by_tf_signal_x_parent_dir_x_gp_dir.csv"] = summarize(trades_df, ["timeframe", "direction", "parent_cycle_dir", "gp_cycle_dir"])
        summaries["by_tf_signal_x_parent_x_gp_x_progress.csv"] = summarize(
            trades_df,
            ["timeframe", "direction", "parent_cycle_dir", "gp_cycle_dir", "parent_cycle_progress_bucket"],
        )
    if "ggp_cycle_dir" in trades_df.columns:
        summaries["by_15m_signal_x_full_hierarchy.csv"] = summarize(
            trades_df[trades_df["timeframe"].eq("15m")],
            ["timeframe", "direction", "parent_cycle_dir", "gp_cycle_dir", "ggp_cycle_dir"],
        )
    summaries["by_tf_signal_x_own_progress.csv"] = summarize(trades_df, ["timeframe", "direction", "own_cycle_progress_bucket"])

    for name, frame in summaries.items():
        if frame is not None and not frame.empty:
            frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig")

    extremes: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for tf in args.timeframes:
        tf_combos: list[pd.DataFrame] = []
        for name in (
            "by_tf_signal_x_parent_dir.csv",
            "by_tf_signal_x_parent_progress.csv",
            "by_tf_signal_x_parent_dir_x_progress.csv",
            "by_tf_own_dir_x_parent_dir.csv",
            "by_tf_signal_x_parent_dir_x_gp_dir.csv",
            "by_tf_signal_x_parent_x_gp_x_progress.csv",
            "by_15m_signal_x_full_hierarchy.csv",
            "by_tf_signal_x_own_progress.csv",
        ):
            df = summaries.get(name)
            if df is None or df.empty:
                continue
            sub = df[df["timeframe"].eq(tf)].copy()
            if sub.empty:
                continue
            sub["combo_source"] = name.replace(".csv", "")
            non_metric = [c for c in sub.columns if c not in {"n_trades", "win_rate_pct", "avg_net_pct", "median_net_pct", "total_compounded_pct", "profit_factor", "max_dd_pct", "low_sample", "baseline_avg_net_pct", "lift_vs_baseline_pct_pt", "combo_source"}]
            sub["condition"] = sub[non_metric].fillna("na").astype(str).agg(" | ".join, axis=1)
            tf_combos.append(sub[["combo_source", "condition", "n_trades", "win_rate_pct", "avg_net_pct", "lift_vs_baseline_pct_pt", "total_compounded_pct", "max_dd_pct"]])
        if not tf_combos:
            continue
        combined = pd.concat(tf_combos, ignore_index=True)
        eligible = combined[(combined["n_trades"] >= LOW_SAMPLE_N) & combined["avg_net_pct"].notna()].copy()
        if eligible.empty:
            continue
        must_enter = eligible.sort_values("avg_net_pct", ascending=False).head(20).reset_index(drop=True)
        avoid = eligible.sort_values("avg_net_pct", ascending=True).head(20).reset_index(drop=True)
        extremes[tf] = (must_enter, avoid)
        must_enter.to_csv(out_dir / f"top_must_enter_{tf}.csv", index=False, encoding="utf-8-sig")
        avoid.to_csv(out_dir / f"top_avoid_{tf}.csv", index=False, encoding="utf-8-sig")

    write_report(out_dir, summaries, extremes)
    print(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
