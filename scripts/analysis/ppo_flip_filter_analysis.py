"""PPO flip filter analysis: which reversal candles should we flip on?

Builds the same always-in-market flip ledger as
``ppo_zone_long_short_flip_backtest.py`` but enriches every trade with
signal-time context features and then bucketizes those features to surface
conditions where flipping is +EV vs -EV.

Per-signal features:
- ``prev_run_length``: bars the prior direction held for (1, 2, 3+).
- ``signal_hist_value`` and ``|signal_hist_value|``: PPO_hist magnitude at
  the reversal bar.
- ``signal_hist_diff_magnitude``: |ppo_hist diff| at the signal — strength of
  the flip.
- ``signal_diff_z``: signal hist-diff magnitude / its expanding std.
- ``signal_body_pct``: |close-open|/open at signal — candle conviction.
- ``signal_body_z``: body / 64-bar expanding std of body.
- ``ppo_at_signal_extreme``: bottom20 / top20 vs mid (look-ahead-free
  expanding quantile bucket).
- ``hist_at_signal_extreme``: same for ppo_hist.
- ``upper_tf_align``: count of upper TFs whose last-closed PPO_hist sign
  matches the new (post-flip) direction.

Trade PnL is the always-in-market flip return between consecutive signals,
direction taken from ``raw_direction`` at the signal bar, executed at close
of next bar (1-bar delay).

Outputs land in
``outputs/analysis_results/ppo_flip_filter_analysis/``.
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
UPPER_OF = {"15m": ("1h", "4h", "1d"), "1h": ("4h", "1d"), "4h": ("1d",), "1d": ()}


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_flip_filter_analysis"


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


def _expanding_quantile_bucket(values: pd.Series) -> pd.Series:
    nums = pd.to_numeric(values, errors="coerce")
    shifted = nums.shift(1)
    q10 = shifted.expanding(min_periods=200).quantile(0.10)
    q20 = shifted.expanding(min_periods=200).quantile(0.20)
    q80 = shifted.expanding(min_periods=200).quantile(0.80)
    q90 = shifted.expanding(min_periods=200).quantile(0.90)
    return pd.Series(
        np.select(
            [nums <= q10, nums <= q20, nums >= q90, nums >= q80],
            ["bottom10", "bottom20", "top10", "top20"],
            default="mid",
        ),
        index=values.index,
    )


def _zone4(ppo: pd.Series, hist: pd.Series) -> pd.Series:
    ppo_sign = np.where(pd.to_numeric(ppo, errors="coerce") >= 0, "ppo_pos", "ppo_neg")
    hist_sign = np.where(pd.to_numeric(hist, errors="coerce") >= 0, "hist_pos", "hist_neg")
    return pd.Series(ppo_sign + "__" + hist_sign, index=ppo.index)


def load_candles(timeframe: str, zero_mode: str) -> pd.DataFrame:
    path = _raw_market_path(timeframe)
    cols_present = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in ("date", "timestamp", "open_time", "open", "high", "low", "close", "ppo", "ppo_hist") if c in cols_present]
    df = pd.read_csv(path, usecols=usecols).copy()
    ts_col = next((c for c in ("timestamp", "open_time", "date") if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"{path} missing timestamp column")
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

    body_pct = (df["close"] - df["open"]).abs() / df["open"].replace(0, np.nan) * 100.0
    df["body_pct"] = body_pct
    body_std = body_pct.shift(1).expanding(min_periods=200).std()
    df["body_z"] = (body_pct / body_std).replace([np.inf, -np.inf], np.nan)

    diff_abs = df["ppo_hist_diff"].abs()
    diff_std = diff_abs.shift(1).expanding(min_periods=200).std()
    df["hist_diff_z"] = (diff_abs / diff_std).replace([np.inf, -np.inf], np.nan)
    df["abs_hist_diff"] = diff_abs

    df["zone4"] = _zone4(df["ppo"], df["ppo_hist"])
    df["ppo_expanding_bin"] = _expanding_quantile_bucket(df["ppo"])
    df["hist_expanding_bin"] = _expanding_quantile_bucket(df["ppo_hist"])
    df["ppo_sign_zone"] = np.where(df["ppo"] >= 0, "ppo_pos", "ppo_neg")
    df["hist_sign_zone"] = np.where(df["ppo_hist"] >= 0, "hist_pos", "hist_neg")
    return df


def compute_run_length_at_each_signal(raw: np.ndarray) -> np.ndarray:
    """Return prior run length for each bar — bars that the prior direction
    held before the bar's value flipped at index i.  Zeros wherever no flip."""
    n = len(raw)
    out = np.zeros(n, dtype=np.int64)
    if n < 2:
        return out
    run_start = 0
    prev_dir = raw[0]
    for i in range(1, n):
        cur = raw[i]
        if cur == 0 or prev_dir == 0:
            prev_dir = cur if cur != 0 else prev_dir
            run_start = i
            continue
        if cur != prev_dir:
            out[i] = i - run_start
            run_start = i
            prev_dir = cur
    return out


def build_signals(candles: pd.DataFrame) -> list[int]:
    raw = candles["raw_direction"].to_numpy(dtype=np.int64)
    signals: list[int] = []
    for i in range(1, len(raw)):
        if raw[i] == 0 or raw[i - 1] == 0:
            continue
        if raw[i] != raw[i - 1]:
            signals.append(i)
    return signals


def upper_align_count(signal_time: pd.Timestamp, new_direction_sign: int, upper_candles: dict[str, pd.DataFrame], upper_tfs: tuple[str, ...]) -> int:
    """For each upper TF, find the last fully-closed bar at signal_time and
    check whether its ppo_hist sign matches new_direction_sign."""
    count = 0
    for tf in upper_tfs:
        u = upper_candles[tf]
        # last closed bar's start_time + tf <= signal_time
        delta = pd.to_timedelta(TF_SECONDS[tf], unit="s")
        cutoff = signal_time - delta
        idx = u["timestamp"].searchsorted(cutoff, side="right") - 1
        if idx < 0 or idx >= len(u):
            continue
        hist_val = u["ppo_hist"].iat[int(idx)]
        if pd.isna(hist_val) or hist_val == 0:
            continue
        u_sign = 1 if hist_val > 0 else -1
        if u_sign == new_direction_sign:
            count += 1
    return count


def build_filtered_trades(candles: pd.DataFrame, timeframe: str, upper_candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    raw = candles["raw_direction"].to_numpy(dtype=np.int64)
    signals = build_signals(candles)
    if len(signals) < 2:
        return pd.DataFrame()
    run_lengths = compute_run_length_at_each_signal(raw)
    upper_tfs = UPPER_OF[timeframe]

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

        align_count = upper_align_count(pd.Timestamp(sig_row["timestamp"]), new_dir_sign, upper_candles, upper_tfs) if upper_tfs else np.nan
        rows.append(
            {
                "timeframe": timeframe,
                "signal_time": sig_row["timestamp"],
                "entry_time": entry_row["timestamp"],
                "exit_time": exit_row["timestamp"],
                "direction": direction,
                "entry_price": float(entry_row["close"]),
                "exit_price": float(exit_row["close"]),
                "gross_return": gross,
                "net_return": net,
                "holding_bars": int(exit_idx - entry_idx),
                "ppo_at_signal": float(sig_row["ppo"]),
                "hist_at_signal": float(sig_row["ppo_hist"]),
                "abs_hist_at_signal": float(abs(sig_row["ppo_hist"])),
                "ppo_sign_zone": str(sig_row["ppo_sign_zone"]),
                "hist_sign_zone": str(sig_row["hist_sign_zone"]),
                "zone4": str(sig_row["zone4"]),
                "ppo_expanding_bin": str(sig_row["ppo_expanding_bin"]),
                "hist_expanding_bin": str(sig_row["hist_expanding_bin"]),
                "prev_run_length": int(run_lengths[sig_i]),
                "abs_hist_diff": float(sig_row["abs_hist_diff"]),
                "hist_diff_z": float(sig_row["hist_diff_z"]),
                "body_pct": float(sig_row["body_pct"]),
                "body_z": float(sig_row["body_z"]),
                "upper_tf_align_count": align_count,
                "n_upper_tfs": len(upper_tfs),
            }
        )
    return pd.DataFrame(rows)


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


def bucket_runlen(value: int) -> str:
    if value <= 1:
        return "1 (1-bar reversal)"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    if value <= 5:
        return "4-5"
    if value <= 10:
        return "6-10"
    return "11+"


def bucket_z(value: float, edges: tuple[float, ...] = (0.5, 1.0, 1.5, 2.5)) -> str:
    if pd.isna(value):
        return "na"
    if value < edges[0]:
        return f"<{edges[0]}"
    for left, right in zip(edges[:-1], edges[1:]):
        if value < right:
            return f"{left}-{right}"
    return f">={edges[-1]}"


def bucket_align(value: float, n_upper: int) -> str:
    if pd.isna(value):
        return "na"
    if n_upper == 0:
        return "no_upper"
    return f"{int(value)}/{n_upper}"


def bucket_abs_hist(value: float, edges: tuple[float, ...]) -> str:
    if pd.isna(value):
        return "na"
    if value < edges[0]:
        return f"<{edges[0]:g}"
    for left, right in zip(edges[:-1], edges[1:]):
        if value < right:
            return f"{left:g}-{right:g}"
    return f">={edges[-1]:g}"


def _summarize_buckets(trades: pd.DataFrame, feature_col: str, group_cols: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in trades.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update(_summary(group))
        # baseline = same TF avg_net_pct
        tf = row.get("timeframe")
        baseline = trades[trades["timeframe"].eq(tf)]["net_return"].mean()
        row["baseline_avg_net_pct"] = float(baseline)
        row["lift_vs_baseline_pct_pt"] = row["avg_net_pct"] - float(baseline)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True)


def filter_simulation(trades: pd.DataFrame, mask: pd.Series, label: str) -> dict[str, Any]:
    sub = trades[mask]
    if sub.empty:
        return {
            "filter": label,
            "n_trades": 0,
            "win_rate_pct": np.nan,
            "avg_net_pct": np.nan,
            "total_compounded_pct": np.nan,
            "max_dd_pct": np.nan,
            "fraction_kept_pct": 0.0,
        }
    summary = _summary(sub)
    summary.pop("low_sample", None)
    summary["filter"] = label
    summary["fraction_kept_pct"] = len(sub) / len(trades) * 100
    return summary


def run_filter_simulations(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for tf, frame in trades.groupby("timeframe", sort=False):
        n_upper = int(frame["n_upper_tfs"].iloc[0])
        masks = {
            "all_signals": pd.Series(True, index=frame.index),
            "prev_run_length>=2": frame["prev_run_length"] >= 2,
            "prev_run_length>=3": frame["prev_run_length"] >= 3,
            "prev_run_length>=4": frame["prev_run_length"] >= 4,
            "hist_diff_z>=1": frame["hist_diff_z"] >= 1.0,
            "hist_diff_z>=1.5": frame["hist_diff_z"] >= 1.5,
            "body_z>=1": frame["body_z"] >= 1.0,
            "body_z>=1.5": frame["body_z"] >= 1.5,
            "ppo_extreme(top20|bottom20)": frame["ppo_expanding_bin"].isin(["top10", "top20", "bottom10", "bottom20"]),
            "ppo_extreme & dir mean-reverts": (
                ((frame["ppo_expanding_bin"].isin(["top10", "top20"])) & frame["direction"].eq("short"))
                | ((frame["ppo_expanding_bin"].isin(["bottom10", "bottom20"])) & frame["direction"].eq("long"))
            ),
            "hist_extreme & dir mean-reverts": (
                ((frame["hist_expanding_bin"].isin(["top10", "top20"])) & frame["direction"].eq("short"))
                | ((frame["hist_expanding_bin"].isin(["bottom10", "bottom20"])) & frame["direction"].eq("long"))
            ),
        }
        if n_upper > 0:
            masks[f"upper_align>={max(1, n_upper - 1)}/{n_upper}"] = frame["upper_tf_align_count"] >= max(1, n_upper - 1)
            masks[f"upper_align={n_upper}/{n_upper}"] = frame["upper_tf_align_count"] >= n_upper
            masks["upper_align=0 (skip!)"] = frame["upper_tf_align_count"].eq(0)
            masks["combo: prev_run>=3 & upper_align>=1"] = (frame["prev_run_length"] >= 3) & (frame["upper_tf_align_count"] >= 1)
            masks["combo: prev_run>=3 & body_z>=1"] = (frame["prev_run_length"] >= 3) & (frame["body_z"] >= 1.0)
            masks["combo: prev_run>=3 & hist_diff_z>=1"] = (frame["prev_run_length"] >= 3) & (frame["hist_diff_z"] >= 1.0)
            masks["combo: prev_run>=4 & upper_align=full"] = (frame["prev_run_length"] >= 4) & (frame["upper_tf_align_count"] >= n_upper)
        for label, mask in masks.items():
            rec = filter_simulation(frame, mask, label)
            rec["timeframe"] = tf
            rows.append(rec)
    return pd.DataFrame(rows)[["timeframe", "filter", "n_trades", "fraction_kept_pct", "win_rate_pct", "avg_net_pct", "total_compounded_pct", "max_dd_pct"]]


def write_report(out_dir: Path, summaries: dict[str, pd.DataFrame], filters_df: pd.DataFrame) -> None:
    def fmt(df: pd.DataFrame, cols: list[str], n: int = 30) -> str:
        if df.empty:
            return "_no rows_"
        cols = [c for c in cols if c in df.columns]
        return df[cols].head(n).to_markdown(index=False, floatfmt=".2f")

    parts: list[str] = ["# PPO Flip Filter Analysis\n"]
    parts.append(
        "## 0. 분석 목적\n"
        "각 전환(반대 방향) 캔들 시그널에서 포지션을 **플립할지 / 무시할지**를 가르는 조건을 찾는다.\n"
        "기존 always-in-market 플립 백테스트(다음 캔들 종료 시점 종가에서 플립)에 동일하게 트레이드를 만들고,\n"
        "각 시그널의 컨텍스트 특성(직전 방향 지속 길이, hist 변화 강도, 캔들 몸통, 상위 TF 정렬,"
        " PPO 극단 여부)별로 플립 후 trade의 net 평균/승률/누적복리/MDD를 비교한다.\n"
        "기준: 같은 TF baseline 평균 대비 +EV 방향이고 표본 50건 이상이면 *'플립 우선 후보'*,"
        " baseline 대비 음수가 크면 *'플립 회피 후보'* 로 본다.\n"
    )

    parts.append("\n## 1. 직전 방향 지속 길이 (prev_run_length) 별\n")
    parts.append(
        fmt(
            summaries["by_run_length.csv"].sort_values(["timeframe", "prev_run_length_bucket"]),
            ["timeframe", "prev_run_length_bucket", "n_trades", "win_rate_pct", "avg_net_pct", "lift_vs_baseline_pct_pt", "total_compounded_pct", "max_dd_pct"],
            n=40,
        )
    )

    parts.append("\n## 2. 시그널 hist 변화 강도 z-score 구간별\n")
    parts.append(
        fmt(
            summaries["by_hist_diff_z.csv"].sort_values(["timeframe", "hist_diff_z_bucket"]),
            ["timeframe", "hist_diff_z_bucket", "n_trades", "win_rate_pct", "avg_net_pct", "lift_vs_baseline_pct_pt"],
            n=40,
        )
    )

    parts.append("\n## 3. 시그널 캔들 몸통 z-score 구간별\n")
    parts.append(
        fmt(
            summaries["by_body_z.csv"].sort_values(["timeframe", "body_z_bucket"]),
            ["timeframe", "body_z_bucket", "n_trades", "win_rate_pct", "avg_net_pct", "lift_vs_baseline_pct_pt"],
            n=40,
        )
    )

    parts.append("\n## 4. 상위 TF PPO_hist 정렬 개수별\n")
    parts.append(
        fmt(
            summaries["by_upper_align.csv"].sort_values(["timeframe", "upper_align_bucket"]),
            ["timeframe", "upper_align_bucket", "n_trades", "win_rate_pct", "avg_net_pct", "lift_vs_baseline_pct_pt"],
            n=40,
        )
    )

    parts.append("\n## 5. 시그널 시점의 PPO expanding-quantile bucket × 새 포지션 방향\n")
    parts.append(
        fmt(
            summaries["by_ppo_bin_direction.csv"].sort_values(["timeframe", "ppo_expanding_bin", "direction"]),
            ["timeframe", "ppo_expanding_bin", "direction", "n_trades", "win_rate_pct", "avg_net_pct", "lift_vs_baseline_pct_pt"],
            n=80,
        )
    )

    parts.append("\n## 6. zone4 (PPO sign × hist sign) × 새 포지션 방향\n")
    parts.append(
        fmt(
            summaries["by_zone4_direction.csv"].sort_values(["timeframe", "zone4", "direction"]),
            ["timeframe", "zone4", "direction", "n_trades", "win_rate_pct", "avg_net_pct", "lift_vs_baseline_pct_pt"],
            n=80,
        )
    )

    parts.append("\n## 7. 후보 필터 시뮬레이션 (TF별 비교)\n")
    parts.append(
        fmt(
            filters_df,
            ["timeframe", "filter", "n_trades", "fraction_kept_pct", "win_rate_pct", "avg_net_pct", "total_compounded_pct", "max_dd_pct"],
            n=80,
        )
    )

    parts.append("\n## 8. 해석 가이드\n")
    parts.append(
        "- `lift_vs_baseline_pct_pt` > 0 이면 해당 조건의 트레이드 평균 net이 같은 TF 전체 평균보다 좋다 → **플립을 권장하는 조건**.\n"
        "- `lift_vs_baseline_pct_pt` < 0 이면 평균이 전체보다 나쁘다 → **플립 회피 후보**.\n"
        "- `prev_run_length=1`은 직전 시그널 직후 다시 반대로 흔드는 \"노이즈 더블 플립\" 케이스이며 단기 TF에서 가장 비용이 많이 든다.\n"
        "- 상위 TF 정렬이 0이면 단기 시그널이 상위 추세를 거스르는 신호 → 보통 회피.\n"
        "- 표본 50 미만 (`low_sample`) 셀은 결론에서 제외 권장.\n"
    )
    (out_dir / "PPO_flip_filter_report.md").write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-mode", choices=("ffill", "zero"), default="ffill")
    parser.add_argument("--timeframes", nargs="*", default=list(TIMEFRAMES), choices=TIMEFRAMES)
    args = parser.parse_args()

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    candles_by_tf: dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        candles_by_tf[tf] = load_candles(tf, args.zero_mode)

    all_trades: list[pd.DataFrame] = []
    for tf in args.timeframes:
        trades = build_filtered_trades(candles_by_tf[tf], tf, candles_by_tf)
        if not trades.empty:
            all_trades.append(trades)
        print(f"{tf}: candles={len(candles_by_tf[tf]):,}  trades={len(trades):,}")
    if not all_trades:
        print("No trades.")
        return 1

    trades_df = pd.concat(all_trades, ignore_index=True).sort_values(["timeframe", "entry_time"]).reset_index(drop=True)

    trades_df["prev_run_length_bucket"] = trades_df["prev_run_length"].apply(bucket_runlen)
    trades_df["hist_diff_z_bucket"] = trades_df["hist_diff_z"].apply(bucket_z)
    trades_df["body_z_bucket"] = trades_df["body_z"].apply(bucket_z)
    trades_df["upper_align_bucket"] = trades_df.apply(lambda r: bucket_align(r["upper_tf_align_count"], int(r["n_upper_tfs"])), axis=1)

    trades_df.to_csv(out_dir / "00_trades_with_features.csv", index=False, encoding="utf-8-sig")

    summaries: dict[str, pd.DataFrame] = {}
    summaries["by_run_length.csv"] = _summarize_buckets(trades_df, "prev_run_length_bucket", ["timeframe", "prev_run_length_bucket"])
    summaries["by_run_length_direction.csv"] = _summarize_buckets(trades_df, "prev_run_length_bucket", ["timeframe", "prev_run_length_bucket", "direction"])
    summaries["by_hist_diff_z.csv"] = _summarize_buckets(trades_df, "hist_diff_z_bucket", ["timeframe", "hist_diff_z_bucket"])
    summaries["by_body_z.csv"] = _summarize_buckets(trades_df, "body_z_bucket", ["timeframe", "body_z_bucket"])
    summaries["by_upper_align.csv"] = _summarize_buckets(trades_df, "upper_align_bucket", ["timeframe", "upper_align_bucket"])
    summaries["by_ppo_bin_direction.csv"] = _summarize_buckets(trades_df, "ppo_expanding_bin", ["timeframe", "ppo_expanding_bin", "direction"])
    summaries["by_zone4_direction.csv"] = _summarize_buckets(trades_df, "zone4", ["timeframe", "zone4", "direction"])
    summaries["by_run_x_align.csv"] = _summarize_buckets(trades_df, "prev_run_length_bucket", ["timeframe", "prev_run_length_bucket", "upper_align_bucket"])
    summaries["by_run_x_body.csv"] = _summarize_buckets(trades_df, "prev_run_length_bucket", ["timeframe", "prev_run_length_bucket", "body_z_bucket"])

    for name, frame in summaries.items():
        frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig")

    filters_df = run_filter_simulations(trades_df)
    filters_df.to_csv(out_dir / "10_filter_simulations.csv", index=False, encoding="utf-8-sig")

    write_report(out_dir, summaries, filters_df)
    print(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
