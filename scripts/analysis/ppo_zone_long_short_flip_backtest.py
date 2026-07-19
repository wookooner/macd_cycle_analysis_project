"""PPO zone × long/short flip backtest.

Rules (per user spec):
- raw_direction[i] = sign(ppo_hist[i] - ppo_hist[i-1])
- A "reversal candle" is bar_i where raw_direction[i] != raw_direction[i-1].
- React to that single opposite candle and execute at the close of the NEXT bar
  (bar_i+1).  After entering, if bar_i+2 is again a reversal candle, flip again
  at close[i+3].  In effect: stop-and-reverse with one-bar execution delay,
  always in market.
- Each trade is split by (a) direction (long / short) and (b) PPO position at
  entry (sign zone, 4-zone, decile, expanding-quantile bucket).  Reports compare
  long vs short profitability inside every zone bucket.

Data: BTCUSD raw candles with `ppo`, `ppo_hist` per timeframe (15m / 1h / 4h /
1d) under `PROJECT_PATHS.raw_market_dir`.
Output: `analysis_results/ppo_zone_long_short_flip_backtest/`.
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
LOW_SAMPLE_N = 30


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_zone_long_short_flip_backtest"


def _raw_market_path(timeframe: str) -> Path:
    candidates = [
        PROJECT_PATHS.raw_market_dir / f"BTCUSD_{timeframe}.csv",
        PROJECT_PATHS.raw_market_dir / f"BTCUSDT_{timeframe}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing raw candle file for {timeframe}: tried {candidates}")


def _read_timestamp(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def _zone4(ppo: pd.Series, hist: pd.Series) -> pd.Series:
    ppo_sign = np.where(pd.to_numeric(ppo, errors="coerce") >= 0, "ppo_pos", "ppo_neg")
    hist_sign = np.where(pd.to_numeric(hist, errors="coerce") >= 0, "hist_pos", "hist_neg")
    return pd.Series(ppo_sign + "__" + hist_sign, index=ppo.index)


def _full_history_decile(values: pd.Series) -> pd.Series:
    ranked = pd.to_numeric(values, errors="coerce").rank(method="first")
    try:
        out = pd.qcut(ranked, 10, labels=False)
    except ValueError:
        return pd.Series(np.nan, index=values.index)
    return (out.astype("float") + 1).astype("Int64")


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
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")
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
    df["zone4"] = _zone4(df["ppo"], df["ppo_hist"])
    df["ppo_decile"] = _full_history_decile(df["ppo"])
    df["hist_decile"] = _full_history_decile(df["ppo_hist"])
    df["ppo_expanding_bin"] = _expanding_quantile_bucket(df["ppo"])
    df["hist_expanding_bin"] = _expanding_quantile_bucket(df["ppo_hist"])
    df["ppo_sign_zone"] = np.where(df["ppo"] >= 0, "ppo_pos", "ppo_neg")
    df["hist_sign_zone"] = np.where(df["ppo_hist"] >= 0, "hist_pos", "hist_neg")
    return df


def build_flip_trades(candles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Always-in-market stop-and-reverse trades using user-defined rule.

    - Signal at bar_i: raw_direction[i] != raw_direction[i-1] (a reversal candle).
    - Execute the flip at close[i+1]; new position direction = raw_direction[i].
    - Each trade runs from close[signal_i+1] to close[next_signal_j+1].
    - Direction at PPO snapshot is taken from bar_signal (the reversal candle),
      which is what the trader actually sees when deciding to flip.
    """
    raw = candles["raw_direction"].to_numpy(dtype=np.int64)
    n = len(candles)
    if n < 4:
        return pd.DataFrame()

    signals: list[int] = []
    for i in range(1, n):
        if raw[i] == 0 or raw[i - 1] == 0:
            continue
        if raw[i] != raw[i - 1]:
            signals.append(i)

    trades: list[dict[str, Any]] = []
    for k in range(len(signals) - 1):
        sig_i = signals[k]
        sig_next = signals[k + 1]
        entry_idx = sig_i + 1
        exit_idx = sig_next + 1
        if entry_idx >= n or exit_idx >= n or exit_idx <= entry_idx:
            continue
        sig_row = candles.iloc[sig_i]
        entry_row = candles.iloc[entry_idx]
        exit_row = candles.iloc[exit_idx]
        direction = "long" if raw[sig_i] > 0 else "short"
        sign = 1.0 if direction == "long" else -1.0
        gross = (float(exit_row["close"]) / float(entry_row["close"]) - 1.0) * 100.0 * sign
        net = gross - COST_PER_FLIP_PCT
        trades.append(
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
                # PPO position snapshot at the SIGNAL candle (the reversal bar)
                "ppo_at_signal": float(sig_row["ppo"]),
                "hist_at_signal": float(sig_row["ppo_hist"]),
                "ppo_sign_zone": str(sig_row["ppo_sign_zone"]),
                "hist_sign_zone": str(sig_row["hist_sign_zone"]),
                "zone4": str(sig_row["zone4"]),
                "ppo_decile": int(sig_row["ppo_decile"]) if pd.notna(sig_row["ppo_decile"]) else np.nan,
                "hist_decile": int(sig_row["hist_decile"]) if pd.notna(sig_row["hist_decile"]) else np.nan,
                "ppo_expanding_bin": str(sig_row["ppo_expanding_bin"]),
                "hist_expanding_bin": str(sig_row["hist_expanding_bin"]),
            }
        )
    return pd.DataFrame(trades)


def _max_drawdown_pct(returns: pd.Series) -> float:
    rets = pd.to_numeric(returns, errors="coerce").fillna(0) / 100.0
    if rets.empty:
        return np.nan
    equity = (1.0 + rets).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min() * 100)


def _summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    net = pd.to_numeric(group["net_return"], errors="coerce").dropna()
    gross = pd.to_numeric(group["gross_return"], errors="coerce").dropna()
    wins = net[net > 0]
    losses = net[net < 0]
    compounded = (np.prod(1 + net / 100.0) - 1.0) * 100.0 if not net.empty else np.nan
    if abs(losses.sum()) > 0:
        profit_factor = float(wins.sum() / abs(losses.sum()))
    elif wins.sum() > 0:
        profit_factor = math.inf
    else:
        profit_factor = np.nan
    holding_mean = float(pd.to_numeric(group["holding_bars"], errors="coerce").mean()) if len(group) else np.nan
    return {
        "n_trades": int(len(group)),
        "win_rate_pct": float((net > 0).mean() * 100) if len(net) else np.nan,
        "avg_gross_return_pct": float(gross.mean()) if len(gross) else np.nan,
        "avg_net_return_pct": float(net.mean()) if len(net) else np.nan,
        "median_net_return_pct": float(net.median()) if len(net) else np.nan,
        "total_net_compounded_pct": float(compounded) if pd.notna(compounded) else np.nan,
        "total_net_sum_pct": float(net.sum()) if len(net) else np.nan,
        "profit_factor": profit_factor,
        "max_drawdown_pct": _max_drawdown_pct(group.sort_values("entry_time")["net_return"]),
        "sharpe_like": float(net.mean() / net.std() * math.sqrt(len(net))) if len(net) > 1 and net.std() else np.nan,
        "avg_holding_bars": holding_mean,
        "low_sample": len(group) < LOW_SAMPLE_N,
    }


def summarize_by(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in trades.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row.update(_summarize_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def long_short_pivot(trades: pd.DataFrame, zone_col: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (tf, zone), group in trades.groupby(["timeframe", zone_col], dropna=False, sort=False):
        long_g = group[group["direction"].eq("long")]
        short_g = group[group["direction"].eq("short")]
        long_metrics = _summarize_group(long_g) if len(long_g) else None
        short_metrics = _summarize_group(short_g) if len(short_g) else None
        all_metrics = _summarize_group(group)
        row = {
            "timeframe": tf,
            zone_col: zone,
            "n_trades_total": all_metrics["n_trades"],
            "n_long": long_metrics["n_trades"] if long_metrics else 0,
            "n_short": short_metrics["n_trades"] if short_metrics else 0,
            "long_win_rate_pct": long_metrics["win_rate_pct"] if long_metrics else np.nan,
            "short_win_rate_pct": short_metrics["win_rate_pct"] if short_metrics else np.nan,
            "long_avg_net_pct": long_metrics["avg_net_return_pct"] if long_metrics else np.nan,
            "short_avg_net_pct": short_metrics["avg_net_return_pct"] if short_metrics else np.nan,
            "long_total_compounded_pct": long_metrics["total_net_compounded_pct"] if long_metrics else np.nan,
            "short_total_compounded_pct": short_metrics["total_net_compounded_pct"] if short_metrics else np.nan,
            "long_profit_factor": long_metrics["profit_factor"] if long_metrics else np.nan,
            "short_profit_factor": short_metrics["profit_factor"] if short_metrics else np.nan,
            "long_max_dd_pct": long_metrics["max_drawdown_pct"] if long_metrics else np.nan,
            "short_max_dd_pct": short_metrics["max_drawdown_pct"] if short_metrics else np.nan,
            "long_minus_short_avg_net_pct": (
                (long_metrics["avg_net_return_pct"] if long_metrics else np.nan)
                - (short_metrics["avg_net_return_pct"] if short_metrics else np.nan)
            ),
            "all_avg_net_pct": all_metrics["avg_net_return_pct"],
            "low_sample_long": (long_metrics["n_trades"] if long_metrics else 0) < LOW_SAMPLE_N,
            "low_sample_short": (short_metrics["n_trades"] if short_metrics else 0) < LOW_SAMPLE_N,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out_dir: Path, trades: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> None:
    def fmt(df: pd.DataFrame, cols: list[str], n: int = 12) -> str:
        if df.empty:
            return "_no rows_"
        cols = [c for c in cols if c in df.columns]
        return df[cols].head(n).to_markdown(index=False, floatfmt=".2f")

    by_dir_tf = summaries["by_direction_timeframe.csv"]
    by_zone4 = summaries["by_zone4_long_short.csv"]
    by_ppo_sign = summaries["by_ppo_sign_long_short.csv"]
    by_ppo_decile = summaries["by_ppo_decile_long_short.csv"]
    by_ppo_exp = summaries["by_ppo_expanding_bin_long_short.csv"]

    parts: list[str] = []
    parts.append("# PPO Zone × Long/Short Flip Backtest Report\n")
    parts.append(
        "## 1. 백테스트 규칙\n"
        "- 시그널: `raw_direction[i] = sign(ppo_hist[i] - ppo_hist[i-1])`가 직전 바와 달라지는 캔들 (반대 움직임 1개 캔들).\n"
        "- 진입/청산: 시그널 캔들 다음 캔들이 종료된 시점의 종가에서 새 방향으로 플립.\n"
        "- 항상 시장에 포지션 보유 (long↔short 스왑). 다음 시그널까지 보유.\n"
        "- 비용: 라운드트립 수수료 0.08% + 진입/청산 슬리피지 각 0.02% (총 0.12%)을 net 수익률에서 차감.\n"
        "- PPO 위치는 시그널 캔들의 종가 기준 PPO/PPO hist 값.\n"
    )
    parts.append("## 2. 타임프레임별 long/short 종합 결과\n")
    parts.append(
        fmt(
            by_dir_tf,
            [
                "timeframe",
                "direction",
                "n_trades",
                "win_rate_pct",
                "avg_net_return_pct",
                "median_net_return_pct",
                "total_net_compounded_pct",
                "profit_factor",
                "max_drawdown_pct",
                "avg_holding_bars",
            ],
            n=20,
        )
    )

    parts.append("\n## 3. PPO sign zone (PPO≥0 vs PPO<0) 별 long vs short\n")
    parts.append(
        fmt(
            by_ppo_sign.sort_values(["timeframe", "ppo_sign_zone"]),
            [
                "timeframe",
                "ppo_sign_zone",
                "n_long",
                "n_short",
                "long_avg_net_pct",
                "short_avg_net_pct",
                "long_minus_short_avg_net_pct",
                "long_win_rate_pct",
                "short_win_rate_pct",
                "long_total_compounded_pct",
                "short_total_compounded_pct",
            ],
            n=20,
        )
    )

    parts.append("\n## 4. zone4 (PPO sign × hist sign) 별 long vs short\n")
    parts.append(
        fmt(
            by_zone4.sort_values(["timeframe", "zone4"]),
            [
                "timeframe",
                "zone4",
                "n_long",
                "n_short",
                "long_avg_net_pct",
                "short_avg_net_pct",
                "long_minus_short_avg_net_pct",
                "long_win_rate_pct",
                "short_win_rate_pct",
                "long_total_compounded_pct",
                "short_total_compounded_pct",
            ],
            n=40,
        )
    )

    parts.append("\n## 5. PPO 전체 히스토리 decile (1=최저 ~ 10=최고) 별 long vs short\n")
    parts.append(
        fmt(
            by_ppo_decile.sort_values(["timeframe", "ppo_decile"]),
            [
                "timeframe",
                "ppo_decile",
                "n_long",
                "n_short",
                "long_avg_net_pct",
                "short_avg_net_pct",
                "long_minus_short_avg_net_pct",
                "long_win_rate_pct",
                "short_win_rate_pct",
            ],
            n=80,
        )
    )

    parts.append("\n## 6. PPO expanding-quantile bucket (look-ahead 없는 분위) 별 long vs short\n")
    parts.append(
        fmt(
            by_ppo_exp.sort_values(["timeframe", "ppo_expanding_bin"]),
            [
                "timeframe",
                "ppo_expanding_bin",
                "n_long",
                "n_short",
                "long_avg_net_pct",
                "short_avg_net_pct",
                "long_minus_short_avg_net_pct",
                "long_win_rate_pct",
                "short_win_rate_pct",
            ],
            n=40,
        )
    )

    parts.append("\n## 7. 해석 가이드\n")
    parts.append(
        "- `long_minus_short_avg_net_pct` 값이 양수면 해당 PPO 위치에서 long이 short보다 평균 수익이 높음.\n"
        "- `low_sample` 표본은 30건 미만이므로 결론을 내릴 때 제외 권장.\n"
        "- decile은 전체 데이터로 산출했기에 사후적 정보가 섞일 수 있음 (해석용). 실거래용으로는 6장 expanding-quantile bucket이 더 적절.\n"
        "- 수익률은 Bitcoin 무한 보유의 시장 베타가 포함되어 있음. 진정한 alpha 분석을 원하면 BTC buy-and-hold 대비로 추가 비교 필요.\n"
    )
    (out_dir / "PPO_zone_long_short_flip_report.md").write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-mode", choices=("ffill", "zero"), default="ffill")
    parser.add_argument("--timeframes", nargs="*", default=list(TIMEFRAMES), choices=TIMEFRAMES)
    args = parser.parse_args()

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trades: list[pd.DataFrame] = []
    for tf in args.timeframes:
        candles = load_candles(tf, args.zero_mode)
        trades = build_flip_trades(candles, tf)
        if not trades.empty:
            all_trades.append(trades)
        print(f"{tf}: candles={len(candles):,}  trades={len(trades):,}")

    if not all_trades:
        print("No trades produced.")
        return 1

    trades_df = pd.concat(all_trades, ignore_index=True).sort_values(["timeframe", "entry_time"]).reset_index(drop=True)
    trades_df.to_csv(out_dir / "00_all_trades.csv", index=False, encoding="utf-8-sig")

    summaries: dict[str, pd.DataFrame] = {}
    summaries["by_direction_timeframe.csv"] = summarize_by(trades_df, ["timeframe", "direction"])
    summaries["by_ppo_sign_long_short.csv"] = long_short_pivot(trades_df, "ppo_sign_zone")
    summaries["by_hist_sign_long_short.csv"] = long_short_pivot(trades_df, "hist_sign_zone")
    summaries["by_zone4_long_short.csv"] = long_short_pivot(trades_df, "zone4")
    summaries["by_ppo_decile_long_short.csv"] = long_short_pivot(trades_df, "ppo_decile")
    summaries["by_hist_decile_long_short.csv"] = long_short_pivot(trades_df, "hist_decile")
    summaries["by_ppo_expanding_bin_long_short.csv"] = long_short_pivot(trades_df, "ppo_expanding_bin")
    summaries["by_hist_expanding_bin_long_short.csv"] = long_short_pivot(trades_df, "hist_expanding_bin")

    for name, frame in summaries.items():
        frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig")

    write_report(out_dir, trades_df, summaries)
    print(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
