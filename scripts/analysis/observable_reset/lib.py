from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import PROJECT_PATHS  # noqa: E402


TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d", "1w")
TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
DEFAULT_HORIZONS = (1, 2, 3, 5, 8, 13, 21, 34)
DEFAULT_ENTRY_OFFSETS = (2, 3)
DEFAULT_COST_PCT = 0.10
OUT_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "observable_reset"
FEATURE_PATH = OUT_DIR / "observable_features.parquet"
SPLIT_PATH = OUT_DIR / "time_split.json"
TRAIN_PATH = OUT_DIR / "train_features.parquet"
TEST_PATH = OUT_DIR / "test_features.locked.parquet"

OBSERVABLE_FEATURES = [
    "hist",
    "hist_sign",
    "hist_slope_3",
    "hist_streak",
    "rsi",
    "rsi_slope_3",
    "ppo",
    "ppo_hist",
    "dist_ma7",
    "dist_ma25",
    "dist_ma99",
    "ma_alignment",
    "atr_over_price",
    "vol_regime",
    "nup_legal_hist",
    "nup_legal_count",
]


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def market_path(timeframe: str) -> Path:
    candidates = [
        PROJECT_PATHS.base_data_dir / f"BTCUSD_{timeframe}.csv",
        PROJECT_PATHS.base_data_dir / f"BTCUSDT_{timeframe}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing market file for {timeframe}: tried {candidates}")


def read_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def load_market(timeframe: str) -> pd.DataFrame:
    path = market_path(timeframe)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [
        c for c in [
            "date", "timestamp", "open_time",
            "open", "high", "low", "close",
            "macd_hist", "rsi", "ppo", "ppo_hist",
            "ma_7", "ma_25", "ma_99",
        ]
        if c in header
    ]
    df = pd.read_csv(path, usecols=usecols)
    ts_col = next((c for c in ("date", "timestamp", "open_time") if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"{path} has no timestamp column")
    df = df.rename(columns={ts_col: "timestamp", "ma_7": "ma7", "ma_25": "ma25", "ma_99": "ma99"})
    df["timestamp"] = read_ts(df["timestamp"])
    for col in [c for c in df.columns if c != "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    required = {"timestamp", "open", "high", "low", "close", "macd_hist"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return df.dropna(subset=list(required)).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def signed_nonzero(series: pd.Series) -> pd.Series:
    signs = np.sign(pd.to_numeric(series, errors="coerce")).replace(0, np.nan).ffill()
    return signs


def add_observable_columns(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    out = df.copy()
    out["hist"] = out["macd_hist"]
    out["hist_delta"] = out["hist"].diff()
    out["hist_delta_sign"] = signed_nonzero(out["hist_delta"])
    out["hist_sign"] = np.sign(out["hist"]).replace(0, np.nan).ffill()
    out["hist_slope_3"] = out["hist_delta"].rolling(3, min_periods=3).mean()
    streak_change = out["hist_delta_sign"].ne(out["hist_delta_sign"].shift()).cumsum()
    out["hist_streak"] = out.groupby(streak_change).cumcount() + 1
    if "rsi" in out.columns:
        out["rsi_slope_3"] = out["rsi"].diff().rolling(3, min_periods=3).mean()
    else:
        out["rsi"] = np.nan
        out["rsi_slope_3"] = np.nan
    for ma_col, window in [("ma7", 7), ("ma25", 25), ("ma99", 99)]:
        if ma_col not in out.columns:
            out[ma_col] = out["close"].rolling(window, min_periods=window).mean()
    out["dist_ma7"] = (out["close"] / out["ma7"] - 1.0) * 100.0
    out["dist_ma25"] = (out["close"] / out["ma25"] - 1.0) * 100.0
    out["dist_ma99"] = (out["close"] / out["ma99"] - 1.0) * 100.0
    out["ma_alignment"] = np.select(
        [out["ma7"].gt(out["ma25"]) & out["ma25"].gt(out["ma99"]), out["ma7"].lt(out["ma25"]) & out["ma25"].lt(out["ma99"])],
        [1, -1],
        default=0,
    )
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr_over_price"] = tr.rolling(14, min_periods=14).mean() / out["close"] * 100.0
    ret = out["close"].pct_change()
    trailing_vol = ret.rolling(50, min_periods=30).std()
    # Expanding percentile rank of the current trailing volatility using only values up to the current row.
    out["vol_regime"] = trailing_vol.expanding(min_periods=100).rank(pct=True)
    out["bar_close_time"] = out["timestamp"] + pd.to_timedelta(TF_SECONDS[timeframe], unit="s")
    return out


def add_legal_nup(base: pd.DataFrame, timeframe: str, markets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    order = list(TIMEFRAMES)
    higher = order[order.index(timeframe) + 1 :] if timeframe in order else []
    out = base.sort_values("entry_time").copy()
    signs: list[str] = []
    for tf in higher:
        src = markets.get(tf)
        if src is None or src.empty:
            continue
        hist_sign = np.sign(src["macd_hist"]).replace(0, np.nan).ffill()
        ctx = pd.DataFrame({"bar_close_time": src["bar_close_time"], f"legal_hist_sign_{tf}": hist_sign})
        out = pd.merge_asof(
            out.sort_values("entry_time"),
            ctx.dropna().sort_values("bar_close_time"),
            left_on="entry_time",
            right_on="bar_close_time",
            direction="backward",
        ).drop(columns=["bar_close_time"], errors="ignore")
        signs.append(f"legal_hist_sign_{tf}")
    if signs:
        out["nup_legal_hist"] = (out[signs] > 0).sum(axis=1)
        out["nup_legal_count"] = out[signs].notna().sum(axis=1)
    else:
        out["nup_legal_hist"] = np.nan
        out["nup_legal_count"] = 0
    return out


def max_drawdown(returns_pct: Iterable[float]) -> float:
    returns = pd.Series(list(returns_pct), dtype="float64").fillna(0.0) / 100.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min() * 100.0) if not dd.empty else np.nan


def profit_factor(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    wins = r[r > 0].sum()
    losses = r[r < 0].abs().sum()
    if losses == 0:
        return float("inf") if wins > 0 else np.nan
    return float(wins / losses)


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    phat = wins / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return (center - margin) * 100.0, (center + margin) * 100.0


def metrics_for_returns(frame: pd.DataFrame, return_col: str = "net_return") -> dict[str, Any]:
    r = pd.to_numeric(frame[return_col], errors="coerce").dropna()
    n = int(len(r))
    if n == 0:
        return {"n": 0}
    wins = int((r > 0).sum())
    lo, hi = wilson_ci(wins, n)
    return {
        "n": n,
        "win_rate_pct": wins / n * 100.0,
        "wilson_low_pct": lo,
        "wilson_high_pct": hi,
        "net_avg_pct": float(r.mean()),
        "net_median_pct": float(r.median()),
        "profit_factor": profit_factor(r),
        "expectancy_pct": float(r.mean()),
        "mdd_pct": max_drawdown(r),
    }


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in OBSERVABLE_FEATURES if col in df.columns]


def load_features(path: Path = FEATURE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing observable features: {path}")
    return pd.read_parquet(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def qbin(series: pd.Series, bins: int = 5) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    ranked = clean.rank(method="first")
    try:
        return pd.qcut(ranked, bins, labels=[f"q{i + 1}" for i in range(bins)]).astype(str)
    except ValueError:
        return pd.Series(["na"] * len(series), index=series.index)


def spearman_like(x: pd.Series, y: pd.Series) -> float:
    xy = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(xy) < 3:
        return np.nan
    return float(xy["x"].rank().corr(xy["y"].rank()))


def bh_fdr(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype="float64")
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = m - rank + 1
        val = min(prev, p_values[idx] * m / original_rank)
        adjusted[idx] = val
        prev = val
    return adjusted.tolist()
