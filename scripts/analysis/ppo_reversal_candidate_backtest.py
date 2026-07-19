from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import PROJECT_PATHS  # noqa: E402


TIMEFRAMES = ("15m", "1h", "4h", "1d")
ENTRY_TIMEFRAMES = ("15m", "1h", "4h")
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
MAX_TOLERANCE = 2
MIN_CYCLE_DURATION = 3
LOW_SAMPLE_N = 30
ROUND_TRIP_FEE_PCT = 0.08
SLIPPAGE_PER_SIDE_PCT = 0.02
MAX_BACKTEST_ENTRIES_PER_STRATEGY = 5000
EXTREME_THRESHOLDS = {
    "15m": {"long": -0.42, "short": 0.42},
    "1h": {"long": -0.90, "short": 0.93},
    "4h": {"long": -2.03, "short": 2.06},
    "1d": {"long": -4.07, "short": 5.43},
}
FIXED_BARS = {"15m": (4, 8, 16, 32), "1h": (2, 4, 8, 16), "4h": (2, 4, 8, 12)}
TP_SL_GRID = {
    "15m": {"tp": (0.5, 0.8, 1.2, 1.8), "sl": (0.3, 0.5, 0.8, 1.2)},
    "1h": {"tp": (1.0, 1.5, 2.5, 4.0), "sl": (0.6, 1.0, 1.5, 2.0)},
    "4h": {"tp": (2.0, 3.5, 5.0, 8.0), "sl": (1.0, 1.8, 2.5, 4.0)},
}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    entry_tf: str
    required_align: tuple[str, ...] = ()
    supportive_4h_extreme: bool = False
    required_upper: tuple[str, ...] = ()


STRATEGIES = (
    StrategySpec("S0_15m_raw_reversal", "15m"),
    StrategySpec("S1_15m_1h_bias", "15m", required_align=("1h",)),
    StrategySpec("S2_15m_4h_bias", "15m", required_align=("4h",)),
    StrategySpec("S3_15m_1h_4h_alignment", "15m", required_align=("1h", "4h")),
    StrategySpec("S4_15m_1h_4h_1d_alignment", "15m", required_align=("1h", "4h", "1d")),
    StrategySpec("S5_15m_4h_extreme", "15m", supportive_4h_extreme=True),
    StrategySpec("S6_1h_raw_reversal", "1h"),
    StrategySpec("S7_1h_4h_bias", "1h", required_align=("4h",)),
    StrategySpec("S8_1h_4h_1d_alignment", "1h", required_align=("4h", "1d")),
    StrategySpec("S9_4h_1d_bias", "4h", required_align=("1d",)),
)


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_reversal_candidate_backtest"


def _read_timestamp(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def _sign_label(value: Any) -> float:
    label = str(value or "").strip().lower()
    if label in {"up", "long", "1", "+1"}:
        return 1.0
    if label in {"down", "short", "-1"}:
        return -1.0
    return np.nan


def _candidate_sign(direction: Any) -> int:
    return 1 if str(direction) == "long" else -1


def _tf_delta(timeframe: str) -> pd.Timedelta:
    return pd.to_timedelta(TF_SECONDS[timeframe], unit="s")


def _safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _zone4(ppo: pd.Series, hist: pd.Series) -> pd.Series:
    ppo_sign = np.where(pd.to_numeric(ppo, errors="coerce") >= 0, "ppo_pos", "ppo_neg")
    hist_sign = np.where(pd.to_numeric(hist, errors="coerce") >= 0, "hist_pos", "hist_neg")
    return pd.Series(ppo_sign + "__" + hist_sign, index=ppo.index)


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


def _add_full_deciles(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    ranked = pd.to_numeric(df[col], errors="coerce").rank(method="first")
    try:
        df[f"{prefix}_decile"] = pd.qcut(ranked, 10, labels=False).astype("float") + 1
    except ValueError:
        df[f"{prefix}_decile"] = np.nan
    df[f"{prefix}_bin"] = _bin_from_decile(df[f"{prefix}_decile"])
    return df


def _expanding_bin(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    shifted = values.shift(1)
    q10 = shifted.expanding(min_periods=200).quantile(0.10)
    q20 = shifted.expanding(min_periods=200).quantile(0.20)
    q80 = shifted.expanding(min_periods=200).quantile(0.80)
    q90 = shifted.expanding(min_periods=200).quantile(0.90)
    return pd.Series(
        np.select(
            [values <= q10, values <= q20, values >= q90, values >= q80],
            ["bottom10", "bottom20", "top10", "top20"],
            default="mid",
        ),
        index=series.index,
    )


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
        PROJECT_PATHS.asset_cycle_dir("btc") / f"cycles_{timeframe}.parquet",
        PROJECT_PATHS.cycle_structured_dir / "btc" / f"cycles_{timeframe}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing cycle parquet for {timeframe}: tried {candidates}")


def load_candles(timeframe: str, zero_mode: str) -> pd.DataFrame:
    path = _raw_market_path(timeframe)
    usecols = [col for col in pd.read_csv(path, nrows=0).columns if col in {"date", "timestamp", "open_time", "open", "high", "low", "close", "ppo", "ppo_hist"}]
    df = pd.read_csv(path, usecols=usecols).copy()
    ts_col = next((col for col in ("timestamp", "open_time", "date") if col in df.columns), None)
    if ts_col is None:
        raise ValueError(f"{path} has no timestamp/date/open_time column")
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
    df = _add_full_deciles(df, "ppo", "ppo")
    df = _add_full_deciles(df, "ppo_hist", "hist")
    df["ppo_expanding_bin"] = _expanding_bin(df["ppo"])
    df["hist_expanding_bin"] = _expanding_bin(df["ppo_hist"])
    for bars in (1, 3, 5, 10):
        df[f"forward_return_{bars}"] = (df["close"].shift(-bars) / df["close"] - 1.0) * 100.0
    return df


def load_cycles(timeframe: str) -> pd.DataFrame:
    df = pd.read_parquet(_cycle_path(timeframe)).copy()
    df["start_date"] = _read_timestamp(df["start_date"])
    df["end_date"] = _read_timestamp(df["end_date"])
    df = df.dropna(subset=["start_date", "end_date"]).sort_values("start_date").reset_index(drop=True)
    df["cycle_sign"] = df["cycle_type"].map(_sign_label)
    df["end_exclusive"] = df["end_date"] + _tf_delta(timeframe)
    df["duration_candles"] = pd.to_numeric(df.get("duration_candles"), errors="coerce")
    return df


def add_cycle_state(candles: pd.DataFrame, cycles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    state_cols = ["cycle_id", "cycle_type", "cycle_sign", "cycle_start", "cycle_end", "cycle_progress"]
    if cycles.empty:
        for col in state_cols:
            candles[col] = np.nan
        return candles
    intervals = cycles[["cycle_id", "cycle_type", "cycle_sign", "start_date", "end_exclusive", "duration_candles"]].copy()
    merged = pd.merge_asof(
        candles.sort_values("timestamp"),
        intervals.sort_values("start_date"),
        left_on="timestamp",
        right_on="start_date",
        direction="backward",
    )
    in_cycle = merged["timestamp"].lt(merged["end_exclusive"])
    merged.loc[~in_cycle, ["cycle_id", "cycle_type", "cycle_sign", "start_date", "end_exclusive", "duration_candles"]] = np.nan
    elapsed = (merged["timestamp"] - merged["start_date"]) / _tf_delta(timeframe)
    denom = pd.to_numeric(merged["duration_candles"], errors="coerce").replace(0, np.nan)
    merged["cycle_progress"] = (elapsed / denom).clip(lower=0, upper=1)
    merged = merged.rename(columns={"start_date": "cycle_start", "end_exclusive": "cycle_end"})
    return merged


def candidate_run_lengths(raw_direction: pd.Series) -> np.ndarray:
    raw = raw_direction.to_numpy()
    out = np.zeros(len(raw), dtype=np.int64)
    idx = 0
    while idx < len(raw):
        direction = raw[idx]
        end = idx + 1
        while end < len(raw) and raw[end] == direction:
            end += 1
        out[idx:end] = end - idx
        idx = end
    return out


def build_candidates(candles: dict[str, pd.DataFrame], cycles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for tf, df in candles.items():
        base = df.copy()
        base["prev_active_dir"] = base["raw_direction"].replace(0, np.nan).ffill().shift(1)
        base["candidate_run_length_afterward"] = candidate_run_lengths(base["raw_direction"])
        mask = (
            base["raw_direction"].isin([-1, 1])
            & base["prev_active_dir"].isin([-1, 1])
            & (base["raw_direction"] == -base["prev_active_dir"])
        )
        cand = base.loc[mask].copy()
        cand["candidate_tf"] = tf
        cand["candidate_direction"] = np.where(cand["raw_direction"] > 0, "long", "short")
        cand["close_at_entry"] = cand["close"]
        cand["rule_based_true_reversal"] = cand["candidate_run_length_afterward"] > MAX_TOLERANCE
        cand["rule_based_noise"] = ~cand["rule_based_true_reversal"]
        start_lookup = cycles[tf][["start_date", "cycle_type", "cycle_sign", "cycle_id"]].rename(
            columns={
                "start_date": "cycle_label_start_date",
                "cycle_type": "cycle_label_type",
                "cycle_sign": "cycle_label_sign",
                "cycle_id": "cycle_label_id",
            }
        )
        cand = pd.merge_asof(
            cand.sort_values("timestamp"),
            start_lookup.sort_values("cycle_label_start_date"),
            left_on="timestamp",
            right_on="cycle_label_start_date",
            direction="nearest",
            tolerance=_tf_delta(tf),
        )
        cand_sign = cand["candidate_direction"].map({"long": 1.0, "short": -1.0})
        close_to_start = (cand["timestamp"] - cand["cycle_label_start_date"]).abs().le(_tf_delta(tf))
        cand["cycle_based_true_reversal"] = close_to_start & cand["cycle_label_sign"].eq(cand_sign)
        cand["cycle_based_noise"] = ~cand["cycle_based_true_reversal"]
        cand["label_mismatch"] = cand["rule_based_true_reversal"] != cand["cycle_based_true_reversal"]
        cand["is_true_reversal"] = cand["rule_based_true_reversal"].astype(int)
        cand["is_noise"] = cand["rule_based_noise"].astype(int)
        for bars in (1, 3, 5, 10):
            signed = cand[f"forward_return_{bars}"] * cand_sign
            cand[f"candidate_forward_return_{bars}"] = signed
        frames.append(cand)
    return pd.concat(frames, ignore_index=True).sort_values(["candidate_tf", "timestamp"]).reset_index(drop=True)


def upper_timeframes(candidate_tf: str) -> tuple[str, ...]:
    idx = TIMEFRAMES.index(candidate_tf)
    return TIMEFRAMES[idx + 1 :]


def add_upper_features(candidates: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out_frames: list[pd.DataFrame] = []
    for tf, group in candidates.groupby("candidate_tf", sort=False):
        enriched = group.sort_values("timestamp").copy()
        for upper_tf in upper_timeframes(tf):
            upper = candles[upper_tf].copy().sort_values("timestamp")
            upper["available_at"] = upper["timestamp"] + _tf_delta(upper_tf)
            keep = [
                "available_at",
                "ppo",
                "ppo_hist",
                "ppo_hist_diff",
                "zone4",
                "ppo_bin",
                "hist_bin",
                "ppo_expanding_bin",
                "hist_expanding_bin",
                "cycle_type",
                "cycle_sign",
                "cycle_progress",
            ]
            upper_keep = upper[keep].rename(
                columns={
                    "ppo": f"upper_{upper_tf}_ppo",
                    "ppo_hist": f"upper_{upper_tf}_hist",
                    "ppo_hist_diff": f"upper_{upper_tf}_hist_diff",
                    "zone4": f"upper_{upper_tf}_zone4",
                    "ppo_bin": f"upper_{upper_tf}_ppo_bin",
                    "hist_bin": f"upper_{upper_tf}_hist_bin",
                    "ppo_expanding_bin": f"upper_{upper_tf}_ppo_expanding_bin",
                    "hist_expanding_bin": f"upper_{upper_tf}_hist_expanding_bin",
                    "cycle_type": f"upper_{upper_tf}_cycle_direction",
                    "cycle_sign": f"upper_{upper_tf}_cycle_sign",
                    "cycle_progress": f"upper_{upper_tf}_cycle_progress",
                }
            )
            enriched = pd.merge_asof(
                enriched.sort_values("timestamp"),
                upper_keep,
                left_on="timestamp",
                right_on="available_at",
                direction="backward",
            ).drop(columns=["available_at"])
        out_frames.append(enriched)
    result = pd.concat(out_frames, ignore_index=True).sort_values(["candidate_tf", "timestamp"]).reset_index(drop=True)
    result = add_bias_features(result)
    return result


def add_bias_features(candidates: pd.DataFrame) -> pd.DataFrame:
    for upper_tf in ("1h", "4h", "1d"):
        hist_col = f"upper_{upper_tf}_hist"
        ppo_col = f"upper_{upper_tf}_ppo"
        if hist_col not in candidates.columns:
            candidates[f"upper_{upper_tf}_align"] = 0
            continue
        sign = candidates["candidate_direction"].map({"long": 1, "short": -1})
        upper_bias = np.select(
            [pd.to_numeric(candidates[hist_col], errors="coerce") < 0, pd.to_numeric(candidates[hist_col], errors="coerce") > 0],
            [1, -1],
            default=0,
        )
        candidates[f"upper_{upper_tf}_align"] = (upper_bias == sign).astype(int)
        thresholds = EXTREME_THRESHOLDS[upper_tf]
        supportive_long = (candidates["candidate_direction"].eq("long")) & (candidates[ppo_col] <= thresholds["long"]) & (candidates[hist_col] < 0)
        supportive_short = (candidates["candidate_direction"].eq("short")) & (candidates[ppo_col] >= thresholds["short"]) & (candidates[hist_col] > 0)
        contra_long = (candidates["candidate_direction"].eq("long")) & (candidates[ppo_col] >= thresholds["short"]) & (candidates[hist_col] > 0)
        contra_short = (candidates["candidate_direction"].eq("short")) & (candidates[ppo_col] <= thresholds["long"]) & (candidates[hist_col] < 0)
        candidates[f"has_supportive_{upper_tf}_extreme"] = (supportive_long | supportive_short).astype(int)
        candidates[f"has_contra_{upper_tf}_extreme"] = (contra_long | contra_short).astype(int)
    align_cols = [col for col in ("upper_1h_align", "upper_4h_align", "upper_1d_align") if col in candidates.columns]
    candidates["align_count"] = candidates[align_cols].sum(axis=1) if align_cols else 0
    major_cols = [col for col in ("upper_4h_align", "upper_1d_align") if col in candidates.columns]
    candidates["major_align_count"] = candidates[major_cols].sum(axis=1) if major_cols else 0
    candidates["has_supportive_4h_extreme"] = candidates.get("has_supportive_4h_extreme", 0)
    candidates["has_contra_4h_extreme"] = candidates.get("has_contra_4h_extreme", 0)
    return candidates


def _summary_counts(group: pd.DataFrame) -> pd.Series:
    n = len(group)
    true_count = int(group["is_true_reversal"].sum())
    noise_count = int(group["is_noise"].sum())
    return pd.Series(
        {
            "n": n,
            "true_reversal_count": true_count,
            "noise_count": noise_count,
            "true_reversal_pct": true_count / n * 100 if n else np.nan,
            "noise_pct": noise_count / n * 100 if n else np.nan,
            "low_sample": n < LOW_SAMPLE_N,
        }
    )


def write_noise_summaries(candidates: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    by_tf = candidates.groupby(["candidate_tf", "candidate_direction"], dropna=False).apply(_summary_counts, include_groups=False).reset_index()
    outputs["21_candidate_noise_by_tf.csv"] = by_tf

    by_zone = candidates.groupby(["candidate_tf", "candidate_direction", "zone4"], dropna=False).apply(_summary_counts, include_groups=False).reset_index()
    for bars in (1, 3, 5, 10):
        ret = candidates.groupby(["candidate_tf", "candidate_direction", "zone4"], dropna=False)[f"candidate_forward_return_{bars}"].mean().reset_index(name=f"avg_forward_return_{bars}")
        by_zone = by_zone.merge(ret, on=["candidate_tf", "candidate_direction", "zone4"], how="left")
    outputs["22_candidate_noise_by_zone.csv"] = by_zone

    by_bins = candidates.groupby(["candidate_tf", "candidate_direction", "ppo_bin", "hist_bin"], dropna=False).apply(_summary_counts, include_groups=False).reset_index()
    perf = candidates.groupby(["candidate_tf", "candidate_direction", "ppo_bin", "hist_bin"], dropna=False)["candidate_forward_return_5"].agg(
        trade_return_mean="mean",
        win_rate=lambda s: float((s > 0).mean() * 100),
    ).reset_index()
    by_bins = by_bins.merge(perf, on=["candidate_tf", "candidate_direction", "ppo_bin", "hist_bin"], how="left")
    by_bins["long_return_mean"] = np.where(by_bins["candidate_direction"].eq("long"), by_bins["trade_return_mean"], np.nan)
    by_bins["short_return_mean"] = np.where(by_bins["candidate_direction"].eq("short"), by_bins["trade_return_mean"], np.nan)
    outputs["23_candidate_noise_by_ppo_hist_bins.csv"] = by_bins

    rows: list[pd.DataFrame] = []
    for tf in ("1h", "4h", "1d"):
        align_col = f"upper_{tf}_align"
        if align_col not in candidates.columns:
            continue
        cols = ["candidate_tf", "candidate_direction", align_col, "has_supportive_4h_extreme", "has_contra_4h_extreme"]
        part = candidates.groupby(cols, dropna=False).apply(_summary_counts, include_groups=False).reset_index()
        part["upper_tf_set"] = tf
        perf = candidates.groupby(cols, dropna=False)["candidate_forward_return_5"].agg(
            avg_trade_return="mean",
            win_rate=lambda s: float((s > 0).mean() * 100),
        ).reset_index()
        part = part.merge(perf, on=cols, how="left")
        part = part.rename(columns={align_col: "align_count"})
        rows.append(part)
    upper_summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    outputs["24_candidate_noise_by_upper_tf.csv"] = upper_summary

    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
    return outputs


def train_noise_models(candidates: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:
        result = pd.DataFrame([{"model": "unavailable", "error": str(exc)}])
        result.to_csv(out_dir / "25_noise_model_feature_importance.csv", index=False, encoding="utf-8-sig")
        return result

    rows: list[dict[str, Any]] = []
    feature_cols = [
        "ppo",
        "ppo_hist",
        "ppo_hist_diff",
        "zone4",
        "ppo_bin",
        "hist_bin",
        "upper_1h_ppo",
        "upper_1h_hist",
        "upper_1h_zone4",
        "upper_4h_ppo",
        "upper_4h_hist",
        "upper_4h_zone4",
        "upper_1d_ppo",
        "upper_1d_hist",
        "upper_1d_zone4",
        "align_count",
        "major_align_count",
        "has_supportive_4h_extreme",
        "has_contra_4h_extreme",
    ]
    feature_cols = [col for col in feature_cols if col in candidates.columns]
    data = candidates.dropna(subset=["is_true_reversal"]).sort_values("timestamp").copy()
    if data["is_true_reversal"].nunique() < 2 or len(data) < 200:
        result = pd.DataFrame([{"model": "skipped", "reason": "not enough labelled class diversity"}])
        result.to_csv(out_dir / "25_noise_model_feature_importance.csv", index=False, encoding="utf-8-sig")
        return result

    split = int(len(data) * 0.7)
    train, test = data.iloc[:split], data.iloc[split:]
    y_train = train["is_true_reversal"].astype(int)
    y_test = test["is_true_reversal"].astype(int)
    numeric = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]
    categorical = [col for col in feature_cols if col not in numeric]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=160, max_depth=7, min_samples_leaf=30, random_state=42, class_weight="balanced_subsample", n_jobs=-1),
    }
    for model_name, estimator in models.items():
        pipe = Pipeline([("pre", pre), ("model", estimator)])
        pipe.fit(train[feature_cols], y_train)
        probs = pipe.predict_proba(test[feature_cols])[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probs) if y_test.nunique() == 2 else np.nan
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
        names = pipe.named_steps["pre"].get_feature_names_out()
        if model_name == "random_forest":
            importances = pipe.named_steps["model"].feature_importances_
        else:
            importances = np.abs(pipe.named_steps["model"].coef_[0])
        order = np.argsort(importances)[::-1][:40]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "model": model_name,
                    "rank": rank,
                    "feature": names[idx],
                    "importance": float(importances[idx]),
                    "auc": float(auc),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "confusion_matrix": json.dumps(cm),
                    "train_n": len(train),
                    "test_n": len(test),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "25_noise_model_feature_importance.csv", index=False, encoding="utf-8-sig")
    return result


def _entry_filter(candidates: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    df = candidates[candidates["candidate_tf"].eq(spec.entry_tf)].copy()
    for upper_tf in spec.required_align:
        df = df[df.get(f"upper_{upper_tf}_align", 0).eq(1)]
    if spec.supportive_4h_extreme:
        df = df[df["has_supportive_4h_extreme"].eq(1)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) > MAX_BACKTEST_ENTRIES_PER_STRATEGY:
        # Keep the sample chronological and spread across the full history so slower
        # path-dependent exit grids stay reproducible on this research dataset.
        take = np.linspace(0, len(df) - 1, MAX_BACKTEST_ENTRIES_PER_STRATEGY).round().astype(int)
        df = df.iloc[np.unique(take)].reset_index(drop=True)
    return df


def _direction_return(entry: float, exit_: float, direction: str) -> float:
    if pd.isna(entry) or pd.isna(exit_) or entry == 0:
        return np.nan
    sign = _candidate_sign(direction)
    return (exit_ / entry - 1.0) * 100.0 * sign


def _find_next_candidate(candidates: pd.DataFrame, entry_time: pd.Timestamp, direction: str, true_only: bool = False) -> pd.Series | None:
    opp = "short" if direction == "long" else "long"
    subset = candidates[(candidates["timestamp"] > entry_time) & candidates["candidate_direction"].eq(opp)]
    if true_only:
        subset = subset[subset["is_true_reversal"].eq(1)]
    if subset.empty:
        return None
    return subset.iloc[0]


def _find_hist_cross(candles: pd.DataFrame, entry_time: pd.Timestamp, direction: str) -> pd.Series | None:
    future = candles[candles["timestamp"] > entry_time]
    if direction == "long":
        future = future[future["ppo_hist"] >= 0]
    else:
        future = future[future["ppo_hist"] <= 0]
    if future.empty:
        return None
    return future.iloc[0]


def _fixed_bar_exit(candles: pd.DataFrame, entry_time: pd.Timestamp, bars: int) -> pd.Series | None:
    pos = candles.index[candles["timestamp"] == entry_time]
    if len(pos) == 0:
        future = candles[candles["timestamp"] > entry_time]
        return future.iloc[min(bars - 1, len(future) - 1)] if not future.empty else None
    idx = int(pos[0]) + bars
    if idx >= len(candles):
        idx = len(candles) - 1
    return candles.iloc[idx] if idx > int(pos[0]) else None


def _tp_sl_exit(candles: pd.DataFrame, entry_time: pd.Timestamp, entry_price: float, direction: str, tp: float, sl: float) -> tuple[pd.Series | None, str]:
    future = candles[candles["timestamp"] > entry_time]
    sign = _candidate_sign(direction)
    for _, row in future.iterrows():
        high_ret = _direction_return(entry_price, row["high"] if sign > 0 else row["low"], direction)
        low_ret = _direction_return(entry_price, row["low"] if sign > 0 else row["high"], direction)
        if low_ret <= -sl:
            item = row.copy()
            item["close"] = entry_price * (1 - sl / 100 * sign)
            return item, f"tp_sl_tp{tp}_sl{sl}_stop"
        if high_ret >= tp:
            item = row.copy()
            item["close"] = entry_price * (1 + tp / 100 * sign)
            return item, f"tp_sl_tp{tp}_sl{sl}_take"
    return None, f"tp_sl_tp{tp}_sl{sl}_eod"


def _trailing_exit(candles: pd.DataFrame, entry_time: pd.Timestamp, entry_price: float, direction: str, trail_pct: float) -> tuple[pd.Series | None, str]:
    future = candles[candles["timestamp"] > entry_time]
    sign = _candidate_sign(direction)
    best = entry_price
    for _, row in future.iterrows():
        if sign > 0:
            best = max(best, row["high"])
            stop = best * (1 - trail_pct / 100)
            if row["low"] <= stop:
                item = row.copy()
                item["close"] = stop
                return item, f"trailing_stop_{trail_pct}"
        else:
            best = min(best, row["low"])
            stop = best * (1 + trail_pct / 100)
            if row["high"] >= stop:
                item = row.copy()
                item["close"] = stop
                return item, f"trailing_stop_{trail_pct}"
    return None, f"trailing_stop_{trail_pct}_eod"


def _exit_variants(entry: pd.Series, candidates: dict[str, pd.DataFrame], candles: dict[str, pd.DataFrame]) -> list[tuple[str, pd.Timestamp, float]]:
    entry_tf = str(entry["candidate_tf"])
    direction = str(entry["candidate_direction"])
    entry_time = entry["timestamp"]
    entry_price = float(entry["close_at_entry"])
    variants: list[tuple[str, pd.Timestamp, float]] = []

    same_cands = candidates[entry_tf]
    same_cand = _find_next_candidate(same_cands, entry_time, direction, false := False)
    if same_cand is not None:
        variants.append(("same_tf_opposite_candidate", same_cand["timestamp"], same_cand["close_at_entry"]))
    same_true = _find_next_candidate(same_cands, entry_time, direction, true_only=True)
    if same_true is not None:
        variants.append(("same_tf_opposite_true_reversal_confirmed", same_true["timestamp"], same_true["close_at_entry"]))

    upper = upper_timeframes(entry_tf)
    if upper:
        upper_tf = upper[0]
        up_cand = _find_next_candidate(candidates[upper_tf], entry_time, direction, true_only=False)
        if up_cand is not None:
            variants.append((f"{upper_tf}_opposite_candidate", up_cand["timestamp"], up_cand["close_at_entry"]))
        up_true = _find_next_candidate(candidates[upper_tf], entry_time, direction, true_only=True)
        if up_true is not None:
            variants.append((f"{upper_tf}_opposite_true_reversal_confirmed", up_true["timestamp"], up_true["close_at_entry"]))
        hist_cross = _find_hist_cross(candles[upper_tf], entry_time, direction)
        if hist_cross is not None:
            variants.append((f"{upper_tf}_hist_cross_zero", hist_cross["timestamp"], hist_cross["close"]))

    for bars in FIXED_BARS[entry_tf]:
        row = _fixed_bar_exit(candles[entry_tf], entry_time, bars)
        if row is not None:
            variants.append((f"fixed_bars_{bars}", row["timestamp"], row["close"]))

    for tp in TP_SL_GRID[entry_tf]["tp"][:2]:
        for sl in TP_SL_GRID[entry_tf]["sl"][:2]:
            row, reason = _tp_sl_exit(candles[entry_tf], entry_time, entry_price, direction, tp, sl)
            if row is not None:
                variants.append((reason, row["timestamp"], row["close"]))

    row, reason = _trailing_exit(candles[entry_tf], entry_time, entry_price, direction, trail_pct=TP_SL_GRID[entry_tf]["sl"][1])
    if row is not None:
        variants.append((reason, row["timestamp"], row["close"]))
    return variants


def run_backtests(candidates_all: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidate_by_tf = {tf: candidates_all[candidates_all["candidate_tf"].eq(tf)].sort_values("timestamp").reset_index(drop=True) for tf in TIMEFRAMES}
    candidate_events: dict[tuple[str, str, bool], pd.DataFrame] = {}
    for tf, frame in candidate_by_tf.items():
        for direction in ("long", "short"):
            subset = frame[frame["candidate_direction"].eq(direction)].sort_values("timestamp").reset_index(drop=True)
            candidate_events[(tf, direction, False)] = subset
            candidate_events[(tf, direction, True)] = subset[subset["is_true_reversal"].eq(1)].reset_index(drop=True)

    hist_cross_events: dict[tuple[str, str], pd.DataFrame] = {}
    candle_lookup: dict[str, pd.DataFrame] = {}
    for tf, frame in candles.items():
        clean = frame.sort_values("timestamp").reset_index(drop=True).copy()
        candle_lookup[tf] = clean
        hist_cross_events[(tf, "long")] = clean[clean["ppo_hist"] >= 0].reset_index(drop=True)
        hist_cross_events[(tf, "short")] = clean[clean["ppo_hist"] <= 0].reset_index(drop=True)

    trades: list[dict[str, Any]] = []
    cost = ROUND_TRIP_FEE_PCT + SLIPPAGE_PER_SIDE_PCT * 2

    def next_event(events: pd.DataFrame, when: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
        if events.empty:
            return None, None
        times = events["timestamp"].to_numpy(dtype="datetime64[ns]")
        idx = int(np.searchsorted(times, np.datetime64(when), side="right"))
        if idx >= len(events):
            return None, None
        row = events.iloc[idx]
        price = row.get("close_at_entry", row.get("close", np.nan))
        return row["timestamp"], float(price)

    def fixed_exit(tf: str, entry_bar: int, bars: int) -> tuple[pd.Timestamp | None, float | None]:
        frame = candle_lookup[tf]
        idx = int(entry_bar) + bars
        if idx >= len(frame):
            idx = len(frame) - 1
        if idx <= int(entry_bar):
            return None, None
        row = frame.iloc[idx]
        return row["timestamp"], float(row["close"])

    def tp_sl_exit(tf: str, entry_bar: int, entry_price: float, direction: str, tp: float, sl: float) -> tuple[pd.Timestamp | None, float | None, str]:
        frame = candle_lookup[tf]
        sign = _candidate_sign(direction)
        max_scan = max(FIXED_BARS[tf]) * 2
        start = int(entry_bar) + 1
        end = min(len(frame), start + max_scan)
        for idx in range(start, end):
            row = frame.iloc[idx]
            if sign > 0:
                stop_hit = (row["low"] / entry_price - 1.0) * 100.0 <= -sl
                take_hit = (row["high"] / entry_price - 1.0) * 100.0 >= tp
                stop_price = entry_price * (1 - sl / 100.0)
                take_price = entry_price * (1 + tp / 100.0)
            else:
                stop_hit = (entry_price / row["high"] - 1.0) * 100.0 <= -sl
                take_hit = (entry_price / row["low"] - 1.0) * 100.0 >= tp
                stop_price = entry_price * (1 + sl / 100.0)
                take_price = entry_price * (1 - tp / 100.0)
            if stop_hit:
                return row["timestamp"], float(stop_price), f"tp_sl_tp{tp}_sl{sl}_stop"
            if take_hit:
                return row["timestamp"], float(take_price), f"tp_sl_tp{tp}_sl{sl}_take"
        if end > start:
            row = frame.iloc[end - 1]
            return row["timestamp"], float(row["close"]), f"tp_sl_tp{tp}_sl{sl}_timeout"
        return None, None, f"tp_sl_tp{tp}_sl{sl}_eod"

    def trailing_exit(tf: str, entry_bar: int, entry_price: float, direction: str, trail_pct: float) -> tuple[pd.Timestamp | None, float | None, str]:
        frame = candle_lookup[tf]
        sign = _candidate_sign(direction)
        best = entry_price
        start = int(entry_bar) + 1
        end = min(len(frame), start + max(FIXED_BARS[tf]) * 3)
        for idx in range(start, end):
            row = frame.iloc[idx]
            if sign > 0:
                best = max(best, float(row["high"]))
                stop = best * (1 - trail_pct / 100.0)
                if row["low"] <= stop:
                    return row["timestamp"], float(stop), f"trailing_stop_{trail_pct}"
            else:
                best = min(best, float(row["low"]))
                stop = best * (1 + trail_pct / 100.0)
                if row["high"] >= stop:
                    return row["timestamp"], float(stop), f"trailing_stop_{trail_pct}"
        if end > start:
            row = frame.iloc[end - 1]
            return row["timestamp"], float(row["close"]), f"trailing_stop_{trail_pct}_timeout"
        return None, None, f"trailing_stop_{trail_pct}_eod"

    def add_trade(spec: StrategySpec, entry: pd.Series, exit_reason: str, exit_time: pd.Timestamp, exit_price: float) -> dict[str, Any] | None:
        if pd.isna(exit_time) or exit_time <= entry["timestamp"]:
            return None
        gross = _direction_return(entry["close_at_entry"], exit_price, entry["candidate_direction"])
        net = gross - cost if not pd.isna(gross) else np.nan
        holding_bars = (exit_time - entry["timestamp"]) / _tf_delta(spec.entry_tf)
        return {
            "strategy_name": spec.name,
            "entry_time": entry["timestamp"],
            "exit_time": exit_time,
            "direction": entry["candidate_direction"],
            "entry_price": entry["close_at_entry"],
            "exit_price": exit_price,
            "gross_return": gross,
            "net_return": net,
            "holding_bars": float(holding_bars),
            "entry_tf": spec.entry_tf,
            "exit_reason": exit_reason,
            "candidate_ppo": entry["ppo"],
            "candidate_hist": entry["ppo_hist"],
            "candidate_zone": entry["zone4"],
            "upper_1h_zone": entry.get("upper_1h_zone4"),
            "upper_4h_zone": entry.get("upper_4h_zone4"),
            "upper_1d_zone": entry.get("upper_1d_zone4"),
            "is_noise": entry["is_noise"],
            "is_true_reversal": entry["is_true_reversal"],
        }

    def append_method_trades(spec: StrategySpec, entries: pd.DataFrame, method: str) -> None:
        last_exit = pd.Timestamp.min
        for _, entry in entries.iterrows():
            if entry["timestamp"] <= last_exit:
                continue
            direction = str(entry["candidate_direction"])
            opposite = "short" if direction == "long" else "long"
            exit_time: pd.Timestamp | None = None
            exit_price: float | None = None
            exit_reason = method
            if method == "same_tf_opposite_candidate":
                exit_time, exit_price = next_event(candidate_events[(spec.entry_tf, opposite, False)], entry["timestamp"])
            elif method == "same_tf_opposite_true_reversal_confirmed":
                exit_time, exit_price = next_event(candidate_events[(spec.entry_tf, opposite, True)], entry["timestamp"])
            elif method.startswith("fixed_bars_"):
                bars = int(method.rsplit("_", 1)[1])
                exit_time, exit_price = fixed_exit(spec.entry_tf, int(entry["bar_index"]), bars)
            elif method.endswith("_opposite_candidate"):
                upper_tf = method.split("_", 1)[0]
                exit_time, exit_price = next_event(candidate_events[(upper_tf, opposite, False)], entry["timestamp"])
            elif method.endswith("_opposite_true_reversal_confirmed"):
                upper_tf = method.split("_", 1)[0]
                exit_time, exit_price = next_event(candidate_events[(upper_tf, opposite, True)], entry["timestamp"])
            elif method.endswith("_hist_cross_zero"):
                upper_tf = method.split("_", 1)[0]
                exit_time, exit_price = next_event(hist_cross_events[(upper_tf, direction)], entry["timestamp"])
            elif method.startswith("tp_sl_"):
                _, _, tp_part, sl_part = method.split("_")
                tp = float(tp_part[2:])
                sl = float(sl_part[2:])
                exit_time, exit_price, _detail_reason = tp_sl_exit(spec.entry_tf, int(entry["bar_index"]), float(entry["close_at_entry"]), direction, tp, sl)
                exit_reason = method
            elif method.startswith("trailing_stop_"):
                trail_pct = float(method.rsplit("_", 1)[1])
                exit_time, exit_price, _detail_reason = trailing_exit(spec.entry_tf, int(entry["bar_index"]), float(entry["close_at_entry"]), direction, trail_pct)
                exit_reason = method
            if exit_time is None or exit_price is None:
                continue
            trade = add_trade(spec, entry, exit_reason, exit_time, exit_price)
            if trade is not None:
                trades.append(trade)
                last_exit = exit_time

    for spec in STRATEGIES:
        entries = _entry_filter(candidates_all, spec)
        methods = ["same_tf_opposite_candidate", "same_tf_opposite_true_reversal_confirmed"]
        methods.extend([f"fixed_bars_{bars}" for bars in FIXED_BARS[spec.entry_tf]])
        upper = upper_timeframes(spec.entry_tf)
        if upper:
            upper_tf = upper[0]
            methods.extend(
                [
                    f"{upper_tf}_opposite_candidate",
                    f"{upper_tf}_opposite_true_reversal_confirmed",
                    f"{upper_tf}_hist_cross_zero",
                ]
            )
        for tp, sl in [(TP_SL_GRID[spec.entry_tf]["tp"][1], TP_SL_GRID[spec.entry_tf]["sl"][1])]:
            methods.append(f"tp_sl_tp{tp}_sl{sl}")
        methods.append(f"trailing_stop_{TP_SL_GRID[spec.entry_tf]['sl'][1]}")
        for method in methods:
            append_method_trades(spec, entries, method)
    return pd.DataFrame(trades)


def _max_drawdown_pct(returns: pd.Series) -> float:
    rets = pd.to_numeric(returns, errors="coerce").fillna(0) / 100.0
    equity = (1.0 + rets).cumprod()
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min() * 100)


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame()
    for (strategy, exit_reason), group in trades.groupby(["strategy_name", "exit_reason"], dropna=False):
        net = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        gross = pd.to_numeric(group["gross_return"], errors="coerce").dropna()
        wins = net[net > 0]
        losses = net[net < 0]
        compounded = (np.prod(1 + net / 100.0) - 1.0) * 100.0 if not net.empty else np.nan
        profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.inf if wins.sum() > 0 else np.nan
        long_group = group[group["direction"].eq("long")]
        short_group = group[group["direction"].eq("short")]
        rows.append(
            {
                "strategy_name": strategy,
                "entry_tf": group["entry_tf"].iloc[0],
                "filter_set": strategy,
                "exit_method": exit_reason,
                "n_trades": len(group),
                "win_rate": float((net > 0).mean() * 100) if len(net) else np.nan,
                "avg_return_gross": float(gross.mean()) if len(gross) else np.nan,
                "avg_return_net": float(net.mean()) if len(net) else np.nan,
                "median_return_net": float(net.median()) if len(net) else np.nan,
                "total_return_compounded": float(compounded),
                "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else profit_factor,
                "max_drawdown": _max_drawdown_pct(group.sort_values("entry_time")["net_return"]),
                "sharpe_like": float(net.mean() / net.std() * math.sqrt(len(net))) if len(net) > 1 and net.std() else np.nan,
                "avg_holding_bars": float(group["holding_bars"].mean()),
                "avg_holding_hours": float(group["holding_bars"].mean() * TF_SECONDS[group["entry_tf"].iloc[0]] / 3600),
                "long_trades": len(long_group),
                "short_trades": len(short_group),
                "long_win_rate": float((long_group["net_return"] > 0).mean() * 100) if len(long_group) else np.nan,
                "short_win_rate": float((short_group["net_return"] > 0).mean() * 100) if len(short_group) else np.nan,
                "low_sample": len(group) < LOW_SAMPLE_N,
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_return_net", "n_trades"], ascending=[False, False]).reset_index(drop=True)


def strategy_by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for cols in (["strategy_name", "upper_4h_zone"], ["strategy_name", "upper_1d_zone"]):
        zone_col = cols[1]
        for keys, group in trades.groupby(cols, dropna=False):
            net = group["net_return"]
            rows.append(
                {
                    "strategy_name": keys[0],
                    "regime_type": zone_col,
                    "regime": keys[1],
                    "n_trades": len(group),
                    "win_rate": float((net > 0).mean() * 100),
                    "avg_return_net": float(net.mean()),
                    "median_return_net": float(net.median()),
                    "low_sample": len(group) < LOW_SAMPLE_N,
                }
            )
    return pd.DataFrame(rows)


def noise_vs_trade_performance(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (strategy, tf, is_noise), group in trades.groupby(["strategy_name", "entry_tf", "is_noise"], dropna=False):
        net = group["net_return"]
        rows.append(
            {
                "strategy_name": strategy,
                "candidate_tf": tf,
                "label_type": "noise" if is_noise else "true_reversal",
                "n_trades": len(group),
                "win_rate": float((net > 0).mean() * 100),
                "avg_return_net": float(net.mean()),
                "median_return_net": float(net.median()),
                "max_drawdown_contribution": _max_drawdown_pct(group.sort_values("entry_time")["net_return"]),
                "low_sample": len(group) < LOW_SAMPLE_N,
            }
        )
    return pd.DataFrame(rows)


def feature_noise_reduction(candidates: pd.DataFrame) -> pd.DataFrame:
    base_noise = candidates.groupby(["candidate_tf", "candidate_direction"])["is_noise"].mean().rename("base_noise_rate")
    rows = []
    checks = [
        ("align_count_ge_1", candidates["align_count"] >= 1),
        ("align_count_ge_2", candidates["align_count"] >= 2),
        ("major_align_count_ge_1", candidates["major_align_count"] >= 1),
        ("supportive_4h_extreme", candidates["has_supportive_4h_extreme"] == 1),
        ("no_contra_4h_extreme", candidates["has_contra_4h_extreme"] == 0),
        ("ppo_bottom20", candidates["ppo_bin"].eq("bottom20") | candidates["ppo_bin"].eq("bottom10")),
        ("ppo_top20", candidates["ppo_bin"].eq("top20") | candidates["ppo_bin"].eq("top10")),
        ("hist_bottom20", candidates["hist_bin"].eq("bottom20") | candidates["hist_bin"].eq("bottom10")),
        ("hist_top20", candidates["hist_bin"].eq("top20") | candidates["hist_bin"].eq("top10")),
    ]
    for name, mask in checks:
        subset = candidates[mask]
        for key, group in subset.groupby(["candidate_tf", "candidate_direction"]):
            base = base_noise.get(key, np.nan)
            noise = float(group["is_noise"].mean())
            rows.append(
                {
                    "feature": name,
                    "candidate_tf": key[0],
                    "candidate_direction": key[1],
                    "n": len(group),
                    "base_noise_pct": base * 100,
                    "filtered_noise_pct": noise * 100,
                    "noise_reduction_pct_point": (base - noise) * 100,
                    "low_sample": len(group) < LOW_SAMPLE_N,
                }
            )
    return pd.DataFrame(rows).sort_values("noise_reduction_pct_point", ascending=False)


def write_report(out_dir: Path, summaries: dict[str, pd.DataFrame], model_rows: pd.DataFrame, backtest_summary: pd.DataFrame, feature_reductions: pd.DataFrame) -> None:
    by_tf = summaries["21_candidate_noise_by_tf.csv"]
    by_bins = summaries["23_candidate_noise_by_ppo_hist_bins.csv"]
    by_upper = summaries["24_candidate_noise_by_upper_tf.csv"]
    best = backtest_summary[~backtest_summary["low_sample"]].head(5) if not backtest_summary.empty else pd.DataFrame()
    worst = backtest_summary[~backtest_summary["low_sample"]].sort_values("avg_return_net").head(5) if not backtest_summary.empty else pd.DataFrame()

    def table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
        if df.empty:
            return "_No rows._"
        return df[cols].head(n).to_markdown(index=False)

    tf_lines = []
    for tf in ("15m", "1h", "4h"):
        part = by_tf[by_tf["candidate_tf"].eq(tf)]
        if part.empty:
            tf_lines.append(f"- {tf}: no candidates")
        else:
            noise = part["noise_count"].sum() / part["n"].sum() * 100
            tf_lines.append(f"- {tf}: noise {noise:.2f}% across {int(part['n'].sum())} candidates")

    top_ppo = by_bins[~by_bins["low_sample"]].sort_values("true_reversal_pct", ascending=False).head(8)
    upper_15m = by_upper[(by_upper["candidate_tf"].eq("15m")) & (by_upper["upper_tf_set"].eq("4h"))]
    reduction_note = "Not enough 15m/4h grouped rows."
    if not upper_15m.empty:
        aligned = upper_15m[upper_15m["align_count"].eq(1)]
        raw = by_tf[by_tf["candidate_tf"].eq("15m")]
        if not aligned.empty and not raw.empty:
            raw_noise = raw["noise_count"].sum() / raw["n"].sum() * 100
            aligned_noise = aligned["noise_count"].sum() / aligned["n"].sum() * 100
            reduction_note = f"15m with 4h same-direction bias noise: {aligned_noise:.2f}% vs raw {raw_noise:.2f}% ({raw_noise - aligned_noise:.2f} pct-point change)."

    report = f"""# PPO Reversal Noise Backtest Report

## 1. 분석 목적

이 분석은 확정된 cycle start가 아니라 실전에서 먼저 보이는 PPO hist 방향 전환 캔들을 entry candidate로 삼아 noise, true reversal, 그리고 거래 성과를 분해한다.

## 2. 기존 확정 사이클 분석의 한계

확정 cycle start는 사후적으로 깔끔하지만 실시간 진입 시점에는 아직 확정되지 않는다. 따라서 모든 반전 후보에 진입하는 baseline을 먼저 만들고, PPO/HIST/상위 TF 필터가 실제로 기대값을 개선하는지 비교했다.

## 3. 반전 후보 캔들 정의

`ppo_hist.diff()`의 부호를 raw direction으로 계산했다. 직전 유효 방향이 down이고 현재 up이면 long candidate, 직전 유효 방향이 up이고 현재 down이면 short candidate로 정의했다.

## 4. noise / true reversal label 정의

기본 label은 반대 방향 run length가 `MAX_TOLERANCE={MAX_TOLERANCE}`를 초과하면 true reversal, 이하면 noise다. cycle start와 가까운 candidate는 별도 cycle-based label로도 계산했고 mismatch는 CSV로 저장했다.

## 5. TF별 기본 noise 확률

{chr(10).join(tf_lines)}

## 6. PPO/HIST 구간별 true reversal 확률

{table(top_ppo, ["candidate_tf", "candidate_direction", "ppo_bin", "hist_bin", "n", "true_reversal_pct", "noise_pct", "trade_return_mean"], 10)}

## 7. 상위 TF 상태별 noise 감소 효과

{reduction_note}

## 8. 15m 타점 전략 결과

{table(backtest_summary[backtest_summary["entry_tf"].eq("15m") & ~backtest_summary["low_sample"]], ["strategy_name", "exit_method", "n_trades", "win_rate", "avg_return_net", "total_return_compounded", "max_drawdown", "avg_holding_hours"], 10)}

## 9. 1h 타점 전략 결과

{table(backtest_summary[backtest_summary["entry_tf"].eq("1h") & ~backtest_summary["low_sample"]], ["strategy_name", "exit_method", "n_trades", "win_rate", "avg_return_net", "total_return_compounded", "max_drawdown", "avg_holding_hours"], 10)}

## 10. 4h/1d 필터의 효과

상위 TF 필터 효과는 `24_candidate_noise_by_upper_tf.csv`, `28_strategy_by_market_regime.csv`, 그리고 전략 S2/S3/S4/S7/S8/S9 비교로 확인한다. 표본 30건 미만은 low_sample로 결론에서 제외한다.

## 11. 청산 방식별 비교

{table(backtest_summary[~backtest_summary["low_sample"]], ["strategy_name", "exit_method", "n_trades", "win_rate", "avg_return_net", "profit_factor", "max_drawdown", "avg_holding_hours"], 15)}

## 12. 최적 전략 후보 3개

{table(best, ["strategy_name", "exit_method", "n_trades", "win_rate", "avg_return_net", "total_return_compounded", "max_drawdown"], 3)}

## 13. 피해야 할 신호 조건

{table(feature_reductions.sort_values("noise_reduction_pct_point").head(10), ["feature", "candidate_tf", "candidate_direction", "n", "base_noise_pct", "filtered_noise_pct", "noise_reduction_pct_point"], 10)}

## 14. 실전 적용 룰

상위 TF alignment와 supportive 4h extreme이 noise를 낮추면서 net return도 개선되는 조합을 우선 후보로 본다. 단일 후보 candle만으로 자동매매 룰을 확정하지 말고, `26_backtest_summary.csv`에서 거래 수, 평균 net return, MDD, 보유시간을 함께 확인해야 한다.

## 15. 한계와 다음 분석

상위 TF join은 기본적으로 마지막으로 닫힌 candle만 사용해 lookahead를 줄였다. 다만 TP/SL 체결은 candle high/low 기반의 보수적 stop-first 규칙이며, intrabar 체결 순서는 실제 tick 데이터 없이 완전히 알 수 없다.

## 질문별 답

- 15m/1h/4h noise 비율: 위 TF별 기본 noise 확률 참조.
- PPO/HIST가 어느 구간일 때 true reversal 확률이 높은가: 6장 표와 `23_candidate_noise_by_ppo_hist_bins.csv` 참조.
- 상위 4h가 같은 방향이면 15m noise가 얼마나 줄어드는가: 7장 참조.
- 상위 4h가 반대 extreme이면 피해야 하는가: `24_candidate_noise_by_upper_tf.csv`와 `feature_noise_reduction.csv`에서 contra extreme 조건의 noise/성과를 확인한다.
- 15m timing + 1h/4h 확인 vs 1h timing + 4h/1d 확인: 8장, 9장과 TOP 전략 표 기준으로 비교한다.
- 어디까지 참고 파는 것이 좋은가: 11장 청산 방식 비교 기준.
- 필터링이 baseline보다 개선되는가: S0/S6 raw와 S1-S5/S7-S9의 net return, MDD, 거래 수를 같이 비교한다.
- 자동매매 룰 후보: 12장 후보 중 low_sample이 아니고 MDD가 감당 가능한 조합.
"""
    (out_dir / "PPO_reversal_noise_backtest_report.md").write_text(report, encoding="utf-8")


def scan_data() -> pd.DataFrame:
    rows = []
    for tf in TIMEFRAMES:
        raw = _raw_market_path(tf)
        cycle = _cycle_path(tf)
        raw_cols = list(pd.read_csv(raw, nrows=0).columns)
        cycle_cols = list(pd.read_parquet(cycle).columns)
        rows.append(
            {
                "timeframe": tf,
                "raw_path": str(raw),
                "cycle_path": str(cycle),
                "raw_columns": "|".join(raw_cols),
                "cycle_columns": "|".join(cycle_cols),
                "has_required_raw": all(col in raw_cols for col in ("open", "high", "low", "close", "ppo", "ppo_hist")) and any(col in raw_cols for col in ("date", "timestamp", "open_time")),
                "raw_direction_source": "computed_from_ppo_hist_diff",
                "ppo_hist_diff_source": "computed",
                "cycle_label_source": "cycle_start_date/cycle_type",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO reversal candidate noise and backtest analysis.")
    parser.add_argument("--zero-mode", choices=("ffill", "zero"), default="ffill")
    args = parser.parse_args()

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    scan = scan_data()
    scan.to_csv(out_dir / "20_data_column_scan.csv", index=False, encoding="utf-8-sig")

    cycles = {tf: load_cycles(tf) for tf in TIMEFRAMES}
    candles = {tf: add_cycle_state(load_candles(tf, args.zero_mode), cycles[tf], tf) for tf in TIMEFRAMES}
    candidates = build_candidates(candles, cycles)
    candidates = add_upper_features(candidates, candles)
    candidates.to_csv(out_dir / "20_reversal_candidates.csv", index=False, encoding="utf-8-sig")
    candidates[candidates["label_mismatch"]].to_csv(out_dir / "20_label_mismatch_cases.csv", index=False, encoding="utf-8-sig")

    summaries = write_noise_summaries(candidates, out_dir)
    model_rows = train_noise_models(candidates, out_dir)
    feature_reductions = feature_noise_reduction(candidates)
    feature_reductions.to_csv(out_dir / "25_noise_reduction_features.csv", index=False, encoding="utf-8-sig")

    trades = run_backtests(candidates, candles)
    trades.to_csv(out_dir / "27_backtest_trades.csv", index=False, encoding="utf-8-sig")
    backtest_summary = summarize_trades(trades)
    backtest_summary.to_csv(out_dir / "26_backtest_summary.csv", index=False, encoding="utf-8-sig")
    strategy_regime = strategy_by_regime(trades)
    strategy_regime.to_csv(out_dir / "28_strategy_by_market_regime.csv", index=False, encoding="utf-8-sig")
    best_exit = backtest_summary.sort_values(["strategy_name", "avg_return_net"], ascending=[True, False]).groupby("strategy_name", as_index=False).head(5)
    best_exit.to_csv(out_dir / "29_best_exit_by_entry_filter.csv", index=False, encoding="utf-8-sig")
    noise_perf = noise_vs_trade_performance(trades)
    noise_perf.to_csv(out_dir / "30_noise_vs_trade_performance.csv", index=False, encoding="utf-8-sig")

    write_report(out_dir, summaries, model_rows, backtest_summary, feature_reductions)
    print(f"Wrote analysis outputs to {out_dir}")
    print(f"Candidates: {len(candidates):,}; trades: {len(trades):,}; summary rows: {len(backtest_summary):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
