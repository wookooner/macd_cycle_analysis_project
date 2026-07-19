from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.microstructure.features import normalize_timeframe
from data_pipeline.microstructure.paths import feature_dir, report_dir


LOGGER = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    "cvd_delta",
    "taker_buy_ratio",
    "book_imbalance_last",
    "open_interest_change_pct",
    "funding_rate_percentile",
    "long_short_ratio",
    "top_account_long_short_ratio",
    "top_position_long_short_ratio",
]


def load_feature_frame(symbol: str, timeframe: str) -> pd.DataFrame:
    timeframe = normalize_timeframe(timeframe)
    path = feature_dir(symbol) / f"microstructure_features_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing feature parquet: {path}")
    df = pd.read_parquet(path)
    if "bar_start" not in df.columns:
        raise ValueError(f"{path} has no bar_start column")
    df["bar_start"] = pd.to_datetime(df["bar_start"], errors="coerce", utc=True)
    for col in ["feature_source_max_time", "label_start_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df.sort_values("bar_start").reset_index(drop=True)


def add_forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    price_col = "label_start_price" if "label_start_price" in df.columns else "close"
    if price_col not in df.columns:
        raise ValueError("feature frame needs label_start_price or close for forward-return validation")
    out = df.copy()
    close = pd.to_numeric(out[price_col], errors="coerce")
    for horizon in horizons:
        out[f"forward_return_{horizon}b"] = (close.shift(-horizon) / close - 1.0) * 100.0
        future_closes = pd.concat([close.shift(-step) for step in range(1, horizon + 1)], axis=1)
        out[f"forward_max_return_{horizon}b"] = (future_closes.max(axis=1) / close - 1.0) * 100.0
        out[f"forward_min_return_{horizon}b"] = (future_closes.min(axis=1) / close - 1.0) * 100.0
    return out


def audit_observation_boundary(df: pd.DataFrame) -> dict[str, Any]:
    required = {"feature_source_max_time", "label_start_time"}
    missing = sorted(required.difference(df.columns))
    if missing:
        return {"status": "fail", "missing_columns": missing}
    feature_time = pd.to_datetime(df["feature_source_max_time"], errors="coerce", utc=True)
    label_time = pd.to_datetime(df["label_start_time"], errors="coerce", utc=True)
    valid = feature_time.notna() & label_time.notna()
    violations = valid & (feature_time > label_time)
    lag_seconds = (label_time[valid] - feature_time[valid]).dt.total_seconds()
    checked_rows = int(valid.sum())
    return {
        "status": "no_data" if checked_rows == 0 else ("pass" if int(violations.sum()) == 0 else "fail"),
        "checked_rows": checked_rows,
        "violation_rows": int(violations.sum()),
        "min_label_lag_seconds": None if lag_seconds.empty else float(lag_seconds.min()),
        "max_label_lag_seconds": None if lag_seconds.empty else float(lag_seconds.max()),
        "non_positive_lag_rows": int((lag_seconds <= 0).sum()) if not lag_seconds.empty else 0,
    }


def audit_staleness(df: pd.DataFrame) -> dict[str, Any]:
    stale_cols = [col for col in df.columns if col.endswith("_staleness_seconds")]
    report: dict[str, Any] = {"columns": stale_cols}
    for col in stale_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        report[col] = {
            "n": int(len(values)),
            "max_seconds": None if values.empty else float(values.max()),
            "p95_seconds": None if values.empty else float(values.quantile(0.95)),
        }
    return report


def audit_funding_boundaries(df: pd.DataFrame, lookback_minutes: int = 5) -> dict[str, Any]:
    if "funding_rate_source_time" not in df.columns or "feature_source_max_time" not in df.columns:
        return {"status": "skip", "reason": "funding source time unavailable"}
    feature_time = pd.to_datetime(df["feature_source_max_time"], errors="coerce", utc=True)
    funding_time = pd.to_datetime(df["funding_rate_source_time"], errors="coerce", utc=True)
    settlement_hours = {0, 8, 16}
    next_hour = feature_time.dt.ceil("h")
    near_settlement = (
        next_hour.dt.hour.isin(settlement_hours)
        & ((next_hour - feature_time) <= pd.Timedelta(minutes=lookback_minutes))
        & ((next_hour - feature_time) >= pd.Timedelta(0))
    )
    violations = near_settlement & funding_time.notna() & (funding_time > feature_time)
    checked_rows = int(near_settlement.sum())
    return {
        "status": "no_data" if checked_rows == 0 else ("pass" if int(violations.sum()) == 0 else "fail"),
        "checked_rows": checked_rows,
        "violation_rows": int(violations.sum()),
        "lookback_minutes": lookback_minutes,
    }


def filter_stale_rows(df: pd.DataFrame, max_staleness_seconds: float | None) -> pd.DataFrame:
    if max_staleness_seconds is None:
        return df
    stale_cols = [col for col in df.columns if col.endswith("_staleness_seconds")]
    if not stale_cols:
        return df
    mask = pd.Series(True, index=df.index)
    for col in stale_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        mask &= values.isna() | (values <= max_staleness_seconds)
    return df[mask].copy()


def qbin(series: pd.Series, bins: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    ranked = clean.rank(method="first")
    try:
        return pd.qcut(ranked, bins, labels=[f"q{i + 1}" for i in range(bins)]).astype("string")
    except ValueError:
        return pd.Series(["na"] * len(series), index=series.index, dtype="string")


def distribution_metrics(values: pd.Series) -> dict[str, Any]:
    r = pd.to_numeric(values, errors="coerce").dropna()
    if r.empty:
        return {"n": 0}
    q10 = float(r.quantile(0.10))
    q25 = float(r.quantile(0.25))
    q50 = float(r.quantile(0.50))
    q75 = float(r.quantile(0.75))
    q90 = float(r.quantile(0.90))
    return {
        "n": int(len(r)),
        "mean_pct": float(r.mean()),
        "std_pct": float(r.std(ddof=0)),
        "skew": float(r.skew()),
        "q10_pct": q10,
        "q25_pct": q25,
        "median_pct": q50,
        "q75_pct": q75,
        "q90_pct": q90,
        "left_tail_mean_pct": float(r[r <= q10].mean()) if (r <= q10).any() else np.nan,
        "right_tail_mean_pct": float(r[r >= q90].mean()) if (r >= q90).any() else np.nan,
        "positive_rate_pct": float((r > 0).mean() * 100.0),
        "tail_asymmetry_pct": float(abs(q90) - abs(q10)),
    }


def profile_feature(df: pd.DataFrame, feature: str, return_col: str, bins: int, min_rows: int) -> pd.DataFrame:
    if feature not in df.columns or return_col not in df.columns:
        return pd.DataFrame()
    work = df[[feature, return_col]].copy()
    if pd.api.types.is_numeric_dtype(work[feature]):
        work["bucket"] = qbin(work[feature], bins)
    else:
        work["bucket"] = work[feature].astype("string").fillna("na")

    rows = []
    for bucket, group in work.groupby("bucket", dropna=False):
        metrics = distribution_metrics(group[return_col])
        if metrics.get("n", 0) < min_rows:
            continue
        rows.append({"feature": feature, "bucket": str(bucket), "return_col": return_col, **metrics})
    return pd.DataFrame(rows)


def profile_all_features(
    df: pd.DataFrame,
    selected_features: list[str],
    horizons: list[int],
    bins: int,
    min_rows: int,
) -> pd.DataFrame:
    all_profiles = []
    for horizon in horizons:
        return_col = f"forward_return_{horizon}b"
        for feature in selected_features:
            profile = profile_feature(df, feature, return_col, bins=bins, min_rows=min_rows)
            if not profile.empty:
                all_profiles.append(profile)
    return pd.concat(all_profiles, ignore_index=True, sort=False) if all_profiles else pd.DataFrame()


def shuffled_feature_frame(df: pd.DataFrame, selected_features: list[str], seed: int) -> pd.DataFrame:
    shuffled = df.copy()
    rng = np.random.default_rng(seed)
    for feature in selected_features:
        if feature in shuffled.columns:
            values = shuffled[feature].to_numpy(copy=True)
            rng.shuffle(values)
            shuffled[feature] = values
    return shuffled


def validate_forward_distributions(
    symbol: str = "BTCUSDT",
    timeframe: str = "1min",
    horizons: list[int] | None = None,
    features: list[str] | None = None,
    bins: int = 5,
    min_rows: int = 30,
    warmup_rows: int = 100,
    shuffle_seeds: list[int] | None = None,
    max_staleness_seconds: float | None = 600.0,
) -> dict[str, Path]:
    timeframe = normalize_timeframe(timeframe)
    horizons = horizons or [5, 15, 60]
    shuffle_seeds = shuffle_seeds or [11, 42, 101]
    raw_df = load_feature_frame(symbol, timeframe)
    boundary_audit = audit_observation_boundary(raw_df)
    staleness_audit = audit_staleness(raw_df)
    funding_boundary_audit = audit_funding_boundaries(raw_df)
    filtered_df = filter_stale_rows(raw_df, max_staleness_seconds)
    df = add_forward_returns(filtered_df.iloc[warmup_rows:].reset_index(drop=True), horizons)
    selected_features = [feature for feature in (features or DEFAULT_FEATURES) if feature in df.columns]
    out_dir = report_dir(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_path = out_dir / f"microstructure_forward_frame_{timeframe}.parquet"
    df.to_parquet(feature_path, index=False)

    profiles = profile_all_features(df, selected_features, horizons, bins=bins, min_rows=min_rows)

    profile_path = out_dir / f"microstructure_forward_distribution_profile_{timeframe}.csv"
    profiles.to_csv(profile_path, index=False, encoding="utf-8-sig")

    shuffled_runs = []
    for seed in shuffle_seeds:
        shuffled = shuffled_feature_frame(df, selected_features, seed=seed)
        shuffled_profile = profile_all_features(shuffled, selected_features, horizons, bins=bins, min_rows=min_rows)
        if not shuffled_profile.empty:
            shuffled_profile["shuffle_seed"] = seed
            shuffled_runs.append(shuffled_profile)
    shuffled_profiles = pd.concat(shuffled_runs, ignore_index=True, sort=False) if shuffled_runs else pd.DataFrame()
    shuffled_profile_path = out_dir / f"microstructure_forward_distribution_profile_shuffled_{timeframe}.csv"
    shuffled_profiles.to_csv(shuffled_profile_path, index=False, encoding="utf-8-sig")

    meta = {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "rows_raw": int(len(raw_df)),
        "rows_after_staleness_filter": int(len(filtered_df)),
        "rows_after_warmup": int(len(df)),
        "warmup_rows_removed": int(warmup_rows),
        "max_staleness_seconds": max_staleness_seconds,
        "horizons_bars": horizons,
        "features": selected_features,
        "profile_rows": int(len(profiles)),
        "shuffled_profile_rows": int(len(shuffled_profiles)),
        "shuffle_seeds": shuffle_seeds,
        "observation_boundary_audit": boundary_audit,
        "staleness_audit": staleness_audit,
        "funding_boundary_audit": funding_boundary_audit,
        "focus": "conditional forward-return distribution: quantiles, tails, skew; not directional claims",
        "force_order_limit_note": "!forceOrder@arr is throttled by Binance and should be interpreted as a lower-bound/spike proxy, not total liquidation volume.",
    }
    meta_path = out_dir / f"microstructure_forward_validation_meta_{timeframe}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"frame": feature_path, "profile": profile_path, "shuffled_profile": shuffled_profile_path, "meta": meta_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate microstructure features with conditional forward-return distributions.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1min")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 15, 60])
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--warmup-rows", type=int, default=100)
    parser.add_argument("--shuffle-seeds", nargs="+", type=int, default=[11, 42, 101])
    parser.add_argument("--max-staleness-seconds", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    try:
        paths = validate_forward_distributions(
            symbol=args.symbol,
            timeframe=args.timeframe,
            horizons=args.horizons,
            features=args.features,
            bins=args.bins,
            min_rows=args.min_rows,
            warmup_rows=args.warmup_rows,
            shuffle_seeds=args.shuffle_seeds,
            max_staleness_seconds=args.max_staleness_seconds,
        )
    except Exception as exc:
        LOGGER.error("Validation failed: %s", exc)
        sys.exit(1)
    for label, path in paths.items():
        LOGGER.info("Saved %s: %s", label, path)


if __name__ == "__main__":
    main()
