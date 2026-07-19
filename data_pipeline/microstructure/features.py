from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.microstructure.io import read_parquet_trees_prefer_first
from data_pipeline.microstructure.paths import feature_dir, raw_stream_read_dirs
from src.common.paths import PROJECT_PATHS


LOGGER = logging.getLogger(__name__)


def normalize_timeframe(timeframe: str) -> str:
    aliases = {
        "1m": "1min",
        "1min": "1min",
        "5m": "5min",
        "5min": "5min",
        "15m": "15min",
        "15min": "15min",
        "30m": "30min",
        "30min": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    normalized = str(timeframe).strip().lower()
    return aliases.get(normalized, normalized)


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(normalize_timeframe(timeframe))


def ns_to_utc_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="ns", utc=True).astype("datetime64[ns, UTC]")


def utc_datetime_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed", utc=True).astype("datetime64[ns, UTC]")


def read_stream(symbol: str, stream_name: str) -> pd.DataFrame:
    """Read canonical data plus any pre-layout files during migration."""
    return read_parquet_trees_prefer_first(raw_stream_read_dirs(symbol, stream_name))


def _deduplicate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "agg_trade_id" not in trades.columns:
        return trades
    sort_cols = [col for col in ["event_time_ns", "trade_time_ns", "received_time_ns"] if col in trades.columns]
    out = trades.sort_values(sort_cols) if sort_cols else trades.copy()
    subset = ["agg_trade_id"]
    if "symbol" in out.columns:
        subset = ["symbol", "agg_trade_id"]
    return out.drop_duplicates(subset=subset, keep="first")


def audit_trade_quality(
    trades: pd.DataFrame,
    timeframe: str,
    max_single_trade_quantity: float = 1_000.0,
    max_bar_volume: float = 20_000.0,
    extreme_ratio_volume: float = 100.0,
    extreme_ratio_cutoff: float = 0.001,
) -> dict[str, Any]:
    indexed = _bar_index(trades, timeframe)
    if indexed.empty:
        return {"status": "no_data", "rows": 0}

    work = indexed.copy()
    for col in ["quantity", "signed_quantity"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")

    if "agg_trade_id" in work.columns:
        duplicate_subset = ["agg_trade_id"]
        if "symbol" in work.columns:
            duplicate_subset = ["symbol", "agg_trade_id"]
        duplicate_trade_ids = int(work.duplicated(subset=duplicate_subset).sum())
    else:
        duplicate_trade_ids = 0

    grouped = work.groupby("bar_start", as_index=False).agg(
        trade_count=("quantity", "size"),
        trade_volume=("quantity", "sum"),
        buy_volume=("signed_quantity", lambda s: s.clip(lower=0).sum()),
        sell_volume=("signed_quantity", lambda s: (-s.clip(upper=0)).sum()),
    )
    total = grouped["buy_volume"] + grouped["sell_volume"]
    grouped["taker_buy_ratio"] = grouped["buy_volume"] / total.replace(0, np.nan)

    full_index = pd.date_range(
        start=grouped["bar_start"].min(),
        end=grouped["bar_start"].max(),
        freq=timeframe_delta(timeframe),
        tz="UTC",
    )
    missing_bars = int(len(full_index.difference(pd.DatetimeIndex(grouped["bar_start"]))))
    max_trade_quantity = float(work["quantity"].max(skipna=True)) if work["quantity"].notna().any() else 0.0
    max_observed_bar_volume = float(grouped["trade_volume"].max(skipna=True)) if not grouped.empty else 0.0
    extreme_ratio_mask = (
        grouped["trade_volume"].ge(extreme_ratio_volume)
        & (
            grouped["taker_buy_ratio"].le(extreme_ratio_cutoff)
            | grouped["taker_buy_ratio"].ge(1.0 - extreme_ratio_cutoff)
        )
    )
    extreme_ratio_bars = int(extreme_ratio_mask.sum())

    issues = []
    warnings = []
    if duplicate_trade_ids:
        issues.append(f"duplicate agg_trade_id rows={duplicate_trade_ids}")
    if max_trade_quantity > max_single_trade_quantity:
        warnings.append(f"max single aggTrade quantity={max_trade_quantity:.4f}")
    if max_observed_bar_volume > max_bar_volume:
        warnings.append(f"max bar volume={max_observed_bar_volume:.4f}")
    if extreme_ratio_bars:
        examples = grouped.loc[
            extreme_ratio_mask,
            ["bar_start", "trade_volume", "buy_volume", "sell_volume", "taker_buy_ratio"],
        ].head(5)
        warnings.append(f"extreme one-sided bars={extreme_ratio_bars}; examples={examples.to_dict('records')}")

    return {
        "status": "fail" if issues else "pass",
        "rows": int(len(work)),
        "bars": int(len(grouped)),
        "missing_bars": missing_bars,
        "duplicate_trade_ids": duplicate_trade_ids,
        "max_trade_quantity": max_trade_quantity,
        "max_bar_volume": max_observed_bar_volume,
        "extreme_ratio_bars": extreme_ratio_bars,
        "issues": issues,
        "warnings": warnings,
    }


def ensure_trade_quality(trades: pd.DataFrame, timeframe: str) -> None:
    audit = audit_trade_quality(trades, timeframe)
    if audit["status"] == "fail":
        raise ValueError("Invalid agg_trade raw data: " + "; ".join(audit["issues"]))
    if audit.get("warnings"):
        LOGGER.warning("agg_trade quality warnings: %s", "; ".join(audit["warnings"]))


def _bar_index(frame: pd.DataFrame, timeframe: str, time_col: str = "event_time_ns") -> pd.DataFrame:
    if frame.empty or time_col not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out["timestamp"] = ns_to_utc_datetime(out[time_col])
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out["bar_start"] = out["timestamp"].dt.floor(normalize_timeframe(timeframe)).astype("datetime64[ns, UTC]")
    return out


def build_trade_bars(trades: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    trades = _deduplicate_trades(trades)
    ensure_trade_quality(trades, timeframe)
    trades = _bar_index(trades, timeframe)
    if trades.empty:
        return pd.DataFrame()
    bar_delta = timeframe_delta(timeframe)

    for col in ["quantity", "signed_quantity", "price"]:
        trades[col] = pd.to_numeric(trades.get(col), errors="coerce")
    trades["notional"] = trades["quantity"] * trades["price"]
    trades["buy_quantity"] = trades["signed_quantity"].clip(lower=0)
    trades["sell_quantity"] = (-trades["signed_quantity"].clip(upper=0))

    grouped = trades.groupby("bar_start", as_index=False).agg(
        trade_count=("quantity", "size"),
        trade_volume=("quantity", "sum"),
        max_trade_quantity=("quantity", "max"),
        buy_volume=("buy_quantity", "sum"),
        sell_volume=("sell_quantity", "sum"),
        cvd_delta=("signed_quantity", "sum"),
        trade_notional=("notional", "sum"),
        vwap=("price", lambda s: np.nan),
    )
    notional = trades.groupby("bar_start")["notional"].sum()
    qty = trades.groupby("bar_start")["quantity"].sum()
    vwap = (notional / qty.replace(0, np.nan)).rename("vwap").reset_index()
    grouped = grouped.drop(columns=["vwap"]).merge(vwap, on="bar_start", how="left")
    grouped["cvd"] = grouped["cvd_delta"].cumsum()
    total = grouped["buy_volume"] + grouped["sell_volume"]
    grouped["taker_buy_ratio"] = grouped["buy_volume"] / total.replace(0, np.nan)
    grouped["trade_quality_extreme_one_sided"] = (
        grouped["trade_volume"].ge(100.0)
        & (
            grouped["taker_buy_ratio"].le(0.001)
            | grouped["taker_buy_ratio"].ge(0.999)
        )
    )
    grouped["trade_quality_large_bar"] = grouped["trade_volume"].ge(20_000.0)
    grouped["trade_quality_large_agg_trade"] = grouped["max_trade_quantity"].ge(1_000.0)
    grouped["trade_source_max_time"] = grouped["bar_start"] + bar_delta
    return grouped


def build_depth_bars(depth: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if not depth.empty and "source_endpoint" in depth.columns:
        depth = depth[~depth["source_endpoint"].astype("string").str.contains("binance_public_data_bookDepth", na=False)]
    depth = _bar_index(depth, timeframe)
    if depth.empty:
        return pd.DataFrame()
    bar_delta = timeframe_delta(timeframe)

    for col in ["book_imbalance_top_n", "bid_qty_top_n", "ask_qty_top_n"]:
        depth[col] = pd.to_numeric(depth.get(col), errors="coerce")

    grouped = depth.groupby("bar_start", as_index=False).agg(
        depth_updates=("book_imbalance_top_n", "size"),
        book_imbalance_mean=("book_imbalance_top_n", "mean"),
        book_imbalance_last=("book_imbalance_top_n", "last"),
        book_imbalance_min=("book_imbalance_top_n", "min"),
        book_imbalance_max=("book_imbalance_top_n", "max"),
        bid_qty_top_n_last=("bid_qty_top_n", "last"),
        ask_qty_top_n_last=("ask_qty_top_n", "last"),
    )
    grouped["depth_source_max_time"] = grouped["bar_start"] + bar_delta
    return grouped


def build_book_depth_percent_bars(depth: pd.DataFrame, timeframe: str, target_percentage: float = 1.0) -> pd.DataFrame:
    depth = _bar_index(depth, timeframe)
    if depth.empty:
        return pd.DataFrame()
    bar_delta = timeframe_delta(timeframe)

    for col in ["percentage", "depth", "notional"]:
        depth[col] = pd.to_numeric(depth.get(col), errors="coerce")
    # Binance bookDepth percentage rows are cumulative to that distance from mid,
    # so never sum multiple percentage bands. Use one target band, e.g. +/-1%.
    depth = depth[np.isclose(depth["percentage"].abs(), float(target_percentage), equal_nan=False)].copy()
    if depth.empty:
        return pd.DataFrame()

    depth["side"] = np.where(depth["percentage"] < 0, "bid", "ask")
    last = depth.sort_values("timestamp").groupby(["bar_start", "side"], as_index=False).last()
    qty = last.pivot(index="bar_start", columns="side", values="depth").reset_index()
    notional = last.pivot(index="bar_start", columns="side", values="notional").reset_index()
    qty = qty.rename(columns={
        "bid": f"book_depth_pct_{target_percentage:g}_bid_qty",
        "ask": f"book_depth_pct_{target_percentage:g}_ask_qty",
    })
    notional = notional.rename(columns={
        "bid": f"book_depth_pct_{target_percentage:g}_bid_notional",
        "ask": f"book_depth_pct_{target_percentage:g}_ask_notional",
    })
    grouped = qty.merge(notional, on="bar_start", how="outer")
    bid_col = f"book_depth_pct_{target_percentage:g}_bid_qty"
    ask_col = f"book_depth_pct_{target_percentage:g}_ask_qty"
    denom = grouped[bid_col] + grouped[ask_col]
    grouped[f"book_depth_pct_{target_percentage:g}_imbalance"] = (
        grouped[bid_col] - grouped[ask_col]
    ) / denom.replace(0, np.nan)
    grouped["book_depth_percent_source_max_time"] = grouped["bar_start"] + bar_delta
    return grouped


def build_force_order_bars(force_orders: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    force_orders = _bar_index(force_orders, timeframe)
    if force_orders.empty:
        return pd.DataFrame()
    bar_delta = timeframe_delta(timeframe)

    for col in ["quantity", "signed_quantity", "notional"]:
        force_orders[col] = pd.to_numeric(force_orders.get(col), errors="coerce")
    force_orders["buy_liq_qty"] = force_orders["signed_quantity"].clip(lower=0)
    force_orders["sell_liq_qty"] = (-force_orders["signed_quantity"].clip(upper=0))
    force_orders["buy_liq_notional"] = np.where(force_orders["signed_quantity"] > 0, force_orders["notional"], 0.0)
    force_orders["sell_liq_notional"] = np.where(force_orders["signed_quantity"] < 0, force_orders["notional"], 0.0)

    grouped = force_orders.groupby("bar_start", as_index=False).agg(
        force_order_count=("quantity", "size"),
        force_order_qty=("quantity", "sum"),
        force_order_signed_qty=("signed_quantity", "sum"),
        buy_liq_qty=("buy_liq_qty", "sum"),
        sell_liq_qty=("sell_liq_qty", "sum"),
        buy_liq_notional=("buy_liq_notional", "sum"),
        sell_liq_notional=("sell_liq_notional", "sum"),
    )
    grouped["force_order_source_max_time"] = grouped["bar_start"] + bar_delta
    return grouped


def build_last_value_bars(frame: pd.DataFrame, timeframe: str, columns: list[str], source_label: str) -> pd.DataFrame:
    frame = _bar_index(frame, timeframe)
    if frame.empty:
        return pd.DataFrame()
    use_cols = ["bar_start", "timestamp"] + [col for col in columns if col in frame.columns]
    if len(use_cols) == 1:
        return pd.DataFrame()
    for col in use_cols:
        if col not in {"bar_start", "timestamp"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    grouped = frame[use_cols].groupby("bar_start", as_index=False).last()
    return grouped.rename(columns={"timestamp": f"{source_label}_source_time"})


def build_bar_anchor(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    frame = _bar_index(frame, timeframe)
    if frame.empty:
        return pd.DataFrame()
    return frame[["bar_start"]].drop_duplicates().sort_values("bar_start")


class _FenwickTree:
    def __init__(self, size: int) -> None:
        self.values = [0] * (size + 1)

    def add(self, index: int, value: int) -> None:
        while index < len(self.values):
            self.values[index] += value
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        total = 0
        while index > 0:
            total += self.values[index]
            index -= index & -index
        return total


def expanding_percentile(series: pd.Series, min_periods: int = 100) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    clean = values.dropna()
    if clean.empty:
        return pd.Series(np.nan, index=values.index, dtype="float64")

    uniques = np.sort(clean.unique())
    ranks = pd.Series(np.searchsorted(uniques, clean.to_numpy(), side="left") + 1, index=clean.index)
    tree = _FenwickTree(len(uniques))
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    seen = 0
    for idx, rank in ranks.items():
        rank_int = int(rank)
        tree.add(rank_int, 1)
        seen += 1
        if seen >= min_periods:
            out.loc[idx] = tree.prefix_sum(rank_int) / seen
    return out


def load_market_close(symbol: str, timeframe: str) -> pd.DataFrame:
    display_symbol = "BTCUSD" if symbol.upper() == "BTCUSDT" else symbol.upper()
    normalized = normalize_timeframe(timeframe)
    suffix_map = {"1min": "1min", "5min": "5m", "15min": "15m", "30min": "30m"}
    suffix = suffix_map.get(normalized, normalized)
    path = PROJECT_PATHS.base_data_dir / f"{display_symbol}_{suffix}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=lambda col: col in {"date", "timestamp", "open_time", "close"})
    ts_col = next((col for col in ["date", "timestamp", "open_time"] if col in df.columns), None)
    if ts_col is None or "close" not in df.columns:
        return pd.DataFrame()
    return pd.DataFrame({
        "bar_start": utc_datetime_ns(df[ts_col]),
        "label_start_price": pd.to_numeric(df["close"], errors="coerce"),
    }).dropna(subset=["bar_start"]).sort_values("bar_start").drop_duplicates("bar_start")


def load_kline_close(symbol: str, timeframe: str) -> pd.DataFrame:
    frame = read_stream(symbol, "kline")
    frame = _bar_index(frame, timeframe)
    if frame.empty or "close" not in frame.columns:
        return pd.DataFrame()
    return pd.DataFrame({
        "bar_start": frame["bar_start"],
        "label_start_price": pd.to_numeric(frame["close"], errors="coerce"),
    }).dropna(subset=["bar_start", "label_start_price"]).sort_values("bar_start").drop_duplicates("bar_start")


def _merge_exact(base: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return frame
    if frame.empty:
        return base
    return base.merge(frame, on="bar_start", how="outer")


def _merge_last_value(
    base: pd.DataFrame,
    frame: pd.DataFrame,
    source_label: str,
    tolerance: pd.Timedelta,
    bar_delta: pd.Timedelta,
) -> pd.DataFrame:
    if frame.empty:
        return base
    if base.empty:
        base = frame[["bar_start"]].copy()
    source_time_col = f"{source_label}_source_time"
    merged = pd.merge_asof(
        base.sort_values("bar_start"),
        frame.sort_values("bar_start"),
        on="bar_start",
        direction="backward",
        tolerance=tolerance,
    )
    if source_time_col in merged.columns:
        merged[f"{source_label}_staleness_seconds"] = (
            merged["bar_start"] + bar_delta - merged[source_time_col]
        ).dt.total_seconds()
    return merged


def _regularize_bars(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    bar_delta = timeframe_delta(timeframe)
    start = frame["bar_start"].min()
    end = frame["bar_start"].max()
    grid = pd.DataFrame({"bar_start": pd.date_range(start=start, end=end, freq=bar_delta, tz="UTC")})
    grid["bar_start"] = grid["bar_start"].astype("datetime64[ns, UTC]")
    frame = frame.copy()
    frame["bar_start"] = frame["bar_start"].astype("datetime64[ns, UTC]")
    out = grid.merge(frame, on="bar_start", how="left")

    zero_fill_cols = [
        "trade_count",
        "trade_volume",
        "buy_volume",
        "sell_volume",
        "cvd_delta",
        "trade_notional",
        "depth_updates",
        "force_order_count",
        "force_order_qty",
        "force_order_signed_qty",
        "buy_liq_qty",
        "sell_liq_qty",
        "buy_liq_notional",
        "sell_liq_notional",
        "trade_quality_extreme_one_sided",
        "trade_quality_large_bar",
        "trade_quality_large_agg_trade",
    ]
    for col in zero_fill_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "cvd" in out.columns:
        out["cvd"] = pd.to_numeric(out["cvd"], errors="coerce").ffill()
    return out


def build_microstructure_features(
    symbol: str = "BTCUSDT",
    timeframe: str = "1min",
    aux_tolerance: str | pd.Timedelta = "10min",
    funding_tolerance: str | pd.Timedelta = "9h",
    close_tolerance: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    timeframe = normalize_timeframe(timeframe)
    bar_delta = timeframe_delta(timeframe)
    aux_tol = pd.Timedelta(aux_tolerance)
    funding_tol = pd.Timedelta(funding_tolerance)
    close_tol = pd.Timedelta(close_tolerance) if close_tolerance is not None else bar_delta / 2
    mark_price_stream = read_stream(symbol, "mark_price")
    premium_index_stream = read_stream(symbol, "premium_index")
    kline_stream = read_stream(symbol, "kline")

    event_frames = [
        build_trade_bars(read_stream(symbol, "agg_trade"), timeframe),
        build_depth_bars(read_stream(symbol, "book_depth"), timeframe),
        build_book_depth_percent_bars(read_stream(symbol, "book_depth_percent"), timeframe),
        build_force_order_bars(read_stream(symbol, "force_order"), timeframe),
    ]
    result = pd.DataFrame()
    for frame in event_frames:
        result = _merge_exact(result, frame)
    for frame in [
        build_bar_anchor(kline_stream, timeframe),
        build_bar_anchor(mark_price_stream, timeframe),
        build_bar_anchor(premium_index_stream, timeframe),
    ]:
        result = _merge_exact(result, frame)

    funding_settled_stream = read_stream(symbol, "funding_rate_settled")
    if funding_settled_stream.empty:
        funding_settled_stream = read_stream(symbol, "funding_rate")
        if not funding_settled_stream.empty and "funding_rate" in funding_settled_stream.columns:
            funding_settled_stream = funding_settled_stream.rename(columns={"funding_rate": "funding_rate_settled"})
    funding_settled_frame = build_last_value_bars(
        funding_settled_stream,
        timeframe,
        ["funding_rate_settled"],
        "funding_rate_settled",
    )
    funding_predicted_frame = pd.DataFrame()
    if "funding_rate" in mark_price_stream.columns:
        funding_predicted_stream = mark_price_stream.rename(columns={"funding_rate": "funding_rate_predicted"})
        funding_predicted_frame = build_last_value_bars(
            funding_predicted_stream,
            timeframe,
            ["funding_rate_predicted"],
            "funding_rate_predicted",
        )
    if not premium_index_stream.empty and "close" in premium_index_stream.columns:
        premium_index_stream = premium_index_stream.rename(columns={"close": "premium_index_close"})

    last_value_frames = [
        ("mark_price", build_last_value_bars(mark_price_stream, timeframe, ["mark_price", "index_price"], "mark_price")),
        ("funding_rate_predicted", funding_predicted_frame),
        ("premium_index", build_last_value_bars(premium_index_stream, timeframe, ["premium_index_close"], "premium_index")),
        ("open_interest", build_last_value_bars(read_stream(symbol, "open_interest_snapshot"), timeframe, ["open_interest"], "open_interest")),
        ("global_ls", build_last_value_bars(read_stream(symbol, "global_ls_account_ratio"), timeframe, ["long_short_ratio", "long_account", "short_account"], "global_ls")),
        ("top_account_ls", build_last_value_bars(read_stream(symbol, "top_ls_account_ratio"), timeframe, ["long_short_ratio", "long_account", "short_account"], "top_account_ls").rename(
            columns={
                "long_short_ratio": "top_account_long_short_ratio",
                "long_account": "top_account_long",
                "short_account": "top_account_short",
            }
        )),
        ("top_position_ls", build_last_value_bars(read_stream(symbol, "top_ls_position_ratio"), timeframe, ["long_short_ratio", "long_account", "short_account"], "top_position_ls").rename(
            columns={
                "long_short_ratio": "top_position_long_short_ratio",
                "long_account": "top_position_long",
                "short_account": "top_position_short",
            }
        )),
    ]

    for source_label, frame in last_value_frames:
        result = _merge_last_value(result, frame, source_label, aux_tol, bar_delta)
    result = _merge_last_value(result, funding_settled_frame, "funding_rate_settled", funding_tol, bar_delta)

    if result.empty:
        return pd.DataFrame()

    result = result.sort_values("bar_start").reset_index(drop=True)
    result = _regularize_bars(result, timeframe)

    if "open_interest" in result.columns:
        result["open_interest_change"] = result["open_interest"].diff()
        result["open_interest_change_pct"] = result["open_interest"].pct_change() * 100.0

    for col in ["funding_rate_settled", "funding_rate_predicted", "premium_index_close"]:
        if col in result.columns:
            percentile_col = f"{col}_percentile"
            result[percentile_col] = expanding_percentile(result[col], min_periods=100)
            result[f"{col}_bucket"] = pd.cut(
                result[percentile_col],
                bins=[-np.inf, 0.1, 0.25, 0.75, 0.9, np.inf],
                labels=["p0_10", "p10_25", "p25_75", "p75_90", "p90_100"],
            ).astype("string")

    close = load_kline_close(symbol, timeframe)
    if not close.empty:
        close = close.copy()
        close["label_start_time"] = close["bar_start"] + bar_delta
        result["label_price_bar_start"] = result["bar_start"] + bar_delta
        close["bar_start"] = close["bar_start"].astype("datetime64[ns, UTC]")
        close["label_start_time"] = close["label_start_time"].astype("datetime64[ns, UTC]")
        result["bar_start"] = result["bar_start"].astype("datetime64[ns, UTC]")
        result["label_price_bar_start"] = result["label_price_bar_start"].astype("datetime64[ns, UTC]")
        result = result.merge(
            close.rename(columns={"bar_start": "close_bar_start"}),
            left_on="label_price_bar_start",
            right_on="close_bar_start",
            how="left",
        )
        if close_tol is not None:
            matched_age = (result["label_price_bar_start"] - result["close_bar_start"]).abs()
            stale_label = result["close_bar_start"].notna() & (matched_age > close_tol)
            result.loc[stale_label, ["label_start_time", "label_start_price"]] = pd.NA
        result = result.drop(columns=["close_bar_start"], errors="ignore")

    result["feature_source_max_time"] = result["bar_start"] + bar_delta
    if "mark_price" in result.columns:
        fallback_label_price = result["mark_price"].shift(-1)
        fallback_label_time = result["bar_start"] + (bar_delta * 2)
        if "label_start_price" not in result.columns:
            result["label_start_price"] = pd.NA
        if "label_start_time" not in result.columns:
            result["label_start_time"] = pd.NaT
        missing_label = result["label_start_price"].isna()
        result.loc[missing_label, "label_start_price"] = fallback_label_price[missing_label]
        result.loc[missing_label, "label_start_time"] = fallback_label_time[missing_label]
    if "label_start_time" not in result.columns:
        result["label_start_time"] = result["feature_source_max_time"]
    result["label_start_lag_seconds"] = (
        pd.to_datetime(result["label_start_time"], utc=True) - pd.to_datetime(result["feature_source_max_time"], utc=True)
    ).dt.total_seconds()
    return result


def write_features(symbol: str, timeframe: str) -> Path:
    timeframe = normalize_timeframe(timeframe)
    df = build_microstructure_features(symbol=symbol, timeframe=timeframe)
    if df.empty:
        raise RuntimeError(f"No microstructure rows found for {symbol} {timeframe}")
    out_dir = feature_dir(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"microstructure_features_{timeframe}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bar-level microstructure features from raw Binance Parquet streams.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1min", help="Pandas frequency, e.g. 1min, 5min, 15min.")
    parser.add_argument("--aux-tolerance", default="10min", help="Maximum age for REST/mark-price asof joins.")
    parser.add_argument("--funding-tolerance", default="9h", help="Maximum age for settled funding-rate asof joins.")
    parser.add_argument("--close-tolerance", default=None, help="Maximum age for market close joins; defaults to half the bar length.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    try:
        df = build_microstructure_features(
            args.symbol,
            args.timeframe,
            args.aux_tolerance,
            args.funding_tolerance,
            args.close_tolerance,
        )
        if df.empty:
            raise RuntimeError(f"No microstructure rows found for {args.symbol} {args.timeframe}")
        out_dir = feature_dir(args.symbol)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"microstructure_features_{normalize_timeframe(args.timeframe)}.parquet"
        df.to_parquet(out_path, index=False)
    except Exception as exc:
        LOGGER.error("Feature build failed: %s", exc)
        sys.exit(1)
    LOGGER.info("Saved microstructure features: %s", out_path)


if __name__ == "__main__":
    main()
