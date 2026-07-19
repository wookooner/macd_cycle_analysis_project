"""Historical order-flow footprint surface API.

The raw Binance aggTrade partitions are deliberately read here instead of from
indicator CSVs: a footprint must preserve aggressor side at each price level.
"""

from __future__ import annotations

import math
from datetime import date as Date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from data_pipeline.microstructure.paths import raw_stream_read_dirs


router = APIRouter(prefix="/api/footprint", tags=["footprint"])
DEFAULT_SYMBOL = "BTCUSDT"
MAX_MINUTES = 240
MAX_PRICE_LEVELS = 600
IMBALANCE_RATIO = 3.0


def _partition_files(symbol: str, session_date: str) -> list[Path]:
    """Select same-day parquet partitions, preferring canonical layout paths."""
    selected: dict[Path, Path] = {}
    for root in raw_stream_read_dirs(symbol, "agg_trade"):
        partition = root / f"date={session_date}"
        if not partition.exists():
            continue
        for path in sorted(partition.glob("*.parquet")):
            selected.setdefault(path.name, path)
    return list(selected.values())


def _read_session_trades(symbol: str, session_date: str) -> tuple[pd.DataFrame, int, int]:
    paths = _partition_files(symbol, session_date)
    if not paths:
        raise HTTPException(status_code=404, detail=f"No aggTrade parquet data found for {symbol} on {session_date} UTC.")

    # Keep this concat explicit: duplicate index labels corrupt side assignment
    # when grouped or joined later.
    frames = [pd.read_parquet(path) for path in paths]
    trades = pd.concat(frames, ignore_index=True, sort=False).reset_index(drop=True)
    combined_row_count = len(trades)
    required = {"agg_trade_id", "price", "quantity", "is_buyer_maker"}
    if "trade_time_ns" not in trades.columns and "event_time_ns" not in trades.columns:
        required.add("trade_time_ns")
    missing = required.difference(trades.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"aggTrade parquet is missing required columns: {sorted(missing)}")

    subset = ["agg_trade_id"]
    if "symbol" in trades.columns:
        subset.insert(0, "symbol")
    sort_col = "trade_time_ns" if "trade_time_ns" in trades.columns else "event_time_ns"
    trades = trades.sort_values(sort_col, kind="stable").drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return trades, len(paths), combined_row_count


def _maker_flags(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes"})


def _parse_window(session_date: str, start: str, minutes: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        start_dt = datetime.fromisoformat(f"{session_date}T{start}:00").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="start must use HH:MM (UTC).") from exc
    start_ts = pd.Timestamp(start_dt)
    return start_ts, start_ts + pd.Timedelta(minutes=minutes)


def _serialize_number(value: float) -> float:
    return round(float(value), 8)


@router.get("/dates")
def footprint_dates(symbol: str = Query(DEFAULT_SYMBOL)) -> dict[str, Any]:
    symbol = symbol.upper()
    dates: set[str] = set()
    for root in raw_stream_read_dirs(symbol, "agg_trade"):
        if not root.exists():
            continue
        for partition in root.glob("date=*"):
            if partition.is_dir() and any(partition.glob("*.parquet")):
                dates.add(partition.name.removeprefix("date="))
    ordered_dates = sorted(dates)
    latest_start = None
    if ordered_dates:
        try:
            latest_trades, _, _ = _read_session_trades(symbol, ordered_dates[-1])
            time_col = "trade_time_ns" if "trade_time_ns" in latest_trades.columns else "event_time_ns"
            first_timestamp = pd.to_datetime(pd.to_numeric(latest_trades[time_col], errors="coerce"), unit="ns", utc=True).min()
            if pd.notna(first_timestamp):
                latest_start = first_timestamp.strftime("%H:%M")
        except HTTPException:
            pass
    return {"symbol": symbol, "dates": ordered_dates, "latestStart": latest_start}


@router.get("/surface")
def footprint_surface(
    session_date: str = Query(..., alias="date"),
    start: str = Query("00:00"),
    minutes: int = Query(30, ge=1, le=MAX_MINUTES),
    tick: float = Query(5.0, gt=0, le=1_000),
    symbol: str = Query(DEFAULT_SYMBOL),
) -> dict[str, Any]:
    symbol = symbol.upper()
    try:
        Date.fromisoformat(session_date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="date must use YYYY-MM-DD.") from exc

    start_ts, end_ts = _parse_window(session_date, start, minutes)
    trades, source_file_count, combined_row_count = _read_session_trades(symbol, session_date)
    time_col = "trade_time_ns" if "trade_time_ns" in trades.columns else "event_time_ns"
    work = trades.copy()
    work["timestamp"] = pd.to_datetime(pd.to_numeric(work[time_col], errors="coerce"), unit="ns", utc=True)
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce")
    work = work.dropna(subset=["timestamp", "price", "quantity"])
    work = work[(work["timestamp"] >= start_ts) & (work["timestamp"] < end_ts)].copy().reset_index(drop=True)
    if work.empty:
        raise HTTPException(status_code=404, detail="No aggTrade data exists in the selected UTC window.")

    work["minute"] = work["timestamp"].dt.floor("min")
    work["is_sell"] = _maker_flags(work["is_buyer_maker"])
    work["buy_volume"] = np.where(work["is_sell"], 0.0, work["quantity"])
    work["sell_volume"] = np.where(work["is_sell"], work["quantity"], 0.0)
    work["price_level"] = np.floor((work["price"] + tick * 1e-9) / tick) * tick
    work["price_level"] = work["price_level"].round(8)

    minute_quality = work.groupby("minute", as_index=False).agg(
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        trade_count=("quantity", "size"),
    )
    minute_quality["buy_ratio"] = minute_quality["buy_volume"] / (minute_quality["buy_volume"] + minute_quality["sell_volume"])
    extreme_minutes = minute_quality[(minute_quality["buy_ratio"] <= 0) | (minute_quality["buy_ratio"] >= 1)]
    if not extreme_minutes.empty:
        raise HTTPException(
            status_code=422,
            detail="AggTrade side validation failed: a populated minute has a 0% or 100% buy ratio. Refusing to render a potentially misparsed footprint.",
        )

    grouped = work.groupby(["minute", "price_level"], as_index=False).agg(
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        trade_count=("quantity", "size"),
    )
    grouped["total_volume"] = grouped["buy_volume"] + grouped["sell_volume"]
    grouped["delta"] = grouped["buy_volume"] - grouped["sell_volume"]

    min_price = math.floor(float(grouped["price_level"].min()) / tick) * tick
    max_price = math.ceil(float(grouped["price_level"].max()) / tick) * tick
    level_count = int(round((max_price - min_price) / tick)) + 1
    if level_count > MAX_PRICE_LEVELS:
        raise HTTPException(status_code=422, detail=f"Selected range needs {level_count} price levels. Increase tick size above {tick:.2f}.")
    price_levels = [round(max_price - index * tick, 8) for index in range(level_count)]
    time_slots = pd.date_range(start_ts, periods=minutes, freq="min", tz="UTC")
    time_index = {slot: index for index, slot in enumerate(time_slots)}

    cells: dict[tuple[int, float], dict[str, Any]] = {}
    for row in grouped.itertuples(index=False):
        index = time_index.get(row.minute)
        if index is None:
            continue
        cells[(index, float(row.price_level))] = {
            "timeIndex": index,
            "price": _serialize_number(row.price_level),
            "buyVolume": _serialize_number(row.buy_volume),
            "sellVolume": _serialize_number(row.sell_volume),
            "totalVolume": _serialize_number(row.total_volume),
            "delta": _serialize_number(row.delta),
            "tradeCount": int(row.trade_count),
            "isPoc": False,
            "buyImbalance": False,
            "sellImbalance": False,
        }

    poc_by_time: list[float | None] = [None] * minutes
    for index in range(minutes):
        column = [cell for (time_index_value, _), cell in cells.items() if time_index_value == index]
        if not column:
            continue
        poc = max(column, key=lambda cell: cell["totalVolume"])
        poc["isPoc"] = True
        poc_by_time[index] = poc["price"]

    # True diagonal imbalance: sell(P) versus buy(P + one tick).  Horizontal
    # buy/sell comparisons at the same price are intentionally never used.
    for index in range(minutes):
        for lower_price in price_levels[1:]:
            upper_price = round(lower_price + tick, 8)
            lower = cells.get((index, lower_price))
            upper = cells.get((index, upper_price))
            sell_at_lower = lower["sellVolume"] if lower else 0.0
            buy_at_upper = upper["buyVolume"] if upper else 0.0
            if sell_at_lower > 0 and buy_at_upper > 0:
                if sell_at_lower >= buy_at_upper * IMBALANCE_RATIO and lower:
                    lower["sellImbalance"] = True
                if buy_at_upper >= sell_at_lower * IMBALANCE_RATIO and upper:
                    upper["buyImbalance"] = True

    quality_rows = minute_quality.sort_values("minute")
    return {
        "meta": {
            "symbol": symbol,
            "timezone": "UTC",
            "date": session_date,
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "minutes": minutes,
            "tick": tick,
            "sourceFileCount": source_file_count,
            "rawTradeCount": int(combined_row_count),
            "windowTradeCount": int(len(work)),
            "deduplicatedTradeCount": int(len(trades)),
            "imbalanceRatio": IMBALANCE_RATIO,
            "quality": {
                "status": "pass",
                "buyRatioMin": _serialize_number(quality_rows["buy_ratio"].min()),
                "buyRatioMax": _serialize_number(quality_rows["buy_ratio"].max()),
                "extremeMinuteCount": 0,
            },
        },
        "timeSlots": [slot.isoformat() for slot in time_slots],
        "priceLevels": price_levels,
        "pocByTime": poc_by_time,
        "cells": list(cells.values()),
    }
