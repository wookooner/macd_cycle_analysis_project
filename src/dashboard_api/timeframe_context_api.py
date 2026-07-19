from __future__ import annotations

from collections import OrderedDict
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.common.paths import PROJECT_PATHS


router = APIRouter(prefix="/api/timeframe-context")

SUPPORTED_RESOLUTIONS = ("15m", "1h", "1min")
CONTEXT_TIMEFRAMES = ("1w", "1d", "4h", "1h", "15m")
MAJOR_TIMEFRAMES = ("1w", "1d", "4h", "1h")
TIMEFRAME_COLUMNS = ("key", "type", "time_prog", "candle_prog", "changed")
SUMMARY_COLUMNS = ("n_up_4", "combo_4", "major_late_count")
DEFAULT_ASSET = "btc"
SERIES_CACHE: "OrderedDict[tuple[str, str, int, str | None, str | None, float, float], dict[str, Any]]" = OrderedDict()
SERIES_CACHE_MAX = 24
FALLBACK_MARKET_FILES = {
    "btc": {
        "15m": "BTCUSD_15m.csv",
        "1h": "BTCUSD_1h.csv",
    }
}
CANDLE_SECONDS = {
    "15m": 15 * 60,
    "1h": 3600,
    "4h": 4 * 3600,
    "1d": 24 * 3600,
    "1w": 7 * 24 * 3600,
}


def _normalize_asset(asset: str) -> str:
    return str(asset or DEFAULT_ASSET).strip().lower()


def _normalize_resolution(resolution: str) -> str:
    return str(resolution or "1h").strip().lower()


def _context_path(asset: str, resolution: str) -> Path:
    return PROJECT_PATHS.context_dir(asset) / f"timeframe_context_{resolution}.parquet"


def _cycle_dim_path(asset: str) -> Path:
    return PROJECT_PATHS.context_dir(asset) / "cycle_dim.parquet"


def _cycle_parquet_path(asset: str, timeframe: str) -> Path:
    return PROJECT_PATHS.cycle_structured_dir / asset / f"cycles_{timeframe}.parquet"


def _market_path(asset: str, resolution: str) -> Path | None:
    file_name = FALLBACK_MARKET_FILES.get(asset, {}).get(resolution)
    return PROJECT_PATHS.base_data_dir / file_name if file_name else None


def _load_market_timestamps(market_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            market = pd.read_csv(market_path)
            break
        except Exception as exc:  # pragma: no cover - defensive path for live-written CSVs
            last_error = exc
            time.sleep(0.15)
    else:
        raise last_error or ValueError(f"Failed to read market file: {market_path}")

    timestamp_series = None
    if "date" in market.columns:
        timestamp_series = pd.to_datetime(market["date"], errors="coerce")
    elif "unix" in market.columns:
        timestamp_series = pd.to_datetime(market["unix"], unit="s", errors="coerce")
    else:
        raise ValueError(f"Market file missing timestamp columns: {market_path}")

    result = pd.DataFrame({"timestamp": timestamp_series})
    result = result.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return result


def _parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value or not isinstance(value, str):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise HTTPException(status_code=400, detail=f"Invalid datetime value: {value}")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)
    return parsed


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _direction_label(value: Any) -> str | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    try:
        numeric = int(cleaned)
    except (TypeError, ValueError):
        text = str(cleaned).lower()
        if "up" in text or text == "u":
            return "up"
        if "down" in text or text == "d":
            return "down"
        return text
    if numeric == 1:
        return "up"
    if numeric == -1:
        return "down"
    return "gap"


def _load_cycle_lookup(dim_path: Path) -> dict[int, dict[str, Any]]:
    dim = pd.read_parquet(dim_path)
    required = ["cycle_key", "cycle_id", "timeframe", "cycle_type", "start_date", "end_date", "duration_candles"]
    dim = dim[[column for column in required if column in dim.columns]].copy()
    lookup: dict[int, dict[str, Any]] = {}
    for row in dim.itertuples(index=False):
        item = row._asdict()
        cycle_key = _clean_value(item.get("cycle_key"))
        if cycle_key is None:
            continue
        lookup[int(cycle_key)] = {
            "cycle_id": _clean_value(item.get("cycle_id")),
            "timeframe": _clean_value(item.get("timeframe")),
            "cycle_type": _direction_label(item.get("cycle_type")),
            "start_date": _clean_value(item.get("start_date")),
            "end_date": _clean_value(item.get("end_date")),
            "duration_candles": _clean_value(item.get("duration_candles")),
        }
    return lookup


def _serialize_rows(df: pd.DataFrame, cycle_lookup: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in df.to_dict(orient="records"):
        timestamp = pd.to_datetime(item["timestamp"])
        row: dict[str, Any] = {
            "unix": int(timestamp.value // 1_000_000_000),
            "date": timestamp.isoformat(),
        }

        for timeframe in CONTEXT_TIMEFRAMES:
            key = _clean_value(item.get(f"{timeframe}_key"))
            key_int = int(key) if key not in (None, 0) else 0
            row[f"{timeframe}_key"] = key_int
            row[f"{timeframe}_type"] = _direction_label(item.get(f"{timeframe}_type"))
            row[f"{timeframe}_time_prog"] = _clean_value(item.get(f"{timeframe}_time_prog"))
            row[f"{timeframe}_candle_prog"] = _clean_value(item.get(f"{timeframe}_candle_prog"))
            row[f"{timeframe}_changed"] = bool(_clean_value(item.get(f"{timeframe}_changed")) or False)

            cycle = cycle_lookup.get(key_int) if key_int else None
            if cycle:
                prefix = f"{timeframe}_cycle"
                row[f"{prefix}_id"] = cycle["cycle_id"]
                row[f"{prefix}_type"] = cycle["cycle_type"]
                row[f"{prefix}_start_date"] = cycle["start_date"]
                row[f"{prefix}_end_date"] = cycle["end_date"]
                row[f"{prefix}_duration_candles"] = cycle["duration_candles"]

        for column in SUMMARY_COLUMNS:
            row[column] = _clean_value(item.get(column))
        rows.append(row)
    return rows


def _load_cycle_frame(asset: str, timeframe: str, max_ts: pd.Timestamp | None) -> pd.DataFrame:
    path = _cycle_parquet_path(asset, timeframe)
    if not path.exists():
        return pd.DataFrame()
    columns = ["cycle_id", "cycle_type", "start_date", "end_date", "duration_candles"]
    df = pd.read_parquet(path, columns=columns).copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df = df.dropna(subset=["start_date", "end_date"]).sort_values("start_date").reset_index(drop=True)
    if df.empty:
        return df

    effective_end = df["end_date"].copy()
    candle_delta = pd.to_timedelta(CANDLE_SECONDS.get(timeframe, 3600), unit="s")
    effective_end = effective_end + candle_delta
    if max_ts is not None:
        effective_end.iloc[-1] = max(effective_end.iloc[-1], max_ts + pd.to_timedelta(3600, unit="s"))
    df["_effective_end"] = effective_end
    df["_fallback_key"] = range(1, len(df) + 1)
    return df


def _active_cycle(cycles: pd.DataFrame, timestamp: pd.Timestamp) -> dict[str, Any] | None:
    if cycles.empty:
        return None
    starts = cycles["start_date"].values.astype("datetime64[ns]")
    ts_value = timestamp.to_datetime64()
    idx = starts.searchsorted(ts_value, side="right") - 1
    if idx < 0:
        return None
    row = cycles.iloc[int(idx)]
    if timestamp >= row["_effective_end"]:
        return None
    return row.to_dict()


def _progress(timestamp: pd.Timestamp, cycle: dict[str, Any] | None) -> float:
    if not cycle:
        return 0.0
    start_date = cycle.get("start_date")
    end_date = cycle.get("_effective_end")
    if start_date is None or end_date is None or end_date <= start_date:
        return 0.0
    value = (timestamp - start_date).total_seconds() / (end_date - start_date).total_seconds()
    return round(max(0.0, min(1.0, value)), 4)


def _serialize_fallback_row(timestamp: pd.Timestamp, cycles_by_tf: dict[str, pd.DataFrame]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "unix": int(timestamp.value // 1_000_000_000),
        "date": timestamp.isoformat(),
    }
    n_up = 0
    combo_bits: list[str] = []
    major_late_count = 0

    for timeframe in CONTEXT_TIMEFRAMES:
        cycle = _active_cycle(cycles_by_tf.get(timeframe, pd.DataFrame()), timestamp)
        direction = _direction_label(cycle.get("cycle_type")) if cycle else "gap"
        progress = _progress(timestamp, cycle)
        key = int(cycle.get("_fallback_key")) if cycle else 0
        row[f"{timeframe}_key"] = key
        row[f"{timeframe}_type"] = direction
        row[f"{timeframe}_time_prog"] = progress
        row[f"{timeframe}_candle_prog"] = progress
        row[f"{timeframe}_changed"] = timestamp == cycle.get("start_date") if cycle else False
        if cycle:
            prefix = f"{timeframe}_cycle"
            row[f"{prefix}_id"] = _clean_value(cycle.get("cycle_id"))
            row[f"{prefix}_type"] = direction
            row[f"{prefix}_start_date"] = _clean_value(cycle.get("start_date"))
            row[f"{prefix}_end_date"] = _clean_value(cycle.get("_effective_end"))
            row[f"{prefix}_duration_candles"] = _clean_value(cycle.get("duration_candles"))

        if timeframe in MAJOR_TIMEFRAMES:
            if direction == "up":
                n_up += 1
                combo_bits.append("U")
            else:
                combo_bits.append("D")
            if progress > 0.8:
                major_late_count += 1

    row["n_up_4"] = n_up
    row["combo_4"] = "".join(combo_bits)
    row["major_late_count"] = major_late_count
    return row


def _build_cycle_fallback_rows(
    asset: str,
    resolution: str,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
    limit: int,
) -> tuple[list[dict[str, Any]], int, str | None]:
    if resolution not in ("15m", "1h"):
        return [], 0, None
    market_path = _market_path(asset, resolution)
    if not market_path or not market_path.exists():
        return [], 0, None

    market = _load_market_timestamps(market_path)
    if start_ts is not None:
        market = market[market["timestamp"] >= start_ts]
    if end_ts is not None:
        market = market[market["timestamp"] <= end_ts]
    filtered_rows = len(market)
    if limit > 0:
        market = market.tail(limit)
    if market.empty:
        return [], filtered_rows, str(market_path)

    max_ts = market["timestamp"].max()
    cycles_by_tf = {timeframe: _load_cycle_frame(asset, timeframe, max_ts) for timeframe in CONTEXT_TIMEFRAMES}
    rows = [_serialize_fallback_row(timestamp, cycles_by_tf) for timestamp in market["timestamp"]]
    return rows, filtered_rows, str(market_path)


def _context_rows_from_parquet(
    context_path: Path,
    dim_path: Path,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
    limit: int,
) -> tuple[list[dict[str, Any]], int, int]:
    columns = ["timestamp"]
    for timeframe in CONTEXT_TIMEFRAMES:
        columns.extend(f"{timeframe}_{suffix}" for suffix in TIMEFRAME_COLUMNS)
    columns.extend(SUMMARY_COLUMNS)

    df = pd.read_parquet(context_path, columns=columns).sort_values("timestamp").reset_index(drop=True)
    total_rows = len(df)
    if start_ts is not None:
        df = df[df["timestamp"] >= start_ts]
    if end_ts is not None:
        df = df[df["timestamp"] <= end_ts]
    filtered_rows = len(df)
    if limit > 0:
        df = df.tail(limit)

    cycle_lookup = _load_cycle_lookup(dim_path)
    return _serialize_rows(df, cycle_lookup), filtered_rows, total_rows


@router.get("/series")
def get_timeframe_context_series(
    asset: str = Query(DEFAULT_ASSET),
    resolution: str = Query("1h"),
    start: str | None = Query(None),
    end: str | None = Query(None),
    limit: int = Query(0, ge=0),
) -> dict[str, Any]:
    normalized_asset = _normalize_asset(asset)
    normalized_resolution = _normalize_resolution(resolution)
    if normalized_resolution not in SUPPORTED_RESOLUTIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported context resolution: {resolution}")

    context_path = _context_path(normalized_asset, normalized_resolution)
    dim_path = _cycle_dim_path(normalized_asset)
    fallback_market_path = _market_path(normalized_asset, normalized_resolution)
    if not context_path.exists() and not fallback_market_path:
        raise HTTPException(status_code=404, detail=f"Context parquet not found: {context_path}")
    if context_path.exists() and not dim_path.exists():
        raise HTTPException(status_code=404, detail=f"Cycle dimension parquet not found: {dim_path}")

    start_ts = _parse_timestamp(start)
    end_ts = _parse_timestamp(end)
    source_mtimes = [
        context_path.stat().st_mtime if context_path.exists() else 0.0,
        dim_path.stat().st_mtime if dim_path.exists() else 0.0,
    ]
    for timeframe in CONTEXT_TIMEFRAMES:
        path = _cycle_parquet_path(normalized_asset, timeframe)
        source_mtimes.append(path.stat().st_mtime if path.exists() else 0.0)
    if fallback_market_path and fallback_market_path.exists():
        source_mtimes.append(fallback_market_path.stat().st_mtime)
    cache_key = (
        normalized_asset,
        normalized_resolution,
        limit,
        start_ts.isoformat() if start_ts is not None else None,
        end_ts.isoformat() if end_ts is not None else None,
        max(source_mtimes),
        sum(source_mtimes),
    )
    cached = SERIES_CACHE.get(cache_key)
    if cached is not None:
        SERIES_CACHE.move_to_end(cache_key)
        return cached

    fallback_rows, fallback_filtered_rows, fallback_source = _build_cycle_fallback_rows(
        normalized_asset,
        normalized_resolution,
        start_ts,
        end_ts,
        limit,
    )
    context_rows: list[dict[str, Any]] = []
    context_filtered_rows = 0
    context_total_rows = 0
    if context_path.exists() and dim_path.exists():
        context_rows, context_filtered_rows, context_total_rows = _context_rows_from_parquet(
            context_path,
            dim_path,
            start_ts,
            end_ts,
            limit,
        )

    row_by_unix = {row["unix"]: row for row in context_rows}
    row_by_unix.update({row["unix"]: row for row in fallback_rows})
    rows = [row_by_unix[key] for key in sorted(row_by_unix)]
    if limit > 0 and len(rows) > limit:
        rows = rows[-limit:]
    source_paths = [str(context_path)] if context_path.exists() else []
    if fallback_source:
        source_paths.append(fallback_source)
    payload = {
        "meta": {
            "asset": normalized_asset,
            "resolution": normalized_resolution,
            "sourcePath": " + ".join(source_paths),
            "rowCount": len(rows),
            "filteredRowCount": max(context_filtered_rows, fallback_filtered_rows),
            "totalRowCount": max(context_total_rows, fallback_filtered_rows),
            "contextRows": len(context_rows),
            "fallbackRows": len(fallback_rows),
            "startDate": rows[0]["date"] if rows else None,
            "endDate": rows[-1]["date"] if rows else None,
        },
        "rows": rows,
    }
    SERIES_CACHE[cache_key] = payload
    SERIES_CACHE.move_to_end(cache_key)
    while len(SERIES_CACHE) > SERIES_CACHE_MAX:
        SERIES_CACHE.popitem(last=False)
    return payload
