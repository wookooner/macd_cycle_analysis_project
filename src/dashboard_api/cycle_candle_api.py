from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.common.paths import PROJECT_PATHS


router = APIRouter(prefix="/api/cycle-candles")
SERIES_CACHE: "OrderedDict[tuple[str, str, int, float, float | None], dict[str, Any]]" = OrderedDict()
SERIES_CACHE_MAX = 32
HIERARCHY_CACHE: dict[tuple[str, float | None], dict[str, Any]] = {}
PARENT_METRIC_CACHE: dict[tuple[str, str, float], dict[str, dict[str, float | None]]] = {}

SUPPORTED_TIMEFRAMES = ("15m", "1h", "4h", "1d", "1w")
TIMEFRAME_SECONDS = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
}
DEFAULT_ASSET = "btc"
MARKET_FILES = {
    "btc": {
        "15m": "BTCUSD_15m.csv",
        "1h": "BTCUSD_1h.csv",
        "4h": "BTCUSD_4h.csv",
        "1d": "BTCUSD_1d.csv",
        "1w": "BTCUSD_1w.csv",
    }
}


def _normalize_asset(asset: str) -> str:
    return str(asset or DEFAULT_ASSET).strip().lower()


def _normalize_timeframe(timeframe: str) -> str:
    return str(timeframe or "").strip().lower()


def _clean_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return str(value)


def _to_datetime(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def _to_datetime_label(value: Any) -> str | None:
    parsed = _to_datetime(value)
    if parsed is None:
        return None
    return parsed.isoformat()


def _find_cycle_parquet(asset: str, timeframe: str) -> Path | None:
    base_dir = PROJECT_PATHS.cycle_structured_dir
    candidates = [
        base_dir / asset / f"cycles_{timeframe}.parquet",
        base_dir / f"cycles_{timeframe}.parquet",
        base_dir / asset / f"cycles_{timeframe}_enriched.parquet",
        base_dir / f"cycles_{timeframe}_enriched.parquet",
    ]
    return next((path for path in candidates if path.exists()), None)


def _find_hierarchy(asset: str) -> dict[str, Any]:
    base_dir = PROJECT_PATHS.cycle_structured_dir
    candidates = [
        base_dir / asset / "cycle_hierarchy_map.json",
        base_dir / "cycle_hierarchy_map.json",
    ]
    for path in candidates:
        if path.exists():
            cache_key = (asset, path.stat().st_mtime)
            if cache_key not in HIERARCHY_CACHE:
                HIERARCHY_CACHE.clear()
                HIERARCHY_CACHE[cache_key] = json.loads(path.read_text(encoding="utf-8"))
            return HIERARCHY_CACHE[cache_key]
    return {}


def _available_assets() -> list[str]:
    base_dir = PROJECT_PATHS.cycle_structured_dir
    assets = set()
    if (base_dir / DEFAULT_ASSET).exists():
        assets.add(DEFAULT_ASSET)
    for path in base_dir.glob("*/cycles_*.parquet"):
        if path.parent.is_dir():
            asset = path.parent.name.lower()
            if asset != "archive":
                assets.add(asset)
    if not assets and any(_find_cycle_parquet(DEFAULT_ASSET, tf) for tf in SUPPORTED_TIMEFRAMES):
        assets.add(DEFAULT_ASSET)
    return sorted(assets)


def _available_timeframes(asset: str) -> list[str]:
    return [timeframe for timeframe in SUPPORTED_TIMEFRAMES if _find_cycle_parquet(asset, timeframe)]


def _as_candle_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _cycle_type_label(value: Any) -> str | None:
    label = _clean_text(value)
    if not label:
        return None
    lowered = label.lower()
    if "up" in lowered or lowered == "u":
        return "up"
    if "down" in lowered or lowered == "d":
        return "down"
    return label


def _feature_value(features: Any, *path: str) -> Any:
    current = features if isinstance(features, dict) else {}
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _candle_value(candle: dict[str, Any], key: str) -> float | None:
    return _clean_number(candle.get(key))


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _market_end_date(asset: str, timeframe: str) -> str | None:
    file_name = MARKET_FILES.get(asset, {}).get(timeframe)
    if not file_name:
        return None
    path = PROJECT_PATHS.base_data_dir / file_name
    if not path.exists():
        return None
    try:
        tail = pd.read_csv(path, usecols=["date"]).tail(1)
    except Exception:
        return None
    if tail.empty:
        return None
    return _to_datetime_label(tail.iloc[0]["date"])


def _extend_latest_cycle_with_market(rows_df: pd.DataFrame, asset: str, timeframe: str) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df
    file_name = MARKET_FILES.get(asset, {}).get(timeframe)
    if not file_name:
        return rows_df
    path = PROJECT_PATHS.base_data_dir / file_name
    if not path.exists():
        return rows_df

    latest_index = rows_df.index[-1]
    cycle_start = _to_datetime(rows_df.at[latest_index, "start_date"])
    cycle_end = _to_datetime(rows_df.at[latest_index, "end_date"])
    if cycle_start is None or cycle_end is None:
        return rows_df

    try:
        market_df = pd.read_csv(path)
    except Exception:
        return rows_df
    if "date" not in market_df.columns:
        return rows_df

    market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce")
    market_df = market_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if market_df.empty or market_df["date"].iloc[-1] <= cycle_end:
        return rows_df

    cycle_market = market_df[(market_df["date"] >= cycle_start) & (market_df["date"] <= market_df["date"].iloc[-1])].copy()
    if cycle_market.empty:
        return rows_df

    def _market_row_to_candle(row: pd.Series) -> dict[str, Any]:
        candle = {
            "timestamp": _to_datetime_label(row.get("date")),
            "open": _clean_number(row.get("open")),
            "high": _clean_number(row.get("high")),
            "low": _clean_number(row.get("low")),
            "close": _clean_number(row.get("close")),
            "volume": _clean_number(row.get("volume")),
            "macd": _clean_number(row.get("macd")),
            "macd_signal": _clean_number(row.get("macd_signal")),
            "macd_hist": _clean_number(row.get("macd_hist")),
            "rsi": _clean_number(row.get("rsi")),
            "stoch_rsi_k": _clean_number(row.get("stoch_rsi_k")),
            "stoch_rsi_d": _clean_number(row.get("stoch_rsi_d")),
            "ppo": _clean_number(row.get("ppo")),
            "ppo_signal": _clean_number(row.get("ppo_signal")),
            "ppo_hist": _clean_number(row.get("ppo_hist")),
            "cvd_rolling": _clean_number(row.get("cvd_rolling")),
            "taker_buy_base": _clean_number(row.get("taker_buy_base")),
            "funding_rate": _clean_number(row.get("funding_rate")),
        }
        return {key: value for key, value in candle.items() if value is not None}

    extended = rows_df.copy()
    extended.at[latest_index, "candle_data"] = [_market_row_to_candle(row) for _, row in cycle_market.iterrows()]
    extended.at[latest_index, "end_date"] = str(cycle_market["date"].iloc[-1])
    extended.at[latest_index, "duration_candles"] = len(cycle_market)
    return extended


def _lag_hours(source_end_date: str | None, raw_end_date: str | None) -> float | None:
    source_end = _to_datetime(source_end_date)
    raw_end = _to_datetime(raw_end_date)
    if source_end is None or raw_end is None:
        return None
    return round((raw_end - source_end).total_seconds() / 3600, 2)


def _is_stale(timeframe: str, source_lag_hours: float | None) -> bool:
    if source_lag_hours is None:
        return False
    threshold_hours = max(2.0, (TIMEFRAME_SECONDS.get(timeframe, 3600) * 3) / 3600)
    return source_lag_hours > threshold_hours


def _parent_field(hierarchy: dict[str, Any], parent_tf: str, parent_id: str | None, key: str) -> Any:
    if not parent_id:
        return None
    return hierarchy.get(parent_tf, {}).get(parent_id, {}).get(key)


def _cycle_metric_from_row(row: pd.Series) -> dict[str, float | None]:
    candles = _as_candle_list(row.get("candle_data"))
    first = candles[0] if candles else {}
    last = candles[-1] if candles else {}
    features = row.get("cycle_features")
    return {
        "start_ppo": _first_present(_candle_value(first, "ppo"), _clean_number(_feature_value(features, "start", "ppo"))),
        "end_ppo": _first_present(_candle_value(last, "ppo"), _clean_number(_feature_value(features, "end", "ppo"))),
        "start_ppo_hist": _first_present(_candle_value(first, "ppo_hist"), _clean_number(_feature_value(features, "start", "ppo_hist"))),
        "end_ppo_hist": _first_present(_candle_value(last, "ppo_hist"), _clean_number(_feature_value(features, "end", "ppo_hist"))),
    }


def _load_parent_metrics(asset: str, timeframes: tuple[str, ...]) -> dict[str, dict[str, dict[str, float | None]]]:
    metrics: dict[str, dict[str, dict[str, float | None]]] = {}
    for timeframe in timeframes:
        path = _find_cycle_parquet(asset, timeframe)
        if not path:
            metrics[timeframe] = {}
            continue

        cache_key = (asset, timeframe, path.stat().st_mtime)
        if cache_key not in PARENT_METRIC_CACHE:
            df = pd.read_parquet(path, columns=["cycle_id", "candle_data", "cycle_features"])
            row_metrics = {str(row.get("cycle_id")): _cycle_metric_from_row(row) for _, row in df.iterrows()}
            metric_df = pd.DataFrame.from_dict(row_metrics, orient="index")
            for key in ("end_ppo", "end_ppo_hist"):
                if key not in metric_df:
                    continue
                ranks = pd.to_numeric(metric_df[key], errors="coerce").rank(pct=True)
                for cycle_id, rank in ranks.items():
                    row_metrics[cycle_id][f"{key}_rank_score"] = None if pd.isna(rank) else (float(rank) * 2) - 1
            PARENT_METRIC_CACHE[cache_key] = row_metrics
        metrics[timeframe] = PARENT_METRIC_CACHE[cache_key]
    return metrics


def _parent_metric(
    parent_metrics: dict[str, dict[str, dict[str, float | None]]],
    parent_tf: str,
    parent_id: str | None,
    key: str,
) -> float | None:
    if not parent_id:
        return None
    return parent_metrics.get(parent_tf, {}).get(parent_id, {}).get(key)


def _extract_parent_ids(hierarchy: dict[str, Any], timeframe: str, cycle_id: str) -> tuple[str | None, str | None, str | None]:
    def _first_parent(tf: str, cid: str | None, parent_tf: str) -> str | None:
        if not cid:
            return None
        node = hierarchy.get(tf, {}).get(cid, {})
        parent_ids = node.get("parent_cycle_ids", {}) if isinstance(node, dict) else {}
        return (parent_ids.get(parent_tf) or [None])[0]

    parent_4h = _first_parent(timeframe, cycle_id, "4h")
    parent_1d = _first_parent(timeframe, cycle_id, "1d")
    parent_1w = _first_parent(timeframe, cycle_id, "1w")

    if not parent_1d:
        parent_1d = _first_parent("4h", parent_4h, "1d")

    if not parent_1d:
        parent_1h = _first_parent(timeframe, cycle_id, "1h")
        parent_4h = _first_parent("1h", parent_1h, "4h")
        parent_1d = _first_parent("4h", parent_4h, "1d")

    if not parent_1w and parent_1d:
        parent_1w = _first_parent("1d", parent_1d, "1w")

    return parent_4h, parent_1d, parent_1w


def _build_cycle_row(
    row: pd.Series,
    hierarchy: dict[str, Any],
    parent_metrics: dict[str, dict[str, dict[str, float | None]]],
) -> dict[str, Any] | None:
    candles = _as_candle_list(row.get("candle_data"))
    if not candles:
        return None

    first = candles[0]
    last = candles[-1]
    open_price = _clean_number(first.get("open"))
    close_price = _clean_number(last.get("close"))
    highs = [_clean_number(candle.get("high")) for candle in candles]
    lows = [_clean_number(candle.get("low")) for candle in candles]
    highs = [value for value in highs if value is not None]
    lows = [value for value in lows if value is not None]
    if open_price is None or close_price is None or not highs or not lows:
        return None

    start_date = row.get("start_date") or first.get("timestamp")
    end_date = row.get("end_date") or last.get("timestamp")
    parsed_start = _to_datetime(start_date)
    if parsed_start is None:
        return None

    features = row.get("cycle_features")
    cycle_id = str(row.get("cycle_id"))
    parent_4h, parent_1d, parent_1w = _extract_parent_ids(hierarchy, str(row.get("timeframe")), cycle_id)
    change_price_pct = _clean_number(_feature_value(features, "change", "price_pct"))
    if change_price_pct is None and open_price:
        change_price_pct = ((close_price - open_price) / open_price) * 100
    high_price = max(highs)
    low_price = min(lows)
    body_pct = ((close_price - open_price) / open_price) * 100 if open_price else None
    range_pct = ((high_price - low_price) / open_price) * 100 if open_price else None
    upper_wick_pct = ((high_price - max(open_price, close_price)) / open_price) * 100 if open_price else None
    lower_wick_pct = ((min(open_price, close_price) - low_price) / open_price) * 100 if open_price else None
    start_ppo = _first_present(_candle_value(first, "ppo"), _clean_number(_feature_value(features, "start", "ppo")))
    end_ppo = _first_present(_candle_value(last, "ppo"), _clean_number(_feature_value(features, "end", "ppo")))
    start_ppo_hist = _first_present(_candle_value(first, "ppo_hist"), _clean_number(_feature_value(features, "start", "ppo_hist")))
    end_ppo_hist = _first_present(_candle_value(last, "ppo_hist"), _clean_number(_feature_value(features, "end", "ppo_hist")))
    start_rsi = _first_present(_candle_value(first, "rsi"), _clean_number(_feature_value(features, "start", "rsi")))
    end_rsi = _first_present(_candle_value(last, "rsi"), _clean_number(_feature_value(features, "end", "rsi")))
    start_cvd_rolling = _first_present(_candle_value(first, "cvd_rolling"), _clean_number(_feature_value(features, "start", "cvd_rolling")))
    end_cvd_rolling = _first_present(_candle_value(last, "cvd_rolling"), _clean_number(_feature_value(features, "end", "cvd")))

    def _delta(end_value: float | None, start_value: float | None) -> float | None:
        if end_value is None or start_value is None:
            return None
        return end_value - start_value

    return {
        "unix": int(parsed_start.timestamp()),
        "date": parsed_start.isoformat(),
        "start_date": parsed_start.isoformat(),
        "end_date": _to_datetime_label(end_date),
        "cycle_id": cycle_id,
        "cycle_type": _cycle_type_label(row.get("cycle_type")),
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "duration_candles": int(_clean_number(row.get("duration_candles")) or len(candles)),
        "change_price_pct": change_price_pct,
        "cycle_return_pct": change_price_pct,
        "cycle_body_pct": body_pct,
        "cycle_range_pct": range_pct,
        "upper_wick_pct": upper_wick_pct,
        "lower_wick_pct": lower_wick_pct,
        "direction_strength_pct": _clean_number(_feature_value(features, "strength", "direction_pct")),
        "avg_true_range": _clean_number(_feature_value(features, "volatility", "avg_true_range")),
        "start_ppo": start_ppo,
        "end_ppo": end_ppo,
        "start_ppo_signal": _candle_value(first, "ppo_signal"),
        "end_ppo_signal": _candle_value(last, "ppo_signal"),
        "start_ppo_hist": start_ppo_hist,
        "end_ppo_hist": end_ppo_hist,
        "ppo_delta": _delta(end_ppo, start_ppo),
        "ppo_hist_delta": _delta(end_ppo_hist, start_ppo_hist),
        "ppo": _candle_value(last, "ppo"),
        "ppo_signal": _candle_value(last, "ppo_signal"),
        "ppo_hist": _candle_value(last, "ppo_hist"),
        "ppo_rank_score": _parent_metric(parent_metrics, str(row.get("timeframe")), cycle_id, "end_ppo_rank_score"),
        "ppo_hist_rank_score": _parent_metric(parent_metrics, str(row.get("timeframe")), cycle_id, "end_ppo_hist_rank_score"),
        "start_ppo_series": _candle_value(first, "ppo"),
        "area_ppo_hist": _clean_number(_feature_value(features, "aggregate", "area_ppo_hist")),
        "start_rsi": start_rsi,
        "end_rsi": end_rsi,
        "rsi_delta": _delta(end_rsi, start_rsi),
        "start_cvd_rolling": start_cvd_rolling,
        "end_cvd_rolling": end_cvd_rolling,
        "cvd_rolling_delta": _delta(end_cvd_rolling, start_cvd_rolling),
        "cycle_direction_value": 1 if _cycle_type_label(row.get("cycle_type")) == "up" else -1,
        "parent_4h_cycle_id": parent_4h,
        "parent_4h_cycle_type": _cycle_type_label(_parent_field(hierarchy, "4h", parent_4h, "cycle_type")),
        "parent_4h_start_ppo": _parent_metric(parent_metrics, "4h", parent_4h, "start_ppo"),
        "parent_4h_end_ppo": _parent_metric(parent_metrics, "4h", parent_4h, "end_ppo"),
        "parent_4h_start_ppo_hist": _parent_metric(parent_metrics, "4h", parent_4h, "start_ppo_hist"),
        "parent_4h_end_ppo_hist": _parent_metric(parent_metrics, "4h", parent_4h, "end_ppo_hist"),
        "parent_4h_ppo_rank_score": _parent_metric(parent_metrics, "4h", parent_4h, "end_ppo_rank_score"),
        "parent_4h_ppo_hist_rank_score": _parent_metric(parent_metrics, "4h", parent_4h, "end_ppo_hist_rank_score"),
        "parent_1d_cycle_id": parent_1d,
        "parent_1d_cycle_type": _cycle_type_label(_parent_field(hierarchy, "1d", parent_1d, "cycle_type")),
        "parent_1d_start_ppo": _parent_metric(parent_metrics, "1d", parent_1d, "start_ppo"),
        "parent_1d_end_ppo": _parent_metric(parent_metrics, "1d", parent_1d, "end_ppo"),
        "parent_1d_start_ppo_hist": _parent_metric(parent_metrics, "1d", parent_1d, "start_ppo_hist"),
        "parent_1d_end_ppo_hist": _parent_metric(parent_metrics, "1d", parent_1d, "end_ppo_hist"),
        "parent_1d_ppo_rank_score": _parent_metric(parent_metrics, "1d", parent_1d, "end_ppo_rank_score"),
        "parent_1d_ppo_hist_rank_score": _parent_metric(parent_metrics, "1d", parent_1d, "end_ppo_hist_rank_score"),
        "parent_1w_cycle_id": parent_1w,
        "parent_1w_cycle_type": _cycle_type_label(_parent_field(hierarchy, "1w", parent_1w, "cycle_type")),
        "parent_1w_start_ppo": _parent_metric(parent_metrics, "1w", parent_1w, "start_ppo"),
        "parent_1w_end_ppo": _parent_metric(parent_metrics, "1w", parent_1w, "end_ppo"),
        "parent_1w_start_ppo_hist": _parent_metric(parent_metrics, "1w", parent_1w, "start_ppo_hist"),
        "parent_1w_end_ppo_hist": _parent_metric(parent_metrics, "1w", parent_1w, "end_ppo_hist"),
        "parent_1w_ppo_rank_score": _parent_metric(parent_metrics, "1w", parent_1w, "end_ppo_rank_score"),
        "parent_1w_ppo_hist_rank_score": _parent_metric(parent_metrics, "1w", parent_1w, "end_ppo_hist_rank_score"),
    }


@router.get("/files")
def list_cycle_candle_files() -> dict[str, Any]:
    assets = _available_assets()
    files = []
    for asset in assets:
        for timeframe in _available_timeframes(asset):
            files.append({"asset": asset, "symbol": asset.upper(), "timeframe": timeframe})
    return {"files": files, "defaultAsset": DEFAULT_ASSET}


@router.get("/series")
def get_cycle_candle_series(
    asset: str = Query(DEFAULT_ASSET),
    timeframe: str = Query("4h"),
    limit: int = Query(0, ge=0),
) -> dict[str, Any]:
    normalized_asset = _normalize_asset(asset)
    normalized_timeframe = _normalize_timeframe(timeframe)
    if normalized_timeframe not in SUPPORTED_TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {timeframe}")

    source_path = _find_cycle_parquet(normalized_asset, normalized_timeframe)
    if not source_path:
        raise HTTPException(status_code=404, detail=f"Cycle parquet not found for {normalized_asset} {normalized_timeframe}")

    raw_end_date = _market_end_date(normalized_asset, normalized_timeframe)
    cache_key = (
        normalized_asset,
        normalized_timeframe,
        limit,
        source_path.stat().st_mtime,
        (PROJECT_PATHS.base_data_dir / MARKET_FILES.get(normalized_asset, {}).get(normalized_timeframe, "")).stat().st_mtime
        if MARKET_FILES.get(normalized_asset, {}).get(normalized_timeframe)
        and (PROJECT_PATHS.base_data_dir / MARKET_FILES[normalized_asset][normalized_timeframe]).exists()
        else None,
    )
    cached = SERIES_CACHE.get(cache_key)
    if cached is not None:
        SERIES_CACHE.move_to_end(cache_key)
        return cached

    source_df = pd.read_parquet(source_path).sort_values("start_date").reset_index(drop=True)
    if "candle_data" not in source_df.columns:
        raise HTTPException(status_code=422, detail=f"{source_path.name} does not include candle_data")

    total_rows = len(source_df)
    rows_df = source_df.tail(limit).copy() if limit > 0 else source_df.copy()
    rows_df = _extend_latest_cycle_with_market(rows_df, normalized_asset, normalized_timeframe)
    hierarchy = _find_hierarchy(normalized_asset)
    parent_metrics = _load_parent_metrics(normalized_asset, tuple(dict.fromkeys((normalized_timeframe, "4h", "1d", "1w"))))
    rows = [_build_cycle_row(row, hierarchy, parent_metrics) for _, row in rows_df.iterrows()]
    rows = [row for row in rows if row is not None]
    source_end_date = rows[-1]["end_date"] if rows else None
    source_lag_hours = _lag_hours(source_end_date, raw_end_date)

    payload = {
        "meta": {
            "asset": normalized_asset,
            "symbol": normalized_asset.upper(),
            "timeframe": normalized_timeframe,
            "sourcePath": str(source_path),
            "rowCount": len(rows),
            "totalRowCount": total_rows,
            "startDate": rows[0]["start_date"] if rows else None,
            "endDate": source_end_date,
            "rawEndDate": raw_end_date,
            "sourceLagHours": source_lag_hours,
            "isStale": _is_stale(normalized_timeframe, source_lag_hours),
        },
        "rows": rows,
    }
    SERIES_CACHE[cache_key] = payload
    SERIES_CACHE.move_to_end(cache_key)
    while len(SERIES_CACHE) > SERIES_CACHE_MAX:
        SERIES_CACHE.popitem(last=False)
    return payload
