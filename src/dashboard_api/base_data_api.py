import csv
from pathlib import Path
from typing import Any
from collections import OrderedDict, deque
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, HTTPException, Query

from src.common.paths import PROJECT_PATHS

BASE_DATA_DIR = PROJECT_PATHS.base_data_dir

router = APIRouter()

SERIES_CACHE: "OrderedDict[tuple[str, float, int], dict[str, Any]]" = OrderedDict()
SERIES_CACHE_MAX = 32
TIMEFRAME_SECONDS = {
    "1min": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
}

# These files are maintained independently from the candle CSVs.  Merge them
# at read time so an OI value is not lost merely because the candle enrichment
# job has not yet copied it into BTCUSD_<timeframe>.csv.
OI_SIDECAR_FIELDS = {
    "oi": ("oi", "oi_usd", "oi_change", "oi_change_pct"),
    "oi_contracts": ("oi_contracts", "oi_contracts_change", "oi_contracts_change_pct"),
    "oi_notional": ("oi_notional", "oi_notional_change", "oi_notional_change_pct"),
}


def _to_number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_datetime_local(value: str | None) -> str | None:
    if not value:
        return None
    return str(value).replace(" ", "T")[:16]


def _parse_datetime_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("T", " "))
    return parsed.replace(tzinfo=timezone.utc)


def _resolve_csv_path(file_name: str) -> Path:
    resolved = (BASE_DATA_DIR / file_name).resolve()
    if resolved.parent != BASE_DATA_DIR.resolve():
        raise HTTPException(status_code=400, detail="The requested file path is not allowed.")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {file_name}")
    return resolved


def _read_csv_tail(csv_path: Path, limit: int) -> tuple[list[str], list[str], int]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline().strip()
        if not header_line:
            return [], [], 0
        if limit > 0:
            data_buffer: deque[str] = deque(maxlen=limit + 4)
            total_count = 0
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                data_buffer.append(line)
                total_count += 1
            return [item.strip() for item in header_line.split(",")], list(data_buffer), total_count

        lines = [line.strip() for line in handle if line.strip()]
        return [item.strip() for item in header_line.split(",")], lines, len(lines)


def _oi_sidecar_paths(timeframe: str) -> list[Path]:
    return [BASE_DATA_DIR / f"BTCUSDT_{suffix}_{timeframe}.csv" for suffix in OI_SIDECAR_FIELDS]


def _load_oi_sidecars(timeframe: str) -> dict[int, dict[str, float | None]]:
    """Return OI values keyed by candle unix time from the optional sidecar CSVs."""
    values_by_unix: dict[int, dict[str, float | None]] = {}
    for suffix, fields in OI_SIDECAR_FIELDS.items():
        path = BASE_DATA_DIR / f"BTCUSDT_{suffix}_{timeframe}.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                unix = _to_number(row.get("unix"))
                if unix is None:
                    continue
                target = values_by_unix.setdefault(int(unix), {})
                for field in fields:
                    value = _to_number(row.get(field))
                    if value is not None:
                        target[field] = value
    return values_by_unix


@router.get("/api/base-data/files")
def list_base_data_files() -> dict[str, Any]:
    if not BASE_DATA_DIR.exists():
        raise HTTPException(status_code=404, detail="Base data directory not found.")

    files = sorted(
        (
            {
                "name": path.name,
                "timeframe": path.stem.split("_")[-1],
            }
            for path in BASE_DATA_DIR.glob("*.csv")
            if path.is_file()
        ),
        key=lambda item: item["name"],
    )
    default_file = next((item["name"] for item in files if item["name"] == "BTCUSD_1h.csv"), files[0]["name"] if files else "")
    return {"files": files, "defaultFile": default_file}


@router.get("/api/base-data/series")
def get_base_data_series(file: str = Query(...), limit: int = Query(0, ge=0)) -> dict[str, Any]:
    csv_path = _resolve_csv_path(file)
    stat = csv_path.stat()
    timeframe = file.removesuffix(".csv").split("_")[-1]
    sidecar_mtimes = tuple(path.stat().st_mtime if path.exists() else None for path in _oi_sidecar_paths(timeframe))
    cache_key = (csv_path.name, stat.st_mtime, sidecar_mtimes, limit)
    cached = SERIES_CACHE.get(cache_key)
    if cached is not None:
        SERIES_CACHE.move_to_end(cache_key)
        return cached

    headers, sliced_lines, total_count = _read_csv_tail(csv_path, limit)
    if not headers or total_count < 1:
        raise HTTPException(status_code=404, detail="The CSV file is empty.")

    timeframe_seconds = TIMEFRAME_SECONDS.get(timeframe)
    oi_sidecars = _load_oi_sidecars(timeframe)
    now = datetime.now(timezone.utc)
    dropped_partial = 0

    rows: list[dict[str, Any]] = []
    for line in sliced_lines:
        values = line.split(",")
        row = {header: values[index] if index < len(values) else "" for index, header in enumerate(headers)}
        row_date = _parse_datetime_utc(row.get("date"))
        if timeframe_seconds and row_date and row_date + timedelta(seconds=timeframe_seconds) > now:
            dropped_partial += 1
            continue
        sidecar_values = oi_sidecars.get(int(_to_number(row.get("unix")) or -1), {})
        rows.append(
            {
                "unix": _to_number(row.get("unix")),
                "date": _to_datetime_local(row.get("date")),
                "open": _to_number(row.get("open")),
                "high": _to_number(row.get("high")),
                "low": _to_number(row.get("low")),
                "close": _to_number(row.get("close")),
                "volumeUsd": _to_number(row.get("Volume USD")),
                "symbol": row.get("symbol") or file.split("_")[0],
                "macd": _to_number(row.get("macd")),
                "macd_signal": _to_number(row.get("macd_signal")),
                "macd_hist": _to_number(row.get("macd_hist")),
                "rsi": _to_number(row.get("rsi")),
                "stoch_rsi_k": _to_number(row.get("stoch_rsi_k")),
                "stoch_rsi_d": _to_number(row.get("stoch_rsi_d")),
                "volume": _to_number(row.get("volume")),
                "taker_buy_base": _to_number(row.get("taker_buy_base")),
                "volume_delta": _to_number(row.get("volume_delta")),
                "cvd": _to_number(row.get("cvd")),
                "cvd_rolling": _to_number(row.get("cvd_rolling")),
                "ppo": _to_number(row.get("ppo")),
                "ppo_signal": _to_number(row.get("ppo_signal")),
                "ppo_hist": _to_number(row.get("ppo_hist")),
                "delta": _to_number(row.get("delta")),
                "ma_7": _to_number(row.get("ma_7")),
                "ma_25": _to_number(row.get("ma_25")),
                "ma_99": _to_number(row.get("ma_99")),
                "oi": sidecar_values.get("oi", _to_number(row.get("oi"))),
                "oi_contracts": sidecar_values.get("oi_contracts", _to_number(row.get("oi_contracts"))),
                "oi_contracts_change": sidecar_values.get("oi_contracts_change", _to_number(row.get("oi_contracts_change"))),
                "oi_contracts_change_pct": sidecar_values.get("oi_contracts_change_pct", _to_number(row.get("oi_contracts_change_pct"))),
                "oi_usd": sidecar_values.get("oi_usd", _to_number(row.get("oi_usd"))),
                "oi_notional": sidecar_values.get("oi_notional", _to_number(row.get("oi_notional"))),
                "oi_notional_change": sidecar_values.get("oi_notional_change", _to_number(row.get("oi_notional_change"))),
                "oi_notional_change_pct": sidecar_values.get("oi_notional_change_pct", _to_number(row.get("oi_notional_change_pct"))),
                "oi_change": sidecar_values.get("oi_change", _to_number(row.get("oi_change"))),
                "oi_change_pct": sidecar_values.get("oi_change_pct", _to_number(row.get("oi_change_pct"))),
                "funding_rate": _to_number(row.get("funding_rate")),
            }
        )

    if limit > 0 and len(rows) > limit:
        rows = rows[-limit:]

    payload = {
        "meta": {
            "fileName": file,
            "symbol": rows[0]["symbol"] if rows else "",
            "timeframe": timeframe,
            "rowCount": len(rows),
            "totalRowCount": total_count,
            "startDate": rows[0]["date"] if rows else "",
            "endDate": rows[-1]["date"] if rows else "",
            "columns": headers,
            "droppedPartialRows": dropped_partial,
        },
        "rows": rows,
    }
    SERIES_CACHE[cache_key] = payload
    SERIES_CACHE.move_to_end(cache_key)
    while len(SERIES_CACHE) > SERIES_CACHE_MAX:
        SERIES_CACHE.popitem(last=False)
    return payload
