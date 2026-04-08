from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DATA_DIR = PROJECT_ROOT / "data" / "base_data"

router = APIRouter()


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


def _resolve_csv_path(file_name: str) -> Path:
    resolved = (BASE_DATA_DIR / file_name).resolve()
    if resolved.parent != BASE_DATA_DIR.resolve():
        raise HTTPException(status_code=400, detail="The requested file path is not allowed.")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {file_name}")
    return resolved


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
    lines = [line for line in csv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise HTTPException(status_code=404, detail="The CSV file is empty.")

    headers = [item.strip() for item in lines[0].split(",")]
    data_lines = lines[1:]
    sliced_lines = data_lines[-limit:] if limit > 0 else data_lines

    rows: list[dict[str, Any]] = []
    for line in sliced_lines:
        values = line.split(",")
        row = {header: values[index] if index < len(values) else "" for index, header in enumerate(headers)}
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
                "oi": _to_number(row.get("oi")),
                "oi_contracts": _to_number(row.get("oi_contracts")),
                "oi_contracts_change": _to_number(row.get("oi_contracts_change")),
                "oi_contracts_change_pct": _to_number(row.get("oi_contracts_change_pct")),
                "oi_usd": _to_number(row.get("oi_usd")),
                "oi_notional": _to_number(row.get("oi_notional")),
                "oi_notional_change": _to_number(row.get("oi_notional_change")),
                "oi_notional_change_pct": _to_number(row.get("oi_notional_change_pct")),
                "oi_change": _to_number(row.get("oi_change")),
                "oi_change_pct": _to_number(row.get("oi_change_pct")),
                "funding_rate": _to_number(row.get("funding_rate")),
            }
        )

    return {
        "meta": {
            "fileName": file,
            "symbol": rows[0]["symbol"] if rows else "",
            "timeframe": file.removesuffix(".csv").split("_")[-1],
            "rowCount": len(rows),
            "totalRowCount": len(data_lines),
            "startDate": rows[0]["date"] if rows else "",
            "endDate": rows[-1]["date"] if rows else "",
            "columns": headers,
        },
        "rows": rows,
    }
