from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

from data_pipeline.microstructure.paths import (
    legacy_public_archive_dir,
    public_archive_dir,
    raw_stream_dir,
)
from data_pipeline.storage.manifests import write_ingestion_manifest


LOGGER = logging.getLogger(__name__)

S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PUBLIC_DATA_URL = "https://data.binance.vision"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass(frozen=True)
class PublicDataset:
    name: str
    periods: tuple[str, ...]
    interval_required: bool = False


DATASETS: dict[str, PublicDataset] = {
    "aggTrades": PublicDataset("aggTrades", ("monthly", "daily")),
    "metrics": PublicDataset("metrics", ("daily",)),
    "bookTicker": PublicDataset("bookTicker", ("monthly", "daily")),
    "bookDepth": PublicDataset("bookDepth", ("daily",)),
    "markPriceKlines": PublicDataset("markPriceKlines", ("monthly", "daily"), interval_required=True),
    "premiumIndexKlines": PublicDataset("premiumIndexKlines", ("monthly", "daily"), interval_required=True),
    "klines": PublicDataset("klines", ("monthly", "daily"), interval_required=True),
    "fundingRate": PublicDataset("fundingRate", ("monthly",)),
}

DEFAULT_DATASETS = tuple(DATASETS)
DEFAULT_PERIOD = {
    "aggTrades": "daily",
    "metrics": "daily",
    "bookTicker": "monthly",
    "bookDepth": "daily",
    "markPriceKlines": "monthly",
    "premiumIndexKlines": "monthly",
    "klines": "monthly",
    "fundingRate": "monthly",
}

AGG_TRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "trade_time",
    "is_buyer_maker",
]
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
BOOK_TICKER_COLUMNS = [
    "update_id",
    "symbol",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "transaction_time",
    "event_time",
]
BOOK_DEPTH_COLUMNS = ["timestamp", "percentage", "depth", "notional"]
FUNDING_RATE_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
CSV_CHUNKSIZE = 250_000


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    last_modified: str


@dataclass(frozen=True)
class BackfillResult:
    dataset: str
    zip_key: str
    archive_path: Path
    parquet_paths: tuple[Path, ...]
    rows: int


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _next_month(value: date) -> date:
    return date(value.year + (1 if value.month == 12 else 0), 1 if value.month == 12 else value.month + 1, 1)


def _iter_months(start: date, end: date) -> Iterable[str]:
    cursor = _month_start(start)
    stop = _month_start(end)
    while cursor <= stop:
        yield cursor.strftime("%Y-%m")
        cursor = _next_month(cursor)


def _iter_days(start: date, end: date) -> Iterable[str]:
    for day in pd.date_range(start=start, end=end, freq="D"):
        yield day.strftime("%Y-%m-%d")


def _object_date_from_key(key: str, period: str) -> date | None:
    pattern = r"(\d{4}-\d{2})\.zip$" if period == "monthly" else r"(\d{4}-\d{2}-\d{2})\.zip$"
    match = re.search(pattern, key)
    if not match:
        return None
    value = match.group(1)
    return datetime.strptime(value, "%Y-%m" if period == "monthly" else "%Y-%m-%d").date()


def _key_in_range(key: str, period: str, start: date, end: date) -> bool:
    key_date = _object_date_from_key(key, period)
    if key_date is None:
        return False
    if period == "monthly":
        return _month_start(start) <= key_date <= _month_start(end)
    return start <= key_date <= end


def dataset_prefix(dataset: str, symbol: str, period: str, interval: str) -> str:
    symbol = symbol.upper()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if period not in DATASETS[dataset].periods:
        raise ValueError(f"{dataset} does not support period={period}")
    base = f"data/futures/um/{period}/{dataset}/{symbol}"
    if DATASETS[dataset].interval_required:
        return f"{base}/{interval}/"
    return f"{base}/"


def list_s3_objects(prefix: str) -> list[S3Object]:
    objects: list[S3Object] = []
    marker: str | None = None
    while True:
        params = {"delimiter": "/", "prefix": prefix}
        if marker:
            params["marker"] = marker
        response = requests.get(f"{S3_LIST_URL}?{urlencode(params)}", timeout=30)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        for item in root.findall("s3:Contents", S3_NS):
            key = item.findtext("s3:Key", default="", namespaces=S3_NS)
            if key:
                objects.append(
                    S3Object(
                        key=key,
                        size=int(item.findtext("s3:Size", default="0", namespaces=S3_NS)),
                        last_modified=item.findtext("s3:LastModified", default="", namespaces=S3_NS),
                    )
                )
        if root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS) != "true":
            break
        marker = root.findtext("s3:NextMarker", namespaces=S3_NS)
        if not marker and objects:
            marker = objects[-1].key
        if not marker:
            break
    return objects


def select_zip_objects(dataset: str, symbol: str, period: str, interval: str, start: date, end: date) -> list[S3Object]:
    prefix = dataset_prefix(dataset, symbol, period, interval)
    objects = list_s3_objects(prefix)
    zips = [
        obj
        for obj in objects
        if obj.key.endswith(".zip")
        and not obj.key.endswith(".zip.CHECKSUM")
        and _key_in_range(obj.key, period, start, end)
    ]
    return sorted(zips, key=lambda obj: obj.key)


def _download_file(key: str, out_path: Path, overwrite: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path
    url = f"{PUBLIC_DATA_URL}/{key}"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp_path.replace(out_path)
    return out_path


def _read_checksum_text(key: str) -> str | None:
    response = requests.get(f"{PUBLIC_DATA_URL}/{key}.CHECKSUM", timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text.strip()


def _verify_checksum(zip_path: Path, checksum_text: str | None) -> bool:
    if not checksum_text:
        return True
    expected = checksum_text.split()[0].strip()
    if not expected:
        return True
    hasher = hashlib.sha256()
    with zip_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(f"Checksum mismatch for {zip_path}: expected={expected} actual={actual}")
    return True


def archive_path_for_key(key: str) -> Path:
    relative_path = Path(key).relative_to("data/futures/um")
    canonical = public_archive_dir() / relative_path
    legacy = legacy_public_archive_dir() / relative_path
    # Reuse an already-downloaded archive during the transition. New downloads
    # are stored in archive/binance/futures/um.
    return legacy if legacy.exists() and not canonical.exists() else canonical


def _expected_columns(dataset: str) -> list[str] | None:
    if dataset == "aggTrades":
        return AGG_TRADE_COLUMNS
    if dataset in {"klines", "markPriceKlines", "premiumIndexKlines"}:
        return KLINE_COLUMNS
    if dataset == "bookTicker":
        return BOOK_TICKER_COLUMNS
    if dataset == "bookDepth":
        return BOOK_DEPTH_COLUMNS
    if dataset == "fundingRate":
        return FUNDING_RATE_COLUMNS
    return None


def _csv_read_options(first_line: str, columns: list[str] | None) -> dict[str, Any]:
    if columns and not any(ch.isalpha() for ch in first_line.split(",")[0]):
        return {"header": None, "names": columns}
    return {"header": 0}


def _read_csv_from_zip(zip_path: Path, dataset: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()
        with zf.open(csv_names[0]) as fh:
            first = fh.readline().decode("utf-8", errors="replace").strip()
            fh.seek(0)
            return pd.read_csv(fh, **_csv_read_options(first, _expected_columns(dataset)))


def _iter_csv_chunks_from_zip(zip_path: Path, dataset: str, chunksize: int) -> Iterable[pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            return
        with zf.open(csv_names[0]) as fh:
            first = fh.readline().decode("utf-8", errors="replace").strip()
            fh.seek(0)
            options = _csv_read_options(first, _expected_columns(dataset))
            yield from pd.read_csv(fh, chunksize=chunksize, **options)


def _to_ns(series: pd.Series, unit: str = "ms") -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    factor = 1_000_000 if unit == "ms" else 1
    return (values * factor).round().astype("Int64")


def _datetime_to_ns(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True).astype("datetime64[ns, UTC]")
    return pd.Series(dt.astype("int64"), index=series.index).where(dt.notna(), pd.NA).astype("Int64")


def _date_from_event_ns(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["event_time_ns"], unit="ns", utc=True).dt.strftime("%Y-%m-%d")


def _write_partitioned(frame: pd.DataFrame, symbol: str, stream: str, stem: str, chunk_index: int | None = None) -> tuple[Path, ...]:
    if frame.empty:
        return ()
    out_paths: list[Path] = []
    out = frame.copy()
    out["event_date"] = _date_from_event_ns(out)
    for event_date, group in out.groupby("event_date"):
        group = group.drop(columns=["event_date"])
        out_dir = raw_stream_dir(symbol, stream) / f"date={event_date}"
        out_dir.mkdir(parents=True, exist_ok=True)
        chunk_suffix = "" if chunk_index is None else f"-chunk{chunk_index:05d}"
        out_path = out_dir / f"public-{stem}{chunk_suffix}-{event_date}.parquet"
        group.to_parquet(out_path, index=False)
        write_ingestion_manifest(
            provider="binance",
            market="usdm",
            symbol=symbol,
            dataset=stream,
            data_path=out_path,
            rows=group,
            source="data.binance.vision",
            extra={"archive_stem": stem},
        )
        out_paths.append(out_path)
    return tuple(out_paths)


def normalize_agg_trades(df: pd.DataFrame, symbol: str) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    work = df.copy()
    if "agg_trade_id" not in work.columns:
        work.columns = AGG_TRADE_COLUMNS[: len(work.columns)]
    for col in ["agg_trade_id", "first_trade_id", "last_trade_id"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("Int64")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce")
    work["is_buyer_maker"] = work["is_buyer_maker"].astype(str).str.lower().isin({"true", "1"})
    time_col = "trade_time" if "trade_time" in work.columns else "transact_time"
    work["event_time_ns"] = _to_ns(work[time_col], "ms")
    work["trade_time_ns"] = work["event_time_ns"]
    work["signed_quantity"] = np.where(work["is_buyer_maker"], -work["quantity"], work["quantity"])
    work["symbol"] = symbol.upper()
    work["source_endpoint"] = "binance_public_data"
    work["symbol_type"] = 1
    cols = [
        "event_time_ns",
        "trade_time_ns",
        "symbol",
        "agg_trade_id",
        "price",
        "quantity",
        "signed_quantity",
        "is_buyer_maker",
        "first_trade_id",
        "last_trade_id",
        "source_endpoint",
        "symbol_type",
    ]
    return {"agg_trade": work[cols].dropna(subset=["event_time_ns", "quantity", "price"])}


def normalize_metrics(df: pd.DataFrame, symbol: str) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    work = df.copy()
    work["event_time_ns"] = _datetime_to_ns(work["create_time"])
    work["symbol"] = symbol.upper()
    numeric_cols = [
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    common = ["event_time_ns", "symbol"]
    out: dict[str, pd.DataFrame] = {
        "metrics": work[[*common, *[col for col in numeric_cols if col in work.columns]]].dropna(subset=["event_time_ns"]),
        "open_interest_snapshot": pd.DataFrame(
            {
                "event_time_ns": work["event_time_ns"],
                "symbol": work["symbol"],
                "open_interest": work.get("sum_open_interest"),
                "open_interest_value": work.get("sum_open_interest_value"),
                "source_endpoint": "binance_public_data_metrics",
            }
        ),
        "global_ls_account_ratio": pd.DataFrame(
            {
                "event_time_ns": work["event_time_ns"],
                "symbol": work["symbol"],
                "period": "5m",
                "long_short_ratio": work.get("count_long_short_ratio"),
                "source_endpoint": "binance_public_data_metrics",
            }
        ),
        "top_ls_account_ratio": pd.DataFrame(
            {
                "event_time_ns": work["event_time_ns"],
                "symbol": work["symbol"],
                "period": "5m",
                "long_short_ratio": work.get("count_toptrader_long_short_ratio"),
                "source_endpoint": "binance_public_data_metrics",
            }
        ),
        "top_ls_position_ratio": pd.DataFrame(
            {
                "event_time_ns": work["event_time_ns"],
                "symbol": work["symbol"],
                "period": "5m",
                "long_short_ratio": work.get("sum_toptrader_long_short_ratio"),
                "source_endpoint": "binance_public_data_metrics",
            }
        ),
        "taker_ls_volume_ratio": pd.DataFrame(
            {
                "event_time_ns": work["event_time_ns"],
                "symbol": work["symbol"],
                "period": "5m",
                "long_short_ratio": work.get("sum_taker_long_short_vol_ratio"),
                "source_endpoint": "binance_public_data_metrics",
            }
        ),
    }
    return {name: frame.dropna(subset=["event_time_ns"]) for name, frame in out.items()}


def normalize_book_depth(df: pd.DataFrame, symbol: str) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    raw = df.copy()
    raw["event_time_ns"] = _datetime_to_ns(raw["timestamp"])
    raw["symbol"] = symbol.upper()
    raw["percentage"] = pd.to_numeric(raw["percentage"], errors="coerce")
    raw["depth"] = pd.to_numeric(raw["depth"], errors="coerce")
    raw["notional"] = pd.to_numeric(raw["notional"], errors="coerce")
    raw["source_endpoint"] = "binance_public_data_bookDepth"
    raw_out = raw[["event_time_ns", "symbol", "percentage", "depth", "notional", "source_endpoint"]].dropna(
        subset=["event_time_ns", "percentage"]
    )

    # bookDepth percentage bands are cumulative to the given +/- distance.
    # Keep the raw percent-band stream only; feature builders must select one
    # band or explicitly difference adjacent bands.
    return {"book_depth_percent": raw_out}


def normalize_book_ticker(df: pd.DataFrame, symbol: str) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    work = df.copy()
    if "update_id" not in work.columns:
        work.columns = BOOK_TICKER_COLUMNS[: len(work.columns)]
    work["symbol"] = work.get("symbol", symbol).astype(str).str.upper() if "symbol" in work.columns else symbol.upper()
    time_col = "event_time" if "event_time" in work.columns else "transaction_time"
    work["event_time_ns"] = _to_ns(work[time_col], "ms")
    for col in ["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["source_endpoint"] = "binance_public_data_bookTicker"
    cols = [
        "event_time_ns",
        "symbol",
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "source_endpoint",
    ]
    return {"book_ticker": work[[col for col in cols if col in work.columns]].dropna(subset=["event_time_ns"])}


def normalize_klines(df: pd.DataFrame, symbol: str, dataset: str) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    work = df.copy()
    if "open_time" not in work.columns:
        work.columns = KLINE_COLUMNS[: len(work.columns)]
    for col in KLINE_COLUMNS:
        if col in work.columns and col not in {"open_time", "close_time"}:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["event_time_ns"] = _to_ns(work["open_time"], "ms")
    work["close_time_ns"] = _to_ns(work["close_time"], "ms")
    work["symbol"] = symbol.upper()
    work["source_endpoint"] = f"binance_public_data_{dataset}"
    base_cols = [
        "event_time_ns",
        "close_time_ns",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "source_endpoint",
    ]
    if dataset == "markPriceKlines":
        return {
            "mark_price": pd.DataFrame(
                {
                    "event_time_ns": work["event_time_ns"],
                    "symbol": work["symbol"],
                    "mark_price": work["close"],
                    "source_endpoint": work["source_endpoint"],
                }
            ).dropna(subset=["event_time_ns", "mark_price"])
        }
    stream = "premium_index" if dataset == "premiumIndexKlines" else "kline"
    return {stream: work[[col for col in base_cols if col in work.columns]].dropna(subset=["event_time_ns"])}


def normalize_funding_rate(df: pd.DataFrame, symbol: str) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    work = df.copy()
    work["event_time_ns"] = _to_ns(work["calc_time"], "ms")
    work["symbol"] = symbol.upper()
    work["funding_interval_hours"] = pd.to_numeric(work["funding_interval_hours"], errors="coerce")
    work["funding_rate_settled"] = pd.to_numeric(work["last_funding_rate"], errors="coerce")
    work["source_endpoint"] = "binance_public_data_fundingRate"
    return {
        "funding_rate_settled": work[
            ["event_time_ns", "symbol", "funding_interval_hours", "funding_rate_settled", "source_endpoint"]
        ].dropna(subset=["event_time_ns", "funding_rate_settled"])
    }


def normalize_dataset(df: pd.DataFrame, dataset: str, symbol: str) -> dict[str, pd.DataFrame]:
    if dataset == "aggTrades":
        return normalize_agg_trades(df, symbol)
    if dataset == "metrics":
        return normalize_metrics(df, symbol)
    if dataset == "bookDepth":
        return normalize_book_depth(df, symbol)
    if dataset == "bookTicker":
        return normalize_book_ticker(df, symbol)
    if dataset in {"klines", "markPriceKlines", "premiumIndexKlines"}:
        return normalize_klines(df, symbol, dataset)
    if dataset == "fundingRate":
        return normalize_funding_rate(df, symbol)
    raise ValueError(f"No normalizer for {dataset}")


def convert_archive(zip_path: Path, dataset: str, symbol: str) -> tuple[Path, ...]:
    stem = zip_path.stem
    out_paths: list[Path] = []
    if dataset in {"aggTrades", "bookTicker"}:
        for chunk_index, df in enumerate(_iter_csv_chunks_from_zip(zip_path, dataset, CSV_CHUNKSIZE)):
            for stream, frame in normalize_dataset(df, dataset, symbol).items():
                out_paths.extend(_write_partitioned(frame, symbol, stream, stem, chunk_index=chunk_index))
        return tuple(out_paths)

    df = _read_csv_from_zip(zip_path, dataset)
    for stream, frame in normalize_dataset(df, dataset, symbol).items():
        out_paths.extend(_write_partitioned(frame, symbol, stream, stem))
    return tuple(out_paths)


def download_and_convert_object(obj: S3Object, dataset: str, symbol: str, overwrite: bool = False) -> BackfillResult:
    archive_path = archive_path_for_key(obj.key)
    LOGGER.info("Downloading %s", obj.key)
    _download_file(obj.key, archive_path, overwrite=overwrite)
    _verify_checksum(archive_path, _read_checksum_text(obj.key))
    parquet_paths = convert_archive(archive_path, dataset, symbol)
    rows = 0
    for path in parquet_paths:
        try:
            rows += len(pd.read_parquet(path, columns=[]))
        except Exception:
            pass
    return BackfillResult(dataset, obj.key, archive_path, parquet_paths, rows)


def backfill_public_data(
    symbol: str,
    datasets: Iterable[str],
    start: date,
    end: date,
    interval: str = "1m",
    period: str = "auto",
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[BackfillResult]:
    results: list[BackfillResult] = []
    for dataset in datasets:
        selected_period = DEFAULT_PERIOD[dataset] if period == "auto" else period
        objects = select_zip_objects(dataset, symbol, selected_period, interval, start, end)
        LOGGER.info(
            "Selected %s %s objects for %s %s %s..%s",
            len(objects),
            selected_period,
            symbol.upper(),
            dataset,
            start,
            end,
        )
        if dry_run:
            for obj in objects:
                LOGGER.info("Dry run: %s (%s bytes)", obj.key, obj.size)
            continue
        for obj in objects:
            results.append(download_and_convert_object(obj, dataset, symbol, overwrite=overwrite))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Binance USD-M public data from data.binance.vision.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--datasets", nargs="+", default=["all"], help=f"Datasets or all. Options: {', '.join(DATASETS)}")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default="1m", help="Kline interval for klines/markPriceKlines/premiumIndexKlines.")
    parser.add_argument("--period", choices=["auto", "daily", "monthly"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    datasets = DEFAULT_DATASETS if args.datasets == ["all"] else tuple(args.datasets)
    unknown = sorted(set(datasets).difference(DATASETS))
    if unknown:
        LOGGER.error("Unknown datasets: %s", ", ".join(unknown))
        return 2
    try:
        results = backfill_public_data(
            symbol=args.symbol,
            datasets=datasets,
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            interval=args.interval,
            period=args.period,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        LOGGER.error("Backfill failed: %s", exc)
        return 1
    for result in results:
        LOGGER.info(
            "Converted %s -> %s parquet files",
            result.archive_path,
            len(result.parquet_paths),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
