from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
import websockets

from data_pipeline.microstructure.io import utc_now_ns, write_partitioned_parquet
from data_pipeline.microstructure.paths import raw_stream_dir
from data_pipeline.storage.manifests import write_ingestion_manifest


LOGGER = logging.getLogger(__name__)

FSTREAM_MARKET_URL = "wss://fstream.binance.com/market/stream?streams={streams}"
FSTREAM_PUBLIC_URL = "wss://fstream.binance.com/public/stream?streams={streams}"
FAPI_BASE_URL = "https://fapi.binance.com"


def _ms_to_ns(value: Any) -> int | None:
    try:
        return int(value) * 1_000_000
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _level_rows(levels: list[list[Any]], prefix: str, max_levels: int) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    for index in range(max_levels):
        price = qty = None
        if index < len(levels):
            price = _to_float(levels[index][0])
            qty = _to_float(levels[index][1])
        row[f"{prefix}_px_{index + 1}"] = price
        row[f"{prefix}_qty_{index + 1}"] = qty
    return row


def normalize_ws_payload(payload: dict[str, Any], max_depth_levels: int = 20) -> tuple[str, dict[str, Any]] | None:
    stream = str(payload.get("stream", ""))
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return None

    received_ns = utc_now_ns()
    event_type = data.get("e")
    symbol = str(data.get("s") or "").upper()

    if event_type == "aggTrade":
        price = _to_float(data.get("p"))
        qty = _to_float(data.get("q"))
        is_buyer_maker = bool(data.get("m"))
        signed_qty = None if qty is None else (-qty if is_buyer_maker else qty)
        return "agg_trade", {
            "stream": stream,
            "event_time_ns": _ms_to_ns(data.get("E")),
            "trade_time_ns": _ms_to_ns(data.get("T")),
            "received_time_ns": received_ns,
            "symbol": symbol,
            "agg_trade_id": data.get("a"),
            "price": price,
            "quantity": qty,
            "normal_quantity": _to_float(data.get("nq")),
            "signed_quantity": signed_qty,
            "is_buyer_maker": is_buyer_maker,
            "first_trade_id": data.get("f"),
            "last_trade_id": data.get("l"),
            "symbol_type": data.get("st"),
        }

    if event_type == "depthUpdate" or "depth" in stream:
        bids = data.get("bids", data.get("b", [])) or []
        asks = data.get("asks", data.get("a", [])) or []
        bid_sum = sum(_to_float(level[1]) or 0.0 for level in bids[:max_depth_levels])
        ask_sum = sum(_to_float(level[1]) or 0.0 for level in asks[:max_depth_levels])
        denom = bid_sum + ask_sum
        row = {
            "stream": stream,
            "event_time_ns": _ms_to_ns(data.get("E")) or received_ns,
            "transaction_time_ns": _ms_to_ns(data.get("T")),
            "received_time_ns": received_ns,
            "symbol": symbol,
            "first_update_id": data.get("U"),
            "final_update_id": data.get("u"),
            "previous_final_update_id": data.get("pu"),
            "symbol_type": data.get("st"),
            "bid_qty_top_n": bid_sum,
            "ask_qty_top_n": ask_sum,
            "book_imbalance_top_n": None if denom == 0 else (bid_sum - ask_sum) / denom,
        }
        row.update(_level_rows(bids, "bid", max_depth_levels))
        row.update(_level_rows(asks, "ask", max_depth_levels))
        return "book_depth", row

    if event_type == "forceOrder":
        order = data.get("o", {}) if isinstance(data.get("o"), dict) else {}
        symbol = str(order.get("s") or symbol).upper()
        price = _to_float(order.get("p"))
        qty = _to_float(order.get("q"))
        side = str(order.get("S") or "").upper()
        signed_qty = None if qty is None else (-qty if side == "SELL" else qty)
        return "force_order", {
            "stream": stream,
            "event_time_ns": _ms_to_ns(data.get("E")),
            "trade_time_ns": _ms_to_ns(order.get("T")),
            "received_time_ns": received_ns,
            "symbol": symbol,
            "side": side,
            "order_type": order.get("o"),
            "time_in_force": order.get("f"),
            "quantity": qty,
            "price": price,
            "average_price": _to_float(order.get("ap")),
            "order_status": order.get("X"),
            "pair_symbol": data.get("ps"),
            "symbol_type": data.get("st"),
            "last_filled_quantity": _to_float(order.get("l")),
            "accumulated_filled_quantity": _to_float(order.get("z")),
            "signed_quantity": signed_qty,
            "notional": None if price is None or qty is None else price * qty,
        }

    if event_type == "markPriceUpdate":
        return "mark_price", {
            "stream": stream,
            "event_time_ns": _ms_to_ns(data.get("E")),
            "received_time_ns": received_ns,
            "symbol": symbol,
            "mark_price": _to_float(data.get("p")),
            "index_price": _to_float(data.get("i")),
            "estimated_settle_price": _to_float(data.get("P")),
            "funding_rate": _to_float(data.get("r")),
            "next_funding_time_ns": _ms_to_ns(data.get("T")),
            "symbol_type": data.get("st"),
        }

    return None


@dataclass
class ParquetBatchBuffer:
    symbol: str
    max_rows: int = 200_000
    keep_recent_files: int | None = None

    def __post_init__(self) -> None:
        self.rows_by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(self, stream_name: str, row: dict[str, Any]) -> list[Any]:
        self.rows_by_stream[stream_name].append(row)
        if len(self.rows_by_stream[stream_name]) >= self.max_rows:
            return self.flush(stream_name)
        return []

    def flush(self, stream_name: str | None = None) -> list[Any]:
        written = []
        names = [stream_name] if stream_name else list(self.rows_by_stream)
        for name in names:
            rows = self.rows_by_stream.get(name, [])
            if not rows:
                continue
            out_path = write_partitioned_parquet(
                rows,
                raw_stream_dir(self.symbol, name),
                keep_recent_files=self.keep_recent_files,
            )
            self.rows_by_stream[name] = []
            if out_path:
                write_ingestion_manifest(
                    provider="binance",
                    market="usdm",
                    symbol=self.symbol,
                    dataset=name,
                    data_path=out_path,
                    rows=pd.DataFrame(rows),
                    source="binance_futures_websocket_or_rest",
                )
                written.append(out_path)
                LOGGER.info("Wrote %s rows to %s", len(rows), out_path)
        return written


def rest_get(path: str, params: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    response = requests.get(f"{FAPI_BASE_URL}{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def collect_rest_snapshot(symbol: str, periods: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    now_ns = utc_now_ns()
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    oi = rest_get("/fapi/v1/openInterest", {"symbol": symbol})
    if isinstance(oi, dict):
        rows["open_interest_snapshot"].append({
            "event_time_ns": _ms_to_ns(oi.get("time")) or now_ns,
            "received_time_ns": now_ns,
            "symbol": symbol,
            "open_interest": _to_float(oi.get("openInterest")),
        })

    for period in periods:
        common = {"symbol": symbol, "period": period, "limit": 1}
        for stream_name, path in [
            ("global_ls_account_ratio", "/futures/data/globalLongShortAccountRatio"),
            ("top_ls_account_ratio", "/futures/data/topLongShortAccountRatio"),
            ("top_ls_position_ratio", "/futures/data/topLongShortPositionRatio"),
        ]:
            data = rest_get(path, common)
            if isinstance(data, list) and data:
                item = data[-1]
                rows[stream_name].append({
                    "event_time_ns": _ms_to_ns(item.get("timestamp")) or now_ns,
                    "received_time_ns": now_ns,
                    "symbol": symbol,
                    "period": period,
                    "long_short_ratio": _to_float(item.get("longShortRatio")),
                    "long_account": _to_float(item.get("longAccount")),
                    "short_account": _to_float(item.get("shortAccount")),
                })
    return rows


async def rest_poll_loop(symbol: str, buffer: ParquetBatchBuffer, interval_seconds: int, periods: tuple[str, ...]) -> None:
    while True:
        try:
            for stream_name, rows in collect_rest_snapshot(symbol, periods).items():
                for row in rows:
                    buffer.add(stream_name, row)
                buffer.flush(stream_name)
        except Exception:
            LOGGER.exception("REST poll failed")
        await asyncio.sleep(interval_seconds)


async def websocket_collect_loop(
    symbol: str,
    streams: tuple[str, ...],
    buffer: ParquetBatchBuffer,
    flush_seconds: int,
    max_depth_levels: int,
    url_template: str,
    source_endpoint: str,
) -> None:
    stream_path = "/".join(streams)
    url = url_template.format(streams=stream_path)
    last_flush = time.monotonic()
    backoff_seconds = 1.0

    while True:
        try:
            LOGGER.info("Connecting Binance futures %s streams: %s", source_endpoint, stream_path)
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=5) as websocket:
                backoff_seconds = 1.0
                async for raw_message in websocket:
                    with suppress(json.JSONDecodeError):
                        normalized = normalize_ws_payload(json.loads(raw_message), max_depth_levels=max_depth_levels)
                        if normalized:
                            stream_name, row = normalized
                            if row.get("symbol") and str(row["symbol"]).upper() != symbol:
                                continue
                            row["source_endpoint"] = source_endpoint
                            buffer.add(stream_name, row)

                    if time.monotonic() - last_flush >= flush_seconds:
                        buffer.flush()
                        last_flush = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("WebSocket collection failed")
            buffer.flush()
            delay = min(backoff_seconds, 60.0) + random.uniform(0, 0.5)
            await asyncio.sleep(delay)
            backoff_seconds = min(backoff_seconds * 2, 60.0)


def default_streams(symbol: str, depth_stream: str) -> tuple[str, ...]:
    lower = symbol.lower()
    return (
        f"{lower}@aggTrade",
        f"{lower}@{depth_stream}",
        f"{lower}@markPrice@1s",
        "!forceOrder@arr",
    )


def _stream_url_template(stream: str) -> tuple[str, str]:
    if "depth" in stream or stream.endswith("@bookTicker"):
        return FSTREAM_PUBLIC_URL, "binance_usdm_public"
    return FSTREAM_MARKET_URL, "binance_usdm_market"


async def run_collector(args: argparse.Namespace) -> None:
    symbol = args.symbol.upper()
    streams = tuple(args.streams) if args.streams else default_streams(symbol, args.depth_stream)
    buffer = ParquetBatchBuffer(symbol=symbol, max_rows=args.batch_rows, keep_recent_files=args.keep_files)
    streams_by_url: dict[tuple[str, str], list[str]] = defaultdict(list)
    for stream in streams:
        streams_by_url[_stream_url_template(stream)].append(stream)

    tasks = [
        asyncio.create_task(
            websocket_collect_loop(
                symbol,
                tuple(grouped_streams),
                buffer,
                args.flush_seconds,
                args.depth_levels,
                url_template,
                source_endpoint,
            )
        )
        for (url_template, source_endpoint), grouped_streams in streams_by_url.items()
    ]
    if args.rest_poll_seconds > 0:
        tasks.append(
            asyncio.create_task(rest_poll_loop(symbol, buffer, args.rest_poll_seconds, tuple(args.rest_periods)))
        )
    try:
        await asyncio.gather(*tasks)
    finally:
        buffer.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Binance USD-M futures microstructure data to partitioned Parquet.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--depth-stream", default="depth20@100ms", help="Use depth20@100ms for top-book snapshots or depth@100ms for diff updates.")
    parser.add_argument("--streams", nargs="*", help="Override combined stream names, e.g. btcusdt@aggTrade btcusdt@depth20@100ms")
    parser.add_argument("--depth-levels", type=int, default=20)
    parser.add_argument("--batch-rows", type=int, default=200_000)
    parser.add_argument("--flush-seconds", type=int, default=3600)
    parser.add_argument("--keep-files", type=int, default=0, help="Keep only the newest N raw parquet files per stream. Use 0 to keep all files.")
    parser.add_argument("--rest-poll-seconds", type=int, default=60)
    parser.add_argument("--rest-periods", nargs="+", default=["5m", "15m", "1h"])
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    asyncio.run(run_collector(parse_args()))


if __name__ == "__main__":
    main()
