from __future__ import annotations

import asyncio
import json
import logging
import random
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

import websockets

from src.realtime.event_bus import EventBus


LOGGER = logging.getLogger(__name__)

SUPPORTED_TIMEFRAMES = ("5m", "15m", "1h")
DISPLAY_TO_EXCHANGE_SYMBOL = {
    "BTCUSD": "BTCUSDT",
    "BTCUSDT": "BTCUSDT",
}
EXCHANGE_TO_DISPLAY_SYMBOL = {
    "BTCUSDT": "BTCUSD",
}
BINANCE_FUTURES_WS_URL = "wss://fstream.binancefuture.com/stream?streams={streams}"


def normalize_symbol(symbol: str) -> str | None:
    return DISPLAY_TO_EXCHANGE_SYMBOL.get(str(symbol or "").strip().upper())


def normalize_timeframe(timeframe: str) -> str | None:
    normalized = str(timeframe or "").strip().lower()
    return normalized if normalized in SUPPORTED_TIMEFRAMES else None


def subscription_key(display_symbol: str, timeframe: str) -> str:
    return f"time:{display_symbol.upper()}:{timeframe.lower()}"


def _iso_from_millis(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_kline_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    data = payload.get("data") if "data" in payload else payload
    kline = data.get("k") if isinstance(data, dict) else None
    if not isinstance(kline, dict):
        return None

    exchange_symbol = str(kline.get("s") or data.get("s") or "").upper()
    timeframe = normalize_timeframe(str(kline.get("i") or ""))
    start_ms = kline.get("t")
    if not exchange_symbol or not timeframe or start_ms is None:
        return None

    try:
        start_ms_int = int(start_ms)
        is_closed = bool(kline.get("x"))
        return {
            "type": "candle_update",
            "source": "binance_futures_ws",
            "candleType": "time",
            "symbol": exchange_symbol,
            "displaySymbol": EXCHANGE_TO_DISPLAY_SYMBOL.get(exchange_symbol, exchange_symbol),
            "exchange": "binance_usdm",
            "marketType": "futures_usdm",
            "timeframe": timeframe,
            "data": {
                "unix": start_ms_int // 1000,
                "date": _iso_from_millis(start_ms_int),
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"]),
                "isClosed": is_closed,
                "isPartial": not is_closed,
            },
        }
    except (KeyError, TypeError, ValueError) as error:
        LOGGER.debug("Failed to normalize Binance kline payload: %s", error)
        return None


class BinanceKlineSupervisor:
    def __init__(
        self,
        event_bus: EventBus,
        symbols: tuple[str, ...] = ("BTCUSDT",),
        timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
    ) -> None:
        self.event_bus = event_bus
        self.symbols = tuple(symbol.upper() for symbol in symbols)
        self.timeframes = tuple(timeframe.lower() for timeframe in timeframes if timeframe in SUPPORTED_TIMEFRAMES)
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        backoff_seconds = 1.0
        while not self._stop_event.is_set():
            try:
                await self._run_once()
                backoff_seconds = 1.0
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Binance kline stream failed")

            if self._stop_event.is_set():
                break

            delay = min(backoff_seconds, 60.0) + random.uniform(0, 0.5)
            await asyncio.sleep(delay)
            backoff_seconds = min(backoff_seconds * 2, 60.0)

    async def stop(self) -> None:
        self._stop_event.set()

    async def _run_once(self) -> None:
        streams = "/".join(
            f"{symbol.lower()}@kline_{timeframe}"
            for symbol in self.symbols
            for timeframe in self.timeframes
        )
        if not streams:
            LOGGER.warning("No Binance kline streams configured")
            await asyncio.sleep(60)
            return

        url = BINANCE_FUTURES_WS_URL.format(streams=streams)
        LOGGER.info("Connecting Binance kline stream: %s", streams)
        async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=5) as websocket:
            async for raw_message in websocket:
                if self._stop_event.is_set():
                    break
                with suppress(json.JSONDecodeError):
                    event = normalize_kline_event(json.loads(raw_message))
                    if event:
                        await self.event_bus.publish(event)
