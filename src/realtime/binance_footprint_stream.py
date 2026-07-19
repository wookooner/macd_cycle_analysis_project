from __future__ import annotations

import asyncio
import bisect
import json
import logging
import random
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import websockets

from src.realtime.event_bus import EventBus


LOGGER = logging.getLogger(__name__)

BINANCE_FUTURES_WS_URL = "wss://fstream.binancefuture.com/stream?streams={streams}"
DISPLAY_SYMBOLS = {"BTCUSDT": "BTCUSD"}
FOOTPRINT_TIMEFRAMES = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
}


def normalize_agg_trade(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize Binance aggregated trades to the minimum footprint contract.

    ``m`` means the buyer was the maker, therefore the aggressive side is sell.
    Aggregated trades are intentionally used instead of individual fills: they are
    sufficient for a local footprint and substantially reduce message volume.
    """
    data = payload.get("data", payload)
    if not isinstance(data, dict) or data.get("e") != "aggTrade":
        return None
    try:
        symbol = str(data["s"]).upper()
        return {
            "symbol": symbol,
            "trade_time_ms": int(data.get("T") or data["E"]),
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "is_buyer_maker": bool(data["m"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


@dataclass
class _FootprintBar:
    start_ms: int
    levels: dict[float, list[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0]))
    last_price: float | None = None
    trade_count: int = 0


class FootprintAggregator:
    """In-memory, bounded footprint aggregation for a small number of live bars."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        price_bin_size: float = 20.0,
        max_price_levels: int = 48,
        history_bars: int = 10,
    ) -> None:
        if timeframe not in FOOTPRINT_TIMEFRAMES:
            raise ValueError(f"Unsupported footprint timeframe: {timeframe}")
        if price_bin_size <= 0:
            raise ValueError("price_bin_size must be positive")
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.interval_ms = FOOTPRINT_TIMEFRAMES[timeframe]
        self.price_bin_size = price_bin_size
        self.max_price_levels = max_price_levels
        self.history_bars = max(2, history_bars)
        self._bars: dict[int, _FootprintBar] = {}

    def ingest(self, trade: dict[str, Any]) -> None:
        if trade.get("symbol") != self.symbol:
            return
        trade_time_ms = int(trade["trade_time_ms"])
        start_ms = trade_time_ms - (trade_time_ms % self.interval_ms)
        self._drop_old_bars(start_ms)
        bar = self._bars.setdefault(start_ms, _FootprintBar(start_ms=start_ms))
        price = float(trade["price"])
        price_bin = round(price / self.price_bin_size) * self.price_bin_size
        quantity = float(trade["quantity"])
        # buy index 0 / sell index 1, both expressed as base-asset quantity.
        side_index = 1 if trade["is_buyer_maker"] else 0
        bar.levels[price_bin][side_index] += quantity
        bar.last_price = price
        bar.trade_count += 1

    def snapshot(self, now_ms: int | None = None) -> dict[str, Any] | None:
        if not self._bars:
            return None
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        current_start = now_ms - (now_ms % self.interval_ms)
        start_ms = max((key for key in self._bars if key <= current_start), default=max(self._bars))
        bar = self._bars.get(start_ms)
        if not bar:
            return None

        visible_bars = [self._bars[key] for key in sorted(self._bars)[-self.history_bars :]]
        surface_prices = self._select_surface_prices(visible_bars, bar.last_price)
        bars = [self._serialize_bar(item, surface_prices) for item in visible_bars]
        current_bar = next((item for item in bars if item["barStartMs"] == bar.start_ms), bars[-1])
        return {
            "barStartMs": current_bar["barStartMs"],
            "barEndMs": current_bar["barEndMs"],
            "priceBinSize": self.price_bin_size,
            "lastPrice": current_bar["lastPrice"],
            "tradeCount": current_bar["tradeCount"],
            "levels": current_bar["levels"],
            "bars": bars,
        }

    def _select_surface_prices(self, bars: list[_FootprintBar], anchor_price: float | None) -> list[float]:
        prices = sorted({price for bar in bars for price in bar.levels})
        if len(prices) <= self.max_price_levels or anchor_price is None:
            return prices
        insert_at = bisect.bisect_left(prices, anchor_price)
        lower = max(0, insert_at - self.max_price_levels // 2)
        upper = min(len(prices), lower + self.max_price_levels)
        return prices[max(0, upper - self.max_price_levels) : upper]

    def _serialize_bar(self, bar: _FootprintBar, prices: list[float]) -> dict[str, Any]:
        raw_levels = []
        for price in reversed(prices):
            buy_volume, sell_volume = bar.levels.get(price, (0.0, 0.0))
            raw_levels.append(
                {
                    "price": price,
                    "buyVolume": buy_volume,
                    "sellVolume": sell_volume,
                    "delta": buy_volume - sell_volume,
                    "totalVolume": buy_volume + sell_volume,
                }
            )
        point_of_control = max(raw_levels, key=lambda level: level["totalVolume"], default=None)
        for level in raw_levels:
            buy_volume = level["buyVolume"]
            sell_volume = level["sellVolume"]
            level["isPoc"] = bool(point_of_control and level["price"] == point_of_control["price"] and level["totalVolume"] > 0)
            level["buyImbalance"] = buy_volume > 0 and buy_volume >= sell_volume * 3
            level["sellImbalance"] = sell_volume > 0 and sell_volume >= buy_volume * 3
        return {
            "barStartMs": bar.start_ms,
            "barEndMs": bar.start_ms + self.interval_ms,
            "lastPrice": bar.last_price,
            "tradeCount": bar.trade_count,
            "totalVolume": sum(level["totalVolume"] for level in raw_levels),
            "delta": sum(level["delta"] for level in raw_levels),
            "pocPrice": point_of_control["price"] if point_of_control and point_of_control["totalVolume"] > 0 else None,
            "levels": raw_levels,
        }

    def _drop_old_bars(self, current_start_ms: int) -> None:
        # Retain a compact rolling surface, never an unbounded in-memory tape.
        cutoff = current_start_ms - self.interval_ms * (self.history_bars - 1)
        for start_ms in list(self._bars):
            if start_ms < cutoff:
                del self._bars[start_ms]


class BinanceFootprintSupervisor:
    def __init__(
        self,
        event_bus: EventBus,
        symbols: tuple[str, ...] = ("BTCUSDT",),
        timeframes: tuple[str, ...] = tuple(FOOTPRINT_TIMEFRAMES),
        price_bin_size: float = 5.0,
        publish_interval_seconds: float = 0.25,
    ) -> None:
        self.event_bus = event_bus
        self.symbols = tuple(symbol.upper() for symbol in symbols)
        self.timeframes = tuple(timeframe for timeframe in timeframes if timeframe in FOOTPRINT_TIMEFRAMES)
        self.publish_interval_seconds = publish_interval_seconds
        self._aggregators = {
            (symbol, timeframe): FootprintAggregator(symbol, timeframe, price_bin_size=price_bin_size)
            for symbol in self.symbols
            for timeframe in self.timeframes
        }
        self._stop_event = asyncio.Event()
        self._last_publish = 0.0

    async def run(self) -> None:
        backoff_seconds = 1.0
        while not self._stop_event.is_set():
            try:
                await self._run_once()
                backoff_seconds = 1.0
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Binance footprint stream failed")
            if self._stop_event.is_set():
                break
            await asyncio.sleep(min(backoff_seconds, 60.0) + random.uniform(0, 0.5))
            backoff_seconds = min(backoff_seconds * 2, 60.0)

    async def stop(self) -> None:
        self._stop_event.set()

    async def _run_once(self) -> None:
        streams = "/".join(f"{symbol.lower()}@aggTrade" for symbol in self.symbols)
        if not streams or not self._aggregators:
            return
        url = BINANCE_FUTURES_WS_URL.format(streams=streams)
        LOGGER.info("Connecting Binance footprint stream: %s", streams)
        async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=5) as websocket:
            async for raw_message in websocket:
                if self._stop_event.is_set():
                    return
                with suppress(json.JSONDecodeError):
                    trade = normalize_agg_trade(json.loads(raw_message))
                    if not trade:
                        continue
                    for timeframe in self.timeframes:
                        aggregator = self._aggregators.get((trade["symbol"], timeframe))
                        if aggregator:
                            aggregator.ingest(trade)
                    await self._publish_if_due()

    async def _publish_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_publish < self.publish_interval_seconds:
            return
        self._last_publish = now
        for (symbol, timeframe), aggregator in self._aggregators.items():
            snapshot = aggregator.snapshot()
            if snapshot:
                await self.event_bus.publish(
                    {
                        "type": "footprint_update",
                        "source": "binance_futures_agg_trade_ws",
                        "symbol": symbol,
                        "displaySymbol": DISPLAY_SYMBOLS.get(symbol, symbol),
                        "exchange": "binance_usdm",
                        "marketType": "futures_usdm",
                        "timeframe": timeframe,
                        "data": snapshot,
                    }
                )
