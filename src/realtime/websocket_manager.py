from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import suppress
from typing import Any

from fastapi import WebSocket

from src.realtime.event_bus import EventBus


LOGGER = logging.getLogger(__name__)


def event_subscription_key(event: dict[str, Any]) -> str | None:
    if event.get("type") == "candle_update" and event.get("candleType") == "time":
        display_symbol = event.get("displaySymbol") or event.get("symbol")
        timeframe = event.get("timeframe")
        if display_symbol and timeframe:
            return f"time:{str(display_symbol).upper()}:{str(timeframe).lower()}"

    if event.get("type") == "footprint_update":
        display_symbol = event.get("displaySymbol") or event.get("symbol")
        timeframe = event.get("timeframe")
        if display_symbol and timeframe:
            return f"time:{str(display_symbol).upper()}:{str(timeframe).lower()}"

    if event.get("type") == "cycle_data_updated":
        asset = event.get("asset")
        timeframe = event.get("timeframe")
        if asset and timeframe:
            return f"cycle:{str(asset).lower()}:{str(timeframe).lower()}"

    return None


class WebSocketManager:
    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._clients: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._forward_task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def connect(self, websocket: WebSocket, subscription_key: str) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients[subscription_key].add(websocket)
        LOGGER.info("WebSocket connected: %s", subscription_key)

    async def disconnect(self, websocket: WebSocket, subscription_key: str) -> None:
        async with self._lock:
            clients = self._clients.get(subscription_key)
            if clients:
                clients.discard(websocket)
                if not clients:
                    self._clients.pop(subscription_key, None)
        LOGGER.info("WebSocket disconnected: %s", subscription_key)

    async def send_error_and_close(self, websocket: WebSocket, code: str, message: str) -> None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "code": code, "message": message})
        await websocket.close(code=1008)

    async def run(self) -> None:
        if self._forward_task and not self._forward_task.done():
            return
        self._stopping.clear()
        queue = await self.event_bus.subscribe()
        try:
            while not self._stopping.is_set():
                event = await queue.get()
                key = event_subscription_key(event)
                if key:
                    await self.broadcast(key, event)
        finally:
            await self.event_bus.unsubscribe(queue)

    async def stop(self) -> None:
        self._stopping.set()

    async def broadcast(self, subscription_key: str, event: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients.get(subscription_key, set()))

        stale_clients: list[WebSocket] = []
        for websocket in clients:
            try:
                await websocket.send_json(event)
            except Exception:
                stale_clients.append(websocket)

        if stale_clients:
            async with self._lock:
                clients_set = self._clients.get(subscription_key)
                if clients_set:
                    for websocket in stale_clients:
                        clients_set.discard(websocket)

    async def close_all(self) -> None:
        async with self._lock:
            entries = [(key, websocket) for key, clients in self._clients.items() for websocket in clients]
            self._clients.clear()
        for _, websocket in entries:
            with suppress(Exception):
                await websocket.close()
