from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any


LOGGER = logging.getLogger(__name__)


class EventBus:
    """Small in-process pub/sub bus.

    The bus deliberately knows nothing about WebSocket clients. Producers publish
    normalized dashboard events, and consumers decide how to route them.
    """

    def __init__(self, queue_size: int = 512) -> None:
        self.queue_size = queue_size
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.queue_size)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)
        with suppress(asyncio.QueueEmpty):
            while True:
                queue.get_nowait()

    async def publish(self, event: dict[str, Any]) -> None:
        async with self._lock:
            subscribers = list(self._subscribers)

        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                LOGGER.warning("Dropping realtime event for a slow subscriber: %s", event.get("type"))


event_bus = EventBus()

