from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.realtime.binance_kline_stream import normalize_symbol, normalize_timeframe, subscription_key
from src.realtime.websocket_manager import WebSocketManager


LOGGER = logging.getLogger(__name__)


def create_realtime_router(websocket_manager: WebSocketManager) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/market")
    async def market_websocket(websocket: WebSocket) -> None:
        display_symbol = str(websocket.query_params.get("symbol") or "").strip().upper()
        timeframe = normalize_timeframe(str(websocket.query_params.get("timeframe") or ""))
        exchange_symbol = normalize_symbol(display_symbol)

        if not exchange_symbol:
            await websocket_manager.send_error_and_close(
                websocket,
                "UNSUPPORTED_SYMBOL",
                f"Unsupported symbol: {display_symbol or '-'}",
            )
            return

        if not timeframe:
            await websocket_manager.send_error_and_close(
                websocket,
                "UNSUPPORTED_TIMEFRAME",
                f"Unsupported timeframe: {websocket.query_params.get('timeframe') or '-'}",
            )
            return

        key = subscription_key(display_symbol, timeframe)
        await websocket_manager.connect(websocket, key)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            LOGGER.exception("Market WebSocket failed: %s", key)
        finally:
            await websocket_manager.disconnect(websocket, key)

    return router
