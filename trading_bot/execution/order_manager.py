"""
trading_bot/execution/order_manager.py
=====================================
Manage Binance Futures account reads and order placement.

- Account reads can use the real exchange even when dry_run=True.
- Order placement honors dry_run and will not send live orders unless disabled.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("bot.order")


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    side: str = ""
    size: float = 0.0
    price: float = 0.0
    error: str = ""

    def __str__(self):
        if self.success:
            return f"OK {self.side} {self.size} @ {self.price} (id={self.order_id})"
        return f"ERROR {self.error}"


class OrderManager:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        leverage: int = 5,
        testnet: bool = True,
        dry_run: bool = True,
    ):
        self.symbol = symbol
        self.leverage = leverage
        self.dry_run = dry_run
        self._exchange = None
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet

    def _can_query_live(self) -> bool:
        return bool(self._api_key and self._api_secret)

    def _get_exchange(self):
        if self._exchange is None:
            import ccxt

            cfg = {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "options": {
                    "defaultType": "future",
                    "adjustForTimeDifference": True,
                    "recvWindow": 10000,
                },
                "enableRateLimit": True,
            }
            if self._testnet:
                cfg["options"]["sandboxMode"] = True
            self._exchange = ccxt.binance(cfg)
            try:
                self._exchange.load_time_difference()
            except Exception as exc:
                logger.warning("Time difference sync failed: %s", exc)
            try:
                self._exchange.set_leverage(self.leverage, self.symbol)
            except Exception as exc:
                logger.warning("Leverage setup failed: %s", exc)
        return self._exchange

    def get_balance(self) -> float:
        if not self._can_query_live():
            return 10000.0 if self.dry_run else 0.0
        try:
            bal = self._get_exchange().fetch_balance()
            return float((bal.get("USDT", {}) or {}).get("free", 0) or 0)
        except Exception as exc:
            logger.error("Balance lookup failed: %s", exc)
            return 10000.0 if self.dry_run else 0.0

    def get_account_snapshot(self) -> dict:
        if not self._can_query_live():
            fallback = 10000.0 if self.dry_run else 0.0
            return {
                "free_usdt": fallback,
                "used_usdt": 0.0,
                "total_usdt": fallback,
                "source": "dry_run_placeholder" if self.dry_run else "unavailable",
            }
        try:
            bal = self._get_exchange().fetch_balance()
            usdt = bal.get("USDT", {}) or {}
            free_usdt = float(usdt.get("free", 0) or 0)
            used_usdt = float(usdt.get("used", 0) or 0)
            total_usdt = float(usdt.get("total", free_usdt + used_usdt) or (free_usdt + used_usdt))
            return {
                "free_usdt": free_usdt,
                "used_usdt": used_usdt,
                "total_usdt": total_usdt,
                "source": "exchange",
            }
        except Exception as exc:
            logger.error("Account snapshot lookup failed: %s", exc)
            fallback = 10000.0 if self.dry_run else 0.0
            return {
                "free_usdt": fallback,
                "used_usdt": 0.0,
                "total_usdt": fallback,
                "source": "fallback",
            }

    def get_position(self) -> Optional[dict]:
        if not self._can_query_live():
            return None
        try:
            positions = self._get_exchange().fetch_positions([self.symbol])
            for position in positions:
                contracts = float(position.get("contracts", 0) or 0)
                if contracts > 0:
                    return {
                        "side": str(position.get("side", "")).upper(),
                        "size": contracts,
                        "entry_price": float(position.get("entryPrice", 0) or 0),
                        "unrealized_pnl": float(position.get("unrealizedPnl", 0) or 0),
                        "notional": float(position.get("notional", 0) or 0),
                        "leverage": float(position.get("leverage", self.leverage) or self.leverage),
                        "mark_price": float(position.get("markPrice", 0) or 0),
                        "liquidation_price": float(position.get("liquidationPrice", 0) or 0),
                        "margin_mode": position.get("marginMode", ""),
                    }
            return None
        except Exception as exc:
            logger.error("Position lookup failed: %s", exc)
            return None

    def get_current_price(self) -> float:
        try:
            ticker = self._get_exchange().fetch_ticker(self.symbol)
            return float(ticker.get("last", 0) or 0)
        except Exception as exc:
            logger.error("Price lookup failed: %s", exc)
            return 0.0

    def calculate_order_size(self, balance: float, size_pct: int, price: float) -> float:
        if price <= 0:
            return 0.0
        usdt = balance * (size_pct / 100.0) * self.leverage
        return round(usdt / price, 3)

    def enter_position(
        self,
        side: str,
        size_pct: int,
        sl_pct: float = 2.0,
        tp_pct: float = 4.0,
    ) -> OrderResult:
        balance = self.get_balance()
        price = self.get_current_price()
        size = self.calculate_order_size(balance, size_pct, price)

        if size <= 0:
            return OrderResult(success=False, error="order size <= 0")

        if self.dry_run:
            logger.info(
                "[DRY] %s %s BTC @ ~%s (bal=%.0f, %s%%, %sx)",
                side,
                size,
                price,
                balance,
                size_pct,
                self.leverage,
            )
            return OrderResult(True, "DRY_RUN", side, size, price)

        try:
            exchange = self._get_exchange()
            order_side = "buy" if side == "LONG" else "sell"
            order = exchange.create_market_order(self.symbol, order_side, size)
            entry_price = float(order.get("average", price) or price)
            logger.info("Filled: %s %s @ %s", side, size, entry_price)
            self._set_sl_tp(side, size, entry_price, sl_pct, tp_pct)
            return OrderResult(True, order.get("id", ""), side, size, entry_price)
        except Exception as exc:
            logger.error("Order failed: %s", exc, exc_info=True)
            return OrderResult(success=False, error=str(exc))

    def close_position(self, side: str) -> OrderResult:
        if self.dry_run:
            logger.info("[DRY] Close %s", side)
            return OrderResult(True, "DRY_CLOSE", side)

        pos = self.get_position()
        if not pos:
            return OrderResult(success=False, error="no open position")

        try:
            exchange = self._get_exchange()
            close_side = "sell" if side == "LONG" else "buy"
            order = exchange.create_market_order(
                self.symbol,
                close_side,
                pos["size"],
                params={"reduceOnly": True},
            )
            close_price = float(order.get("average", 0) or 0)
            logger.info("Closed: %s %s @ %s", side, pos["size"], close_price)
            return OrderResult(True, order.get("id", ""), side, pos["size"], close_price)
        except Exception as exc:
            logger.error("Close failed: %s", exc, exc_info=True)
            return OrderResult(success=False, error=str(exc))

    def _set_sl_tp(self, side: str, size: float, entry: float, sl_pct: float, tp_pct: float):
        try:
            exchange = self._get_exchange()
            if side == "LONG":
                sl_price = round(entry * (1 - sl_pct / 100), 2)
                tp_price = round(entry * (1 + tp_pct / 100), 2)
                exit_side = "sell"
            else:
                sl_price = round(entry * (1 + sl_pct / 100), 2)
                tp_price = round(entry * (1 - tp_pct / 100), 2)
                exit_side = "buy"

            exchange.create_order(
                self.symbol,
                "stop_market",
                exit_side,
                size,
                params={"stopPrice": sl_price, "reduceOnly": True},
            )
            exchange.create_order(
                self.symbol,
                "take_profit_market",
                exit_side,
                size,
                params={"stopPrice": tp_price, "reduceOnly": True},
            )
            logger.info("SL=%s (%s%%), TP=%s (%s%%)", sl_price, sl_pct, tp_price, tp_pct)
        except Exception as exc:
            logger.warning("SL/TP setup failed: %s", exc)
