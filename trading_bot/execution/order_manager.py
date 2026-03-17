"""
trading_bot/execution/order_manager.py
=======================================
ccxt → Binance Futures 주문 관리.
dry_run=True 시 실제 주문 없이 로그만 기록.
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
            return f"✅ {self.side} {self.size} @ {self.price} (id={self.order_id})"
        return f"❌ {self.error}"


class OrderManager:
    def __init__(self, api_key: str, api_secret: str, symbol: str,
                 leverage: int = 5, testnet: bool = True, dry_run: bool = True):
        self.symbol = symbol
        self.leverage = leverage
        self.dry_run = dry_run
        self._exchange = None
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet

    def _get_exchange(self):
        if self._exchange is None:
            import ccxt
            cfg = {
                'apiKey': self._api_key,
                'secret': self._api_secret,
                'options': {'defaultType': 'future'},
                'enableRateLimit': True,
            }
            if self._testnet:
                cfg['options']['sandboxMode'] = True
            self._exchange = ccxt.binance(cfg)
            try:
                self._exchange.set_leverage(self.leverage, self.symbol)
            except Exception as e:
                logger.warning(f"Leverage 설정 실패: {e}")
        return self._exchange

    def get_balance(self) -> float:
        if self.dry_run:
            return 10000.0
        try:
            bal = self._get_exchange().fetch_balance()
            return float(bal.get('USDT', {}).get('free', 0))
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            return 0.0

    def get_position(self) -> Optional[dict]:
        if self.dry_run:
            return None
        try:
            positions = self._get_exchange().fetch_positions([self.symbol])
            for p in positions:
                if float(p.get('contracts', 0)) > 0:
                    return {
                        "side": p.get('side', '').upper(),
                        "size": float(p['contracts']),
                        "entry_price": float(p.get('entryPrice', 0)),
                        "unrealized_pnl": float(p.get('unrealizedPnl', 0)),
                        "notional": float(p.get('notional', 0)),
                    }
            return None
        except Exception as e:
            logger.error(f"포지션 조회 실패: {e}")
            return None

    def get_current_price(self) -> float:
        try:
            t = self._get_exchange().fetch_ticker(self.symbol)
            return float(t.get('last', 0))
        except Exception as e:
            logger.error(f"가격 조회 실패: {e}")
            return 0.0

    def calculate_order_size(self, balance: float, size_pct: int, price: float) -> float:
        if price <= 0:
            return 0.0
        usdt = balance * (size_pct / 100.0) * self.leverage
        return round(usdt / price, 3)

    def enter_position(self, side: str, size_pct: int,
                       sl_pct: float = 2.0, tp_pct: float = 4.0) -> OrderResult:
        balance = self.get_balance()
        price = self.get_current_price()
        size = self.calculate_order_size(balance, size_pct, price)

        if size <= 0:
            return OrderResult(success=False, error="주문 수량 0 이하")

        if self.dry_run:
            logger.info(
                f"[DRY] {side} {size} BTC @ ~{price} "
                f"(bal={balance:.0f}, {size_pct}%, {self.leverage}x)"
            )
            return OrderResult(True, "DRY_RUN", side, size, price)

        try:
            ex = self._get_exchange()
            order_side = "buy" if side == "LONG" else "sell"
            order = ex.create_market_order(self.symbol, order_side, size)
            ep = float(order.get('average', price))
            logger.info(f"Filled: {side} {size} @ {ep}")
            self._set_sl_tp(side, size, ep, sl_pct, tp_pct)
            return OrderResult(True, order.get('id', ''), side, size, ep)
        except Exception as e:
            logger.error(f"주문 실패: {e}", exc_info=True)
            return OrderResult(success=False, error=str(e))

    def close_position(self, side: str) -> OrderResult:
        if self.dry_run:
            logger.info(f"[DRY] Close {side}")
            return OrderResult(True, "DRY_CLOSE", side)

        pos = self.get_position()
        if not pos:
            return OrderResult(success=False, error="포지션 없음")

        try:
            ex = self._get_exchange()
            close_side = "sell" if side == "LONG" else "buy"
            order = ex.create_market_order(
                self.symbol, close_side, pos["size"],
                params={"reduceOnly": True},
            )
            cp = float(order.get('average', 0))
            logger.info(f"Closed: {side} {pos['size']} @ {cp}")
            return OrderResult(True, order.get('id', ''), side, pos['size'], cp)
        except Exception as e:
            logger.error(f"청산 실패: {e}", exc_info=True)
            return OrderResult(success=False, error=str(e))

    def _set_sl_tp(self, side: str, size: float, entry: float,
                   sl_pct: float, tp_pct: float):
        try:
            ex = self._get_exchange()
            if side == "LONG":
                sl_p = round(entry * (1 - sl_pct / 100), 2)
                tp_p = round(entry * (1 + tp_pct / 100), 2)
                sl_side = tp_side = "sell"
            else:
                sl_p = round(entry * (1 + sl_pct / 100), 2)
                tp_p = round(entry * (1 - tp_pct / 100), 2)
                sl_side = tp_side = "buy"

            ex.create_order(self.symbol, "stop_market", sl_side, size,
                            params={"stopPrice": sl_p, "reduceOnly": True})
            ex.create_order(self.symbol, "take_profit_market", tp_side, size,
                            params={"stopPrice": tp_p, "reduceOnly": True})
            logger.info(f"SL={sl_p} ({sl_pct}%), TP={tp_p} ({tp_pct}%)")
        except Exception as e:
            logger.warning(f"SL/TP 설정 실패: {e}")
