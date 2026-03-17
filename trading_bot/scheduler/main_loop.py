"""
trading_bot/scheduler/main_loop.py
====================================
APScheduler 기반 메인 루프.
매 XX:55 파이프라인 → 상태 추출 → AI 판단 → 로그 → 알림/실행.

모든 AI 의사결정은 DecisionLogger에 기록됨.
"""

import asyncio
import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import Settings
from data.pipeline_runner import PipelineRunner
from cycle.state_extractor import CycleStateExtractor
from strategy.ai_signal_engine import AISignalEngine, Signal, TradeAction
from execution.order_manager import OrderManager
from telegram_bot.bot import TradingTelegramBot
from utils.decision_logger import DecisionLogger

logger = logging.getLogger("bot.scheduler")


class TradingScheduler:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.scheduler = AsyncIOScheduler()

        self.pipeline = PipelineRunner(settings.paths.analysis_project_root)

        self.state_extractor = CycleStateExtractor(
            parquet_dir=settings.paths.parquet_dir,
            hierarchy_map_path=settings.paths.hierarchy_map_path,
            timeframes=settings.timeframes,
        )

        self.ai_engine = AISignalEngine(
            api_key=settings.ai.api_key,
            model=settings.ai.model,
            rules_doc_path=settings.paths.rules_doc_path,
            max_tokens=settings.ai.max_tokens,
            temperature=settings.ai.temperature,
        )

        self.order_manager = OrderManager(
            api_key=settings.binance.api_key,
            api_secret=settings.binance.api_secret,
            symbol=settings.binance.symbol,
            leverage=settings.binance.leverage,
            testnet=settings.binance.testnet,
            dry_run=settings.trading.dry_run,
        )

        self.telegram = TradingTelegramBot(
            token=settings.telegram.bot_token,
            chat_id=settings.telegram.chat_id,
            approval_timeout_minutes=settings.telegram.approval_timeout_minutes,
        )

        # 의사결정 로그
        self.decision_log = DecisionLogger(settings.paths.log_dir)

        # Telegram 콜백 연결
        self.telegram.on_approve = self._handle_approval
        self.telegram.on_close = self._handle_close
        self.telegram.on_status = self._handle_status

        # 마지막 market_state 캐시 (status에서 사용)
        self._last_market_state: dict = {}

    async def start(self):
        await self.telegram.initialize()

        self.scheduler.add_job(
            self._hourly_cycle,
            CronTrigger(
                minute=self.settings.scheduler.cron_minute,
                second=self.settings.scheduler.cron_second,
            ),
            id="hourly_cycle",
            name="Hourly Pipeline + Signal (XX:55)",
            misfire_grace_time=120,
        )

        self.scheduler.start()

        mode = "DRY RUN" if self.settings.trading.dry_run else "LIVE"
        bot_mode = self.telegram.mode_label
        await self.telegram.send_message(
            f"🚀 *Bot Started* [{mode}]\n"
            f"Symbol: {self.settings.binance.symbol}\n"
            f"Leverage: {self.settings.binance.leverage}x\n"
            f"Schedule: XX:{self.settings.scheduler.cron_minute:02d}\n"
            f"Mode: {bot_mode}\n"
            f"AI: {self.settings.ai.model}"
        )
        logger.info(f"Scheduler started [{mode}] [{bot_mode}]")

    async def stop(self):
        self.scheduler.shutdown(wait=False)
        await self.telegram.send_message("🛑 Bot stopped")
        await self.telegram.shutdown()

    # ── 매 XX:55 실행 ──────────────────────────────────

    async def _hourly_cycle(self):
        now = datetime.utcnow()
        logger.info(f"=== Hourly cycle: {now.strftime('%H:%M:%S')} ===")

        if self.telegram.is_paused:
            logger.info("Paused — skip")
            return

        try:
            # 1. 파이프라인
            ok = await asyncio.get_event_loop().run_in_executor(
                None, self.pipeline.run
            )
            if not ok:
                await self.telegram.send_message(
                    "⚠️ Pipeline fail. Using previous data."
                )
                self.decision_log.log_error("Pipeline failed")

            # 2. 상태 추출
            pos_info = None
            pos = self.order_manager.get_position()
            if pos:
                pos_info = {
                    "side": pos["side"],
                    "entry_price": pos["entry_price"],
                    "size_usdt": pos.get("notional", 0),
                    "unrealized_pnl_pct": (
                        pos["unrealized_pnl"] /
                        max(pos.get("notional", 1), 1) * 100
                    ),
                }

            market_state = self.state_extractor.extract_current_state(
                current_position=pos_info,
            )
            self._last_market_state = market_state

            # 3. AI 판단
            signal = self.ai_engine.evaluate(market_state)
            mode = self.telegram.mode_label

            # 4. 로그 + 처리
            if signal.is_actionable():
                # 액션 가능한 신호 → 로그 + 알림
                self.decision_log.log_signal(
                    market_state, signal.to_dict(), mode
                )
                logger.info(f"Signal: {signal.action.value} [{mode}]")
                await self.telegram.send_signal_alert(signal)
                # 자동 모드에서는 send_signal_alert 내부에서 즉시 실행됨
            else:
                # HOLD → 로그만
                self.decision_log.log_hold(
                    market_state, signal.to_dict(), mode
                )
                logger.info(f"HOLD — {signal.reasoning[:100]}")

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            self.decision_log.log_error(str(e))
            await self.telegram.send_message(f"❌ Error: {str(e)[:200]}")

    # ── 콜백 ──────────────────────────────────────────

    async def _handle_approval(self, signal: Signal):
        """승인된 신호 또는 자동 모드 신호 실행"""
        mode = self.telegram.mode_label
        user_response = "auto_executed" if self.telegram.is_auto_mode else "approved"
        logger.info(f"Executing: {signal.action.value} [{user_response}]")

        try:
            result = None

            if signal.action == TradeAction.ENTER_LONG:
                result = self.order_manager.enter_position(
                    "LONG", signal.size_pct, signal.sl_pct, signal.tp_pct)
            elif signal.action == TradeAction.ENTER_SHORT:
                result = self.order_manager.enter_position(
                    "SHORT", signal.size_pct, signal.sl_pct, signal.tp_pct)
            elif signal.action == TradeAction.CLOSE_LONG:
                result = self.order_manager.close_position("LONG")
            elif signal.action == TradeAction.CLOSE_SHORT:
                result = self.order_manager.close_position("SHORT")
            elif signal.action == TradeAction.REVERSE_TO_LONG:
                self.order_manager.close_position("SHORT")
                result = self.order_manager.enter_position(
                    "LONG", signal.size_pct, signal.sl_pct, signal.tp_pct)
            elif signal.action == TradeAction.REVERSE_TO_SHORT:
                self.order_manager.close_position("LONG")
                result = self.order_manager.enter_position(
                    "SHORT", signal.size_pct, signal.sl_pct, signal.tp_pct)

            if result:
                # 실행 결과 로그
                self.decision_log.log_execution(
                    signal.to_dict(),
                    {"success": result.success, "side": result.side,
                     "size": result.size, "price": result.price,
                     "error": result.error},
                    mode, user_response,
                )
                await self.telegram.send_order_result(result)

        except Exception as e:
            logger.error(f"Execution error: {e}", exc_info=True)
            self.decision_log.log_error(str(e), signal.to_dict())
            await self.telegram.send_message(f"❌ Execution error: {str(e)[:200]}")

    async def _handle_close(self, side: str):
        try:
            pos = self.order_manager.get_position()
            if pos:
                result = self.order_manager.close_position(pos["side"])
                self.decision_log.log_execution(
                    {"action": "EMERGENCY_CLOSE"}, 
                    {"success": result.success, "side": result.side,
                     "price": result.price},
                    "MANUAL", "emergency_close",
                )
                await self.telegram.send_order_result(result)
            else:
                await self.telegram.send_message("No position")
        except Exception as e:
            await self.telegram.send_message(f"❌ Close error: {str(e)[:200]}")

    async def _handle_status(self) -> str:
        try:
            pos = self.order_manager.get_position()
            pos_text = "None"
            if pos:
                pnl = pos.get('unrealized_pnl', 0)
                pos_text = (
                    f"{pos['side']} {pos['size']} BTC\n"
                    f"  Entry: {pos['entry_price']}\n"
                    f"  PnL: {pnl:+.2f} USDT"
                )

            bal = self.order_manager.get_balance()
            chain = self._last_market_state.get("chain", {})
            price = self._last_market_state.get("price", "?")

            # 최근 의사결정 요약
            recent = self.decision_log.get_recent_decisions(3)
            recent_text = ""
            for r in recent:
                evt = r.get("event", "")
                ts = r.get("timestamp", "")[-8:]  # HH:MM:SS
                sig = r.get("signal", {})
                act = sig.get("action", evt)
                recent_text += f"  {ts} {act}\n"

            if not recent_text:
                recent_text = "  (none)\n"

            return (
                f"📊 *현재 상태*\n"
                f"━━━━━━━━━━━━━━━\n"
                f"*Price*: {price}\n"
                f"*Balance*: {bal:,.0f} USDT\n"
                f"*Position*: {pos_text}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"*Combo*: {chain.get('combo', '?')}\n"
                f"*n\\_up*: {chain.get('n_up', '?')}\n"
                f"*4h-1h align*: {'Y' if chain.get('alignment_4h_1h') else 'N'}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"*Mode*: {'DRY' if self.settings.trading.dry_run else 'LIVE'}\n"
                f"*Trade mode*: {self.telegram.mode_label}\n"
                f"*Paused*: {'Y' if self.telegram.is_paused else 'N'}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"*Recent decisions*:\n{recent_text}"
            )
        except Exception as e:
            return f"❌ Status error: {str(e)[:200]}"
