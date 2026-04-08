"""
Main scheduler loop for the trading bot.

Default behavior:
- consume already-prepared data from live_update_service / pipeline outputs
- do not rerun the heavy pipeline unless explicitly configured
- evaluate AI decisions on schedule
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import BotMode, DataSourceMode, Settings
from cycle.state_extractor import CycleStateExtractor
from execution.order_manager import OrderManager
from strategy.ai_signal_engine import AISignalEngine, Signal, TradeAction
from strategy.rule_filter import RuleFilter
from telegram_bot.bot import TradingTelegramBot
from utils.decision_logger import DecisionLogger

try:
    from data.pipeline_runner import PipelineRunner
except Exception:
    PipelineRunner = None

logger = logging.getLogger("bot.scheduler")
KST = timezone(timedelta(hours=9))


@dataclass
class CandleCloseCondition:
    operator: str
    price: float
    created_at: datetime


class TradingScheduler:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.scheduler = AsyncIOScheduler()
        self.state_extractor = CycleStateExtractor(
            parquet_dir=settings.paths.parquet_dir,
            hierarchy_map_path=settings.paths.hierarchy_map_path,
            base_data_dir=settings.paths.base_data_dir,
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
        self.decision_log = DecisionLogger(settings.paths.log_dir)
        self.rule_filter = RuleFilter()
        self._last_market_state: dict = {}
        self._candle_close_condition: CandleCloseCondition | None = None
        self._last_execution_time: datetime | None = None  # RULE 6 쿨다운 추적
        self._cooldown_minutes: int = 30

        self.pipeline = None
        if settings.trading.data_source_mode == DataSourceMode.PIPELINE and PipelineRunner is not None:
            self.pipeline = PipelineRunner(settings.paths.analysis_project_root)

        self.telegram = None
        if settings.trading.use_telegram:
            self.telegram = TradingTelegramBot(
                token=settings.telegram.bot_token,
                chat_id=settings.telegram.chat_id,
                approval_timeout_minutes=settings.telegram.approval_timeout_minutes,
                auto_mode=settings.trading.mode == BotMode.AUTO,
            )
            self.telegram.on_approve = self._handle_approval
            self.telegram.on_close = self._handle_close
            self.telegram.on_status = self._handle_status
            self.telegram.on_analyze = self._handle_analyze
            self.telegram.on_set_close_condition = self._handle_set_close_condition
            self.telegram.on_clear_close_condition = self._handle_clear_close_condition
            self.telegram.on_close_condition_status = self._handle_close_condition_status

    async def start(self):
        if self.telegram:
            await self.telegram.initialize()

        self.scheduler.add_job(
            self._hourly_cycle,
            CronTrigger(
                minute=self.settings.scheduler.cron_minute,
                second=self.settings.scheduler.cron_second,
            ),
            id="hourly_cycle",
            name="Hourly signal evaluation",
            misfire_grace_time=120,
        )
        self.scheduler.start()

        mode_label = self.settings.trading.mode.value.upper()
        dry_label = "DRY RUN" if self.settings.trading.dry_run else "LIVE"
        decision_source = "AI+RULES" if self.ai_engine.is_configured() else "RULES ONLY"
        logger.info("Scheduler started [%s] [%s]", dry_label, mode_label)
        if self.telegram:
            await self.telegram.send_message(
                f"*Bot Started* [{dry_label}]\n"
                f"Mode: {mode_label}\n"
                f"Data source: {self.settings.trading.data_source_mode.value}\n"
                f"Schedule: XX:{self.settings.scheduler.cron_minute:02d}:{self.settings.scheduler.cron_second:02d}\n"
                f"Decision source: {decision_source}\n"
                f"AI: {self.settings.ai.model}"
            )

    async def stop(self):
        self.scheduler.shutdown(wait=False)
        if self.telegram:
            await self.telegram.send_message("Bot stopped.")
            await self.telegram.shutdown()

    async def _hourly_cycle(self):
        now = datetime.now(timezone.utc)
        logger.info("=== Scheduled evaluation: %s ===", now.strftime("%H:%M:%S UTC"))

        if self.telegram and self.telegram.is_paused:
            logger.info("Telegram pause is active. Skipping scheduled evaluation.")
            return

        try:
            if self.pipeline is not None:
                ok = await asyncio.get_event_loop().run_in_executor(None, self.pipeline.run)
                if not ok:
                    self.decision_log.log_error("Pipeline refresh failed before evaluation")
                    if self.telegram:
                        await self.telegram.send_message("Pipeline refresh failed. Using current available data.")

            pos_info = self._get_position_info()
            market_state = self.state_extractor.extract_current_state(current_position=pos_info, as_of=now)
            self._last_market_state = market_state
            logger.info(
                "Signal reference locked to closed 1h candle ending at %s",
                market_state.get("reference_time") or "unknown",
            )

            # ── RULE 6: 청산 후 쿨다운 ────────────────────────────────
            if await self._check_candle_close_exit(market_state, pos_info):
                return

            if self._last_execution_time is not None:
                elapsed = (now - self._last_execution_time).total_seconds() / 60
                if elapsed < self._cooldown_minutes:
                    remaining = int(self._cooldown_minutes - elapsed)
                    logger.info("RULE 6 쿨다운 중 — 잔여 %s분, 건너뜀", remaining)
                    self.decision_log.log_hold(
                        market_state,
                        {"action": "HOLD", "reasoning": f"RULE 6 쿨다운 중 (잔여 {remaining}분)"},
                        "COOLDOWN",
                    )
                    return

            # ── RULE 7: 일일 거래 한도 ────────────────────────────────
            daily_count = self.decision_log.get_today_execution_count()
            if daily_count >= 3:
                logger.info("RULE 7 일일 한도 도달 (%s건) — 건너뜀", daily_count)
                if self.telegram:
                    await self.telegram.send_message(f"RULE 7: 오늘 거래 {daily_count}건 한도 도달. 신규 진입 중단.")
                return

            # ── 룰 기반 Pre-Filter ────────────────────────────────────
            filter_result = self.rule_filter.check(market_state)
            if not filter_result.allowed:
                logger.info("RuleFilter BLOCK: %s", filter_result.block_reason)
                self.decision_log.log_hold(
                    market_state,
                    {"action": "HOLD", "reasoning": filter_result.block_reason},
                    "RULE_FILTER",
                )
                return

            signal = self._build_signal(market_state, pos_info)

            # ── Post-AI 방향 검증 + 사이즈 배율 적용 ──────────────────
            signal = self.rule_filter.validate_signal(signal, filter_result)

            mode_label = self.settings.trading.mode.value.upper()

            if not signal.is_actionable():
                self.decision_log.log_hold(market_state, signal.to_dict(), mode_label)
                logger.info("HOLD: %s", signal.reasoning[:160])
                if self.settings.trading.mode == BotMode.ANALYZE and self.telegram:
                    await self.telegram.send_message(
                        f"*Analyze Mode*\nDecision: `{signal.action.value}`\nReasoning: {signal.reasoning}"
                    )
                return

            self.decision_log.log_signal(market_state, signal.to_dict(), mode_label)
            logger.info("Actionable signal: %s", signal.action.value)

            if self.settings.trading.mode == BotMode.ANALYZE:
                if self.telegram:
                    await self.telegram.send_message(
                        f"*Analyze Mode Signal*\nAction: `{signal.action.value}`\nReasoning: {signal.reasoning}"
                    )
                return

            if self.telegram:
                await self.telegram.send_signal_alert(signal)
            elif self.settings.trading.mode == BotMode.AUTO:
                await self._handle_approval(signal)

        except Exception as exc:
            logger.error("Scheduled evaluation failed: %s", exc, exc_info=True)
            self.decision_log.log_error(str(exc))
            if self.telegram:
                await self.telegram.send_message(f"Bot error: {str(exc)[:300]}")

    def _get_position_info(self):
        pos = self.order_manager.get_position()
        if not pos:
            return None
        notional = max(pos.get("notional", 1), 1)
        return {
            "side": pos["side"],
            "entry_price": pos["entry_price"],
            "size_usdt": pos.get("notional", 0),
            "unrealized_pnl_pct": pos["unrealized_pnl"] / notional * 100,
        }

    def _build_signal(self, market_state: dict, current_position: dict | None) -> Signal:
        if not self.ai_engine.is_configured():
            logger.info("AI key is not configured. Using rule-only signal engine.")
            return self.rule_filter.generate_signal(market_state, current_position)

        signal = self.ai_engine.evaluate(market_state)
        if signal.reasoning.startswith("AI evaluation failed:"):
            logger.warning("AI evaluation failed. Falling back to rule-only signal engine.")
            fallback = self.rule_filter.generate_signal(market_state, current_position)
            fallback.alerts = list(fallback.alerts) + ["AI failed, rule-only fallback used"]
            return fallback
        return signal

    async def _check_candle_close_exit(self, market_state: dict, current_position: dict | None) -> bool:
        condition = self._candle_close_condition
        if condition is None or not current_position:
            return False

        close_price = market_state.get("price")
        if close_price is None:
            return False

        triggered = (
            close_price >= condition.price
            if condition.operator == ">="
            else close_price <= condition.price
        )
        if not triggered:
            return False

        side = str(current_position.get("side", "")).upper()
        result = self.order_manager.close_position(side)
        signal_dict = {
            "action": f"CLOSE_{side}",
            "reasoning": (
                f"Closed by candle-close condition: 1h close {close_price} "
                f"{condition.operator} {condition.price}"
            ),
        }
        self.decision_log.log_execution(
            signal_dict,
            {
                "success": result.success,
                "side": result.side,
                "size": result.size,
                "price": result.price,
                "error": result.error,
            },
            "TELEGRAM_RULE",
            "candle_close_trigger",
        )
        if result.success:
            self._last_execution_time = datetime.now(timezone.utc)
            if self.telegram:
                await self.telegram.send_message(
                    f"Candle-close exit triggered.\n1h close: {close_price}\nCondition: {condition.operator} {condition.price}"
                )
                await self.telegram.send_order_result(result)
            self._candle_close_condition = None
        else:
            if self.telegram:
                await self.telegram.send_message(
                    f"Candle-close exit triggered but close failed: {result.error or 'unknown error'}"
                )
        return True

    async def _handle_set_close_condition(self, operator: str, price: float) -> str:
        self._candle_close_condition = CandleCloseCondition(
            operator=operator,
            price=price,
            created_at=datetime.now(timezone.utc),
        )
        return (
            "Candle-close exit armed.\n"
            f"Condition: close {operator} {price}\n"
            "It will trigger on the next evaluated closed 1h candle."
        )

    async def _handle_clear_close_condition(self) -> str:
        if self._candle_close_condition is None:
            return "No candle-close exit condition is active."
        self._candle_close_condition = None
        return "Candle-close exit condition cleared."

    async def _handle_close_condition_status(self) -> str:
        if self._candle_close_condition is None:
            return "No candle-close exit condition is active."
        condition = self._candle_close_condition
        return (
            "Active candle-close exit condition\n"
            f"- Condition: close {condition.operator} {condition.price}\n"
            f"- Created: {condition.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    async def _handle_approval(self, signal: Signal):
        mode_label = self.settings.trading.mode.value.upper()
        user_response = "auto_executed" if self.settings.trading.mode == BotMode.AUTO else "approved"
        logger.info("Executing signal: %s [%s]", signal.action.value, user_response)

        try:
            result = None
            if signal.action == TradeAction.ENTER_LONG:
                result = self.order_manager.enter_position("LONG", signal.size_pct, signal.sl_pct, signal.tp_pct)
            elif signal.action == TradeAction.ENTER_SHORT:
                result = self.order_manager.enter_position("SHORT", signal.size_pct, signal.sl_pct, signal.tp_pct)
            elif signal.action == TradeAction.CLOSE_LONG:
                result = self.order_manager.close_position("LONG")
            elif signal.action == TradeAction.CLOSE_SHORT:
                result = self.order_manager.close_position("SHORT")
            elif signal.action == TradeAction.REVERSE_TO_LONG:
                self.order_manager.close_position("SHORT")
                result = self.order_manager.enter_position("LONG", signal.size_pct, signal.sl_pct, signal.tp_pct)
            elif signal.action == TradeAction.REVERSE_TO_SHORT:
                self.order_manager.close_position("LONG")
                result = self.order_manager.enter_position("SHORT", signal.size_pct, signal.sl_pct, signal.tp_pct)

            if result:
                self.decision_log.log_execution(
                    signal.to_dict(),
                    {
                        "success": result.success,
                        "side": result.side,
                        "size": result.size,
                        "price": result.price,
                        "error": result.error,
                    },
                    mode_label,
                    user_response,
                )
                if result.success:
                    # RULE 6: 청산·진입 후 쿨다운 타이머 시작
                    self._last_execution_time = datetime.now(timezone.utc)
                if self.telegram:
                    await self.telegram.send_order_result(result)
        except Exception as exc:
            logger.error("Execution failed: %s", exc, exc_info=True)
            self.decision_log.log_error(str(exc), signal.to_dict())
            if self.telegram:
                await self.telegram.send_message(f"Execution error: {str(exc)[:300]}")

    async def _handle_close(self, side: str):
        try:
            pos = self.order_manager.get_position()
            if not pos:
                if self.telegram:
                    await self.telegram.send_message("No open position.")
                return

            result = self.order_manager.close_position(pos["side"])
            self._candle_close_condition = None
            self.decision_log.log_execution(
                {"action": "EMERGENCY_CLOSE"},
                {"success": result.success, "side": result.side, "price": result.price, "error": result.error},
                "MANUAL",
                "emergency_close",
            )
            if self.telegram:
                await self.telegram.send_order_result(result)
        except Exception as exc:
            if self.telegram:
                await self.telegram.send_message(f"Close error: {str(exc)[:300]}")

    async def _handle_status(self) -> str:
        try:
            market_state = self.state_extractor.extract_current_state(
                current_position=self._get_position_info(),
                as_of=datetime.now(timezone.utc),
            )
            self._last_market_state = market_state
            pos = self.order_manager.get_position()
            account = self.order_manager.get_account_snapshot()
            chain = market_state.get("chain", {})
            price = market_state.get("price", "?")
            reference_time = self._format_dual_time(market_state.get("reference_time"))
            evaluation_time = self._format_dual_time(market_state.get("evaluation_time"))
            close_condition_text = "none"
            if self._candle_close_condition is not None:
                close_condition_text = (
                    f"{self._candle_close_condition.operator} {self._candle_close_condition.price}"
                )
            position_text = "포지션 없음"
            position_text_en = "No open position."
            if pos:
                pnl = pos.get("unrealized_pnl", 0)
                total_usdt = max(account.get("total_usdt", 0), 0)
                notional = abs(float(pos.get("notional", 0) or 0))
                exposure_pct = (notional / total_usdt * 100) if total_usdt > 0 else 0.0
                position_text_en = (
                    f"- Side: {pos.get('side', '-')}\n"
                    f"- Size: {pos.get('size', 0):,.4f} BTC\n"
                    f"- Entry: {pos.get('entry_price', 0):,.2f}\n"
                    f"- Notional: {notional:,.2f} USDT\n"
                    f"- Exposure: {exposure_pct:.2f}%\n"
                    f"- Unrealized PnL: {pnl:+,.2f} USDT\n"
                    f"- Leverage: {pos.get('leverage', self.settings.binance.leverage)}x\n"
                    f"- Mark: {pos.get('mark_price', 0):,.2f}\n"
                    f"- Liquidation: {pos.get('liquidation_price', 0):,.2f}"
                )

            timeframes = market_state.get("timeframes", {})
            cycle_lines_en = []
            for tf in ["1w", "1d", "4h", "1h"]:
                tf_state = timeframes.get(tf)
                if not tf_state:
                    cycle_lines_en.append(f"- {tf}: no data")
                    continue
                cycle_lines_en.append(
                    f"- {tf}: {tf_state.get('cycle_type', '?')} / duration {tf_state.get('duration', '?')} / "
                    f"pos {tf_state.get('position_pct', '?')}"
                )

            recent = self.decision_log.get_recent_decisions(3)
            recent_text_en = "\n".join(
                f"- {record.get('timestamp', '')[-8:]} {record.get('signal', {}).get('action', record.get('event', ''))}"
                for record in recent
            ) or "- none"

            return (
                "Current Status\n\n"
                "[Market]\n"
                f"- Price: {price}\n"
                f"- Reference 1h candle: {reference_time}\n"
                f"- Checked at: {evaluation_time}\n"
                f"- Combo: {chain.get('combo', '?')}\n"
                f"- n_up: {chain.get('n_up', '?')}\n"
                f"- 4h-1h aligned: {'Y' if chain.get('alignment_4h_1h') else 'N'}\n"
                f"- Mode: {self.settings.trading.mode.value.upper()}\n"
                f"- Order mode: {'DRY RUN' if self.settings.trading.dry_run else 'LIVE'}\n"
                f"- Candle-close exit: {close_condition_text}\n\n"
                "[Cycle]\n"
                f"{chr(10).join(cycle_lines_en)}\n\n"
                "[Position]\n"
                f"{position_text_en}\n\n"
                "[Account]\n"
                f"- Total USDT: {account.get('total_usdt', 0):,.2f}\n"
                f"- Free USDT: {account.get('free_usdt', 0):,.2f}\n"
                f"- Used USDT: {account.get('used_usdt', 0):,.2f}\n"
                f"- Source: {account.get('source', '-')}\n\n"
                "[Recent]\n"
                f"{recent_text_en}"
            )

            if pos:
                pnl = pos.get("unrealized_pnl", 0)
                total_usdt = max(account.get("total_usdt", 0), 0)
                notional = abs(float(pos.get("notional", 0) or 0))
                exposure_pct = (notional / total_usdt * 100) if total_usdt > 0 else 0.0
                position_text = (
                    f"- 방향: {pos.get('side', '-')}\n"
                    f"- 수량: {pos.get('size', 0):,.4f} BTC\n"
                    f"- 진입가: {pos.get('entry_price', 0):,.2f}\n"
                    f"- 현재 노셔널: {notional:,.2f} USDT\n"
                    f"- 계좌 대비 비중: {exposure_pct:.2f}%\n"
                    f"- 미실현 손익: {pnl:+,.2f} USDT\n"
                    f"- 레버리지: {pos.get('leverage', self.settings.binance.leverage)}x\n"
                    f"- 마크가: {pos.get('mark_price', 0):,.2f}\n"
                    f"- 청산가: {pos.get('liquidation_price', 0):,.2f}"
                )

            chain = market_state.get("chain", {})
            price = market_state.get("price", "?")
            reference_time = self._format_dual_time(market_state.get("reference_time"))
            evaluation_time = self._format_dual_time(market_state.get("evaluation_time"))
            timeframes = market_state.get("timeframes", {})
            cycle_lines = []
            for tf in ["1w", "1d", "4h", "1h"]:
                tf_state = timeframes.get(tf)
                if not tf_state:
                    cycle_lines.append(f"- {tf}: 없음")
                    continue
                cycle_lines.append(
                    f"- {tf}: {tf_state.get('cycle_type', '?')} / duration {tf_state.get('duration', '?')} / "
                    f"pos {tf_state.get('position_pct', '?')}"
                )
            recent = self.decision_log.get_recent_decisions(3)
            recent_text = "\n".join(
                f"- {record.get('timestamp', '')[-8:]} {record.get('signal', {}).get('action', record.get('event', ''))}"
                for record in recent
            ) or "- none"

            return (
                "Current Status\n\n"
                "[시장정보]\n"
                f"- 현재가: {price}\n"
                f"- 조합: {chain.get('combo', '?')}\n"
                f"- 상승 개수: {chain.get('n_up', '?')}\n"
                f"- 4h-1h 정렬: {'Y' if chain.get('alignment_4h_1h') else 'N'}\n"
                f"- 모드: {self.settings.trading.mode.value.upper()}\n"
                f"- 주문 상태: {'DRY RUN' if self.settings.trading.dry_run else 'LIVE'}\n\n"
                "[내기준 사이클정보]\n"
                f"{chr(10).join(cycle_lines)}\n\n"
                "[내 포지션]\n"
                f"{position_text}\n\n"
                "[계좌정보]\n"
                f"- 총 USDT: {account.get('total_usdt', 0):,.2f}\n"
                f"- 사용 가능 USDT: {account.get('free_usdt', 0):,.2f}\n"
                f"- 사용 중 USDT: {account.get('used_usdt', 0):,.2f}\n"
                f"- 조회 출처: {account.get('source', '-')}\n\n"
                "[최근 이벤트]\n"
                f"{recent_text}"
            )
        except Exception as exc:
            return f"Status error: {str(exc)[:300]}"

    def _format_dual_time(self, value: str | None) -> str:
        if not value:
            return "unknown"
        timestamp = self._parse_timestamp(value)
        if timestamp is None:
            return str(value)
        utc_text = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        kst_text = timestamp.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")
        return f"{utc_text} / {kst_text}"

    def _parse_timestamp(self, value: str) -> datetime | None:
        try:
            text = value.strip().replace("Z", "+00:00")
            timestamp = datetime.fromisoformat(text)
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc)
        except Exception:
            return None

    async def _handle_analyze(self, user_request: str) -> str:
        try:
            market_state = self.state_extractor.extract_current_state(
                current_position=self._get_position_info(),
                as_of=datetime.now(timezone.utc),
            )
            self._last_market_state = market_state
            return self.ai_engine.analyze(market_state, user_request)
        except Exception as exc:
            logger.error("On-demand analysis failed: %s", exc, exc_info=True)
            return f"On-demand analysis failed: {exc}"
