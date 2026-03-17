"""
trading_bot/telegram_bot/bot.py
================================
Telegram 봇: 자동/승인 모드 전환 + 알림 + 명령어

모드:
  /manual — 승인 모드 (신호 → 버튼 → 승인 시 실행)
  /auto   — 자동 모드 (신호 → 즉시 실행 + 알림만 발송)
"""

import asyncio
import logging
from datetime import timedelta
from typing import Optional, Callable, Awaitable

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes
)

from strategy.ai_signal_engine import Signal, TradeAction

logger = logging.getLogger("bot.telegram")


class TradingTelegramBot:
    def __init__(self, token: str, chat_id: str,
                 approval_timeout_minutes: int = 30):
        self.token = token
        self.chat_id = chat_id
        self.approval_timeout = timedelta(minutes=approval_timeout_minutes)
        self.app: Optional[Application] = None
        self.bot: Optional[Bot] = None

        # 외부 콜백
        self.on_approve: Optional[Callable[[Signal], Awaitable]] = None
        self.on_close: Optional[Callable[[str], Awaitable]] = None
        self.on_status: Optional[Callable[[], Awaitable[str]]] = None

        self._pending_signals: dict[str, Signal] = {}
        self._paused = False
        self._auto_mode = False  # False=승인모드, True=자동모드

    async def initialize(self):
        self.app = Application.builder().token(self.token).build()
        self.bot = self.app.bot

        # 명령어 등록
        commands = {
            "status": self._cmd_status,
            "close": self._cmd_close,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "auto": self._cmd_auto,
            "manual": self._cmd_manual,
            "mode": self._cmd_mode,
            "help": self._cmd_help,
        }
        for cmd, handler in commands.items():
            self.app.add_handler(CommandHandler(cmd, handler))

        self.app.add_handler(CallbackQueryHandler(self._callback_handler))

        await self.app.initialize()
        await self.app.start()
        asyncio.create_task(self._start_polling())
        logger.info("Telegram bot initialized")

    async def _start_polling(self):
        try:
            await self.app.updater.start_polling(drop_pending_updates=True)
        except Exception as e:
            logger.error(f"Polling error: {e}")

    async def shutdown(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_auto_mode(self) -> bool:
        return self._auto_mode

    @property
    def mode_label(self) -> str:
        return "AUTO" if self._auto_mode else "MANUAL"

    # ── 신호 알림 ──────────────────────────────────────

    async def send_signal_alert(self, signal: Signal) -> Optional[str]:
        """
        신호 알림 발송.
        - 자동 모드: 즉시 실행 알림 (버튼 없음)
        - 승인 모드: 승인/거부 버튼 포함
        """
        if self._paused:
            logger.info("Paused — skip alert")
            return None

        emoji = {
            TradeAction.ENTER_LONG: "🟢", TradeAction.ENTER_SHORT: "🔴",
            TradeAction.CLOSE_LONG: "⚪", TradeAction.CLOSE_SHORT: "⚪",
            TradeAction.REVERSE_TO_LONG: "🔄🟢",
            TradeAction.REVERSE_TO_SHORT: "🔄🔴",
        }.get(signal.action, "⚪")

        mode_tag = "🤖 AUTO" if self._auto_mode else "👤 MANUAL"

        text = (
            f"{emoji} *매매 신호* [{mode_tag}]\n"
            f"━━━━━━━━━━━━━━━\n"
            f"*Action*: `{signal.action.value}`\n"
            f"*방향*: {signal.direction}\n"
            f"*사이즈*: {signal.size_pct}%\n"
            f"*신뢰도*: {signal.confidence:.1%}\n"
            f"*SL*: {signal.sl_pct}% | *TP*: {signal.tp_pct}%\n"
            f"━━━━━━━━━━━━━━━\n"
            f"*근거*: {signal.reasoning}\n"
        )
        if signal.alerts:
            text += f"\n⚠️ {' | '.join(signal.alerts)}\n"

        try:
            if self._auto_mode:
                # 자동 모드: 즉시 실행, 알림만 발송
                text += "\n🤖 *자동 실행 중...*"
                await self.bot.send_message(
                    self.chat_id, text, parse_mode="Markdown",
                )
                # 즉시 실행
                if self.on_approve:
                    await self.on_approve(signal)
                return "auto"

            else:
                # 승인 모드: 버튼 포함
                text += f"\n⏰ 승인 대기: {self.approval_timeout.seconds // 60}분"
                kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton("✅ 승인", callback_data="approve"),
                    InlineKeyboardButton("❌ 거부", callback_data="reject"),
                ]])
                msg = await self.bot.send_message(
                    self.chat_id, text, parse_mode="Markdown", reply_markup=kb,
                )
                mid = str(msg.message_id)
                self._pending_signals[mid] = signal
                asyncio.create_task(self._approval_timeout(mid))
                return mid

        except Exception as e:
            logger.error(f"Alert failed: {e}")
            return None

    async def send_message(self, text: str):
        try:
            await self.bot.send_message(self.chat_id, text, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Send failed: {e}")

    async def send_order_result(self, result):
        await self.send_message(f"📋 *주문 결과*\n{str(result)}")

    # ── 콜백 ──────────────────────────────────────────

    async def _callback_handler(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        mid = str(query.message.message_id)
        signal = self._pending_signals.pop(mid, None)
        if not signal:
            await query.edit_message_text("⏰ 만료된 신호입니다.")
            return

        if query.data == "approve":
            await query.edit_message_text(f"✅ *승인* — {signal.action.value} 실행 중...")
            if self.on_approve:
                await self.on_approve(signal)
        else:
            await query.edit_message_text("❌ *거부* — 취소됨")
            logger.info(f"Rejected: {signal.action.value}")

    async def _approval_timeout(self, mid: str):
        await asyncio.sleep(self.approval_timeout.total_seconds())
        signal = self._pending_signals.pop(mid, None)
        if signal:
            try:
                await self.bot.edit_message_text(
                    self.chat_id, int(mid),
                    text=f"⏰ *타임아웃* — {signal.action.value} 자동 취소",
                )
            except Exception:
                pass

    # ── 명령어 ────────────────────────────────────────

    async def _cmd_auto(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._auto_mode = True
        await update.message.reply_text(
            "🤖 *자동 모드 활성화*\n"
            "AI 신호 발생 시 승인 없이 즉시 실행됩니다.\n"
            "/manual 로 승인 모드로 전환",
            parse_mode="Markdown",
        )
        logger.info("Mode: AUTO")

    async def _cmd_manual(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._auto_mode = False
        await update.message.reply_text(
            "👤 *승인 모드 활성화*\n"
            "AI 신호 발생 시 승인 버튼을 눌러야 실행됩니다.\n"
            "/auto 로 자동 모드로 전환",
            parse_mode="Markdown",
        )
        logger.info("Mode: MANUAL")

    async def _cmd_mode(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        mode = "🤖 자동 (AUTO)" if self._auto_mode else "👤 승인 (MANUAL)"
        await update.message.reply_text(
            f"현재 모드: *{mode}*",
            parse_mode="Markdown",
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = await self.on_status() if self.on_status else "상태 조회 미연결"
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_close(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.on_close:
            await update.message.reply_text("⚡ 청산 실행 중...")
            await self.on_close("ALL")
        else:
            await update.message.reply_text("청산 기능 미연결")

    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._paused = True
        await update.message.reply_text("⏸ 봇 일시정지. /resume 으로 재개")

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._paused = False
        await update.message.reply_text("▶️ 봇 재개됨")

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 *BTC Cycle Trading Bot*\n\n"
            "*모드 전환*\n"
            "/auto — 자동 모드 (즉시 실행)\n"
            "/manual — 승인 모드 (버튼 승인)\n"
            "/mode — 현재 모드 확인\n\n"
            "*제어*\n"
            "/status — 현재 상태\n"
            "/close — 긴급 청산\n"
            "/pause — 일시정지\n"
            "/resume — 재개\n"
            "/help — 도움말",
            parse_mode="Markdown",
        )
