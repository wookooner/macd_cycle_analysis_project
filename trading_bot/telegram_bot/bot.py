"""
Telegram control surface for the trading bot.

Supports:
- analyze/approve/auto mode visibility
- pause/resume
- status / emergency close
- approval workflow for signals
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import timedelta
from typing import Awaitable, Callable, Optional

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from strategy.ai_signal_engine import Signal

logger = logging.getLogger("bot.telegram")


class TradingTelegramBot:
    def __init__(
        self,
        token: str,
        chat_id: str,
        approval_timeout_minutes: int = 30,
        auto_mode: bool = False,
    ):
        self.token = token
        self.chat_id = chat_id
        self.approval_timeout = timedelta(minutes=approval_timeout_minutes)
        self._auto_mode = auto_mode
        self._paused = False
        self._pending_signals: dict[str, Signal] = {}

        self.app: Optional[Application] = None
        self.bot: Optional[Bot] = None

        self.on_approve: Optional[Callable[[Signal], Awaitable[None]]] = None
        self.on_close: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_status: Optional[Callable[[], Awaitable[str]]] = None
        self.on_analyze: Optional[Callable[[str], Awaitable[str]]] = None
        self.on_set_close_condition: Optional[Callable[[str, float], Awaitable[str]]] = None
        self.on_clear_close_condition: Optional[Callable[[], Awaitable[str]]] = None
        self.on_close_condition_status: Optional[Callable[[], Awaitable[str]]] = None

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_auto_mode(self) -> bool:
        return self._auto_mode

    @property
    def mode_label(self) -> str:
        return "AUTO" if self._auto_mode else "MANUAL"

    async def initialize(self):
        self.app = Application.builder().token(self.token).build()
        self.bot = self.app.bot

        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("close", self._cmd_close))
        self.app.add_handler(CommandHandler("pause", self._cmd_pause))
        self.app.add_handler(CommandHandler("resume", self._cmd_resume))
        self.app.add_handler(CommandHandler("auto", self._cmd_auto))
        self.app.add_handler(CommandHandler("manual", self._cmd_manual))
        self.app.add_handler(CommandHandler("mode", self._cmd_mode))
        self.app.add_handler(CommandHandler("analyze", self._cmd_analyze))
        self.app.add_handler(CommandHandler("signal", self._cmd_signal))
        self.app.add_handler(CommandHandler("closeif", self._cmd_closeif))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        self.app.add_handler(CallbackQueryHandler(self._callback_handler))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._text_handler))

        await self.app.initialize()
        await self.app.start()
        asyncio.create_task(self._start_polling())
        logger.info("Telegram bot initialized")

    async def _start_polling(self):
        try:
            await self.app.updater.start_polling(drop_pending_updates=True)
        except Exception as exc:
            logger.error("Telegram polling failed: %s", exc, exc_info=True)

    async def shutdown(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

    async def send_message(self, text: str):
        if not self.bot:
            return
        try:
            await self.bot.send_message(self.chat_id, text, parse_mode="Markdown")
        except Exception as exc:
            logger.warning("Telegram markdown send failed, retrying as plain text: %s", exc)
            try:
                await self.bot.send_message(self.chat_id, text)
            except Exception as retry_exc:
                logger.error("Telegram send failed: %s", retry_exc, exc_info=True)

    async def _reply_text(self, update: Update, text: str, markdown: bool = False):
        if not update.message:
            return
        try:
            if markdown:
                await update.message.reply_text(text, parse_mode="Markdown")
            else:
                await update.message.reply_text(text)
        except Exception as exc:
            if markdown:
                logger.warning("Telegram markdown reply failed, retrying as plain text: %s", exc)
                try:
                    await update.message.reply_text(text)
                    return
                except Exception as retry_exc:
                    logger.error("Telegram reply failed: %s", retry_exc, exc_info=True)
                    return
            logger.error("Telegram reply failed: %s", exc, exc_info=True)

    async def send_order_result(self, result):
        await self.send_message(f"*Order Result*\n`{result}`")

    async def send_signal_alert(self, signal: Signal) -> Optional[str]:
        if self._paused or not self.bot:
            return None

        text = (
            f"*Trading Signal* [{self.mode_label}]\n"
            f"*Action*: `{signal.action.value}`\n"
            f"*Direction*: {signal.direction}\n"
            f"*Size*: {signal.size_pct}%\n"
            f"*Confidence*: {signal.confidence:.1%}\n"
            f"*SL*: {signal.sl_pct}%\n"
            f"*TP*: {signal.tp_pct}%\n"
            f"*Reasoning*: {signal.reasoning}"
        )
        if signal.alerts:
            text += "\n*Alerts*: " + " | ".join(signal.alerts)

        if self._auto_mode:
            await self.send_message(text + "\n\n*Auto execution started.*")
            if self.on_approve:
                await self.on_approve(signal)
            return "auto"

        keyboard = InlineKeyboardMarkup(
            [[
                InlineKeyboardButton("Approve", callback_data="approve"),
                InlineKeyboardButton("Reject", callback_data="reject"),
            ]]
        )
        message = await self.bot.send_message(
            self.chat_id,
            text + f"\n\nApproval timeout: {int(self.approval_timeout.total_seconds() // 60)} min",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
        message_id = str(message.message_id)
        self._pending_signals[message_id] = signal
        asyncio.create_task(self._approval_timeout(message_id))
        return message_id

    async def _callback_handler(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        message_id = str(query.message.message_id)
        signal = self._pending_signals.pop(message_id, None)

        if not signal:
            await query.edit_message_text("This signal is no longer available.")
            return

        if query.data == "approve":
            await query.edit_message_text(f"Approved: {signal.action.value}")
            if self.on_approve:
                await self.on_approve(signal)
        else:
            await query.edit_message_text("Rejected.")
            logger.info("Signal rejected: %s", signal.action.value)

    async def _approval_timeout(self, message_id: str):
        await asyncio.sleep(self.approval_timeout.total_seconds())
        signal = self._pending_signals.pop(message_id, None)
        if signal and self.bot:
            try:
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=int(message_id),
                    text=f"Approval timed out: {signal.action.value}",
                )
            except Exception:
                pass

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = await self.on_status() if self.on_status else "Status handler is not connected."
        await self._reply_text(update, text)

    async def _cmd_close(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.on_close:
            await self._reply_text(update, "Emergency close requested.")
            await self.on_close("ALL")
        else:
            await self._reply_text(update, "Close handler is not connected.")

    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._paused = True
        await self._reply_text(update, "Bot paused.")

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._paused = False
        await self._reply_text(update, "Bot resumed.")

    async def _cmd_auto(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._auto_mode = True
        await self._reply_text(update, "Auto mode enabled.")

    async def _cmd_manual(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._auto_mode = False
        await self._reply_text(update, "Manual approval mode enabled.")

    async def _cmd_mode(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._reply_text(update, f"Current mode: {self.mode_label}")

    async def _cmd_analyze(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self.on_analyze:
            await self._reply_text(update, "Analyze handler is not connected.")
            return
        question = " ".join(ctx.args).strip()
        await self._reply_text(update, "Analyzing current state...")
        answer = await self.on_analyze(question)
        await self._reply_text(update, answer)

    async def _cmd_signal(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self.on_analyze:
            await self._reply_text(update, "Signal analysis handler is not connected.")
            return
        await self._reply_text(update, "Checking current signal...")
        answer = await self.on_analyze("Give the current cycle judgment, likely action, and the main reasons in concise form.")
        await self._reply_text(update, answer)

    async def _cmd_closeif(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        raw = " ".join(ctx.args).strip()
        if not raw:
            await self._reply_text(
                update,
                "Usage: /closeif >= 68000 | /closeif <= 65000 | /closeif status | /closeif clear",
            )
            return

        lowered = raw.lower()
        if lowered == "status":
            if not self.on_close_condition_status:
                await self._reply_text(update, "Close condition status handler is not connected.")
                return
            await self._reply_text(update, await self.on_close_condition_status())
            return

        if lowered == "clear":
            if not self.on_clear_close_condition:
                await self._reply_text(update, "Clear close condition handler is not connected.")
                return
            await self._reply_text(update, await self.on_clear_close_condition())
            return

        match = re.match(r"^(>=|<=|above|below)\s*([0-9]+(?:\.[0-9]+)?)$", raw, re.IGNORECASE)
        if not match:
            await self._reply_text(
                update,
                "Usage: /closeif >= 68000 | /closeif <= 65000",
            )
            return

        operator, price_text = match.groups()
        if operator.lower() == "above":
            operator = ">="
        elif operator.lower() == "below":
            operator = "<="

        if not self.on_set_close_condition:
            await self._reply_text(update, "Set close condition handler is not connected.")
            return

        await self._reply_text(update, await self.on_set_close_condition(operator, float(price_text)))

    async def _text_handler(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self.on_analyze:
            await self._reply_text(update, "Analyze handler is not connected.")
            return
        text = (update.message.text or "").strip()
        if not text:
            return
        await self._reply_text(update, "Thinking...")
        answer = await self.on_analyze(text)
        await self._reply_text(update, answer)

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._reply_text(
            update,
            "/status - current bot status\n"
            "/analyze [question] - analyze the latest market_state with the rule document\n"
            "/signal - summarize the current cycle judgment and likely action\n"
            "/closeif >= PRICE - close if the next closed 1h candle closes at or above PRICE\n"
            "/closeif <= PRICE - close if the next closed 1h candle closes at or below PRICE\n"
            "/closeif status - show the active candle-close condition\n"
            "/closeif clear - clear the active candle-close condition\n"
            "/auto - enable auto execution\n"
            "/manual - enable manual approval mode\n"
            "/pause - pause alerts and execution\n"
            "/resume - resume alerts and execution\n"
            "/close - emergency close current position\n"
            "/mode - show current mode\n"
            "You can also send a plain text question to chat with the bot.",
        )
