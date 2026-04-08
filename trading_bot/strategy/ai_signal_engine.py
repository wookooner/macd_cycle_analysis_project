"""
AI-based decision engine.

The model receives a structured market_state and a rules document, then
returns a normalized trading decision as JSON.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger("bot.ai_signal")


class TradeAction(str, Enum):
    HOLD = "HOLD"
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    REVERSE_TO_LONG = "REVERSE_TO_LONG"
    REVERSE_TO_SHORT = "REVERSE_TO_SHORT"


@dataclass
class Signal:
    action: TradeAction = TradeAction.HOLD
    direction: str = "NEUTRAL"
    size_pct: int = 0
    confidence: float = 0.0
    reasoning: str = ""
    sl_pct: float = 2.0
    tp_pct: float = 4.0
    alerts: list[str] = field(default_factory=list)
    raw_response: str = ""
    latency_ms: int = 0
    timestamp: str = ""

    def is_actionable(self) -> bool:
        return self.action != TradeAction.HOLD

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload


class AISignalEngine:
    VALID_ACTIONS = {action.value for action in TradeAction}

    def __init__(self, api_key: str, model: str, rules_doc_path: Path, max_tokens: int = 1024, temperature: float = 0.0):
        self.api_key = api_key
        self.model = model
        self.rules_doc_path = rules_doc_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError as exc:
                raise RuntimeError("anthropic package is required: pip install anthropic") from exc
        return self._client

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def evaluate(self, market_state: dict) -> Signal:
        from config.ai_prompt import build_system_prompt, build_user_message

        system_prompt = build_system_prompt(self.rules_doc_path)
        user_message = build_user_message(market_state)
        started_at = time.time()

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            raw_text = response.content[0].text.strip()
            signal = self._parse_response(raw_text)
            signal.raw_response = raw_text
            signal.latency_ms = int((time.time() - started_at) * 1000)
            signal.timestamp = market_state.get("timestamp", "")
            logger.info(
                "AI signal generated: action=%s confidence=%.2f size=%s latency=%sms",
                signal.action.value,
                signal.confidence,
                signal.size_pct,
                signal.latency_ms,
            )
            return signal
        except Exception as exc:
            logger.error("AI evaluation failed: %s", exc, exc_info=True)
            return Signal(
                action=TradeAction.HOLD,
                reasoning=f"AI evaluation failed: {exc}",
                alerts=["AI evaluation failed, defaulting to HOLD"],
                timestamp=market_state.get("timestamp", ""),
            )

    def analyze(self, market_state: dict, user_request: str = "") -> str:
        from config.ai_prompt import build_analysis_system_prompt, build_analysis_user_message

        system_prompt = build_analysis_system_prompt(self.rules_doc_path)
        user_message = build_analysis_user_message(market_state, user_request)

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.error("AI analysis failed: %s", exc, exc_info=True)
            return f"AI analysis failed: {exc}"

    def _parse_response(self, raw_text: str) -> Signal:
        try:
            text = raw_text.strip()
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()

            data = json.loads(text)
            action_str = str(data.get("action", "HOLD")).upper()
            if action_str not in self.VALID_ACTIONS:
                action_str = TradeAction.HOLD.value

            return Signal(
                action=TradeAction(action_str),
                direction=str(data.get("direction", "NEUTRAL")),
                size_pct=max(0, min(100, int(data.get("size_pct", 0)))),
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
                reasoning=str(data.get("reasoning", "")),
                sl_pct=max(0.0, min(10.0, float(data.get("sl_pct", 2.0)))),
                tp_pct=max(0.0, min(20.0, float(data.get("tp_pct", 4.0)))),
                alerts=[str(item) for item in data.get("alerts", [])],
            )
        except Exception as exc:
            logger.error("Failed to parse AI response: %s", exc, exc_info=True)
            return Signal(
                action=TradeAction.HOLD,
                reasoning=f"Failed to parse AI response: {exc}",
                alerts=["Invalid AI response, defaulting to HOLD"],
                raw_response=raw_text,
            )
