"""
trading_bot/strategy/ai_signal_engine.py
=========================================
Claude API를 호출하여 매매 신호 및 포지션 사이징 결정.

핵심: 매 호출 시 trading_rules.md를 읽어서 시스템 프롬프트에 포함.
"""

import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

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
    """AI가 생성한 매매 신호"""
    action: TradeAction = TradeAction.HOLD
    direction: str = "NEUTRAL"
    size_pct: int = 0
    confidence: float = 0.0
    reasoning: str = ""
    sl_pct: float = 2.0
    tp_pct: float = 4.0
    alerts: list = field(default_factory=list)
    raw_response: str = ""
    latency_ms: int = 0
    timestamp: str = ""

    def is_actionable(self) -> bool:
        return self.action != TradeAction.HOLD

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d


class AISignalEngine:
    """Claude API 호출 → 매매 신호 생성"""

    VALID_ACTIONS = {a.value for a in TradeAction}

    def __init__(self, api_key: str, model: str,
                 rules_doc_path: Path,
                 max_tokens: int = 1024,
                 temperature: float = 0.0):
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
            except ImportError:
                raise RuntimeError("pip install anthropic 필요")
        return self._client

    def evaluate(self, market_state: dict) -> Signal:
        """시장 상태 분석 → Signal 반환 (실패 시 HOLD)"""
        # 매 호출마다 규칙 문서를 새로 읽음 (수정 즉시 반영)
        from config.ai_prompt import build_system_prompt, build_user_message

        system_prompt = build_system_prompt(self.rules_doc_path)
        user_message = build_user_message(market_state)
        start_time = time.time()

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            latency_ms = int((time.time() - start_time) * 1000)
            raw_text = response.content[0].text.strip()

            signal = self._parse_response(raw_text)
            signal.latency_ms = latency_ms
            signal.raw_response = raw_text
            signal.timestamp = market_state.get("timestamp", "")

            logger.info(
                f"AI Signal: action={signal.action.value}, "
                f"confidence={signal.confidence:.2f}, "
                f"size={signal.size_pct}%, latency={latency_ms}ms"
            )
            return signal

        except Exception as e:
            logger.error(f"AI API 호출 실패: {e}", exc_info=True)
            return Signal(
                action=TradeAction.HOLD,
                reasoning=f"AI API 오류: {str(e)}",
                alerts=["API 호출 실패 — HOLD 기본값"],
            )

    def _parse_response(self, raw_text: str) -> Signal:
        """AI 응답 JSON → Signal. 비정상 → HOLD 폴백."""
        try:
            text = raw_text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            text = text.strip()

            data = json.loads(text)

            action_str = data.get("action", "HOLD").upper()
            if action_str not in self.VALID_ACTIONS:
                logger.warning(f"Invalid action '{action_str}' → HOLD")
                action_str = "HOLD"

            return Signal(
                action=TradeAction(action_str),
                direction=data.get("direction", "NEUTRAL"),
                size_pct=max(0, min(100, int(data.get("size_pct", 0)))),
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
                reasoning=data.get("reasoning", ""),
                sl_pct=max(0.0, min(10.0, float(data.get("sl_pct", 2.0)))),
                tp_pct=max(0.0, min(20.0, float(data.get("tp_pct", 4.0)))),
                alerts=data.get("alerts", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"응답 파싱 실패: {e}\nRaw: {raw_text[:500]}")
            return Signal(
                action=TradeAction.HOLD,
                reasoning=f"파싱 오류: {str(e)}",
                alerts=["AI 응답 비정상 — HOLD"],
                raw_response=raw_text,
            )
