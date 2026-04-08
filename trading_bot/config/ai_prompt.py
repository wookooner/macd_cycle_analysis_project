"""
Prompt builders for:
- structured trading decisions
- conversational Telegram analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("bot.ai_prompt")


DECISION_RESPONSE_FORMAT = """
Return only valid JSON in this shape:

```json
{
  "action": "HOLD | ENTER_LONG | ENTER_SHORT | CLOSE_LONG | CLOSE_SHORT | REVERSE_TO_LONG | REVERSE_TO_SHORT",
  "direction": "LONG | SHORT | NEUTRAL",
  "size_pct": 0,
  "confidence": 0.0,
  "reasoning": "short explanation",
  "sl_pct": 2.0,
  "tp_pct": 4.0,
  "alerts": []
}
```
"""


DECISION_SYSTEM_PREFIX = """
You are the AI decision layer for a BTC cycle-based trading bot.
You must read the provided rule document carefully and judge whether the current market_state matches the user's framework.

Important constraints:
- If the rule document is incomplete or ambiguous, default conservatively to HOLD.
- Use the provided structured data as the source of truth.
- Do not invent missing indicators.
- Prefer safety over action.
"""


ANALYSIS_SYSTEM_PREFIX = """
You are the AI analysis layer for a BTC cycle-based trading system.
Read the user's rule document and analyze the current market_state in that context.

Important constraints:
- Be explicit about what the data supports and what it does not.
- If the rule document is empty or incomplete, say so and give a descriptive analysis rather than pretending rules exist.
- Use concise markdown, not JSON.
- Focus on multi-timeframe cycle structure, indicator context, and practical interpretation.
"""


def build_system_prompt(rules_doc_path: Path) -> str:
    rules_content = _load_rules_document(rules_doc_path)
    return DECISION_SYSTEM_PREFIX.strip() + "\n\n[Rule Document]\n" + rules_content + "\n\n" + DECISION_RESPONSE_FORMAT.strip()


def build_user_message(market_state: dict) -> str:
    return "Evaluate the current market_state and return a structured trading decision.\n\n" + json.dumps(
        market_state,
        ensure_ascii=False,
        indent=2,
    )


def build_analysis_system_prompt(rules_doc_path: Path) -> str:
    rules_content = _load_rules_document(rules_doc_path)
    return ANALYSIS_SYSTEM_PREFIX.strip() + "\n\n[Rule Document]\n" + rules_content


def build_analysis_user_message(market_state: dict, user_request: str = "") -> str:
    request = user_request.strip() or "Explain the current cycle structure, important indicators, likely bias, and key caution points."
    return (
        f"User request:\n{request}\n\n"
        "Current market_state:\n"
        + json.dumps(market_state, ensure_ascii=False, indent=2)
    )


def _load_rules_document(rules_doc_path: Path) -> str:
    if rules_doc_path.exists():
        content = rules_doc_path.read_text(encoding="utf-8")
        logger.info("Rules document loaded: %s (%s chars)", rules_doc_path, len(content))
        return content if content.strip() else "[The rule document is currently empty.]"

    logger.warning("Rules document not found: %s", rules_doc_path)
    return "[The rule document file was not found.]"
