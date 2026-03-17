"""
trading_bot/config/ai_prompt.py
================================
Claude API 프롬프트 구성.

핵심: 매 호출 시 trading_rules.md 파일을 읽어서 시스템 프롬프트에 포함.
→ 규칙 수정 시 코드 변경 없이 .md 파일만 수정하면 됨.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("bot.ai_prompt")

# 규칙 문서와 별개로, AI 응답 형식을 지정하는 고정 프롬프트
_RESPONSE_FORMAT_PROMPT = """
## 응답 형식

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.

```json
{
  "action": "HOLD | ENTER_LONG | ENTER_SHORT | CLOSE_LONG | CLOSE_SHORT | REVERSE_TO_LONG | REVERSE_TO_SHORT",
  "direction": "LONG | SHORT | NEUTRAL",
  "size_pct": 0-100,
  "confidence": 0.0-1.0,
  "reasoning": "판단 근거 (한국어, 2-3문장)",
  "sl_pct": 0.0-5.0,
  "tp_pct": 0.0-10.0,
  "alerts": ["주의사항 리스트"]
}
```

### action 결정 우선순위
1. 현재 포지션이 있고, 사이클 전환으로 방향 불일치 → CLOSE 또는 REVERSE
2. 포지션 없고, 조건 충족 → ENTER
3. 조건 불충분 또는 회피 콤보 → HOLD
4. dur ≤ 4 → 무조건 HOLD

### confidence 기준
- 0.9+: n_up=4/0 + 스위트스팟
- 0.7-0.9: n_up=3/1 + 필터 충족
- 0.5-0.7: n_up=2 거래가능 콤보
- <0.5: HOLD 권장
"""

_SYSTEM_PREFIX = """당신은 BTC 선물 매매 시스템의 의사결정 엔진입니다.
아래 규칙서에 따라 현재 시장 상태를 분석하고, 매매 신호와 포지션 사이징을 결정합니다.
규칙서에 명시되지 않은 판단은 HOLD로 처리하세요.

"""


def build_system_prompt(rules_doc_path: Path) -> str:
    """
    시스템 프롬프트 생성.
    trading_rules.md를 읽어서 동적으로 구성.
    
    Args:
        rules_doc_path: trading_rules.md 경로
        
    Returns:
        완성된 시스템 프롬프트 문자열
    """
    rules_content = _load_rules_document(rules_doc_path)
    return _SYSTEM_PREFIX + rules_content + "\n\n" + _RESPONSE_FORMAT_PROMPT


def build_user_message(market_state: dict) -> str:
    """market_state JSON을 user 메시지로 변환"""
    return (
        "현재 시장 상태를 분석하고 매매 결정을 내려주세요.\n\n"
        + json.dumps(market_state, ensure_ascii=False, indent=2)
    )


def _load_rules_document(rules_doc_path: Path) -> str:
    """
    규칙 문서 로드. 파일이 없으면 경고 + 기본 규칙 반환.
    """
    if rules_doc_path.exists():
        content = rules_doc_path.read_text(encoding="utf-8")
        logger.info(f"Rules document loaded: {rules_doc_path} ({len(content)} chars)")
        return content

    logger.warning(f"Rules document not found: {rules_doc_path} — using fallback")
    return _FALLBACK_RULES


# 규칙 문서가 없을 때 최소한의 폴백
_FALLBACK_RULES = """
## 기본 규칙 (폴백)
- n_up=4 → ENTER_LONG (100%)
- n_up=0 → ENTER_SHORT (100%)
- n_up=3 → ENTER_LONG (50~100%)
- n_up=1 → ENTER_SHORT (50~100%)
- n_up=2 → HOLD (기본)
- dur ≤ 4 → 무조건 HOLD
- 확실하지 않으면 HOLD
"""
