"""
trading_bot/utils/decision_logger.py
=====================================
모든 AI 의사결정을 JSONL 파일에 기록.

기록 내용:
  - 시장 상태 (market_state)
  - AI 판단 결과 (signal)
  - 실행 결과 (order_result)
  - 모드 (auto/manual)
  - 사용자 응답 (approved/rejected/timeout)
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger("bot.decision_log")


class DecisionLogger:
    """
    JSONL 형식으로 의사결정 로그를 기록.
    
    파일: {log_dir}/decisions_{YYYY-MM}.jsonl (월별 분리)
    
    각 행은 하나의 의사결정 이벤트:
    {
        "timestamp": "2026-03-17T09:55:00",
        "event": "signal_generated | order_executed | order_rejected | order_timeout",
        "mode": "auto | manual",
        "market_state": { ... },
        "signal": { ... },
        "order_result": { ... },
        "user_response": "approved | rejected | timeout | auto_executed",
        "notes": "..."
    }
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_path(self) -> Path:
        """월별 로그 파일 경로"""
        now = datetime.utcnow()
        return self.log_dir / f"decisions_{now.strftime('%Y-%m')}.jsonl"

    def _write(self, record: dict):
        """JSONL 한 행 기록"""
        record["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        try:
            path = self._get_log_path()
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Decision log write failed: {e}")

    def log_signal(self, market_state: dict, signal_dict: dict,
                   mode: str):
        """AI 신호 생성 시 기록"""
        self._write({
            "event": "signal_generated",
            "mode": mode,
            "market_state": market_state,
            "signal": signal_dict,
        })

    def log_hold(self, market_state: dict, signal_dict: dict,
                 mode: str):
        """HOLD 판단 시 기록 (신호 없음)"""
        self._write({
            "event": "hold",
            "mode": mode,
            "chain": market_state.get("chain", {}),
            "price": market_state.get("price"),
            "signal": signal_dict,
        })

    def log_execution(self, signal_dict: dict, order_result: dict,
                      mode: str, user_response: str):
        """주문 실행 결과 기록"""
        self._write({
            "event": "order_executed",
            "mode": mode,
            "user_response": user_response,
            "signal": signal_dict,
            "order_result": order_result,
        })

    def log_rejection(self, signal_dict: dict, reason: str, mode: str):
        """주문 거부/타임아웃 기록"""
        self._write({
            "event": "order_rejected",
            "mode": mode,
            "user_response": reason,
            "signal": signal_dict,
        })

    def log_error(self, error_msg: str, context: Optional[dict] = None):
        """오류 기록"""
        self._write({
            "event": "error",
            "error": error_msg,
            "context": context or {},
        })

    def get_recent_decisions(self, n: int = 10) -> list[dict]:
        """최근 N개 의사결정 조회 (status 명령 등에서 사용)"""
        path = self._get_log_path()
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            records = []
            for line in lines[-n:]:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
            return records
        except Exception:
            return []
