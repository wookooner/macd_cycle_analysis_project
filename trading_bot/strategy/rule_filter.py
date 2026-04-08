"""
Rule-based pre-trade filter and fallback signal generator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from strategy.ai_signal_engine import Signal, TradeAction

logger = logging.getLogger("bot.rule_filter")

_HARD_AVOID: frozenset[str] = frozenset({"UUUD", "DDDU", "UDDU", "DUUD"})
_DANGER_HIGH: frozenset[str] = frozenset({"UUUD", "DDDU"})
_DANGER_MED: frozenset[str] = frozenset({"UDDU", "DUUD"})

_LONG_ENTRY_ACTIONS = frozenset({TradeAction.ENTER_LONG, TradeAction.REVERSE_TO_LONG})
_SHORT_ENTRY_ACTIONS = frozenset({TradeAction.ENTER_SHORT, TradeAction.REVERSE_TO_SHORT})


@dataclass
class FilterResult:
    allowed: bool
    block_reason: str = ""
    size_modifier: float = 1.0
    alerts: list[str] = field(default_factory=list)
    danger_score: int = 0
    n_up: int = -1
    suggested_direction: str = "NEUTRAL"


class RuleFilter:
    MIN_DURATION = 5

    def check(self, market_state: dict) -> FilterResult:
        chain = market_state.get("chain", {})
        timeframes = market_state.get("timeframes", {})
        price = market_state.get("price")

        n_up = int(chain.get("n_up", -1))
        combo = str(chain.get("combo", "")).upper()
        alignment_4h_1h = bool(chain.get("alignment_4h_1h", False))

        tf_1h = timeframes.get("1h") or {}
        tf_4h = timeframes.get("4h") or {}
        snap_1h = tf_1h.get("analysis_snapshot") or {}

        dur_1h = int(tf_1h.get("duration", 0) or 0)
        pos_4h = float(tf_4h.get("position_pct", 0.0) or 0.0)
        ppo_hist = snap_1h.get("ppo_hist")
        ma_25 = snap_1h.get("ma_25")

        if n_up < 0:
            return FilterResult(False, "n_up data is missing.")

        if n_up >= 3:
            suggested = "LONG"
        elif n_up <= 1:
            suggested = "SHORT"
        else:
            suggested = "NEUTRAL"

        if combo in _HARD_AVOID:
            return FilterResult(
                False,
                f"Avoid combo {combo} blocked by rule.",
                alerts=[f"avoid combo {combo}"],
                n_up=n_up,
                suggested_direction=suggested,
            )

        if dur_1h < self.MIN_DURATION:
            return FilterResult(
                False,
                f"1h duration {dur_1h} is below minimum {self.MIN_DURATION}.",
                alerts=[f"duration too short: {dur_1h}"],
                n_up=n_up,
                suggested_direction=suggested,
            )

        if n_up == 2:
            result = self._check_n2_setup(ma_25, price, dur_1h, n_up)
            if result is not None:
                return result
            if ma_25 is not None and price is not None and price > 0:
                dist = (price - ma_25) / ma_25 * 100
                if dist < 0:
                    suggested = "LONG"
                elif dist > 0:
                    suggested = "SHORT"

        danger = self._danger_score(ppo_hist, ma_25, price, pos_4h, combo, alignment_4h_1h)

        if danger >= 7:
            return FilterResult(
                False,
                f"Danger score {danger} is in the red zone.",
                alerts=[f"danger RED: {danger}"],
                danger_score=danger,
                n_up=n_up,
                suggested_direction=suggested,
            )

        size_mod = 1.0
        alerts: list[str] = []

        if danger >= 5:
            size_mod *= 0.5
            alerts.append(f"danger ORANGE ({danger}) -> size x0.5")

        if not alignment_4h_1h:
            size_mod *= 0.5
            alerts.append("4h/1h misaligned -> size x0.5")

        logger.info(
            "RuleFilter OK: n_up=%s dir=%s combo=%s danger=%s size_mod=%.2f",
            n_up, suggested, combo, danger, size_mod,
        )
        return FilterResult(
            allowed=True,
            size_modifier=size_mod,
            alerts=alerts,
            danger_score=danger,
            n_up=n_up,
            suggested_direction=suggested,
        )

    def generate_signal(self, market_state: dict, current_position: Optional[dict] = None) -> Signal:
        result = self.check(market_state)
        if not result.allowed:
            return Signal(
                action=TradeAction.HOLD,
                direction="NEUTRAL",
                reasoning=result.block_reason,
                alerts=list(result.alerts),
                timestamp=market_state.get("timestamp", ""),
            )

        action = self._resolve_action(result, current_position)
        reasoning = (
            "Rule-only engine: no actionable setup."
            if action == TradeAction.HOLD
            else f"Rule-only engine: n_up={result.n_up}, direction={result.suggested_direction}, danger={result.danger_score}."
        )

        signal = Signal(
            action=action,
            direction=result.suggested_direction,
            size_pct=self._size_for_result(result),
            confidence=self._confidence_for_result(result),
            reasoning=reasoning,
            alerts=list(result.alerts),
            timestamp=market_state.get("timestamp", ""),
        )
        return self.validate_signal(signal, result)

    def validate_signal(self, signal: Signal, result: FilterResult) -> Signal:
        if result.suggested_direction == "LONG" and signal.action in _SHORT_ENTRY_ACTIONS:
            logger.warning("Rule 1 override: short entry blocked while n_up=%s suggests LONG", result.n_up)
            return Signal(
                action=TradeAction.HOLD,
                reasoning=f"Rule 1: n_up={result.n_up} only allows LONG-side positioning.",
                alerts=["Rule1 direction override: forced HOLD"],
            )

        if result.suggested_direction == "SHORT" and signal.action in _LONG_ENTRY_ACTIONS:
            logger.warning("Rule 1 override: long entry blocked while n_up=%s suggests SHORT", result.n_up)
            return Signal(
                action=TradeAction.HOLD,
                reasoning=f"Rule 1: n_up={result.n_up} only allows SHORT-side positioning.",
                alerts=["Rule1 direction override: forced HOLD"],
            )

        if result.size_modifier < 1.0 and signal.size_pct > 0:
            original = signal.size_pct
            signal.size_pct = max(1, int(signal.size_pct * result.size_modifier))
            logger.info("size_pct %s -> %s (x%.2f)", original, signal.size_pct, result.size_modifier)
            signal.alerts = list(signal.alerts) + result.alerts

        return signal

    def _check_n2_setup(
        self,
        ma_25: Optional[float],
        price: Optional[float],
        dur: int,
        n_up: int,
    ) -> Optional[FilterResult]:
        if ma_25 is None or price is None or ma_25 <= 0:
            return FilterResult(
                False,
                "n_up=2 but MA25 data is missing.",
                alerts=["n_up=2 requires MA25 distance and duration filter"],
                n_up=n_up,
                suggested_direction="NEUTRAL",
            )

        dist = (price - ma_25) / ma_25 * 100
        long_ok = dist < 0 and dur >= 8
        short_ok = dist > 0 and dur >= 8

        if not long_ok and not short_ok:
            return FilterResult(
                False,
                f"n_up=2 setup not ready: dist_MA25={dist:.1f}% dur={dur}",
                alerts=["n_up=2 MA25/duration filter not met"],
                n_up=n_up,
                suggested_direction="NEUTRAL",
            )
        return None

    def _resolve_action(self, result: FilterResult, current_position: Optional[dict]) -> TradeAction:
        side = str((current_position or {}).get("side", "")).upper()
        direction = result.suggested_direction

        if direction == "LONG":
            if side == "SHORT":
                return TradeAction.REVERSE_TO_LONG
            if side != "LONG":
                return TradeAction.ENTER_LONG
            return TradeAction.HOLD

        if direction == "SHORT":
            if side == "LONG":
                return TradeAction.REVERSE_TO_SHORT
            if side != "SHORT":
                return TradeAction.ENTER_SHORT
            return TradeAction.HOLD

        return TradeAction.HOLD

    def _size_for_result(self, result: FilterResult) -> int:
        if result.n_up in (4, 0):
            base_size = 90
        elif result.n_up in (3, 1):
            base_size = 70
        elif result.n_up == 2:
            base_size = 50
        else:
            base_size = 0
        return max(0, min(100, int(base_size)))

    def _confidence_for_result(self, result: FilterResult) -> float:
        if result.n_up in (4, 0):
            base_confidence = 0.9
        elif result.n_up in (3, 1):
            base_confidence = 0.75
        elif result.n_up == 2:
            base_confidence = 0.6
        else:
            base_confidence = 0.0
        adjusted = base_confidence - max(0, result.danger_score - 4) * 0.03
        return max(0.0, min(1.0, round(adjusted, 2)))

    @staticmethod
    def _danger_score(
        ppo_hist: Optional[float],
        ma_25: Optional[float],
        price: Optional[float],
        pos_4h: float,
        combo: str,
        alignment_4h_1h: bool,
    ) -> int:
        score = 0

        if ppo_hist is not None:
            ah = abs(ppo_hist)
            if ah < 20:
                score += 3
            elif ah < 40:
                score += 2
            elif ah < 60:
                score += 1

        if ma_25 is not None and price is not None and ma_25 > 0:
            dp = abs((price - ma_25) / ma_25 * 100)
            if dp > 3:
                score += 3
            elif dp > 2:
                score += 2
            elif dp > 1.5:
                score += 1

        if pos_4h > 0.8:
            score += 2
        elif pos_4h > 0.5:
            score += 1

        if combo in _DANGER_HIGH:
            score += 2
        elif combo in _DANGER_MED:
            score += 1

        if not alignment_4h_1h and 0.3 < pos_4h < 0.7:
            score += 1

        return score
