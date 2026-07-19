from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.backtest.types import Costs, ExitPolicy, ExitPolicyKind, RuleKind


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_rule_kind(value: str) -> str:
    legacy = {"event_based": "production", "backtest_only": "research_only"}
    return legacy.get(str(value), str(value))


def load_rules_config(path: str | Path) -> list[dict[str, Any]]:
    rules = load_yaml(path).get("rules", [])
    for rule in rules:
        rule["rule_kind"] = normalize_rule_kind(rule.get("rule_kind", RuleKind.RESEARCH_ONLY.value))
    return rules


def load_stops_config(path: str | Path) -> list[ExitPolicy]:
    policies: list[ExitPolicy] = []
    for stop in load_yaml(path).get("stops", []):
        kind = str(stop.get("stop_kind", "cycle_exit"))
        if kind == "cycle_exit":
            policy_kind = ExitPolicyKind.OPPOSITE_TRUE
        elif kind == "partial_tp":
            policy_kind = ExitPolicyKind.PARTIAL_TP
        elif kind == "tp_sl":
            policy_kind = ExitPolicyKind.TP_SL
        elif kind == "fixed_sl":
            policy_kind = ExitPolicyKind.FIXED_SL
        elif kind == "atr_stop":
            policy_kind = ExitPolicyKind.ATR_STOP
        else:
            policy_kind = ExitPolicyKind.STRUCTURAL_INVALIDATION
        policies.append(ExitPolicy(name=stop["stop_name"], kind=policy_kind, params=stop.get("params", {}) or {}))
    return policies


def load_grid_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def costs_from_grid(grid: dict[str, Any]) -> Costs:
    return Costs(fee_pct=float(grid.get("fee", 0.08)), slippage_pct_per_side=float(grid.get("slippage", 0.02)))
