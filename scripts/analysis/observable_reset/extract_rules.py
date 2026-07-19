from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lib import DEFAULT_HORIZONS, OBSERVABLE_FEATURES, OUT_DIR, TRAIN_PATH, metrics_for_returns, save_json


@dataclass
class Node:
    conditions: list[dict[str, Any]]
    depth: int
    rows: pd.DataFrame


def gini(y: pd.Series) -> float:
    if y.empty:
        return 0.0
    p = float(y.mean())
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


def candidate_thresholds(series: pd.Series) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.nunique() <= 1:
        return []
    qs = values.quantile([0.2, 0.4, 0.6, 0.8]).drop_duplicates()
    return [float(v) for v in qs.tolist() if np.isfinite(v)]


def best_split(df: pd.DataFrame, features: list[str], target: str, min_leaf: int) -> tuple[str, float, float] | None:
    base = gini(df[target])
    best: tuple[str, float, float] | None = None
    for feature in features:
        for threshold in candidate_thresholds(df[feature]):
            left = df[df[feature] <= threshold]
            right = df[df[feature] > threshold]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            gain = base - (len(left) / len(df)) * gini(left[target]) - (len(right) / len(df)) * gini(right[target])
            if best is None or gain > best[2]:
                best = (feature, threshold, float(gain))
    return best


def grow_tree(df: pd.DataFrame, features: list[str], target: str, return_col: str, max_depth: int, min_leaf: int) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []
    stack = [Node(conditions=[], depth=0, rows=df)]
    while stack:
        node = stack.pop()
        split = best_split(node.rows, features, target, min_leaf) if node.depth < max_depth else None
        if split is None:
            metrics = metrics_for_returns(node.rows.rename(columns={return_col: "net_return"}), "net_return")
            if metrics.get("n", 0) >= min_leaf:
                leaves.append({"conditions": node.conditions, **metrics})
            continue
        feature, threshold, gain = split
        left_cond = {"feature": feature, "op": "<=", "value": threshold}
        right_cond = {"feature": feature, "op": ">", "value": threshold}
        stack.append(Node(node.conditions + [right_cond], node.depth + 1, node.rows[node.rows[feature] > threshold]))
        stack.append(Node(node.conditions + [left_cond], node.depth + 1, node.rows[node.rows[feature] <= threshold]))
    return leaves


def render_rule(rule: dict[str, Any]) -> str:
    cond = " and ".join(f"{c['feature']} {c['op']} {c['value']:.6g}" for c in rule["conditions"]) or "always"
    return (
        f"[{rule['timeframe']} {rule['direction']} h{rule['horizon']}] if {cond} "
        f"=> n={rule['n']} WR={rule['win_rate_pct']:.2f}% net={rule['net_avg_pct']:.4f}% PF={rule['profit_factor']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-leaf", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_parquet(TRAIN_PATH)
    features = [f for f in OBSERVABLE_FEATURES if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    rules: list[dict[str, Any]] = []
    for horizon in args.horizons:
        ret_col = f"net_return_h{horizon}"
        if ret_col not in df.columns:
            continue
        for (tf, direction), group in df.dropna(subset=[ret_col]).groupby(["timeframe", "direction"], observed=True):
            work = group[features + [ret_col]].dropna().copy()
            if len(work) < args.min_leaf * 2:
                continue
            work["target_win"] = work[ret_col] > 0
            leaves = grow_tree(work, features, "target_win", ret_col, args.max_depth, args.min_leaf)
            for leaf in leaves:
                leaf.update({
                    "timeframe": tf,
                    "direction": direction,
                    "horizon": horizon,
                    "return_col": ret_col,
                })
                leaf["rule_id"] = f"r{len(rules) + 1:04d}"
                rules.append(leaf)

    rules = sorted(rules, key=lambda r: (r.get("net_avg_pct", -999), r.get("profit_factor", 0), r.get("n", 0)), reverse=True)
    save_json(OUT_DIR / "candidate_rules.json", rules)
    (OUT_DIR / "candidate_rules.txt").write_text("\n".join(render_rule(rule) for rule in rules) + "\n", encoding="utf-8")
    print(json.dumps({"rules": len(rules), "features": features}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
