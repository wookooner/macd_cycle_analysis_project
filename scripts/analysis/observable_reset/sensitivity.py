from __future__ import annotations

import argparse
import json

import pandas as pd

from lib import DEFAULT_COST_PCT, OUT_DIR, TEST_PATH, load_json, metrics_for_returns
from robustness_filter import apply_conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--costs", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    args = parser.parse_args()

    test = pd.read_parquet(TEST_PATH)
    frozen = load_json(OUT_DIR / "frozen_rules.json")
    rows: list[dict] = []
    for rule in frozen:
        ret_col = rule["return_col"]
        subset = test[
            (test["timeframe"] == rule["timeframe"])
            & (test["direction"] == rule["direction"])
        ].dropna(subset=[ret_col]).copy()
        matched = subset[apply_conditions(subset, rule["conditions"])].copy()
        if matched.empty:
            continue
        gross_col = "_gross_est"
        matched[gross_col] = matched[ret_col] + DEFAULT_COST_PCT
        signs = []
        for cost in args.costs:
            tmp = matched.copy()
            tmp["net_return"] = tmp[gross_col] - cost
            metrics = metrics_for_returns(tmp, "net_return")
            signs.append(metrics.get("net_avg_pct", 0) > 0)
            rows.append({
                "rule_id": rule["rule_id"],
                "timeframe": rule["timeframe"],
                "direction": rule["direction"],
                "horizon": rule["horizon"],
                "cost_pct": cost,
                **metrics,
            })
        rows[-len(args.costs)]["discard_cost_sign_flip"] = not all(signs)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["discard_cost_sign_flip"] = out.groupby("rule_id")["net_avg_pct"].transform(lambda s: not (s > 0).all())
    else:
        out["discard_cost_sign_flip"] = pd.Series(dtype="bool")
    out.to_csv(OUT_DIR / "sensitivity.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": int(len(out)), "rules": len(frozen)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
