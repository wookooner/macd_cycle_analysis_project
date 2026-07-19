from __future__ import annotations

import argparse
import json

import pandas as pd

from lib import FEATURE_PATH, OUT_DIR, SPLIT_PATH, TEST_PATH, load_features, load_json, metrics_for_returns, save_json
from robustness_filter import apply_conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlock-test", action="store_true", help="Required to read and materialize OOS rows.")
    args = parser.parse_args()
    if not args.unlock_test:
        raise SystemExit("Refusing to load test set without --unlock-test")

    split = load_json(SPLIT_PATH)
    train_end = pd.Timestamp(split["train_end"])
    features = load_features(FEATURE_PATH).copy()
    features["entry_time"] = pd.to_datetime(features["entry_time"], errors="coerce")
    test = features[features["entry_time"] > train_end].copy()
    test.to_parquet(TEST_PATH, index=False)

    frozen = load_json(OUT_DIR / "frozen_rules.json")
    rows: list[dict] = []
    for rule in frozen:
        ret_col = rule["return_col"]
        subset = test[
            (test["timeframe"] == rule["timeframe"])
            & (test["direction"] == rule["direction"])
        ].dropna(subset=[ret_col]).copy()
        matched = subset[apply_conditions(subset, rule["conditions"])]
        metrics = metrics_for_returns(matched.rename(columns={ret_col: "net_return"}), "net_return")
        train_metrics = rule.get("train_metrics_recomputed", {})
        uses_nup = any(cond["feature"].startswith("nup_legal") for cond in rule["conditions"])
        rows.append({
            "rule_id": rule["rule_id"],
            "timeframe": rule["timeframe"],
            "direction": rule["direction"],
            "horizon": rule["horizon"],
            "uses_nup_legal_hist": uses_nup,
            "train_n": train_metrics.get("n"),
            "train_wr": train_metrics.get("win_rate_pct"),
            "train_net": train_metrics.get("net_avg_pct"),
            "train_pf": train_metrics.get("profit_factor"),
            "test_n": metrics.get("n"),
            "test_wr": metrics.get("win_rate_pct"),
            "test_net": metrics.get("net_avg_pct"),
            "test_pf": metrics.get("profit_factor"),
            "net_decay_pct": None if train_metrics.get("net_avg_pct") in (None, 0) else metrics.get("net_avg_pct", 0) / train_metrics.get("net_avg_pct"),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "oos_validation.csv", index=False, encoding="utf-8-sig")
    report = [
        "# OOS Validation",
        "",
        f"- Train end: `{train_end}`",
        f"- Test rows: `{len(test)}`",
        f"- Frozen rules: `{len(frozen)}`",
        "",
    ]
    if out.empty:
        report.append("No frozen rules survived robustness filtering.")
    else:
        compare = out.groupby("uses_nup_legal_hist")[["test_n", "test_wr", "test_net", "test_pf"]].mean(numeric_only=True).reset_index()
        report.extend(["## n_up Comparison", "", compare.to_markdown(index=False), "", "## Rules", "", out.to_markdown(index=False)])
    (OUT_DIR / "oos_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    save_json(OUT_DIR / "oos_validation_metadata.json", {"test_rows": int(len(test)), "rules": len(frozen)})
    print(json.dumps({"test_rows": int(len(test)), "rules": len(frozen), "rows": int(len(out))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
