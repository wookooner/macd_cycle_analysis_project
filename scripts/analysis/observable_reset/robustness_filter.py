from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from lib import OUT_DIR, TRAIN_PATH, bh_fdr, load_json, metrics_for_returns, save_json


def apply_conditions(df: pd.DataFrame, conditions: list[dict]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for cond in conditions:
        if cond["op"] == "<=":
            mask &= pd.to_numeric(df[cond["feature"]], errors="coerce") <= float(cond["value"])
        elif cond["op"] == ">":
            mask &= pd.to_numeric(df[cond["feature"]], errors="coerce") > float(cond["value"])
        else:
            raise ValueError(f"unsupported op: {cond['op']}")
    return mask


def normal_p_from_wr(win_rate_pct: float, n: int) -> float:
    if n <= 0:
        return 1.0
    p = win_rate_pct / 100.0
    z = (p - 0.5) / math.sqrt(0.25 / n)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-sample", type=int, default=200)
    parser.add_argument("--fdr-alpha", type=float, default=0.10)
    parser.add_argument("--min-net", type=float, default=0.0)
    args = parser.parse_args()

    train = pd.read_parquet(TRAIN_PATH)
    train["entry_time"] = pd.to_datetime(train["entry_time"], errors="coerce")
    candidates = load_json(OUT_DIR / "candidate_rules.json")
    pvals = [normal_p_from_wr(rule.get("win_rate_pct", 50.0), int(rule.get("n", 0))) for rule in candidates]
    qvals = bh_fdr(pvals)

    total_configs = len(candidates)
    for name in ("univariate_scan.csv", "bivariate_scan.csv"):
        path = OUT_DIR / name
        if path.exists():
            total_configs += len(pd.read_csv(path))

    frozen: list[dict] = []
    rejected: list[dict] = []
    for rule, pval, qval in zip(candidates, pvals, qvals):
        ret_col = rule["return_col"]
        subset = train[
            (train["timeframe"] == rule["timeframe"])
            & (train["direction"] == rule["direction"])
        ].dropna(subset=[ret_col]).copy()
        mask = apply_conditions(subset, rule["conditions"])
        matched = subset[mask].copy()
        metrics = metrics_for_returns(matched.rename(columns={ret_col: "net_return"}), "net_return")
        yearly = matched.assign(year=pd.to_datetime(matched["entry_time"]).dt.year).groupby("year")[ret_col].mean().dropna()
        year_sign_ok = bool((yearly > 0).all()) if len(yearly) >= 2 else False
        reasons = []
        if metrics.get("n", 0) < args.min_sample:
            reasons.append("min_sample")
        if metrics.get("net_avg_pct", -999) <= args.min_net:
            reasons.append("non_positive_net")
        if metrics.get("profit_factor", 0) <= 1.0:
            reasons.append("pf_le_1")
        if not year_sign_ok:
            reasons.append("year_sign_not_stable")
        if qval > args.fdr_alpha:
            reasons.append("bh_fdr")
        enriched = {
            **rule,
            "train_metrics_recomputed": metrics,
            "p_value": pval,
            "q_value": qval,
            "total_configs": total_configs,
            "year_sign_ok": year_sign_ok,
        }
        if reasons:
            rejected.append({**enriched, "reject_reasons": reasons})
        else:
            frozen.append(enriched)

    save_json(OUT_DIR / "frozen_rules.json", frozen)
    save_json(OUT_DIR / "rejected_rules.json", rejected)
    print(json.dumps({"frozen": len(frozen), "rejected": len(rejected), "total_configs": total_configs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
