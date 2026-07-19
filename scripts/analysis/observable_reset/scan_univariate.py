from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from lib import DEFAULT_HORIZONS, OBSERVABLE_FEATURES, OUT_DIR, TRAIN_PATH, metrics_for_returns, qbin, spearman_like


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--min-sample", type=int, default=100)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(TRAIN_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    rows: list[dict] = []
    ranking_rows: list[dict] = []

    for horizon in args.horizons:
        ret_col = f"net_return_h{horizon}"
        if ret_col not in df.columns:
            continue
        for feature in [f for f in OBSERVABLE_FEATURES if f in df.columns]:
            working = df[["timeframe", "direction", "entry_time", feature, ret_col]].dropna().copy()
            if working.empty:
                continue
            if working[feature].nunique(dropna=True) <= args.bins:
                working["bin"] = working[feature].astype(str)
                monotonic_source = working[feature].rank(method="dense")
            else:
                working["bin"] = qbin(working[feature], args.bins)
                monotonic_source = pd.to_numeric(working[feature], errors="coerce")
            working = working.rename(columns={ret_col: "net_return"})
            rho = spearman_like(monotonic_source, working["net_return"])
            yearly = (
                working.assign(year=working["entry_time"].dt.year)
                .groupby("year")["net_return"]
                .mean()
                .dropna()
            )
            stability = float(np.sign(yearly).mode().iloc[0] == np.sign(yearly).mean()) if len(yearly) else np.nan
            best_abs = -np.inf
            best: dict | None = None
            for keys, group in working.groupby(["timeframe", "direction", "bin"], dropna=False, observed=True):
                metrics = metrics_for_returns(group, "net_return")
                if metrics["n"] < args.min_sample:
                    continue
                row = {
                    "feature": feature,
                    "horizon": horizon,
                    "timeframe": keys[0],
                    "direction": keys[1],
                    "bin": keys[2],
                    "spearman": rho,
                    "year_sign_stability": stability,
                    **metrics,
                }
                rows.append(row)
                if abs(metrics["net_avg_pct"]) > best_abs:
                    best_abs = abs(metrics["net_avg_pct"])
                    best = row
            if best:
                ranking_rows.append({
                    "feature": feature,
                    "horizon": horizon,
                    "best_abs_net_avg_pct": best_abs,
                    "best_net_avg_pct": best["net_avg_pct"],
                    "best_n": best["n"],
                    "best_timeframe": best["timeframe"],
                    "best_direction": best["direction"],
                    "best_bin": best["bin"],
                    "spearman": rho,
                    "year_sign_stability": stability,
                })

    result = pd.DataFrame(rows).sort_values(["horizon", "timeframe", "direction", "feature", "bin"])
    ranking = pd.DataFrame(ranking_rows).sort_values(["best_abs_net_avg_pct", "best_n"], ascending=[False, False])
    result.to_csv(OUT_DIR / "univariate_scan.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(OUT_DIR / "univariate_feature_ranking.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": int(len(result)), "ranking_rows": int(len(ranking)), "top_features": ranking["feature"].head(5).tolist() if not ranking.empty else []}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
