from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd

from lib import DEFAULT_HORIZONS, OUT_DIR, TRAIN_PATH, metrics_for_returns, qbin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--min-sample", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_parquet(TRAIN_PATH)
    ranking_path = OUT_DIR / "univariate_feature_ranking.csv"
    ranking = pd.read_csv(ranking_path)
    top_features = ranking["feature"].drop_duplicates().head(args.top_n).tolist()

    rows: list[dict] = []
    for horizon in args.horizons:
        ret_col = f"net_return_h{horizon}"
        if ret_col not in df.columns:
            continue
        for f1, f2 in itertools.combinations(top_features, 2):
            if f1 not in df.columns or f2 not in df.columns:
                continue
            working = df[["timeframe", "direction", f1, f2, ret_col]].dropna().copy()
            if working.empty:
                continue
            working["bin1"] = working[f1].astype(str) if working[f1].nunique() <= args.bins else qbin(working[f1], args.bins)
            working["bin2"] = working[f2].astype(str) if working[f2].nunique() <= args.bins else qbin(working[f2], args.bins)
            working = working.rename(columns={ret_col: "net_return"})
            for keys, group in working.groupby(["timeframe", "direction", "bin1", "bin2"], dropna=False, observed=True):
                metrics = metrics_for_returns(group, "net_return")
                if metrics["n"] < args.min_sample:
                    continue
                rows.append({
                    "feature1": f1,
                    "feature2": f2,
                    "horizon": horizon,
                    "timeframe": keys[0],
                    "direction": keys[1],
                    "bin1": keys[2],
                    "bin2": keys[3],
                    **metrics,
                })

    result = pd.DataFrame(rows).sort_values(["horizon", "timeframe", "direction", "feature1", "feature2", "bin1", "bin2"])
    result.to_csv(OUT_DIR / "bivariate_scan.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": int(len(result)), "top_features": top_features}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
