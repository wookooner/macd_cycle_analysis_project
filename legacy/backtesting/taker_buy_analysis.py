import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TIMEFRAMES = ["1h", "4h", "1d", "1w"]


def find_project_root():
    here = Path(__file__).resolve()
    for current in [here.parent] + list(here.parents):
        if (current / "data" / "base_data").exists() and (current / "data" / "cycle_data" / "structured").exists():
            return current
    for current in [Path.cwd().resolve()] + list(Path.cwd().resolve().parents):
        if (current / "data" / "base_data").exists() and (current / "data" / "cycle_data" / "structured").exists():
            return current
    raise FileNotFoundError("Could not locate project root.")


def resolve_cycle_data_dir(project_root):
    structured_dir = project_root / "data" / "cycle_data" / "structured"
    btc_dir = structured_dir / "btc"
    required = ["cycles_1h.parquet", "cycles_4h.parquet", "cycles_1d.parquet", "cycles_1w.parquet"]
    if all((btc_dir / name).exists() for name in required):
        return btc_dir
    if all((structured_dir / name).exists() for name in required):
        return structured_dir
    raise FileNotFoundError("Could not find cycle parquet files.")


def ensure_output_dir(project_root):
    output_dir = project_root / "analysis_results" / "taker_buy_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_base_csv(project_root, timeframe):
    path = project_root / "data" / "base_data" / f"BTCUSD_{timeframe}.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["taker_buy_base"] = pd.to_numeric(df.get("taker_buy_base"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    df["Volume USD"] = pd.to_numeric(df.get("Volume USD"), errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_cycles(cycle_data_dir, timeframe):
    path = cycle_data_dir / f"cycles_{timeframe}.parquet"
    df = pd.read_parquet(path)
    keep_cols = ["cycle_id", "cycle_type", "duration_candles", "start_date", "end_date"]
    df = df[keep_cols].copy()
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["start_date", "end_date"]).sort_values("start_date").reset_index(drop=True)
    return df


def normal_pvalue_from_z(z_value):
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def regression_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return {
            "n": n,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "corr": np.nan,
            "z_score": np.nan,
            "p_value": np.nan,
        }

    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    ssx = np.sum(x_centered ** 2)
    if ssx == 0:
        return {
            "n": n,
            "slope": 0.0,
            "intercept": y_mean,
            "r2": 0.0,
            "corr": 0.0,
            "z_score": 0.0,
            "p_value": 1.0,
        }

    slope = np.sum(x_centered * y_centered) / ssx
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * x
    resid = y - fitted
    sse = np.sum(resid ** 2)
    sst = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    corr = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0.0
    se_slope = math.sqrt((sse / max(n - 2, 1)) / ssx) if n > 2 and ssx > 0 else np.nan
    z_score = slope / se_slope if se_slope and np.isfinite(se_slope) and se_slope > 0 else np.nan
    p_value = normal_pvalue_from_z(z_score) if np.isfinite(z_score) else np.nan
    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "corr": corr,
        "z_score": z_score,
        "p_value": p_value,
    }


def year_group_effect_size(df, value_col):
    sample = df[["year", value_col]].dropna()
    if sample.empty:
        return np.nan
    grand_mean = sample[value_col].mean()
    ss_total = ((sample[value_col] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for _, group in sample.groupby("year"):
        ss_between += len(group) * (group[value_col].mean() - grand_mean) ** 2
    return ss_between / ss_total


def largest_year_jump(yearly_df, col):
    ordered = yearly_df[["year", col]].dropna().sort_values("year").reset_index(drop=True)
    if len(ordered) < 2:
        return None
    ordered["delta"] = ordered[col].diff()
    row = ordered.iloc[ordered["delta"].abs().idxmax()]
    return {
        "year": int(row["year"]),
        "delta": float(row["delta"]),
        "from_year": int(ordered.iloc[max(row.name - 1, 0)]["year"]),
        "to_year": int(row["year"]),
    }


def add_scaled_features(df):
    out = df.copy()
    out["taker_buy_quote"] = out["taker_buy_base"] * out["close"]
    out["taker_buy_ratio"] = np.where(out["volume"] > 0, out["taker_buy_base"] / out["volume"], np.nan)
    out["taker_buy_usd_ratio"] = np.where(out["Volume USD"] > 0, out["taker_buy_quote"] / out["Volume USD"], np.nan)
    out["log_taker_buy_base"] = np.log1p(out["taker_buy_base"].clip(lower=0))
    out["year"] = out["date"].dt.year
    out["days_from_start"] = (out["date"] - out["date"].min()).dt.total_seconds() / 86400.0
    return out


def analyze_time_series(df):
    sample = df.dropna(subset=["taker_buy_base"]).copy()
    if sample.empty:
        return {
            "coverage": {},
            "trend": {},
            "yearly": pd.DataFrame(),
            "verdict": "No taker_buy_base data available.",
        }

    yearly = (
        sample.groupby("year", as_index=False)
        .agg(
            count=("taker_buy_base", "size"),
            raw_mean=("taker_buy_base", "mean"),
            raw_median=("taker_buy_base", "median"),
            raw_std=("taker_buy_base", "std"),
            ratio_mean=("taker_buy_ratio", "mean"),
            ratio_median=("taker_buy_ratio", "median"),
            usd_ratio_mean=("taker_buy_usd_ratio", "mean"),
            usd_ratio_median=("taker_buy_usd_ratio", "median"),
        )
    )
    yearly["raw_cv"] = yearly["raw_std"] / yearly["raw_mean"].replace(0, np.nan)

    raw_log_trend = regression_stats(sample["days_from_start"], sample["log_taker_buy_base"])
    ratio_trend = regression_stats(sample["days_from_start"], sample["taker_buy_ratio"])
    usd_ratio_trend = regression_stats(sample["days_from_start"], sample["taker_buy_usd_ratio"])

    slope_to_year = 365.25
    raw_growth_per_year = math.expm1(raw_log_trend["slope"] * slope_to_year) if np.isfinite(raw_log_trend["slope"]) else np.nan

    raw_year_median_ratio = (
        yearly["raw_median"].max() / yearly["raw_median"].min()
        if len(yearly) and yearly["raw_median"].min() > 0
        else np.nan
    )
    ratio_year_median_ratio = (
        yearly["ratio_median"].max() / yearly["ratio_median"].min()
        if len(yearly) and yearly["ratio_median"].min() > 0
        else np.nan
    )

    raw_eta = year_group_effect_size(sample, "taker_buy_base")
    ratio_eta = year_group_effect_size(sample, "taker_buy_ratio")
    usd_ratio_eta = year_group_effect_size(sample, "taker_buy_usd_ratio")

    raw_jump = largest_year_jump(yearly, "raw_median")
    ratio_jump = largest_year_jump(yearly, "ratio_median")

    raw_stable = (
        np.isfinite(raw_growth_per_year)
        and abs(raw_growth_per_year) < 0.10
        and np.isfinite(raw_year_median_ratio)
        and raw_year_median_ratio < 1.5
        and np.isfinite(raw_eta)
        and raw_eta < 0.10
    )
    ratio_stable = (
        np.isfinite(ratio_year_median_ratio)
        and ratio_year_median_ratio < 1.5
        and np.isfinite(ratio_eta)
        and ratio_eta < 0.10
        and np.isfinite(usd_ratio_eta)
        and usd_ratio_eta < 0.10
    )

    if raw_stable:
        verdict = "Raw taker_buy_base looks reasonably stable across time."
    elif ratio_stable:
        verdict = "Raw taker_buy_base drifts over time. Use a ratio or normalized version instead."
    else:
        verdict = "Both raw and simple ratios shift across time. Use regime-aware normalization rather than a fixed absolute threshold."

    return {
        "coverage": {
            "rows_total": int(len(df)),
            "rows_with_taker_buy": int(len(sample)),
            "coverage_pct": float(len(sample) / len(df) * 100.0) if len(df) else np.nan,
            "first_valid_date": str(sample["date"].min()),
            "last_valid_date": str(sample["date"].max()),
        },
        "trend": {
            "raw_log_slope_per_day": raw_log_trend["slope"],
            "raw_growth_per_year": raw_growth_per_year,
            "raw_r2": raw_log_trend["r2"],
            "raw_p_value": raw_log_trend["p_value"],
            "ratio_slope_per_day": ratio_trend["slope"],
            "ratio_r2": ratio_trend["r2"],
            "ratio_p_value": ratio_trend["p_value"],
            "usd_ratio_slope_per_day": usd_ratio_trend["slope"],
            "usd_ratio_r2": usd_ratio_trend["r2"],
            "usd_ratio_p_value": usd_ratio_trend["p_value"],
            "raw_year_median_ratio_max_min": raw_year_median_ratio,
            "ratio_year_median_ratio_max_min": ratio_year_median_ratio,
            "raw_year_effect_eta2": raw_eta,
            "ratio_year_effect_eta2": ratio_eta,
            "usd_ratio_year_effect_eta2": usd_ratio_eta,
            "raw_largest_year_jump": raw_jump,
            "ratio_largest_year_jump": ratio_jump,
        },
        "yearly": yearly,
        "verdict": verdict,
    }


def assign_cycles(base_df, cycles_df):
    if base_df.empty or cycles_df.empty:
        return pd.DataFrame()

    sample = base_df.dropna(subset=["date", "taker_buy_base"]).copy()
    if sample.empty:
        return pd.DataFrame()

    cycle_starts = cycles_df["start_date"].to_numpy()
    cycle_ends = cycles_df["end_date"].to_numpy()
    timestamps = sample["date"].to_numpy()

    idx = np.searchsorted(cycle_ends, timestamps, side="left")
    valid = idx < len(cycles_df)
    sample = sample.loc[valid].copy()
    idx = idx[valid]

    start_ok = sample["date"].to_numpy() >= cycle_starts[idx]
    sample = sample.loc[start_ok].copy()
    idx = idx[start_ok]
    sample["cycle_id"] = cycles_df.iloc[idx]["cycle_id"].to_numpy()
    return sample


def analyze_cycles(base_df, cycles_df):
    assigned = assign_cycles(base_df, cycles_df)
    if assigned.empty:
        return {
            "cycle_summary": pd.DataFrame(),
            "cycle_type_summary": pd.DataFrame(),
            "overall": {},
            "duration_summary": pd.DataFrame(),
            "yearly": pd.DataFrame(),
            "trend": {},
        }

    cycle_agg = (
        assigned.groupby("cycle_id", as_index=False)
        .agg(
            taker_buy_sum=("taker_buy_base", "sum"),
            taker_buy_mean=("taker_buy_base", "mean"),
            taker_buy_median=("taker_buy_base", "median"),
            taker_buy_quote_sum=("taker_buy_quote", "sum"),
            volume_sum=("volume", "sum"),
            volume_usd_sum=("Volume USD", "sum"),
            candle_count=("date", "size"),
        )
    )
    cycle_agg["cycle_taker_buy_ratio"] = np.where(
        cycle_agg["volume_sum"] > 0,
        cycle_agg["taker_buy_sum"] / cycle_agg["volume_sum"],
        np.nan,
    )
    cycle_agg["cycle_taker_buy_usd_ratio"] = np.where(
        cycle_agg["volume_usd_sum"] > 0,
        cycle_agg["taker_buy_quote_sum"] / cycle_agg["volume_usd_sum"],
        np.nan,
    )

    merged = cycle_agg.merge(cycles_df, on="cycle_id", how="left")
    merged["year"] = merged["start_date"].dt.year
    merged["log_taker_buy_sum"] = np.log1p(merged["taker_buy_sum"].clip(lower=0))
    merged["days_from_start"] = (merged["start_date"] - merged["start_date"].min()).dt.total_seconds() / 86400.0
    merged["coverage_vs_duration"] = merged["candle_count"] / merged["duration_candles"].replace(0, np.nan)

    cycle_type_summary = (
        merged.groupby("cycle_type", as_index=False)
        .agg(
            cycles=("cycle_id", "size"),
            mean_sum=("taker_buy_sum", "mean"),
            median_sum=("taker_buy_sum", "median"),
            mean_ratio=("cycle_taker_buy_ratio", "mean"),
            median_ratio=("cycle_taker_buy_ratio", "median"),
            mean_duration=("duration_candles", "mean"),
        )
    )

    sum_trend = regression_stats(merged["days_from_start"], merged["log_taker_buy_sum"])
    ratio_trend = regression_stats(merged["days_from_start"], merged["cycle_taker_buy_ratio"])
    yearly = (
        merged.groupby("year", as_index=False)
        .agg(
            cycles=("cycle_id", "size"),
            taker_buy_sum_median=("taker_buy_sum", "median"),
            taker_buy_ratio_median=("cycle_taker_buy_ratio", "median"),
        )
    )

    duration_summary = (
        merged.assign(
            duration_bucket=pd.cut(
                merged["duration_candles"],
                bins=[0, 4, 7, 12, 20, np.inf],
                labels=["1-4", "5-7", "8-12", "13-20", "21+"],
                right=True,
            )
        )
        .groupby("duration_bucket", as_index=False)
        .agg(
            cycles=("cycle_id", "size"),
            sum_median=("taker_buy_sum", "median"),
            ratio_median=("cycle_taker_buy_ratio", "median"),
            usd_ratio_median=("cycle_taker_buy_usd_ratio", "median"),
        )
    )

    mean_cycle_sum = merged["taker_buy_sum"].mean()
    std_cycle_sum = merged["taker_buy_sum"].std()
    overall = {
        "cycles": int(len(merged)),
        "matched_candles": int(merged["candle_count"].sum()),
        "median_cycle_sum": float(merged["taker_buy_sum"].median()),
        "mean_cycle_sum": float(mean_cycle_sum),
        "std_cycle_sum": float(std_cycle_sum),
        "cv_cycle_sum": float(std_cycle_sum / mean_cycle_sum) if pd.notna(mean_cycle_sum) and mean_cycle_sum != 0 else np.nan,
        "median_cycle_ratio": float(merged["cycle_taker_buy_ratio"].median()),
        "mean_cycle_ratio": float(merged["cycle_taker_buy_ratio"].mean()),
        "median_cycle_usd_ratio": float(merged["cycle_taker_buy_usd_ratio"].median()),
        "mean_cycle_usd_ratio": float(merged["cycle_taker_buy_usd_ratio"].mean()),
        "median_coverage_vs_duration": float(merged["coverage_vs_duration"].median()),
        "mean_coverage_vs_duration": float(merged["coverage_vs_duration"].mean()),
        "year_effect_eta2_sum": float(year_group_effect_size(merged, "taker_buy_sum")),
        "year_effect_eta2_ratio": float(year_group_effect_size(merged, "cycle_taker_buy_ratio")),
        "cycle_sum_yearly_max_min": float(
            yearly["taker_buy_sum_median"].max() / yearly["taker_buy_sum_median"].min()
            if len(yearly) and yearly["taker_buy_sum_median"].min() > 0
            else np.nan
        ),
        "cycle_ratio_yearly_max_min": float(
            yearly["taker_buy_ratio_median"].max() / yearly["taker_buy_ratio_median"].min()
            if len(yearly) and yearly["taker_buy_ratio_median"].min() > 0
            else np.nan
        ),
    }

    return {
        "cycle_summary": merged.sort_values("start_date").reset_index(drop=True),
        "cycle_type_summary": cycle_type_summary,
        "overall": overall,
        "duration_summary": duration_summary,
        "yearly": yearly,
        "trend": {
            "cycle_sum_growth_per_year": math.expm1(sum_trend["slope"] * 365.25) if np.isfinite(sum_trend["slope"]) else np.nan,
            "cycle_sum_r2": sum_trend["r2"],
            "cycle_sum_p_value": sum_trend["p_value"],
            "cycle_ratio_slope_per_day": ratio_trend["slope"],
            "cycle_ratio_r2": ratio_trend["r2"],
            "cycle_ratio_p_value": ratio_trend["p_value"],
        },
    }


def write_timeframe_outputs(output_dir, timeframe, ts_result, cycle_result):
    ts_result["yearly"].to_csv(output_dir / f"{timeframe}_yearly_summary.csv", index=False)
    cycle_result["cycle_summary"].to_csv(output_dir / f"{timeframe}_cycle_summary.csv", index=False)
    cycle_result["cycle_type_summary"].to_csv(output_dir / f"{timeframe}_cycle_type_summary.csv", index=False)
    cycle_result["duration_summary"].to_csv(output_dir / f"{timeframe}_cycle_duration_summary.csv", index=False)
    if "yearly" in cycle_result:
        cycle_result["yearly"].to_csv(output_dir / f"{timeframe}_cycle_yearly_summary.csv", index=False)


def format_float(value, digits=4):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def build_markdown_report(results):
    lines = [
        "# Taker Buy Base Stability Analysis",
        "",
        "Goal: decide whether `taker_buy_base` can be used as a raw absolute value, or whether it needs normalization.",
        "",
    ]

    for timeframe, payload in results.items():
        coverage = payload["time_series"]["coverage"]
        trend = payload["time_series"]["trend"]
        cycle_trend = payload["cycles"]["trend"]
        lines.extend(
            [
                f"## {timeframe}",
                "",
                f"- Verdict: {payload['time_series']['verdict']}",
                f"- Coverage: {coverage.get('rows_with_taker_buy', 0)}/{coverage.get('rows_total', 0)} rows "
                f"({format_float(coverage.get('coverage_pct', np.nan), 2)}%), "
                f"{coverage.get('first_valid_date', 'n/a')} to {coverage.get('last_valid_date', 'n/a')}",
                f"- Raw yearly growth estimate: {format_float(trend.get('raw_growth_per_year', np.nan) * 100 if np.isfinite(trend.get('raw_growth_per_year', np.nan)) else np.nan, 2)}% / year",
                f"- Raw trend R^2: {format_float(trend.get('raw_r2', np.nan), 4)}, p≈{format_float(trend.get('raw_p_value', np.nan), 6)}",
                f"- Raw median max/min by year: {format_float(trend.get('raw_year_median_ratio_max_min', np.nan), 3)}",
                f"- Ratio median max/min by year: {format_float(trend.get('ratio_year_median_ratio_max_min', np.nan), 3)}",
                f"- Year effect eta^2: raw={format_float(trend.get('raw_year_effect_eta2', np.nan), 3)}, "
                f"ratio={format_float(trend.get('ratio_year_effect_eta2', np.nan), 3)}, "
                f"usd_ratio={format_float(trend.get('usd_ratio_year_effect_eta2', np.nan), 3)}",
                f"- Cycle sum growth estimate: {format_float(cycle_trend.get('cycle_sum_growth_per_year', np.nan) * 100 if np.isfinite(cycle_trend.get('cycle_sum_growth_per_year', np.nan)) else np.nan, 2)}% / year",
                f"- Cycle ratio trend R^2: {format_float(cycle_trend.get('cycle_ratio_r2', np.nan), 4)}, p≈{format_float(cycle_trend.get('cycle_ratio_p_value', np.nan), 6)}",
                "",
            ]
        )

        raw_jump = trend.get("raw_largest_year_jump")
        ratio_jump = trend.get("ratio_largest_year_jump")
        if raw_jump:
            lines.append(
                f"- Largest raw median jump: {raw_jump['from_year']} -> {raw_jump['to_year']} "
                f"({format_float(raw_jump['delta'], 4)})"
            )
        if ratio_jump:
            lines.append(
                f"- Largest ratio median jump: {ratio_jump['from_year']} -> {ratio_jump['to_year']} "
                f"({format_float(ratio_jump['delta'], 6)})"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_cycle_window_markdown_report(results):
    lines = [
        "# Cycle-Window Taker Buy Analysis",
        "",
        "This report uses only candles inside each cycle window, from the first candle to the last candle inclusive.",
        "",
    ]

    for timeframe, payload in results.items():
        cycle_payload = payload["cycles"]
        overall = cycle_payload["overall"]
        trend = cycle_payload["trend"]
        type_summary = cycle_payload["cycle_type_summary"]
        duration_summary = cycle_payload["duration_summary"]

        lines.extend(
            [
                f"## {timeframe}",
                "",
                f"- Cycles analyzed: {overall.get('cycles', 0)}",
                f"- Matched candles inside cycle windows: {overall.get('matched_candles', 0)}",
                f"- Coverage vs duration: median={format_float(overall.get('median_coverage_vs_duration', np.nan), 3)}, "
                f"mean={format_float(overall.get('mean_coverage_vs_duration', np.nan), 3)}",
                f"- Cycle taker_buy_sum: median={format_float(overall.get('median_cycle_sum', np.nan), 2)}, "
                f"mean={format_float(overall.get('mean_cycle_sum', np.nan), 2)}, "
                f"std={format_float(overall.get('std_cycle_sum', np.nan), 2)}, "
                f"cv={format_float(overall.get('cv_cycle_sum', np.nan), 3)}",
                f"- Cycle taker_buy_ratio: median={format_float(overall.get('median_cycle_ratio', np.nan), 4)}, "
                f"mean={format_float(overall.get('mean_cycle_ratio', np.nan), 4)}",
                f"- Cycle taker_buy_usd_ratio: median={format_float(overall.get('median_cycle_usd_ratio', np.nan), 4)}, "
                f"mean={format_float(overall.get('mean_cycle_usd_ratio', np.nan), 4)}",
                f"- Year effect eta^2: sum={format_float(overall.get('year_effect_eta2_sum', np.nan), 3)}, "
                f"ratio={format_float(overall.get('year_effect_eta2_ratio', np.nan), 3)}",
                f"- Yearly median max/min: cycle_sum={format_float(overall.get('cycle_sum_yearly_max_min', np.nan), 3)}, "
                f"cycle_ratio={format_float(overall.get('cycle_ratio_yearly_max_min', np.nan), 3)}",
                f"- Cycle-sum growth estimate: {format_float(trend.get('cycle_sum_growth_per_year', np.nan) * 100 if np.isfinite(trend.get('cycle_sum_growth_per_year', np.nan)) else np.nan, 2)}% / year",
                f"- Cycle-sum trend R^2={format_float(trend.get('cycle_sum_r2', np.nan), 4)}, p~{format_float(trend.get('cycle_sum_p_value', np.nan), 6)}",
                f"- Cycle-ratio trend R^2={format_float(trend.get('cycle_ratio_r2', np.nan), 4)}, p~{format_float(trend.get('cycle_ratio_p_value', np.nan), 6)}",
                "",
                "Cycle type summary:",
            ]
        )

        if type_summary.empty:
            lines.append("- No cycle type summary available.")
        else:
            for _, row in type_summary.iterrows():
                lines.append(
                    f"- {row['cycle_type']}: cycles={int(row['cycles'])}, "
                    f"median_sum={format_float(row['median_sum'], 2)}, "
                    f"median_ratio={format_float(row['median_ratio'], 4)}, "
                    f"mean_duration={format_float(row['mean_duration'], 2)}"
                )

        lines.extend(["", "Duration bucket summary:"])
        if duration_summary.empty:
            lines.append("- No duration summary available.")
        else:
            for _, row in duration_summary.iterrows():
                lines.append(
                    f"- {row['duration_bucket']}: cycles={int(row['cycles'])}, "
                    f"sum_median={format_float(row['sum_median'], 2)}, "
                    f"ratio_median={format_float(row['ratio_median'], 4)}, "
                    f"usd_ratio_median={format_float(row['usd_ratio_median'], 4)}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_ratio_markdown_report(results):
    lines = [
        "# Taker Buy Ratio Analysis",
        "",
        "This report focuses on normalized metrics rather than raw taker_buy_base.",
        "- `taker_buy_ratio` = taker_buy_base / volume",
        "- `cycle_taker_buy_ratio` = sum(taker_buy_base inside cycle) / sum(volume inside cycle)",
        "- `taker_buy_usd_ratio` = taker_buy_quote / Volume USD",
        "",
    ]

    for timeframe, payload in results.items():
        ts = payload["time_series"]
        cycle_payload = payload["cycles"]
        trend = ts["trend"]
        overall = cycle_payload["overall"]
        cycle_trend = cycle_payload["trend"]
        type_summary = cycle_payload["cycle_type_summary"]
        duration_summary = cycle_payload["duration_summary"]

        if not overall:
            continue

        if overall.get("year_effect_eta2_ratio", np.nan) < 0.02 and overall.get("cycle_ratio_yearly_max_min", np.nan) < 1.05:
            ratio_verdict = "Ratio is very stable across time and looks usable as a shared threshold feature."
        elif overall.get("year_effect_eta2_ratio", np.nan) < 0.08 and overall.get("cycle_ratio_yearly_max_min", np.nan) < 1.10:
            ratio_verdict = "Ratio is fairly stable, but mild regime awareness is still safer."
        else:
            ratio_verdict = "Ratio is better than raw values, but still regime-sensitive at this timeframe."

        lines.extend(
            [
                f"## {timeframe}",
                "",
                f"- Ratio verdict: {ratio_verdict}",
                f"- Candle-level ratio median max/min by year: {format_float(trend.get('ratio_year_median_ratio_max_min', np.nan), 3)}",
                f"- Candle-level ratio year effect eta^2: {format_float(trend.get('ratio_year_effect_eta2', np.nan), 3)}",
                f"- Candle-level ratio trend R^2: {format_float(trend.get('ratio_r2', np.nan), 4)}, p~{format_float(trend.get('ratio_p_value', np.nan), 6)}",
                f"- Cycle-level ratio median: {format_float(overall.get('median_cycle_ratio', np.nan), 4)}",
                f"- Cycle-level ratio mean: {format_float(overall.get('mean_cycle_ratio', np.nan), 4)}",
                f"- Cycle-level yearly max/min: {format_float(overall.get('cycle_ratio_yearly_max_min', np.nan), 3)}",
                f"- Cycle-level year effect eta^2: {format_float(overall.get('year_effect_eta2_ratio', np.nan), 3)}",
                f"- Cycle-level ratio trend R^2: {format_float(cycle_trend.get('cycle_ratio_r2', np.nan), 4)}, p~{format_float(cycle_trend.get('cycle_ratio_p_value', np.nan), 6)}",
                f"- USD ratio year effect eta^2: {format_float(trend.get('usd_ratio_year_effect_eta2', np.nan), 3)}",
                "",
                "Cycle type summary:",
            ]
        )

        if type_summary.empty:
            lines.append("- No cycle type summary available.")
        else:
            for _, row in type_summary.iterrows():
                lines.append(
                    f"- {row['cycle_type']}: median_ratio={format_float(row['median_ratio'], 4)}, "
                    f"mean_ratio={format_float(row['mean_ratio'], 4)}, cycles={int(row['cycles'])}"
                )

        lines.extend(["", "Duration bucket ratio summary:"])
        if duration_summary.empty:
            lines.append("- No duration summary available.")
        else:
            for _, row in duration_summary.iterrows():
                lines.append(
                    f"- {row['duration_bucket']}: ratio_median={format_float(row['ratio_median'], 4)}, "
                    f"usd_ratio_median={format_float(row['usd_ratio_median'], 4)}, cycles={int(row['cycles'])}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def print_console_summary(results):
    print("\n" + "=" * 120)
    print(f"{'TF':<5} {'Coverage%':>10} {'RawGrowth/Y':>12} {'RawMedMaxMin':>14} {'RatioMedMaxMin':>16} {'Verdict':<50}")
    print("=" * 120)
    for timeframe, payload in results.items():
        coverage = payload["time_series"]["coverage"]
        trend = payload["time_series"]["trend"]
        raw_growth = trend.get("raw_growth_per_year", np.nan)
        print(
            f"{timeframe:<5} "
            f"{format_float(coverage.get('coverage_pct', np.nan), 2):>10} "
            f"{format_float(raw_growth * 100 if np.isfinite(raw_growth) else np.nan, 2):>12} "
            f"{format_float(trend.get('raw_year_median_ratio_max_min', np.nan), 3):>14} "
            f"{format_float(trend.get('ratio_year_median_ratio_max_min', np.nan), 3):>16} "
            f"{payload['time_series']['verdict']:<50}"
        )


def main():
    project_root = find_project_root()
    cycle_data_dir = resolve_cycle_data_dir(project_root)
    output_dir = ensure_output_dir(project_root)

    print(f"Project root: {project_root}")
    print(f"Cycle data dir: {cycle_data_dir}")
    print(f"Output dir: {output_dir}")

    all_results = {}
    summary_for_json = {}

    for timeframe in TIMEFRAMES:
        base_df = add_scaled_features(load_base_csv(project_root, timeframe))
        cycles_df = load_cycles(cycle_data_dir, timeframe)
        ts_result = analyze_time_series(base_df)
        cycle_result = analyze_cycles(base_df, cycles_df)
        write_timeframe_outputs(output_dir, timeframe, ts_result, cycle_result)

        all_results[timeframe] = {
            "time_series": ts_result,
            "cycles": cycle_result,
        }
        summary_for_json[timeframe] = {
            "coverage": ts_result["coverage"],
            "trend": ts_result["trend"],
            "cycle_overall": cycle_result["overall"],
            "cycle_trend": cycle_result["trend"],
            "verdict": ts_result["verdict"],
        }

    report = build_markdown_report(all_results)
    cycle_window_report = build_cycle_window_markdown_report(all_results)
    ratio_report = build_ratio_markdown_report(all_results)
    (output_dir / "taker_buy_analysis_report.md").write_text(report, encoding="utf-8")
    (output_dir / "taker_buy_cycle_window_report.md").write_text(cycle_window_report, encoding="utf-8")
    (output_dir / "taker_buy_ratio_report.md").write_text(ratio_report, encoding="utf-8")
    (output_dir / "taker_buy_analysis_summary.json").write_text(
        json.dumps(summary_for_json, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print_console_summary(all_results)
    print(f"\nSaved report: {output_dir / 'taker_buy_analysis_report.md'}")
    print(f"Saved cycle-window report: {output_dir / 'taker_buy_cycle_window_report.md'}")
    print(f"Saved ratio report: {output_dir / 'taker_buy_ratio_report.md'}")


if __name__ == "__main__":
    main()
