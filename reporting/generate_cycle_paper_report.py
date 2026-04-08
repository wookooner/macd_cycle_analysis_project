from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = PROJECT_ROOT / "data" / "cycle_data" / "structured"
BASE_DATA_DIR = PROJECT_ROOT / "data" / "base_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_results" / "paper_report"
TIMEFRAMES = ("1w", "1d", "4h", "1h")
ADJACENT_PAIRS = (("1w", "1d"), ("1d", "4h"), ("4h", "1h"))


def _find_cycle_parquet(timeframe: str, asset: str = "btc") -> Path:
    candidates = [
        STRUCTURED_DIR / asset / f"cycles_{timeframe}.parquet",
        STRUCTURED_DIR / f"cycles_{timeframe}.parquet",
        STRUCTURED_DIR / asset / f"cycles_{timeframe}_enriched.parquet",
        STRUCTURED_DIR / f"cycles_{timeframe}_enriched.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing cycle parquet for timeframe={timeframe}")


def _find_hierarchy_map(asset: str = "btc") -> Path:
    candidates = [
        STRUCTURED_DIR / asset / "cycle_hierarchy_map.json",
        STRUCTURED_DIR / "cycle_hierarchy_map.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Missing cycle_hierarchy_map.json")


def _normalize_direction(value: object) -> str:
    text = str(value or "").strip().lower()
    if "up" in text or text == "u":
        return "UP"
    if "down" in text or text == "d":
        return "DOWN"
    return "UNKNOWN"


def _flatten_dict(data: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in data.items():
        next_prefix = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, next_prefix))
        else:
            flat[next_prefix] = value
    return flat


def _extract_cycle_close_prices(row: pd.Series) -> tuple[float | None, float | None]:
    candle_data = row.get("candle_data")
    if candle_data is None or not hasattr(candle_data, "__len__") or len(candle_data) == 0:
        return None, None
    try:
        first = candle_data[0]
        last = candle_data[-1]
        first_close = pd.to_numeric(first.get("close"), errors="coerce")
        last_close = pd.to_numeric(last.get("close"), errors="coerce")
        return (
            float(first_close) if pd.notna(first_close) else None,
            float(last_close) if pd.notna(last_close) else None,
        )
    except Exception:
        return None, None


def _pct_change(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start == 0:
        return None
    return float((end / start - 1.0) * 100.0)


def _safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 2:
        return None
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _quantile_summary(series: pd.Series) -> dict[str, float | int]:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return {"n": 0}
    return {
        "n": int(cleaned.count()),
        "mean": float(cleaned.mean()),
        "median": float(cleaned.median()),
        "std": float(cleaned.std(ddof=1)) if len(cleaned) > 1 else 0.0,
        "min": float(cleaned.min()),
        "q05": float(cleaned.quantile(0.05)),
        "q25": float(cleaned.quantile(0.25)),
        "q75": float(cleaned.quantile(0.75)),
        "q95": float(cleaned.quantile(0.95)),
        "max": float(cleaned.max()),
    }


def _save_table(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"


def _annotate_bar_containers(ax: plt.Axes, fmt: str = "{:.2f}", fontsize: int = 9) -> None:
    for container in ax.containers:
        labels = []
        has_value = False
        for patch in container:
            height = patch.get_height()
            if pd.isna(height):
                labels.append("")
                continue
            has_value = True
            labels.append(fmt.format(height))
        if has_value:
            ax.bar_label(container, labels=labels, padding=3, fontsize=fontsize)


def _move_legend_below(ax: plt.Axes, ncol: int = 2, title: str | None = None) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    if title is not None:
        legend.set_title(title)
    sns.move_legend(
        ax,
        "upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        frameon=False,
        title=title,
    )


def load_cycle_data(timeframes: tuple[str, ...] = TIMEFRAMES, asset: str = "btc") -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for timeframe in timeframes:
        raw = pd.read_parquet(_find_cycle_parquet(timeframe, asset=asset)).copy()
        flat_rows = []
        for _, row in raw.iterrows():
            first_close, last_close = _extract_cycle_close_prices(row)
            record = {
                "cycle_id": row.get("cycle_id"),
                "timeframe": timeframe,
                "cycle_type": _normalize_direction(row.get("cycle_type")),
                "start_date": pd.to_datetime(row.get("start_date"), errors="coerce"),
                "end_date": pd.to_datetime(row.get("end_date"), errors="coerce"),
                "duration_candles": pd.to_numeric(row.get("duration_candles"), errors="coerce"),
                "cycle_first_close": first_close,
                "cycle_last_close": last_close,
            }
            features = row.get("cycle_features")
            if isinstance(features, dict):
                record.update(_flatten_dict(features))
            flat_rows.append(record)

        df = pd.DataFrame(flat_rows)
        df["theoretical_price_pct"] = pd.to_numeric(df.get("change_price_pct"), errors="coerce")
        df["ppo_hist_change"] = pd.to_numeric(df.get("change_ppo_hist"), errors="coerce")
        df["area_ppo_hist"] = pd.to_numeric(df.get("aggregate_area_ppo_hist"), errors="coerce")
        df["noise_count"] = pd.to_numeric(df.get("shape_noise_count"), errors="coerce")
        df["direction_pct"] = pd.to_numeric(df.get("strength_direction_pct"), errors="coerce")
        df["peak_price_position"] = pd.to_numeric(df.get("shape_peak_price_position"), errors="coerce")
        df["trough_price_position"] = pd.to_numeric(df.get("shape_trough_price_position"), errors="coerce")
        df["duration_candles"] = pd.to_numeric(df.get("duration_candles"), errors="coerce")
        df = df.sort_values("start_date").reset_index(drop=True)
        df["next_cycle_id"] = df["cycle_id"].shift(-1)
        df["next_cycle_type"] = df["cycle_type"].shift(-1)
        df["next_cycle_start_date"] = df["start_date"].shift(-1)
        df["next_cycle_first_close"] = pd.to_numeric(df["cycle_first_close"].shift(-1), errors="coerce")
        df["realized_price_pct"] = [
            _pct_change(start, end)
            for start, end in zip(df["cycle_first_close"], df["next_cycle_first_close"])
        ]
        df["confirmation_gap_pct"] = [
            _pct_change(start, end)
            for start, end in zip(df["cycle_last_close"], df["next_cycle_first_close"])
        ]
        df["theoretical_match_expected_direction"] = np.where(
            df["cycle_type"].eq("UP"),
            df["theoretical_price_pct"] > 0,
            np.where(df["cycle_type"].eq("DOWN"), df["theoretical_price_pct"] < 0, False),
        )
        df["realized_match_expected_direction"] = np.where(
            df["cycle_type"].eq("UP"),
            df["realized_price_pct"] > 0,
            np.where(df["cycle_type"].eq("DOWN"), df["realized_price_pct"] < 0, False),
        )
        df["theoretical_flat_price_move"] = df["theoretical_price_pct"].eq(0)
        df["realized_flat_price_move"] = df["realized_price_pct"].eq(0)
        df["theoretical_abs_loss_pct"] = np.where(df["cycle_type"].eq("DOWN"), df["theoretical_price_pct"].abs(), np.nan)
        df["realized_abs_loss_pct"] = np.where(df["cycle_type"].eq("DOWN"), df["realized_price_pct"].abs(), np.nan)
        df["noise_ratio"] = np.where(
            df["duration_candles"] > 0,
            df["noise_count"] / df["duration_candles"],
            np.nan,
        )
        data[timeframe] = df
    return data


def load_candle_counts(timeframes: tuple[str, ...] = TIMEFRAMES, symbol: str = "BTCUSD") -> pd.DataFrame:
    rows = []
    for timeframe in timeframes:
        csv_path = BASE_DATA_DIR / f"{symbol}_{timeframe}.csv"
        count = len(pd.read_csv(csv_path, usecols=["date"])) if csv_path.exists() else np.nan
        rows.append({"timeframe": timeframe, "total_candles": count})
    return pd.DataFrame(rows)


def section_3_1_dataset_overview(data: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    candle_counts = load_candle_counts()
    rows = []
    for timeframe, df in data.items():
        total_cycles = len(df)
        up_cycles = int(df["cycle_type"].eq("UP").sum())
        down_cycles = int(df["cycle_type"].eq("DOWN").sum())
        candle_count = candle_counts.loc[candle_counts["timeframe"].eq(timeframe), "total_candles"].iloc[0]
        rows.append(
            {
                "timeframe": timeframe,
                "total_candles": int(candle_count) if pd.notna(candle_count) else None,
                "detected_cycles": total_cycles,
                "up_cycles": up_cycles,
                "down_cycles": down_cycles,
                "up_ratio": up_cycles / total_cycles if total_cycles else np.nan,
                "down_ratio": down_cycles / total_cycles if total_cycles else np.nan,
                "avg_candles_per_cycle": candle_count / total_cycles if total_cycles and pd.notna(candle_count) else np.nan,
                "realized_valid_cycles": int(df["realized_price_pct"].notna().sum()),
                "realized_valid_ratio": float(df["realized_price_pct"].notna().mean()),
            }
        )

    overview = pd.DataFrame(rows)
    _save_table(overview, output_dir, "3_1_dataset_overview")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(bottom=0.2, wspace=0.25)
    sns.barplot(data=overview, x="timeframe", y="detected_cycles", ax=axes[0], color="#4c78a8")
    axes[0].set_title("Detected cycles by timeframe")
    axes[0].set_xlabel("Timeframe")
    axes[0].set_ylabel("Cycles")
    _annotate_bar_containers(axes[0], "{:.0f}")

    ratio_plot = overview.melt(
        id_vars="timeframe",
        value_vars=["up_ratio", "down_ratio"],
        var_name="direction",
        value_name="ratio",
    )
    sns.barplot(data=ratio_plot, x="timeframe", y="ratio", hue="direction", ax=axes[1], palette=["#2ca02c", "#d62728"])
    axes[1].set_title("UP/DOWN cycle ratio")
    axes[1].set_xlabel("Timeframe")
    axes[1].set_ylabel("Ratio")
    axes[1].set_ylim(0, 1)
    _annotate_bar_containers(axes[1], "{:.2f}")
    _move_legend_below(axes[1], ncol=2, title="")

    fig.savefig(output_dir / "3_1_dataset_overview.png")
    plt.close(fig)
    return overview


def section_3_2_direction_accuracy(data: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    rows = []
    basis_specs = [
        ("theoretical", "theoretical_price_pct", "theoretical_match_expected_direction", "theoretical_flat_price_move"),
        ("realized", "realized_price_pct", "realized_match_expected_direction", "realized_flat_price_move"),
    ]
    for timeframe, df in data.items():
        for basis, pct_col, match_col, flat_col in basis_specs:
            for direction in ("UP", "DOWN"):
                subset = df[df["cycle_type"].eq(direction)].copy()
                subset = subset[subset[pct_col].notna()].copy()
                if subset.empty:
                    continue
                match_count = int(subset[match_col].sum())
                total = len(subset)
                rows.append(
                    {
                        "timeframe": timeframe,
                        "basis": basis,
                        "cycle_type": direction,
                        "sample_size": total,
                        "matched_direction_count": match_count,
                        "flat_move_count": int(subset[flat_col].sum()),
                        "direction_accuracy": match_count / total if total else np.nan,
                        "mean_price_pct": float(subset[pct_col].mean()),
                        "median_price_pct": float(subset[pct_col].median()),
                    }
                )

    accuracy = pd.DataFrame(rows)
    _save_table(accuracy, output_dir, "3_2_direction_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.subplots_adjust(bottom=0.22, wspace=0.25)
    for ax, basis in zip(axes, ["theoretical", "realized"]):
        subset = accuracy[accuracy["basis"].eq(basis)]
        sns.barplot(
            data=subset,
            x="timeframe",
            y="direction_accuracy",
            hue="cycle_type",
            ax=ax,
            palette={"UP": "#2ca02c", "DOWN": "#d62728"},
        )
        ax.set_title(f"{basis.capitalize()} direction accuracy")
        ax.set_xlabel("Timeframe")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        _annotate_bar_containers(ax, "{:.2f}")
        _move_legend_below(ax, ncol=2, title="")
    fig.savefig(output_dir / "3_2_direction_accuracy.png")
    plt.close(fig)
    return accuracy


def section_3_3_price_distribution(data: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    rows = []
    corr_rows = []
    plot_rows = []
    basis_specs = [
        ("theoretical", "theoretical_price_pct"),
        ("realized", "realized_price_pct"),
    ]

    for timeframe, df in data.items():
        for basis, pct_col in basis_specs:
            for direction in ("UP", "DOWN"):
                subset = df[df["cycle_type"].eq(direction)].copy()
                subset = subset[subset[pct_col].notna()].copy()
                if subset.empty:
                    continue
                price_stats = _quantile_summary(subset[pct_col])
                ppo_stats = _quantile_summary(subset["ppo_hist_change"])
                rows.append(
                    {
                        "timeframe": timeframe,
                        "basis": basis,
                        "cycle_type": direction,
                        **{f"price_pct_{k}": v for k, v in price_stats.items()},
                        **{f"ppo_hist_change_{k}": v for k, v in ppo_stats.items()},
                    }
                )
                corr_rows.append(
                    {
                        "timeframe": timeframe,
                        "basis": basis,
                        "cycle_type": direction,
                        "corr_price_pct_vs_ppo_hist_change": _safe_corr(subset[pct_col], subset["ppo_hist_change"]),
                        "corr_price_pct_vs_area_ppo_hist": _safe_corr(subset[pct_col], subset["area_ppo_hist"]),
                        "corr_price_pct_vs_confirmation_gap": _safe_corr(subset[pct_col], subset["confirmation_gap_pct"]),
                    }
                )
                sample = subset[["timeframe", "cycle_type", "area_ppo_hist"]].copy()
                sample["basis"] = basis
                sample["price_pct"] = subset[pct_col]
                plot_rows.append(sample.dropna())

    distribution = pd.DataFrame(rows)
    corr_df = pd.DataFrame(corr_rows)
    _save_table(distribution, output_dir, "3_3_price_pct_distribution")
    _save_table(corr_df, output_dir, "3_3_price_pct_ppo_relationship")

    plot_df = pd.concat(plot_rows, ignore_index=True) if plot_rows else pd.DataFrame()
    if not plot_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.subplots_adjust(bottom=0.14, hspace=0.38, wspace=0.22)
        for ax, timeframe in zip(axes.flatten(), TIMEFRAMES):
            subset = plot_df[plot_df["timeframe"].eq(timeframe)].copy()
            if subset.empty:
                ax.set_visible(False)
                continue
            sns.boxplot(
                data=subset,
                x="basis",
                y="price_pct",
                hue="cycle_type",
                ax=ax,
                palette={"UP": "#2ca02c", "DOWN": "#d62728"},
            )
            ax.axhline(0, color="black", linewidth=1, alpha=0.45)
            ax.set_title(f"{timeframe} price_pct distribution")
            ax.set_xlabel("")
            ax.set_ylabel("price_pct (%)")
            _move_legend_below(ax, ncol=2, title="")
        fig.savefig(output_dir / "3_3_price_pct_distribution.png")
        plt.close(fig)

        ppo_fig, ppo_axes = plt.subplots(2, 2, figsize=(18, 12))
        ppo_fig.subplots_adjust(bottom=0.14, hspace=0.38, wspace=0.22)
        for ax, timeframe in zip(ppo_axes.flatten(), TIMEFRAMES):
            subset = plot_df[plot_df["timeframe"].eq(timeframe)].copy()
            if subset.empty:
                ax.set_visible(False)
                continue
            sns.scatterplot(
                data=subset,
                x="area_ppo_hist",
                y="price_pct",
                hue="cycle_type",
                style="basis",
                alpha=0.5,
                s=45,
                ax=ax,
                palette={"UP": "#2ca02c", "DOWN": "#d62728"},
            )
            ax.axhline(0, color="black", linewidth=1, alpha=0.45)
            ax.set_title(f"{timeframe} price_pct vs PPO-hist area")
            ax.set_xlabel("area_ppo_hist")
            ax.set_ylabel("price_pct (%)")
            _move_legend_below(ax, ncol=4, title="")
        ppo_fig.savefig(output_dir / "3_3_price_pct_ppo_relationship.png")
        plt.close(ppo_fig)
    return distribution


def section_3_4_duration_distribution(data: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    rows = []
    plot_rows = []
    for timeframe, df in data.items():
        for direction in ("UP", "DOWN"):
            subset = df[df["cycle_type"].eq(direction)].copy()
            if subset.empty:
                continue
            rows.append(
                {
                    "timeframe": timeframe,
                    "cycle_type": direction,
                    **_quantile_summary(subset["duration_candles"]),
                }
            )
            plot_rows.append(subset[["timeframe", "cycle_type", "duration_candles"]].dropna())

    duration = pd.DataFrame(rows)
    _save_table(duration, output_dir, "3_4_duration_distribution")

    plot_df = pd.concat(plot_rows, ignore_index=True) if plot_rows else pd.DataFrame()
    if not plot_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.subplots_adjust(bottom=0.24, wspace=0.25)
        sns.boxplot(
            data=plot_df,
            x="timeframe",
            y="duration_candles",
            hue="cycle_type",
            ax=axes[0],
            palette={"UP": "#2ca02c", "DOWN": "#d62728"},
        )
        axes[0].set_title("Cycle duration distribution")
        axes[0].set_xlabel("Timeframe")
        axes[0].set_ylabel("Duration (candles)")
        _move_legend_below(axes[0], ncol=2, title="")

        fig.savefig(output_dir / "3_4_duration_distribution.png")
        plt.close(fig)

        hist_fig, hist_axes = plt.subplots(2, 2, figsize=(18, 12))
        hist_fig.subplots_adjust(bottom=0.08, hspace=0.45, wspace=0.25)
        for ax, timeframe in zip(hist_axes.flatten(), TIMEFRAMES):
            subset = plot_df[plot_df["timeframe"].eq(timeframe)]
            if subset.empty:
                ax.set_visible(False)
                continue
            sns.histplot(
                data=subset,
                x="duration_candles",
                hue="cycle_type",
                multiple="layer",
                kde=True,
                bins=25,
                palette={"UP": "#2ca02c", "DOWN": "#d62728"},
                ax=ax,
            )
            ax.set_title(f"{timeframe} duration histogram")
            ax.set_xlabel("Duration (candles)")
            ax.set_ylabel("Count")
            _move_legend_below(ax, ncol=2, title="")
        hist_fig.savefig(output_dir / "3_4_duration_histograms.png")
        plt.close(hist_fig)
    return duration


def section_3_5_structural_asymmetry(data: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    rows = []
    basis_specs = [
        ("theoretical", "theoretical_price_pct", "theoretical_abs_loss_pct", "theoretical_match_expected_direction"),
        ("realized", "realized_price_pct", "realized_abs_loss_pct", "realized_match_expected_direction"),
    ]
    for timeframe, df in data.items():
        for basis, pct_col, abs_loss_col, match_col in basis_specs:
            up = df[df["cycle_type"].eq("UP") & df[pct_col].notna()].copy()
            down = df[df["cycle_type"].eq("DOWN") & df[pct_col].notna()].copy()
            if up.empty or down.empty:
                continue
            up_mean_gain = float(up[pct_col].mean())
            down_mean_loss_abs = float(down[abs_loss_col].mean())
            rows.append(
                {
                    "timeframe": timeframe,
                    "basis": basis,
                    "up_cycle_count": int(len(up)),
                    "down_cycle_count": int(len(down)),
                    "count_ratio_up_to_down": len(up) / len(down) if len(down) else np.nan,
                    "up_mean_price_pct": up_mean_gain,
                    "down_mean_abs_loss_pct": down_mean_loss_abs,
                    "payoff_ratio_up_gain_to_down_loss": up_mean_gain / down_mean_loss_abs if down_mean_loss_abs else np.nan,
                    "up_mean_duration": float(up["duration_candles"].mean()),
                    "down_mean_duration": float(down["duration_candles"].mean()),
                    "duration_ratio_up_to_down": float(up["duration_candles"].mean() / down["duration_candles"].mean()) if down["duration_candles"].mean() else np.nan,
                    "up_direction_accuracy": float(up[match_col].mean()),
                    "down_direction_accuracy": float(down[match_col].mean()),
                    "mean_confirmation_gap_pct": float(df["confirmation_gap_pct"].dropna().mean()) if df["confirmation_gap_pct"].notna().any() else np.nan,
                }
            )

    asymmetry = pd.DataFrame(rows)
    _save_table(asymmetry, output_dir, "3_5_structural_asymmetry")

    plot_df = asymmetry.melt(
        id_vars=["timeframe", "basis"],
        value_vars=["up_mean_price_pct", "down_mean_abs_loss_pct", "up_mean_duration", "down_mean_duration"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(bottom=0.24)
    sns.barplot(data=plot_df, x="timeframe", y="value", hue="metric", ax=ax)
    ax.set_title("Structural asymmetry summary")
    ax.set_xlabel("Timeframe")
    ax.set_ylabel("Value")
    _annotate_bar_containers(ax, "{:.2f}")
    _move_legend_below(ax, ncol=2, title="")
    fig.savefig(output_dir / "3_5_structural_asymmetry.png")
    plt.close(fig)
    return asymmetry


def section_3_6_noise_characteristics(data: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    rows = []
    plot_rows = []
    for timeframe, df in data.items():
        for direction in ("UP", "DOWN"):
            subset = df[df["cycle_type"].eq(direction)].copy()
            if subset.empty:
                continue
            rows.append(
                {
                    "timeframe": timeframe,
                    "cycle_type": direction,
                    "sample_size": int(len(subset)),
                    "noise_count_mean": float(subset["noise_count"].mean()),
                    "noise_count_median": float(subset["noise_count"].median()),
                    "noise_count_q75": float(subset["noise_count"].quantile(0.75)),
                    "noise_ratio_mean": float(subset["noise_ratio"].mean()),
                    "direction_pct_mean": float(subset["direction_pct"].mean()),
                    "direction_pct_median": float(subset["direction_pct"].median()),
                    "corr_duration_vs_noise_count": _safe_corr(subset["duration_candles"], subset["noise_count"]),
                    "corr_duration_vs_direction_pct": _safe_corr(subset["duration_candles"], subset["direction_pct"]),
                }
            )
            plot_rows.append(
                subset[["timeframe", "cycle_type", "duration_candles", "noise_count", "direction_pct"]].dropna(subset=["duration_candles", "noise_count"])
            )

    noise = pd.DataFrame(rows)
    _save_table(noise, output_dir, "3_6_noise_characteristics")

    plot_df = pd.concat(plot_rows, ignore_index=True) if plot_rows else pd.DataFrame()
    if not plot_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.subplots_adjust(bottom=0.24, wspace=0.28)
        sns.scatterplot(
            data=plot_df,
            x="duration_candles",
            y="noise_count",
            hue="cycle_type",
            style="timeframe",
            alpha=0.5,
            ax=axes[0],
            palette={"UP": "#2ca02c", "DOWN": "#d62728"},
        )
        axes[0].set_title("Noise count vs duration")
        axes[0].set_xlabel("Duration (candles)")
        axes[0].set_ylabel("noise_count")
        _move_legend_below(axes[0], ncol=4, title="")

        sns.boxplot(
            data=plot_df,
            x="timeframe",
            y="direction_pct",
            hue="cycle_type",
            ax=axes[1],
            palette={"UP": "#2ca02c", "DOWN": "#d62728"},
        )
        axes[1].set_title("direction_pct distribution")
        axes[1].set_xlabel("Timeframe")
        axes[1].set_ylabel("direction_pct")
        _move_legend_below(axes[1], ncol=2, title="")

        fig.savefig(output_dir / "3_6_noise_characteristics.png")
        plt.close(fig)
    return noise


def section_3_7_shape_positions(data: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    rows = []
    peak_rows = []
    trough_rows = []

    for timeframe, df in data.items():
        up = df[df["cycle_type"].eq("UP")].copy()
        down = df[df["cycle_type"].eq("DOWN")].copy()
        if not up.empty:
            rows.append(
                {
                    "timeframe": timeframe,
                    "cycle_type": "UP",
                    "metric": "peak_price_position",
                    **_quantile_summary(up["peak_price_position"]),
                }
            )
            peak_rows.append(up[["timeframe", "peak_price_position"]].dropna())
        if not down.empty:
            rows.append(
                {
                    "timeframe": timeframe,
                    "cycle_type": "DOWN",
                    "metric": "trough_price_position",
                    **_quantile_summary(down["trough_price_position"]),
                }
            )
            trough_rows.append(down[["timeframe", "trough_price_position"]].dropna())

    shape_df = pd.DataFrame(rows)
    _save_table(shape_df, output_dir, "3_7_cycle_shape_positions")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    if peak_rows:
        peak_df = pd.concat(peak_rows, ignore_index=True)
        sns.histplot(data=peak_df, x="peak_price_position", hue="timeframe", bins=20, kde=True, ax=axes[0])
        axes[0].set_title("Peak position in UP cycles")
        axes[0].set_xlabel("peak_price_position")
        _move_legend_below(axes[0], ncol=4, title="")
    if trough_rows:
        trough_df = pd.concat(trough_rows, ignore_index=True)
        sns.histplot(data=trough_df, x="trough_price_position", hue="timeframe", bins=20, kde=True, ax=axes[1])
        axes[1].set_title("Trough position in DOWN cycles")
        axes[1].set_xlabel("trough_price_position")
        _move_legend_below(axes[1], ncol=4, title="")

    fig.savefig(output_dir / "3_7_cycle_shape_positions.png")
    plt.close(fig)
    return shape_df


def section_3_8_hierarchy(asset: str, output_dir: Path) -> pd.DataFrame:
    hierarchy = json.loads(_find_hierarchy_map(asset=asset).read_text(encoding="utf-8"))
    rows = []
    for parent_tf, child_tf in ADJACENT_PAIRS:
        nodes = hierarchy.get(parent_tf, {})
        for cycle_id, node in nodes.items():
            child_ids = node.get("child_cycle_ids", {}).get(child_tf, [])
            if not child_ids:
                continue
            child_types = [
                _normalize_direction(hierarchy.get(child_tf, {}).get(child_id, {}).get("cycle_type"))
                for child_id in child_ids
                if child_id in hierarchy.get(child_tf, {})
            ]
            child_up_count = sum(direction == "UP" for direction in child_types)
            child_down_count = sum(direction == "DOWN" for direction in child_types)
            rows.append(
                {
                    "parent_tf": parent_tf,
                    "child_tf": child_tf,
                    "parent_cycle_id": cycle_id,
                    "parent_cycle_type": _normalize_direction(node.get("cycle_type")),
                    "child_count": len(child_ids),
                    "child_up_count": child_up_count,
                    "child_down_count": child_down_count,
                    "child_up_ratio": child_up_count / len(child_ids) if child_ids else np.nan,
                    "child_down_ratio": child_down_count / len(child_ids) if child_ids else np.nan,
                }
            )

    pair_df = pd.DataFrame(rows)
    summary = (
        pair_df.groupby(["parent_tf", "child_tf", "parent_cycle_type"], as_index=False)
        .agg(
            parent_cycles=("parent_cycle_id", "count"),
            avg_child_count=("child_count", "mean"),
            median_child_count=("child_count", "median"),
            q25_child_count=("child_count", lambda s: s.quantile(0.25)),
            q75_child_count=("child_count", lambda s: s.quantile(0.75)),
            avg_child_up_ratio=("child_up_ratio", "mean"),
            avg_child_down_ratio=("child_down_ratio", "mean"),
        )
    )
    _save_table(summary, output_dir, "3_8_timeframe_hierarchy")

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(bottom=0.24)
    sns.barplot(
        data=summary,
        x="parent_tf",
        y="avg_child_count",
        hue="parent_cycle_type",
        ax=ax,
        palette={"UP": "#2ca02c", "DOWN": "#d62728"},
    )
    ax.set_title("Average lower-timeframe cycles inside each parent cycle")
    ax.set_xlabel("Parent timeframe")
    ax.set_ylabel("Average child count")
    _annotate_bar_containers(ax, "{:.2f}")
    _move_legend_below(ax, ncol=2, title="")
    fig.savefig(output_dir / "3_8_timeframe_hierarchy.png")
    plt.close(fig)
    return summary


def build_summary_index(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    summary = {
        name: {
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
        for name, df in tables.items()
    }
    (output_dir / "report_index.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready cycle statistics and charts.")
    parser.add_argument("--asset", default="btc")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()
    data = load_cycle_data(asset=args.asset)

    tables = {
        "3_1_dataset_overview": section_3_1_dataset_overview(data, output_dir),
        "3_2_direction_accuracy": section_3_2_direction_accuracy(data, output_dir),
        "3_3_price_distribution": section_3_3_price_distribution(data, output_dir),
        "3_4_duration_distribution": section_3_4_duration_distribution(data, output_dir),
        "3_5_structural_asymmetry": section_3_5_structural_asymmetry(data, output_dir),
        "3_6_noise_characteristics": section_3_6_noise_characteristics(data, output_dir),
        "3_7_shape_positions": section_3_7_shape_positions(data, output_dir),
        "3_8_hierarchy": section_3_8_hierarchy(args.asset, output_dir),
    }
    build_summary_index(output_dir, tables)
    print(f"Saved paper report outputs to: {output_dir}")


if __name__ == "__main__":
    main()
