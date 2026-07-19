"""MTF conflict / alternating direction chain analysis (15m / 1h / 4h / 1d / 1w + 5m).

Builds a state table anchored on closed 15m bars. For each 15m bar it joins the
current cycle direction and PPO/HIST features from each TF, the direction chain
strings, conflict pattern label, conflict score, and forward-looking target
labels (1h / 4h / 1d direction transitions, rebound-failure flags, forward
returns and MFE/MAE). Outputs:

  64_direction_chain_outcome_summary.csv
  65_m15_up_h1_down_reversal_or_noise.csv
  66_h1_down_h4_up_pullback_or_reversal.csv
  67_h4_up_inside_1d_down_analysis.csv
  68_alternating_chain_current_pattern_analysis.csv
  73_current_state_scenario_mapping.csv

Closed-only guard: for every TF higher than the 15m anchor we use the most
recent bar whose close time + tf_delta <= anchor (merge_asof backward against
``available_at``). Cycle direction is taken as the cycle whose
[start_date, end_exclusive) interval contains the anchor — same convention as
state_extractor.py and ppo_reversal_candidate_backtest.add_cycle_state.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import PROJECT_PATHS  # noqa: E402


TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d", "1w")
ANCHOR_TF = "15m"
TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
LOW_SAMPLE_N = 30
MEDIUM_SAMPLE_N = 100
RELIABLE_SAMPLE_N = 1000

# How many future 15m bars to scan when scoring forward labels for the 1h / 4h /
# 1d transitions. 32 bars = 8h, enough for the longest "wait window" cases.
FORWARD_15M_BARS = (4, 8, 12, 16, 32)
FORWARD_1H_BARS = (1, 2, 4, 8)
FORWARD_4H_BARS = (1, 2, 3, 6)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _market_path(tf: str) -> Path:
    candidates = [
        PROJECT_PATHS.raw_market_dir / f"BTCUSD_{tf}.csv",
        PROJECT_PATHS.raw_market_dir / f"BTCUSDT_{tf}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing raw candle file for {tf}: tried {candidates}")


def _cycle_path(tf: str) -> Path:
    candidates = [
        PROJECT_PATHS.asset_cycle_dir("btc") / f"cycles_{tf}.parquet",
        PROJECT_PATHS.cycle_structured_dir / "btc" / f"cycles_{tf}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing cycle parquet for {tf}: tried {candidates}")


def output_dir() -> Path:
    target = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_mtf_conflict_chain_analysis"
    target.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# Loading + feature engineering helpers
# ---------------------------------------------------------------------------


def _read_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _zone4(ppo: pd.Series, hist: pd.Series) -> pd.Series:
    ppo_sign = np.where(pd.to_numeric(ppo, errors="coerce") >= 0, "ppo_pos", "ppo_neg")
    hist_sign = np.where(pd.to_numeric(hist, errors="coerce") >= 0, "hist_pos", "hist_neg")
    return pd.Series(ppo_sign + "__" + hist_sign, index=ppo.index)


def _bin_from_decile(decile: pd.Series) -> pd.Series:
    dec = pd.to_numeric(decile, errors="coerce")
    return pd.Series(
        np.select(
            [dec <= 1, dec <= 2, dec >= 10, dec >= 9],
            ["bottom10", "bottom20", "top10", "top20"],
            default="mid",
        ),
        index=decile.index,
    )


def _add_full_deciles(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    ranked = pd.to_numeric(df[col], errors="coerce").rank(method="first")
    try:
        df[f"{prefix}_decile"] = pd.qcut(ranked, 10, labels=False).astype("float") + 1
    except ValueError:
        df[f"{prefix}_decile"] = np.nan
    df[f"{prefix}_bin"] = _bin_from_decile(df[f"{prefix}_decile"])
    return df


def load_candles(tf: str) -> pd.DataFrame:
    path = _market_path(tf)
    cols_avail = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in cols_avail if c in {"date", "timestamp", "open_time", "open", "high", "low", "close", "ppo", "ppo_hist", "ma_25"}]
    df = pd.read_csv(path, usecols=usecols).copy()
    ts_col = next((c for c in ("timestamp", "open_time", "date") if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"{path} has no timestamp column")
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _read_ts(df["timestamp"])
    for col in ("open", "high", "low", "close", "ppo", "ppo_hist"):
        if col not in df.columns:
            raise ValueError(f"{path} missing column {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ma_25" in df.columns:
        df["ma_25"] = pd.to_numeric(df["ma_25"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close", "ppo", "ppo_hist"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["ppo_hist_diff"] = df["ppo_hist"].diff()
    df["zone4"] = _zone4(df["ppo"], df["ppo_hist"])
    df = _add_full_deciles(df, "ppo", "ppo")
    df = _add_full_deciles(df, "ppo_hist", "hist")
    df["hist_improving"] = (df["ppo_hist_diff"] > 0).astype("int8")
    df["hist_worsening"] = (df["ppo_hist_diff"] < 0).astype("int8")
    if "ma_25" in df.columns:
        df["dist_ma25"] = (df["close"] / df["ma_25"] - 1.0) * 100.0
    else:
        df["dist_ma25"] = np.nan
    df["timeframe"] = tf
    df["available_at"] = df["timestamp"] + pd.Timedelta(seconds=TF_SECONDS[tf])
    return df


def load_cycles(tf: str) -> pd.DataFrame:
    df = pd.read_parquet(_cycle_path(tf)).copy()
    df["start_date"] = _read_ts(df["start_date"])
    df["end_date"] = _read_ts(df["end_date"])
    df = df.dropna(subset=["start_date", "end_date"]).sort_values("start_date").reset_index(drop=True)
    df["end_exclusive"] = df["end_date"] + pd.Timedelta(seconds=TF_SECONDS[tf])
    df["cycle_type_norm"] = df["cycle_type"].astype(str).str.lower().map({"up": "U", "down": "D"}).fillna("X")
    df["duration_candles"] = pd.to_numeric(df.get("duration_candles"), errors="coerce")
    return df[["cycle_id", "cycle_type", "cycle_type_norm", "start_date", "end_date", "end_exclusive", "duration_candles"]]


# ---------------------------------------------------------------------------
# State-table construction
# ---------------------------------------------------------------------------


def join_cycle_dir(anchor: pd.DataFrame, cycles: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Join the cycle that contains each anchor timestamp on `cycles_<tf>`."""
    rename = {
        "cycle_id": f"cycle_id_{tf}",
        "cycle_type_norm": f"dir_{tf}",
        "start_date": f"cycle_start_{tf}",
        "end_exclusive": f"cycle_end_excl_{tf}",
        "duration_candles": f"cycle_duration_{tf}",
    }
    src = cycles[list(rename)].rename(columns=rename).sort_values(f"cycle_start_{tf}")
    merged = pd.merge_asof(
        anchor.sort_values("timestamp"),
        src,
        left_on="timestamp",
        right_on=f"cycle_start_{tf}",
        direction="backward",
    )
    in_cycle = merged["timestamp"].lt(merged[f"cycle_end_excl_{tf}"])
    for col in [f"cycle_id_{tf}", f"dir_{tf}", f"cycle_start_{tf}", f"cycle_end_excl_{tf}", f"cycle_duration_{tf}"]:
        merged.loc[~in_cycle, col] = np.nan
    elapsed = (merged["timestamp"] - merged[f"cycle_start_{tf}"]) / pd.Timedelta(seconds=TF_SECONDS[tf])
    duration = pd.to_numeric(merged[f"cycle_duration_{tf}"], errors="coerce").replace(0, np.nan)
    merged[f"cycle_progress_{tf}"] = (elapsed / duration).clip(lower=0, upper=1)
    merged[f"dir_{tf}"] = merged[f"dir_{tf}"].fillna("X")
    return merged


def join_tf_features(anchor: pd.DataFrame, candles: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Join PPO/HIST features from `candles` onto `anchor`. For higher TFs we use
    `available_at` (close + tf_delta) so we never see future data; for the
    anchor TF (15m) we just align timestamps directly."""
    keep_cols = [
        "ppo",
        "ppo_hist",
        "ppo_hist_diff",
        "ppo_bin",
        "hist_bin",
        "zone4",
        "hist_improving",
        "hist_worsening",
        "dist_ma25",
        "close",
    ]
    rename = {c: f"{c}_{tf}" for c in keep_cols}
    if tf == ANCHOR_TF:
        sub = candles[["timestamp"] + keep_cols].rename(columns=rename)
        out = anchor.merge(sub, on="timestamp", how="left")
    else:
        sub = candles[["available_at"] + keep_cols].rename(columns=rename)
        out = pd.merge_asof(
            anchor.sort_values("timestamp"),
            sub.sort_values("available_at"),
            left_on="timestamp",
            right_on="available_at",
            direction="backward",
        ).drop(columns=["available_at"])
    return out


def build_state_table() -> pd.DataFrame:
    anchor_candles = load_candles(ANCHOR_TF)
    state = anchor_candles[["timestamp", "open", "high", "low", "close"]].copy()
    state["bar_index"] = np.arange(len(state), dtype=np.int64)

    # Per-TF features
    for tf in TIMEFRAMES:
        candles = anchor_candles if tf == ANCHOR_TF else load_candles(tf)
        state = join_tf_features(state, candles, tf)

    # Per-TF cycle direction
    for tf in TIMEFRAMES:
        cycles = load_cycles(tf)
        state = join_cycle_dir(state, cycles, tf)

    # Direction chain strings
    state["chain_15m_to_1w"] = (
        state["dir_15m"].fillna("X")
        + state["dir_1h"].fillna("X")
        + state["dir_4h"].fillna("X")
        + state["dir_1d"].fillna("X")
        + state["dir_1w"].fillna("X")
    )
    state["chain_5m_to_1w"] = state["dir_5m"].fillna("X") + state["chain_15m_to_1w"]

    # Conflict descriptors
    chain5 = state["chain_15m_to_1w"]
    state["n_up_5tf"] = chain5.str.count("U")
    state["n_down_5tf"] = chain5.str.count("D")
    state["n_up_major"] = (
        state["dir_1h"].eq("U").astype(int)
        + state["dir_4h"].eq("U").astype(int)
        + state["dir_1d"].eq("U").astype(int)
        + state["dir_1w"].eq("U").astype(int)
    )
    state["n_up_minor"] = (
        state["dir_5m"].eq("U").astype(int) + state["dir_15m"].eq("U").astype(int)
    )

    def _alt_count(s: str) -> int:
        if not isinstance(s, str) or len(s) < 2 or "X" in s:
            return -1
        return sum(1 for i in range(len(s) - 1) if s[i] != s[i + 1])

    state["alternation_count_5tf"] = chain5.map(_alt_count)
    state["conflict_score_5tf"] = state["alternation_count_5tf"] / 4.0  # max 4 transitions across 5 TFs
    state.loc[state["alternation_count_5tf"] < 0, "conflict_score_5tf"] = np.nan

    # Conflict pattern label
    def _classify_pattern(row: pd.Series) -> str:
        c5 = row["chain_15m_to_1w"]
        c6 = row["chain_5m_to_1w"]
        if "X" in c5:
            return "unknown"
        if c5 == "UUUUU":
            return "full_long_alignment"
        if c5 == "DDDDD":
            return "full_short_alignment"
        if c5 == "UDUDU":
            return "alternating_up_start"
        if c5 == "DUDUD":
            return "alternating_down_start"
        if "X" not in c6 and c6 == "DUDUDU":
            return "full_alternating_down_start_with_5m"
        if "X" not in c6 and c6 == "UDUDUD":
            return "full_alternating_up_start_with_5m"
        # 4h vs 1d split families
        if row["dir_4h"] == "U" and row["dir_1d"] == "D":
            return "h4_up_inside_1d_down"
        if row["dir_4h"] == "D" and row["dir_1d"] == "U":
            return "h4_down_inside_1d_up"
        if row["dir_15m"] == "U" and row["dir_1h"] == "D":
            return "m15_rebound_inside_1h_down"
        if row["dir_15m"] == "D" and row["dir_1h"] == "U":
            return "m15_pullback_inside_1h_up"
        if row["dir_1h"] == "D" and row["dir_4h"] == "U":
            return "h1_pullback_inside_4h_up"
        if row["dir_1h"] == "U" and row["dir_4h"] == "D":
            return "h1_rebound_inside_4h_down"
        return "other_mixed"

    state["conflict_pattern"] = state.apply(_classify_pattern, axis=1)

    # Pair-level conflict flags
    state["has_15m_vs_1h_conflict"] = (state["dir_15m"] != state["dir_1h"]).astype(int)
    state["has_1h_vs_4h_conflict"] = (state["dir_1h"] != state["dir_4h"]).astype(int)
    state["has_4h_vs_1d_conflict"] = (state["dir_4h"] != state["dir_1d"]).astype(int)
    state["has_1d_vs_1w_conflict"] = (state["dir_1d"] != state["dir_1w"]).astype(int)

    # "Current-like" pattern flags
    state["pattern_current_like_UDUDU"] = state["chain_15m_to_1w"].eq("UDUDU").astype(int)
    state["pattern_current_like_DUDUDU"] = state["chain_5m_to_1w"].eq("DUDUDU").astype(int)

    return state


# ---------------------------------------------------------------------------
# Forward / target label generation
# ---------------------------------------------------------------------------


def add_forward_labels(state: pd.DataFrame) -> pd.DataFrame:
    """Generate forward-looking target labels and forward returns. Labels here
    are the *only* place we look beyond the anchor timestamp."""
    state = state.copy()
    state["future_close_15m_1bar"] = state["close"].shift(-1)

    # Forward returns at 15m grain (% change of close)
    for n in FORWARD_15M_BARS:
        future = state["close"].shift(-n)
        state[f"forward_return_15m_{n}bar"] = (future / state["close"] - 1.0) * 100.0

    # MFE / MAE windows at 15m grain
    high = state["high"]
    low = state["low"]
    close = state["close"]
    for n in (8, 16, 32):
        roll_high = high.shift(-1).rolling(n, min_periods=1).max().shift(-(n - 1))
        roll_low = low.shift(-1).rolling(n, min_periods=1).min().shift(-(n - 1))
        state[f"mfe_15m_{n}bar"] = (roll_high / close - 1.0) * 100.0
        state[f"mae_15m_{n}bar"] = (roll_low / close - 1.0) * 100.0

    # Forward 1h direction snapshots — at T + N*15m, what is dir_1h?
    for n in FORWARD_15M_BARS:
        state[f"future_dir_1h_at_+{n}_15mbar"] = state["dir_1h"].shift(-n)
        state[f"future_dir_4h_at_+{n}_15mbar"] = state["dir_4h"].shift(-n)
        state[f"future_dir_1d_at_+{n}_15mbar"] = state["dir_1d"].shift(-n)
        state[f"future_dir_15m_at_+{n}_15mbar"] = state["dir_15m"].shift(-n)

    # Target: 1h turns UP within next N 15m bars (only meaningful when current
    # dir_1h == "D"). We mark NaN when current state is not the pre-condition.
    for n in FORWARD_15M_BARS:
        any_up = pd.Series(False, index=state.index)
        for i in range(1, n + 1):
            any_up = any_up | state["dir_1h"].shift(-i).eq("U")
        col = f"target_1h_turns_up_within_{n}_15m_bars"
        state[col] = np.where(state["dir_1h"].eq("D"), any_up.astype(int), np.nan)

    # Target: 1h turns DOWN within next N 15m bars (mirror)
    for n in FORWARD_15M_BARS:
        any_down = pd.Series(False, index=state.index)
        for i in range(1, n + 1):
            any_down = any_down | state["dir_1h"].shift(-i).eq("D")
        col = f"target_1h_turns_down_within_{n}_15m_bars"
        state[col] = np.where(state["dir_1h"].eq("U"), any_down.astype(int), np.nan)

    # Target: 1d turns UP within next K 4h bars (4h = 16 15m bars)
    bars_per_4h = 16
    for k in FORWARD_4H_BARS:
        n = k * bars_per_4h
        any_up = pd.Series(False, index=state.index)
        for i in range(1, n + 1):
            any_up = any_up | state["dir_1d"].shift(-i).eq("U")
        col = f"target_1d_turns_up_within_{k}_4h_bars"
        state[col] = np.where(state["dir_1d"].eq("D"), any_up.astype(int), np.nan)

    # Target: 15m rebound fails (current dir_15m == "U" and dir_1h == "D"):
    # within next 8 15m bars, 1h has NOT turned UP and 15m has flipped back to DOWN.
    n = 8
    rebound_pre = state["dir_15m"].eq("U") & state["dir_1h"].eq("D")
    any_1h_up = pd.Series(False, index=state.index)
    any_15m_down = pd.Series(False, index=state.index)
    for i in range(1, n + 1):
        any_1h_up = any_1h_up | state["dir_1h"].shift(-i).eq("U")
        any_15m_down = any_15m_down | state["dir_15m"].shift(-i).eq("D")
    state["target_15m_rebound_fails_8bars"] = np.where(
        rebound_pre, ((~any_1h_up) & any_15m_down).astype(int), np.nan
    )

    # Target: 4h rebound fails inside 1d down (current dir_4h == "U" and
    # dir_1d == "D"): within next 3 4h bars, 1d has NOT turned UP and 4h has
    # flipped back to DOWN.
    k = 3
    n4 = k * bars_per_4h
    pre_47 = state["dir_4h"].eq("U") & state["dir_1d"].eq("D")
    any_1d_up = pd.Series(False, index=state.index)
    any_4h_down = pd.Series(False, index=state.index)
    for i in range(1, n4 + 1):
        any_1d_up = any_1d_up | state["dir_1d"].shift(-i).eq("U")
        any_4h_down = any_4h_down | state["dir_4h"].shift(-i).eq("D")
    state["target_4h_rebound_fails_inside_1d_down_3bars"] = np.where(
        pre_47, ((~any_1d_up) & any_4h_down).astype(int), np.nan
    )

    # Resolution direction (used by 68): does the chain resolve to long-aligned
    # UUU on 15m/1h/4h or short-aligned DDD within next 32 15m bars?
    n_res = 32
    long_resolved = pd.Series(False, index=state.index)
    short_resolved = pd.Series(False, index=state.index)
    for i in range(1, n_res + 1):
        d15 = state["dir_15m"].shift(-i)
        d1h = state["dir_1h"].shift(-i)
        d4h = state["dir_4h"].shift(-i)
        long_resolved = long_resolved | (d15.eq("U") & d1h.eq("U") & d4h.eq("U"))
        short_resolved = short_resolved | (d15.eq("D") & d1h.eq("D") & d4h.eq("D"))
    state["resolves_to_UUU_within_32_15m_bars"] = long_resolved.astype(int)
    state["resolves_to_DDD_within_32_15m_bars"] = short_resolved.astype(int)
    # First-resolution direction
    def _first_resolution(row_idx: int) -> str:
        l = long_resolved.iloc[row_idx]
        s = short_resolved.iloc[row_idx]
        if l and not s:
            return "long"
        if s and not l:
            return "short"
        if l and s:
            # earliest wins — re-scan
            for i in range(1, n_res + 1):
                pos = row_idx + i
                if pos >= len(state):
                    return "neither"
                d15 = state["dir_15m"].iat[pos]
                d1h = state["dir_1h"].iat[pos]
                d4h = state["dir_4h"].iat[pos]
                if d15 == "U" and d1h == "U" and d4h == "U":
                    return "long"
                if d15 == "D" and d1h == "D" and d4h == "D":
                    return "short"
            return "neither"
        return "neither"

    state["resolution_direction_32bars"] = [
        _first_resolution(i) for i in range(len(state))
    ]

    return state


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------


def sample_class(n: int) -> str:
    if n < LOW_SAMPLE_N:
        return "very_low"
    if n < MEDIUM_SAMPLE_N:
        return "low_sample"
    if n < RELIABLE_SAMPLE_N:
        return "medium_sample"
    return "reliable_sample"


def _agg_metrics(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    out = {
        "n": n,
        "sample_class": sample_class(n),
        "avg_forward_return_15m_4bar": g["forward_return_15m_4bar"].mean(),
        "avg_forward_return_15m_8bar": g["forward_return_15m_8bar"].mean(),
        "avg_forward_return_15m_16bar": g["forward_return_15m_16bar"].mean(),
        "avg_forward_return_15m_32bar": g["forward_return_15m_32bar"].mean(),
        "median_forward_return_15m_8bar": g["forward_return_15m_8bar"].median(),
        "long_win_rate_8bar": (g["forward_return_15m_8bar"] > 0).mean() * 100.0,
        "short_win_rate_8bar": (g["forward_return_15m_8bar"] < 0).mean() * 100.0,
        "avg_mfe_15m_16bar": g["mfe_15m_16bar"].mean(),
        "avg_mae_15m_16bar": g["mae_15m_16bar"].mean(),
        "avg_mfe_15m_32bar": g["mfe_15m_32bar"].mean(),
        "avg_mae_15m_32bar": g["mae_15m_32bar"].mean(),
        "mfe_mae_ratio_16bar": (
            g["mfe_15m_16bar"].mean() / abs(g["mae_15m_16bar"].mean())
            if g["mae_15m_16bar"].mean() not in (0, np.nan)
            and not pd.isna(g["mae_15m_16bar"].mean())
            else np.nan
        ),
        "pct_1h_turns_up_within_8_15m_bars": g["target_1h_turns_up_within_8_15m_bars"].mean() * 100.0,
        "pct_1h_turns_up_within_16_15m_bars": g["target_1h_turns_up_within_16_15m_bars"].mean() * 100.0,
        "pct_1h_turns_down_within_8_15m_bars": g["target_1h_turns_down_within_8_15m_bars"].mean() * 100.0,
        "pct_1d_turns_up_within_3_4h_bars": g["target_1d_turns_up_within_3_4h_bars"].mean() * 100.0,
        "pct_15m_rebound_fails_8bars": g["target_15m_rebound_fails_8bars"].mean() * 100.0,
        "pct_4h_rebound_fails_3bars": g["target_4h_rebound_fails_inside_1d_down_3bars"].mean() * 100.0,
        "pct_resolves_to_UUU_32bars": g["resolves_to_UUU_within_32_15m_bars"].mean() * 100.0,
        "pct_resolves_to_DDD_32bars": g["resolves_to_DDD_within_32_15m_bars"].mean() * 100.0,
    }
    return pd.Series(out)


def _summary_table(state: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    df = state.dropna(subset=["chain_15m_to_1w"]).copy()
    df = df[~df["chain_15m_to_1w"].str.contains("X", na=False)]
    grouped = df.groupby(group_cols, dropna=False, observed=True)
    summary = grouped.apply(_agg_metrics).reset_index()
    summary = summary.sort_values("n", ascending=False)
    return summary


# ---------------------------------------------------------------------------
# Output writers (64..68, 73)
# ---------------------------------------------------------------------------


def write_64_direction_chain_outcome(state: pd.DataFrame, out: Path) -> Path:
    df = _summary_table(state, ["chain_15m_to_1w", "conflict_pattern", "n_up_5tf", "conflict_score_5tf"])
    path = out / "64_direction_chain_outcome_summary.csv"
    df.to_csv(path, index=False)
    return path


def write_65_m15_up_h1_down(state: pd.DataFrame, out: Path) -> Path:
    sub = state[state["dir_15m"].eq("U") & state["dir_1h"].eq("D")].copy()
    df = sub.groupby(
        [
            "ppo_bin_15m",
            "hist_bin_15m",
            "hist_improving_15m",
            "ppo_bin_1h",
            "hist_bin_1h",
            "hist_improving_1h",
            "dir_4h",
            "ppo_bin_4h",
            "hist_bin_4h",
            "dir_1d",
            "ppo_bin_1d",
            "hist_bin_1d",
        ],
        dropna=False,
        observed=True,
    ).apply(_agg_metrics).reset_index()
    df = df.sort_values("n", ascending=False)
    path = out / "65_m15_up_h1_down_reversal_or_noise.csv"
    df.to_csv(path, index=False)
    return path


def write_66_h1_down_h4_up(state: pd.DataFrame, out: Path) -> Path:
    sub = state[state["dir_1h"].eq("D") & state["dir_4h"].eq("U")].copy()
    df = sub.groupby(
        [
            "ppo_bin_1h",
            "hist_bin_1h",
            "hist_improving_1h",
            "ppo_bin_4h",
            "hist_bin_4h",
            "hist_improving_4h",
            "dir_1d",
            "hist_improving_1d",
        ],
        dropna=False,
        observed=True,
    ).apply(_agg_metrics).reset_index()
    df = df.sort_values("n", ascending=False)
    path = out / "66_h1_down_h4_up_pullback_or_reversal.csv"
    df.to_csv(path, index=False)
    return path


def write_67_h4_up_inside_1d_down(state: pd.DataFrame, out: Path) -> Path:
    sub = state[state["dir_4h"].eq("U") & state["dir_1d"].eq("D")].copy()
    df = sub.groupby(
        [
            "ppo_bin_4h",
            "hist_bin_4h",
            "hist_improving_4h",
            "ppo_bin_1d",
            "hist_bin_1d",
            "hist_improving_1d",
            "dir_1w",
            "hist_bin_1w",
        ],
        dropna=False,
        observed=True,
    ).apply(_agg_metrics).reset_index()
    df = df.sort_values("n", ascending=False)
    path = out / "67_h4_up_inside_1d_down_analysis.csv"
    df.to_csv(path, index=False)
    return path


def write_68_alternating_pattern(state: pd.DataFrame, out: Path) -> Path:
    mask = state["chain_15m_to_1w"].eq("UDUDU") | state["chain_5m_to_1w"].eq("DUDUDU")
    sub = state[mask].copy()
    rows: list[dict] = []
    for label, sel in [
        ("UDUDU_15m_to_1w", state["chain_15m_to_1w"].eq("UDUDU")),
        ("DUDUDU_5m_to_1w", state["chain_5m_to_1w"].eq("DUDUDU")),
        ("either_alternating", mask),
    ]:
        g = state[sel]
        if g.empty:
            continue
        n = len(g)
        rows.append(
            {
                "subset": label,
                "n": n,
                "sample_class": sample_class(n),
                "long_resolution_pct": g["resolves_to_UUU_within_32_15m_bars"].mean() * 100.0,
                "short_resolution_pct": g["resolves_to_DDD_within_32_15m_bars"].mean() * 100.0,
                "neither_resolution_pct": (
                    g["resolution_direction_32bars"].eq("neither").mean() * 100.0
                ),
                "pct_1h_turns_up_within_8_15m_bars": g["target_1h_turns_up_within_8_15m_bars"].mean() * 100.0,
                "pct_1h_turns_up_within_16_15m_bars": g["target_1h_turns_up_within_16_15m_bars"].mean() * 100.0,
                "pct_4h_rebound_fails_3bars": g["target_4h_rebound_fails_inside_1d_down_3bars"].mean() * 100.0,
                "pct_1d_turns_up_within_3_4h_bars": g["target_1d_turns_up_within_3_4h_bars"].mean() * 100.0,
                "avg_forward_return_15m_8bar": g["forward_return_15m_8bar"].mean(),
                "avg_forward_return_15m_16bar": g["forward_return_15m_16bar"].mean(),
                "avg_forward_return_15m_32bar": g["forward_return_15m_32bar"].mean(),
                "long_win_rate_8bar": (g["forward_return_15m_8bar"] > 0).mean() * 100.0,
                "short_win_rate_8bar": (g["forward_return_15m_8bar"] < 0).mean() * 100.0,
                "avg_mfe_15m_32bar": g["mfe_15m_32bar"].mean(),
                "avg_mae_15m_32bar": g["mae_15m_32bar"].mean(),
            }
        )

    # Per-row breakdowns within UDUDU split by 1h/4h hist regime
    if not sub.empty:
        for ppo_bin_1h in sorted(sub["ppo_bin_1h"].dropna().unique()):
            g = sub[sub["ppo_bin_1h"].eq(ppo_bin_1h)]
            n = len(g)
            rows.append(
                {
                    "subset": f"alt_split_by_1h_ppo_bin_{ppo_bin_1h}",
                    "n": n,
                    "sample_class": sample_class(n),
                    "long_resolution_pct": g["resolves_to_UUU_within_32_15m_bars"].mean() * 100.0,
                    "short_resolution_pct": g["resolves_to_DDD_within_32_15m_bars"].mean() * 100.0,
                    "neither_resolution_pct": (
                        g["resolution_direction_32bars"].eq("neither").mean() * 100.0
                    ),
                    "pct_1h_turns_up_within_8_15m_bars": g["target_1h_turns_up_within_8_15m_bars"].mean() * 100.0,
                    "pct_1h_turns_up_within_16_15m_bars": g["target_1h_turns_up_within_16_15m_bars"].mean() * 100.0,
                    "pct_4h_rebound_fails_3bars": g["target_4h_rebound_fails_inside_1d_down_3bars"].mean() * 100.0,
                    "pct_1d_turns_up_within_3_4h_bars": g["target_1d_turns_up_within_3_4h_bars"].mean() * 100.0,
                    "avg_forward_return_15m_8bar": g["forward_return_15m_8bar"].mean(),
                    "avg_forward_return_15m_16bar": g["forward_return_15m_16bar"].mean(),
                    "avg_forward_return_15m_32bar": g["forward_return_15m_32bar"].mean(),
                    "long_win_rate_8bar": (g["forward_return_15m_8bar"] > 0).mean() * 100.0,
                    "short_win_rate_8bar": (g["forward_return_15m_8bar"] < 0).mean() * 100.0,
                    "avg_mfe_15m_32bar": g["mfe_15m_32bar"].mean(),
                    "avg_mae_15m_32bar": g["mae_15m_32bar"].mean(),
                }
            )

    df = pd.DataFrame(rows)
    path = out / "68_alternating_chain_current_pattern_analysis.csv"
    df.to_csv(path, index=False)
    return path


def write_73_current_state_mapping(state: pd.DataFrame, out: Path, override_chain: str | None = None) -> Path:
    """Find historical analogues to the current observed chain. We pick the
    most recent row whose chain has no unresolved 'X' codes (cycle parquets
    typically lag the latest market candles by a few bars). The CLI flag
    --current-chain can override the detection with a literal 6-char chain."""
    rows: list[dict] = []
    if override_chain and len(override_chain) == 6 and set(override_chain) <= {"U", "D"}:
        current_chain_5m = override_chain
        anchor_ts: pd.Timestamp | None = None
    else:
        clean = state[~state["chain_5m_to_1w"].str.contains("X", na=True)]
        if clean.empty:
            current_chain_5m = "DUDUDU"
            anchor_ts = None
        else:
            last_row = clean.tail(1).squeeze()
            current_chain_5m = str(last_row["chain_5m_to_1w"])
            anchor_ts = last_row["timestamp"]
    current_chain_15m = current_chain_5m[1:] if len(current_chain_5m) == 6 else "UDUDU"

    for label, mask in [
        (f"exact_chain_5m_to_1w={current_chain_5m}", state["chain_5m_to_1w"].eq(current_chain_5m)),
        (f"exact_chain_15m_to_1w={current_chain_15m}", state["chain_15m_to_1w"].eq(current_chain_15m)),
        (
            "loose_4h_up_inside_1d_down_with_1w_up",
            state["dir_4h"].eq("U") & state["dir_1d"].eq("D") & state["dir_1w"].eq("U"),
        ),
        (
            "loose_15m_up_in_1h_down_with_4h_up",
            state["dir_15m"].eq("U") & state["dir_1h"].eq("D") & state["dir_4h"].eq("U"),
        ),
    ]:
        g = state[mask]
        n = len(g)
        if n == 0:
            continue
        rows.append(
            {
                "scenario": label,
                "similar_case_count": n,
                "sample_class": sample_class(n),
                "long_resolution_pct": g["resolves_to_UUU_within_32_15m_bars"].mean() * 100.0,
                "short_resolution_pct": g["resolves_to_DDD_within_32_15m_bars"].mean() * 100.0,
                "neither_resolution_pct": (
                    g["resolution_direction_32bars"].eq("neither").mean() * 100.0
                ),
                "pct_1h_turns_up_within_8_15m_bars": g["target_1h_turns_up_within_8_15m_bars"].mean() * 100.0,
                "pct_1h_turns_up_within_16_15m_bars": g["target_1h_turns_up_within_16_15m_bars"].mean() * 100.0,
                "pct_15m_rebound_fails_8bars": g["target_15m_rebound_fails_8bars"].mean() * 100.0,
                "pct_4h_rebound_fails_3bars": g["target_4h_rebound_fails_inside_1d_down_3bars"].mean() * 100.0,
                "pct_1d_turns_up_within_3_4h_bars": g["target_1d_turns_up_within_3_4h_bars"].mean() * 100.0,
                "avg_forward_return_15m_8bar": g["forward_return_15m_8bar"].mean(),
                "avg_forward_return_15m_16bar": g["forward_return_15m_16bar"].mean(),
                "avg_forward_return_15m_32bar": g["forward_return_15m_32bar"].mean(),
                "long_win_rate_8bar": (g["forward_return_15m_8bar"] > 0).mean() * 100.0,
                "short_win_rate_8bar": (g["forward_return_15m_8bar"] < 0).mean() * 100.0,
                "avg_mfe_15m_32bar": g["mfe_15m_32bar"].mean(),
                "avg_mae_15m_32bar": g["mae_15m_32bar"].mean(),
                "current_chain_5m_to_1w": current_chain_5m,
                "current_chain_15m_to_1w": current_chain_15m,
                "anchor_timestamp": str(anchor_ts) if anchor_ts is not None else "(override)",
            }
        )

    df = pd.DataFrame(rows)
    path = out / "73_current_state_scenario_mapping.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Persisted state-table dump (debug + downstream backtest input)
# ---------------------------------------------------------------------------


def write_state_table(state: pd.DataFrame, out: Path) -> Path:
    keep = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "dir_5m",
        "dir_15m",
        "dir_1h",
        "dir_4h",
        "dir_1d",
        "dir_1w",
        "chain_15m_to_1w",
        "chain_5m_to_1w",
        "conflict_pattern",
        "conflict_score_5tf",
        "n_up_5tf",
        "n_up_major",
        "n_up_minor",
        "alternation_count_5tf",
        "has_15m_vs_1h_conflict",
        "has_1h_vs_4h_conflict",
        "has_4h_vs_1d_conflict",
        "has_1d_vs_1w_conflict",
        "pattern_current_like_UDUDU",
        "pattern_current_like_DUDUDU",
        "ppo_15m",
        "ppo_hist_15m",
        "ppo_bin_15m",
        "hist_bin_15m",
        "hist_improving_15m",
        "ppo_1h",
        "ppo_hist_1h",
        "ppo_bin_1h",
        "hist_bin_1h",
        "hist_improving_1h",
        "ppo_4h",
        "ppo_hist_4h",
        "ppo_bin_4h",
        "hist_bin_4h",
        "hist_improving_4h",
        "ppo_1d",
        "ppo_hist_1d",
        "ppo_bin_1d",
        "hist_bin_1d",
        "hist_improving_1d",
        "ppo_1w",
        "ppo_hist_1w",
        "ppo_bin_1w",
        "hist_bin_1w",
        "forward_return_15m_4bar",
        "forward_return_15m_8bar",
        "forward_return_15m_16bar",
        "forward_return_15m_32bar",
        "mfe_15m_16bar",
        "mae_15m_16bar",
        "mfe_15m_32bar",
        "mae_15m_32bar",
        "target_1h_turns_up_within_8_15m_bars",
        "target_1h_turns_up_within_16_15m_bars",
        "target_1h_turns_down_within_8_15m_bars",
        "target_1d_turns_up_within_3_4h_bars",
        "target_15m_rebound_fails_8bars",
        "target_4h_rebound_fails_inside_1d_down_3bars",
        "resolves_to_UUU_within_32_15m_bars",
        "resolves_to_DDD_within_32_15m_bars",
        "resolution_direction_32bars",
    ]
    keep = [c for c in keep if c in state.columns]
    path = out / "mtf_conflict_state_table.parquet"
    state[keep].to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="optional row limit for fast iteration")
    parser.add_argument("--skip-state-dump", action="store_true")
    parser.add_argument("--current-chain", type=str, default=None,
                        help="override the auto-detected current chain with a literal 6-char chain like DUDUDU")
    args = parser.parse_args()

    out = output_dir()
    print(f"[mtf-conflict] output_dir = {out}", flush=True)

    print("[mtf-conflict] building state table...", flush=True)
    state = build_state_table()
    if args.limit:
        state = state.head(args.limit).copy()
    print(f"[mtf-conflict] state table rows = {len(state)}", flush=True)

    print("[mtf-conflict] computing forward labels...", flush=True)
    state = add_forward_labels(state)

    if not args.skip_state_dump:
        path = write_state_table(state, out)
        print(f"[mtf-conflict] wrote {path}", flush=True)

    for writer in (
        write_64_direction_chain_outcome,
        write_65_m15_up_h1_down,
        write_66_h1_down_h4_up,
        write_67_h4_up_inside_1d_down,
        write_68_alternating_pattern,
    ):
        path = writer(state, out)
        print(f"[mtf-conflict] wrote {path}", flush=True)

    path = write_73_current_state_mapping(state, out, override_chain=args.current_chain)
    print(f"[mtf-conflict] wrote {path}", flush=True)

    print("[mtf-conflict] done.", flush=True)


if __name__ == "__main__":
    main()
