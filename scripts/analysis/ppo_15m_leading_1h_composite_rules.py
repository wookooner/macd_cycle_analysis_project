from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.ppo_reversal_candidate_backtest import (  # noqa: E402
    LOW_SAMPLE_N,
    PROJECT_PATHS,
    ROUND_TRIP_FEE_PCT,
    SLIPPAGE_PER_SIDE_PCT,
    _raw_market_path,
    _read_timestamp,
)


SOURCE_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_15m_leading_1h_reversal_analysis"
OUT_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_15m_leading_1h_composite_rules"
POSITION_COST_PCT = ROUND_TRIP_FEE_PCT + SLIPPAGE_PER_SIDE_PCT * 2


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    mask_fn: Callable[[pd.DataFrame], pd.Series]


def output_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def is_bottom20(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(["bottom10", "bottom20"])


def is_top20(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(["top10", "top20"])


def safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_lead_candidates() -> pd.DataFrame:
    path = SOURCE_DIR / "31_15m_leads_1h_reversal_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing source candidate summary: {path}")
    needed = [
        "timestamp",
        "close_at_entry",
        "m15_bar_index",
        "m15_ppo_bin",
        "m15_hist_bin",
        "m15_zone4",
        "m15_ppo",
        "m15_ppo_hist",
        "m15_ppo_hist_diff",
        "h1_timestamp",
        "h1_cycle_type",
        "h1_ppo_bin",
        "h1_hist_bin",
        "h1_ppo",
        "h1_ppo_hist",
        "h1_ppo_hist_diff",
        "h1_is_hist_improving",
        "h1_is_down_accelerating",
        "h4_cycle_type",
        "h4_ppo_bin",
        "h4_hist_bin",
        "h4_ppo",
        "h4_ppo_hist",
        "h4_hist_improving",
        "h4_long_bias",
        "h4_short_bias",
        "h4_extreme_long",
        "h4_extreme_short",
        "d1_cycle_type",
        "d1_ppo_bin",
        "d1_hist_bin",
        "d1_ppo",
        "d1_ppo_hist",
        "d1_long_regime",
        "d1_short_regime",
        "target_1h_up_confirm_time",
        "target_1h_up_confirm_price",
        "target_common_exit_1h_down_time",
        "target_common_exit_1h_down_price",
        "target_1h_turns_up_within_4bars",
        "target_1h_turns_up_within_8bars",
        "target_1h_turns_up_within_12bars",
        "target_1h_turns_up_before_new_low",
        "target_max_favorable_excursion",
        "target_max_adverse_excursion",
        "target_pre_entry_alpha",
        "target_early_common_return",
        "target_confirmed_common_return",
    ]
    available = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[col for col in needed if col in available], low_memory=False)
    for col in ["timestamp", "h1_timestamp", "target_1h_up_confirm_time", "target_common_exit_1h_down_time"]:
        if col in df:
            df[col] = _read_timestamp(df[col])
    for col in [
        "h1_is_hist_improving",
        "h1_is_down_accelerating",
        "h4_hist_improving",
        "h4_long_bias",
        "h4_short_bias",
        "h4_extreme_long",
        "h4_extreme_short",
        "d1_long_regime",
        "d1_short_regime",
        "target_1h_turns_up_within_4bars",
        "target_1h_turns_up_within_8bars",
        "target_1h_turns_up_within_12bars",
        "target_1h_turns_up_before_new_low",
    ]:
        if col in df:
            df[col] = safe_bool(df[col])
    df["m15_bottom"] = is_bottom20(df["m15_ppo_bin"]) & is_bottom20(df["m15_hist_bin"])
    df["m15_deep_bottom"] = df["m15_ppo_bin"].astype(str).eq("bottom10") & is_bottom20(df["m15_hist_bin"])
    df["h1_bottom"] = is_bottom20(df["h1_ppo_bin"]) & is_bottom20(df["h1_hist_bin"])
    df["h1_bottom_improving"] = df["h1_bottom"] & df["h1_is_hist_improving"]
    df["h4_supportive"] = df["h4_long_bias"] | df["h4_extreme_long"]
    df["h4_contra"] = df["h4_short_bias"] | df["h4_extreme_short"]
    df["d1_down"] = df["d1_cycle_type"].astype(str).str.lower().eq("down")
    df["d1_strong_short"] = df["d1_short_regime"] & is_top20(df["d1_ppo_bin"])
    return df


def load_15m_candles() -> pd.DataFrame:
    path = _raw_market_path("15m")
    usecols = [col for col in pd.read_csv(path, nrows=0).columns if col in {"date", "timestamp", "open_time", "open", "high", "low", "close", "ppo_hist"}]
    df = pd.read_csv(path, usecols=usecols)
    ts_col = next(col for col in ("timestamp", "open_time", "date") if col in df.columns)
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _read_timestamp(df["timestamp"])
    for col in ["open", "high", "low", "close", "ppo_hist"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "close", "ppo_hist"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df), dtype=np.int64)
    df["ppo_hist_diff"] = df["ppo_hist"].diff()
    return df


def rules() -> list[Rule]:
    return [
        Rule("R0_all_15m_long_when_1h_down", "All 15m long candidates while closed 1h cycle is DOWN", lambda d: pd.Series(True, index=d.index)),
        Rule("R1_15m_bottom", "15m PPO bottom20 and 15m HIST bottom20", lambda d: d["m15_bottom"]),
        Rule("R2_15m_bottom__1h_bottom_improving", "15m bottom + 1h PPO/HIST bottom20 + 1h hist improving", lambda d: d["m15_bottom"] & d["h1_bottom_improving"]),
        Rule("R3_R2__4h_supportive", "R2 + 4h long bias or 4h extreme long", lambda d: d["m15_bottom"] & d["h1_bottom_improving"] & d["h4_supportive"]),
        Rule("R4_R2__4h_not_contra", "R2 + no 4h short bias/extreme", lambda d: d["m15_bottom"] & d["h1_bottom_improving"] & ~d["h4_contra"]),
        Rule("R5_R2__1d_down", "R2 while 1d cycle is DOWN", lambda d: d["m15_bottom"] & d["h1_bottom_improving"] & d["d1_down"]),
        Rule("R6_R2__not_1d_strong_short", "R2 excluding 1d strong short regime", lambda d: d["m15_bottom"] & d["h1_bottom_improving"] & ~d["d1_strong_short"]),
        Rule("R7_R2__1d_strong_short", "R2 inside 1d strong short regime", lambda d: d["m15_bottom"] & d["h1_bottom_improving"] & d["d1_strong_short"]),
        Rule("R8_15m_deep_bottom__1h_bottom_improving", "15m PPO bottom10 + HIST bottom20/bottom10 + 1h bottom improving", lambda d: d["m15_deep_bottom"] & d["h1_bottom_improving"]),
        Rule("R9_R8__4h_supportive__not_1d_strong_short", "R8 + 4h supportive + not 1d strong short", lambda d: d["m15_deep_bottom"] & d["h1_bottom_improving"] & d["h4_supportive"] & ~d["d1_strong_short"]),
    ]


def summarize_subset(df: pd.DataFrame, rule_name: str, description: str) -> dict[str, object]:
    n = len(df)
    return {
        "rule_name": rule_name,
        "description": description,
        "n": n,
        "within_4bars_pct": pct(df, "target_1h_turns_up_within_4bars"),
        "within_8bars_pct": pct(df, "target_1h_turns_up_within_8bars"),
        "within_12bars_pct": pct(df, "target_1h_turns_up_within_12bars"),
        "before_new_low_pct": pct(df, "target_1h_turns_up_before_new_low"),
        "avg_pre_entry_alpha": mean(df, "target_pre_entry_alpha"),
        "avg_mfe": mean(df, "target_max_favorable_excursion"),
        "avg_mae": mean(df, "target_max_adverse_excursion"),
        "avg_early_common_return": mean(df, "target_early_common_return"),
        "avg_confirmed_common_return": mean(df, "target_confirmed_common_return"),
        "low_sample": n < LOW_SAMPLE_N,
    }


def pct(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean() * 100) if len(df) and col in df else np.nan


def mean(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df[col], errors="coerce").mean()) if len(df) and col in df else np.nan


def max_drawdown(returns: pd.Series) -> float:
    net = pd.to_numeric(returns, errors="coerce").fillna(0) / 100.0
    equity = (1 + net).cumprod()
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1).min() * 100)


def common_exit_return(entry_price: float, exit_price: float) -> float:
    if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
        return np.nan
    return (exit_price / entry_price - 1.0) * 100.0 - POSITION_COST_PCT


def entry_mode_summaries(df: pd.DataFrame, candles_15m: pd.DataFrame, rule_name: str) -> list[dict[str, object]]:
    rows = []
    for mode in ["15m_immediate", "wait_1_15m_bar", "wait_2_15m_bars", "1h_hist_improving_confirm", "1h_up_confirmed"]:
        returns: list[float] = []
        delays: list[float] = []
        success: list[bool] = []
        for item in df.itertuples(index=False):
            exit_price = getattr(item, "target_common_exit_1h_down_price", np.nan)
            exit_time = getattr(item, "target_common_exit_1h_down_time", pd.NaT)
            if pd.isna(exit_price) or pd.isna(exit_time):
                continue
            entry_price = np.nan
            entry_delay = np.nan
            if mode == "15m_immediate":
                entry_price = getattr(item, "close_at_entry", np.nan)
                entry_delay = 0
            elif mode in {"wait_1_15m_bar", "wait_2_15m_bars"}:
                wait = 1 if mode == "wait_1_15m_bar" else 2
                idx = getattr(item, "m15_bar_index", np.nan)
                if pd.isna(idx):
                    continue
                idx = int(idx)
                entry_idx = idx + wait
                if entry_idx >= len(candles_15m):
                    continue
                segment = candles_15m.iloc[idx + 1 : entry_idx + 1]
                if segment.empty or not (segment["ppo_hist_diff"] > 0).all():
                    continue
                entry_price = float(candles_15m.iloc[entry_idx]["close"])
                entry_delay = wait
            elif mode == "1h_hist_improving_confirm":
                if getattr(item, "h1_is_hist_improving", False):
                    entry_price = getattr(item, "close_at_entry", np.nan)
                    entry_delay = 0
                else:
                    confirm_time = getattr(item, "target_1h_up_confirm_time", pd.NaT)
                    if pd.isna(confirm_time):
                        continue
                    # This conservative proxy waits until 1h UP confirmation when
                    # candidate-time 1h hist is not improving.
                    entry_price = getattr(item, "target_1h_up_confirm_price", np.nan)
                    entry_delay = (confirm_time - getattr(item, "timestamp")) / pd.Timedelta(minutes=15)
            elif mode == "1h_up_confirmed":
                confirm_time = getattr(item, "target_1h_up_confirm_time", pd.NaT)
                if pd.isna(confirm_time):
                    continue
                entry_price = getattr(item, "target_1h_up_confirm_price", np.nan)
                entry_delay = (confirm_time - getattr(item, "timestamp")) / pd.Timedelta(minutes=15)
            if pd.isna(entry_price):
                continue
            ret = common_exit_return(float(entry_price), float(exit_price))
            returns.append(ret)
            delays.append(float(entry_delay))
            success.append(bool(getattr(item, "target_1h_turns_up_within_8bars", False)))
        s = pd.Series(returns, dtype="float64")
        rows.append(
            {
                "rule_name": rule_name,
                "entry_mode": mode,
                "n": len(s),
                "turn_up_within_8bars_pct": float(np.mean(success) * 100) if success else np.nan,
                "win_rate": float((s > 0).mean() * 100) if len(s) else np.nan,
                "avg_return": float(s.mean()) if len(s) else np.nan,
                "median_return": float(s.median()) if len(s) else np.nan,
                "mdd": max_drawdown(s),
                "avg_entry_delay_bars": float(np.mean(delays)) if delays else np.nan,
                "low_sample": len(s) < LOW_SAMPLE_N,
            }
        )
    return rows


def grouped_summary(df: pd.DataFrame, group_cols: list[str], label: str) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"analysis": label, **{col: val for col, val in zip(group_cols, keys)}}
        row.update(summarize_subset(group, "group", ""))
        row.pop("rule_name", None)
        row.pop("description", None)
        rows.append(row)
    return pd.DataFrame(rows)


def rank_rules(rule_summary: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    best_timing = (
        timing[~timing["low_sample"]]
        .sort_values(["rule_name", "avg_return"], ascending=[True, False])
        .groupby("rule_name", as_index=False)
        .head(1)
    )
    merged = rule_summary.merge(
        best_timing[["rule_name", "entry_mode", "avg_return", "win_rate", "mdd", "avg_entry_delay_bars"]],
        on="rule_name",
        how="left",
    )
    merged["score"] = (
        merged["within_8bars_pct"].fillna(0) * 0.35
        + merged["avg_return"].fillna(-5) * 8
        + merged["win_rate"].fillna(0) * 0.15
        - merged["low_sample"].astype(int) * 50
    )
    return merged.sort_values("score", ascending=False)


def markdown_table(df: pd.DataFrame, cols: list[str], n: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    return df[cols].head(n).to_markdown(index=False)


def write_report(
    out_dir: Path,
    rule_summary: pd.DataFrame,
    state_summary: pd.DataFrame,
    timing: pd.DataFrame,
    final_rules: pd.DataFrame,
) -> None:
    r0 = rule_summary[rule_summary["rule_name"].eq("R0_all_15m_long_when_1h_down")].iloc[0]
    r2 = rule_summary[rule_summary["rule_name"].eq("R2_15m_bottom__1h_bottom_improving")]
    r7 = rule_summary[rule_summary["rule_name"].eq("R7_R2__1d_strong_short")]
    r2_text = "No R2 rows."
    if not r2.empty:
        row = r2.iloc[0]
        r2_text = f"R2 success within 8 bars: {row['within_8bars_pct']:.2f}% on n={int(row['n'])}, avg alpha {row['avg_pre_entry_alpha']:.4f}%."
    r7_text = "No 1d strong short sample for R2."
    if not r7.empty:
        row = r7.iloc[0]
        r7_text = f"R2 inside 1d strong short: n={int(row['n'])}, within 8 bars {row['within_8bars_pct']:.2f}%, avg alpha {row['avg_pre_entry_alpha']:.4f}%."
    report = f"""# PPO 15m Leading 1h Composite Rules

## Purpose

This expands the prior 15m-leading-1h analysis into practical composite rules using closed-only 15m/1h/4h/1d features. Candidate rows are not duplicated; this report uses the existing source candidate table and writes only summaries.

## Baseline

- Candidates: {int(r0['n']):,}
- 1h UP within 8 bars: {r0['within_8bars_pct']:.2f}%
- Average pre-entry alpha: {r0['avg_pre_entry_alpha']:.4f}%

## Main Composite Result

{r2_text}

## 1d Strong Short Check

{r7_text}

## Rule Ranking

{markdown_table(final_rules, ['rule_name', 'n', 'within_8bars_pct', 'avg_pre_entry_alpha', 'entry_mode', 'avg_return', 'win_rate', 'mdd', 'score'], 12)}

## 4h and 1d State Breakdown

{markdown_table(state_summary.sort_values(['within_8bars_pct', 'n'], ascending=[False, False]), ['analysis', 'h4_cycle_type', 'h4_ppo_bin', 'h4_hist_bin', 'd1_cycle_type', 'd1_ppo_bin', 'd1_hist_bin', 'n', 'within_8bars_pct', 'avg_pre_entry_alpha'], 15)}

## Entry Timing

{markdown_table(timing.sort_values(['rule_name', 'avg_return'], ascending=[True, False]), ['rule_name', 'entry_mode', 'n', 'turn_up_within_8bars_pct', 'win_rate', 'avg_return', 'mdd', 'avg_entry_delay_bars'], 30)}

## Practical Takeaway

The rule should not only maximize early 1h-turn probability. It must keep enough trades, avoid 1d strong short regimes when they damage alpha, and prefer the entry mode with acceptable drawdown. Use `41_final_rule_candidates.csv` for the shortlist.
"""
    (out_dir / "PPO_15m_leading_1h_composite_rules_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Composite rules for 15m leading 1h reversal analysis.")
    _ = parser.parse_args()
    out_dir = output_dir()
    df = load_lead_candidates()
    candles_15m = load_15m_candles()

    rule_rows = []
    timing_rows = []
    for rule in rules():
        mask = rule.mask_fn(df).fillna(False)
        subset = df[mask].copy()
        rule_rows.append(summarize_subset(subset, rule.name, rule.description))
        timing_rows.extend(entry_mode_summaries(subset, candles_15m, rule.name))
    rule_summary = pd.DataFrame(rule_rows).sort_values(["within_8bars_pct", "n"], ascending=[False, False])
    timing = pd.DataFrame(timing_rows).sort_values(["rule_name", "avg_return"], ascending=[True, False])

    state_base = df[df["m15_bottom"] & df["h1_bottom_improving"]].copy()
    state_summary = pd.concat(
        [
            grouped_summary(state_base, ["h4_cycle_type", "h4_ppo_bin", "h4_hist_bin"], "R2_by_4h"),
            grouped_summary(state_base, ["d1_cycle_type", "d1_ppo_bin", "d1_hist_bin"], "R2_by_1d"),
            grouped_summary(state_base, ["h4_cycle_type", "h4_ppo_bin", "h4_hist_bin", "d1_cycle_type", "d1_ppo_bin", "d1_hist_bin"], "R2_by_4h_1d"),
        ],
        ignore_index=True,
    )
    final_rules = rank_rules(rule_summary, timing)

    rule_summary.to_csv(out_dir / "37_composite_rule_success.csv", index=False, encoding="utf-8-sig")
    state_summary.to_csv(out_dir / "38_composite_rule_by_4h_1d_state.csv", index=False, encoding="utf-8-sig")
    timing.to_csv(out_dir / "39_entry_mode_by_composite_rule.csv", index=False, encoding="utf-8-sig")
    final_rules.to_csv(out_dir / "40_final_rule_candidates.csv", index=False, encoding="utf-8-sig")
    write_report(out_dir, rule_summary, state_summary, timing, final_rules)

    print(f"Wrote composite rule analysis to {out_dir}")
    print(f"Rules: {len(rule_summary)}; timing rows: {len(timing)}; state rows: {len(state_summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
