from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.config import costs_from_grid, load_grid_config, load_rules_config, load_stops_config
from src.backtest.manifest import build_manifest_entry, write_manifest
from src.backtest.metrics import BasisConfig, compute_metrics
from src.backtest.simulator import _direction_return, simulate_trades
from src.backtest.types import BacktestResults, EntrySignal, MarketContext, SizingPolicy
from src.backtest.validation import validate_dataframe, validate_decision_table, validate_manifest
from src.common.paths import PROJECT_PATHS


TIMEFRAMES = ("15m", "1h", "4h", "1d")
BASE_CANDIDATE_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_reversal_candidate_backtest"
BASE_LEAD_DIR = PROJECT_PATHS.outputs_root / "analysis_results" / "ppo_15m_leading_1h_reversal_analysis"


def _read_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", errors="coerce")


def _raw_market_path(timeframe: str) -> Path:
    return PROJECT_PATHS.base_data_dir / f"BTCUSD_{timeframe}.csv"


def _is_bottom20(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(["bottom10", "bottom20"])


def _is_top20(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(["top10", "top20"])


def load_raw_candles(timeframe: str) -> pd.DataFrame:
    path = _raw_market_path(timeframe)
    wanted = {"date", "timestamp", "open_time", "open", "high", "low", "close", "ppo", "ppo_hist", "rsi", "ma_25", "ma_99", "cvd", "volume", "volume_delta"}
    available = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[col for col in available if col in wanted])
    ts_col = next(col for col in ("timestamp", "open_time", "date") if col in df.columns)
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _read_timestamp(df["timestamp"])
    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df), dtype=np.int64)
    df["ppo_hist_diff"] = df["ppo_hist"].diff() if "ppo_hist" in df else np.nan
    df["distance_from_ma25"] = (df["close"] / df["ma_25"] - 1.0) * 100.0 if "ma_25" in df else np.nan
    df["distance_from_ma99"] = (df["close"] / df["ma_99"] - 1.0) * 100.0 if "ma_99" in df else np.nan
    df["cvd_delta"] = df["cvd"].diff() if "cvd" in df else np.nan
    return df


def load_candidates() -> pd.DataFrame:
    path = BASE_CANDIDATE_DIR / "20_reversal_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing prior candidate file: {path}")
    cols = pd.read_csv(path, nrows=0).columns
    wanted = [
        "timestamp", "open", "high", "low", "close", "bar_index", "candidate_tf",
        "candidate_direction", "close_at_entry", "ppo_bin", "hist_bin",
        "upper_1h_hist", "upper_1h_hist_diff", "upper_1h_ppo_bin", "upper_1h_hist_bin",
        "upper_1h_cycle_direction", "upper_4h_hist", "upper_4h_hist_diff",
        "upper_4h_hist_bin", "upper_4h_cycle_direction",
        "upper_1d_hist", "upper_1d_hist_diff", "upper_1d_ppo_bin", "upper_1d_cycle_direction",
        "same_tf_opposite_true_time", "return_until_same_tf_opposite",
    ]
    df = pd.read_csv(path, usecols=[col for col in wanted if col in cols], engine="python")
    df["timestamp"] = _read_timestamp(df["timestamp"])
    for col in [c for c in df.columns if c not in {"timestamp", "candidate_tf", "candidate_direction", "ppo_bin", "hist_bin"} and not c.endswith("_direction") and not c.endswith("_bin")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [c for c in df.columns if c.endswith("_direction")]:
        df[col] = df[col].astype(str).str.lower()
    df["source_id"] = np.arange(len(df))
    return df


def add_own_features(candidates: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for tf, group in candidates.groupby("candidate_tf", sort=False):
        own = candles[tf][["timestamp", "bar_index", "distance_from_ma25", "distance_from_ma99", "cvd_delta"]].rename(
            columns={"bar_index": "own_bar_index", "distance_from_ma25": "own_distance_from_ma25", "distance_from_ma99": "own_distance_from_ma99", "cvd_delta": "own_cvd_delta"}
        )
        merged = group.merge(own, on="timestamp", how="left")
        merged["bar_index"] = pd.to_numeric(merged["bar_index"], errors="coerce").fillna(merged["own_bar_index"]).astype("int64")
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def build_event_tables(candidates: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    if "is_true_reversal" not in candidates:
        # The base CSV normally has this column even if not in minimal reads.
        full = pd.read_csv(BASE_CANDIDATE_DIR / "20_reversal_candidates.csv", usecols=["timestamp", "candidate_tf", "candidate_direction", "close_at_entry", "is_true_reversal"], engine="python")
        full["timestamp"] = _read_timestamp(full["timestamp"])
    else:
        full = candidates
    for tf in TIMEFRAMES:
        for direction in ("long", "short"):
            sub = full[(full["candidate_tf"].eq(tf)) & (full["candidate_direction"].eq(direction)) & (full["is_true_reversal"].eq(1))].copy()
            tables[(tf, direction)] = sub.sort_values("timestamp").reset_index(drop=True)
    return tables


def next_opposite_event_resolver(events: dict[tuple[str, str], pd.DataFrame]):
    def resolve(tf: str, direction: str, when: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
        opposite = "short" if direction == "long" else "long"
        frame = events.get((tf, opposite), pd.DataFrame())
        if frame.empty:
            return None, None
        times = frame["timestamp"].to_numpy(dtype="datetime64[ns]")
        idx = int(np.searchsorted(times, np.datetime64(when), side="right"))
        if idx >= len(frame):
            return None, None
        row = frame.iloc[idx]
        return row["timestamp"], float(row["close_at_entry"])

    return resolve


def build_rule_entries(candidates: pd.DataFrame, rules: list[dict[str, Any]]) -> pd.DataFrame:
    c = candidates
    own_bottom = _is_bottom20(c["ppo_bin"]) & _is_bottom20(c["hist_bin"])
    own_top = _is_top20(c["ppo_bin"]) & _is_top20(c["hist_bin"])
    h1_bottom = _is_bottom20(c["upper_1h_ppo_bin"]) & _is_bottom20(c["upper_1h_hist_bin"])
    h1_long = c["upper_1h_hist"] < 0
    h1_short = c["upper_1h_hist"] > 0
    h4_long = c["upper_4h_hist"] < 0
    h4_short = c["upper_4h_hist"] > 0
    d1_short = c["upper_1d_hist"] > 0
    d1_strong_short = d1_short & _is_top20(c["upper_1d_ppo_bin"])
    d1_strong_long = (c["upper_1d_hist"] < 0) & _is_bottom20(c["upper_1d_ppo_bin"])
    cvd_neg_threshold = c["own_cvd_delta"].quantile(0.35)
    h4_up_cycle = c["upper_4h_cycle_direction"].eq("up")
    h4_down_cycle = c["upper_4h_cycle_direction"].eq("down")
    h1_down_cycle = c["upper_1h_cycle_direction"].eq("down")
    d1_down_cycle = c["upper_1d_cycle_direction"].eq("down")
    h4_hist_top20 = _is_top20(c["upper_4h_hist_bin"]) if "upper_4h_hist_bin" in c.columns else pd.Series(False, index=c.index)
    h4_hist_weakening = pd.to_numeric(c.get("upper_4h_hist_diff", 0), errors="coerce") < 0
    masks = {
        "L1_long_trend_following": c["candidate_tf"].eq("15m") & c["candidate_direction"].eq("long") & h1_long & h4_long & ~d1_strong_short,
        "S1_short_trend_following": c["candidate_tf"].eq("15m") & c["candidate_direction"].eq("short") & h1_short & h4_short & d1_short,
        "S3_short_1d_strong_short": c["candidate_tf"].isin(["15m", "1h"]) & c["candidate_direction"].eq("short") & d1_strong_short & h4_short & ~d1_strong_long,
        "S4_short_rebound_in_downtrend": c["candidate_tf"].isin(["15m", "1h"]) & c["candidate_direction"].eq("short") & c["upper_4h_cycle_direction"].eq("down") & ~d1_strong_long & (c["own_distance_from_ma25"] > 0) & (c["own_cvd_delta"] <= cvd_neg_threshold) & own_top,
        "CL1_1h_long_while_4h_down_own_extreme": c["candidate_tf"].eq("1h") & c["candidate_direction"].eq("long") & c["upper_4h_cycle_direction"].eq("down") & own_bottom & ~d1_strong_short,
        # Conflict-state strategies (UDUDU family).
        "C0_conflict_immediate_long": c["candidate_tf"].eq("15m") & c["candidate_direction"].eq("long") & h1_down_cycle & h4_up_cycle,
        "C3_conflict_rejection_short": c["candidate_tf"].eq("15m") & c["candidate_direction"].eq("short") & h1_down_cycle & h4_up_cycle & h4_hist_weakening,
        "C4_4h_rebound_fails_in_1d_down_short": c["candidate_tf"].eq("1h") & c["candidate_direction"].eq("short") & h4_up_cycle & d1_down_cycle & h4_hist_weakening,
    }
    rule_map = {r["rule_name"]: r for r in rules}
    frames = []
    for rule_name, mask in masks.items():
        if rule_name not in rule_map:
            continue
        sub = c[mask].copy()
        sub["rule_name"] = rule_name
        sub["rule_kind"] = rule_map[rule_name]["rule_kind"]
        sub["entry_time"] = sub["timestamp"]
        sub["entry_price"] = sub["close_at_entry"]
        sub["signal_time"] = sub["timestamp"]
        sub["entry_delay_bars"] = 0
        sub["missed_move_before_entry"] = 0.0
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_l4_event_entries(candles: dict[str, pd.DataFrame], rules: list[dict[str, Any]]) -> pd.DataFrame:
    rule_names = {rule["rule_name"] for rule in rules}
    needs_l4 = "L4_event_based" in rule_names
    needs_conflict = bool(rule_names & {"C1_conflict_wait_1h_confirmed_long", "C2_conflict_4h_strong_long", "C5_1d_reversal_leading_long"})
    if not (needs_l4 or needs_conflict):
        return pd.DataFrame()
    path = BASE_LEAD_DIR / "31_15m_leads_1h_reversal_candidates.csv"
    if not path.exists():
        return pd.DataFrame()
    wanted = [
        "timestamp", "close_at_entry", "target_1h_up_confirm_time",
        "target_1h_up_confirm_price", "target_1h_turn_delay_15m_bars",
        "target_close_return_to_confirm", "upper_4h_hist", "upper_1d_hist",
        "upper_1d_ppo_bin",
        "upper_4h_cycle_direction", "upper_1d_cycle_direction",
        "upper_4h_hist_bin", "d1_hist_improving",
    ]
    cols = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[col for col in wanted if col in cols], engine="python")
    df["timestamp"] = _read_timestamp(df["timestamp"])
    df["target_1h_up_confirm_time"] = _read_timestamp(df["target_1h_up_confirm_time"])
    df = df.dropna(subset=["target_1h_up_confirm_time", "target_1h_up_confirm_price"])
    if "upper_4h_cycle_direction" in df.columns:
        df["upper_4h_cycle_direction"] = df["upper_4h_cycle_direction"].astype(str).str.lower()
    if "upper_1d_cycle_direction" in df.columns:
        df["upper_1d_cycle_direction"] = df["upper_1d_cycle_direction"].astype(str).str.lower()
    d1_strong_short = (pd.to_numeric(df.get("upper_1d_hist", 0), errors="coerce") > 0) & _is_top20(df.get("upper_1d_ppo_bin", pd.Series("", index=df.index)))
    df_l4 = df.copy()
    if "upper_4h_hist" in df_l4:
        df_l4 = df_l4[(pd.to_numeric(df_l4["upper_4h_hist"], errors="coerce") <= 0) | (~d1_strong_short)].copy()
    m15 = candles["15m"]
    times = m15["timestamp"].to_numpy(dtype="datetime64[ns]")
    frames: list[pd.DataFrame] = []

    def _emit(rule_name: str, rule_kind: str, source: pd.DataFrame, wait_bars: int) -> None:
        sub = source[source["target_1h_turn_delay_15m_bars"].le(wait_bars)].copy()
        if sub.empty:
            return
        pos = np.searchsorted(times, sub["target_1h_up_confirm_time"].to_numpy(dtype="datetime64[ns]"), side="left")
        valid = pos < len(m15)
        sub = sub.loc[valid].copy()
        pos = pos[valid]
        if sub.empty:
            return
        candles_at_entry = m15.iloc[pos]
        sub["rule_name"] = rule_name if wait_bars is None else f"{rule_name}_{wait_bars}bars"
        sub["rule_kind"] = rule_kind
        sub["candidate_direction"] = "long"
        sub["candidate_tf"] = "15m"
        sub["entry_time"] = sub["target_1h_up_confirm_time"]
        sub["entry_price"] = pd.to_numeric(sub["target_1h_up_confirm_price"], errors="coerce")
        sub["bar_index"] = pos.astype(int)
        sub["signal_time"] = sub["timestamp"]
        sub["entry_delay_bars"] = wait_bars
        sub["missed_move_before_entry"] = pd.to_numeric(sub["target_close_return_to_confirm"], errors="coerce")
        sub["open"] = candles_at_entry["open"].to_numpy()
        sub["high"] = candles_at_entry["high"].to_numpy()
        sub["low"] = candles_at_entry["low"].to_numpy()
        sub["close"] = candles_at_entry["close"].to_numpy()
        frames.append(sub)

    if needs_l4:
        for wait_bars in (4, 8, 12, 16):
            _emit("L4_event_based", "production", df_l4, wait_bars)

    if needs_conflict:
        # Conflict-state confirmation strategies are anchored on the 4h cycle
        # direction at signal time. C1 = 4h UP at signal. C2 = C1 plus 4h hist
        # top20 and not d1_strong_short. C5 = 4h UP, 1d DOWN, 1d hist improving.
        h4_up = df.get("upper_4h_cycle_direction", pd.Series(index=df.index)).eq("up")
        d1_down = df.get("upper_1d_cycle_direction", pd.Series(index=df.index)).eq("down")
        h4_hist_top20 = _is_top20(df.get("upper_4h_hist_bin", pd.Series("", index=df.index)))
        d1_hist_improving = pd.to_numeric(df.get("d1_hist_improving", 0), errors="coerce").fillna(0).astype(int).gt(0)
        df_c1 = df[h4_up].copy()
        df_c2 = df[h4_up & h4_hist_top20 & ~d1_strong_short].copy()
        df_c5 = df[h4_up & d1_down & d1_hist_improving].copy()
        for wait_bars in (4, 8, 12, 16):
            if "C1_conflict_wait_1h_confirmed_long" in rule_names:
                _emit("C1_conflict_wait_1h_confirmed_long", "production", df_c1, wait_bars)
            if "C2_conflict_4h_strong_long" in rule_names:
                _emit("C2_conflict_4h_strong_long", "research_only", df_c2, wait_bars)
            if "C5_1d_reversal_leading_long" in rule_names:
                _emit("C5_1d_reversal_leading_long", "research_only", df_c5, wait_bars)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def entries_to_signals(entries: pd.DataFrame) -> list[EntrySignal]:
    signals: list[EntrySignal] = []
    for row in entries.itertuples(index=False):
        signals.append(
            EntrySignal(
                signal_time=pd.Timestamp(row.signal_time),
                entry_time=pd.Timestamp(row.entry_time),
                entry_price=float(row.entry_price),
                direction=str(row.candidate_direction),
                entry_tf=str(row.candidate_tf),
                bar_index=int(row.bar_index),
                rule_name=str(row.rule_name),
                rule_kind=str(row.rule_kind),
                entry_delay_bars=int(getattr(row, "entry_delay_bars", 0) or 0),
                missed_move_before_entry=float(getattr(row, "missed_move_before_entry", 0.0) or 0.0),
                metadata={
                    "source_id": getattr(row, "source_id", None),
                    "high": getattr(row, "high", None),
                    "low": getattr(row, "low", None),
                },
            )
        )
    return signals


def structural_levels_from_policy(candles: dict[str, pd.DataFrame]):
    def levels(signal: EntrySignal, policy) -> tuple[float | None, float | None]:
        direction = str(signal.direction)
        name = policy.name
        if "candidate" in name:
            low = signal.metadata.get("low")
            high = signal.metadata.get("high")
            return (float(low), None) if direction == "long" and pd.notna(low) else (None, float(high)) if direction == "short" and pd.notna(high) else (None, None)
        lookback = 8
        if "recent_swing_4" in name:
            lookback = 4
        elif "recent_swing_16" in name:
            lookback = 16
        tf = "1h" if "h1_swing" in name else signal.entry_tf
        market = candles[tf]
        if tf == signal.entry_tf:
            pos = signal.bar_index
        else:
            pos = int(np.searchsorted(market["timestamp"].to_numpy(dtype="datetime64[ns]"), np.datetime64(signal.entry_time), side="right") - 1)
        start = max(0, pos - lookback)
        recent = market.iloc[start : pos + 1]
        if recent.empty:
            return None, None
        if direction == "long":
            return float(recent["low"].min()), None
        return None, float(recent["high"].max())

    return levels


def run_backtest_grid(
    rules_yaml: str | Path,
    stops_yaml: str | Path,
    periods: list[str] | None = None,
    asset: str = "btc",
    grid_yaml: str | Path | None = None,
) -> BacktestResults:
    grid = load_grid_config(grid_yaml or Path("configs/backtest_grid.yaml"))
    rules = load_rules_config(rules_yaml)
    stops_all = load_stops_config(stops_yaml)
    stop_names = set(grid.get("stop_names", []))
    stops = [stop for stop in stops_all if stop.name in stop_names]
    costs = costs_from_grid(grid)

    candles = {tf: load_raw_candles(tf) for tf in TIMEFRAMES}
    candidates = add_own_features(load_candidates(), candles)
    events = build_event_tables(load_candidates())
    entries = pd.concat([build_rule_entries(candidates, rules), build_l4_event_entries(candles, rules)], ignore_index=True)
    rule_names = set(grid.get("rule_names", []))
    # Event-based variants use a "<rule_name>_<wait_bars>bars" suffix; allow any
    # selected rule_name to match either its bare form or that suffixed form.
    suffix_prefixes = tuple(f"{name}_" for name in rule_names if name in {
        "L4_event_based",
        "C1_conflict_wait_1h_confirmed_long",
        "C2_conflict_4h_strong_long",
        "C5_1d_reversal_leading_long",
    })
    rname = entries["rule_name"].astype(str)
    mask = rname.isin(rule_names)
    for prefix in suffix_prefixes:
        mask = mask | rname.str.startswith(prefix)
    entries = entries[mask]
    context = MarketContext(
        candles=candles,
        opposite_event_resolver=next_opposite_event_resolver(events),
        structural_levels=structural_levels_from_policy(candles),
    )

    ledgers = []
    for stop in stops:
        ledgers.append(simulate_trades(entries_to_signals(entries), None, stop, SizingPolicy(), costs, context))
    ledger = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    comparison = compute_metrics(ledger, BasisConfig(), ["rule_name", "rule_kind", "direction", "exit_policy"])
    reliable = comparison[(comparison["sample_class"].eq("reliable_sample")) & (~comparison["rule_kind"].eq("proxy"))].copy()
    medium = comparison[comparison["sample_class"].eq("medium_sample")].copy()
    low = comparison[comparison["sample_class"].eq("low_sample")].copy()

    out_dir = PROJECT_PATHS.outputs_root / "analysis_results" / str(grid.get("output_dir", "shared_backtest_grid"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ledger": out_dir / "trade_ledger.csv",
        "comparison": out_dir / "comparison.csv",
        "reliable": out_dir / "comparison_reliable.csv",
        "medium": out_dir / "comparison_medium.csv",
        "low": out_dir / "comparison_low.csv",
    }
    ledger.to_csv(paths["ledger"], index=False, encoding="utf-8-sig")
    comparison.to_csv(paths["comparison"], index=False, encoding="utf-8-sig")
    reliable.to_csv(paths["reliable"], index=False, encoding="utf-8-sig")
    medium.to_csv(paths["medium"], index=False, encoding="utf-8-sig")
    low.to_csv(paths["low"], index=False, encoding="utf-8-sig")

    manifest = [
        build_manifest_entry(path, purpose=name, derived_from="20_reversal_candidates.csv, 31_15m_leads_1h_reversal_candidates.csv, raw candles", view_of="shared_backtest")
        for name, path in paths.items()
    ]
    write_manifest(out_dir / "output_manifest.json", manifest)
    validation = {
        "ledger": validate_dataframe(ledger, ["entry_time", "exit_time", "net_return", "mfe_until_exit", "mae_until_exit"]),
        "comparison": validate_dataframe(comparison, ["rule_name", "exit_policy", "n_trades", "avg_return_net"]),
        "decision_table": validate_decision_table(reliable),
        "manifest": validate_manifest(manifest),
    }
    (out_dir / "validation_report.json").write_text(pd.Series(validation).to_json(force_ascii=False, indent=2), encoding="utf-8")
    return BacktestResults(ledger, comparison, reliable, medium, low, manifest, validation, out_dir)
