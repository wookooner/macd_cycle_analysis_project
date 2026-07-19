"""Successful down-cycle hierarchy analysis.

For each timeframe (15m / 1h / 4h / 1d):
1. Take all down-classified cycles.
2. Enrich each cycle with start/end price, drop%, duration, PPO+hist at both
   start and end.
3. Classify success tier by absolute drop magnitude:
   - "strong" = top 25% biggest drops in that TF
   - "mid"    = middle 50%
   - "weak"   = bottom 25% (smallest drops; may even be flat/positive)
4. For each down cycle, look up:
   - PARENT TF cycle state at start_date and at end_date
     (cycle direction, progress fraction, bars_since_parent_start)
   - PARENT TF cycle state PPO/PPO_hist at those moments
   - CHILD TF cycle composition inside [start_date, end_date]:
     count of children, up vs down, first child direction, last child direction,
     average/longest child duration, fraction of time spent in down children.
5. Aggregate features by success tier and timeframe.
6. Compare strong vs weak down cycles → identify what differs at the *start*,
   what differs at the *end*, and what differs in the *child composition*.

Outputs go to ``outputs/analysis_results/successful_down_cycle_hierarchy/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import PROJECT_PATHS  # noqa: E402

TIMEFRAMES = ("15m", "1h", "4h", "1d")
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

# (parent_tf, child_tf) — parent is the LARGER timeframe, child is the SMALLER.
# 15m is the smallest TF available, so child_tf=None.
# 1d is the largest TF analysed, so parent_tf=None.
HIERARCHY = {
    "15m": ("1h", None),
    "1h": ("4h", "15m"),
    "4h": ("1d", "1h"),
    "1d": (None, "4h"),
}

# fraction edges for tiering by absolute drop %
TIER_QUANTILES = (0.25, 0.75)

LOW_SAMPLE_N = 30


def output_dir() -> Path:
    return PROJECT_PATHS.outputs_root / "analysis_results" / "successful_down_cycle_hierarchy"


def _read_timestamp(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def _raw_market_path(timeframe: str) -> Path:
    candidates = [
        PROJECT_PATHS.raw_market_dir / f"BTCUSD_{timeframe}.csv",
        PROJECT_PATHS.raw_market_dir / f"BTCUSDT_{timeframe}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing raw candle file for {timeframe}: tried {candidates}")


def _cycle_path(timeframe: str) -> Path:
    candidates = [
        PROJECT_PATHS.cycle_structured_dir / "btc" / f"cycles_{timeframe}.parquet",
        PROJECT_PATHS.asset_cycle_dir("btc") / f"cycles_{timeframe}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing cycle parquet for {timeframe}: tried {candidates}")


def _tf_delta(tf: str) -> pd.Timedelta:
    return pd.to_timedelta(TF_SECONDS[tf], unit="s")


def load_candles(timeframe: str) -> pd.DataFrame:
    path = _raw_market_path(timeframe)
    cols_present = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in ("date", "timestamp", "open_time", "open", "high", "low", "close", "ppo", "ppo_hist") if c in cols_present]
    df = pd.read_csv(path, usecols=usecols).copy()
    ts_col = next((c for c in ("timestamp", "open_time", "date") if c in df.columns), None)
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = _read_timestamp(df["timestamp"])
    for col in ("open", "high", "low", "close", "ppo", "ppo_hist"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "ppo", "ppo_hist"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["timeframe"] = timeframe
    return df


def load_cycles(timeframe: str) -> pd.DataFrame:
    df = pd.read_parquet(_cycle_path(timeframe), columns=["cycle_id", "start_date", "end_date", "cycle_type", "duration_candles", "category"]).copy()
    df["start_date"] = _read_timestamp(df["start_date"])
    df["end_date"] = _read_timestamp(df["end_date"])
    df = df.dropna(subset=["start_date", "end_date"]).sort_values("start_date").reset_index(drop=True)
    df["cycle_sign"] = np.where(df["cycle_type"].astype(str).str.lower().isin({"up", "long", "1", "+1"}), 1, -1).astype(np.int8)
    df["end_exclusive"] = df["end_date"] + _tf_delta(timeframe)
    df["duration_candles"] = pd.to_numeric(df["duration_candles"], errors="coerce")
    return df


def lookup_close(candles: pd.DataFrame, ts: pd.Timestamp) -> tuple[float, float, float]:
    """Return (close, ppo, ppo_hist) at the candle whose start <= ts < next_start.
    Falls back to the last bar at-or-before ts.
    """
    if pd.isna(ts):
        return np.nan, np.nan, np.nan
    times = candles["timestamp"].to_numpy()
    idx = int(np.searchsorted(times.astype("datetime64[ns]"), np.datetime64(ts), side="right") - 1)
    if idx < 0 or idx >= len(candles):
        return np.nan, np.nan, np.nan
    row = candles.iloc[idx]
    return float(row["close"]), float(row["ppo"]), float(row["ppo_hist"])


def enrich_cycles(cycles: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    starts: list[float] = []
    ends: list[float] = []
    ppo_starts: list[float] = []
    ppo_ends: list[float] = []
    hist_starts: list[float] = []
    hist_ends: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for _, row in cycles.iterrows():
        s_close, s_ppo, s_hist = lookup_close(candles, row["start_date"])
        e_close, e_ppo, e_hist = lookup_close(candles, row["end_date"])
        starts.append(s_close)
        ends.append(e_close)
        ppo_starts.append(s_ppo)
        ppo_ends.append(e_ppo)
        hist_starts.append(s_hist)
        hist_ends.append(e_hist)
        # Low/high during cycle window
        mask = (candles["timestamp"] >= row["start_date"]) & (candles["timestamp"] < row["end_exclusive"])
        sub = candles.loc[mask]
        if sub.empty:
            lows.append(np.nan)
            highs.append(np.nan)
        else:
            lows.append(float(sub["low"].min()))
            highs.append(float(sub["high"].max()))
    df = cycles.copy()
    df["close_at_start"] = starts
    df["close_at_end"] = ends
    df["ppo_at_start"] = ppo_starts
    df["ppo_at_end"] = ppo_ends
    df["hist_at_start"] = hist_starts
    df["hist_at_end"] = hist_ends
    df["min_low_in_cycle"] = lows
    df["max_high_in_cycle"] = highs
    df["price_change_pct"] = (df["close_at_end"] / df["close_at_start"] - 1.0) * 100.0
    df["max_drawdown_pct"] = (df["min_low_in_cycle"] / df["close_at_start"] - 1.0) * 100.0
    df["max_runup_pct"] = (df["max_high_in_cycle"] / df["close_at_start"] - 1.0) * 100.0
    return df


def assign_success_tier(down_cycles: pd.DataFrame) -> pd.DataFrame:
    """Tier by *absolute* end-to-end drop in down cycles."""
    out = down_cycles.copy()
    drop = -out["price_change_pct"]  # positive value means actual drop magnitude
    q_low, q_high = drop.quantile(TIER_QUANTILES[0]), drop.quantile(TIER_QUANTILES[1])
    out["drop_pct"] = drop
    out["success_tier"] = np.select(
        [drop <= q_low, drop >= q_high],
        ["weak", "strong"],
        default="mid",
    )
    out["tier_quantile_low"] = float(q_low)
    out["tier_quantile_high"] = float(q_high)
    return out


def parent_state_at(parent_cycles: pd.DataFrame, parent_tf: str, ts: pd.Timestamp) -> dict[str, Any]:
    """Return parent cycle state at timestamp ts (no look-ahead beyond
    retrospective duration_candles)."""
    out = {
        "parent_cycle_id": np.nan,
        "parent_cycle_dir": np.nan,
        "parent_cycle_progress": np.nan,
        "parent_bars_since_start": np.nan,
    }
    if parent_cycles.empty or pd.isna(ts):
        return out
    sub = parent_cycles[(parent_cycles["start_date"] <= ts) & (ts < parent_cycles["end_exclusive"])]
    if sub.empty:
        return out
    row = sub.iloc[0]
    elapsed = (ts - row["start_date"]) / _tf_delta(parent_tf)
    duration = row["duration_candles"] if pd.notna(row["duration_candles"]) and row["duration_candles"] > 0 else np.nan
    out["parent_cycle_id"] = row["cycle_id"]
    out["parent_cycle_dir"] = "up" if row["cycle_sign"] > 0 else "down"
    out["parent_cycle_progress"] = float(elapsed / duration) if duration else np.nan
    out["parent_bars_since_start"] = float(elapsed)
    return out


def child_composition(child_cycles: pd.DataFrame, child_tf: str, start: pd.Timestamp, end_excl: pd.Timestamp) -> dict[str, Any]:
    out = {
        "n_children": 0,
        "n_children_up": 0,
        "n_children_down": 0,
        "first_child_dir": np.nan,
        "last_child_dir": np.nan,
        "avg_child_duration": np.nan,
        "max_child_duration": np.nan,
        "frac_time_in_down_children": np.nan,
        "first_child_progress_at_parent_start": np.nan,
    }
    if child_cycles.empty or pd.isna(start) or pd.isna(end_excl):
        return out
    # children whose start is within [start, end_excl)
    inside = child_cycles[(child_cycles["start_date"] >= start) & (child_cycles["start_date"] < end_excl)].copy()
    out["n_children"] = int(len(inside))
    if not inside.empty:
        ups = (inside["cycle_sign"] > 0).sum()
        downs = (inside["cycle_sign"] < 0).sum()
        out["n_children_up"] = int(ups)
        out["n_children_down"] = int(downs)
        out["first_child_dir"] = "up" if inside["cycle_sign"].iat[0] > 0 else "down"
        out["last_child_dir"] = "up" if inside["cycle_sign"].iat[-1] > 0 else "down"
        out["avg_child_duration"] = float(inside["duration_candles"].mean())
        out["max_child_duration"] = float(inside["duration_candles"].max())

    # fraction of time spent in *any* down child (overlap-clipped)
    child_delta = _tf_delta(child_tf)
    child_intervals = child_cycles[(child_cycles["end_exclusive"] > start) & (child_cycles["start_date"] < end_excl)].copy()
    if not child_intervals.empty:
        overlap_start = child_intervals["start_date"].clip(lower=start)
        overlap_end = child_intervals["end_exclusive"].clip(upper=end_excl)
        overlap_seconds = (overlap_end - overlap_start).dt.total_seconds().clip(lower=0)
        is_down = child_intervals["cycle_sign"] < 0
        total = overlap_seconds.sum()
        down_total = overlap_seconds[is_down].sum()
        out["frac_time_in_down_children"] = float(down_total / total) if total > 0 else np.nan

        # progress of first child still active at parent's start_date
        active_at_start = child_intervals[(child_intervals["start_date"] <= start) & (child_intervals["end_exclusive"] > start)]
        if not active_at_start.empty:
            row = active_at_start.iloc[0]
            elapsed = (start - row["start_date"]) / child_delta
            duration = row["duration_candles"] if pd.notna(row["duration_candles"]) and row["duration_candles"] > 0 else np.nan
            if duration:
                out["first_child_progress_at_parent_start"] = float(elapsed / duration)
    return out


def build_features(tf: str, enriched: pd.DataFrame, parent_cycles: pd.DataFrame | None, parent_tf: str | None, child_cycles: pd.DataFrame | None, child_tf: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, c in enriched.iterrows():
        feat: dict[str, Any] = {
            "timeframe": tf,
            "cycle_id": c["cycle_id"],
            "category": c["category"],
            "duration_candles": c["duration_candles"],
            "start_date": c["start_date"],
            "end_date": c["end_date"],
            "close_at_start": c["close_at_start"],
            "close_at_end": c["close_at_end"],
            "price_change_pct": c["price_change_pct"],
            "drop_pct": c["drop_pct"],
            "max_drawdown_pct": c["max_drawdown_pct"],
            "max_runup_pct": c["max_runup_pct"],
            "ppo_at_start": c["ppo_at_start"],
            "ppo_at_end": c["ppo_at_end"],
            "hist_at_start": c["hist_at_start"],
            "hist_at_end": c["hist_at_end"],
            "ppo_delta": c["ppo_at_end"] - c["ppo_at_start"] if pd.notna(c["ppo_at_start"]) and pd.notna(c["ppo_at_end"]) else np.nan,
            "success_tier": c["success_tier"],
        }
        if parent_cycles is not None and parent_tf is not None:
            for label, ts in (("at_start", c["start_date"]), ("at_end", c["end_date"])):
                p = parent_state_at(parent_cycles, parent_tf, ts)
                feat[f"parent_{label}_dir"] = p["parent_cycle_dir"]
                feat[f"parent_{label}_progress"] = p["parent_cycle_progress"]
                feat[f"parent_{label}_bars_since_start"] = p["parent_bars_since_start"]
        if child_cycles is not None and child_tf is not None:
            comp = child_composition(child_cycles, child_tf, c["start_date"], c["end_exclusive"])
            for k, v in comp.items():
                feat[f"child_{k}"] = v
        rows.append(feat)
    return pd.DataFrame(rows)


def progress_bucket(value: float) -> str:
    if pd.isna(value):
        return "na"
    if value < 0.33:
        return "early"
    if value < 0.66:
        return "mid"
    return "late"


def summarize_by_tier(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows = []
    numeric_cols = [
        "duration_candles", "drop_pct", "max_drawdown_pct", "max_runup_pct",
        "ppo_at_start", "ppo_at_end", "hist_at_start", "hist_at_end", "ppo_delta",
        "parent_at_start_progress", "parent_at_end_progress",
        "parent_at_start_bars_since_start", "parent_at_end_bars_since_start",
        "child_n_children", "child_n_children_up", "child_n_children_down",
        "child_avg_child_duration", "child_max_child_duration",
        "child_frac_time_in_down_children", "child_first_child_progress_at_parent_start",
    ]
    cat_cols = [
        "parent_at_start_dir", "parent_at_end_dir",
        "child_first_child_dir", "child_last_child_dir",
    ]
    for (tf, tier), group in features.groupby(["timeframe", "success_tier"], sort=False):
        row = {"timeframe": tf, "success_tier": tier, "n_cycles": int(len(group))}
        for col in numeric_cols:
            if col in group.columns:
                val = pd.to_numeric(group[col], errors="coerce")
                row[f"{col}_mean"] = float(val.mean())
                row[f"{col}_median"] = float(val.median())
        for col in cat_cols:
            if col in group.columns:
                vc = group[col].fillna("na").value_counts(normalize=True)
                for k, v in vc.items():
                    row[f"{col}__{k}_pct"] = float(v * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_breakdown(features: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in features.groupby(["timeframe", "success_tier"] + group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(group)
        row = {"timeframe": keys[0], "success_tier": keys[1]}
        for i, col in enumerate(group_cols):
            row[col] = keys[2 + i]
        row.update(
            {
                "n_cycles": n,
                "avg_drop_pct": float(group["drop_pct"].mean()),
                "median_drop_pct": float(group["drop_pct"].median()),
                "avg_duration_candles": float(group["duration_candles"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out_dir: Path, by_tier: pd.DataFrame, parent_start_breakdown: pd.DataFrame, parent_end_breakdown: pd.DataFrame, child_dir_breakdown: pd.DataFrame, all_features: pd.DataFrame) -> None:
    def fmt(df: pd.DataFrame, cols: list[str], n: int = 60) -> str:
        if df.empty:
            return "_no rows_"
        cols = [c for c in cols if c in df.columns]
        return df[cols].head(n).to_markdown(index=False, floatfmt=".2f")

    lines: list[str] = ["# Successful Down-Cycle Hierarchy Analysis\n"]
    lines.append(
        "## 0. 분석 정의\n"
        "- 각 시간대(15m/1h/4h/1d)의 *down* 분류 사이클만 대상.\n"
        "- 성공 등급(`success_tier`)은 사이클 시작가에서 종료가까지의 실제 하락폭(`drop_pct = -price_change_pct`) 기준 분위:\n"
        "  - `strong` = 그 시간대 down 사이클 중 하락폭 상위 25%\n"
        "  - `mid`    = 중간 50%\n"
        "  - `weak`   = 하락폭 하위 25% (가장 약한 하락 또는 거의 횡보)\n"
        "- 시작점/종료점에서 부모 사이클(상위 시간대) 상태와 자식 사이클(하위 시간대) 구성을 함께 기록.\n"
        "- 사이클 진행도(early < 0.33, mid 0.33–0.66, late >= 0.66)는 사후적(`duration_candles`) 기준.\n"
    )

    lines.append("\n## 1. 등급별 평균 통계 (시간대 × success_tier)\n")
    lines.append(
        fmt(
            by_tier,
            [
                "timeframe", "success_tier", "n_cycles",
                "drop_pct_mean", "drop_pct_median",
                "duration_candles_mean", "duration_candles_median",
                "max_drawdown_pct_mean", "max_runup_pct_mean",
                "ppo_at_start_mean", "ppo_at_end_mean",
                "hist_at_start_mean", "hist_at_end_mean",
                "ppo_delta_mean",
                "parent_at_start_progress_mean", "parent_at_end_progress_mean",
                "parent_at_start_bars_since_start_mean", "parent_at_end_bars_since_start_mean",
                "child_n_children_mean", "child_n_children_up_mean", "child_n_children_down_mean",
                "child_frac_time_in_down_children_mean",
            ],
            n=20,
        )
    )

    lines.append("\n## 2. 시작점에서의 부모 사이클 방향 분포\n")
    lines.append(
        fmt(
            parent_start_breakdown.sort_values(["timeframe", "success_tier", "parent_at_start_dir"]),
            ["timeframe", "success_tier", "parent_at_start_dir", "n_cycles", "avg_drop_pct", "avg_duration_candles"],
            n=80,
        )
    )

    lines.append("\n## 3. 종료점에서의 부모 사이클 방향 분포\n")
    lines.append(
        fmt(
            parent_end_breakdown.sort_values(["timeframe", "success_tier", "parent_at_end_dir"]),
            ["timeframe", "success_tier", "parent_at_end_dir", "n_cycles", "avg_drop_pct", "avg_duration_candles"],
            n=80,
        )
    )

    lines.append("\n## 4. 자식 사이클의 첫·마지막 방향 조합\n")
    lines.append(
        fmt(
            child_dir_breakdown.sort_values(["timeframe", "success_tier", "child_first_child_dir", "child_last_child_dir"]),
            ["timeframe", "success_tier", "child_first_child_dir", "child_last_child_dir", "n_cycles", "avg_drop_pct", "avg_duration_candles"],
            n=80,
        )
    )

    # 자체 progress 등급별
    if not all_features.empty:
        df = all_features.copy()
        df["parent_at_start_progress_bucket"] = df["parent_at_start_progress"].apply(progress_bucket)
        df["parent_at_end_progress_bucket"] = df["parent_at_end_progress"].apply(progress_bucket)
        ps = categorical_breakdown(df, ["parent_at_start_progress_bucket"])
        pe = categorical_breakdown(df, ["parent_at_end_progress_bucket"])
        lines.append("\n## 5. 시작점에서의 부모 사이클 진행도 분포\n")
        lines.append(
            fmt(
                ps.sort_values(["timeframe", "success_tier", "parent_at_start_progress_bucket"]),
                ["timeframe", "success_tier", "parent_at_start_progress_bucket", "n_cycles", "avg_drop_pct"],
                n=80,
            )
        )
        lines.append("\n## 6. 종료점에서의 부모 사이클 진행도 분포\n")
        lines.append(
            fmt(
                pe.sort_values(["timeframe", "success_tier", "parent_at_end_progress_bucket"]),
                ["timeframe", "success_tier", "parent_at_end_progress_bucket", "n_cycles", "avg_drop_pct"],
                n=80,
            )
        )

    lines.append(
        "\n## 7. 해석 가이드\n"
        "- 동일 시간대에서 strong vs weak 행을 비교하면, *그 등급에 들어간 사이클들이 시작/종료 시점에 어느 부모 방향·부모 진행도·자식 구성을 갖고 있었는지* 가 드러난다.\n"
        "- 예) strong에서 `parent_at_start_dir=down` 비중이 weak보다 크게 높으면, *상위 추세가 이미 아래일 때 시작한 하락 사이클이 더 깊이 떨어진다* 는 신호.\n"
        "- 예) strong에서 `parent_at_end_dir=up` 비중이 weak보다 크게 높으면, *부모 추세가 상승으로 막 전환되는 시점에 이 하락이 자연스레 끝남* 을 시사.\n"
        "- 자식 first/last 방향 조합은 큰 흐름의 도입부와 마무리 모양을 알려준다. (예: first=up, last=down → 위로 잠깐 튄 뒤 본격 하락 시작)\n"
        "- 표본 30 미만(`n_cycles<30`)은 결론에서 제외 권장.\n"
    )
    (out_dir / "successful_down_cycle_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframes", nargs="*", default=list(TIMEFRAMES), choices=TIMEFRAMES)
    args = parser.parse_args()

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    candles = {tf: load_candles(tf) for tf in TIMEFRAMES}
    cycles = {tf: load_cycles(tf) for tf in TIMEFRAMES}
    print({tf: (len(candles[tf]), len(cycles[tf])) for tf in TIMEFRAMES})

    enriched_by_tf: dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        e = enrich_cycles(cycles[tf], candles[tf])
        e_down = e[e["cycle_sign"] < 0].copy()
        e_down = assign_success_tier(e_down)
        enriched_by_tf[tf] = e_down
        print(f"{tf}: down cycles={len(e_down)}  q25={e_down['drop_pct'].quantile(0.25):.2f}%  q75={e_down['drop_pct'].quantile(0.75):.2f}%")

    feat_frames: list[pd.DataFrame] = []
    for tf in args.timeframes:
        parent_tf, child_tf = HIERARCHY[tf]
        feat = build_features(
            tf=tf,
            enriched=enriched_by_tf[tf],
            parent_cycles=cycles[parent_tf] if parent_tf else None,
            parent_tf=parent_tf,
            child_cycles=cycles[child_tf] if child_tf else None,
            child_tf=child_tf,
        )
        feat.to_csv(out_dir / f"down_cycles_{tf}_features.csv", index=False, encoding="utf-8-sig")
        feat_frames.append(feat)

    all_features = pd.concat(feat_frames, ignore_index=True)
    all_features.to_csv(out_dir / "00_all_down_cycles_features.csv", index=False, encoding="utf-8-sig")

    by_tier = summarize_by_tier(all_features)
    by_tier.to_csv(out_dir / "10_summary_by_tier.csv", index=False, encoding="utf-8-sig")

    parent_start = categorical_breakdown(all_features, ["parent_at_start_dir"])
    parent_start.to_csv(out_dir / "20_parent_at_start_dir.csv", index=False, encoding="utf-8-sig")
    parent_end = categorical_breakdown(all_features, ["parent_at_end_dir"])
    parent_end.to_csv(out_dir / "21_parent_at_end_dir.csv", index=False, encoding="utf-8-sig")

    child_dir = categorical_breakdown(all_features, ["child_first_child_dir", "child_last_child_dir"])
    child_dir.to_csv(out_dir / "30_child_first_last_dir.csv", index=False, encoding="utf-8-sig")

    # also produce richer parent x progress breakdowns
    enriched_features = all_features.copy()
    enriched_features["parent_at_start_progress_bucket"] = enriched_features["parent_at_start_progress"].apply(progress_bucket)
    enriched_features["parent_at_end_progress_bucket"] = enriched_features["parent_at_end_progress"].apply(progress_bucket)
    p_start_prog = categorical_breakdown(enriched_features, ["parent_at_start_dir", "parent_at_start_progress_bucket"])
    p_start_prog.to_csv(out_dir / "22_parent_at_start_dir_x_progress.csv", index=False, encoding="utf-8-sig")
    p_end_prog = categorical_breakdown(enriched_features, ["parent_at_end_dir", "parent_at_end_progress_bucket"])
    p_end_prog.to_csv(out_dir / "23_parent_at_end_dir_x_progress.csv", index=False, encoding="utf-8-sig")

    write_report(out_dir, by_tier, parent_start, parent_end, child_dir, all_features)
    print(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
