"""
Build cycle dimension table, timeframe context, and enriched cycle parquets.

Pipeline Step 5 — replaces cycle_hierarchy_map.json with:
  - cycle_dim.parquet          : surrogate-keyed dimension table for all TF cycles
  - timeframe_context_1min.parquet : per-minute snapshot of all TF cycle states
  - timeframe_context_1h.parquet   : hourly subset of the above
  - context_meta.json          : metadata
  - cycles_*.parquet (enriched): parent/child/sibling relation columns appended
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("context_builder")

TF_ORDER = ["1M", "1w", "1d", "4h", "1h", "30m", "15m", "5m", "1min"]
TF_MAJOR = ["1w", "1d", "4h", "1h"]
TF_MINOR = ["30m", "15m", "5m", "1min"]
TF_MACRO = ["1M"]

# Maps TF label to raw CSV filename suffix
_RAW_CSV_NAMES: dict[str, str] = {
    "1M": "BTCUSD_1M.csv",
    "1w": "BTCUSD_1w.csv",
    "1d": "BTCUSD_1d.csv",
    "4h": "BTCUSD_4h.csv",
    "1h": "BTCUSD_1h.csv",
    "30m": "BTCUSD_30m.csv",
    "15m": "BTCUSD_15m.csv",
    "5m": "BTCUSD_5m.csv",
    "1min": "BTCUSD_1min.csv",
}

# Candle duration in nanoseconds — end_date stores start of last candle,
# so effective cycle end = end_date + candle_duration_ns
_CANDLE_NS: dict[str, int] = {
    "1M": 30 * 24 * 3600 * 10**9,
    "1w":  7 * 24 * 3600 * 10**9,
    "1d":      24 * 3600 * 10**9,
    "4h":       4 * 3600 * 10**9,
    "1h":           3600 * 10**9,
    "30m":        30 * 60 * 10**9,
    "15m":        15 * 60 * 10**9,
    "5m":          5 * 60 * 10**9,
    "1min":        1 * 60 * 10**9,
}

_TF_PARENT: dict[str, str | None] = {
    "1M": None,
    "1w": "1M",
    "1d": "1w",
    "4h": "1d",
    "1h": "4h",
    "30m": "1h",
    "15m": "30m",
    "5m": "15m",
    "1min": "5m",
}

_TF_CHILD: dict[str, str | None] = {v: k for k, v in _TF_PARENT.items() if v is not None}
_TF_CHILD["1min"] = None


def _elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """Convert timestamps robustly (string / unix-s / unix-ms)."""
    for kwargs in ({}, {"unit": "s"}, {"unit": "ms"}):
        try:
            result = pd.to_datetime(series, **kwargs, utc=False)
            if hasattr(result, "dt"):
                result = result.dt.tz_localize(None)
            return result
        except (ValueError, TypeError, AttributeError):
            continue
    return series


def _to_ns_int64(dt_series: pd.Series) -> np.ndarray:
    """Convert any datetime series to int64 nanoseconds since epoch."""
    s = _safe_to_datetime(dt_series)
    # normalise to ns regardless of internal dtype (s/ms/us/ns)
    return s.values.astype("datetime64[ns]").astype("int64")


def _type_int(cycle_type_series: pd.Series) -> pd.Series:
    """Normalise 'up'/'down' strings to int8 1/-1 (handles Arrow-backed strings)."""
    s = cycle_type_series
    # Convert Arrow string or any non-numeric dtype to plain Python strings first
    if hasattr(s, "dtype") and s.dtype != np.dtype("int8"):
        try:
            s_str = s.astype(str).str.lower()
            return s_str.map({"up": 1, "down": -1, "1": 1, "-1": -1}).fillna(0).astype("int8")
        except Exception:
            pass
    return s.astype("int8")


class CycleContextBuilder:
    """Orchestrates all 5 phases of the context build."""

    def __init__(
        self,
        asset_cycle_dir: Path,
        context_dir: Path,
        raw_market_dir: Path,
        asset: str = "btc",
    ) -> None:
        self.asset_cycle_dir = Path(asset_cycle_dir)
        self.context_dir = Path(context_dir)
        self.raw_market_dir = Path(raw_market_dir)
        self.asset = asset

        self.dim_path = self.context_dir / "cycle_dim.parquet"
        self.ctx_1min_path = self.context_dir / "timeframe_context_1min.parquet"
        self.ctx_1h_path = self.context_dir / "timeframe_context_1h.parquet"
        self.meta_path = self.context_dir / "context_meta.json"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_all(self, skip_phases: list[int] | None = None) -> bool:
        """Run all build phases. Pass skip_phases=[1,2] to reuse existing dim/context."""
        self.context_dir.mkdir(parents=True, exist_ok=True)
        t_total = time.time()
        skip = set(skip_phases or [])
        try:
            if 1 not in skip:
                LOGGER.info("=== Phase 1: cycle_dim ===")
                dim = self._phase1_build_cycle_dim()
            else:
                LOGGER.info("=== Phase 1: cycle_dim (skipped, loading existing) ===")
                dim = pd.read_parquet(self.dim_path)

            if 2 not in skip:
                LOGGER.info("=== Phase 2: timeframe_context ===")
                self._phase2_build_timeframe_context(dim)
            else:
                LOGGER.info("=== Phase 2: timeframe_context (skipped) ===")

            LOGGER.info("=== Phase 3: enrich cycle parquets ===")
            ctx_1min = pd.read_parquet(self.ctx_1min_path)
            self._phase3_enrich_parquets(dim, ctx_1min)

            LOGGER.info("=== Phase 4: validate ===")
            self._phase4_validate(dim)

            LOGGER.info("=== Phase 5: context_meta.json ===")
            self._phase5_write_meta(dim)

            LOGGER.info("Context build complete (%s)", _elapsed(t_total))
            return True
        except Exception:
            LOGGER.error("Context build failed\n%s", traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Phase 1 — cycle_dim
    # ------------------------------------------------------------------

    def _phase1_build_cycle_dim(self) -> pd.DataFrame:
        t0 = time.time()
        frames: list[pd.DataFrame] = []

        for tf in TF_ORDER:
            path = self.asset_cycle_dir / f"cycles_{tf}.parquet"
            if not path.exists():
                LOGGER.warning("  [skip] %s: not found", tf)
                continue

            df = pd.read_parquet(path)

            # normalise required columns
            for col in ["start_date", "end_date"]:
                if col in df.columns:
                    df[col] = _safe_to_datetime(df[col])

            # handle parquets that store timeframe as a different value (e.g. 1M file has '1M')
            if "timeframe" not in df.columns:
                df["timeframe"] = tf

            keep = ["cycle_id", "timeframe", "cycle_type", "start_date", "end_date", "duration_candles"]
            if "category" in df.columns:
                keep.append("category")
            df = df[[c for c in keep if c in df.columns]].copy()
            df["_tf_order"] = TF_ORDER.index(tf)
            frames.append(df)
            LOGGER.info("  [loaded] %s: %d cycles", tf, len(df))

        if not frames:
            raise RuntimeError("No cycle parquets found in %s" % self.asset_cycle_dir)

        dim = pd.concat(frames, ignore_index=True)
        dim = dim.sort_values(["_tf_order", "start_date"]).drop(columns=["_tf_order"]).reset_index(drop=True)
        dim.insert(0, "cycle_key", dim.index.astype("int32"))
        dim["cycle_type"] = _type_int(dim["cycle_type"])
        dim["start_date"] = dim["start_date"].astype("datetime64[s]")
        dim["end_date"] = dim["end_date"].astype("datetime64[s]")
        if "duration_candles" in dim.columns:
            dim["duration_candles"] = dim["duration_candles"].astype("int16")

        dim.to_parquet(self.dim_path, index=False)
        LOGGER.info("  cycle_dim saved: %d rows, %s (%s)", len(dim), self.dim_path.name, _elapsed(t0))
        return dim

    # ------------------------------------------------------------------
    # Phase 2 — timeframe_context
    # ------------------------------------------------------------------

    def _phase2_build_timeframe_context(self, dim: pd.DataFrame) -> None:
        t0 = time.time()

        ts_start = dim["start_date"].min()
        ts_end = dim["end_date"].max()
        LOGGER.info("  timestamp range: %s → %s", ts_start, ts_end)

        timestamps = pd.date_range(start=ts_start, end=ts_end, freq="1min").astype("datetime64[s]")
        ts_np = timestamps.values.astype("datetime64[ns]").astype("int64")  # nanoseconds since epoch
        LOGGER.info("  timestamp count: %d", len(timestamps))

        ctx = pd.DataFrame({"timestamp": timestamps})

        # --- per-TF mapping ---
        candle_ts_cache: dict[str, np.ndarray] = {}
        for tf in TF_ORDER:
            tf_dim = dim[dim["timeframe"] == tf].sort_values("start_date").reset_index(drop=True)
            if tf_dim.empty:
                LOGGER.warning("  [skip] %s: no cycles in dim", tf)
                ctx[f"{tf}_key"] = np.int32(0)
                ctx[f"{tf}_type"] = np.int8(0)
                ctx[f"{tf}_time_prog"] = np.float16(0)
                ctx[f"{tf}_candle_prog"] = np.float16(0)
                continue

            starts = _to_ns_int64(tf_dim["start_date"])
            ends = _to_ns_int64(tf_dim["end_date"])
            keys = tf_dim["cycle_key"].values.astype("int32")
            types = tf_dim["cycle_type"].values.astype("int8")
            durs = tf_dim["duration_candles"].values.astype("int16") if "duration_candles" in tf_dim.columns else None

            # searchsorted mapping
            idx = np.searchsorted(starts, ts_np, side="right") - 1
            idx_clipped = np.clip(idx, 0, len(starts) - 1)
            valid = (idx >= 0) & (ts_np < ends[idx_clipped])

            result_key = np.where(valid, keys[idx_clipped], np.int32(0)).astype("int32")
            result_type = np.where(valid, types[idx_clipped], np.int8(0)).astype("int8")

            # time_prog
            safe_starts = np.where(valid, starts[idx_clipped], ts_np)
            safe_ends = np.where(valid, ends[idx_clipped], ts_np + 1)
            duration_ns = (safe_ends - safe_starts).astype("float64")
            duration_ns = np.where(duration_ns == 0, 1, duration_ns)
            time_prog = np.where(valid, (ts_np - safe_starts) / duration_ns, np.float32(0)).astype("float32")
            time_prog = np.clip(time_prog, 0.0, 1.0)

            # candle_prog (best effort from raw CSV)
            candle_prog = time_prog.copy()
            raw_csv = self.raw_market_dir / _RAW_CSV_NAMES.get(tf, "")
            if raw_csv.exists() and tf not in candle_ts_cache:
                try:
                    raw_df = pd.read_csv(raw_csv, usecols=["unix"], dtype={"unix": "int64"})
                    raw_df = raw_df.sort_values("unix").drop_duplicates()
                    # unix values may be in seconds or milliseconds
                    unix_vals = raw_df["unix"].values
                    if unix_vals[0] > 1e12:  # milliseconds → seconds
                        unix_vals = unix_vals // 1000
                    # convert seconds → nanoseconds to match ts_np
                    candle_ts_cache[tf] = (unix_vals * 1_000_000_000).astype("int64")
                except Exception:
                    LOGGER.warning("  [candle_prog] could not read raw CSV for %s, falling back to time_prog", tf)

            if tf in candle_ts_cache:
                c_ts = candle_ts_cache[tf]
                # for each active cycle: count candles elapsed up to ts
                # vectorised approach: for each ts, find position in candle_ts within cycle range
                cycle_start_candle_idx = np.searchsorted(c_ts, starts, side="left")
                elapsed_up_to_ts = np.searchsorted(c_ts, ts_np, side="right")

                for i in range(len(starts)):
                    mask = valid & (idx_clipped == i)
                    if not np.any(mask):
                        continue
                    candle_count_in_cycle = np.searchsorted(c_ts, ends[i], side="left") - cycle_start_candle_idx[i]
                    if candle_count_in_cycle == 0:
                        continue
                    elapsed = elapsed_up_to_ts[mask] - cycle_start_candle_idx[i]
                    candle_prog[mask] = np.clip(elapsed / candle_count_in_cycle, 0.0, 1.0).astype("float32")

            ctx[f"{tf}_key"] = result_key
            ctx[f"{tf}_type"] = result_type
            ctx[f"{tf}_time_prog"] = time_prog.astype("float16")
            ctx[f"{tf}_candle_prog"] = candle_prog.astype("float16")
            LOGGER.info("  [mapped] %s", tf)

        # --- transition detection ---
        for tf in TF_ORDER:
            key_col = f"{tf}_key"
            type_col = f"{tf}_type"
            ctx[f"{tf}_changed"] = (ctx[key_col] != ctx[key_col].shift(1)).astype(bool)
            prev_type = ctx[type_col].shift(1).fillna(0).astype("int8")
            curr_type = ctx[type_col]
            ctx[f"{tf}_chg_type"] = np.where(
                ~ctx[f"{tf}_changed"], np.int8(0),
                np.where((prev_type == 1) & (curr_type == -1), np.int8(1),
                np.where((prev_type == -1) & (curr_type == 1), np.int8(2),
                np.int8(0)))
            ).astype("int8")

        ctx["major_changed_any"] = ctx[[f"{tf}_changed" for tf in TF_MAJOR]].any(axis=1)
        ctx["minor_changed_any"] = ctx[[f"{tf}_changed" for tf in TF_MINOR]].any(axis=1)

        # --- derived columns ---
        major_types = ctx[[f"{tf}_type" for tf in TF_MAJOR]]
        minor_types = ctx[[f"{tf}_type" for tf in TF_MINOR]]
        all8_types = ctx[[f"{tf}_type" for tf in TF_ORDER if tf != "1M"]]

        ctx["n_up_4"] = (major_types == 1).sum(axis=1).astype("int8")
        ctx["n_up_8"] = (all8_types == 1).sum(axis=1).astype("int8")
        ctx["major_up_count"] = ctx["n_up_4"]
        ctx["minor_up_count"] = (minor_types == 1).sum(axis=1).astype("int8")

        ctx["combo_4"] = ctx.apply(
            lambda r: "".join("U" if r[f"{tf}_type"] == 1 else "D" for tf in TF_MAJOR), axis=1
        ).astype("category")
        ctx["combo_8"] = ctx.apply(
            lambda r: "".join("U" if r[f"{tf}_type"] == 1 else "D" for tf in TF_ORDER if tf != "1M"), axis=1
        ).astype("category")
        ctx["minor_combo_4"] = ctx.apply(
            lambda r: "".join("U" if r[f"{tf}_type"] == 1 else "D" for tf in TF_MINOR), axis=1
        ).astype("category")

        ctx["major_late_count"] = sum(
            (ctx[f"{tf}_time_prog"] > 0.8).astype("int8") for tf in TF_MAJOR
        ).astype("int8")

        # --- boundary detection ---
        boundary_pairs = [("4h", "1h"), ("1h", "30m"), ("30m", "15m"), ("15m", "5m")]
        for parent_tf, child_tf in boundary_pairs:
            ctx[f"{child_tf}_is_boundary"] = (
                (ctx[f"{parent_tf}_key"] != ctx[f"{parent_tf}_key"].shift(1)) &
                (ctx[f"{child_tf}_key"] == ctx[f"{child_tf}_key"].shift(1))
            ).astype(bool)

        # --- save ---
        ctx.to_parquet(self.ctx_1min_path, index=False)
        LOGGER.info("  timeframe_context_1min saved: %d rows (%s)", len(ctx), _elapsed(t0))

        ctx_1h = ctx[ctx["timestamp"].dt.minute == 0].reset_index(drop=True)
        ctx_1h.to_parquet(self.ctx_1h_path, index=False)
        LOGGER.info("  timeframe_context_1h saved: %d rows", len(ctx_1h))

    # ------------------------------------------------------------------
    # Phase 3 — enrich cycle parquets
    # ------------------------------------------------------------------

    def _phase3_enrich_parquets(self, dim: pd.DataFrame, ctx_1min: pd.DataFrame) -> None:
        t0 = time.time()

        # --- Pass 1: parent relation + sibling + major shortcut + chain (top-down 1M→1min) ---
        for tf in TF_ORDER:
            path = self.asset_cycle_dir / f"cycles_{tf}.parquet"
            if not path.exists():
                continue
            LOGGER.info("  [pass1] %s", tf)
            df = pd.read_parquet(path)
            df["start_date"] = _safe_to_datetime(df["start_date"])
            df["end_date"] = _safe_to_datetime(df["end_date"])
            df = df.sort_values("start_date").reset_index(drop=True)

            # add cycle_key from dim
            id_to_key = dim.set_index("cycle_id")["cycle_key"].to_dict()
            df["cycle_key"] = df["cycle_id"].map(id_to_key).astype("Int32")

            # --- sibling ---
            df["prev_key"] = df["cycle_key"].shift(1).astype("Int32")
            df["prev_type"] = _type_int(df["cycle_type"]).shift(1).astype("Int8")
            df["prev_dur"] = (df["duration_candles"].shift(1) if "duration_candles" in df.columns else pd.NA)
            if df["prev_dur"].notna().any():
                df["prev_dur"] = df["prev_dur"].astype("Int16")
            price_col = next((c for c in ["price_pct", "price_change_pct", "change_price_pct"] if c in df.columns), None)
            df["prev_price_pct"] = df[price_col].shift(1).astype("float32") if price_col else pd.array([0.0] * len(df), dtype="float32")

            parent_tf = _TF_PARENT.get(tf)

            if parent_tf is not None:
                parent_dim = dim[dim["timeframe"] == parent_tf].sort_values("start_date").reset_index(drop=True)
                p_starts = _to_ns_int64(parent_dim["start_date"])
                # extend parent end by one candle duration (end_date = start of last candle)
                p_ends = _to_ns_int64(parent_dim["end_date"]) + _CANDLE_NS.get(parent_tf, 0)
                p_keys = parent_dim["cycle_key"].values.astype("int32")
                p_types = parent_dim["cycle_type"].values.astype("int8")

                c_starts = _to_ns_int64(df["start_date"])
                c_ends = _to_ns_int64(df["end_date"])

                # for each child, find which parents overlap
                parent_key_arr = np.full(len(df), -1, dtype="int32")
                parent_type_arr = np.zeros(len(df), dtype="int8")
                boundary_type_arr = np.zeros(len(df), dtype="int8")
                parent_prev_key_arr = np.full(len(df), -1, dtype="int32")
                parent_prev_type_arr = np.zeros(len(df), dtype="int8")
                parent_next_key_arr = np.full(len(df), -1, dtype="int32")
                parent_next_type_arr = np.zeros(len(df), dtype="int8")
                overlap_prev_arr = np.zeros(len(df), dtype="float32")
                overlap_next_arr = np.zeros(len(df), dtype="float32")
                order_in_parent_arr = np.zeros(len(df), dtype="int16")
                parent_prog_start_arr = np.zeros(len(df), dtype="float32")
                parent_prog_end_arr = np.zeros(len(df), dtype="float32")
                parent_assign_rule_arr = np.zeros(len(df), dtype="int8")
                parent_assign_rule_arr[:] = 0  # 0=contained

                # find overlapping parents per child via two-pointer
                # for each child: overlapping parents are those where p_start < c_end AND p_end > c_start
                p_left = 0
                parent_groups: dict[int, list[int]] = {}  # parent_key → [child indices]

                for ci in range(len(df)):
                    c_s = c_starts[ci]
                    c_e = c_ends[ci]
                    child_dur = float(c_e - c_s) if c_e > c_s else 1.0

                    # advance left pointer
                    while p_left < len(p_ends) and p_ends[p_left] <= c_s:
                        p_left += 1

                    overlapping: list[tuple[int, float, float]] = []  # (parent_idx, overlap_dur, p_dur)
                    pi = p_left
                    while pi < len(p_starts) and p_starts[pi] < c_e:
                        if p_ends[pi] > c_s:
                            ov_s = max(c_s, p_starts[pi])
                            ov_e = min(c_e, p_ends[pi])
                            ov_dur = float(ov_e - ov_s)
                            overlapping.append((pi, ov_dur, float(p_ends[pi] - p_starts[pi])))
                        pi += 1

                    if not overlapping:
                        continue

                    if len(overlapping) == 1:
                        pi0 = overlapping[0][0]
                        parent_key_arr[ci] = p_keys[pi0]
                        parent_type_arr[ci] = p_types[pi0]
                        boundary_type_arr[ci] = 0
                        parent_assign_rule_arr[ci] = 0  # contained
                        p_dur = float(p_ends[pi0] - p_starts[pi0])
                        if p_dur > 0:
                            parent_prog_start_arr[ci] = (c_s - p_starts[pi0]) / p_dur
                            parent_prog_end_arr[ci] = (c_e - p_starts[pi0]) / p_dur
                        parent_groups.setdefault(int(p_keys[pi0]), []).append(ci)
                    else:
                        # boundary: by_start rule — assign to parent that contains c_s
                        primary_pi = None
                        for pi_o, _, _ in overlapping:
                            if p_starts[pi_o] <= c_s < p_ends[pi_o]:
                                primary_pi = pi_o
                                break
                        if primary_pi is None:
                            primary_pi = overlapping[0][0]  # fallback to largest overlap

                        # prev / next parent
                        prev_pi = overlapping[0][0]
                        next_pi = overlapping[1][0] if len(overlapping) > 1 else overlapping[0][0]

                        parent_key_arr[ci] = p_keys[primary_pi]
                        parent_type_arr[ci] = p_types[primary_pi]
                        parent_prev_key_arr[ci] = p_keys[prev_pi]
                        parent_prev_type_arr[ci] = p_types[prev_pi]
                        parent_next_key_arr[ci] = p_keys[next_pi]
                        parent_next_type_arr[ci] = p_types[next_pi]
                        overlap_prev_arr[ci] = overlapping[0][1] / child_dur
                        overlap_next_arr[ci] = overlapping[1][1] / child_dur if len(overlapping) > 1 else 0.0

                        # boundary_type
                        if p_types[prev_pi] != p_types[next_pi]:
                            child_type_val = int(_type_int(df["cycle_type"].iloc[[ci]]).iloc[0])
                            if child_type_val != int(p_types[prev_pi]):
                                boundary_type_arr[ci] = 2  # transition_trigger
                            else:
                                boundary_type_arr[ci] = 1  # straddle
                        else:
                            boundary_type_arr[ci] = 1

                        parent_assign_rule_arr[ci] = 1  # by_start
                        p_dur = float(p_ends[primary_pi] - p_starts[primary_pi])
                        if p_dur > 0:
                            parent_prog_start_arr[ci] = (c_s - p_starts[primary_pi]) / p_dur
                            parent_prog_end_arr[ci] = (c_e - p_starts[primary_pi]) / p_dur
                        parent_groups.setdefault(int(p_keys[primary_pi]), []).append(ci)

                # order_in_parent / total_siblings
                for pk, child_indices in parent_groups.items():
                    # sort child_indices by start_date
                    child_indices_sorted = sorted(child_indices, key=lambda i: c_starts[i])
                    for order, ci in enumerate(child_indices_sorted, start=1):
                        order_in_parent_arr[ci] = order
                total_siblings_arr = np.zeros(len(df), dtype="int16")
                for pk, child_indices in parent_groups.items():
                    n = len(child_indices)
                    for ci in child_indices:
                        total_siblings_arr[ci] = n

                df["parent_key"] = pd.array(np.where(parent_key_arr == -1, pd.NA, parent_key_arr), dtype="Int32")
                df["parent_type"] = pd.array(np.where(parent_key_arr == -1, pd.NA, parent_type_arr.astype("int16")), dtype="Int8")
                df["order_in_parent"] = order_in_parent_arr.astype("int16")
                df["total_siblings"] = total_siblings_arr
                df["parent_progress_at_start"] = np.clip(parent_prog_start_arr, 0.0, None).astype("float32")
                df["parent_progress_at_end"] = np.clip(parent_prog_end_arr, 0.0, None).astype("float32")
                df["parent_assign_rule"] = parent_assign_rule_arr
                df["boundary_type"] = boundary_type_arr
                df["parent_prev_key"] = pd.array(np.where(parent_prev_key_arr == -1, pd.NA, parent_prev_key_arr), dtype="Int32")
                df["parent_prev_type"] = pd.array(np.where(parent_prev_key_arr == -1, pd.NA, parent_prev_type_arr.astype("int16")), dtype="Int8")
                df["parent_next_key"] = pd.array(np.where(parent_next_key_arr == -1, pd.NA, parent_next_key_arr), dtype="Int32")
                df["parent_next_type"] = pd.array(np.where(parent_next_key_arr == -1, pd.NA, parent_next_type_arr.astype("int16")), dtype="Int8")
                df["overlap_prev_ratio"] = np.where(boundary_type_arr == 0, np.nan, overlap_prev_arr).astype("float32")
                df["overlap_next_ratio"] = np.where(boundary_type_arr == 0, np.nan, overlap_next_arr).astype("float32")

            # --- major shortcut (minor TFs only) ---
            if tf in TF_MINOR:
                _ctx_lookup = ctx_1min[["timestamp", "1h_key", "1h_type", "4h_key", "4h_type"]].copy()
                _ctx_lookup["timestamp"] = _ctx_lookup["timestamp"].astype("datetime64[s]")
                _ctx_lookup = _ctx_lookup.sort_values("timestamp").reset_index(drop=True)
                _child_ts = df["start_date"].dt.floor("min").rename("timestamp").to_frame()
                _child_ts["timestamp"] = _child_ts["timestamp"].astype("datetime64[s]")
                _child_ts["_row_idx"] = np.arange(len(df))
                _child_ts = _child_ts.sort_values("timestamp").reset_index(drop=True)
                _merged = pd.merge_asof(_child_ts, _ctx_lookup, on="timestamp", direction="nearest")
                _merged = _merged.sort_values("_row_idx").reset_index(drop=True)
                for major_tf in ["1h", "4h"]:
                    df[f"major_{major_tf}_key"] = _merged[f"{major_tf}_key"].astype("Int32")
                    df[f"major_{major_tf}_type"] = _merged[f"{major_tf}_type"].astype("Int8")

            # --- chain info (major TFs only) ---
            if tf in TF_MAJOR:
                _ctx_lookup = ctx_1min[["timestamp", "n_up_4", "combo_4"]].copy()
                _ctx_lookup["timestamp"] = _ctx_lookup["timestamp"].astype("datetime64[s]")
                _ctx_lookup = _ctx_lookup.sort_values("timestamp").reset_index(drop=True)
                _child_ts = df["start_date"].dt.floor("min").rename("timestamp").to_frame()
                _child_ts["timestamp"] = _child_ts["timestamp"].astype("datetime64[s]")
                _child_ts["_row_idx"] = np.arange(len(df))
                _child_ts = _child_ts.sort_values("timestamp").reset_index(drop=True)
                _merged = pd.merge_asof(_child_ts, _ctx_lookup, on="timestamp", direction="nearest")
                _merged = _merged.sort_values("_row_idx").reset_index(drop=True)
                df["n_up_4"] = _merged["n_up_4"].astype("Int8")
                df["combo_4"] = _merged["combo_4"].astype(str)

            # --- save enriched parquet ---
            # No local backup needed: Step 3 (detect) always regenerates the
            # pre-enrichment base file, and archive/ holds historical snapshots.
            df.to_parquet(path, index=False)
            LOGGER.info("  [pass1 saved] %s (%d cols)", tf, len(df.columns))

        # --- Pass 2: child summary (bottom-up 1min→1M) ---
        for tf in reversed(TF_ORDER):
            if tf == "1min":
                continue
            child_tf = _TF_CHILD.get(tf)
            if child_tf is None:
                continue
            child_path = self.asset_cycle_dir / f"cycles_{child_tf}.parquet"
            parent_path = self.asset_cycle_dir / f"cycles_{tf}.parquet"
            if not child_path.exists() or not parent_path.exists():
                continue
            LOGGER.info("  [pass2] %s ← %s", tf, child_tf)

            child_df = pd.read_parquet(child_path, columns=["cycle_key", "parent_key", "cycle_type", "start_date"])
            child_df = child_df.dropna(subset=["parent_key"])
            child_df["cycle_type_int"] = _type_int(child_df["cycle_type"])

            # aggregate per parent_key
            agg = (
                child_df.groupby("parent_key")
                .agg(
                    child_count=("cycle_key", "count"),
                    child_up_count=("cycle_type_int", lambda x: (x == 1).sum()),
                    child_down_count=("cycle_type_int", lambda x: (x == -1).sum()),
                )
                .reset_index()
                .rename(columns={"parent_key": "cycle_key"})
            )
            agg["cycle_key"] = agg["cycle_key"].astype("int32")

            # max_opposite_child_streak
            def _max_streak(group: pd.DataFrame, parent_type_series: pd.Series) -> int:
                # parent direction: need to join back
                return 0  # placeholder; computed below

            parent_df = pd.read_parquet(parent_path)
            if "cycle_key" not in parent_df.columns:
                id_to_key = dim.set_index("cycle_id")["cycle_key"].to_dict()
                parent_df["cycle_key"] = parent_df["cycle_id"].map(id_to_key).astype("Int32")

            # drop stale child summary columns written as 0 in pass 1
            _child_cols = ["child_count", "child_up_count", "child_down_count",
                           "opposite_child_ratio", "max_opposite_child_streak"]
            parent_df = parent_df.drop(columns=[c for c in _child_cols if c in parent_df.columns])
            parent_df = parent_df.merge(agg, on="cycle_key", how="left")
            parent_df["child_count"] = parent_df["child_count"].fillna(0).astype("Int16")
            parent_df["child_up_count"] = parent_df["child_up_count"].fillna(0).astype("Int16")
            parent_df["child_down_count"] = parent_df["child_down_count"].fillna(0).astype("Int16")

            parent_type_int = _type_int(parent_df["cycle_type"])
            parent_df["opposite_child_ratio"] = (
                np.where(
                    parent_type_int == 1,
                    parent_df["child_down_count"],
                    parent_df["child_up_count"],
                ) / parent_df["child_count"].replace(0, np.nan)
            ).astype("float32")

            # max_opposite_child_streak: iterate per parent
            streak_col = np.zeros(len(parent_df), dtype="int8")
            child_sorted = child_df.sort_values("start_date")
            for pi, row in parent_df.iterrows():
                pk = row.get("cycle_key")
                if pd.isna(pk):
                    continue
                children = child_sorted[child_sorted["parent_key"] == pk]
                if children.empty:
                    continue
                p_type = int(parent_type_int.iloc[pi])
                opposite_mask = (children["cycle_type_int"] != p_type).values
                max_streak = 0
                cur = 0
                for v in opposite_mask:
                    if v:
                        cur += 1
                        max_streak = max(max_streak, cur)
                    else:
                        cur = 0
                streak_col[pi] = min(max_streak, 127)

            parent_df["max_opposite_child_streak"] = streak_col.astype("int8")
            parent_df.to_parquet(parent_path, index=False)
            LOGGER.info("  [pass2 saved] %s", tf)

        LOGGER.info("  Phase 3 complete (%s)", _elapsed(t0))

    # ------------------------------------------------------------------
    # Phase 4 — validate
    # ------------------------------------------------------------------

    def _phase4_validate(self, dim: pd.DataFrame) -> None:
        t0 = time.time()
        report: dict[str, Any] = {"passed": [], "warnings": [], "errors": []}

        # 1. alternation check
        for tf in TF_ORDER:
            path = self.asset_cycle_dir / f"cycles_{tf}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=["cycle_type", "start_date"])
            df = df.sort_values("start_date").reset_index(drop=True)
            types = _type_int(df["cycle_type"])
            violations = int((types == types.shift(1)).sum())
            if violations == 0:
                report["passed"].append(f"alternation_{tf}")
            else:
                report["warnings"].append(f"alternation_{tf}: {violations} consecutive same-type cycles")

        # 2. parent range check (normal cycles)
        for tf in TF_ORDER:
            parent_tf = _TF_PARENT.get(tf)
            if parent_tf is None:
                continue
            path = self.asset_cycle_dir / f"cycles_{tf}.parquet"
            if not path.exists() or "boundary_type" not in pd.read_parquet(path, columns=["boundary_type"]).columns:
                continue
            cols = ["start_date", "end_date", "boundary_type", "parent_key"]
            df = pd.read_parquet(path, columns=[c for c in cols])
            df["start_date"] = _safe_to_datetime(df["start_date"])
            df["end_date"] = _safe_to_datetime(df["end_date"])
            normal = df[df["boundary_type"] == 0].dropna(subset=["parent_key"])
            if normal.empty:
                continue
            parent_dim = dim[dim["timeframe"] == parent_tf].set_index("cycle_key")[["start_date", "end_date"]]
            merged = normal.join(parent_dim.rename(columns={"start_date": "p_start", "end_date": "p_end"}), on="parent_key")
            # p_end is start of last parent candle; effective end = p_end + parent_candle_dur
            parent_candle_ns = pd.Timedelta(nanoseconds=_CANDLE_NS.get(parent_tf, 0))
            p_end_eff = merged["p_end"] + parent_candle_ns
            # child effective end = child end_date + child candle_dur
            child_candle_ns = pd.Timedelta(nanoseconds=_CANDLE_NS.get(tf, 0))
            c_end_eff = merged["end_date"] + child_candle_ns
            violations = int(((merged["start_date"] < merged["p_start"]) | (c_end_eff > p_end_eff)).sum())
            if violations == 0:
                report["passed"].append(f"parent_range_{tf}")
            else:
                report["warnings"].append(f"parent_range_{tf}: {violations} normal cycles exceed parent bounds")

        # 3. child sum check
        for tf in TF_ORDER:
            if tf == "1min":
                continue
            path = self.asset_cycle_dir / f"cycles_{tf}.parquet"
            if not path.exists():
                continue
            cols_needed = ["child_count", "child_up_count", "child_down_count"]
            df = pd.read_parquet(path, columns=cols_needed)
            mismatch = int((df["child_count"] != df["child_up_count"] + df["child_down_count"]).sum())
            if mismatch == 0:
                report["passed"].append(f"child_sum_{tf}")
            else:
                report["errors"].append(f"child_sum_{tf}: {mismatch} rows where count ≠ up+down")

        LOGGER.info("  Validation: %d passed, %d warnings, %d errors",
                    len(report["passed"]), len(report["warnings"]), len(report["errors"]))
        for msg in report["warnings"]:
            LOGGER.warning("  [WARN] %s", msg)
        for msg in report["errors"]:
            LOGGER.error("  [ERR] %s", msg)

        report_path = self.context_dir / "validation_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("  Validation report: %s (%s)", report_path.name, _elapsed(t0))

    # ------------------------------------------------------------------
    # Phase 5 — context_meta.json
    # ------------------------------------------------------------------

    def _phase5_write_meta(self, dim: pd.DataFrame) -> None:
        total = len(dim)
        meta: dict[str, Any] = {
            "version": "2.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "asset": self.asset,
            "data_range": {
                "start": str(dim["start_date"].min()),
                "end": str(dim["end_date"].max()),
            },
            "timeframes": {
                "ordered": TF_ORDER,
                "adjacency": {
                    tf: {"parent": _TF_PARENT.get(tf), "child": _TF_CHILD.get(tf)}
                    for tf in TF_ORDER
                },
                "groups": {"major": TF_MAJOR, "minor": TF_MINOR, "macro": TF_MACRO},
                "candle_minutes": {
                    "1M": 43200, "1w": 10080, "1d": 1440, "4h": 240,
                    "1h": 60, "30m": 30, "15m": 15, "5m": 5, "1min": 1,
                },
            },
            "files": {
                "dimension_table": {"path": "cycle_dim.parquet", "total_cycles": total},
                "context_tables": {
                    "1min": {"path": "timeframe_context_1min.parquet", "resolution_minutes": 1},
                    "1h": {"path": "timeframe_context_1h.parquet", "resolution_minutes": 60},
                },
            },
            "boundary_handling": {
                "default_assign_rule": "by_start",
                "types": {
                    "0": "normal - fully contained in single parent",
                    "1": "straddle - spans two parents",
                    "2": "transition_trigger - spans two parents, child opposes prev parent",
                },
            },
            "deprecated": {
                "cycle_hierarchy_map.json": {
                    "status": "read-only validation source",
                    "reason": "replaced by cycle_dim + context tables",
                    "action": "retain until new structure validation complete, then archive",
                }
            },
        }
        self.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("  context_meta.json saved")
