"""
trading_bot/cycle/state_extractor.py
=====================================
Builds the current market_state JSON from parquet cycle data.

In addition to the lightweight timeframe summary used by the existing bot,
this extractor now attaches the full 1h-rooted cycle chain:
1h -> containing 4h -> containing 1d -> containing 1w.
Each cycle payload includes candle_data and cycle_features so the later
signal engine / AI layer can reason over the raw indicators and candles.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("bot.state_extractor")
DEFAULT_TIMEFRAME_ORDER = ["1w", "1d", "4h", "1h"]
ANALYSIS_COLUMNS = [
    "taker_buy_base",
    "volume_delta",
    "cvd",
    "cvd_rolling",
    "ppo",
    "ppo_signal",
    "ppo_hist",
    "delta",
    "ma_7",
    "ma_25",
    "ma_99",
    "oi",
    "oi_usd",
    "funding_rate",
]


class CycleStateExtractor:
    def __init__(self, parquet_dir: Path, hierarchy_map_path: Path, timeframes: list[str] = None):
        self.parquet_dir = parquet_dir
        self.hierarchy_map_path = hierarchy_map_path
        self.timeframes = self._normalize_timeframes(timeframes or DEFAULT_TIMEFRAME_ORDER)
        self._cycle_frame_cache: dict[str, tuple[int, pd.DataFrame]] = {}
        self._hierarchy_cache: tuple[int, dict[str, Any]] | None = None
        self._base_frame_cache: dict[str, tuple[int, pd.DataFrame]] = {}

    def extract_current_state(
        self,
        current_position: Optional[dict] = None,
        recent_signals: Optional[list] = None,
    ) -> dict:
        cycle_frames = self._load_cycle_frames()
        base_frames = self._load_base_frames()
        hierarchy = self._load_hierarchy_map()

        tf_states = {}
        latest_price = None

        for tf in self.timeframes:
            state = self._extract_tf_state(tf, cycle_frames, base_frames)
            if state:
                tf_states[tf] = state
                if tf == "1h":
                    latest_price = state.pop("_latest_close", None)
            if state and "_latest_close" in state:
                state.pop("_latest_close")

        cycle_context = self._extract_cycle_context(cycle_frames, base_frames, hierarchy)
        if latest_price is None:
            latest_price = cycle_context.get("cycles", {}).get("1h", {}).get("latest_close")

        chain_info = self._evaluate_chain(tf_states, cycle_context)

        market_state = {
            "timestamp": self._utc_now_iso(),
            "price": latest_price,
            "timeframes": tf_states,
            "chain": chain_info,
            "cycle_context": cycle_context,
            "current_position": current_position,
            "recent_signals": recent_signals or [],
        }

        logger.info(
            "State: price=%s, combo=%s, n_up=%s",
            latest_price,
            chain_info.get("combo"),
            chain_info.get("n_up"),
        )
        return market_state

    def _load_cycle_frames(self) -> dict[str, pd.DataFrame]:
        frames = {}
        for tf in self.timeframes:
            parquet_path = self.parquet_dir / f"cycles_{tf}.parquet"
            if not parquet_path.exists():
                logger.warning("Not found: %s", parquet_path)
                self._cycle_frame_cache.pop(tf, None)
                continue
            try:
                mtime_ns = parquet_path.stat().st_mtime_ns
                cached = self._cycle_frame_cache.get(tf)
                if cached and cached[0] == mtime_ns:
                    frames[tf] = cached[1]
                    continue

                df = pd.read_parquet(parquet_path)
                if df.empty:
                    self._cycle_frame_cache.pop(tf, None)
                    continue
                normalized = df.sort_values("start_date").reset_index(drop=True)
                self._cycle_frame_cache[tf] = (mtime_ns, normalized)
                frames[tf] = normalized
            except Exception as exc:
                logger.error("Failed to read %s: %s", parquet_path.name, exc, exc_info=True)
        return frames

    def _load_hierarchy_map(self) -> dict[str, Any]:
        if not self.hierarchy_map_path.exists():
            logger.warning("Hierarchy map not found: %s", self.hierarchy_map_path)
            self._hierarchy_cache = None
            return {}
        try:
            mtime_ns = self.hierarchy_map_path.stat().st_mtime_ns
            if self._hierarchy_cache and self._hierarchy_cache[0] == mtime_ns:
                return self._hierarchy_cache[1]

            hierarchy = json.loads(self.hierarchy_map_path.read_text(encoding="utf-8"))
            self._hierarchy_cache = (mtime_ns, hierarchy)
            return hierarchy
        except Exception as exc:
            logger.error("Failed to load hierarchy map: %s", exc, exc_info=True)
            return {}

    def _load_base_frames(self) -> dict[str, pd.DataFrame]:
        frames = {}
        base_dir = self.parquet_dir.parent.parent / "base_data"
        for tf in self.timeframes:
            csv_path = base_dir / f"BTCUSD_{tf}.csv"
            if not csv_path.exists():
                self._base_frame_cache.pop(tf, None)
                continue
            try:
                mtime_ns = csv_path.stat().st_mtime_ns
                cached = self._base_frame_cache.get(tf)
                if cached and cached[0] == mtime_ns:
                    frames[tf] = cached[1]
                    continue

                df = pd.read_csv(csv_path)
                if df.empty:
                    self._base_frame_cache.pop(tf, None)
                    continue
                normalized = df.sort_values("unix").reset_index(drop=True)
                self._base_frame_cache[tf] = (mtime_ns, normalized)
                frames[tf] = normalized
            except Exception as exc:
                logger.error("Failed to read base csv for %s: %s", tf, exc, exc_info=True)
        return frames

    def _extract_tf_state(
        self,
        tf: str,
        cycle_frames: dict[str, pd.DataFrame],
        base_frames: dict[str, pd.DataFrame],
    ) -> Optional[dict]:
        df = cycle_frames.get(tf)
        if df is None or df.empty:
            return None

        try:
            last = df.iloc[-1]
            features = self._coerce_dict(last.get("cycle_features"))
            start_f = self._coerce_dict(features.get("start"))
            shape_f = self._coerce_dict(features.get("shape"))

            latest_close = None
            candle_data = self._normalize_candle_data(last.get("candle_data"))
            if candle_data:
                latest_close = candle_data[-1].get("close")

            latest_analysis = self._get_latest_analysis_snapshot(tf, base_frames)

            duration = int(last.get("duration_candles", 0) or 0)
            position_pct = self._estimate_position(df, duration)

            return {
                "cycle_id": str(last.get("cycle_id", "")),
                "cycle_type": str(last.get("cycle_type", "")).upper(),
                "duration": duration,
                "start_date": str(last.get("start_date", "")),
                "end_date": str(last.get("end_date", "")),
                "start_rsi": start_f.get("rsi"),
                "start_macd": start_f.get("macd"),
                "start_hist": start_f.get("hist"),
                "start_price": start_f.get("price"),
                "noise_count": shape_f.get("noise_count", 0),
                "position_pct": round(position_pct, 3),
                "is_current": self._is_current_cycle(last, tf),
                "analysis_snapshot": latest_analysis,
                "_latest_close": latest_close,
            }
        except Exception as exc:
            logger.error("Error extracting %s state: %s", tf, exc, exc_info=True)
            return None

    def _extract_cycle_context(
        self,
        cycle_frames: dict[str, pd.DataFrame],
        base_frames: dict[str, pd.DataFrame],
        hierarchy: dict[str, Any],
    ) -> dict[str, Any]:
        current_1h = self._get_current_cycle_row("1h", cycle_frames)
        if current_1h is None:
            return {"anchor_timeframe": "1h", "cycle_ids": {}, "cycles": {}, "hierarchy": {}}

        cycle_ids = self._resolve_cycle_chain_ids(str(current_1h.get("cycle_id", "")), hierarchy)
        cycles = {}
        for tf in ["1h", "4h", "1d", "1w"]:
            cycle_id = cycle_ids.get(tf)
            row = self._find_cycle_row(tf, cycle_id, cycle_frames)
            if row is not None:
                cycles[tf] = self._build_cycle_payload(tf, row, base_frames, hierarchy)

        return {
            "anchor_timeframe": "1h",
            "cycle_ids": cycle_ids,
            "cycles": cycles,
            "hierarchy": self._build_hierarchy_summary(cycle_ids, hierarchy),
        }

    def _get_current_cycle_row(self, tf: str, cycle_frames: dict[str, pd.DataFrame]):
        df = cycle_frames.get(tf)
        if df is None or df.empty:
            return None
        return df.iloc[-1]

    def _find_cycle_row(self, tf: str, cycle_id: Optional[str], cycle_frames: dict[str, pd.DataFrame]):
        if not cycle_id:
            return None
        df = cycle_frames.get(tf)
        if df is None or df.empty or "cycle_id" not in df.columns:
            return None
        matched = df[df["cycle_id"] == cycle_id]
        if matched.empty:
            return None
        return matched.iloc[-1]

    def _resolve_cycle_chain_ids(self, cycle_1h_id: str, hierarchy: dict[str, Any]) -> dict[str, Optional[str]]:
        ids = {"1h": cycle_1h_id, "4h": None, "1d": None, "1w": None}
        node_1h = self._get_hierarchy_node(hierarchy, "1h", cycle_1h_id)
        parents = self._coerce_dict(node_1h.get("parent_cycle_ids"))

        ids["4h"] = self._first_or_none(parents.get("4h"))
        ids["1d"] = self._first_or_none(parents.get("1d"))
        ids["1w"] = self._first_or_none(parents.get("1w"))

        if not ids["1w"] and ids["1d"]:
            node_1d = self._get_hierarchy_node(hierarchy, "1d", ids["1d"])
            ids["1w"] = self._first_or_none(self._coerce_dict(node_1d.get("parent_cycle_ids")).get("1w"))

        if not ids["4h"] and ids["1d"]:
            node_1d = self._get_hierarchy_node(hierarchy, "1d", ids["1d"])
            child_4h_ids = self._coerce_dict(node_1d.get("child_cycle_ids")).get("4h", [])
            for cycle_4h_id in child_4h_ids:
                node_4h = self._get_hierarchy_node(hierarchy, "4h", cycle_4h_id)
                child_1h_ids = self._coerce_dict(node_4h.get("child_cycle_ids")).get("1h", [])
                if cycle_1h_id in child_1h_ids:
                    ids["4h"] = cycle_4h_id
                    break

        if not ids["1d"] and ids["4h"]:
            node_4h = self._get_hierarchy_node(hierarchy, "4h", ids["4h"])
            ids["1d"] = self._first_or_none(self._coerce_dict(node_4h.get("parent_cycle_ids")).get("1d"))

        if not ids["1w"] and ids["1d"]:
            node_1d = self._get_hierarchy_node(hierarchy, "1d", ids["1d"])
            ids["1w"] = self._first_or_none(self._coerce_dict(node_1d.get("parent_cycle_ids")).get("1w"))

        return ids

    def _build_cycle_payload(self, tf: str, row, base_frames: dict[str, pd.DataFrame], hierarchy: dict[str, Any]) -> dict[str, Any]:
        cycle_id = str(row.get("cycle_id", ""))
        node = self._get_hierarchy_node(hierarchy, tf, cycle_id)
        candle_data = self._normalize_candle_data(row.get("candle_data"))
        cycle_features = self._coerce_dict(row.get("cycle_features"))
        latest_candle = candle_data[-1] if candle_data else {}
        parent_ids = self._coerce_dict(node.get("parent_cycle_ids"))
        child_ids = self._coerce_dict(node.get("child_cycle_ids"))
        analysis_rows = self._get_cycle_analysis_rows(tf, row, base_frames)

        return {
            "timeframe": tf,
            "cycle_id": cycle_id,
            "cycle_type": str(row.get("cycle_type", "")).upper(),
            "start_date": str(row.get("start_date", "")),
            "end_date": str(row.get("end_date", "")),
            "duration_candles": int(row.get("duration_candles", 0) or 0),
            "is_current": self._is_current_cycle(row, tf),
            "latest_close": latest_candle.get("close"),
            "latest_timestamp": latest_candle.get("timestamp"),
            "latest_candle": latest_candle,
            "parent_cycle_ids": parent_ids,
            "child_cycle_ids": child_ids,
            "cycle_features": cycle_features,
            "candle_count": len(candle_data),
            "candle_data": candle_data,
            "analysis_fields": [column for column in ANALYSIS_COLUMNS if analysis_rows and column in analysis_rows[0]],
            "analysis_latest": analysis_rows[-1] if analysis_rows else {},
            "analysis_rows": analysis_rows,
        }

    def _build_hierarchy_summary(self, cycle_ids: dict[str, Optional[str]], hierarchy: dict[str, Any]) -> dict[str, Any]:
        summary = {}
        relationships = [("1h", "4h"), ("4h", "1d"), ("1d", "1w")]

        for child_tf, parent_tf in relationships:
            child_id = cycle_ids.get(child_tf)
            parent_id = cycle_ids.get(parent_tf)
            parent_node = self._get_hierarchy_node(hierarchy, parent_tf, parent_id)
            child_node = self._get_hierarchy_node(hierarchy, child_tf, child_id)
            sibling_ids = self._coerce_dict(parent_node.get("child_cycle_ids")).get(child_tf, [])
            summary[f"{child_tf}_in_{parent_tf}"] = {
                "child_cycle_id": child_id,
                "parent_cycle_id": parent_id,
                "child_index_in_parent": self._index_in_list(sibling_ids, child_id),
                "child_count_in_parent": len(sibling_ids),
                "child_sibling_ids": sibling_ids,
                "child_parent_ids": self._coerce_dict(child_node.get("parent_cycle_ids")),
            }

        return summary

    def _estimate_position(self, df: pd.DataFrame, current_dur: int) -> float:
        if current_dur <= 0:
            return 0.0
        last_type = df.iloc[-1].get("cycle_type", "")
        same = df[df["cycle_type"] == last_type]
        avg = same.tail(20)["duration_candles"].mean() if len(same) >= 5 else same["duration_candles"].mean()
        return min(current_dur / max(avg, 1), 1.5)

    def _is_current_cycle(self, row, tf: str) -> bool:
        try:
            end = self._to_utc_timestamp(row.get("end_date", ""))
            if end is None:
                return False
            gap_h = {"1h": 2, "4h": 8, "1d": 48, "1w": 336}.get(tf, 48)
            return (pd.Timestamp.now(tz=timezone.utc) - end).total_seconds() / 3600 < gap_h
        except Exception:
            return False

    def _evaluate_chain(self, tf_states: dict, cycle_context: dict[str, Any]) -> dict:
        parts = []
        n_up = 0
        for tf in self.timeframes:
            st = tf_states.get(tf)
            if st and st.get("cycle_type"):
                ct = st["cycle_type"].upper()
                parts.append("U" if ct == "UP" else "D")
                if ct == "UP":
                    n_up += 1
            else:
                parts.append("?")

        alignment = self._is_aligned(tf_states, "4h", "1h")
        return {
            "combo": "".join(parts),
            "n_up": n_up,
            "alignment_4h_1h": alignment,
            "cycle_ids": cycle_context.get("cycle_ids", {}),
        }

    def _is_aligned(self, tf_states: dict[str, Any], left_tf: str, right_tf: str) -> bool:
        left = self._cycle_direction(tf_states.get(left_tf))
        right = self._cycle_direction(tf_states.get(right_tf))
        return bool(left and right and left == right)

    def _cycle_direction(self, tf_state: Optional[dict[str, Any]]) -> Optional[str]:
        if not tf_state:
            return None
        cycle_type = str(tf_state.get("cycle_type", "")).upper()
        if cycle_type == "UP":
            return "U"
        if cycle_type == "DOWN":
            return "D"
        return None

    def _normalize_candle_data(self, candle_data: Any) -> list[dict[str, Any]]:
        if candle_data is None:
            return []
        if isinstance(candle_data, str):
            try:
                candle_data = json.loads(candle_data)
            except json.JSONDecodeError:
                return []
        if not hasattr(candle_data, "__len__"):
            return []

        normalized = []
        for candle in list(candle_data):
            if isinstance(candle, dict):
                normalized.append(dict(candle))
            elif hasattr(candle, "asDict"):
                normalized.append(dict(candle.asDict()))
            elif hasattr(candle, "items"):
                normalized.append(dict(candle.items()))
        return normalized

    def _coerce_dict(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        if hasattr(value, "asDict"):
            parsed = value.asDict()
            return parsed if isinstance(parsed, dict) else {}
        return dict(value) if hasattr(value, "items") else {}

    def _get_hierarchy_node(self, hierarchy: dict[str, Any], tf: str, cycle_id: Optional[str]) -> dict[str, Any]:
        if not cycle_id:
            return {}
        return self._coerce_dict(self._coerce_dict(hierarchy.get(tf)).get(cycle_id))

    def _first_or_none(self, value: Any) -> Optional[str]:
        if isinstance(value, list) and value:
            return value[0]
        if isinstance(value, str) and value:
            return value
        return None

    def _index_in_list(self, items: list[Any], target: Optional[str]) -> Optional[int]:
        if not target or target not in items:
            return None
        return items.index(target)

    def _get_latest_analysis_snapshot(self, tf: str, base_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
        df = base_frames.get(tf)
        if df is None or df.empty:
            return {}
        latest = df.iloc[-1].to_dict()
        return self._pick_analysis_columns(latest)

    def _get_cycle_analysis_rows(self, tf: str, row, base_frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        df = base_frames.get(tf)
        if df is None or df.empty:
            return []

        start_ts = self._to_utc_timestamp(row.get("start_date", ""))
        end_ts = self._to_utc_timestamp(row.get("end_date", ""))
        if start_ts is None or end_ts is None:
            return []

        try:
            dates = pd.to_datetime(df["date"], utc=True, errors="coerce")
            matched = df[(dates >= start_ts) & (dates <= end_ts)]
            if matched.empty:
                return []
            rows = []
            for _, matched_row in matched.iterrows():
                payload = {"date": matched_row.get("date"), "unix": matched_row.get("unix")}
                payload.update(self._pick_analysis_columns(matched_row.to_dict()))
                rows.append(payload)
            return rows
        except Exception as exc:
            logger.error("Failed to build analysis rows for %s: %s", tf, exc, exc_info=True)
            return []

    def _pick_analysis_columns(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = {}
        for column in ANALYSIS_COLUMNS:
            if column not in row:
                continue
            value = row.get(column)
            if pd.isna(value):
                continue
            payload[column] = value
        return payload

    def _normalize_timeframes(self, timeframes: list[str]) -> list[str]:
        unique = []
        seen = set()
        for tf in timeframes:
            if tf in seen:
                continue
            seen.add(tf)
            unique.append(tf)
        order_lookup = {tf: index for index, tf in enumerate(DEFAULT_TIMEFRAME_ORDER)}
        return sorted(unique, key=lambda tf: order_lookup.get(tf, len(order_lookup)))

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _to_utc_timestamp(self, value: Any) -> Optional[pd.Timestamp]:
        if value in (None, ""):
            return None
        try:
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                return timestamp.tz_localize("UTC")
            return timestamp.tz_convert("UTC")
        except Exception:
            return None
