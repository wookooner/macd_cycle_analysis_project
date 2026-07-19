from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_PATHS


TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d", "1w"]
HIGHER_TF = {"5m": "15m", "15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}


@dataclass
class TimeframeCycleState:
    timeframe: str
    cycle_id: str | None
    cycle_type: str | None
    duration_so_far: float | None
    ppo: float | None
    ppo_hist: float | None
    dist_ma25: float | None
    parent_cycle_id: str | None
    progress: float | None
    last_closed_at: pd.Timestamp | None
    lag_minutes: float | None


@dataclass
class CycleState:
    asset: str
    timestamp: pd.Timestamp
    mode: str
    states: dict[str, TimeframeCycleState] = field(default_factory=dict)
    combo: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "timestamp": str(self.timestamp),
            "mode": self.mode,
            "combo": self.combo,
            "states": {tf: vars(state) for tf, state in self.states.items()},
        }


def asset_cycle_dir(asset: str = "btc") -> Path:
    candidate = PROJECT_PATHS.asset_cycle_dir(asset)
    return candidate if candidate.exists() else PROJECT_PATHS.cycle_structured_dir


def _read_ts(value: Any) -> pd.Timestamp | None:
    if value in (None, "", "unknown"):
        return None
    try:
        return pd.to_datetime(value, format="mixed")
    except Exception:
        return None


def _load_cycles(asset: str) -> dict[str, pd.DataFrame]:
    base = asset_cycle_dir(asset)
    frames: dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        path = base / f"cycles_{tf}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["start_date"] = pd.to_datetime(df["start_date"], format="mixed", errors="coerce")
            df["end_date"] = pd.to_datetime(df["end_date"], format="mixed", errors="coerce")
            frames[tf] = df.sort_values("start_date").reset_index(drop=True)
    return frames


def _load_market(timeframe: str) -> pd.DataFrame | None:
    path = PROJECT_PATHS.base_data_dir / f"BTCUSD_{timeframe}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=lambda col: col in {"date", "timestamp", "open_time", "close", "ppo", "ppo_hist", "ma_25"})
    ts_col = next((col for col in ("timestamp", "open_time", "date") if col in df.columns), None)
    if ts_col is None:
        return None
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    for col in [c for c in df.columns if c != "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _parent_id_from_hierarchy(hierarchy: dict[str, Any], timeframe: str, cycle_id: str | None, parent_tf: str | None) -> str | None:
    if not cycle_id or not parent_tf:
        return None
    ids = hierarchy.get(timeframe, {}).get(str(cycle_id), {}).get("parent_cycle_ids", {}).get(parent_tf, [])
    return str(ids[-1]) if ids else None


def extract_state_as_of(asset: str, timestamp: Any, mode: str = "closed_only") -> CycleState:
    ts = pd.to_datetime(timestamp, format="mixed")
    frames = _load_cycles(asset)
    hierarchy_path = asset_cycle_dir(asset) / "cycle_hierarchy_map.json"
    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8")) if hierarchy_path.exists() else {}
    states: dict[str, TimeframeCycleState] = {}

    for tf in TIMEFRAMES:
        cycles = frames.get(tf)
        market = _load_market(tf)
        if cycles is None or cycles.empty:
            continue
        eligible = cycles[cycles["start_date"] <= ts]
        containing = eligible[(eligible["end_date"] >= ts)] if not eligible.empty else pd.DataFrame()
        row = containing.iloc[-1] if not containing.empty else (eligible.iloc[-1] if not eligible.empty else None)
        if row is None:
            continue
        market_row = None
        if market is not None and not market.empty:
            if mode == "closed_only":
                market_eligible = market[market["timestamp"] <= ts]
            else:
                market_eligible = market[market["timestamp"] <= ts]
            if not market_eligible.empty:
                market_row = market_eligible.iloc[-1]
        cycle_type = str(row.get("cycle_type")).lower()
        start = _read_ts(row.get("start_date"))
        end = _read_ts(row.get("end_date"))
        duration_so_far = ((ts - start) / pd.Timedelta(minutes=1)) if start is not None else None
        duration_candles = row.get("duration_candles")
        progress = None
        if start is not None and end is not None and end > start:
            progress = max(0.0, min(1.0, float((ts - start) / (end - start))))
        ppo = float(market_row.get("ppo")) if market_row is not None and "ppo" in market_row else None
        hist = float(market_row.get("ppo_hist")) if market_row is not None and "ppo_hist" in market_row else None
        close = float(market_row.get("close")) if market_row is not None and "close" in market_row else None
        ma25 = float(market_row.get("ma_25")) if market_row is not None and "ma_25" in market_row else None
        dist_ma25 = (close / ma25 - 1.0) * 100.0 if close not in (None, 0) and ma25 not in (None, 0) else None
        last_closed = market_row.get("timestamp") if market_row is not None else end
        lag = (ts - last_closed).total_seconds() / 60.0 if isinstance(last_closed, pd.Timestamp) else None
        parent_tf = HIGHER_TF.get(tf)
        states[tf] = TimeframeCycleState(
            timeframe=tf,
            cycle_id=str(row.get("cycle_id")),
            cycle_type=cycle_type.upper() if cycle_type in {"up", "down"} else cycle_type,
            duration_so_far=float(duration_candles) if pd.notna(duration_candles) else duration_so_far,
            ppo=ppo,
            ppo_hist=hist,
            dist_ma25=dist_ma25,
            parent_cycle_id=_parent_id_from_hierarchy(hierarchy, tf, str(row.get("cycle_id")), parent_tf),
            progress=progress,
            last_closed_at=last_closed if isinstance(last_closed, pd.Timestamp) else None,
            lag_minutes=lag,
        )
    combo = "".join("U" if states[tf].cycle_type == "UP" else "D" for tf in TIMEFRAMES if tf in states)
    return CycleState(asset=asset, timestamp=ts, mode=mode, states=states, combo=combo)


def extract_current_status(asset: str = "btc") -> dict[str, Any]:
    latest_times = []
    for tf in TIMEFRAMES:
        market = _load_market(tf)
        if market is not None and not market.empty:
            latest_times.append(market["timestamp"].iloc[-1])
    ts = max(latest_times) if latest_times else pd.Timestamp.now(tz="UTC").tz_localize(None)
    state = extract_state_as_of(asset, ts, mode="closed_only")
    return {
        "asset": asset,
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S KST"),
        "data_roots": {
            "cycle_dir": str(asset_cycle_dir(asset)),
            "market_dir": str(PROJECT_PATHS.base_data_dir),
        },
        "chain": {
            "timeframes": TIMEFRAMES,
            "direction_chain_5m_to_1w": state.combo,
            "n_up": state.combo.count("U"),
            "alignment_pairs": [],
        },
        "timeframes": {
            tf: {
                "cycle": {
                    "cycle_id": tf_state.cycle_id,
                    "cycle_type": None if tf_state.cycle_type is None else tf_state.cycle_type.lower(),
                    "duration_candles": tf_state.duration_so_far,
                },
                "cycle_summary_asof": {
                    "end_ppo": tf_state.ppo,
                    "end_ppo_hist": tf_state.ppo_hist,
                    "dist_ma25": tf_state.dist_ma25,
                },
                "hierarchy": {
                    "mapped_parent_tf": HIGHER_TF.get(tf),
                    "mapped_parent_id": tf_state.parent_cycle_id,
                    "position_in_parent": {"text": "N/A"},
                },
                "latest_cycle_candle": {},
                "latest_market_candle": {},
                "unconfirmed_candles": [],
                "freshness": {
                    "latest_cycle_end": None if tf_state.last_closed_at is None else str(tf_state.last_closed_at),
                    "latest_market_time": str(ts),
                    "cycle_lag_minutes_vs_market": tf_state.lag_minutes,
                    "is_cycle_behind_market": bool(tf_state.lag_minutes and tf_state.lag_minutes > 0),
                    "unconfirmed_candle_count": 0,
                },
                "current_candle_state": {},
                "cycle_candles": [],
            }
            for tf, tf_state in state.states.items()
        },
    }
