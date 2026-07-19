from __future__ import annotations

import json
import sys
import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import PROJECT_PATHS


ASSET = "btc"
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d", "1w"]
HIGHER_TF = {"5m": "15m", "15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}
KST_TZ = "Asia/Seoul"
MAX_CYCLE_CANDLES_IN_REPORT = 80


def output_dir() -> Path:
    path = PROJECT_PATHS.outputs_root / "analysis_results" / "current_cycle_status"
    path.mkdir(parents=True, exist_ok=True)
    return path


def asset_cycle_dir(asset: str = ASSET) -> Path:
    candidate = PROJECT_PATHS.asset_cycle_dir(asset)
    return candidate if candidate.exists() else PROJECT_PATHS.cycle_structured_dir


def to_builtin(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def to_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, "", "unknown"):
        return None
    try:
        return pd.to_datetime(value, format="mixed")
    except Exception:
        return None


def to_kst(value: Any) -> str:
    ts = to_timestamp(value)
    if ts is None:
        return "unknown"
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.tz_convert(KST_TZ).strftime("%Y-%m-%d %H:%M:%S KST")


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return "NA"
    return f"{number:,.{digits}f}"


def pct(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "NA"
    return f"{number:+.2f}%"


def direction_sign(cycle_type: Any) -> int:
    return 1 if str(cycle_type).lower() == "up" else -1


def direction_label(value: Any) -> str:
    text = str(value).lower()
    if text == "up":
        return "UP"
    if text == "down":
        return "DOWN"
    return str(value)


def load_cycles(asset: str = ASSET) -> dict[str, pd.DataFrame]:
    base = asset_cycle_dir(asset)
    frames: dict[str, pd.DataFrame] = {}
    for timeframe in TIMEFRAMES:
        path = base / f"cycles_{timeframe}.parquet"
        if path.exists():
            frames[timeframe] = pd.read_parquet(path)
    return frames


def load_hierarchy(asset: str = ASSET) -> dict[str, Any]:
    path = asset_cycle_dir(asset) / "cycle_hierarchy_map.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_latest_market_state(timeframe: str) -> dict[str, Any] | None:
    path = PROJECT_PATHS.base_data_dir / f"BTCUSD_{timeframe}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    current = df.iloc[-1].to_dict()
    previous = df.iloc[-2].to_dict() if len(df) > 1 else {}
    current = {str(key): to_builtin(val) for key, val in current.items()}
    previous = {str(key): to_builtin(val) for key, val in previous.items()}

    for column in [
        "close",
        "macd_hist",
        "ppo_hist",
        "rsi",
        "stoch_rsi_k",
        "stoch_rsi_d",
        "cvd",
        "oi",
        "oi_contracts",
        "oi_notional",
    ]:
        cur = to_float(current.get(column))
        prev = to_float(previous.get(column))
        if cur is not None and prev is not None:
            current[f"{column}_delta_vs_prev"] = cur - prev

    close = to_float(current.get("close"))
    prev_close = to_float(previous.get("close"))
    if close is not None and prev_close not in (None, 0):
        current["close_return_pct_vs_prev"] = (close / prev_close - 1.0) * 100.0

    hist_delta = to_float(current.get("macd_hist_delta_vs_prev"))
    ppo_hist_delta = to_float(current.get("ppo_hist_delta_vs_prev"))
    current["macd_hist_raw_direction"] = int(np.sign(hist_delta)) if hist_delta is not None else None
    current["ppo_hist_raw_direction"] = int(np.sign(ppo_hist_delta)) if ppo_hist_delta is not None else None
    current["_source_csv"] = str(path)
    return current


def load_market_candles_after(timeframe: str, after_time: Any, limit: int = 20) -> list[dict[str, Any]]:
    path = PROJECT_PATHS.base_data_dir / f"BTCUSD_{timeframe}.csv"
    after_ts = to_timestamp(after_time)
    if not path.exists() or after_ts is None:
        return []

    df = pd.read_csv(path)
    if df.empty or "date" not in df.columns:
        return []

    dates = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    rows = df[dates > after_ts].tail(limit).copy()
    return [{str(key): to_builtin(value) for key, value in row.items()} for row in rows.to_dict("records")]


def candle_list(cycle_row: pd.Series | None) -> list[dict[str, Any]]:
    if cycle_row is None:
        return []
    raw = cycle_row.get("candle_data", [])
    if raw is None:
        return []
    return [to_builtin(item) for item in list(raw) if isinstance(item, Mapping)]


def flatten_dict(data: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(data, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(flatten_dict(value, full_key))
        else:
            result[full_key] = to_builtin(value)
    return result


def cycle_row_to_dict(row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {}
    result: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if key in {"candle_data", "cycle_features"}:
            continue
        result[str(key)] = to_builtin(value)
    result["cycle_features_flat"] = flatten_dict(row.get("cycle_features", {}))
    return result


def summarize_cycle_candles(candles: list[dict[str, Any]], cycle_type: str) -> dict[str, Any]:
    if not candles:
        return {}

    sign = direction_sign(cycle_type)
    first = candles[0]
    last = candles[-1]
    closes = pd.Series([to_float(candle.get("close")) for candle in candles], dtype="float64")
    macd_hist = pd.Series([to_float(candle.get("macd_hist")) for candle in candles], dtype="float64")
    ppo_hist = pd.Series([to_float(candle.get("ppo_hist")) for candle in candles], dtype="float64")
    cvd = pd.Series([to_float(candle.get("cvd")) for candle in candles], dtype="float64")
    volume = pd.Series([to_float(candle.get("volume")) for candle in candles], dtype="float64")

    macd_delta = macd_hist.diff()
    ppo_delta = ppo_hist.diff()
    macd_noise = ((np.sign(macd_delta.fillna(0.0)) * sign) < 0).astype(int)
    ppo_noise = ((np.sign(ppo_delta.fillna(0.0)) * sign) < 0).astype(int)

    start_close = to_float(first.get("close"))
    end_close = to_float(last.get("close"))
    raw_return = None
    if start_close not in (None, 0) and end_close is not None:
        raw_return = (end_close / start_close - 1.0) * 100.0

    return {
        "candle_count": len(candles),
        "start_time": first.get("timestamp", first.get("date")),
        "end_time": last.get("timestamp", last.get("date")),
        "start_close": start_close,
        "end_close": end_close,
        "raw_return_pct": raw_return,
        "cycle_direction_return_pct": None if raw_return is None else raw_return * sign,
        "start_macd_hist": to_float(first.get("macd_hist")),
        "end_macd_hist": to_float(last.get("macd_hist")),
        "macd_hist_change": to_float(last.get("macd_hist"), 0.0) - to_float(first.get("macd_hist"), 0.0),
        "start_ppo": to_float(first.get("ppo")),
        "end_ppo": to_float(last.get("ppo")),
        "start_ppo_hist": to_float(first.get("ppo_hist")),
        "end_ppo_hist": to_float(last.get("ppo_hist")),
        "ppo_hist_change": to_float(last.get("ppo_hist"), 0.0) - to_float(first.get("ppo_hist"), 0.0),
        "macd_noise_count_asof": int(macd_noise.sum()),
        "ppo_noise_count_asof": int(ppo_noise.sum()),
        "macd_noise_ratio_asof": float(macd_noise.mean()),
        "ppo_noise_ratio_asof": float(ppo_noise.mean()),
        "abs_ppo_hist_area_asof": float(ppo_hist.abs().sum(skipna=True)),
        "signed_ppo_hist_area_asof": float(ppo_hist.sum(skipna=True)),
        "cvd_change_asof": float(cvd.iloc[-1] - cvd.iloc[0]) if cvd.notna().sum() >= 2 else None,
        "volume_sum_asof": float(volume.sum(skipna=True)),
    }


def latest_row(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None or df.empty:
        return None
    return df.iloc[-1]


def row_by_cycle_id(df: pd.DataFrame | None, cycle_id: str | None) -> pd.Series | None:
    if df is None or df.empty or not cycle_id or "cycle_id" not in df.columns:
        return None
    matched = df[df["cycle_id"].astype(str) == str(cycle_id)]
    return matched.iloc[0] if not matched.empty else None


def parent_ids_for(hierarchy: dict[str, Any], timeframe: str, cycle_id: str) -> dict[str, list[str]]:
    return hierarchy.get(timeframe, {}).get(cycle_id, {}).get("parent_cycle_ids", {}) or {}


def child_ids_for(hierarchy: dict[str, Any], timeframe: str, cycle_id: str) -> dict[str, list[str]]:
    return hierarchy.get(timeframe, {}).get(cycle_id, {}).get("child_cycle_ids", {}) or {}


def chronological_position(
    frames: dict[str, pd.DataFrame],
    hierarchy: dict[str, Any],
    timeframe: str,
    parent_tf: str,
    parent_id: str | None,
    cycle_id: str | None,
) -> dict[str, Any]:
    df = frames.get(timeframe)
    if df is None or df.empty or not parent_id or not cycle_id:
        return {"order": None, "total": 0, "text": "N/A"}

    rows: list[tuple[pd.Timestamp, str]] = []
    for _, row in df.iterrows():
        cid = str(row.get("cycle_id"))
        parents = parent_ids_for(hierarchy, timeframe, cid).get(parent_tf, [])
        if str(parent_id) in [str(item) for item in parents]:
            ts = to_timestamp(row.get("start_date")) or pd.Timestamp.max
            rows.append((ts, cid))

    rows.sort(key=lambda item: item[0])
    ordered = [cid for _, cid in rows]
    if cycle_id not in ordered:
        return {"order": None, "total": len(ordered), "text": f"N/A/{len(ordered)}"}
    order = ordered.index(cycle_id) + 1
    return {"order": order, "total": len(ordered), "text": f"{order}/{len(ordered)}"}


def containing_cycle_by_time(parent_df: pd.DataFrame | None, timestamp: Any) -> str | None:
    ts = to_timestamp(timestamp)
    if parent_df is None or parent_df.empty or ts is None:
        return None
    starts = pd.to_datetime(parent_df["start_date"], format="mixed", errors="coerce")
    ends = pd.to_datetime(parent_df["end_date"], format="mixed", errors="coerce")
    matched = parent_df[(starts <= ts) & (ends >= ts)]
    if not matched.empty:
        return str(matched.iloc[-1]["cycle_id"])
    before = parent_df[starts <= ts]
    if not before.empty:
        return str(before.iloc[-1]["cycle_id"])
    return str(parent_df.iloc[-1]["cycle_id"])


def build_timeframe_state(
    timeframe: str,
    frames: dict[str, pd.DataFrame],
    hierarchy: dict[str, Any],
    latest_market: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    df = frames.get(timeframe)
    row = latest_row(df)
    cycle_id = None if row is None else str(row.get("cycle_id"))
    cycle_type = None if row is None else str(row.get("cycle_type"))
    candles = candle_list(row)
    candle_summary = summarize_cycle_candles(candles, cycle_type or "unknown")
    market = latest_market.get(timeframe)

    latest_cycle_end = to_timestamp(row.get("end_date")) if row is not None else None
    latest_market_time = to_timestamp(market.get("date")) if market else None
    lag_minutes = None
    if latest_cycle_end is not None and latest_market_time is not None:
        lag_minutes = (latest_market_time - latest_cycle_end).total_seconds() / 60.0
    unconfirmed_candles = load_market_candles_after(timeframe, latest_cycle_end)

    parents = parent_ids_for(hierarchy, timeframe, cycle_id or "")
    children = child_ids_for(hierarchy, timeframe, cycle_id or "")
    higher_tf = HIGHER_TF.get(timeframe)
    mapped_parent_id = None
    if higher_tf:
        ids = parents.get(higher_tf, [])
        mapped_parent_id = str(ids[-1]) if ids else containing_cycle_by_time(
            frames.get(higher_tf),
            row.get("end_date") if row is not None else (market or {}).get("date"),
        )

    pos = chronological_position(frames, hierarchy, timeframe, higher_tf, mapped_parent_id, cycle_id) if higher_tf else {
        "order": None,
        "total": 0,
        "text": "top",
    }

    latest_candle = candles[-1] if candles else {}
    prev_candle = candles[-2] if len(candles) > 1 else {}
    cycle_sign = direction_sign(cycle_type)
    macd_delta = None
    ppo_delta = None
    if latest_candle and prev_candle:
        macd_delta = to_float(latest_candle.get("macd_hist"), 0.0) - to_float(prev_candle.get("macd_hist"), 0.0)
        ppo_delta = to_float(latest_candle.get("ppo_hist"), 0.0) - to_float(prev_candle.get("ppo_hist"), 0.0)

    state = {
        "timeframe": timeframe,
        "cycle": cycle_row_to_dict(row),
        "cycle_summary_asof": candle_summary,
        "hierarchy": {
            "mapped_parent_tf": higher_tf,
            "mapped_parent_id": mapped_parent_id,
            "parent_cycle_ids": parents,
            "child_cycle_ids": children,
            "position_in_parent": pos,
        },
        "latest_cycle_candle": latest_candle,
        "latest_market_candle": market,
        "unconfirmed_candles": unconfirmed_candles,
        "freshness": {
            "latest_cycle_end": None if latest_cycle_end is None else str(latest_cycle_end),
            "latest_market_time": None if latest_market_time is None else str(latest_market_time),
            "cycle_lag_minutes_vs_market": lag_minutes,
            "is_cycle_behind_market": bool(lag_minutes is not None and lag_minutes > 0),
            "unconfirmed_candle_count": len(unconfirmed_candles),
        },
        "current_candle_state": {
            "macd_hist_delta_in_cycle": macd_delta,
            "ppo_hist_delta_in_cycle": ppo_delta,
            "macd_noise_against_cycle": None if macd_delta is None else bool(np.sign(macd_delta) * cycle_sign < 0),
            "ppo_noise_against_cycle": None if ppo_delta is None else bool(np.sign(ppo_delta) * cycle_sign < 0),
        },
        "cycle_candles": candles[-MAX_CYCLE_CANDLES_IN_REPORT:],
    }
    return to_builtin(state)


def build_current_status(asset: str = ASSET) -> dict[str, Any]:
    from src.cycles.state_extractor import extract_current_status

    return extract_current_status(asset)


def build_current_status_legacy(asset: str = ASSET) -> dict[str, Any]:
    frames = load_cycles(asset)
    hierarchy = load_hierarchy(asset)
    latest_market = {timeframe: load_latest_market_state(timeframe) for timeframe in TIMEFRAMES}
    states = {
        timeframe: build_timeframe_state(timeframe, frames, hierarchy, latest_market)
        for timeframe in TIMEFRAMES
        if timeframe in frames
    }

    chain = "".join(
        "U" if str(states[tf]["cycle"].get("cycle_type")).lower() == "up" else "D"
        for tf in TIMEFRAMES
        if tf in states
    )
    n_up = chain.count("U")

    alignments: list[dict[str, Any]] = []
    for low_tf, high_tf in HIGHER_TF.items():
        if low_tf not in states or high_tf not in states:
            continue
        low_type = states[low_tf]["cycle"].get("cycle_type")
        high_type = states[high_tf]["cycle"].get("cycle_type")
        alignments.append({
            "pair": f"{low_tf}->{high_tf}",
            "low_cycle_type": low_type,
            "high_cycle_type": high_type,
            "same_direction": low_type == high_type,
            "mapped_parent_id": states[low_tf]["hierarchy"].get("mapped_parent_id"),
        })

    return {
        "asset": asset,
        "generated_at": pd.Timestamp.now(tz=KST_TZ).strftime("%Y-%m-%d %H:%M:%S KST"),
        "data_roots": {
            "cycle_dir": str(asset_cycle_dir(asset)),
            "market_dir": str(PROJECT_PATHS.base_data_dir),
            "output_dir": str(output_dir()),
        },
        "chain": {
            "timeframes": TIMEFRAMES,
            "direction_chain_5m_to_1w": chain,
            "n_up": n_up,
            "alignment_pairs": alignments,
        },
        "timeframes": states,
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def compact_indicator_lines(data: Mapping[str, Any], skip: set[str] | None = None) -> list[str]:
    skip = skip or set()
    items = []
    for key, value in data.items():
        if key in skip or key.startswith("_"):
            continue
        if isinstance(value, Mapping) or isinstance(value, list):
            continue
        items.append(f"- `{key}`: `{value}`")
    return items


def compact_indicator_text(data: Mapping[str, Any], skip: set[str] | None = None) -> str:
    skip = skip or set()
    parts = []
    for key, value in data.items():
        if key in skip or key.startswith("_"):
            continue
        if isinstance(value, Mapping) or isinstance(value, list):
            continue
        if value is None:
            rendered = "NA"
        elif isinstance(value, float):
            rendered = "nan" if pd.isna(value) else f"{value:.6g}"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return " | ".join(parts)


def candle_brief(candle: Mapping[str, Any]) -> str:
    return (
        f"{candle.get('date', candle.get('timestamp', 'unknown'))} "
        f"O={fmt(candle.get('open'), 2)} H={fmt(candle.get('high'), 2)} "
        f"L={fmt(candle.get('low'), 2)} C={fmt(candle.get('close'), 2)} "
        f"MACDh={fmt(candle.get('macd_hist'), 6)} PPOh={fmt(candle.get('ppo_hist'), 6)} "
        f"RSI={fmt(candle.get('rsi'), 2)} StochRSI_K={fmt(candle.get('stoch_rsi_k'), 2)} "
        f"StochRSI_D={fmt(candle.get('stoch_rsi_d'), 2)} CVD={fmt(candle.get('cvd'), 2)} "
        f"OI={fmt(candle.get('oi', candle.get('oi_contracts')), 2)}"
    )


def build_markdown(status: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Current Cycle Status")
    lines.append("")
    lines.append(f"- Generated: `{status['generated_at']}`")
    lines.append(f"- Asset: `{status['asset']}`")
    lines.append(f"- Direction chain 5m->1w: `{status['chain']['direction_chain_5m_to_1w']}`")
    lines.append(f"- n_up: `{status['chain']['n_up']}`")
    lines.append("")

    freshness_rows = []
    cycle_rows = []
    for tf in TIMEFRAMES:
        state = status["timeframes"].get(tf)
        if not state:
            continue
        cycle = state["cycle"]
        summary = state["cycle_summary_asof"]
        fresh = state["freshness"]
        hierarchy = state["hierarchy"]
        freshness_rows.append({
            "tf": tf,
            "cycle_id": cycle.get("cycle_id"),
            "type": direction_label(cycle.get("cycle_type")),
            "cycle_end": fresh.get("latest_cycle_end"),
            "market_time": fresh.get("latest_market_time"),
            "lag_min": "" if fresh.get("cycle_lag_minutes_vs_market") is None else f"{fresh.get('cycle_lag_minutes_vs_market'):.1f}",
            "behind": fresh.get("is_cycle_behind_market"),
        })
        cycle_rows.append({
            "tf": tf,
            "cycle_id": cycle.get("cycle_id"),
            "type": direction_label(cycle.get("cycle_type")),
            "dur": cycle.get("duration_candles"),
            "pos": hierarchy.get("position_in_parent", {}).get("text"),
            "parent": hierarchy.get("mapped_parent_id"),
            "ret_dir": pct(summary.get("cycle_direction_return_pct")),
            "ppo_noise": fmt(summary.get("ppo_noise_ratio_asof"), 2),
            "last_ppo_h": fmt(summary.get("end_ppo_hist"), 4),
        })

    lines.append("## Freshness")
    lines.append(markdown_table(freshness_rows, ["tf", "cycle_id", "type", "cycle_end", "market_time", "lag_min", "behind"]))
    lines.append("")
    lines.append("## Cycle Chain")
    lines.append(markdown_table(cycle_rows, ["tf", "cycle_id", "type", "dur", "pos", "parent", "ret_dir", "ppo_noise", "last_ppo_h"]))
    lines.append("")

    lines.append("## Alignment")
    align_rows = []
    for item in status["chain"]["alignment_pairs"]:
        align_rows.append({
            "pair": item["pair"],
            "low": direction_label(item["low_cycle_type"]),
            "high": direction_label(item["high_cycle_type"]),
            "same": item["same_direction"],
            "mapped_parent": item["mapped_parent_id"],
        })
    lines.append(markdown_table(align_rows, ["pair", "low", "high", "same", "mapped_parent"]))
    lines.append("")

    for tf in TIMEFRAMES:
        state = status["timeframes"].get(tf)
        if not state:
            continue
        cycle = state["cycle"]
        summary = state["cycle_summary_asof"]
        current_state = state["current_candle_state"]
        market = state.get("latest_market_candle") or {}
        latest_cycle_candle = state.get("latest_cycle_candle") or {}
        unconfirmed = state.get("unconfirmed_candles") or []

        lines.append(f"## {tf} Current State")
        lines.append(f"- Cycle: `{cycle.get('cycle_id')}` `{direction_label(cycle.get('cycle_type'))}`")
        lines.append(f"- Period: `{to_kst(cycle.get('start_date'))}` -> `{to_kst(cycle.get('end_date'))}`")
        lines.append(f"- Parent: `{state['hierarchy'].get('mapped_parent_tf')}` `{state['hierarchy'].get('mapped_parent_id')}`")
        lines.append(f"- Return in cycle direction: `{pct(summary.get('cycle_direction_return_pct'))}`")
        lines.append(f"- PPO noise ratio as-of: `{fmt(summary.get('ppo_noise_ratio_asof'), 2)}`")
        lines.append(f"- Latest cycle candle PPO noise: `{current_state.get('ppo_noise_against_cycle')}`")
        lines.append(f"- Latest cycle candle MACD noise: `{current_state.get('macd_noise_against_cycle')}`")
        lines.append(f"- Unconfirmed candles after cycle end: `{len(unconfirmed)}`")
        lines.append("")
        lines.append("### Processed Cycle Summary")
        lines.extend(compact_indicator_lines(summary))
        lines.append("")
        lines.append("### Latest Processed Cycle Candle")
        lines.extend(compact_indicator_lines(latest_cycle_candle))
        lines.append("")
        lines.append("### Latest Market CSV Candle")
        lines.extend(compact_indicator_lines(market))
        lines.append("")
        lines.append("### Unconfirmed Market Candles")
        if unconfirmed:
            for candle in unconfirmed:
                lines.append(f"- `{candle_brief(candle)}`")
        else:
            lines.append("- None")
        lines.append("")
        lines.append("### Cycle Features")
        lines.extend(compact_indicator_lines(cycle.get("cycle_features_flat", {})))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def print_console_summary(status: dict[str, Any]) -> None:
    print("=" * 120)
    print("CURRENT CYCLE STATUS")
    print("=" * 120)
    print(f"Generated: {status['generated_at']}")
    print(f"Direction chain 5m->1w: {status['chain']['direction_chain_5m_to_1w']}  n_up={status['chain']['n_up']}")
    print()

    rows = []
    for tf in TIMEFRAMES:
        state = status["timeframes"].get(tf)
        if not state:
            continue
        cycle = state["cycle"]
        summary = state["cycle_summary_asof"]
        fresh = state["freshness"]
        current_state = state["current_candle_state"]
        rows.append({
            "tf": tf,
            "cycle": cycle.get("cycle_id"),
            "dir": direction_label(cycle.get("cycle_type")),
            "dur": cycle.get("duration_candles"),
            "parent": state["hierarchy"].get("mapped_parent_id"),
            "pos": state["hierarchy"].get("position_in_parent", {}).get("text"),
            "ret_dir": pct(summary.get("cycle_direction_return_pct")),
            "ppo_h": fmt(summary.get("end_ppo_hist"), 4),
            "ppo_noise": current_state.get("ppo_noise_against_cycle"),
            "lag_min": fresh.get("cycle_lag_minutes_vs_market"),
            "unconfirmed": fresh.get("unconfirmed_candle_count"),
        })

    for row in rows:
        lag_text = "NA" if row["lag_min"] is None else f"{row['lag_min']:.0f}m"
        print(
            f"{row['tf']:>3s} {row['cycle']:<16s} {row['dir']:<4s} "
            f"dur={row['dur']:<4} parent={str(row['parent']):<16s} pos={str(row['pos']):<8s} "
            f"ret_dir={row['ret_dir']:<8s} ppo_h={row['ppo_h']:<10s} "
            f"latest_ppo_noise={row['ppo_noise']} lag={lag_text} unconfirmed={row['unconfirmed']}"
        )

    print()
    print("Alignment:")
    for item in status["chain"]["alignment_pairs"]:
        print(
            f"- {item['pair']}: {direction_label(item['low_cycle_type'])} -> "
            f"{direction_label(item['high_cycle_type'])}, same={item['same_direction']}, "
            f"mapped_parent={item['mapped_parent_id']}"
        )
    print()
    print("=" * 120)
    print("FULL CURRENT DETAILS")
    print("=" * 120)

    for tf in TIMEFRAMES:
        state = status["timeframes"].get(tf)
        if not state:
            continue

        cycle = state["cycle"]
        summary = state["cycle_summary_asof"]
        current_state = state["current_candle_state"]
        latest_cycle_candle = state.get("latest_cycle_candle") or {}
        latest_market_candle = state.get("latest_market_candle") or {}
        unconfirmed = state.get("unconfirmed_candles") or []

        print()
        print("-" * 120)
        print(f"{tf.upper()} | {cycle.get('cycle_id')} | {direction_label(cycle.get('cycle_type'))}")
        print("-" * 120)
        print(
            f"period={cycle.get('start_date')} -> {cycle.get('end_date')} | "
            f"duration={cycle.get('duration_candles')} | "
            f"parent={state['hierarchy'].get('mapped_parent_tf')}:{state['hierarchy'].get('mapped_parent_id')} | "
            f"position={state['hierarchy'].get('position_in_parent', {}).get('text')}"
        )
        print(
            f"cycle_return_dir={pct(summary.get('cycle_direction_return_pct'))} | "
            f"raw_return={pct(summary.get('raw_return_pct'))} | "
            f"ppo_noise_ratio={fmt(summary.get('ppo_noise_ratio_asof'), 2)} | "
            f"macd_noise_ratio={fmt(summary.get('macd_noise_ratio_asof'), 2)}"
        )
        print(
            f"latest_cycle_noise: PPO={current_state.get('ppo_noise_against_cycle')} "
            f"MACD={current_state.get('macd_noise_against_cycle')} | "
            f"unconfirmed_count={len(unconfirmed)}"
        )

        print("cycle_summary_all:")
        print("  " + compact_indicator_text(summary))

        print("latest_confirmed_cycle_candle_all:")
        print("  " + compact_indicator_text(latest_cycle_candle))

        print("unconfirmed_candles_after_cycle_end:")
        if unconfirmed:
            for index, candle in enumerate(unconfirmed, start=1):
                print(f"  [{index}] {candle_brief(candle)}")
                print("      " + compact_indicator_text(candle))
        else:
            print("  none")

        print("latest_market_csv_candle_all:")
        print("  " + compact_indicator_text(latest_market_candle))

        features = cycle.get("cycle_features_flat", {})
        if features:
            print("cycle_features_all:")
            print("  " + compact_indicator_text(features))

    print("=" * 120)


def write_csv_outputs(status: dict[str, Any], out_dir: Path) -> dict[str, str]:
    summary_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    candle_rows: list[dict[str, Any]] = []

    for timeframe in TIMEFRAMES:
        state = status["timeframes"].get(timeframe)
        if not state:
            continue

        cycle = state.get("cycle", {})
        row = {
            "timeframe": timeframe,
            **{f"cycle_{key}": value for key, value in cycle.items() if key != "cycle_features_flat"},
            **{f"feature_{key}": value for key, value in cycle.get("cycle_features_flat", {}).items()},
            **{f"summary_{key}": value for key, value in state.get("cycle_summary_asof", {}).items()},
            **{f"freshness_{key}": value for key, value in state.get("freshness", {}).items()},
            **{f"state_{key}": value for key, value in state.get("current_candle_state", {}).items()},
            "hierarchy_mapped_parent_tf": state.get("hierarchy", {}).get("mapped_parent_tf"),
            "hierarchy_mapped_parent_id": state.get("hierarchy", {}).get("mapped_parent_id"),
            "hierarchy_position_in_parent": state.get("hierarchy", {}).get("position_in_parent", {}).get("text"),
        }
        summary_rows.append(to_builtin(row))

        market = state.get("latest_market_candle") or {}
        if market:
            market_rows.append({"timeframe": timeframe, **to_builtin(market)})

        for idx, candle in enumerate(state.get("cycle_candles", []), start=1):
            candle_rows.append({
                "timeframe": timeframe,
                "cycle_id": cycle.get("cycle_id"),
                "cycle_type": cycle.get("cycle_type"),
                "candle_index_in_report": idx,
                **to_builtin(candle),
            })
        for idx, candle in enumerate(state.get("unconfirmed_candles", []), start=1):
            candle_rows.append({
                "timeframe": timeframe,
                "cycle_id": cycle.get("cycle_id"),
                "cycle_type": cycle.get("cycle_type"),
                "candle_index_in_report": idx,
                "is_unconfirmed_after_cycle_end": True,
                **to_builtin(candle),
            })

    paths = {
        "summary_csv": str(out_dir / "current_cycle_summary.csv"),
        "latest_market_csv": str(out_dir / "latest_market_candles.csv"),
        "cycle_candles_csv": str(out_dir / "current_cycle_candles.csv"),
    }
    pd.DataFrame(summary_rows).to_csv(paths["summary_csv"], index=False, encoding="utf-8-sig")
    pd.DataFrame(market_rows).to_csv(paths["latest_market_csv"], index=False, encoding="utf-8-sig")
    pd.DataFrame(candle_rows).to_csv(paths["cycle_candles_csv"], index=False, encoding="utf-8-sig")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print current multi-timeframe cycle and candle status.")
    parser.add_argument("--save", action="store_true", help="Also save JSON/Markdown/CSV outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = build_current_status()
    print_console_summary(status)

    if args.save:
        out_dir = output_dir()
        json_path = out_dir / "current_cycle_status.json"
        md_path = out_dir / "current_cycle_status.md"
        csv_paths = write_csv_outputs(status, out_dir)
        json_path.write_text(json.dumps(to_builtin(status), ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(build_markdown(status), encoding="utf-8")
        print(f"Saved JSON: {json_path}")
        print(f"Saved report: {md_path}")
        for label, path in csv_paths.items():
            print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
