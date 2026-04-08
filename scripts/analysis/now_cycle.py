"""
now_cycle.py
?꾩옱 ?ъ씠???곹깭 + MACD ?덉뒪?좉렇???꾪솚 媛寃?遺꾩꽍

?섏젙 ?대젰 (v2 2026-03-23):
  - ?꾨줈?앺듃 猷⑦듃 ?먮룞 ?먯깋 (寃쎈줈 ?섎뱶肄붾뵫 ?쒓굅)
  - 4h 遺紐?誘몃ℓ????理쒖떊 4h ?ъ씠?대줈 fallback
  - end 吏?쒕? candle_data?먯꽌 吏곸젒 異붿텧 (cycle_features.end 鍮꾪솢?????
  - 媛???꾪봽?덉엫蹂?MACD ?덉뒪?좉렇???꾪솚 媛寃?怨꾩궛 異붽?
  - n_up 泥댁씤 ?쒖떆 異붽?
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_PATHS


KST_TZ = "Asia/Seoul"
SHORT_CYCLE_CONTEXT_THRESHOLD = 5


# ?? ?꾨줈?앺듃 猷⑦듃 ?먯깋 ?????????????????????????????????????????????????????

def _find_project_root() -> Path:
    return PROJECT_PATHS.project_root


# ?? ?곗씠??濡쒕뱶 ????????????????????????????????????????????????????????????

def load_cycle_data(base_path: str, timeframe: str) -> pd.DataFrame | None:
    file_path = os.path.join(base_path, f"cycles_{timeframe}.parquet")
    if not os.path.exists(file_path):
        return None
    return pd.read_parquet(file_path)


def load_latest_csv_state(csv_path: str) -> dict[str, Any] | None:
    """理쒖떊 CSV 2媛??됱쓣 ?쎌뼱 ?꾩옱媛?吏곸쟾媛?湲곗? ?곹깭瑜?留뚮뱺??"""
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    rows = df.tail(2).to_dict("records")
    current = rows[-1]
    previous = rows[-2] if len(rows) > 1 else None

    def _num(row: dict[str, Any] | None, key: str) -> float | None:
        if not row:
            return None
        value = row.get(key)
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    current_hist = _num(current, "macd_hist")
    previous_hist = _num(previous, "macd_hist")
    raw_direction = 0
    if current_hist is not None and previous_hist is not None:
        if current_hist > previous_hist:
            raw_direction = 1
        elif current_hist < previous_hist:
            raw_direction = -1

    return {
        "date": str(current.get("date", current.get("timestamp", "unknown"))),
        "close": _num(current, "close"),
        "open": _num(current, "open"),
        "high": _num(current, "high"),
        "low": _num(current, "low"),
        "volume": _num(current, "volume"),
        "macd": _num(current, "macd"),
        "macd_signal": _num(current, "macd_signal"),
        "macd_hist": current_hist,
        "rsi": _num(current, "rsi"),
        "ppo": _num(current, "ppo"),
        "ppo_signal": _num(current, "ppo_signal"),
        "ppo_hist": _num(current, "ppo_hist"),
        "cvd": _num(current, "cvd"),
        "cvd_rolling": _num(current, "cvd_rolling"),
        "volume_delta": _num(current, "volume_delta"),
        "oi": _num(current, "oi"),
        "oi_contracts": _num(current, "oi_contracts"),
        "oi_contracts_change": _num(current, "oi_contracts_change"),
        "oi_contracts_change_pct": _num(current, "oi_contracts_change_pct"),
        "oi_usd": _num(current, "oi_usd"),
        "oi_notional": _num(current, "oi_notional"),
        "oi_notional_change": _num(current, "oi_notional_change"),
        "oi_notional_change_pct": _num(current, "oi_notional_change_pct"),
        "oi_change": _num(current, "oi_change"),
        "oi_change_pct": _num(current, "oi_change_pct"),
        "funding_rate": _num(current, "funding_rate"),
        "prev_hist": previous_hist,
        "raw_direction": raw_direction,
    }


def to_kst_string(value: Any) -> str:
    if value in (None, "", "unknown"):
        return "unknown"
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.tz_convert(KST_TZ).strftime("%Y-%m-%d %H:%M:%S KST")
    except Exception:
        return str(value)


def assess_live_cycle_state(
    cycle_type: str,
    live_state: dict[str, Any] | None,
    flip_state: dict[str, float] | None,
) -> dict[str, Any]:
    """?꾩옱 CSV 理쒖떊媛?湲곗??쇰줈 ?ъ씠???좎?/?쏀솕/?꾪솚 媛?μ꽦???붿빟?쒕떎."""
    if not live_state:
        return {
            "status": "라이브 CSV 없음",
            "raw_direction_label": "N/A",
            "change_risk": "알 수 없음",
        }

    cycle_dir = 1 if str(cycle_type).lower() == "up" else -1
    raw_direction = int(live_state.get("raw_direction") or 0)
    current_hist = live_state.get("macd_hist")
    current_close = live_state.get("close")

    if raw_direction == cycle_dir:
        status = "사이클 방향 유지"
    elif raw_direction == 0:
        status = "모멘텀 보합"
    else:
        status = "반대 방향 모멘텀 강화"

    if current_hist is None:
        risk = "hist 값 없음"
    elif cycle_dir == 1 and current_hist < 0:
        risk = "하락 전환 리스크 높음"
    elif cycle_dir == -1 and current_hist > 0:
        risk = "상승 전환 리스크 높음"
    elif raw_direction == -cycle_dir:
        risk = "반대 방향 진행 중"
    else:
        risk = "현재 방향 유지 가능성 높음"

    if flip_state and current_close is not None:
        flip_price = flip_state.get("flip_price")
        if flip_price is not None:
            if cycle_dir == 1 and current_close <= flip_price:
                risk = "flip 가격 근접 또는 하향 이탈"
            elif cycle_dir == -1 and current_close >= flip_price:
                risk = "flip 가격 근접 또는 상향 이탈"

    raw_direction_label = {1: "상승(+1)", -1: "하락(-1)", 0: "보합(0)"}.get(raw_direction, "N/A")
    return {
        "status": status,
        "raw_direction_label": raw_direction_label,
        "change_risk": risk,
    }


# ?? Feature 異붿텧 ???????????????????????????????????????????????????????????

def extract_features(cycle_row: pd.Series) -> dict[str, Any]:
    """?ъ씠??row?먯꽌 ?듭떖 吏??異붿텧. end 媛믪? candle_data?먯꽌 吏곸젒 媛?몄샂."""
    features = cycle_row.get("cycle_features", {})
    if not isinstance(features, dict):
        try:
            features = dict(features)
        except (TypeError, ValueError):
            features = {}

    def safe_get(*keys: str, default: float = 0.0) -> float:
        curr = features
        for key in keys:
            if isinstance(curr, dict):
                curr = curr.get(key)
            else:
                return default
        try:
            if curr is not None and not pd.isna(curr):
                return float(curr)
        except Exception:
            pass
        return default

    start_macd = safe_get("start", "macd")
    start_hist = safe_get("start", "hist")
    start_rsi = safe_get("start", "rsi")
    start_price = safe_get("start", "price")
    price_change_pct = safe_get("change", "price_pct")

    # end 媛믪? candle_data 留덉?留?罹붾뱾?먯꽌 異붿텧 (cycle_features.end??鍮꾪솢??
    end_macd = end_hist = end_rsi = end_price = 0.0
    start_cvd = end_cvd = 0.0
    start_cvd_rolling = end_cvd_rolling = 0.0
    start_oi = end_oi = 0.0
    start_oi_contracts = end_oi_contracts = 0.0
    start_oi_notional = end_oi_notional = 0.0
    candle_data = cycle_row.get("candle_data", [])
    if candle_data is not None and len(candle_data) > 0:
        first = candle_data[0] if isinstance(candle_data[0], dict) else {}
        last = candle_data[-1] if isinstance(candle_data[-1], dict) else {}
        start_cvd = float(first.get("cvd", 0) or 0)
        start_cvd_rolling = float(first.get("cvd_rolling", 0) or 0)
        start_oi = float(first.get("oi", 0) or 0)
        start_oi_contracts = float(first.get("oi_contracts", first.get("oi", 0)) or 0)
        start_oi_notional = float(first.get("oi_notional", first.get("oi_usd", 0)) or 0)
        end_macd = float(last.get("macd", 0) or 0)
        end_hist = float(last.get("macd_hist", 0) or 0)
        end_rsi = float(last.get("rsi", 0) or 0)
        end_price = float(last.get("close", 0) or 0)
        end_cvd = float(last.get("cvd", 0) or 0)
        end_cvd_rolling = float(last.get("cvd_rolling", 0) or 0)
        end_oi = float(last.get("oi", 0) or 0)
        end_oi_contracts = float(last.get("oi_contracts", last.get("oi", 0)) or 0)
        end_oi_notional = float(last.get("oi_notional", last.get("oi_usd", 0)) or 0)

    return {
        "cycle_type": cycle_row.get("cycle_type", "unknown"),
        "start_date": cycle_row.get("start_date", "unknown"),
        "end_date": cycle_row.get("end_date", "unknown"),
        "duration_candles": cycle_row.get("duration_candles", 0),
        "price_change_pct": price_change_pct,
        "start_price": start_price,
        "end_price": end_price,
        "macd_change": end_macd - start_macd,
        "hist_change": end_hist - start_hist,
        "rsi_change": end_rsi - start_rsi,
        "start_macd": start_macd,
        "start_hist": start_hist,
        "start_rsi": start_rsi,
        "end_macd": end_macd,
        "end_hist": end_hist,
        "end_rsi": end_rsi,
        "start_cvd": start_cvd,
        "end_cvd": end_cvd,
        "cvd_change": end_cvd - start_cvd,
        "start_cvd_rolling": start_cvd_rolling,
        "end_cvd_rolling": end_cvd_rolling,
        "cvd_rolling_change": end_cvd_rolling - start_cvd_rolling,
        "start_oi": start_oi,
        "end_oi": end_oi,
        "oi_change": end_oi - start_oi,
        "start_oi_contracts": start_oi_contracts,
        "end_oi_contracts": end_oi_contracts,
        "oi_contracts_change": end_oi_contracts - start_oi_contracts,
        "start_oi_notional": start_oi_notional,
        "end_oi_notional": end_oi_notional,
        "oi_notional_change": end_oi_notional - start_oi_notional,
        "core_count": int(safe_get("shape", "core_count")),
        "noise_count": int(safe_get("shape", "noise_count")),
    }


# ?? ?덉뒪?좉렇???꾪솚 媛寃?怨꾩궛 ??????????????????????????????????????????????

def calc_histogram_flip_price(csv_path: str) -> dict[str, float] | None:
    """?ㅼ쓬 罹붾뱾?먯꽌 MACD hist媛 遺??諛섏쟾?섎뒗 醫낃?(close)瑜???궛.

    ?먮━:
      EMA_fast_new = close * k_f + EMA_fast_prev * (1-k_f)
      EMA_slow_new = close * k_s + EMA_slow_prev * (1-k_s)
      MACD_new = EMA_fast_new - EMA_slow_new
      Signal_new = MACD_new * k_sig + Signal_prev * (1-k_sig)
      Hist_new = MACD_new - Signal_new
              = (1-k_sig) * (MACD_new - Signal_prev)

      Hist_new = 0  ?? MACD_new = Signal_prev
      ??close_flip = [Signal_prev - EMA_fast_prev*(1-k_f) + EMA_slow_prev*(1-k_s)] / (k_f - k_s)
    """
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if "close" not in df.columns or len(df) < 30:
        return None

    close = df["close"].values.astype(float)
    k_f = 2.0 / 13.0   # fast EMA (12)
    k_s = 2.0 / 27.0   # slow EMA (26)
    k_sig = 2.0 / 10.0  # signal EMA (9)

    # EMA ?꾩껜 怨꾩궛
    ef = np.empty(len(close))
    es = np.empty(len(close))
    ef[0] = es[0] = close[0]
    for i in range(1, len(close)):
        ef[i] = close[i] * k_f + ef[i - 1] * (1 - k_f)
        es[i] = close[i] * k_s + es[i - 1] * (1 - k_s)

    macd = ef - es
    sig = np.empty(len(macd))
    sig[0] = macd[0]
    for i in range(1, len(macd)):
        sig[i] = macd[i] * k_sig + sig[i - 1] * (1 - k_sig)

    hist = macd - sig

    # ?꾩옱 ?곹깭
    cur_close = close[-1]
    cur_hist = hist[-1]
    cur_macd = macd[-1]
    cur_signal = sig[-1]

    # 전환 가격 계산
    denom = k_f - k_s
    if abs(denom) < 1e-15:
        return None

    flip_close = (sig[-1] - ef[-1] * (1 - k_f) + es[-1] * (1 - k_s)) / denom
    pct_to_flip = (flip_close - cur_close) / cur_close * 100

    # ?hist=0 媛寃?(hist 蹂?붾웾??0??媛寃?= 媛먯냽?믨???寃쎄퀎)
    # hist_new - hist_prev = 0
    # (1-k_sig)*(MACD_new - sig_prev) - hist_prev = 0
    # MACD_new = sig_prev + hist_prev / (1-k_sig)
    target_macd_decel = sig[-1] + hist[-1] / (1 - k_sig)
    decel_close = (target_macd_decel - ef[-1] * (1 - k_f) + es[-1] * (1 - k_s)) / denom
    pct_to_decel = (decel_close - cur_close) / cur_close * 100

    return {
        "current_close": cur_close,
        "current_hist": cur_hist,
        "current_macd": cur_macd,
        "current_signal": cur_signal,
        "flip_price": flip_close,
        "flip_pct": pct_to_flip,
        "decel_price": decel_close,
        "decel_pct": pct_to_decel,
    }


# ?? ?좏떥 ???????????????????????????????????????????????????????????????????

def format_position(order: int | None, total: int) -> str:
    if not order or not total:
        return "N/A"
    return f"{order}/{total}"


def latest_parent_id(node: dict[str, Any], parent_tf: str) -> str | None:
    parent_ids = node.get("parent_cycle_ids", {}).get(parent_tf, [])
    return parent_ids[-1] if parent_ids else None


def latest_parent_id_from_cycle(
    hierarchy_map: dict, timeframe: str, cycle_id: str, parent_tf: str
) -> str | None:
    node = hierarchy_map.get(timeframe, {}).get(cycle_id, {})
    return latest_parent_id(node, parent_tf)


def candle_order(value: Any) -> tuple[int, str]:
    text = str(value or "")
    if text.startswith("candle_"):
        try:
            return int(text.split("_")[-1]), text
        except ValueError:
            pass
    return 10**18, text


def chronological_position(
    df: pd.DataFrame | None,
    hierarchy_map: dict,
    timeframe: str,
    parent_tf: str,
    parent_id: str | None,
    cycle_id: str | None,
) -> tuple[int | None, int]:
    if df is None or df.empty or not parent_id or not cycle_id:
        return None, 0

    rows = []
    for _, row in df.iterrows():
        cid = row.get("cycle_id")
        if not cid:
            continue
        if latest_parent_id_from_cycle(hierarchy_map, timeframe, cid, parent_tf) != parent_id:
            continue
        rows.append((candle_order(row.get("start_date")), candle_order(row.get("end_date")), str(cid)))

    if not rows:
        return None, 0

    rows.sort()
    ordered = [r[2] for r in rows]
    if cycle_id not in ordered:
        return None, len(ordered)
    return ordered.index(cycle_id) + 1, len(ordered)


def row_by_cycle_id(df: pd.DataFrame | None, cycle_id: str | None) -> pd.Series | None:
    if df is None or df.empty or not cycle_id:
        return None
    matched = df[df["cycle_id"] == cycle_id]
    return matched.iloc[0] if not matched.empty else None


def resolve_parent_or_latest(
    hierarchy_map: dict,
    current_map_node: dict,
    parent_tf: str,
    df_parent: pd.DataFrame | None,
) -> tuple[str | None, bool]:
    """留ㅽ븨??遺紐⑤? 李얜릺, ?놁쑝硫??대떦 ??꾪봽?덉엫??理쒖떊 ?ъ씠?댁쓣 fallback?쇰줈 ?ъ슜."""
    pid = latest_parent_id(current_map_node, parent_tf)
    if pid:
        return pid, False
    # fallback: 理쒖떊 ?ъ씠??    if df_parent is not None and not df_parent.empty:
        return df_parent.iloc[-1]["cycle_id"], True
    return None, True


# ?? 異쒕젰 ?щ㎎ ??????????????????????????????????????????????????????????????

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def cycle_candles(cycle_row: pd.Series | None) -> list[dict[str, Any]]:
    if cycle_row is None:
        return []
    candle_data = cycle_row.get("candle_data", [])
    if candle_data is None:
        return []
    return [candle for candle in candle_data if isinstance(candle, dict)]


def previous_cycle_row(df: pd.DataFrame | None, cycle_id: str | None) -> pd.Series | None:
    if df is None or df.empty or not cycle_id:
        return None
    matched = df.index[df["cycle_id"] == cycle_id].tolist()
    if not matched:
        return None
    idx = matched[0]
    if idx <= 0:
        return None
    return df.iloc[idx - 1]


def build_cycle_candle_context(
    df: pd.DataFrame | None,
    cycle_row: pd.Series | None,
    short_threshold: int = SHORT_CYCLE_CONTEXT_THRESHOLD,
) -> tuple[list[tuple[str, dict[str, Any]]], str | None]:
    current_candles = cycle_candles(cycle_row)
    rows: list[tuple[str, dict[str, Any]]] = [("CUR", candle) for candle in current_candles]
    prev_cycle_id = None

    if cycle_row is None:
        return rows, prev_cycle_id

    if len(current_candles) < short_threshold:
        prev_row = previous_cycle_row(df, cycle_row.get("cycle_id"))
        prev_cycle_id = None if prev_row is None else str(prev_row.get("cycle_id"))
        prev_candles = cycle_candles(prev_row)
        rows = [("PREV", candle) for candle in prev_candles] + rows

    return rows, prev_cycle_id


def print_cycle_candle_context(
    tf: str,
    cycle_id: str,
    df: pd.DataFrame | None,
    cycle_row: pd.Series | None,
    short_threshold: int = SHORT_CYCLE_CONTEXT_THRESHOLD,
) -> None:
    rows, prev_cycle_id = build_cycle_candle_context(df, cycle_row, short_threshold=short_threshold)
    if not rows:
        print(f"  [CANDLES {tf.upper()}] candle_data 없음")
        print()
        return

    current_len = len(cycle_candles(cycle_row))
    context_note = f"현재 사이클 {cycle_id}"
    if prev_cycle_id:
        context_note += f" + 이전 사이클 {prev_cycle_id}"

    print(f"  [CANDLES {tf.upper()}] {context_note}")
    print(
        f"    short<{short_threshold}: 이전 사이클 포함 | "
        f"현재 {current_len}캔들 | 총 {len(rows)}행"
    )
    print("    tag  time(KST)            close      hist    rsi        cvd      cvdr         oi       fund")
    for tag, candle in rows:
        ts = to_kst_string(candle.get("date", candle.get("timestamp", "unknown")))
        time_label = ts.replace(" KST", "")
        close = _safe_float(candle.get("close"))
        hist = _safe_float(candle.get("macd_hist"))
        rsi = _safe_float(candle.get("rsi"))
        cvd = _safe_float(candle.get("cvd"))
        cvdr = _safe_float(candle.get("cvd_rolling"))
        oi = _safe_float(candle.get("oi", candle.get("oi_contracts")))
        fund = _safe_float(candle.get("funding_rate"))
        print(
            f"    {tag:<4s} {time_label:<19s} "
            f"{close:>10.2f} {hist:>9.2f} {rsi:>6.2f} "
            f"{cvd:>10.2f} {cvdr:>10.2f} {oi:>10.2f} {fund:>10.6f}"
        )
    print()


def print_cycle_block(
    title: str,
    cycle_id: str,
    info: dict[str, Any],
    pos_text: str,
    extra_text: str = "",
    fallback: bool = False,
) -> None:
    fb_mark = " (fallback)" if fallback else ""
    direction = str(info["cycle_type"]).upper()
    start_date_kst = to_kst_string(info["start_date"])
    end_date_kst = to_kst_string(info["end_date"])
    period_kst = f"{start_date_kst} ~ {end_date_kst}"
    print(f"┌─ [{title}] {cycle_id}{fb_mark}")
    print(f"│  방향: {direction:4s}  캔들: {info['duration_candles']}  위치: {pos_text}{extra_text}")
    print(f"│  기간: {period_kst}")
    print(f"│  가격: {info['start_price']:>10.2f} -> {info['end_price']:>10.2f}  ({info['price_change_pct']:+.2f}%)")
    print(f"│  RSI : {info['start_rsi']:>8.2f} -> {info['end_rsi']:>8.2f}  ({info['rsi_change']:+.2f})")
    print(f"│  MACD: {info['start_macd']:>10.2f} -> {info['end_macd']:>10.2f}  ({info['macd_change']:+.2f})")
    print(f"│  HIST: {info['start_hist']:>10.2f} -> {info['end_hist']:>10.2f}  ({info['hist_change']:+.2f})")
    print(f"│  CVD : {info['start_cvd']:>10.2f} -> {info['end_cvd']:>10.2f}  ({info['cvd_change']:+.2f})")
    print(
        f"│  CVDR: {info['start_cvd_rolling']:>10.2f} -> {info['end_cvd_rolling']:>10.2f}  "
        f"({info['cvd_rolling_change']:+.2f})"
    )
    print(f"│  OI  : {info['start_oi']:>10.2f} -> {info['end_oi']:>10.2f}  ({info['oi_change']:+.2f})")
    print(f"└  노이즈: {info['noise_count']}  코어: {info['core_count']}")
    print()


def print_flip_block(tf: str, flip: dict[str, float] | None) -> None:
    if not flip:
        print(f"  {tf:>3s}: 데이터 없음")
        return

    cur = flip["current_close"]
    hist = flip["current_hist"]
    fp = flip["flip_price"]
    fpct = flip["flip_pct"]
    dp = flip["decel_price"]
    dpct = flip["decel_pct"]

    direction = "상승" if hist > 0 else "하락"
    flip_dir = "하락" if hist > 0 else "상승"

    print(f"  {tf:>3s}  hist: {hist:>+12.2f} {direction}  |  전환(=0): ${fp:>10.2f} ({fpct:>+6.2f}% {flip_dir} 필요)  |  감속(Δ=0): ${dp:>10.2f} ({dpct:>+6.2f}%)")


# ?? 硫붿씤 遺꾩꽍 ??????????????????????????????????????????????????????????????

def print_live_block(tf: str, live_state: dict[str, Any] | None, live_assessment: dict[str, Any]) -> None:
    if not live_state:
        print(f"  [LIVE {tf.upper()}] CSV 최신값 없음")
        print()
        return

    close = live_state.get("close") or 0.0
    macd = live_state.get("macd") or 0.0
    hist = live_state.get("macd_hist") or 0.0
    rsi = live_state.get("rsi") or 0.0
    print(f"  [LIVE {tf.upper()}] {to_kst_string(live_state.get('date', 'unknown'))}")
    print(f"    현재값: close {close:>10.2f}  macd {macd:>10.2f}  hist {hist:>10.2f}  rsi {rsi:>8.2f}")
    print(
        f"    현재판단: {live_assessment.get('status', 'N/A')}  "
        f"| raw_dir {live_assessment.get('raw_direction_label', 'N/A')}  "
        f"| 전환리스크 {live_assessment.get('change_risk', 'N/A')}"
    )
    print()


def print_latest_candle_block(tf: str, live_state: dict[str, Any] | None) -> None:
    if not live_state:
        return

    open_ = live_state.get("open") or 0.0
    high = live_state.get("high") or 0.0
    low = live_state.get("low") or 0.0
    close = live_state.get("close") or 0.0
    volume = live_state.get("volume") or 0.0
    macd = live_state.get("macd") or 0.0
    hist = live_state.get("macd_hist") or 0.0
    rsi = live_state.get("rsi") or 0.0
    ppo = live_state.get("ppo") or 0.0
    ppo_hist = live_state.get("ppo_hist") or 0.0
    cvd = live_state.get("cvd") or 0.0
    cvd_rolling = live_state.get("cvd_rolling") or 0.0
    volume_delta = live_state.get("volume_delta") or 0.0
    oi = live_state.get("oi") or 0.0
    oi_contracts = live_state.get("oi_contracts") or oi
    oi_contracts_change = live_state.get("oi_contracts_change")
    oi_contracts_change_pct = live_state.get("oi_contracts_change_pct")
    oi_usd = live_state.get("oi_usd") or 0.0
    oi_notional = live_state.get("oi_notional") or oi_usd
    oi_notional_change = live_state.get("oi_notional_change")
    oi_notional_change_pct = live_state.get("oi_notional_change_pct")
    funding_rate = live_state.get("funding_rate") or 0.0

    print(f"    [LATEST {tf.upper()} CANDLE]")
    print(f"      OHLCV : O {open_:>10.2f}  H {high:>10.2f}  L {low:>10.2f}  C {close:>10.2f}  V {volume:>10.2f}")
    print(f"      MOM   : MACD {macd:>10.2f}  HIST {hist:>10.2f}  RSI {rsi:>8.2f}  PPO {ppo:>10.2f}  PPO_H {ppo_hist:>10.2f}")
    print(f"      FLOW  : CVD {cvd:>12.2f}  CVD_R {cvd_rolling:>12.2f}  VOL_D {volume_delta:>10.2f}")
    print(f"      POS   : OI {oi:>12.2f}  OI_USD {oi_usd:>14.2f}  FUND {funding_rate:>10.6f}")
    print(
        f"      OI+   : OIC {oi_contracts:>12.2f}  "
        f"d {0.0 if oi_contracts_change is None else oi_contracts_change:>10.2f}  "
        f"({0.0 if oi_contracts_change_pct is None else oi_contracts_change_pct:>+7.4f}%)"
    )
    print(
        f"      OIN+  : OIN {oi_notional:>14.2f}  "
        f"d {0.0 if oi_notional_change is None else oi_notional_change:>12.2f}  "
        f"({0.0 if oi_notional_change_pct is None else oi_notional_change_pct:>+7.4f}%)"
    )
    print()


def analyze_current_cycle_hierarchy() -> None:
    project_root = _find_project_root()
    base_path = str(PROJECT_PATHS.cycle_structured_dir)
    base_data_path = str(PROJECT_PATHS.base_data_dir)
    hierarchy_path = os.path.join(base_path, "cycle_hierarchy_map.json")

    print(f"프로젝트: {project_root}")
    print(f"사이클 데이터: {base_path}")

    if not os.path.exists(hierarchy_path):
        print(f"계층맵 없음: {hierarchy_path}")
        return

    with open(hierarchy_path, "r", encoding="utf-8") as f:
        hierarchy_map = json.load(f)

    dfs: dict[str, pd.DataFrame | None] = {
        tf: load_cycle_data(base_path, tf) for tf in ["1w", "1d", "4h", "1h"]
    }

    if dfs["1h"] is None or dfs["1h"].empty:
        print("1H 사이클 데이터 없음")
        return

    # ?? ?꾩옱 ?ъ씠??泥댁씤 寃곗젙 ??????????????????????????????????????????????
    current_1h_row = dfs["1h"].iloc[-1]
    current_1h_id = current_1h_row["cycle_id"]
    current_1h_map = hierarchy_map.get("1h", {}).get(current_1h_id, {})

    parent_4h_id, fb_4h = resolve_parent_or_latest(hierarchy_map, current_1h_map, "4h", dfs["4h"])
    parent_1d_id, fb_1d = resolve_parent_or_latest(hierarchy_map, current_1h_map, "1d", dfs["1d"])
    parent_1w_id, fb_1w = resolve_parent_or_latest(hierarchy_map, current_1h_map, "1w", dfs["1w"])

    current_4h_row = row_by_cycle_id(dfs["4h"], parent_4h_id)
    current_1d_row = row_by_cycle_id(dfs["1d"], parent_1d_id)
    current_1w_row = row_by_cycle_id(dfs["1w"], parent_1w_id)

    if current_4h_row is None or current_1d_row is None or current_1w_row is None:
        print("상위 사이클 해석 불가")
        return

    info_1h = extract_features(current_1h_row)
    info_4h = extract_features(current_4h_row)
    info_1d = extract_features(current_1d_row)
    info_1w = extract_features(current_1w_row)

    # ?? n_up 泥댁씤 ?????????????????????????????????????????????????????????
    chain = ""
    n_up = 0
    for tf_label, inf in [("1w", info_1w), ("1d", info_1d), ("4h", info_4h), ("1h", info_1h)]:
        ct = str(inf["cycle_type"]).lower()
        letter = "U" if ct == "up" else "D"
        chain += letter
        if ct == "up":
            n_up += 1

    # ?? ?꾩튂 怨꾩궛 ?????????????????????????????????????????????????????????
    pos_1h_4h = chronological_position(dfs["1h"], hierarchy_map, "1h", "4h", parent_4h_id, current_1h_id)
    pos_1h_1d = chronological_position(dfs["1h"], hierarchy_map, "1h", "1d", parent_1d_id, current_1h_id)
    pos_1h_1w = chronological_position(dfs["1h"], hierarchy_map, "1h", "1w", parent_1w_id, current_1h_id)
    pos_4h_1d = chronological_position(dfs["4h"], hierarchy_map, "4h", "1d", parent_1d_id, parent_4h_id)
    pos_4h_1w = chronological_position(dfs["4h"], hierarchy_map, "4h", "1w", parent_1w_id, parent_4h_id)
    pos_1d_1w = chronological_position(dfs["1d"], hierarchy_map, "1d", "1w", parent_1w_id, parent_1d_id)

    # ?? ?덉뒪?좉렇???꾪솚 媛寃???????????????????????????????????????????????
    TF_FILES = {"1h": "BTCUSD_1h.csv", "4h": "BTCUSD_4h.csv", "1d": "BTCUSD_1d.csv", "1w": "BTCUSD_1w.csv"}
    flips = {}
    live_states = {}
    for tf, fname in TF_FILES.items():
        csv_path = os.path.join(base_data_path, fname)
        flips[tf] = calc_histogram_flip_price(csv_path)
        live_states[tf] = load_latest_csv_state(csv_path)

    live_assessments = {
        "1h": assess_live_cycle_state(info_1h["cycle_type"], live_states.get("1h"), flips.get("1h")),
        "4h": assess_live_cycle_state(info_4h["cycle_type"], live_states.get("4h"), flips.get("4h")),
        "1d": assess_live_cycle_state(info_1d["cycle_type"], live_states.get("1d"), flips.get("1d")),
        "1w": assess_live_cycle_state(info_1w["cycle_type"], live_states.get("1w"), flips.get("1w")),
    }

    # ?? 異쒕젰 ??????????????????????????????????????????????????????????????
    print()
    print("=" * 100)
    print(f"  현재 사이클 계층 분석  |  체인: {chain}  n_up={n_up}")
    print("=" * 100)
    print(f"  1H: {current_1h_id}  ->  4H: {parent_4h_id}{'*' if fb_4h else ''}  ->  1D: {parent_1d_id}{'*' if fb_1d else ''}  ->  1W: {parent_1w_id}{'*' if fb_1w else ''}")
    if fb_4h or fb_1d or fb_1w:
        print("  (* = 매핑이 없어 최신 사이클로 대체)")
    print()

    print_cycle_block("1W", parent_1w_id, info_1w, "기준 상위", fallback=fb_1w)
    print_live_block("1w", live_states.get("1w"), live_assessments["1w"])
    print_latest_candle_block("1w", live_states.get("1w"))
    print_cycle_candle_context("1w", parent_1w_id, dfs["1w"], current_1w_row)
    print_cycle_block(
        "1D in 1W", parent_1d_id, info_1d,
        format_position(*pos_1d_1w),
        fallback=fb_1d,
    )
    print_live_block("1d", live_states.get("1d"), live_assessments["1d"])
    print_latest_candle_block("1d", live_states.get("1d"))
    print_cycle_candle_context("1d", parent_1d_id, dfs["1d"], current_1d_row)
    print_cycle_block(
        "4H in 1D", parent_4h_id, info_4h,
        format_position(*pos_4h_1d),
        extra_text=f"  | in 1W: {format_position(*pos_4h_1w)}",
        fallback=fb_4h,
    )
    print_live_block("4h", live_states.get("4h"), live_assessments["4h"])
    print_latest_candle_block("4h", live_states.get("4h"))
    print_cycle_candle_context("4h", parent_4h_id, dfs["4h"], current_4h_row)
    print_cycle_block(
        "1H in 4H", current_1h_id, info_1h,
        format_position(*pos_1h_4h),
        extra_text=f"  | in 1D: {format_position(*pos_1h_1d)} | in 1W: {format_position(*pos_1h_1w)}",
    )

    # ?? ?덉뒪?좉렇???꾪솚 媛寃?釉붾줉 ?????????????????????????????????????????
    print("-" * 100)
    print("  MACD 히스토그램 전환 가격 (다음 캔들 기준)")
    print("-" * 100)
    print_cycle_candle_context("1h", current_1h_id, dfs["1h"], current_1h_row)
    cur_price = flips.get("1h", {}).get("current_close", 0) if flips.get("1h") else 0
    print(f"  현재가: ${cur_price:,.2f}\n")

    print_live_block("1h", live_states.get("1h"), live_assessments["1h"])
    print_latest_candle_block("1h", live_states.get("1h"))
    for tf in ["1h", "4h", "1d", "1w"]:
        print_flip_block(tf, flips.get(tf))
    print()

    print("=" * 100)
    print(f"  요약: 체인 {chain} (n_up={n_up})")
    print(f"  1H ({info_1h['cycle_type'].upper()}) {format_position(*pos_1h_4h)} in 4H"
          f"  |  4H ({info_4h['cycle_type'].upper()}) {format_position(*pos_4h_1d)} in 1D"
          f"  |  1D ({info_1d['cycle_type'].upper()}) {format_position(*pos_1d_1w)} in 1W"
          f"  |  1W ({info_1w['cycle_type'].upper()})")

    # ?꾪솚 媛寃??쒖쨪 ?붿빟
    for tf in ["1h", "4h", "1d", "1w"]:
        fl = flips.get(tf)
        if fl:
            direction = "상승" if fl["current_hist"] < 0 else "하락"
            print(f"  {tf} hist 전환 {direction}: ${fl['flip_price']:,.2f} ({fl['flip_pct']:+.2f}%)"
                  f"  |  감속: ${fl['decel_price']:,.2f} ({fl['decel_pct']:+.2f}%)")

    print("=" * 100)


if __name__ == "__main__":
    analyze_current_cycle_hierarchy()
