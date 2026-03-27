"""
now_cycle.py
현재 사이클 상태 + MACD 히스토그램 전환 가격 분석

수정 이력 (v2 2026-03-23):
  - 프로젝트 루트 자동 탐색 (경로 하드코딩 제거)
  - 4h 부모 미매핑 시 최신 4h 사이클로 fallback
  - end 지표를 candle_data에서 직접 추출 (cycle_features.end 비활성 대응)
  - 각 타임프레임별 MACD 히스토그램 전환 가격 계산 추가
  - n_up 체인 표시 추가
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── 프로젝트 루트 탐색 ─────────────────────────────────────────────────────

def _find_project_root() -> Path:
    for current in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
        if (current / "data" / "base_data").exists():
            return current
    for current in [Path.cwd()] + list(Path.cwd().parents):
        if (current / "data" / "base_data").exists():
            return current
    return Path(__file__).resolve().parent


# ── 데이터 로드 ────────────────────────────────────────────────────────────

def load_cycle_data(base_path: str, timeframe: str) -> pd.DataFrame | None:
    file_path = os.path.join(base_path, f"cycles_{timeframe}.parquet")
    if not os.path.exists(file_path):
        return None
    return pd.read_parquet(file_path)


# ── Feature 추출 ───────────────────────────────────────────────────────────

def extract_features(cycle_row: pd.Series) -> dict[str, Any]:
    """사이클 row에서 핵심 지표 추출. end 값은 candle_data에서 직접 가져옴."""
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

    # end 값은 candle_data 마지막 캔들에서 추출 (cycle_features.end는 비활성)
    end_macd = end_hist = end_rsi = end_price = 0.0
    candle_data = cycle_row.get("candle_data", [])
    if candle_data is not None and len(candle_data) > 0:
        last = candle_data[-1] if isinstance(candle_data[-1], dict) else {}
        end_macd = float(last.get("macd", 0) or 0)
        end_hist = float(last.get("macd_hist", 0) or 0)
        end_rsi = float(last.get("rsi", 0) or 0)
        end_price = float(last.get("close", 0) or 0)

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
        "core_count": int(safe_get("shape", "core_count")),
        "noise_count": int(safe_get("shape", "noise_count")),
    }


# ── 히스토그램 전환 가격 계산 ──────────────────────────────────────────────

def calc_histogram_flip_price(csv_path: str) -> dict[str, float] | None:
    """다음 캔들에서 MACD hist가 부호 반전되는 종가(close)를 역산.

    원리:
      EMA_fast_new = close * k_f + EMA_fast_prev * (1-k_f)
      EMA_slow_new = close * k_s + EMA_slow_prev * (1-k_s)
      MACD_new = EMA_fast_new - EMA_slow_new
      Signal_new = MACD_new * k_sig + Signal_prev * (1-k_sig)
      Hist_new = MACD_new - Signal_new
              = (1-k_sig) * (MACD_new - Signal_prev)

      Hist_new = 0  ⟹  MACD_new = Signal_prev
      → close_flip = [Signal_prev - EMA_fast_prev*(1-k_f) + EMA_slow_prev*(1-k_s)] / (k_f - k_s)
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

    # EMA 전체 계산
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

    # 현재 상태
    cur_close = close[-1]
    cur_hist = hist[-1]
    cur_macd = macd[-1]
    cur_signal = sig[-1]

    # 전환 가격
    denom = k_f - k_s
    if abs(denom) < 1e-15:
        return None

    flip_close = (sig[-1] - ef[-1] * (1 - k_f) + es[-1] * (1 - k_s)) / denom
    pct_to_flip = (flip_close - cur_close) / cur_close * 100

    # Δhist=0 가격 (hist 변화량이 0인 가격 = 감속→가속 경계)
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


# ── 유틸 ───────────────────────────────────────────────────────────────────

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
    """매핑된 부모를 찾되, 없으면 해당 타임프레임의 최신 사이클을 fallback으로 사용."""
    pid = latest_parent_id(current_map_node, parent_tf)
    if pid:
        return pid, False
    # fallback: 최신 사이클
    if df_parent is not None and not df_parent.empty:
        return df_parent.iloc[-1]["cycle_id"], True
    return None, True


# ── 출력 포맷 ──────────────────────────────────────────────────────────────

def print_cycle_block(
    title: str,
    cycle_id: str,
    info: dict[str, Any],
    pos_text: str,
    extra_text: str = "",
    fallback: bool = False,
) -> None:
    fb_mark = " ⚠️(fallback)" if fallback else ""
    direction = str(info["cycle_type"]).upper()
    print(f"┌─ [{title}] {cycle_id}{fb_mark}")
    print(f"│  방향: {direction:4s}  캔들: {info['duration_candles']}  위치: {pos_text}{extra_text}")
    print(f"│  기간: {info['start_date']} ~ {info['end_date']}")
    print(f"│  가격: {info['start_price']:>10.2f} → {info['end_price']:>10.2f}  ({info['price_change_pct']:+.2f}%)")
    print(f"│  RSI : {info['start_rsi']:>8.2f} → {info['end_rsi']:>8.2f}  ({info['rsi_change']:+.2f})")
    print(f"│  MACD: {info['start_macd']:>10.2f} → {info['end_macd']:>10.2f}  ({info['macd_change']:+.2f})")
    print(f"│  HIST: {info['start_hist']:>10.2f} → {info['end_hist']:>10.2f}  ({info['hist_change']:+.2f})")
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

    direction = "▲" if hist > 0 else "▼"
    flip_dir = "하락" if hist > 0 else "상승"

    print(f"  {tf:>3s}  hist: {hist:>+12.2f} {direction}  │  전환(=0): ${fp:>10.2f} ({fpct:>+6.2f}% {flip_dir}필요)  │  감속(Δ=0): ${dp:>10.2f} ({dpct:>+6.2f}%)")


# ── 메인 분석 ──────────────────────────────────────────────────────────────

def analyze_current_cycle_hierarchy() -> None:
    project_root = _find_project_root()
    base_path = str(project_root / "data" / "cycle_data" / "structured")
    base_data_path = str(project_root / "data" / "base_data")
    hierarchy_path = os.path.join(base_path, "cycle_hierarchy_map.json")

    print(f"📁 프로젝트: {project_root}")
    print(f"📁 사이클:   {base_path}")

    if not os.path.exists(hierarchy_path):
        print(f"❌ 계층맵 없음: {hierarchy_path}")
        return

    with open(hierarchy_path, "r", encoding="utf-8") as f:
        hierarchy_map = json.load(f)

    dfs: dict[str, pd.DataFrame | None] = {
        tf: load_cycle_data(base_path, tf) for tf in ["1w", "1d", "4h", "1h"]
    }

    if dfs["1h"] is None or dfs["1h"].empty:
        print("❌ 1H 사이클 데이터 없음")
        return

    # ── 현재 사이클 체인 결정 ──────────────────────────────────────────────
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
        print("❌ 상위 사이클 해석 불가")
        return

    info_1h = extract_features(current_1h_row)
    info_4h = extract_features(current_4h_row)
    info_1d = extract_features(current_1d_row)
    info_1w = extract_features(current_1w_row)

    # ── n_up 체인 ─────────────────────────────────────────────────────────
    chain = ""
    n_up = 0
    for tf_label, inf in [("1w", info_1w), ("1d", info_1d), ("4h", info_4h), ("1h", info_1h)]:
        ct = str(inf["cycle_type"]).lower()
        letter = "U" if ct == "up" else "D"
        chain += letter
        if ct == "up":
            n_up += 1

    # ── 위치 계산 ─────────────────────────────────────────────────────────
    pos_1h_4h = chronological_position(dfs["1h"], hierarchy_map, "1h", "4h", parent_4h_id, current_1h_id)
    pos_1h_1d = chronological_position(dfs["1h"], hierarchy_map, "1h", "1d", parent_1d_id, current_1h_id)
    pos_1h_1w = chronological_position(dfs["1h"], hierarchy_map, "1h", "1w", parent_1w_id, current_1h_id)
    pos_4h_1d = chronological_position(dfs["4h"], hierarchy_map, "4h", "1d", parent_1d_id, parent_4h_id)
    pos_4h_1w = chronological_position(dfs["4h"], hierarchy_map, "4h", "1w", parent_1w_id, parent_4h_id)
    pos_1d_1w = chronological_position(dfs["1d"], hierarchy_map, "1d", "1w", parent_1w_id, parent_1d_id)

    # ── 히스토그램 전환 가격 ──────────────────────────────────────────────
    TF_FILES = {"1h": "BTCUSD_1h.csv", "4h": "BTCUSD_4h.csv", "1d": "BTCUSD_1d.csv", "1w": "BTCUSD_1w.csv"}
    flips = {}
    for tf, fname in TF_FILES.items():
        csv_path = os.path.join(base_data_path, fname)
        flips[tf] = calc_histogram_flip_price(csv_path)

    # ── 출력 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print(f"  현재 사이클 계층 분석  |  체인: {chain}  n_up={n_up}")
    print("=" * 100)
    print(f"  1H: {current_1h_id}  →  4H: {parent_4h_id}{'*' if fb_4h else ''}  →  1D: {parent_1d_id}{'*' if fb_1d else ''}  →  1W: {parent_1w_id}{'*' if fb_1w else ''}")
    if fb_4h or fb_1d or fb_1w:
        print("  (* = 매핑 없어 최신 사이클로 대체)")
    print()

    print_cycle_block("1W", parent_1w_id, info_1w, "기준 상위", fallback=fb_1w)
    print_cycle_block(
        "1D in 1W", parent_1d_id, info_1d,
        format_position(*pos_1d_1w),
        fallback=fb_1d,
    )
    print_cycle_block(
        "4H in 1D", parent_4h_id, info_4h,
        format_position(*pos_4h_1d),
        extra_text=f"  │ in 1W: {format_position(*pos_4h_1w)}",
        fallback=fb_4h,
    )
    print_cycle_block(
        "1H in 4H", current_1h_id, info_1h,
        format_position(*pos_1h_4h),
        extra_text=f"  │ in 1D: {format_position(*pos_1h_1d)} │ in 1W: {format_position(*pos_1h_1w)}",
    )

    # ── 히스토그램 전환 가격 블록 ─────────────────────────────────────────
    print("─" * 100)
    print("  MACD 히스토그램 전환 가격 (다음 캔들 기준)")
    print("─" * 100)
    cur_price = flips.get("1h", {}).get("current_close", 0) if flips.get("1h") else 0
    print(f"  현재가: ${cur_price:,.2f}\n")

    for tf in ["1h", "4h", "1d", "1w"]:
        print_flip_block(tf, flips.get(tf))
    print()

    # ── 요약 ──────────────────────────────────────────────────────────────
    print("=" * 100)
    print(f"  요약: 체인 {chain} (n_up={n_up})")
    print(f"  1H ({info_1h['cycle_type'].upper()}) {format_position(*pos_1h_4h)} in 4H"
          f"  |  4H ({info_4h['cycle_type'].upper()}) {format_position(*pos_4h_1d)} in 1D"
          f"  |  1D ({info_1d['cycle_type'].upper()}) {format_position(*pos_1d_1w)} in 1W"
          f"  |  1W ({info_1w['cycle_type'].upper()})")

    # 전환 가격 한줄 요약
    for tf in ["1h", "4h", "1d", "1w"]:
        fl = flips.get(tf)
        if fl:
            direction = "↗" if fl["current_hist"] < 0 else "↘"
            print(f"  {tf} hist 전환{direction}: ${fl['flip_price']:,.2f} ({fl['flip_pct']:+.2f}%)"
                  f"  |  감속: ${fl['decel_price']:,.2f} ({fl['decel_pct']:+.2f}%)")

    print("=" * 100)


if __name__ == "__main__":
    analyze_current_cycle_hierarchy()