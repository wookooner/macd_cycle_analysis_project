"""API server for cycle dashboard.
Reads parquet + hierarchy_map directly, same logic as the original working data builder.
"""
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path("data/cycle_data/structured")
_oi_cache: dict = {}


def _dir(value):
    t = str(value or "").strip().lower()
    if "up" in t or t == "u": return "U"
    if "down" in t or t == "d": return "D"
    return None


def _safe(v, default=0):
    if v is None: return default
    try:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return default
    except: pass
    return v


def _get_oi_lookup(tf: str) -> tuple:
    """Returns (idx_lookup: dict[int, float], date_lookup: dict[str, float]).
    idx_lookup: candle_N (0-based) → oi
    date_lookup: normalized date string → oi
    Both cached in _oi_cache[tf]."""
    if tf in _oi_cache:
        return _oi_cache[tf]
    oi_path = Path("data/base_data") / f"BTCUSDT_oi_{tf}.csv"
    ohlcv_path = Path("data/base_data") / f"BTCUSD_{tf}.csv"
    if not oi_path.exists() or not ohlcv_path.exists():
        _oi_cache[tf] = ({}, {})
        return ({}, {})
    try:
        oi_df = pd.read_csv(oi_path, usecols=["date", "oi"])
        oi_df["date"] = pd.to_datetime(oi_df["date"])
        oi_df = oi_df.dropna(subset=["oi"])

        ohlcv = pd.read_csv(ohlcv_path, usecols=["date"])
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        ohlcv["candle_idx"] = range(0, len(ohlcv))  # candle_N = iloc[N] (0-indexed)

        merged = ohlcv.merge(oi_df, on="date", how="inner")
        idx_lookup = dict(zip(merged["candle_idx"], merged["oi"].astype(float)))
        # date_lookup: "YYYY-MM-DD HH:MM:SS" normalized string → oi
        date_lookup = {
            str(d): float(v)
            for d, v in zip(merged["date"], merged["oi"])
        }
        result = (idx_lookup, date_lookup)
        _oi_cache[tf] = result
        print(f"[INFO] OI lookup {tf}: {len(idx_lookup)} entries")
        return result
    except Exception as e:
        print(f"[WARN] OI load failed for {tf}: {e}")
        _oi_cache[tf] = ({}, {})
        return ({}, {})


def _resolve_parents(H: dict, tf: str, cid: str, parents: dict):
    """직접 부모가 없을 때 계층 탐색으로 p4h/p1d/p1w 보완. (p4h, p1d, p1w) 반환."""
    p4h = (parents.get("4h") or [None])[0]
    p1d = (parents.get("1d") or [None])[0]
    p1w = (parents.get("1w") or [None])[0]

    # 1W 없으면 1D 부모의 1W 부모로 보완
    if not p1w and p1d and p1d in H.get("1d", {}):
        via = H["1d"][p1d].get("parent_cycle_ids", {}).get("1w", [])
        if via:
            p1w = via[0]

    # 1H인데 4H 없으면 1D 부모의 4H 자식 중 이 1H를 포함하는 것을 탐색
    if tf == "1h" and not p4h and p1d and p1d in H.get("1d", {}):
        h4 = H.get("4h", {})
        for s4h in H["1d"][p1d].get("child_cycle_ids", {}).get("4h", []):
            if cid in h4.get(s4h, {}).get("child_cycle_ids", {}).get("1h", []):
                p4h = s4h
                break

    return p4h, p1d, p1w


def _find_parquet(tf: str):
    """Prefer non-enriched (has candle_data), fallback to enriched."""
    candidates = [
        BASE_DIR / "btc" / f"cycles_{tf}.parquet",
        BASE_DIR / f"cycles_{tf}.parquet",
        BASE_DIR / "btc" / f"cycles_{tf}_enriched.parquet",
        BASE_DIR / f"cycles_{tf}_enriched.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_hierarchy():
    for p in [BASE_DIR / "btc" / "cycle_hierarchy_map.json", BASE_DIR / "cycle_hierarchy_map.json"]:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _rpos(o, n):
    if n <= 1: return 50
    return round(100 * (o - 1) / max(n - 1, 1))


def _order(H, ptf, pid, ctf, cid):
    if not pid or not cid: return 0, 0, 50
    sibs = H.get(ptf, {}).get(pid, {}).get("child_cycle_ids", {}).get(ctf, [])
    if cid not in sibs: return 0, len(sibs), 50
    o = sibs.index(cid) + 1
    return o, len(sibs), _rpos(o, len(sibs))


def _same_dir_label(H, parent_tf, parent_id, child_tf, child_id, child_dir):
    sibs = H.get(parent_tf, {}).get(parent_id, {}).get("child_cycle_ids", {}).get(child_tf, [])
    if child_id not in sibs: return ""
    count = sum(1 for sid in sibs[:sibs.index(child_id)]
                if _dir(H.get(child_tf, {}).get(sid, {}).get("cycle_type")) == child_dir)
    return child_dir + str(count + 1)


def _get_start_end(row):
    """Extract start/end indicator values. Prefer candle_data, fallback to cycle_features."""
    candles = row.get("candle_data")
    has_candles = candles is not None and hasattr(candles, '__len__') and len(candles) > 0

    if has_candles:
        cl = list(candles)
        fc, lc = cl[0], cl[-1]
        return {
            "rsi": round(float(_safe(fc.get("rsi"), 50)), 1),
            "ppo": round(float(_safe(fc.get("ppo"), 0)), 4),
            "ppoh": round(float(_safe(fc.get("ppo_hist"), 0)), 4),
            "cvd": round(float(_safe(fc.get("cvd_rolling"), 0)), 1),
            "ersi": round(float(_safe(lc.get("rsi"), 50)), 1),
            "eppo": round(float(_safe(lc.get("ppo"), 0)), 4),
            "eppoh": round(float(_safe(lc.get("ppo_hist"), 0)), 4),
            "ecvd": round(float(_safe(lc.get("cvd_rolling"), 0)), 1),
        }

    # Fallback: use cycle_features
    feat = row.get("cycle_features", {})
    if not isinstance(feat, dict): feat = {}
    st = feat.get("start", {})
    en = feat.get("end", {})
    return {
        "rsi": round(float(_safe(st.get("rsi"), 50)), 1),
        "ppo": round(float(_safe(st.get("ppo"), 0)), 4),
        "ppoh": round(float(_safe(st.get("ppo_hist"), 0)), 4),
        "cvd": round(float(_safe(st.get("cvd_rolling"), 0)), 1),
        "ersi": round(float(_safe(en.get("rsi"), 50)), 1),
        "eppo": round(float(_safe(en.get("ppo"), 0)), 4),
        "eppoh": round(float(_safe(en.get("ppo_hist"), 0)), 4),
        "ecvd": round(float(_safe(en.get("cvd_rolling"), 0)), 1),
    }


def _process_tf(tf: str, H: dict) -> list:
    path = _find_parquet(tf)
    if not path:
        print(f"[WARN] {tf} parquet not found")
        return []

    df = pd.read_parquet(path).sort_values("start_date").reset_index(drop=True)
    tf_map = H.get(tf, {})
    print(f"[INFO] {tf}: {len(df)} rows from {path.name}, hierarchy has {len(tf_map)} entries")

    oi_idx_lookup, oi_date_lookup = _get_oi_lookup(tf)  # 캐시된 OI 조회 테이블 사전 로드

    results = []
    skipped = 0

    for _, row in df.iterrows():
        cid = row["cycle_id"]
        node = tf_map.get(cid, {})
        parents = node.get("parent_cycle_ids", {})

        # Direction
        ct = _dir(row.get("cycle_type")) or _dir(node.get("cycle_type")) or "U"

        # 부모 ID 결정 — 직접 없으면 계층 탐색으로 보완 (Bug3 fix)
        p4h, p1d, p1w = _resolve_parents(H, tf, cid, parents)

        # 보완 후에도 필수 부모 없으면 스킵
        if not p1w or p1w not in H.get("1w", {}):
            skipped += 1
            continue
        if tf in ("1h", "4h") and (not p1d or p1d not in H.get("1d", {})):
            skipped += 1
            continue
        if tf == "1h" and (not p4h or p4h not in H.get("4h", {})):
            skipped += 1
            continue

        # Parent directions
        t4h = _dir(H.get("4h", {}).get(p4h, {}).get("cycle_type")) if p4h and p4h in H.get("4h", {}) else ct
        t1d = _dir(H.get("1d", {}).get(p1d, {}).get("cycle_type")) if p1d and p1d in H.get("1d", {}) else ct
        t1w = _dir(H.get("1w", {}).get(p1w, {}).get("cycle_type")) if p1w and p1w in H.get("1w", {}) else ct

        # Features
        feat = row.get("cycle_features", {})
        if not isinstance(feat, dict): feat = {}
        sh = feat.get("shape", {})
        ch = feat.get("change", {})
        st = feat.get("start", {})
        ag = feat.get("aggregate", {})

        indicators = _get_start_end(row)

        # OI 변화율 계산 (Bug1 fix)
        oi_chg = None
        if oi_idx_lookup or oi_date_lookup:
            try:
                sd, ed = str(row["start_date"]), str(row["end_date"])
                # Try date-based lookup first (new parquet format after Bug B fix)
                if oi_date_lookup and not sd.startswith("candle_"):
                    sd_norm = str(pd.to_datetime(sd))
                    ed_norm = str(pd.to_datetime(ed))
                    oi_s = oi_date_lookup.get(sd_norm)
                    oi_e = oi_date_lookup.get(ed_norm)
                else:
                    # Fallback to candle-index-based lookup (old parquet format)
                    si = int(sd.replace("candle_", ""))
                    ei = int(ed.replace("candle_", ""))
                    oi_s = oi_idx_lookup.get(si)
                    oi_e = oi_idx_lookup.get(ei)
                if oi_s and oi_e and oi_s != 0:
                    oi_chg = round((oi_e - oi_s) / oi_s * 100, 4)
            except Exception:
                pass

        r = {
            "ct": ct or "U", "w": t1w or "U", "d": t1d or "U", "h": t4h or "U",
            "pct": round(float(_safe(ch.get("price_pct"), 0)), 4),
            "dur": int(_safe(sh.get("duration_candles"), row.get("duration_candles", 0))),
            **indicators,
            "fr": _safe(st.get("fr_current"), None),
            "frsl": _safe(st.get("fr_slope"), None),
            "tbr": _safe(ag.get("taker_buy_ratio"), None),
            "apph": _safe(ag.get("area_ppo_hist"), None),
            "oi_chg": oi_chg,
        }

        # Positions
        if tf == "1h":
            o4h, n4h, r4h = _order(H, "4h", p4h, "1h", cid)
            r.update({"o4h": o4h, "r4h": r4h})

        o1d, n1d, r1d = _order(H, "1d", p1d, tf, cid)
        o1w, n1w, r1w = _order(H, "1w", p1w, tf, cid)
        r.update({"o1d": o1d, "r1d": r1d, "o1w": o1w, "r1w": r1w})

        # Parent-of-parent
        if tf == "1h":
            p1d_ch4h = H.get("1d", {}).get(p1d, {}).get("child_cycle_ids", {}).get("4h", [])
            o4h1d = (p1d_ch4h.index(p4h) + 1) if p4h in p1d_ch4h else 0
            r.update({"o4h1d": o4h1d, "r4h1d": _rpos(o4h1d, len(p1d_ch4h)) if o4h1d else 0})

        p1w_ch1d = H.get("1w", {}).get(p1w, {}).get("child_cycle_ids", {}).get("1d", [])
        o1d1w = (p1w_ch1d.index(p1d) + 1) if p1d and p1d in p1w_ch1d else 0
        r.update({"o1d1w": o1d1w, "r1d1w": _rpos(o1d1w, len(p1w_ch1d)) if o1d1w else 0})

        # Labels
        if tf == "1h":
            r["p4l"] = _same_dir_label(H, "1d", p1d, "4h", p4h, t4h)
            r["p1dl"] = _same_dir_label(H, "1w", p1w, "1d", p1d, t1d)
        elif tf == "4h":
            r["p1dl"] = _same_dir_label(H, "1w", p1w, "1d", p1d, t1d)
        elif tf == "1d":
            r["lbl"] = _same_dir_label(H, "1w", p1w, "1d", cid, ct)

        results.append(r)

    print(f"[INFO] {tf}: {len(results)} results, {skipped} skipped (no parent)")
    return results


@app.get("/api/dashboard")
def get_dashboard_data():
    try:
        H = _load_hierarchy()
        if not H:
            return {"error": "hierarchy_map.json not found", "h1": [], "h4": [], "d1": []}

        h1 = _process_tf("1h", H)
        h4 = _process_tf("4h", H)
        d1 = _process_tf("1d", H)

        # Quick stats
        from collections import Counter
        if h1:
            wc = Counter(r["w"] for r in h1)
            cc = Counter(r["ct"] for r in h1)
            print(f"[STAT] 1H w={dict(wc)}, ct={dict(cc)}")

        return {"h1": h1, "h4": h4, "d1": d1}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "h1": [], "h4": [], "d1": []}


@app.get("/api/cycles/{timeframe}")
def get_cycle_data(timeframe: str):
    try:
        H = _load_hierarchy()
        return _process_tf(timeframe.lower(), H)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/api/debug")
def debug_info():
    """Diagnostic info for troubleshooting."""
    from collections import Counter
    H = _load_hierarchy()
    info = {"hierarchy": {tf: len(ids) for tf, ids in H.items()}}
    
    for tf in ["1h", "4h", "1d"]:
        path = _find_parquet(tf)
        if not path:
            info[tf] = {"error": "not found"}
            continue
        df = pd.read_parquet(path)
        has_candle = "candle_data" in df.columns
        has_features = "cycle_features" in df.columns
        ids_in_h = sum(1 for cid in df["cycle_id"] if cid in H.get(tf, {}))
        info[tf] = {
            "file": str(path),
            "rows": len(df),
            "has_candle_data": has_candle,
            "has_cycle_features": has_features,
            "ids_in_hierarchy": ids_in_h,
            "cycle_types": dict(Counter(df["cycle_type"].tolist())),
        }
    
    data = _process_tf_debug("1h", H)
    info["1h_sample"] = data
    return info


def _process_tf_debug(tf, H):
    """Quick stats from _process_tf output."""
    from collections import Counter
    results = _process_tf(tf, H)
    if not results:
        return {"count": 0}
    return {
        "count": len(results),
        "w": dict(Counter(r["w"] for r in results)),
        "ct": dict(Counter(r["ct"] for r in results)),
        "d": dict(Counter(r["d"] for r in results)),
        "o1w_zero": sum(1 for r in results if r.get("o1w", 0) == 0),
        "o1d_zero": sum(1 for r in results if r.get("o1d", 0) == 0),
        "sample": results[0] if results else None,
    }