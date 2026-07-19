from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from lib import (
    DEFAULT_COST_PCT,
    DEFAULT_ENTRY_OFFSETS,
    DEFAULT_HORIZONS,
    FEATURE_PATH,
    TIMEFRAMES,
    add_legal_nup,
    add_observable_columns,
    ensure_out_dir,
    load_market,
)


def signal_indices(df: pd.DataFrame) -> np.ndarray:
    direction = df["hist_delta_sign"].to_numpy(dtype="float64")
    changed = np.r_[False, direction[1:] != direction[:-1]]
    return np.flatnonzero(changed)


def build_for_timeframe(
    timeframe: str,
    market: pd.DataFrame,
    markets: dict[str, pd.DataFrame],
    horizons: tuple[int, ...],
    entry_offsets: tuple[int, ...],
    cost_pct: float,
) -> pd.DataFrame:
    sig_idx = signal_indices(market)
    if len(sig_idx) == 0:
        return pd.DataFrame()

    close = market["close"].to_numpy(dtype="float64")
    high = market["high"].to_numpy(dtype="float64")
    low = market["low"].to_numpy(dtype="float64")
    direction = market["hist_delta_sign"].to_numpy(dtype="float64")
    max_h = max(horizons)
    rows: list[dict] = []

    for idx in sig_idx:
        sig_dir = direction[idx]
        for offset in entry_offsets:
            entry_idx = int(idx + offset)
            if entry_idx + max_h >= len(market):
                continue
            if direction[entry_idx] != sig_dir:
                continue
            entry_close = close[entry_idx]
            if not np.isfinite(entry_close) or entry_close <= 0:
                continue
            source = market.iloc[entry_idx]
            row = {
                "timeframe": timeframe,
                "signal_index": int(idx),
                "signal_time": market.iloc[idx]["timestamp"],
                "entry_index": entry_idx,
                "entry_offset_candles": int(offset),
                "entry_time": source["timestamp"],
                "entry_close": float(entry_close),
                "direction": "long" if sig_dir > 0 else "short",
                "direction_sign": float(sig_dir),
                "feature_source_max_time": source["timestamp"],
                "hist": source["hist"],
                "hist_sign": source["hist_sign"],
                "hist_slope_3": source["hist_slope_3"],
                "hist_streak": source["hist_streak"],
                "rsi": source["rsi"],
                "rsi_slope_3": source["rsi_slope_3"],
                "ppo": source.get("ppo", np.nan),
                "ppo_hist": source.get("ppo_hist", np.nan),
                "dist_ma7": source["dist_ma7"],
                "dist_ma25": source["dist_ma25"],
                "dist_ma99": source["dist_ma99"],
                "ma_alignment": source["ma_alignment"],
                "atr_over_price": source["atr_over_price"],
                "vol_regime": source["vol_regime"],
            }
            for h in horizons:
                exit_idx = entry_idx + h
                window_high = high[entry_idx + 1 : exit_idx + 1]
                window_low = low[entry_idx + 1 : exit_idx + 1]
                gross = (close[exit_idx] / entry_close - 1.0) * 100.0 * sig_dir
                if sig_dir > 0:
                    mfe = (np.nanmax(window_high) / entry_close - 1.0) * 100.0
                    mae = (np.nanmin(window_low) / entry_close - 1.0) * 100.0
                else:
                    mfe = (1.0 - np.nanmin(window_low) / entry_close) * 100.0
                    mae = (1.0 - np.nanmax(window_high) / entry_close) * 100.0
                row[f"net_return_h{h}"] = float(gross - cost_pct)
                row[f"mfe_h{h}"] = float(mfe)
                row[f"mae_h{h}"] = float(mae)
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = add_legal_nup(out, timeframe, markets)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframes", nargs="+", default=list(TIMEFRAMES), choices=list(TIMEFRAMES))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--entry-offsets", nargs="+", type=int, default=list(DEFAULT_ENTRY_OFFSETS))
    parser.add_argument("--cost-pct", type=float, default=DEFAULT_COST_PCT)
    args = parser.parse_args()

    out_dir = ensure_out_dir()
    horizons = tuple(sorted(set(args.horizons)))
    entry_offsets = tuple(sorted(set(args.entry_offsets)))
    markets = {tf: add_observable_columns(load_market(tf), tf) for tf in TIMEFRAMES}

    frames = []
    errors: dict[str, str] = {}
    for tf in args.timeframes:
        try:
            frames.append(build_for_timeframe(tf, markets[tf], markets, horizons, entry_offsets, args.cost_pct))
        except Exception as exc:
            errors[tf] = str(exc)

    features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    features = features.dropna(subset=["entry_time", "direction", "hist_slope_3", "atr_over_price", f"net_return_h{horizons[0]}"])
    features.to_parquet(FEATURE_PATH, index=False)
    metadata = {
        "rows": int(len(features)),
        "timeframes": list(args.timeframes),
        "horizons": list(horizons),
        "entry_offsets": list(entry_offsets),
        "cost_pct": args.cost_pct,
        "errors": errors,
        "feature_path": str(FEATURE_PATH),
    }
    (out_dir / "observable_features_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
