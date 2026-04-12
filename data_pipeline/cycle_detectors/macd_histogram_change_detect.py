"""
Multi-timeframe MACD histogram cycle detection wrapper.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data_pipeline.cycle_detectors.algorithms.macd_histogram_change import (
    SimpleConfig,
    SimpleMACDAlgorithm,
)
from data_pipeline.feature_extractors.macd_historgram_change_feature.feature_extract import (
    CycleFeatureCalculator,
    StructuredCycleProcessor,
)
from src.common.paths import PROJECT_PATHS

project_root = PROJECT_PATHS.project_root

_NON_INDICATOR_KEYWORDS = ["oi", "funding", "interest", "backfill"]
_OPTIONAL_CANDLE_COLS = [
    "volume_delta",
    "delta",
    "cvd",
    "cvd_rolling",
    "taker_buy_base",
    "ppo",
    "ppo_signal",
    "ppo_hist",
    "oi",
    "oi_usd",
    "oi_change",
    "oi_change_pct",
    "oi_contracts",
    "oi_contracts_change",
    "oi_contracts_change_pct",
    "oi_notional",
    "oi_notional_change",
    "oi_notional_change_pct",
    "funding_rate",
    "ma_7",
    "ma_25",
    "ma_99",
]


def load_algorithm():
    algorithm = SimpleMACDAlgorithm()
    config = SimpleConfig(max_opposite_consecutive=2, min_cycle_length=3)
    return algorithm, config


def _has_macd_hist(file_path: Path) -> bool:
    try:
        if file_path.suffix == ".csv":
            header = pd.read_csv(file_path, nrows=0)
            return "macd_hist" in header.columns
        if file_path.suffix == ".parquet":
            import pyarrow.parquet as pq

            schema = pq.read_schema(file_path)
            return "macd_hist" in schema.names
    except Exception:
        return False
    return False


def find_timeframe_files():
    base_data_path = PROJECT_PATHS.base_data_dir
    if not base_data_path.exists():
        return {}

    timeframe_files = {}
    patterns = {
        "1min": ["*1min*", "*_1min.*", "*1minute*", "*_1minute.*", "*BTCUSD_1m.csv", "*1m_intraday*"],
        "5m": ["*5m*", "*_5m.*", "*5minute*", "*_5minutes.*"],
        "15m": ["*15m*", "*_15m.*", "*15minute*", "*_15minutes.*"],
        "30m": ["*30m*", "*_30m.*", "*30minute*", "*_30minutes.*"],
        "1h": ["*1h*", "*_1h.*", "*1hour*"],
        "4h": ["*4h*", "*_4h.*", "*4hour*"],
        "1d": ["*1d*", "*_1d.*", "*1day*", "*daily*"],
        "1w": ["*1w*", "*_1w.*", "*1week*", "*weekly*"],
        "1M": ["*1M*", "*_1M.*", "*1month*", "*monthly*"],
    }

    for timeframe, pattern_list in patterns.items():
        for pattern in pattern_list:
            raw_files = list(base_data_path.glob(pattern))
            keyword_filtered = [
                f for f in raw_files if not any(kw in f.name.lower() for kw in _NON_INDICATOR_KEYWORDS)
            ]
            indicator_files = [f for f in keyword_filtered if _has_macd_hist(f)]
            if indicator_files:
                timeframe_files[timeframe] = max(indicator_files, key=lambda f: f.stat().st_mtime)
                break
    return timeframe_files


def _prune_empty_cycle_features(cycle_features: Dict) -> Dict:
    if not isinstance(cycle_features, dict):
        return cycle_features
    return {k: v for k, v in cycle_features.items() if not (isinstance(v, dict) and len(v) == 0)}


def create_cycle_records_v3(
    data: pd.DataFrame,
    cycles: List[Dict],
    timeframe: str,
    algorithm_name: str,
    funding_rate_df: Optional[pd.DataFrame] = None,
):
    cycle_records = []
    feature_calculator = CycleFeatureCalculator()

    for i, cycle in enumerate(cycles):
        start_idx = cycle.start_idx if hasattr(cycle, "start_idx") else cycle["start_idx"]
        end_idx = cycle.end_idx if hasattr(cycle, "end_idx") else cycle["end_idx"]
        cycle_type = cycle.cycle_type if hasattr(cycle, "cycle_type") else cycle["cycle_type"]

        if cycle_type == "rising":
            cycle_type = "up"
        elif cycle_type == "falling":
            cycle_type = "down"

        cycle_data = data.iloc[start_idx : end_idx + 1].copy()

        candle_data = []
        for idx, (timestamp, row) in enumerate(cycle_data.iterrows()):
            if hasattr(timestamp, "strftime"):
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(timestamp, str):
                timestamp_str = timestamp
            else:
                date_val = row.get("date") if hasattr(row, "get") else None
                if date_val is not None and pd.notna(date_val):
                    timestamp_str = str(date_val)
                else:
                    timestamp_str = f"candle_{start_idx + idx}"

            candle_record = {
                "timestamp": timestamp_str,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", row.get("Volume USD", 0))),
                "macd": float(row.get("macd", 0)),
                "macd_signal": float(row.get("macd_signal", 0)),
                "macd_hist": float(row.get("macd_hist", 0)),
                "rsi": float(row.get("rsi", 50.0)) if pd.notna(row.get("rsi")) else 50.0,
            }

            for col in _OPTIONAL_CANDLE_COLS:
                if col in row.index and pd.notna(row[col]):
                    candle_record[col] = float(row[col])

            candle_data.append(candle_record)

        start_date_str = candle_data[0]["timestamp"] if candle_data else f"idx_{start_idx}"
        end_date_str = candle_data[-1]["timestamp"] if candle_data else f"idx_{end_idx}"
        cycle_id = f"cycle_{timeframe}_{i + 1:03d}"

        start_macd = candle_data[0]["macd"] if candle_data else 0
        hist_direction = "rising" if cycle_type == "up" else "falling"
        macd_zone = "positive" if start_macd >= 0 else "negative"
        category = f"{hist_direction}_{macd_zone}"

        context_data: Dict[str, object] = {}
        if funding_rate_df is not None:
            fr_history = StructuredCycleProcessor.get_recent_funding_rates(
                funding_rate_df,
                start_date_str,
                lookback=10,
            )
            if fr_history:
                context_data["funding_rate_history"] = fr_history
                context_data["funding_rate"] = fr_history[-1]

        cycle_features = feature_calculator.extract_features_from_candle_data(
            candle_data,
            context_data=context_data,
        )
        cycle_features = _prune_empty_cycle_features(cycle_features)

        cycle_records.append(
            {
                "cycle_id": cycle_id,
                "timeframe": timeframe,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "cycle_type": cycle_type,
                "duration_candles": len(cycle_data),
                "category": category,
                "algorithm_used": algorithm_name,
                "candle_data": candle_data,
                "cycle_features": cycle_features,
            }
        )

    return cycle_records


def detect_cycles_for_timeframe_v3(
    file_path,
    timeframe,
    algorithm,
    config,
    funding_rate_df: Optional[pd.DataFrame] = None,
):
    if file_path.suffix == ".parquet":
        data = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        data = pd.read_csv(file_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    original_index = data.index.copy()
    valid_mask = data["macd_hist"].notna()
    data = data[valid_mask].copy()
    original_index = original_index[valid_mask]
    data.reset_index(drop=True, inplace=True)

    cycles, classification = algorithm.detect_cycles(data, config)
    classification_clean = classification.fillna(0).astype(int)

    if len(classification_clean) != len(data):
        full_classification = pd.Series(0, index=range(len(data)), dtype=int)
        copy_length = min(len(classification_clean), len(data))
        if copy_length > 0:
            full_classification.iloc[:copy_length] = classification_clean.iloc[:copy_length]
        classification_clean = full_classification

    data_with_original_index = data.copy()
    data_with_original_index.index = original_index

    cycle_records = create_cycle_records_v3(
        data_with_original_index,
        cycles,
        timeframe,
        algorithm.name,
        funding_rate_df,
    )
    return cycle_records, len(cycles)


def save_cycle_results_v3(cycle_records: List[Dict], timeframe: str):
    if not cycle_records:
        return False

    output_path = PROJECT_PATHS.cycle_structured_dir
    output_path.mkdir(parents=True, exist_ok=True)
    cycle_file = output_path / f"cycles_{timeframe}.parquet"
    df = pd.DataFrame(cycle_records)
    df.to_parquet(cycle_file, index=False)
    return True


def run_fixed_detection():
    algorithm, config = load_algorithm()
    funding_rate_df = None
    funding_rate_path = PROJECT_PATHS.base_data_dir / "BTCUSDT_funding_rate.csv"
    if funding_rate_path.exists():
        funding_rate_df = StructuredCycleProcessor.load_funding_rate(funding_rate_path)

    timeframe_files = find_timeframe_files()
    results = {}

    for timeframe, file_path in timeframe_files.items():
        cycle_records, cycle_count = detect_cycles_for_timeframe_v3(
            file_path,
            timeframe,
            algorithm,
            config,
            funding_rate_df,
        )
        if save_cycle_results_v3(cycle_records, timeframe):
            results[timeframe] = {"cycle_count": cycle_count}
    return results


if __name__ == "__main__":
    run_fixed_detection()
