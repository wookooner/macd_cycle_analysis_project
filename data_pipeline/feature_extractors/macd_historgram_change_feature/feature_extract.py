"""
feature_extract.py
Categorized cycle feature extractor.
"""

import inspect
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import linregress as scipy_linregress
except Exception:
    scipy_linregress = None

try:
    from .config import DEFAULT_CONFIG
except Exception:
    from config import DEFAULT_CONFIG

warnings.filterwarnings("ignore")


def _find_project_root() -> Path:
    def _is_project_root(path: Path) -> bool:
        return (path / "data" / "base_data").exists() and (path / "data_pipeline").exists()

    for path in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        candidate = path if path.is_dir() else path.parent
        for current in [candidate] + list(candidate.parents):
            if _is_project_root(current):
                return current

    for current in [Path.cwd()] + list(Path.cwd().parents):
        if _is_project_root(current):
            return current
    return Path.cwd()


project_root = _find_project_root()


class CycleFeatureCalculator:
    """Categorized cycle feature calculator."""

    REQUIRED_DEFAULTS = {
        "open": 0.0,
        "high": 0.0,
        "low": 0.0,
        "close": 0.0,
        "volume": 0.0,
        "macd": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0,
        "ppo": 0.0,
        "ppo_signal": 0.0,
        "ppo_hist": 0.0,
        "rsi": 50.0,
    }

    OPTIONAL_NUMERIC_DEFAULTS = {
        "volume_delta": np.nan,
        "delta": np.nan,
        "cvd": np.nan,
        "cvd_rolling": np.nan,
        "taker_buy_base": np.nan,
    }

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.name = "Categorized Cycle Feature Calculator"
        self.version = "4.0"

    def extract_features_from_candle_data(
        self,
        candle_data: List[Dict],
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict]:
        if candle_data is None or len(candle_data) == 0:
            return self.config.get_default_cycle_features_structure()

        if context_data is None:
            context_data = {}

        df = pd.DataFrame(candle_data)

        for col, default in self.REQUIRED_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        for col, default in self.OPTIONAL_NUMERIC_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        features: Dict[str, Dict] = {}
        for category_name, category_data in self.config.FEATURE_CATEGORIES.items():
            features[category_name] = {}
            for feature_name, feature_config in category_data["features"].items():
                if not feature_config["enabled"]:
                    continue
                try:
                    calculator_method = getattr(self, feature_config["calculator"])
                    sig = inspect.signature(calculator_method)
                    if "context_data" in sig.parameters:
                        value = calculator_method(df, context_data=context_data)
                    else:
                        value = calculator_method(df)

                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        features[category_name][feature_name] = None
                        continue

                    if feature_config["data_type"] == "int":
                        value = int(value)
                    elif feature_config["data_type"] == "float":
                        value = float(value)

                    features[category_name][feature_name] = value
                except Exception:
                    features[category_name][feature_name] = feature_config["default_value"]
        return features

    def _first(self, df: pd.DataFrame, col: str, default=None):
        if len(df) == 0 or col not in df.columns:
            return default
        value = df[col].iloc[0]
        return default if pd.isna(value) else value

    def _last(self, df: pd.DataFrame, col: str, default=None):
        if len(df) == 0 or col not in df.columns:
            return default
        value = df[col].iloc[-1]
        return default if pd.isna(value) else value

    def _change(self, df: pd.DataFrame, col: str, default=0.0):
        start = self._first(df, col, None)
        end = self._last(df, col, None)
        if start is None or end is None:
            return default
        return float(end - start)

    def _fr_history(self, context_data: Optional[Dict[str, Any]]) -> List[float]:
        if not context_data:
            return []
        raw = context_data.get("funding_rate_history", [])
        values: List[float] = []
        for value in raw:
            if value is None or pd.isna(value):
                continue
            values.append(float(value))
        return values

    def _recent_fr_values(self, context_data: Optional[Dict[str, Any]], n: int) -> List[float]:
        history = self._fr_history(context_data)
        if not history:
            return []
        return history[-n:]

    def _fr_mean(self, context_data: Optional[Dict[str, Any]], n: int) -> Optional[float]:
        values = self._recent_fr_values(context_data, n)
        if len(values) < n:
            return None
        return float(np.mean(values))

    def _fr_slope_value(self, values: List[float]) -> Optional[float]:
        if len(values) < 2:
            return None
        x = np.arange(len(values), dtype=float)
        y = np.asarray(values, dtype=float)
        if scipy_linregress is not None:
            return float(scipy_linregress(x, y).slope)
        return float(np.polyfit(x, y, 1)[0])

    def calc_duration_candles(self, df: pd.DataFrame) -> int:
        return len(df)

    def calc_core_count(self, df: pd.DataFrame) -> int:
        if len(df) < 2:
            return 0
        hist_changes = df["macd_hist"].diff().dropna()
        if len(hist_changes) == 0:
            return 0
        overall_change = df["macd_hist"].iloc[-1] - df["macd_hist"].iloc[0]
        trend_direction = 1 if overall_change > 0 else -1
        return int((np.sign(hist_changes) == trend_direction).sum())

    def calc_noise_count(self, df: pd.DataFrame) -> int:
        duration = self.calc_duration_candles(df)
        core_count = self.calc_core_count(df)
        return max(0, duration - core_count - 1)

    def calc_direction_change(self, df: pd.DataFrame) -> int:
        if len(df) < 2:
            return 0
        hist_changes = df["macd_hist"].diff().dropna()
        if len(hist_changes) == 0:
            return 0
        directions = np.sign(hist_changes)
        return int((directions.diff() != 0).sum())

    def calc_peak_price_position(self, df: pd.DataFrame) -> float:
        if len(df) <= 1:
            return 0.5
        peak_index = int(np.argmax(df["high"].values))
        return round(peak_index / (len(df) - 1), 4)

    def calc_trough_price_position(self, df: pd.DataFrame) -> float:
        if len(df) <= 1:
            return 0.5
        trough_index = int(np.argmin(df["low"].values))
        return round(trough_index / (len(df) - 1), 4)

    def calc_direction_pct(self, df: pd.DataFrame) -> float:
        duration = self.calc_duration_candles(df)
        if duration == 0:
            return 0.0
        return self.calc_core_count(df) / duration * 100

    def calc_hist_positive_ratio(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        return (df["macd_hist"] > 0).sum() / len(df) * 100

    def calc_price_up_ratio(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        return (df["close"] > df["open"]).sum() / len(df) * 100

    def calc_price_down_ratio(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        return (df["close"] < df["open"]).sum() / len(df) * 100

    def calc_start_price(self, df: pd.DataFrame) -> float:
        return float(self._first(df, "close", 0.0))

    def calc_start_volume(self, df: pd.DataFrame) -> float:
        return float(self._first(df, "volume", 0.0))

    def calc_start_rsi(self, df: pd.DataFrame) -> float:
        return float(self._first(df, "rsi", 50.0))

    def calc_start_macd(self, df: pd.DataFrame) -> float:
        return float(self._first(df, "macd", 0.0))

    def calc_start_ppo(self, df: pd.DataFrame) -> float:
        return float(self._first(df, "ppo", 0.0))

    def calc_start_macd_signal(self, df: pd.DataFrame) -> float:
        return float(self._first(df, "macd_signal", 0.0))

    def calc_start_hist(self, df: pd.DataFrame) -> float:
        return float(self._first(df, "macd_hist", 0.0))

    def calc_start_ppo_hist(self, df: pd.DataFrame) -> float:
        return float(self._first(df, "ppo_hist", 0.0))

    def calc_start_cvd(self, df: pd.DataFrame) -> Optional[float]:
        value = self._first(df, "cvd", None)
        return None if value is None else float(value)

    def calc_start_cvd_rolling(self, df: pd.DataFrame) -> Optional[float]:
        value = self._first(df, "cvd_rolling", None)
        return None if value is None else float(value)

    def calc_start_funding_rate(self, df: pd.DataFrame, context_data: Optional[Dict[str, Any]] = None) -> Optional[float]:
        return self.calc_fr_current(df, context_data=context_data)

    def calc_fr_current(self, df: pd.DataFrame, context_data: Optional[Dict[str, Any]] = None) -> Optional[float]:
        history = self._fr_history(context_data)
        return history[-1] if history else None

    def calc_fr_24h(self, df: pd.DataFrame, context_data: Optional[Dict[str, Any]] = None) -> Optional[float]:
        return self._fr_mean(context_data, 3)

    def calc_fr_72h(self, df: pd.DataFrame, context_data: Optional[Dict[str, Any]] = None) -> Optional[float]:
        return self._fr_mean(context_data, 9)

    def calc_fr_slope(self, df: pd.DataFrame, context_data: Optional[Dict[str, Any]] = None) -> Optional[float]:
        values = self._recent_fr_values(context_data, 6)
        if len(values) < 6:
            return None
        return self._fr_slope_value(values)

    def calc_fr_vol(self, df: pd.DataFrame, context_data: Optional[Dict[str, Any]] = None) -> Optional[float]:
        values = self._recent_fr_values(context_data, 6)
        if len(values) < 6:
            return None
        return float(np.std(values))

    def calc_fr_neg_streak(self, df: pd.DataFrame, context_data: Optional[Dict[str, Any]] = None) -> int:
        history = self._fr_history(context_data)
        if not history:
            return 0
        streak = 0
        for value in reversed(history):
            if value < 0:
                streak += 1
            else:
                break
        return streak

    def calc_fr_regime(self, df: pd.DataFrame, context_data: Optional[Dict[str, Any]] = None) -> Optional[float]:
        history = self._fr_history(context_data)
        if len(history) < 6:
            return None
        recent_mean = float(np.mean(history[-3:]))
        previous_mean = float(np.mean(history[-6:-3]))
        return recent_mean - previous_mean

    def calc_fr_vs_72h(self, df: pd.DataFrame, context_data: Optional[Dict[str, Any]] = None) -> Optional[float]:
        fr_current = self.calc_fr_current(df, context_data=context_data)
        fr_72h = self.calc_fr_72h(df, context_data=context_data)
        if fr_current is None or fr_72h is None:
            return None
        return float(fr_current - fr_72h)

    def calc_end_price(self, df: pd.DataFrame) -> float:
        return float(self._last(df, "close", 0.0))

    def calc_end_volume(self, df: pd.DataFrame) -> float:
        return float(self._last(df, "volume", 0.0))

    def calc_end_rsi(self, df: pd.DataFrame) -> float:
        return float(self._last(df, "rsi", 50.0))

    def calc_end_macd(self, df: pd.DataFrame) -> float:
        return float(self._last(df, "macd", 0.0))

    def calc_end_ppo(self, df: pd.DataFrame) -> float:
        return float(self._last(df, "ppo", 0.0))

    def calc_end_macd_signal(self, df: pd.DataFrame) -> float:
        return float(self._last(df, "macd_signal", 0.0))

    def calc_end_hist(self, df: pd.DataFrame) -> float:
        return float(self._last(df, "macd_hist", 0.0))

    def calc_end_ppo_hist(self, df: pd.DataFrame) -> float:
        return float(self._last(df, "ppo_hist", 0.0))

    def calc_end_cvd(self, df: pd.DataFrame) -> Optional[float]:
        value = self._last(df, "cvd", None)
        return None if value is None else float(value)

    def calc_price_change_pct(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        start_price = self._first(df, "close", 0.0)
        end_price = self._last(df, "close", 0.0)
        if start_price in (None, 0):
            return 0.0
        return float(((end_price - start_price) / start_price) * 100)

    def calc_rsi_change(self, df: pd.DataFrame) -> float:
        return self._change(df, "rsi", 0.0)

    def calc_macd_change(self, df: pd.DataFrame) -> float:
        return self._change(df, "macd", 0.0)

    def calc_ppo_change(self, df: pd.DataFrame) -> float:
        return self._change(df, "ppo", 0.0)

    def calc_macd_signal_change(self, df: pd.DataFrame) -> float:
        return self._change(df, "macd_signal", 0.0)

    def calc_macd_histogram_change(self, df: pd.DataFrame) -> float:
        return self._change(df, "macd_hist", 0.0)

    def calc_ppo_hist_change(self, df: pd.DataFrame) -> float:
        return self._change(df, "ppo_hist", 0.0)

    def calc_cvd_change(self, df: pd.DataFrame) -> Optional[float]:
        start = self.calc_start_cvd(df)
        end = self.calc_end_cvd(df)
        if start is None or end is None:
            return None
        return float(end - start)

    def calc_max_high_pct(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        start_price = self._first(df, "close", 0.0)
        if start_price in (None, 0):
            return 0.0
        return max(0.0, (df["high"].max() - start_price) / start_price * 100)

    def calc_max_loss_pct(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        start_price = self._first(df, "close", 0.0)
        if start_price in (None, 0):
            return 0.0
        return min(0.0, (df["low"].min() - start_price) / start_price * 100)

    def calc_max_intraday_high_pct(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0
        closes = df["close"].values
        highs = df["high"].values
        future_max_high = np.maximum.accumulate(highs[::-1])[::-1]
        future_max_high_after = np.empty(len(highs))
        future_max_high_after[:-1] = future_max_high[1:]
        future_max_high_after[-1] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            gains = np.where(
                (closes > 0) & ~np.isnan(future_max_high_after),
                (future_max_high_after - closes) / closes * 100,
                np.nan,
            )
        valid = gains[~np.isnan(gains)]
        return float(np.max(valid)) if len(valid) > 0 else 0.0

    def calc_max_intraday_loss_pct(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0
        closes = df["close"].values
        lows = df["low"].values
        future_min_low = np.minimum.accumulate(lows[::-1])[::-1]
        future_min_low_after = np.empty(len(lows))
        future_min_low_after[:-1] = future_min_low[1:]
        future_min_low_after[-1] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            losses = np.where(
                (closes > 0) & ~np.isnan(future_min_low_after),
                (future_min_low_after - closes) / closes * 100,
                np.nan,
            )
        valid = losses[~np.isnan(losses)]
        return float(np.min(valid)) if len(valid) > 0 else 0.0

    def calc_avg_true_range(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        prev_closes = np.concatenate([[closes[0]], closes[:-1]])
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)))
        return float(np.mean(tr))

    def calc_price_change_deviation(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0
        price_changes = df["close"].pct_change().dropna()
        if len(price_changes) == 0:
            return 0.0
        return float(price_changes.std() * 100)

    def calc_all_volume(self, df: pd.DataFrame) -> float:
        return float(df["volume"].sum()) if len(df) > 0 else 0.0

    def calc_aggregate_cvd(self, df: pd.DataFrame) -> Optional[float]:
        if len(df) == 0:
            return None
        source_col = "delta" if "delta" in df.columns and not df["delta"].isna().all() else "volume_delta"
        if source_col not in df.columns or df[source_col].isna().all():
            return None
        total = df[source_col].sum()
        return float(total) if pd.notna(total) else None

    def calc_area_ppo_hist(self, df: pd.DataFrame) -> float:
        if "ppo_hist" not in df.columns or len(df) == 0:
            return 0.0
        return float(np.abs(df["ppo_hist"].fillna(0.0)).sum())

    def calc_taker_buy_ratio(self, df: pd.DataFrame) -> Optional[float]:
        if "taker_buy_base" not in df.columns or "volume" not in df.columns:
            return None
        taker_buy = pd.to_numeric(df["taker_buy_base"], errors="coerce").fillna(0.0).sum()
        total_volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).sum()
        if total_volume <= 0:
            return None
        return float(taker_buy / total_volume)

    def convert_legacy_features_to_categorized(self, legacy_features: Dict[str, Any]) -> Dict[str, Dict]:
        categorized_features = self.config.get_default_cycle_features_structure()
        legacy_mapping = {
            "duration_candles": ("shape", "duration_candles"),
            "core_count": ("shape", "core_count"),
            "noise_count": ("shape", "noise_count"),
            "direction_change": ("shape", "direction_change"),
            "peak_price_position": ("shape", "peak_price_position"),
            "trough_price_position": ("shape", "trough_price_position"),
            "direction_pct": ("strength", "direction_pct"),
            "hist_positive_ratio": ("strength", "hist_positive_ratio"),
            "price_up_ratio": ("strength", "price_up_ratio"),
            "price_down_ratio": ("strength", "price_down_ratio"),
            "start_price": ("start", "price"),
            "start_volume": ("start", "volume"),
            "start_rsi": ("start", "rsi"),
            "start_macd": ("start", "macd"),
            "start_ppo": ("start", "ppo"),
            "start_macd_signal": ("start", "macd_signal"),
            "start_hist": ("start", "hist"),
            "start_ppo_hist": ("start", "ppo_hist"),
            "start_cvd": ("start", "cvd"),
            "end_price": ("end", "price"),
            "end_volume": ("end", "volume"),
            "end_rsi": ("end", "rsi"),
            "end_macd": ("end", "macd"),
            "end_ppo": ("end", "ppo"),
            "end_macd_signal": ("end", "macd_signal"),
            "end_hist": ("end", "hist"),
            "end_ppo_hist": ("end", "ppo_hist"),
            "end_cvd": ("end", "cvd"),
            "price_change_pct": ("change", "price_pct"),
            "rsi_change": ("change", "rsi"),
            "macd_change": ("change", "macd"),
            "ppo_change": ("change", "ppo"),
            "macd_signal_change": ("change", "macd_signal"),
            "macd_histogram_change": ("change", "hist"),
            "ppo_hist_change": ("change", "ppo_hist"),
            "cvd_change": ("change", "cvd"),
            "max_high_pct": ("volatility", "max_high_pct"),
            "max_loss_pct": ("volatility", "max_loss_pct"),
            "max_high_change": ("volatility", "max_intraday_high_pct"),
            "max_loss_change": ("volatility", "max_intraday_loss_pct"),
            "avg_true_range": ("volatility", "avg_true_range"),
            "price_change_deviation": ("volatility", "price_change_deviation"),
            "all_volume": ("aggregate", "volume"),
            "aggregate_cvd": ("aggregate", "cvd"),
            "area_ppo_hist": ("aggregate", "area_ppo_hist"),
            "taker_buy_ratio": ("aggregate", "taker_buy_ratio"),
        }
        for legacy_name, value in legacy_features.items():
            if legacy_name in legacy_mapping:
                category, feature = legacy_mapping[legacy_name]
                if category in categorized_features:
                    categorized_features[category][feature] = value
        return categorized_features


class StructuredCycleProcessor:
    """Structured cycle processor."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.calculator = CycleFeatureCalculator()

    @staticmethod
    def load_funding_rate(funding_rate_path: Path) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(funding_rate_path)
            if "funding_rate" not in df.columns:
                return None

            if "unix" in df.columns:
                df["timestamp"] = pd.to_datetime(df["unix"], unit="s", utc=True)
            elif "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"], utc=True)
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            else:
                return None

            df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception:
            return None

    @staticmethod
    def get_recent_funding_rates(
        funding_rate_df: pd.DataFrame,
        start_date_str: str,
        lookback: int = 10,
    ) -> List[float]:
        try:
            start_dt = pd.to_datetime(start_date_str, utc=True)
            timestamps = funding_rate_df["timestamp"]
            end_idx = int(timestamps.searchsorted(start_dt, side="right"))
            if end_idx <= 0:
                return []
            start_idx = max(0, end_idx - lookback)
            values = funding_rate_df.iloc[start_idx:end_idx]["funding_rate"].dropna().astype(float).tolist()
            return values
        except Exception:
            return []

    @staticmethod
    def get_funding_rate_at(
        funding_rate_df: pd.DataFrame,
        start_date_str: str,
    ) -> Optional[float]:
        history = StructuredCycleProcessor.get_recent_funding_rates(funding_rate_df, start_date_str, lookback=1)
        return history[-1] if history else None

    def process_and_enrich_cycles(
        self,
        output_path: Optional[Path] = None,
        funding_rate_path: Optional[Path] = None,
    ):
        try:
            df = pd.read_parquet(self.data_path)
            funding_rate_df = None
            if funding_rate_path is not None:
                funding_rate_df = self.load_funding_rate(Path(funding_rate_path))

            new_features_list = []
            for _, row in df.iterrows():
                candle_data = row["candle_data"]
                context_data: Dict[str, Any] = {}
                if funding_rate_df is not None:
                    fr_history = self.get_recent_funding_rates(funding_rate_df, row["start_date"], lookback=10)
                    if fr_history:
                        context_data["funding_rate_history"] = fr_history
                        context_data["funding_rate"] = fr_history[-1]

                if isinstance(candle_data, (list, np.ndarray)) and len(candle_data) > 0:
                    enriched_features = self.calculator.extract_features_from_candle_data(
                        list(candle_data),
                        context_data=context_data,
                    )
                else:
                    enriched_features = self.calculator.config.get_default_cycle_features_structure()
                new_features_list.append(enriched_features)

            df["cycle_features"] = [
                {k: v for k, v in feat.items() if not (isinstance(v, dict) and len(v) == 0)}
                for feat in new_features_list
            ]

            if output_path is None:
                output_path = self.data_path.with_name(self.data_path.name.replace(".parquet", "_enriched.parquet"))
            df.to_parquet(output_path, index=False)
            return output_path, len(df)
        except Exception:
            return None, 0

    def convert_existing_cycles_to_new_structure(self, output_path: Optional[Path] = None):
        try:
            df = pd.read_parquet(self.data_path)
            converted_cycles = []

            for _, row in df.iterrows():
                candle_data = row["candle_data"]
                if isinstance(candle_data, (list, np.ndarray)) and len(candle_data) > 0:
                    new_features = self.calculator.extract_features_from_candle_data(list(candle_data))
                else:
                    legacy_features = row["cycle_features"] if isinstance(row["cycle_features"], dict) else {}
                    new_features = self.calculator.convert_legacy_features_to_categorized(legacy_features)

                converted_cycles.append(
                    {
                        "cycle_id": row["cycle_id"],
                        "timeframe": row["timeframe"],
                        "start_date": row["start_date"],
                        "end_date": row["end_date"],
                        "cycle_type": row["cycle_type"],
                        "duration_candles": row["duration_candles"],
                        "category": row["category"],
                        "algorithm_used": row["algorithm_used"],
                        "candle_data": row["candle_data"],
                        "cycle_features": new_features,
                    }
                )

            new_df = pd.DataFrame(converted_cycles)
            if output_path is None:
                output_path = self.data_path.with_name(f"converted_{self.data_path.name}")
            new_df.to_parquet(output_path, index=False)
            return output_path, len(converted_cycles)
        except Exception:
            return None, 0

    def validate_new_structure(self, converted_file: Path):
        try:
            pd.read_parquet(converted_file)
            return True
        except Exception:
            return False


def convert_all_timeframes():
    structured_path = project_root / "data" / "cycle_data" / "structured"
    if not structured_path.exists():
        return
    for file_path in structured_path.glob("cycles_*.parquet"):
        if "_enriched" in file_path.name or "converted_" in file_path.name:
            continue
        processor = StructuredCycleProcessor(file_path)
        processor.convert_existing_cycles_to_new_structure()


def process_all_timeframes_for_enrichment(funding_rate_path: Optional[str] = None):
    structured_path = project_root / "data" / "cycle_data" / "structured"
    if not structured_path.exists():
        return

    if funding_rate_path is None:
        default_fr_path = project_root / "data" / "base_data" / "BTCUSDT_funding_rate.csv"
        if default_fr_path.exists():
            funding_rate_path = str(default_fr_path)

    for file_path in structured_path.glob("cycles_*.parquet"):
        if "_enriched" in file_path.name or "converted_" in file_path.name:
            continue
        processor = StructuredCycleProcessor(file_path)
        processor.process_and_enrich_cycles(funding_rate_path=funding_rate_path)


def test_new_structure():
    sample_candles = [
        {
            "timestamp": "2024-01-01 00:00:00",
            "open": 42000,
            "high": 42500,
            "low": 41800,
            "close": 42150,
            "volume": 120.5,
            "macd": 235.67,
            "macd_signal": 280.90,
            "macd_hist": -45.23,
            "ppo": -0.54,
            "ppo_signal": -0.48,
            "ppo_hist": -0.06,
            "rsi": 55.2,
            "volume_delta": -150.3,
            "delta": -150.3,
            "cvd": -150.3,
            "cvd_rolling": -320.5,
            "taker_buy_base": 40.0,
        },
        {
            "timestamp": "2024-01-01 04:00:00",
            "open": 42150,
            "high": 43200,
            "low": 42000,
            "close": 43000,
            "volume": 150.2,
            "macd": 400.89,
            "macd_signal": 350.15,
            "macd_hist": 50.74,
            "ppo": 0.18,
            "ppo_signal": 0.05,
            "ppo_hist": 0.13,
            "rsi": 62.8,
            "volume_delta": 280.7,
            "delta": 280.7,
            "cvd": 130.4,
            "cvd_rolling": 120.4,
            "taker_buy_base": 100.0,
        },
    ]

    context_data = {"funding_rate_history": [-0.0001, 0.0, 0.00012, 0.00014]}
    context_data["funding_rate"] = context_data["funding_rate_history"][-1]

    calculator = CycleFeatureCalculator()
    features = calculator.extract_features_from_candle_data(sample_candles, context_data=context_data)
    print(json.dumps(features, indent=2, ensure_ascii=False, default=str))
    return features


if __name__ == "__main__":
    print("Categorized feature extractor")
    print("1: test")
    print("2: convert")
    print("3: enrich")
    choice = input("Select (1-3): ").strip()
    if choice == "1":
        test_new_structure()
    elif choice == "2":
        convert_all_timeframes()
    elif choice == "3":
        fr_path = input("Funding rate CSV path (optional): ").strip() or None
        process_all_timeframes_for_enrichment(funding_rate_path=fr_path)
