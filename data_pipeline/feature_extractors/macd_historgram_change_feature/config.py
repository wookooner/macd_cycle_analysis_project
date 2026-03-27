"""
config.py
Categorized feature registry for cycle feature extraction.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


class FeatureConfig:
    """Categorized feature configuration manager."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.FEATURE_CATEGORIES: Dict[str, Any] = {
            "shape": {
                "description": "Cycle structure features",
                "features": {
                    "duration_candles": self._feature("calc_duration_candles", True, 0, "int", "Cycle length"),
                    "core_count": self._feature("calc_core_count", True, 0, "int", "Core candles"),
                    "noise_count": self._feature("calc_noise_count", True, 0, "int", "Noise candles"),
                    "direction_change": self._feature("calc_direction_change", True, 0, "int", "Direction changes"),
                    "peak_price_position": self._feature(
                        "calc_peak_price_position", True, 0.5, "float", "Peak position"
                    ),
                    "trough_price_position": self._feature(
                        "calc_trough_price_position", True, 0.5, "float", "Trough position"
                    ),
                },
            },
            "strength": {
                "description": "Trend strength features",
                "features": {
                    "direction_pct": self._feature("calc_direction_pct", True, 0.0, "float", "Directional ratio"),
                    "hist_positive_ratio": self._feature(
                        "calc_hist_positive_ratio", False, 0.0, "float", "Positive MACD histogram ratio"
                    ),
                    "price_up_ratio": self._feature(
                        "calc_price_up_ratio", False, 0.0, "float", "Up candle ratio"
                    ),
                    "price_down_ratio": self._feature(
                        "calc_price_down_ratio", False, 0.0, "float", "Down candle ratio"
                    ),
                },
            },
            "start": {
                "description": "Start-of-cycle features",
                "features": {
                    "price": self._feature("calc_start_price", True, 0.0, "float", "Start close"),
                    "volume": self._feature("calc_start_volume", True, 0.0, "float", "Start volume"),
                    "rsi": self._feature("calc_start_rsi", True, 50.0, "float", "Start RSI"),
                    "macd": self._feature("calc_start_macd", True, 0.0, "float", "Start MACD"),
                    "ppo": self._feature("calc_start_ppo", True, 0.0, "float", "Start PPO"),
                    "macd_signal": self._feature(
                        "calc_start_macd_signal", False, 0.0, "float", "Start MACD signal"
                    ),
                    "hist": self._feature("calc_start_hist", True, 0.0, "float", "Start MACD histogram"),
                    "ppo_hist": self._feature("calc_start_ppo_hist", True, 0.0, "float", "Start PPO histogram"),
                    "cvd": self._feature("calc_start_cvd", True, None, "float", "Start CVD"),
                    "cvd_rolling": self._feature(
                        "calc_start_cvd_rolling", True, None, "float", "Start rolling CVD"
                    ),
                    "funding_rate": self._feature(
                        "calc_start_funding_rate", True, None, "float", "Legacy start funding rate"
                    ),
                    "fr_current": self._feature("calc_fr_current", True, None, "float", "Current funding rate"),
                    "fr_24h": self._feature("calc_fr_24h", True, None, "float", "3-sample FR average"),
                    "fr_72h": self._feature("calc_fr_72h", True, None, "float", "9-sample FR average"),
                    "fr_slope": self._feature("calc_fr_slope", True, None, "float", "6-sample FR slope"),
                    "fr_vol": self._feature("calc_fr_vol", True, None, "float", "6-sample FR volatility"),
                    "fr_neg_streak": self._feature(
                        "calc_fr_neg_streak", True, 0, "int", "Consecutive negative FR count"
                    ),
                    "fr_regime": self._feature("calc_fr_regime", True, None, "float", "Recent FR regime change"),
                    "fr_vs_72h": self._feature("calc_fr_vs_72h", True, None, "float", "Current FR vs 72h mean"),
                },
            },
            "end": {
                "description": "End-of-cycle features",
                "features": {
                    "price": self._feature("calc_end_price", False, 0.0, "float", "End close"),
                    "volume": self._feature("calc_end_volume", False, 0.0, "float", "End volume"),
                    "rsi": self._feature("calc_end_rsi", False, 50.0, "float", "End RSI"),
                    "macd": self._feature("calc_end_macd", False, 0.0, "float", "End MACD"),
                    "ppo": self._feature("calc_end_ppo", True, 0.0, "float", "End PPO"),
                    "macd_signal": self._feature(
                        "calc_end_macd_signal", False, 0.0, "float", "End MACD signal"
                    ),
                    "hist": self._feature("calc_end_hist", False, 0.0, "float", "End MACD histogram"),
                    "ppo_hist": self._feature("calc_end_ppo_hist", True, 0.0, "float", "End PPO histogram"),
                    "cvd": self._feature("calc_end_cvd", True, None, "float", "End CVD"),
                },
            },
            "change": {
                "description": "Start/end delta features",
                "features": {
                    "price_pct": self._feature("calc_price_change_pct", True, 0.0, "float", "Price change pct"),
                    "rsi": self._feature("calc_rsi_change", False, 0.0, "float", "RSI change"),
                    "macd": self._feature("calc_macd_change", False, 0.0, "float", "MACD change"),
                    "ppo": self._feature("calc_ppo_change", True, 0.0, "float", "PPO change"),
                    "macd_signal": self._feature(
                        "calc_macd_signal_change", False, 0.0, "float", "MACD signal change"
                    ),
                    "hist": self._feature(
                        "calc_macd_histogram_change", False, 0.0, "float", "MACD histogram change"
                    ),
                    "ppo_hist": self._feature("calc_ppo_hist_change", True, 0.0, "float", "PPO histogram change"),
                    "cvd": self._feature("calc_cvd_change", True, None, "float", "CVD change"),
                },
            },
            "volatility": {
                "description": "Volatility features",
                "features": {
                    "max_high_pct": self._feature("calc_max_high_pct", False, 0.0, "float", "Max high pct"),
                    "max_loss_pct": self._feature("calc_max_loss_pct", False, 0.0, "float", "Max loss pct"),
                    "max_intraday_high_pct": self._feature(
                        "calc_max_intraday_high_pct", False, 0.0, "float", "Max intraday high pct"
                    ),
                    "max_intraday_loss_pct": self._feature(
                        "calc_max_intraday_loss_pct", False, 0.0, "float", "Max intraday loss pct"
                    ),
                    "avg_true_range": self._feature(
                        "calc_avg_true_range", True, 0.0, "float", "Average true range"
                    ),
                    "price_change_deviation": self._feature(
                        "calc_price_change_deviation", True, 0.0, "float", "Price change deviation"
                    ),
                },
            },
            "aggregate": {
                "description": "Whole-cycle aggregate features",
                "features": {
                    "volume": self._feature("calc_all_volume", True, 0.0, "float", "Total volume"),
                    "cvd": self._feature("calc_aggregate_cvd", True, None, "float", "Aggregate delta sum"),
                    "area_ppo_hist": self._feature(
                        "calc_area_ppo_hist", True, 0.0, "float", "Absolute PPO histogram area"
                    ),
                    "taker_buy_ratio": self._feature(
                        "calc_taker_buy_ratio", True, None, "float", "Taker buy ratio"
                    ),
                },
            },
        }

        self.CALCULATION_CONFIG = {
            "batch_size": 1000,
            "handle_errors": True,
            "use_default_on_error": True,
        }
        self.import_config()

    @staticmethod
    def _feature(calculator: str, enabled: bool, default_value: Any, data_type: str, description: str) -> Dict[str, Any]:
        return {
            "description": description,
            "calculator": calculator,
            "enabled": enabled,
            "default_value": default_value,
            "data_type": data_type,
        }

    def get_all_features_flat(self) -> Dict[str, Dict[str, Any]]:
        flat_features = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            for feature_name, feature_config in category_data["features"].items():
                flat_name = f"{category_name}_{feature_name}"
                flat_features[flat_name] = {
                    "description": feature_config["description"],
                    "calculator": feature_config["calculator"],
                    "enabled": feature_config["enabled"],
                    "default_value": feature_config["default_value"],
                    "category": category_name,
                    "feature_key": feature_name,
                }
        return flat_features

    def get_enabled_features_by_category(self) -> Dict[str, Dict[str, Dict]]:
        enabled_by_category = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            enabled_features = {
                name: config
                for name, config in category_data["features"].items()
                if config["enabled"]
            }
            if enabled_features:
                enabled_by_category[category_name] = enabled_features
        return enabled_by_category

    def get_feature_names_by_category(self) -> Dict[str, List[str]]:
        names_by_category = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            feature_names = [
                name
                for name, config in category_data["features"].items()
                if config["enabled"]
            ]
            if feature_names:
                names_by_category[category_name] = feature_names
        return names_by_category

    def get_all_calculator_names(self) -> List[str]:
        return [
            feature_config["calculator"]
            for category_data in self.FEATURE_CATEGORIES.values()
            for feature_config in category_data["features"].values()
            if feature_config["enabled"]
        ]

    def get_default_cycle_features_structure(self) -> Dict[str, Dict]:
        structure = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            structure[category_name] = {
                feature_name: feature_config["default_value"]
                for feature_name, feature_config in category_data["features"].items()
                if feature_config["enabled"]
            }
        return structure

    def enable_feature(self, category_name: str, feature_name: str) -> bool:
        try:
            self.FEATURE_CATEGORIES[category_name]["features"][feature_name]["enabled"] = True
            print(f"Enabled '{category_name}.{feature_name}'")
            return True
        except KeyError:
            print(f"Missing feature: '{category_name}.{feature_name}'")
            return False

    def disable_feature(self, category_name: str, feature_name: str) -> bool:
        try:
            self.FEATURE_CATEGORIES[category_name]["features"][feature_name]["enabled"] = False
            print(f"Disabled '{category_name}.{feature_name}'")
            return True
        except KeyError:
            print(f"Missing feature: '{category_name}.{feature_name}'")
            return False

    def validate_calculator_functions(self, calculator_module: Any) -> Dict[str, Dict[str, bool]]:
        validation_results = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            validation_results[category_name] = {
                feature_name: hasattr(calculator_module, feature_config["calculator"])
                for feature_name, feature_config in category_data["features"].items()
                if feature_config["enabled"]
            }
        return validation_results

    def export_config(self):
        export_data = {
            "feature_categories": self.FEATURE_CATEGORIES,
            "calculation_config": self.CALCULATION_CONFIG,
            "version": "4.0",
            "structure_type": "categorized",
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

    def import_config(self):
        if not self.config_path.exists():
            return False
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                import_data = json.load(f)

            for category_name, category_data in import_data.get("feature_categories", {}).items():
                if category_name not in self.FEATURE_CATEGORIES:
                    continue
                for feature_name, feature_config in category_data.get("features", {}).items():
                    if feature_name in self.FEATURE_CATEGORIES[category_name]["features"]:
                        if "enabled" in feature_config:
                            self.FEATURE_CATEGORIES[category_name]["features"][feature_name]["enabled"] = (
                                feature_config["enabled"]
                            )

            if "calculation_config" in import_data:
                self.CALCULATION_CONFIG = import_data["calculation_config"]
            return True
        except Exception as e:
            print(f"Warning: failed to import config, using defaults: {e}")
            return False

    def print_feature_summary(self):
        print("\n" + "=" * 80)
        print("Feature Summary")
        print("=" * 80)

        total_features = 0
        enabled_features = 0

        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            features = category_data["features"]
            enabled_count = sum(1 for cfg in features.values() if cfg["enabled"])
            total_count = len(features)
            total_features += total_count
            enabled_features += enabled_count

            print(f"\n{category_name.upper()} ({category_data['description']})")
            print(f"  enabled: {enabled_count}/{total_count}")
            for feature_name, feature_config in features.items():
                status = "ON" if feature_config["enabled"] else "OFF"
                print(f"  [{status}] {feature_name}: {feature_config['description']}")

        print("\n" + "=" * 80)
        print(f"Total enabled: {enabled_features}/{total_features}")
        print("=" * 80)


DEFAULT_CONFIG_PATH = Path(__file__).parent / "features_config_v2.json"
DEFAULT_CONFIG = FeatureConfig(DEFAULT_CONFIG_PATH)


def main():
    print("Feature config manager")

    while True:
        DEFAULT_CONFIG.print_feature_summary()

        print("\n[Menu]")
        print("1: Toggle feature")
        print("2: Enable all")
        print("3: Disable all")
        print("4: Show enabled counts")
        print("5: Exit")

        choice = input("Select (1-5): ").strip()

        if choice == "5":
            break
        if choice == "1":
            category_input = input("Category name: ").strip()
            feature_input = input("Feature name: ").strip()
            if category_input in DEFAULT_CONFIG.FEATURE_CATEGORIES:
                features = DEFAULT_CONFIG.FEATURE_CATEGORIES[category_input]["features"]
                if feature_input in features:
                    current_status = features[feature_input]["enabled"]
                    if current_status:
                        DEFAULT_CONFIG.disable_feature(category_input, feature_input)
                    else:
                        DEFAULT_CONFIG.enable_feature(category_input, feature_input)
        elif choice == "2":
            for category_name, category_data in DEFAULT_CONFIG.FEATURE_CATEGORIES.items():
                for feature_name in category_data["features"]:
                    DEFAULT_CONFIG.enable_feature(category_name, feature_name)
        elif choice == "3":
            for category_name, category_data in DEFAULT_CONFIG.FEATURE_CATEGORIES.items():
                for feature_name in category_data["features"]:
                    DEFAULT_CONFIG.disable_feature(category_name, feature_name)
        elif choice == "4":
            enabled_by_category = DEFAULT_CONFIG.get_enabled_features_by_category()
            for category_name, features in enabled_by_category.items():
                print(f"{category_name}: {len(features)}")

        DEFAULT_CONFIG.export_config()


if __name__ == "__main__":
    main()
