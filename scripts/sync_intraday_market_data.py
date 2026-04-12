from __future__ import annotations

from data_pipeline.collectors.config import INTRADAY_SOURCE_FILES, RAW_DATA_DIR, DATA_FILES
from data_pipeline.collectors.new_collcetor import AdvancedBTCDataCollectorV2
from data_pipeline.indicators.indicator import IndicatorCalculator
from src.common.paths import PROJECT_PATHS


def main() -> int:
    PROJECT_PATHS.ensure_runtime_dirs()

    collector = AdvancedBTCDataCollectorV2()
    calculator = IndicatorCalculator()

    total = len(INTRADAY_SOURCE_FILES)
    for index, timeframe in enumerate(INTRADAY_SOURCE_FILES.keys(), start=1):
        file_name = DATA_FILES[timeframe]
        print(f"[progress {index}/{total} {index / total * 100:.1f}%] syncing {timeframe}")
        collector.update_ohlcv(timeframe)
        file_path = RAW_DATA_DIR / file_name
        if not file_path.exists():
            print(f"[skip] missing normalized file: {file_path}")
            continue

        ok = calculator.process_file(
            file_path,
            force_recalculate=False,
            recalc_last_n=500,
            cvd_rolling_period=120 if timeframe == "1min" else 60,
        )
        print(f"[{'ok' if ok else 'fail'}] {timeframe} -> {file_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
