import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from data_pipeline.collectors.config import DATA_FILES, RAW_DATA_DIR
from data_pipeline.collectors.new_collcetor import AdvancedBTCDataCollectorV2
from data_pipeline.indicators.indicator import IndicatorCalculator
from data_pipeline.pipeline_runner import _merge_futures_into_ohlcv, run_pipeline


LOGGER = logging.getLogger("live_update_service")
PROJECT_ROOT = Path(__file__).resolve().parent
TIMEFRAMES = ["1h", "4h", "1d", "1w"]
FUTURES_TIMEFRAMES = ["1h", "4h", "1d"]


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def align_to_hour_boundary(now_ts: float | None = None) -> float:
    current = datetime.fromtimestamp(now_ts or time.time(), tz=timezone.utc)
    return current.replace(minute=0, second=0, microsecond=0).timestamp()


class LiveUpdateService:
    def __init__(
        self,
        market_interval_seconds: int,
        futures_interval_seconds: int,
        cycle_interval_seconds: int,
        indicator_recalc_rows: int,
    ) -> None:
        self.market_interval_seconds = market_interval_seconds
        self.futures_interval_seconds = futures_interval_seconds
        self.cycle_interval_seconds = cycle_interval_seconds
        self.indicator_recalc_rows = indicator_recalc_rows

        self.collector = AdvancedBTCDataCollectorV2()
        self.indicator_calculator = IndicatorCalculator()

        now = time.time()
        self.next_market_sync_at = now
        self.next_futures_sync_at = now
        self.next_cycle_sync_at = align_to_hour_boundary(now) + cycle_interval_seconds

    def run_forever(self) -> None:
        LOGGER.info("Live update service started.")
        LOGGER.info("Market sync every %ss", self.market_interval_seconds)
        LOGGER.info("Futures sync every %ss", self.futures_interval_seconds)
        LOGGER.info("Cycle sync every %ss", self.cycle_interval_seconds)

        while True:
            now = time.time()
            try:
                if now >= self.next_market_sync_at:
                    self.sync_market_files()
                    self.next_market_sync_at = now + self.market_interval_seconds

                if now >= self.next_futures_sync_at:
                    self.sync_futures_files()
                    self.next_futures_sync_at = now + self.futures_interval_seconds

                if now >= self.next_cycle_sync_at:
                    self.sync_cycles()
                    self.next_cycle_sync_at += self.cycle_interval_seconds
            except Exception:
                LOGGER.exception("Live update loop failed")
                time.sleep(5)

            time.sleep(1)

    def sync_market_files(self) -> None:
        LOGGER.info("Syncing market CSV files")
        for timeframe in TIMEFRAMES:
            self.collector.update_ohlcv(timeframe)
            self.update_indicators_for_timeframe(timeframe)

    def sync_futures_files(self) -> None:
        LOGGER.info("Syncing futures CSV files")
        self.collector.update_funding_rate()
        for timeframe in FUTURES_TIMEFRAMES:
            self.collector.update_oi(timeframe)

        for timeframe in FUTURES_TIMEFRAMES:
            filename = DATA_FILES[timeframe]
            merged = _merge_futures_into_ohlcv(RAW_DATA_DIR, timeframe, filename)
            if merged:
                self.update_indicators_for_timeframe(timeframe)

    def update_indicators_for_timeframe(self, timeframe: str) -> None:
        file_path = RAW_DATA_DIR / DATA_FILES[timeframe]
        if not file_path.exists():
            LOGGER.warning("Missing CSV file for indicator update: %s", file_path.name)
            return

        LOGGER.info("Updating indicators for %s", file_path.name)
        self.indicator_calculator.process_file(
            file_path,
            force_recalculate=False,
            recalc_last_n=self.indicator_recalc_rows,
            cvd_rolling_period=20,
        )

    def sync_cycles(self) -> None:
        LOGGER.info("Running hourly cycle pipeline")
        run_pipeline(
            asset="btc",
            steps=[3, 4],
            force=False,
            collect_futures=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously update dashboard CSV files and run cycle detection hourly.",
    )
    parser.add_argument("--market-interval", type=int, default=15, help="Seconds between OHLCV/indicator refreshes.")
    parser.add_argument("--futures-interval", type=int, default=60, help="Seconds between funding/OI merges.")
    parser.add_argument("--cycle-interval", type=int, default=3600, help="Seconds between cycle recalculations.")
    parser.add_argument("--indicator-recalc-rows", type=int, default=120, help="Trailing rows to recalculate indicators on each refresh.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    service = LiveUpdateService(
        market_interval_seconds=args.market_interval,
        futures_interval_seconds=args.futures_interval,
        cycle_interval_seconds=args.cycle_interval,
        indicator_recalc_rows=args.indicator_recalc_rows,
    )
    service.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
