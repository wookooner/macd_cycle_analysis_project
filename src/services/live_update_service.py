import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipeline.collectors.config import DATA_FILES, RAW_DATA_DIR, resolve_market_file_path
from data_pipeline.collectors.new_collcetor import AdvancedBTCDataCollectorV2
from data_pipeline.indicators.indicator import IndicatorCalculator
from data_pipeline.pipeline_runner import _merge_futures_into_ohlcv, run_pipeline
from src.common.paths import PROJECT_PATHS


LOGGER = logging.getLogger("live_update_service")
PROJECT_ROOT = PROJECT_PATHS.project_root
MARKET_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d", "1w"]
CYCLE_DASHBOARD_TIMEFRAMES = ["15m", "1h", "4h", "1d", "1w"]
TIMEFRAMES = MARKET_TIMEFRAMES
FUTURES_TIMEFRAMES = ["1h", "4h", "1d"]
TIMEFRAME_CVD_ROLLING = {
    "1min": 120,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 480,
    "4h": 120,
    "1d": 20,
    "1w": 20,
    "1M": 20,
}
INDICATOR_COLUMNS = [
    "macd",
    "macd_signal",
    "macd_hist",
    "ppo",
    "ppo_signal",
    "ppo_hist",
    "rsi",
    "stoch_rsi_k",
    "stoch_rsi_d",
]
INDICATOR_WARMUP_ROWS = 200
CYCLE_STALE_MIN_THRESHOLD_SECONDS = 5 * 60
CYCLE_RESYNC_DEBOUNCE_SECONDS = 5 * 60
CYCLE_TIMEFRAME_SECONDS = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
}


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
        run_initial_sync: bool = False,
    ) -> None:
        self.market_interval_seconds = market_interval_seconds
        self.futures_interval_seconds = futures_interval_seconds
        self.cycle_interval_seconds = cycle_interval_seconds
        self.indicator_recalc_rows = indicator_recalc_rows
        self.run_initial_sync = run_initial_sync

        self.collector = AdvancedBTCDataCollectorV2()
        self.indicator_calculator = IndicatorCalculator()
        PROJECT_PATHS.ensure_runtime_dirs()

        now = time.time()
        self.next_market_sync_at = now if run_initial_sync else now + market_interval_seconds
        self.next_futures_sync_at = now if run_initial_sync else now + futures_interval_seconds
        self.next_cycle_sync_at = align_to_hour_boundary(now) + cycle_interval_seconds
        self.last_cycle_sync_at: float | None = None
        if not run_initial_sync and self.cycle_outputs_are_stale():
            LOGGER.info("Cycle outputs are stale; scheduling initial cycle sync immediately")
            self.next_cycle_sync_at = now

    def run_forever(self) -> None:
        LOGGER.info("Live update service started.")
        LOGGER.info("Market sync every %ss", self.market_interval_seconds)
        LOGGER.info("Futures sync every %ss", self.futures_interval_seconds)
        LOGGER.info("Cycle sync every %ss", self.cycle_interval_seconds)
        LOGGER.info("Initial sync enabled: %s", self.run_initial_sync)

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
        self.schedule_cycle_sync_if_stale("market sync")

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
        self.schedule_cycle_sync_if_stale("futures sync")

    def update_indicators_for_timeframe(self, timeframe: str) -> None:
        file_path = resolve_market_file_path(timeframe)
        if not file_path.exists():
            LOGGER.warning("Missing CSV file for indicator update: %s", file_path.name)
            return

        recalc_rows = max(self.indicator_recalc_rows, self.required_indicator_recalc_rows(file_path))
        LOGGER.info("Updating indicators for %s", file_path.name)
        self.indicator_calculator.process_file(
            file_path,
            force_recalculate=False,
            recalc_last_n=recalc_rows,
            cvd_rolling_period=TIMEFRAME_CVD_ROLLING.get(timeframe, 20),
        )

    def required_indicator_recalc_rows(self, file_path: Path) -> int:
        try:
            header = pd.read_csv(file_path, nrows=0).columns
            usecols = [column for column in INDICATOR_COLUMNS if column in header]
            if not usecols:
                return self.indicator_recalc_rows

            df = pd.read_csv(file_path, usecols=usecols)
            if len(df) <= INDICATOR_WARMUP_ROWS:
                return len(df)

            search = df.iloc[INDICATOR_WARMUP_ROWS:]
            missing = search.isna().any(axis=1)
            if not missing.any():
                return self.indicator_recalc_rows

            first_missing_index = missing[missing].index[0]
            required_rows = len(df) - first_missing_index + INDICATOR_WARMUP_ROWS
            LOGGER.info("Expanding indicator recalculation for %s to %s rows", file_path.name, required_rows)
            return required_rows
        except Exception:
            LOGGER.exception("Failed to inspect indicator coverage for %s", file_path.name)
            return self.indicator_recalc_rows

    def sync_cycles(self) -> None:
        LOGGER.info("Running hourly cycle pipeline")
        run_pipeline(
            asset="btc",
            steps=[3, 4],
            force=False,
            collect_futures=True,
            timeframes=CYCLE_DASHBOARD_TIMEFRAMES,
        )
        self.last_cycle_sync_at = time.time()

    def schedule_cycle_sync_if_stale(self, reason: str) -> None:
        now = time.time()
        if self.last_cycle_sync_at and now - self.last_cycle_sync_at < CYCLE_RESYNC_DEBOUNCE_SECONDS:
            return
        if self.cycle_outputs_are_stale():
            LOGGER.info("Cycle outputs became stale after %s; scheduling cycle sync immediately", reason)
            self.next_cycle_sync_at = min(self.next_cycle_sync_at, now)

    def cycle_outputs_are_stale(self) -> bool:
        for timeframe in CYCLE_DASHBOARD_TIMEFRAMES:
            raw_path = resolve_market_file_path(timeframe)
            cycle_path = PROJECT_PATHS.asset_cycle_dir("btc") / f"cycles_{timeframe}.parquet"
            if not raw_path.exists() or not cycle_path.exists():
                return True
            try:
                raw_tail = pd.read_csv(raw_path, usecols=["date"]).tail(1)
                cycle_tail = pd.read_parquet(cycle_path, columns=["end_date"]).tail(1)
                if raw_tail.empty or cycle_tail.empty:
                    return True
                raw_end = pd.to_datetime(raw_tail.iloc[0]["date"], errors="coerce")
                cycle_end = pd.to_datetime(cycle_tail.iloc[0]["end_date"], errors="coerce")
                if pd.isna(raw_end) or pd.isna(cycle_end):
                    return True
                threshold_seconds = max(
                    CYCLE_STALE_MIN_THRESHOLD_SECONDS,
                    CYCLE_TIMEFRAME_SECONDS.get(timeframe, 60 * 60) * 3,
                )
                if (raw_end - cycle_end).total_seconds() > threshold_seconds:
                    LOGGER.info(
                        "Stale cycle output detected: %s raw=%s cycle=%s",
                        timeframe,
                        raw_end,
                        cycle_end,
                    )
                    return True
            except Exception:
                LOGGER.exception("Failed to inspect cycle freshness for %s", timeframe)
                return True
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously update dashboard CSV files and run cycle detection hourly.",
    )
    parser.add_argument("--market-interval", type=int, default=15, help="Seconds between OHLCV/indicator refreshes.")
    parser.add_argument("--futures-interval", type=int, default=60, help="Seconds between funding/OI merges.")
    parser.add_argument("--cycle-interval", type=int, default=3600, help="Seconds between cycle recalculations.")
    parser.add_argument("--indicator-recalc-rows", type=int, default=120, help="Trailing rows to recalculate indicators on each refresh.")
    parser.add_argument("--run-initial-sync", action="store_true", help="Run data sync immediately on startup.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    LOGGER.info("Using shared path config: %s", PROJECT_PATHS.config_path)
    LOGGER.info("Configured data root: %s", PROJECT_PATHS.data_root)
    LOGGER.info("Legacy raw data dir: %s", RAW_DATA_DIR)

    service = LiveUpdateService(
        market_interval_seconds=args.market_interval,
        futures_interval_seconds=args.futures_interval,
        cycle_interval_seconds=args.cycle_interval,
        indicator_recalc_rows=args.indicator_recalc_rows,
        run_initial_sync=args.run_initial_sync,
    )
    service.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
