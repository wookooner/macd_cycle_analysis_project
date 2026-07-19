from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.microstructure.binance_public_data import DATASETS, DEFAULT_DATASETS, _parse_date, backfill_public_data
from data_pipeline.microstructure.features import build_microstructure_features, feature_dir, normalize_timeframe


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Binance USD-M public data and build microstructure features in one run."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--datasets", nargs="+", default=["all"], help=f"Datasets or all. Options: {', '.join(DATASETS)}")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default="1m", help="Kline interval for klines/markPriceKlines/premiumIndexKlines.")
    parser.add_argument("--backfill-period", choices=["auto", "daily", "monthly"], default="auto")
    parser.add_argument("--timeframe", default="1min", help="Feature timeframe, e.g. 1min, 5min, 15min.")
    parser.add_argument("--aux-tolerance", default="10min")
    parser.add_argument("--funding-tolerance", default="9h")
    parser.add_argument("--close-tolerance", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List backfill objects without downloading or building features.")
    parser.add_argument("--skip-backfill", action="store_true", help="Build features from existing raw parquet only.")
    parser.add_argument("--skip-features", action="store_true", help="Only download/convert raw parquet; do not build features.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    datasets = DEFAULT_DATASETS if args.datasets == ["all"] else tuple(args.datasets)
    unknown = sorted(set(datasets).difference(DATASETS))
    if unknown:
        LOGGER.error("Unknown datasets: %s", ", ".join(unknown))
        return 2

    try:
        if not args.skip_backfill:
            results = backfill_public_data(
                symbol=args.symbol,
                datasets=datasets,
                start=_parse_date(args.start),
                end=_parse_date(args.end),
                interval=args.interval,
                period=args.backfill_period,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                LOGGER.info("Backfill converted %s archive objects.", len(results))

        if args.dry_run or args.skip_features:
            return 0

        timeframe = normalize_timeframe(args.timeframe)
        df = build_microstructure_features(
            symbol=args.symbol,
            timeframe=timeframe,
            aux_tolerance=args.aux_tolerance,
            funding_tolerance=args.funding_tolerance,
            close_tolerance=args.close_tolerance,
        )
        if df.empty:
            raise RuntimeError(f"No microstructure rows found for {args.symbol} {timeframe}")

        out_dir = feature_dir(args.symbol)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"microstructure_features_{timeframe}.parquet"
        df.to_parquet(out_path, index=False)
        LOGGER.info("Saved microstructure features: rows=%s path=%s", len(df), out_path)
    except Exception as exc:
        LOGGER.error("Integrated microstructure run failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
