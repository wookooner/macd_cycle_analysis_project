"""Compatibility wrapper for the consolidated data pipeline package."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.pipeline_runner import _parse_args, run_pipeline


def main() -> int:
    args = _parse_args()
    assets_to_run = ["btc", "gold"] if args.asset == "all" else [args.asset]
    overall_ok = True

    for asset in assets_to_run:
        ok = run_pipeline(
            asset=asset,
            steps=args.steps,
            force=args.force,
            collect_futures=not args.no_futures,
            dry_run=args.dry_run,
            timeframes=args.timeframes,
            microstructure_symbol=args.microstructure_symbol,
            microstructure_timeframe=args.microstructure_timeframe,
            microstructure_batch_rows=args.microstructure_batch_rows,
            microstructure_flush_seconds=args.microstructure_flush_seconds,
            microstructure_keep_files=args.microstructure_keep_files,
            microstructure_rest_poll_seconds=args.microstructure_rest_poll_seconds,
            microstructure_depth_stream=args.microstructure_depth_stream,
            microstructure_depth_levels=args.microstructure_depth_levels,
        )
        if not ok:
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
