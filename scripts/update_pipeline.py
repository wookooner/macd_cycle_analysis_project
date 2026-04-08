"""Compatibility wrapper for the consolidated data pipeline package."""

import sys

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
        )
        if not ok:
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
