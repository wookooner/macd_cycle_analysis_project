"""
Pipeline runner for collection, indicator updates, cycle detection, and hierarchy mapping.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_pipeline.utils.io import atomic_write_csv, prune_backup_files
from src.common.paths import PROJECT_PATHS


LOGGER = logging.getLogger("pipeline")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _section(title: str) -> None:
    bar = "=" * 60
    LOGGER.info(bar)
    LOGGER.info("  %s", title)
    LOGGER.info(bar)


def _elapsed(started_at: float) -> str:
    seconds = int(time.time() - started_at)
    return f"{seconds // 60}m {seconds % 60}s" if seconds >= 60 else f"{seconds}s"


@dataclass(frozen=True)
class AssetSpec:
    asset: str
    label: str
    data_files: dict[str, str]
    cycle_dir: Path
    legacy_cycle_dir: Path | None
    funding_rate_file: str | None
    has_cvd: bool
    cvd_rolling: dict[str, int]


def _asset_specs() -> dict[str, AssetSpec]:
    return {
        "btc": AssetSpec(
            asset="btc",
            label="BTC (Binance)",
            data_files={
                "1h": "BTCUSD_1h.csv",
                "4h": "BTCUSD_4h.csv",
                "1d": "BTCUSD_1d.csv",
                "1w": "BTCUSD_1w.csv",
                "1m": "BTCUSD_1m.csv",
            },
            cycle_dir=PROJECT_PATHS.asset_cycle_dir("btc"),
            legacy_cycle_dir=PROJECT_PATHS.cycle_structured_dir,
            funding_rate_file="BTCUSDT_funding_rate.csv",
            has_cvd=True,
            cvd_rolling={"1h": 480, "4h": 120, "1d": 20, "1w": 20, "1m": 20},
        ),
        "gold": AssetSpec(
            asset="gold",
            label="Gold (API Ninjas)",
            data_files={
                "1h": "GOLD_1h.csv",
                "4h": "GOLD_4h.csv",
                "1d": "GOLD_1d.csv",
                "1w": "GOLD_1w.csv",
                "1m": "GOLD_1m.csv",
            },
            cycle_dir=PROJECT_PATHS.asset_cycle_dir("gold"),
            legacy_cycle_dir=None,
            funding_rate_file=None,
            has_cvd=False,
            cvd_rolling={"1h": 480, "4h": 120, "1d": 20, "1w": 20, "1m": 20},
        ),
    }


def log_path_summary() -> None:
    LOGGER.info("Path configuration")
    for key, value in PROJECT_PATHS.summary().items():
        LOGGER.info("  %s: %s", key, value)


def validate_runtime_paths() -> list[str]:
    PROJECT_PATHS.ensure_runtime_dirs()
    issues = PROJECT_PATHS.validate()
    for issue in issues:
        LOGGER.warning("Path validation: %s", issue)
    return issues


def summarize_pipeline_outputs(asset: str) -> dict[str, str]:
    spec = _asset_specs()[asset]
    outputs = {
        "raw_data_dir": str(PROJECT_PATHS.base_data_dir),
        "cycle_dir": str(spec.cycle_dir),
    }
    if spec.legacy_cycle_dir is not None:
        outputs["legacy_cycle_dir"] = str(spec.legacy_cycle_dir)
    return outputs


def step_collect(asset: str, collect_futures: bool = True, dry_run: bool = False) -> bool:
    spec = _asset_specs()[asset]
    _section(f"Step 1 / 4 : collect - {spec.label}")
    started_at = time.time()

    if dry_run:
        LOGGER.info("Dry run: collection skipped.")
        return True

    if asset == "btc":
        from data_pipeline.collectors.new_collcetor import AdvancedBTCDataCollectorV2

        try:
            collector = AdvancedBTCDataCollectorV2()
            LOGGER.info("Updating BTC OHLCV files")
            collector.update_all_ohlcv()
            if collect_futures:
                LOGGER.info("Updating BTC futures data")
                collector.update_all_futures_data()
        except Exception:
            LOGGER.error("BTC collection failed\n%s", traceback.format_exc())
            return False

    elif asset == "gold":
        from data_pipeline.collectors.gold_collector import GoldDataCollector

        try:
            collector = GoldDataCollector()
            if not collector.api_key:
                LOGGER.error("API_NINJAS_KEY is required for gold collection.")
                return False
            LOGGER.info("Updating Gold OHLCV files")
            collector.update_all_ohlcv()
        except Exception:
            LOGGER.error("Gold collection failed\n%s", traceback.format_exc())
            return False

    LOGGER.info("Step 1 complete (%s)", _elapsed(started_at))
    return True


def step_indicator(asset: str, force: bool = False, dry_run: bool = False) -> bool:
    spec = _asset_specs()[asset]
    _section(f"Step 2 / 4 : indicators - {spec.label}")
    started_at = time.time()

    from data_pipeline.indicators.indicator import IndicatorCalculator

    success_count = 0
    fail_count = 0
    calculator = IndicatorCalculator()

    for timeframe, filename in spec.data_files.items():
        file_path = PROJECT_PATHS.base_data_dir / filename
        if not file_path.exists():
            LOGGER.warning("Missing input file: %s", file_path)
            continue

        LOGGER.info(
            "Indicator input: %s (cvd_rolling=%s, has_cvd=%s)",
            file_path.name,
            spec.cvd_rolling.get(timeframe, 20),
            spec.has_cvd,
        )

        if dry_run:
            success_count += 1
            continue

        ok = calculator.process_file(
            file_path,
            force_recalculate=force,
            recalc_last_n=100,
            cvd_rolling_period=spec.cvd_rolling.get(timeframe, 20),
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    if asset == "btc":
        for timeframe in ["1h", "4h", "1d"]:
            filename = spec.data_files.get(timeframe)
            if not filename:
                continue
            file_path = PROJECT_PATHS.base_data_dir / filename
            if not file_path.exists():
                continue
            if dry_run:
                LOGGER.info("Dry run: would merge futures fields into %s", file_path.name)
                continue
            merged = _merge_futures_into_ohlcv(PROJECT_PATHS.base_data_dir, timeframe, filename)
            LOGGER.info("Futures merge %s for %s", "applied" if merged else "skipped", file_path.name)

    LOGGER.info("Indicator result: success=%s, failed=%s", success_count, fail_count)
    LOGGER.info("Step 2 complete (%s)", _elapsed(started_at))
    return fail_count == 0


def _load_funding_rate_if_available(spec: AssetSpec) -> pd.DataFrame | None:
    if not spec.funding_rate_file:
        return None

    funding_rate_path = PROJECT_PATHS.base_data_dir / spec.funding_rate_file
    if not funding_rate_path.exists():
        return None

    from data_pipeline.feature_extractors.macd_historgram_change_feature.feature_extract import (
        StructuredCycleProcessor,
    )

    return StructuredCycleProcessor.load_funding_rate(funding_rate_path)


def step_detect(asset: str, dry_run: bool = False) -> bool:
    spec = _asset_specs()[asset]
    _section(f"Step 3 / 4 : detect - {spec.label}")
    started_at = time.time()

    spec.cycle_dir.mkdir(parents=True, exist_ok=True)

    from data_pipeline.cycle_detectors.macd_histogram_change_detect import (
        detect_cycles_for_timeframe_v3,
        load_algorithm,
    )

    algorithm, config = load_algorithm()
    funding_rate_df = _load_funding_rate_if_available(spec)

    timeframe_files: dict[str, Path] = {}
    for timeframe, filename in spec.data_files.items():
        candidate = PROJECT_PATHS.base_data_dir / filename
        if candidate.exists():
            timeframe_files[timeframe] = candidate
        else:
            LOGGER.warning("Missing timeframe file: %s", candidate)

    if not timeframe_files:
        LOGGER.error("No indicator-enriched input files found for detection.")
        return False

    results: dict[str, int] = {}
    for timeframe, file_path in timeframe_files.items():
        if dry_run:
            LOGGER.info("Dry run: would detect cycles for %s from %s", timeframe, file_path.name)
            results[timeframe] = 0
            continue

        try:
            cycle_records, cycle_count = detect_cycles_for_timeframe_v3(
                file_path,
                timeframe,
                algorithm,
                config,
                funding_rate_df,
            )
            if _save_cycles(cycle_records, timeframe, spec.cycle_dir, spec.legacy_cycle_dir):
                results[timeframe] = cycle_count
        except Exception:
            LOGGER.error("Cycle detection failed for %s\n%s", timeframe, traceback.format_exc())

    LOGGER.info("Detected cycle outputs for %s timeframes", len(results))
    LOGGER.info("Step 3 complete (%s)", _elapsed(started_at))
    return len(results) > 0


def step_map(asset: str, dry_run: bool = False) -> bool:
    spec = _asset_specs()[asset]
    _section(f"Step 4 / 4 : hierarchy map - {spec.label}")
    started_at = time.time()

    if not spec.cycle_dir.exists():
        LOGGER.error("Cycle directory does not exist: %s", spec.cycle_dir)
        return False

    if dry_run:
        LOGGER.info("Dry run: would build hierarchy map in %s", spec.cycle_dir)
        return True

    from data_pipeline.cycle_detectors.cycle_time_mapper import CycleHierarchyMapper

    try:
        mapper = CycleHierarchyMapper(spec.cycle_dir)
        mapper.load_all_cycles()
        mapper.build_hierarchy_map(min_overlap=0.01)
        output_path = mapper.save_hierarchy_map()
        LOGGER.info("Hierarchy map saved to %s", output_path)

        if spec.legacy_cycle_dir and spec.legacy_cycle_dir != spec.cycle_dir:
            spec.legacy_cycle_dir.mkdir(parents=True, exist_ok=True)
            legacy_map = spec.legacy_cycle_dir / "cycle_hierarchy_map.json"
            shutil.copy2(output_path, legacy_map)
            LOGGER.info("Legacy hierarchy copy saved to %s", legacy_map)
    except Exception:
        LOGGER.error("Hierarchy mapping failed\n%s", traceback.format_exc())
        return False

    LOGGER.info("Step 4 complete (%s)", _elapsed(started_at))
    return True


def _save_cycles(
    cycle_records: list,
    timeframe: str,
    output_dir: Path,
    legacy_dir: Path | None = None,
) -> bool:
    if not cycle_records:
        LOGGER.warning("No cycles to save for %s", timeframe)
        return False

    try:
        df = pd.DataFrame(cycle_records)
        if "cycle_features" in df.columns:
            df["cycle_features"] = df["cycle_features"].apply(_prune_empty_features)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"cycles_{timeframe}.parquet"
        df.to_parquet(out_path, index=False)
        LOGGER.info("Saved %s (%s records)", out_path, len(cycle_records))

        if legacy_dir and legacy_dir != output_dir:
            legacy_dir.mkdir(parents=True, exist_ok=True)
            legacy_path = legacy_dir / f"cycles_{timeframe}.parquet"
            shutil.copy2(out_path, legacy_path)
            LOGGER.info("Saved legacy copy %s", legacy_path)

        return True
    except Exception as exc:
        LOGGER.error("Failed to save cycles for %s: %s", timeframe, exc)
        return False


def _prune_empty_features(features: dict) -> dict:
    if not isinstance(features, dict):
        return features
    return {key: value for key, value in features.items() if isinstance(value, dict) and value}


def _merge_futures_into_ohlcv(data_dir: Path, timeframe: str, ohlcv_filename: str) -> bool:
    ohlcv_path = data_dir / ohlcv_filename
    if not ohlcv_path.exists():
        return False

    try:
        df = pd.read_csv(ohlcv_path)
        df["date"] = pd.to_datetime(df["date"])
        changed = False

        oi_contracts_path = data_dir / f"BTCUSDT_oi_contracts_{timeframe}.csv"
        oi_notional_path = data_dir / f"BTCUSDT_oi_notional_{timeframe}.csv"
        oi_legacy_path = data_dir / f"BTCUSDT_oi_{timeframe}.csv"

        oi_cols_to_drop = [
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
        ]
        df = df.drop(columns=[column for column in oi_cols_to_drop if column in df.columns])

        if oi_contracts_path.exists() and oi_notional_path.exists():
            oi_contracts_df = pd.read_csv(
                oi_contracts_path,
                usecols=["date", "oi_contracts", "oi_contracts_change", "oi_contracts_change_pct"],
            )
            oi_contracts_df["date"] = pd.to_datetime(oi_contracts_df["date"])
            oi_contracts_df = oi_contracts_df.drop_duplicates("date").sort_values("date")

            oi_notional_df = pd.read_csv(
                oi_notional_path,
                usecols=["date", "oi_notional", "oi_notional_change", "oi_notional_change_pct"],
            )
            oi_notional_df["date"] = pd.to_datetime(oi_notional_df["date"])
            oi_notional_df = oi_notional_df.drop_duplicates("date").sort_values("date")

            df = df.merge(oi_contracts_df, on="date", how="left")
            df = df.merge(oi_notional_df, on="date", how="left")
            df["oi"] = df["oi_contracts"]
            df["oi_usd"] = df["oi_notional"]
            df["oi_change"] = df["oi_contracts_change"]
            df["oi_change_pct"] = df["oi_contracts_change_pct"]
            changed = True
        elif oi_legacy_path.exists():
            oi_df = pd.read_csv(oi_legacy_path, usecols=["date", "oi", "oi_usd", "oi_change", "oi_change_pct"])
            oi_df["date"] = pd.to_datetime(oi_df["date"])
            oi_df = oi_df.drop_duplicates("date").sort_values("date")
            df = df.merge(oi_df, on="date", how="left")
            df["oi_contracts"] = df["oi"]
            df["oi_contracts_change"] = df["oi_change"]
            df["oi_contracts_change_pct"] = df["oi_change_pct"]
            df["oi_notional"] = df["oi_usd"]
            changed = True

        funding_rate_path = data_dir / "BTCUSDT_funding_rate.csv"
        if funding_rate_path.exists():
            funding_df = pd.read_csv(funding_rate_path, usecols=["date", "funding_rate"])
            funding_df["date"] = pd.to_datetime(funding_df["date"])
            funding_df = funding_df.drop_duplicates("date").sort_values("date")
            df = df.drop(columns=[column for column in ["funding_rate"] if column in df.columns])
            df_sorted = df.sort_values("date").reset_index(drop=True)
            df_sorted = pd.merge_asof(df_sorted, funding_df, on="date", direction="backward")
            sort_col = "unix" if "unix" in df_sorted.columns else "date"
            df = df_sorted.sort_values(sort_col).reset_index(drop=True)
            changed = True

        if not changed:
            return False

        PROJECT_PATHS.backup_data_dir.mkdir(parents=True, exist_ok=True)
        backup_path = PROJECT_PATHS.backup_data_dir / (
            f"{ohlcv_path.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(ohlcv_path, backup_path)
        prune_backup_files(PROJECT_PATHS.backup_data_dir)

        df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_csv(df, ohlcv_path)
        return True
    except Exception:
        LOGGER.error("Futures merge failed for %s\n%s", ohlcv_filename, traceback.format_exc())
        return False


ALL_STEPS = [1, 2, 3, 4]
STEP_NAMES = {1: "collect", 2: "indicator", 3: "detect", 4: "map"}


def run_pipeline(
    asset: str,
    steps: list[int],
    force: bool = False,
    collect_futures: bool = True,
    dry_run: bool = False,
    log_paths: bool = True,
) -> bool:
    _setup_logging()
    total_started_at = time.time()
    spec = _asset_specs()[asset]

    if log_paths:
        log_path_summary()
    validate_runtime_paths()

    LOGGER.info("Pipeline start for %s", spec.label)
    LOGGER.info("Steps: %s", steps)
    LOGGER.info("Force recalculation: %s", force)
    LOGGER.info("Dry run: %s", dry_run)
    LOGGER.info("Outputs: %s", summarize_pipeline_outputs(asset))

    step_functions = {
        1: lambda: step_collect(asset, collect_futures=collect_futures, dry_run=dry_run),
        2: lambda: step_indicator(asset, force=force, dry_run=dry_run),
        3: lambda: step_detect(asset, dry_run=dry_run),
        4: lambda: step_map(asset, dry_run=dry_run),
    }

    results: dict[int, bool] = {}
    for step_num in sorted(steps):
        if step_num not in step_functions:
            LOGGER.warning("Unknown step: %s", step_num)
            continue
        try:
            results[step_num] = step_functions[step_num]()
        except Exception:
            LOGGER.error("Unhandled exception in step %s\n%s", step_num, traceback.format_exc())
            results[step_num] = False

    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline summary for %s", spec.label)
    LOGGER.info("=" * 60)
    all_ok = True
    for step_num in sorted(results):
        ok = results[step_num]
        LOGGER.info("  [%s] Step %s: %s", "OK" if ok else "FAIL", step_num, STEP_NAMES[step_num])
        all_ok = all_ok and ok
    LOGGER.info("Total elapsed: %s", _elapsed(total_started_at))
    return all_ok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MACD cycle data pipeline.")
    parser.add_argument("--asset", choices=["btc", "gold", "all"], default="btc")
    parser.add_argument("--steps", nargs="+", type=int, default=ALL_STEPS, choices=ALL_STEPS)
    parser.add_argument("--force", action="store_true", help="Recalculate indicators from scratch.")
    parser.add_argument("--no-futures", dest="no_futures", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Log planned work without writing outputs.")
    parser.add_argument("--validate-paths-only", action="store_true", help="Only validate configured paths.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging()

    if args.validate_paths_only:
        log_path_summary()
        issues = validate_runtime_paths()
        return 0 if not issues else 1

    assets_to_run = ["btc", "gold"] if args.asset == "all" else [args.asset]
    overall_ok = True

    for asset in assets_to_run:
        ok = run_pipeline(
            asset=asset,
            steps=args.steps,
            force=args.force,
            collect_futures=not args.no_futures,
            dry_run=args.dry_run,
            log_paths=False,
        )
        overall_ok = overall_ok and ok

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
