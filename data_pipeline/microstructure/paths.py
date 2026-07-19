from __future__ import annotations

from pathlib import Path

from data_pipeline.storage.layout import archive_dataset_dir, raw_dataset_dir
from src.common.paths import PROJECT_PATHS


# New writes use the provider-first layout below.  The legacy root remains
# readable while existing data is migrated; no collector writes to it.
MICROSTRUCTURE_ROOT = PROJECT_PATHS.raw_root / "binance" / "usdm"
LEGACY_MICROSTRUCTURE_ROOT = PROJECT_PATHS.raw_root / "microstructure" / "binance_usdm"
BINANCE_PUBLIC_ARCHIVE_ROOT = archive_dataset_dir("binance", "futures", "um")
LEGACY_BINANCE_PUBLIC_ARCHIVE_ROOT = PROJECT_PATHS.raw_root / "binance_public_data"
MICROSTRUCTURE_FEATURE_ROOT = PROJECT_PATHS.processed_features_dir / "microstructure"
MICROSTRUCTURE_REPORT_ROOT = PROJECT_PATHS.outputs_root / "analysis_results" / "microstructure_validation"


def raw_stream_dir(symbol: str, stream: str) -> Path:
    """Canonical destination for a Binance USD-M normalized stream."""
    return raw_dataset_dir("binance", "usdm", symbol, stream)


def legacy_raw_stream_dir(symbol: str, stream: str) -> Path:
    """Pre-layout destination retained for read compatibility only."""
    return LEGACY_MICROSTRUCTURE_ROOT / symbol.upper() / stream


def raw_stream_read_dirs(symbol: str, stream: str) -> tuple[Path, ...]:
    """Return canonical then legacy locations, omitting duplicate paths."""
    canonical = raw_stream_dir(symbol, stream)
    legacy = legacy_raw_stream_dir(symbol, stream)
    return (canonical,) if canonical == legacy else (canonical, legacy)


def public_archive_dir(market: str = "futures", contract: str = "um") -> Path:
    if market == "futures" and contract == "um":
        return BINANCE_PUBLIC_ARCHIVE_ROOT
    return archive_dataset_dir("binance", market, contract)


def legacy_public_archive_dir(market: str = "futures", contract: str = "um") -> Path:
    return LEGACY_BINANCE_PUBLIC_ARCHIVE_ROOT / market / contract


def feature_dir(symbol: str) -> Path:
    return MICROSTRUCTURE_FEATURE_ROOT / symbol.upper()


def report_dir(symbol: str) -> Path:
    return MICROSTRUCTURE_REPORT_ROOT / symbol.upper()
