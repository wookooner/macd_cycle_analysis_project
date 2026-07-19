from __future__ import annotations

import json

import pandas as pd

from data_pipeline.microstructure.io import read_parquet_trees_prefer_first
from data_pipeline.microstructure.paths import public_archive_dir, raw_stream_dir
from data_pipeline.storage.manifests import write_ingestion_manifest
from src.common.paths import PROJECT_PATHS


def test_canonical_microstructure_paths_use_provider_first_layout():
    assert str(raw_stream_dir("BTCUSDT", "agg_trade")).replace("\\", "/").endswith(
        "raw/binance/usdm/BTCUSDT/agg_trade"
    )
    assert str(public_archive_dir()).replace("\\", "/").endswith("archive/binance/futures/um")


def test_compatible_parquet_reader_prefers_canonical_duplicate(tmp_path):
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    for base, value, name in [
        (legacy, 1, "same.parquet"),
        (canonical, 2, "same.parquet"),
        (legacy, 3, "legacy-only.parquet"),
    ]:
        path = base / "date=2026-07-16" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"value": [value]}).to_parquet(path, index=False)

    merged = read_parquet_trees_prefer_first((canonical, legacy))

    assert sorted(merged["value"].tolist()) == [2, 3]


def test_ingestion_manifest_records_row_and_time_bounds(tmp_path):
    original_manifests_root = PROJECT_PATHS.manifests_root
    object.__setattr__(PROJECT_PATHS, "manifests_root", tmp_path / "manifests")
    try:
        manifest = write_ingestion_manifest(
            provider="binance",
            market="usdm",
            symbol="BTCUSDT",
            dataset="agg_trade",
            data_path=tmp_path / "raw" / "date=2026-07-16" / "part-test.parquet",
            rows=pd.DataFrame({"event_time_ns": [1, 2], "price": [1.0, 2.0]}),
            source="test",
        )
    finally:
        object.__setattr__(PROJECT_PATHS, "manifests_root", original_manifests_root)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["row_count"] == 2
    assert payload["min_event_time_ns"] == 1
    assert payload["max_event_time_ns"] == 2
