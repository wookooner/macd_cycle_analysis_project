from __future__ import annotations

"""Canonical paths for provider data.

The layout deliberately encodes provider, market and symbol before a dataset:
``raw/<provider>/<market>/<symbol>/<dataset>``.  This makes adding a new
exchange, contract type, symbol or stream additive rather than a special case.
"""

from pathlib import Path

from src.common.paths import PROJECT_PATHS


def _component(value: str) -> str:
    """Return a path-safe, stable dataset component."""
    normalized = str(value).strip().lower().replace(" ", "_")
    if not normalized or normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError(f"Invalid storage path component: {value!r}")
    return normalized


def _symbol(value: str) -> str:
    normalized = _component(value).upper()
    return normalized


def raw_dataset_dir(provider: str, market: str, symbol: str, dataset: str, *parts: str) -> Path:
    """Return a canonical normalized raw-data directory."""
    return PROJECT_PATHS.raw_root.joinpath(
        _component(provider),
        _component(market),
        _symbol(symbol),
        _component(dataset),
        *(_component(part) for part in parts),
    )


def archive_dataset_dir(provider: str, market: str, *parts: str) -> Path:
    """Return a provider-original archive directory, separate from raw data."""
    return PROJECT_PATHS.archive_root.joinpath(
        _component(provider),
        _component(market),
        *(_component(part) for part in parts),
    )
