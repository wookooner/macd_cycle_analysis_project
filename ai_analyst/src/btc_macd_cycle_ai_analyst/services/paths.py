from __future__ import annotations

import os
from pathlib import Path

from ..settings import get_settings


class AnalystPaths:
    """Resolve canonical data locations for the analysis app."""

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def app_root(self) -> Path:
        return self.settings.app_root

    @property
    def data_root(self) -> Path:
        env_name = self.settings.data_root_env_name
        env_value = os.getenv(env_name)
        if env_value:
            return Path(env_value).expanduser()
        return self.settings.default_data_root

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    def cycles_enriched_dir(self, asset: str | None = None) -> Path:
        resolved_asset = asset or self.settings.default_asset
        return self.processed_root / "cycles_enriched" / resolved_asset

    def context_dir(self, asset: str | None = None) -> Path:
        resolved_asset = asset or self.settings.default_asset
        return self.processed_root / "context" / resolved_asset

    def cycle_file(self, timeframe: str, asset: str | None = None) -> Path:
        return self.cycles_enriched_dir(asset=asset) / f"cycles_{timeframe}.parquet"

    def context_file(self, timeframe: str, asset: str | None = None) -> Path:
        return self.context_dir(asset=asset) / f"timeframe_context_{timeframe}.parquet"
