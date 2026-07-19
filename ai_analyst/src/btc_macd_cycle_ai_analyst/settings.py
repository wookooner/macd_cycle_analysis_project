from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = APP_ROOT / "configs" / "paths.yaml"


def _load_paths_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        return {}
    return raw


class AnalystSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_ANALYST_",
        env_file=APP_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = Field(default="gpt-3.5-turbo")
    llm_base_url: str = Field(default="http://localhost:1234/v1")
    llm_api_key: str = Field(default="lm-studio")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=8192)
    default_asset: str = Field(default="btc")

    @property
    def app_root(self) -> Path:
        return APP_ROOT

    @property
    def config_path(self) -> Path:
        return CONFIG_PATH

    @property
    def paths_config(self) -> dict[str, Any]:
        return _load_paths_config()

    @property
    def data_root_env_name(self) -> str:
        project = self.paths_config.get("project", {})
        if isinstance(project, dict):
            return str(project.get("data_root_env", "MACD_DATA_ROOT"))
        return "MACD_DATA_ROOT"

    @property
    def default_data_root(self) -> Path:
        project = self.paths_config.get("project", {})
        if isinstance(project, dict) and project.get("default_data_root"):
            return Path(str(project["default_data_root"])).expanduser()
        return Path.home() / "macd-cycle-data"


@lru_cache(maxsize=1)
def get_settings() -> AnalystSettings:
    return AnalystSettings()
