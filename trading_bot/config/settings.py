"""
Central configuration for the trading bot.

Key goals:
- keep runtime modes explicit
- decouple the bot from the data production service by default
- make future Telegram/chat-driven control easier to extend
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


BOT_ROOT = Path(__file__).parent.parent.resolve()
REPO_ROOT = BOT_ROOT.parent.resolve()
DEFAULT_MACD_DATA_ROOT = Path(r"C:\Users\qw370\macd-cycle-data")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ("true", "1", "yes")


class BotMode(str, Enum):
    ANALYZE = "analyze"
    APPROVE = "approve"
    AUTO = "auto"


class DataSourceMode(str, Enum):
    EXTERNAL = "external"
    PIPELINE = "pipeline"


def _env_enum(key: str, enum_cls, default):
    raw = os.environ.get(key, default.value if hasattr(default, "value") else str(default)).lower()
    for member in enum_cls:
        if member.value == raw:
            return member
    return default


@dataclass
class PathConfig:
    analysis_project_root: Path = field(
        default_factory=lambda: Path(_env("ANALYSIS_PROJECT_ROOT", str(REPO_ROOT)))
    )
    macd_data_root: Path = field(
        default_factory=lambda: Path(_env("MACD_DATA_ROOT", str(DEFAULT_MACD_DATA_ROOT)))
    )

    @property
    def legacy_data_root(self) -> Path:
        return self.analysis_project_root / "data"

    @property
    def parquet_dir(self) -> Path:
        override = _env("ANALYSIS_PARQUET_DIR")
        return Path(override) if override else self.legacy_data_root / "cycle_data" / "structured"

    @property
    def hierarchy_map_path(self) -> Path:
        override = _env("ANALYSIS_HIERARCHY_MAP_PATH")
        return Path(override) if override else self.parquet_dir / "cycle_hierarchy_map.json"

    @property
    def base_data_dir(self) -> Path:
        override = _env("ANALYSIS_BASE_DATA_DIR")
        return Path(override) if override else self.legacy_data_root / "base_data"

    @property
    def update_pipeline_path(self) -> Path:
        override = _env("UPDATE_PIPELINE_PATH")
        return Path(override) if override else self.analysis_project_root / "update_pipeline.py"

    @property
    def bot_dir(self) -> Path:
        return BOT_ROOT

    @property
    def bot_db_path(self) -> Path:
        return self.bot_dir / "bot_data.db"

    @property
    def log_dir(self) -> Path:
        return self.bot_dir / "logs"

    @property
    def rules_doc_path(self) -> Path:
        return self.bot_dir / "config" / "trading_rules.md"


@dataclass
class BinanceConfig:
    api_key: str = field(default_factory=lambda: _env("BINANCE_API_KEY"))
    api_secret: str = field(default_factory=lambda: _env("BINANCE_API_SECRET"))
    symbol: str = field(default_factory=lambda: _env("BINANCE_SYMBOL", "BTC/USDT:USDT"))
    leverage: int = field(default_factory=lambda: _env_int("BINANCE_LEVERAGE", 5))
    testnet: bool = field(default_factory=lambda: _env_bool("BINANCE_TESTNET", True))


@dataclass
class TelegramConfig:
    bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))
    approval_timeout_minutes: int = field(default_factory=lambda: _env_int("TELEGRAM_APPROVAL_TIMEOUT_MINUTES", 30))


@dataclass
class AIConfig:
    api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    model: str = field(default_factory=lambda: _env("AI_MODEL", "claude-sonnet-4-20250514"))
    max_tokens: int = field(default_factory=lambda: _env_int("AI_MAX_TOKENS", 1024))
    temperature: float = field(default_factory=lambda: _env_float("AI_TEMPERATURE", 0.0))


@dataclass
class TradingConfig:
    mode: BotMode = field(default_factory=lambda: _env_enum("BOT_MODE", BotMode, BotMode.ANALYZE))
    use_telegram: bool = field(default_factory=lambda: _env_bool("USE_TELEGRAM", True))
    data_source_mode: DataSourceMode = field(
        default_factory=lambda: _env_enum("DATA_SOURCE_MODE", DataSourceMode, DataSourceMode.EXTERNAL)
    )
    max_position_pct: float = field(default_factory=lambda: _env_float("MAX_POSITION_PCT", 30.0))
    daily_max_loss_pct: float = field(default_factory=lambda: _env_float("DAILY_MAX_LOSS_PCT", 5.0))
    max_consecutive_losses: int = field(default_factory=lambda: _env_int("MAX_CONSECUTIVE_LOSSES", 3))
    min_duration_for_trade: int = field(default_factory=lambda: _env_int("MIN_DURATION_FOR_TRADE", 5))
    candle_wait_seconds: int = field(default_factory=lambda: _env_int("CANDLE_WAIT_SECONDS", 15))
    dry_run: bool = field(default_factory=lambda: _env_bool("DRY_RUN", True))


@dataclass
class SchedulerConfig:
    cron_minute: int = field(default_factory=lambda: _env_int("CRON_MINUTE", 0))
    cron_second: int = field(default_factory=lambda: _env_int("CRON_SECOND", _env_int("CANDLE_WAIT_SECONDS", 15)))


@dataclass
class Settings:
    paths: PathConfig = field(default_factory=PathConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    timeframes: list[str] = field(default_factory=lambda: ["1w", "1d", "4h", "1h"])

    def validate(self) -> list[str]:
        missing = []

        if not self.paths.analysis_project_root.exists():
            missing.append(f"ANALYSIS_PROJECT_ROOT not found: {self.paths.analysis_project_root}")
        if not self.paths.macd_data_root.exists():
            missing.append(f"MACD_DATA_ROOT not found: {self.paths.macd_data_root}")

        if self.trading.mode in (BotMode.APPROVE, BotMode.AUTO):
            if not self.binance.api_key:
                missing.append("BINANCE_API_KEY")
            if not self.binance.api_secret:
                missing.append("BINANCE_API_SECRET")

        if self.trading.use_telegram:
            if not self.telegram.bot_token:
                missing.append("TELEGRAM_BOT_TOKEN")
            if not self.telegram.chat_id:
                missing.append("TELEGRAM_CHAT_ID")

        return missing


def load_settings() -> Settings:
    env_path = BOT_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        if env_path.exists():
            load_dotenv(env_path, override=True)
    except ImportError:
        pass

    return Settings()


settings = load_settings()
