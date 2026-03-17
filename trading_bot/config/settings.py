"""
trading_bot/config/settings.py
===============================
전역 설정 관리.

핵심: Settings()는 .env 로드 후에만 생성.
     load_settings()를 통해서만 인스턴스화.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

# 봇 루트 경로 (이 파일 기준으로 고정 — 실행 위치 무관)
BOT_ROOT = Path(__file__).parent.parent.resolve()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ("true", "1", "yes")


@dataclass
class PathConfig:
    analysis_project_root: Path = field(
        default_factory=lambda: Path(
            _env("ANALYSIS_PROJECT_ROOT",
                 r"C:\Users\Administrator\Desktop\macd_cycle_analysis_project")
        )
    )

    @property
    def parquet_dir(self) -> Path:
        return self.analysis_project_root / "data" / "cycle_data" / "structured"

    @property
    def hierarchy_map_path(self) -> Path:
        return self.parquet_dir / "cycle_hierarchy_map.json"

    @property
    def update_pipeline_path(self) -> Path:
        return self.analysis_project_root / "update_pipeline.py"

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
        """AI에게 매 호출 전 읽힐 규칙 문서"""
        return self.bot_dir / "config" / "trading_rules.md"


@dataclass
class BinanceConfig:
    api_key: str = field(default_factory=lambda: _env("BINANCE_API_KEY"))
    api_secret: str = field(default_factory=lambda: _env("BINANCE_API_SECRET"))
    symbol: str = "BTC/USDT:USDT"
    leverage: int = field(default_factory=lambda: _env_int("BINANCE_LEVERAGE", 5))
    testnet: bool = field(default_factory=lambda: _env_bool("BINANCE_TESTNET", True))


@dataclass
class TelegramConfig:
    bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))
    approval_timeout_minutes: int = 30


@dataclass
class AIConfig:
    api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.0


@dataclass
class TradingConfig:
    max_position_pct: float = field(
        default_factory=lambda: _env_float("MAX_POSITION_PCT", 30.0)
    )
    daily_max_loss_pct: float = field(
        default_factory=lambda: _env_float("DAILY_MAX_LOSS_PCT", 5.0)
    )
    max_consecutive_losses: int = 3
    min_duration_for_trade: int = 5
    candle_wait_seconds: int = 15
    dry_run: bool = field(
        default_factory=lambda: _env_bool("DRY_RUN", True)
    )


@dataclass
class SchedulerConfig:
    # 매 시간 55분 실행 (정각 5분 전 — 파이프라인 소요시간 감안)
    cron_minute: int = 55
    cron_second: int = 0


@dataclass
class Settings:
    paths: PathConfig = field(default_factory=PathConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    timeframes: list = field(
        default_factory=lambda: ["1w", "1d", "4h", "1h"]
    )

    def validate(self) -> list[str]:
        missing = []
        if not self.binance.api_key:
            missing.append("BINANCE_API_KEY")
        if not self.binance.api_secret:
            missing.append("BINANCE_API_SECRET")
        if not self.telegram.bot_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.telegram.chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        if not self.ai.api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.paths.analysis_project_root.exists():
            missing.append(
                f"ANALYSIS_PROJECT_ROOT 경로 없음: {self.paths.analysis_project_root}"
            )
        return missing


def load_settings() -> Settings:
    """
    .env → 환경변수 로드 → Settings 생성.
    반드시 이 함수를 통해 Settings를 생성해야 합니다.
    """
    env_path = BOT_ROOT / ".env"
    try:
        from dotenv import load_dotenv
        if env_path.exists():
            load_dotenv(env_path, override=True)
    except ImportError:
        pass

    return Settings()
