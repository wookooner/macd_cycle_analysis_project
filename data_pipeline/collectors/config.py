"""
Data collection configuration.

This module keeps legacy file locations working while the repository moves
toward shared path configuration in `configs/paths.yaml`.
"""

import os

from src.common.paths import PROJECT_PATHS

# =====================================
# File system paths
# =====================================
BASE_DIR = PROJECT_PATHS.project_root
DATA_DIR = PROJECT_PATHS.legacy_data_root
RAW_DATA_DIR = PROJECT_PATHS.base_data_dir
BACKUP_DATA_DIR = PROJECT_PATHS.backup_data_dir

# Market source filenames
DATA_FILES = {
    "1h": "BTCUSD_1h.csv",
    "4h": "BTCUSD_4h.csv",
    "1d": "BTCUSD_1d.csv",
    "1w": "BTCUSD_1w.csv",
    "1m": "BTCUSD_1m.csv",
}

# =====================================
# Binance API settings
# =====================================
BINANCE_SYMBOL = "BTCUSDT"

BINANCE_INTERVALS = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
    "1m": "1M",
}

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
BINANCE_BASE_URL = "https://api.binance.com"

# =====================================
# Collection settings
# =====================================
BASE_COLUMNS = ["unix", "date", "symbol", "open", "high", "low", "close", "Volume USD"]

ALL_COLUMNS = [
    "unix",
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "Volume USD",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi",
]

BINANCE_KLINE_MAPPING = {
    "open_time": 0,
    "open": 1,
    "high": 2,
    "low": 3,
    "close": 4,
    "volume": 5,
    "close_time": 6,
    "quote_volume": 7,
    "trades_count": 8,
    "taker_buy_base": 9,
    "taker_buy_quote": 10,
    "ignore": 11,
}

MAX_RETRIES = 3
RETRY_DELAY = 5
REQUEST_DELAY = 0.1
MAX_REQUESTS_PER_MINUTE = 1200
MAX_LIMIT = 1000

# =====================================
# Backup and logging
# =====================================
ENABLE_BACKUP = True
BACKUP_BEFORE_UPDATE = True
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def ensure_directories() -> None:
    """Create required legacy-compatible directories."""
    PROJECT_PATHS.ensure_runtime_dirs()
    for directory in [DATA_DIR, RAW_DATA_DIR, BACKUP_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("Data collection directories created successfully.")
