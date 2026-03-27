"""
Data Collection Configuration
BTC 데이터 수집을 위한 설정 파일 (Binance API 전용)
"""

import os
from pathlib import Path

# =====================================
# 파일 경로 설정
# =====================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "base_data"
BACKUP_DATA_DIR = DATA_DIR / "backup_data"

# 데이터 파일명 설정
DATA_FILES = {
    '1h': 'BTCUSD_1h.csv',
    '4h': 'BTCUSD_4h.csv', 
    '1d': 'BTCUSD_1d.csv',
    '1w': 'BTCUSD_1w.csv',
    '1m': 'BTCUSD_1m.csv'
}

# =====================================
# Binance API 설정
# =====================================
# Binance 심볼
BINANCE_SYMBOL = "BTCUSDT"

# Binance 타임프레임 매핑
BINANCE_INTERVALS = {
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
    '1w': '1w',
    '1m': '1M'  # Binance는 1M (1 month)
}

# Binance API 키 (환경변수에서 로드, 없어도 Public API 사용 가능)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')

# API 엔드포인트
BINANCE_BASE_URL = "https://api.binance.com"

# =====================================
# 데이터 수집 설정
# =====================================
# 기본 칼럼 (보조지표 제외)
BASE_COLUMNS = ['unix', 'date', 'symbol', 'open', 'high', 'low', 'close', 'Volume USD']

# 전체 칼럼 (보조지표 포함 - 나중에 계산될 예정)
ALL_COLUMNS = ['unix', 'date', 'symbol', 'open', 'high', 'low', 'close', 'Volume USD', 
               'macd', 'macd_signal', 'macd_hist', 'rsi']

# Binance Kline 응답 매핑 (순서대로)
# [Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, 
#  Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore]
BINANCE_KLINE_MAPPING = {
    'open_time': 0,      # 시작 시간 (timestamp)
    'open': 1,           # 시가
    'high': 2,           # 고가  
    'low': 3,            # 저가
    'close': 4,          # 종가
    'volume': 5,         # 거래량 (Base asset)
    'close_time': 6,     # 종료 시간
    'quote_volume': 7,   # 거래량 (Quote asset = USD)
    'trades_count': 8,   # 거래 횟수
    'taker_buy_base': 9, # Taker buy base asset volume
    'taker_buy_quote': 10, # Taker buy quote asset volume  
    'ignore': 11         # Ignore
}

# 데이터 수집 시 재시도 설정
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# API Rate Limiting
REQUEST_DELAY = 0.1  # seconds between requests
MAX_REQUESTS_PER_MINUTE = 1200

# 한 번에 가져올 수 있는 최대 데이터 개수 (Binance 제한)
MAX_LIMIT = 1000

# =====================================
# 백업 설정  
# =====================================
ENABLE_BACKUP = True
BACKUP_BEFORE_UPDATE = True

# =====================================
# 로깅 설정
# =====================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =====================================
# 디렉토리 생성
# =====================================
def ensure_directories():
    """필요한 디렉토리가 없으면 생성"""
    for directory in [DATA_DIR, RAW_DATA_DIR, BACKUP_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("Data collection directories created successfully!")
