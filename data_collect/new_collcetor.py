# C:\Users\Administrator\Desktop\macd_cycle_analysis_project\data_collect\advanced_collector.py

import pandas as pd
from datetime import datetime, timezone
import time
from pathlib import Path
import logging
import shutil

# Binance 공식 커넥터 사용
from binance.spot import Spot as Client
from binance.error import ClientError

# 기존 config.py 파일의 설정을 그대로 사용합니다.
from config import *

class AdvancedBTCDataCollector:
    """
    시간대(UTC) 문제를 해결하고 안정성을 높인 신규 BTC 데이터 수집기.
    - 모든 시간 처리를 UTC 기준으로 통일
    - API 요청 제한(1000개)에 맞춰 데이터 분할 요청 (Pagination) 기능 구현
    - 최신 Binance 공식 라이브러리 사용
    """

    def __init__(self):
        """초기화 메서드"""
        self.logger = self._setup_logging()
        ensure_directories()  # config.py의 디렉토리 생성 함수
        self.client = self._setup_binance_client()

    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
        return logging.getLogger(__name__)

    def _setup_binance_client(self) -> Client | None:
        """Binance 클라이언트 초기화"""
        try:
            # API 키 유무에 따라 인증된 클라이언트 또는 Public 클라이언트 생성
            client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
            # 연결 테스트
            client.ping()
            self.logger.info("✅ Binance 클라이언트 초기화 및 연결 테스트 완료")
            return client
        except Exception as e:
            self.logger.error(f"❌ Binance 클라이언트 초기화 실패: {e}")
            return None

    def _fetch_binance_data(self, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
        """
        Binance API로부터 Klines 데이터를 안정적으로 가져옵니다. (Pagination 지원)
        """
        if self.client is None:
            self.logger.error("Binance 클라이언트가 초기화되지 않았습니다.")
            return None

        symbol = BINANCE_SYMBOL
        interval = BINANCE_INTERVALS[timeframe]
        
        # datetime 객체를 millisecond 단위의 Unix Timestamp로 변환
        start_ts_ms = int(start_dt.timestamp() * 1000)
        end_ts_ms = int(end_dt.timestamp() * 1000)

        self.logger.info(f"데이터 수집 시작: {symbol} ({interval})")
        self.logger.info(f"요청 기간: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ~ {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")

        all_klines = []
        fetch_start_ts = start_ts_ms
        
        while True:
            try:
                self.logger.debug(f"Fetching from {datetime.fromtimestamp(fetch_start_ts / 1000, tz=timezone.utc)}")
                klines = self.client.klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=fetch_start_ts,
                    endTime=end_ts_ms,
                    limit=MAX_LIMIT
                )
                
                if not klines:
                    self.logger.info("더 이상 가져올 데이터가 없습니다. 수집을 종료합니다.")
                    break

                all_klines.extend(klines)
                
                # 다음 요청 시작 시간을 마지막으로 받은 데이터의 시간+1ms로 설정
                last_kline_ts = klines[-1][BINANCE_KLINE_MAPPING['open_time']]
                fetch_start_ts = last_kline_ts + 1

                self.logger.info(f"데이터 {len(klines)}개 수집. 다음 시작: {datetime.fromtimestamp(fetch_start_ts / 1000, tz=timezone.utc)}")
                
                # API Rate Limiting을 위한 지연
                time.sleep(REQUEST_DELAY)

            except ClientError as e:
                self.logger.error(f"Binance API 오류 발생: {e.status_code} - {e.error_message}")
                return None
            except Exception as e:
                self.logger.error(f"데이터 수집 중 예외 발생: {e}")
                return None
        
        if not all_klines:
            return pd.DataFrame() # 빈 데이터프레임 반환

        return self._format_klines(all_klines)

    def _format_klines(self, klines_data: list) -> pd.DataFrame:
        """Binance Klines 데이터를 프로젝트 형식에 맞는 DataFrame으로 변환"""
        cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_volume', 'trades_count', 'taker_buy_base', 
            'taker_buy_quote', 'ignore'
        ]
        df = pd.DataFrame(klines_data, columns=cols)

        # 숫자형 데이터로 변환
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # 'unix'와 'date' 컬럼 생성 (UTC 기준)
        df['unix'] = df['open_time'] // 1000
        df['date'] = pd.to_datetime(df['unix'], unit='s', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')

        # 프로젝트에서 사용하는 컬럼명으로 변경
        df.rename(columns={'quote_volume': 'Volume USD'}, inplace=True)
        
        # 최종 컬럼 선택 및 순서 정리
        final_df = df[[
            'unix', 'date', 'open', 'high', 'low', 'close', 'Volume USD'
        ]].copy()
        
        # symbol 컬럼 추가
        final_df['symbol'] = 'BTCUSD'
        
        # 보조지표 컬럼 추가 (값은 나중에 채워짐)
        for col in ['macd', 'macd_signal', 'macd_hist', 'rsi']:
            final_df[col] = pd.NA

        self.logger.info(f"데이터 포맷팅 완료: {len(final_df)} rows")
        return final_df

    def _merge_and_save_data(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, file_path: Path):
        """기존 데이터와 신규 데이터를 병합하고 중복을 제거한 후 저장"""
        if new_df is None or new_df.empty:
            self.logger.info("추가할 새로운 데이터가 없습니다.")
            return

        # 1. 데이터 백업
        if ENABLE_BACKUP and file_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{file_path.name}.backup_{timestamp}"
            backup_path = BACKUP_DATA_DIR / backup_filename
            try:
                shutil.copy2(file_path, backup_path)
                self.logger.info(f"백업 완료: {backup_path}")
            except Exception as e:
                self.logger.error(f"백업 실패: {e}")
        
        # 2. 데이터 병합
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # 3. 중복 제거 (unix timestamp 기준, 최신 데이터 유지)
        initial_rows = len(combined_df)
        combined_df.drop_duplicates(subset=['unix'], keep='last', inplace=True)
        final_rows = len(combined_df)
        self.logger.info(f"중복 데이터 {initial_rows - final_rows}개 제거 완료.")
        
        # 4. 정렬
        combined_df.sort_values(by='unix', inplace=True)
        
        # 5. 저장
        try:
            combined_df.to_csv(file_path, index=False)
            self.logger.info(f"✅ 데이터 저장 완료: {file_path} ({final_rows} rows)")
        except Exception as e:
            self.logger.error(f"데이터 저장 실패: {e}")


    def update_timeframe_data(self, timeframe: str):
        """특정 타임프레임의 데이터를 업데이트합니다."""
        self.logger.info(f"========== {timeframe} 데이터 업데이트 시작 ==========")
        
        file_path = RAW_DATA_DIR / DATA_FILES[timeframe]
        
        existing_df = pd.DataFrame()
        start_fetch_dt: datetime

        if file_path.exists():
            try:
                existing_df = pd.read_csv(file_path)
                last_unix_time = existing_df['unix'].iloc[-1]
                # 마지막 시간부터 가져와서 진행중인 캔들도 업데이트
                start_fetch_dt = datetime.fromtimestamp(last_unix_time, tz=timezone.utc)
                self.logger.info(f"기존 데이터 로드 완료. 마지막 데이터 시점: {start_fetch_dt}")
            except Exception as e:
                self.logger.error(f"기존 데이터 로드 실패: {e}. 처음부터 다시 수집합니다.")
                # 파일에 문제가 있으면 아주 오래전부터 다시 수집
                start_fetch_dt = datetime(2017, 1, 1, tzinfo=timezone.utc)
        else:
            self.logger.info("기존 데이터 파일이 없습니다. 2017년부터 전체 데이터를 수집합니다.")
            start_fetch_dt = datetime(2017, 1, 1, tzinfo=timezone.utc)
            
        # 데이터 수집 종료 시점은 항상 현재 UTC 시간
        end_fetch_dt = datetime.now(timezone.utc)
        
        # 신규 데이터 가져오기
        new_data_df = self._fetch_binance_data(timeframe, start_fetch_dt, end_fetch_dt)
        
        # 병합 및 저장
        self._merge_and_save_data(existing_df, new_data_df, file_path)
        self.logger.info(f"========== {timeframe} 데이터 업데이트 완료 ==========\n")


    def update_all_timeframes(self):
        """설정된 모든 타임프레임의 데이터를 순차적으로 업데이트합니다."""
        self.logger.info("🚀 전체 타임프레임 데이터 업데이트를 시작합니다.")
        for tf in DATA_FILES.keys():
            self.update_timeframe_data(tf)
        self.logger.info("🎉 모든 타임프레임 데이터 업데이트가 완료되었습니다.")


if __name__ == "__main__":
    collector = AdvancedBTCDataCollector()
    
    print("🚀 고급 BTC 데이터 수집기 (Binance API)")
    print("1. 전체 타임프레임 업데이트")
    print("2. 특정 타임프레임 업데이트")
    
    choice = input("원하는 작업을 선택하세요 (1/2): ").strip()
    
    if choice == "1":
        collector.update_all_timeframes()
    elif choice == "2":
        tf_list = list(DATA_FILES.keys())
        print(f"사용 가능한 타임프레임: {tf_list}")
        timeframe_choice = input("업데이트할 타임프레임을 입력하세요: ").strip()
        if timeframe_choice in tf_list:
            collector.update_timeframe_data(timeframe_choice)
        else:
            print("❌ 잘못된 타임프레임입니다.")
    else:
        print("❌ 잘못된 선택입니다.")