"""
BTC 데이터 수집기 (Binance API 전용)
기존 데이터 확인 후 누락된 부분 업데이트
주간/월간 캔들의 진행중인 캔들 업데이트 지원
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pathlib import Path
import logging
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import *

class BTCDataCollector:
    def __init__(self):
        self.setup_logging()
        ensure_directories()
        self.setup_binance_client()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
        self.logger = logging.getLogger(__name__)
        
    def setup_binance_client(self):
        """Binance 클라이언트 초기화"""
        try:
            # API 키가 있으면 인증된 클라이언트, 없으면 Public API만 사용
            if BINANCE_API_KEY and BINANCE_SECRET_KEY:
                self.client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
                self.logger.info("Binance 인증 클라이언트 초기화 완료")
            else:
                self.client = Client()  # Public API만 사용
                self.logger.info("Binance Public API 클라이언트 초기화 완료")
                
        except Exception as e:
            self.logger.error(f"Binance 클라이언트 초기화 실패: {str(e)}")
            self.client = None
        
    def get_existing_data(self, timeframe):
        """기존 데이터 파일 확인 및 로드"""
        file_path = RAW_DATA_DIR / DATA_FILES[timeframe]
        
        if not file_path.exists():
            self.logger.info(f"기존 데이터 파일이 없습니다: {DATA_FILES[timeframe]}")
            return None
            
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"기존 데이터 로드 완료: {len(df)} rows")
            return df
        except Exception as e:
            self.logger.error(f"기존 데이터 로드 실패: {str(e)}")
            return None
    
    def get_last_date(self, df):
        """마지막 데이터 날짜 확인"""
        if df is None or len(df) == 0:
            return None
            
        # date 컬럼에서 마지막 날짜 추출
        last_date_str = df['date'].iloc[-1]
        try:
            last_date = pd.to_datetime(last_date_str)
            return last_date
        except:
            self.logger.error(f"날짜 파싱 실패: {last_date_str}")
            return None
    
    def fetch_new_data(self, timeframe, start_date=None, end_date=None):
        """Binance API를 사용하여 새로운 데이터 수집"""
        if self.client is None:
            self.logger.error("Binance 클라이언트가 초기화되지 않았습니다.")
            return None
            
        interval = BINANCE_INTERVALS[timeframe]
        self.logger.info(f"데이터 수집 시작: {BINANCE_SYMBOL}, {interval}")
        
        try:
            # 시작/끝 시간을 timestamp로 변환 (Binance는 milliseconds 사용)
            start_str = None
            end_str = None
            
            if start_date:
                start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
                start_str = str(start_timestamp)
                
            if end_date:
                end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)
                end_str = str(end_timestamp)
            
            # Binance Klines 데이터 수집
            klines = self.client.get_historical_klines(
                symbol=BINANCE_SYMBOL,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=MAX_LIMIT
            )
            
            if not klines:
                self.logger.warning("수집된 데이터가 없습니다.")
                return None
                
            # 데이터 형식 변환
            df = self.format_binance_data(klines, timeframe)
            self.logger.info(f"데이터 수집 완료: {len(df)} rows")
            
            return df
            
        except BinanceAPIException as e:
            self.logger.error(f"Binance API 오류: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"데이터 수집 실패: {str(e)}")
            return None
    
    def format_binance_data(self, klines_data, timeframe):
        """Binance Klines 데이터를 기존 형식에 맞게 변환"""
        data_rows = []
        
        for kline in klines_data:
            # Binance kline 형식:
            # [Open time, Open, High, Low, Close, Volume, Close time, Quote volume, ...]
            
            # Unix timestamp (seconds로 변환)
            unix_timestamp = int(kline[BINANCE_KLINE_MAPPING['open_time']]) // 1000
            
            # Date 형식 맞추기 (타임프레임에 따라 다르게)
            date_obj = datetime.fromtimestamp(unix_timestamp)
            if timeframe in ['1h', '4h']:
                # 시간단위 데이터: 날짜 + 시간
                date_str = date_obj.strftime('%Y-%m-%d %H:%M:%S')
            else:
                # 일/주/월 단위 데이터: 날짜만
                date_str = date_obj.strftime('%Y-%m-%d')
            
            # OHLCV 데이터
            open_price = float(kline[BINANCE_KLINE_MAPPING['open']])
            high_price = float(kline[BINANCE_KLINE_MAPPING['high']])
            low_price = float(kline[BINANCE_KLINE_MAPPING['low']])
            close_price = float(kline[BINANCE_KLINE_MAPPING['close']])
            volume = float(kline[BINANCE_KLINE_MAPPING['volume']])
            
            # Volume USD 계산 (Quote asset volume을 사용하거나 close * volume)
            # Binance에서 quote_volume이 이미 USD 기준 거래량
            volume_usd = float(kline[BINANCE_KLINE_MAPPING['quote_volume']])
            
            row = {
                'unix': unix_timestamp,
                'date': date_str,
                'symbol': 'BTCUSD',  # 기존 형식에 맞게 변경
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'Volume USD': round(volume_usd, 2),
                # 보조지표는 나중에 계산
                'macd': np.nan,
                'macd_signal': np.nan,
                'macd_hist': np.nan,
                'rsi': np.nan
            }
            
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        return df
    
    def backup_existing_data(self, timeframe):
        """기존 데이터 백업"""
        if not ENABLE_BACKUP:
            return
            
        source_path = RAW_DATA_DIR / DATA_FILES[timeframe]
        if not source_path.exists():
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{DATA_FILES[timeframe]}.backup_{timestamp}"
        backup_path = BACKUP_DATA_DIR / backup_filename
        
        try:
            import shutil
            shutil.copy2(source_path, backup_path)
            self.logger.info(f"백업 완료: {backup_filename}")
        except Exception as e:
            self.logger.error(f"백업 실패: {str(e)}")
    
    def is_same_period(self, date1, date2, timeframe):
        """타임프레임에 따른 같은 기간 확인"""
        if date1 is None or date2 is None:
            return False
            
        if timeframe in ['1h', '4h']:
            # 시간 단위: 같은 날인지 확인
            return date1.date() == date2.date()
        elif timeframe == '1d':
            # 일 단위: 같은 날인지 확인
            return date1.date() == date2.date()
        elif timeframe == '1w':
            # 주 단위: 같은 주인지 확인 (월요일 기준)
            # 두 날짜가 같은 ISO 주차에 속하는지 확인
            return date1.isocalendar()[:2] == date2.isocalendar()[:2]  # (year, week)
        elif timeframe == '1m':
            # 월 단위: 같은 달인지 확인
            return date1.year == date2.year and date1.month == date2.month
        else:
            return date1.date() == date2.date()
    
    def get_period_start(self, date, timeframe):
        """타임프레임에 따른 기간 시작일 계산"""
        if timeframe == '1w':
            # 해당 주의 월요일 계산
            days_since_monday = date.weekday()
            monday = date - timedelta(days=days_since_monday)
            return monday.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe == '1m':
            # 해당 월의 1일 계산
            return date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            # 1h, 4h, 1d는 해당 날짜 그대로
            return date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def should_update_last_candle(self, last_date, current_date, timeframe):
        """마지막 캔들을 업데이트해야 하는지 판단"""
        if last_date is None or current_date is None:
            return False
        
        if timeframe == '1h':
            # 마지막 데이터 시간으로부터 1시간이 지나지 않았으면 업데이트
            return current_date < last_date + timedelta(hours=1)
        if timeframe == '4h':
            # 마지막 데이터 시간으로부터 4시간이 지나지 않았으면 업데이트
            return current_date < last_date + timedelta(hours=4)
        
        # 현재 진행중인 기간에 속하는지 확인
        is_same_period = self.is_same_period(last_date, current_date, timeframe)
        
        if timeframe in ['1h', '4h', '1d']:
            # 시간/일 단위: 같은 날이면 업데이트
            return is_same_period
        elif timeframe == '1w':
            # 주 단위: 같은 주면 업데이트
            if is_same_period:
                self.logger.info(f"같은 주 ({last_date.isocalendar()[1]}주차) - 마지막 캔들 업데이트")
                return True
            else:
                self.logger.info(f"다른 주 ({last_date.isocalendar()[1]}주차 → {current_date.isocalendar()[1]}주차) - 새 캔들 추가")
                return False
        elif timeframe == '1m':
            # 월 단위: 같은 달이면 업데이트
            if is_same_period:
                self.logger.info(f"같은 달 ({last_date.year}-{last_date.month:02d}) - 마지막 캔들 업데이트")
                return True
            else:
                self.logger.info(f"다른 달 ({last_date.year}-{last_date.month:02d} → {current_date.year}-{current_date.month:02d}) - 새 캔들 추가")
                return False
        
        return is_same_period

    def get_period_info(self, date, timeframe):
        """현재 기간 정보 출력 (디버깅용)"""
        if timeframe == '1w':
            year, week, weekday = date.isocalendar()
            period_start = self.get_period_start(date, timeframe)
            period_end = period_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            return f"{year}년 {week}주차 ({period_start.strftime('%Y-%m-%d')} ~ {period_end.strftime('%Y-%m-%d')})"
        elif timeframe == '1m':
            period_start = self.get_period_start(date, timeframe)
            # 다음 달 1일에서 1초 빼기
            if date.month == 12:
                next_month = date.replace(year=date.year + 1, month=1, day=1)
            else:
                next_month = date.replace(month=date.month + 1, day=1)
            period_end = next_month - timedelta(seconds=1)
            return f"{date.year}년 {date.month}월 ({period_start.strftime('%Y-%m-%d')} ~ {period_end.strftime('%Y-%m-%d')})"
        else:
            return f"{date.strftime('%Y-%m-%d')}"
    
    def get_appropriate_lookback_period(self, timeframe):
        """타임프레임에 따른 적절한 조회 기간 설정"""
        lookback_days = {
            '1h': 30,   # 30일
            '4h': 90,   # 90일  
            '1d': 365,  # 1년
            '1w': 365 * 2,  # 2년
            '1m': 365 * 5   # 5년
        }
        return lookback_days.get(timeframe, 365)
    
    def update_timeframe_data(self, timeframe, mode='auto'):
        """특정 타임프레임 데이터 업데이트 (주간/월간 캔들 고려)"""
        self.logger.info(f"=== {timeframe} 데이터 업데이트 시작 ===")
    
        # 1. 기존 데이터 확인
        existing_df = self.get_existing_data(timeframe)
    
        if existing_df is None:
            # 기존 데이터가 없으면 전체 수집
            self.logger.info("기존 데이터가 없어 전체 데이터를 수집합니다.")
            lookback_days = self.get_appropriate_lookback_period(timeframe)
            start_date = datetime.now() - timedelta(days=lookback_days)
        
            new_df = self.fetch_new_data(timeframe, start_date=start_date)
            if new_df is not None:
                self.save_data(new_df, timeframe)
            return
    
        # 2. 마지막 데이터 날짜 확인
        last_date = self.get_last_date(existing_df)
        current_date = datetime.now()

        self.logger.info(f"마지막 데이터 날짜: {last_date}")
        self.logger.info(f"현재 날짜: {current_date}")
        
        # 기간 정보 출력 (1w, 1m의 경우)
        if timeframe in ['1w', '1m']:
            last_period_info = self.get_period_info(last_date, timeframe)
            current_period_info = self.get_period_info(current_date, timeframe)
            self.logger.info(f"마지막 데이터 기간: {last_period_info}")
            self.logger.info(f"현재 기간: {current_period_info}")

        # 3. 타임프레임별 업데이트 전략 결정
        should_update = self.should_update_last_candle(last_date, current_date, timeframe)
        
        if should_update:
            # 현재 진행중인 캔들 업데이트
            self.logger.info(f"🔄 현재 진행중인 {timeframe} 캔들을 업데이트합니다.")

            if timeframe == '1w':
                start_date = self.get_period_start(current_date, timeframe)
                self.logger.info(f"   업데이트 범위: {start_date} (이번 주 월요일) ~ {current_date}")
            elif timeframe == '1m':
                start_date = self.get_period_start(current_date, timeframe)
                self.logger.info(f"   업데이트 범위: {start_date} (이번 달 1일) ~ {current_date}")
            else:
                # 1h, 4h, 1d → 마지막 데이터 이후부터 현재까지
                start_date = last_date + timedelta(seconds=1)
                self.logger.info(f"   업데이트 범위: {start_date} ~ {current_date}")

            end_date = current_date
            new_df = self.fetch_new_data(timeframe, start_date, end_date)
            if new_df is not None:
                updated_df = pd.concat([existing_df[:-1], new_df], ignore_index=True)
                updated_df = self.remove_duplicates(updated_df)
                self.backup_existing_data(timeframe)
                self.save_data(updated_df, timeframe)
        else:
            # 새로운 캔들 추가
            self.logger.info(f"➕ 새로운 {timeframe} 캔들을 추가합니다.")

            start_date = last_date + timedelta(seconds=1)
            self.logger.info(f"   추가 범위: {last_date} 이후 ~ {current_date}")
            
            end_date = None
            
            if start_date > datetime.now():
                self.logger.info("⏳ 추가할 새로운 데이터가 없습니다. 마지막 데이터가 최신입니다.")
                return
        
            new_df = self.fetch_new_data(timeframe, start_date, end_date)
        
            if new_df is not None and not new_df.empty:
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df = self.remove_duplicates(updated_df)
                self.backup_existing_data(timeframe)
                self.save_data(updated_df, timeframe)
            else:
                self.logger.info("✅ 추가할 새로운 캔들이 없습니다. 데이터가 이미 최신 상태입니다.")
    
    def remove_duplicates(self, df):
        """중복 데이터 제거 (unix timestamp 기준)"""
        before_count = len(df)
        df_clean = df.drop_duplicates(subset=['unix'], keep='last')
        after_count = len(df_clean)
        
        if before_count != after_count:
            self.logger.info(f"중복 데이터 제거: {before_count - after_count}개")
            
        return df_clean.reset_index(drop=True)
    
    def save_data(self, df, timeframe):
        """데이터 저장"""
        file_path = RAW_DATA_DIR / DATA_FILES[timeframe]
        
        try:
            df.to_csv(file_path, index=False)
            self.logger.info(f"데이터 저장 완료: {len(df)} rows -> {DATA_FILES[timeframe]}")
        except Exception as e:
            self.logger.error(f"데이터 저장 실패: {str(e)}")
    
    def update_all_timeframes(self):
        """모든 타임프레임 데이터 업데이트"""
        self.logger.info("전체 데이터 업데이트 시작")
        
        for timeframe in DATA_FILES.keys():
            try:
                self.update_timeframe_data(timeframe)
                time.sleep(REQUEST_DELAY)  # API 호출 간격
            except Exception as e:
                self.logger.error(f"{timeframe} 업데이트 실패: {str(e)}")
                continue
        
        self.logger.info("전체 데이터 업데이트 완료")

# 실행부
if __name__ == "__main__":
    collector = BTCDataCollector()
    
    # 사용 예시
    print("BTC 데이터 수집기 (Binance API)")
    print("1. 전체 업데이트")
    print("2. 특정 타임프레임 업데이트")
    
    choice = input("선택하세요 (1/2): ").strip()
    
    if choice == "1":
        collector.update_all_timeframes()
    elif choice == "2":
        print("타임프레임 목록:", list(DATA_FILES.keys()))
        timeframe = input("타임프레임을 입력하세요: ").strip()
        if timeframe in DATA_FILES:
            collector.update_timeframe_data(timeframe)
        else:
            print("잘못된 타임프레임입니다.")
    else:
        print("잘못된 선택입니다.")