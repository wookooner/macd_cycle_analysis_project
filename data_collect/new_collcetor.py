# C:\Users\Administrator\Desktop\macd_cycle_analysis_project\data_collect\advanced_collector_v2.py
#
# 기존 AdvancedBTCDataCollector 대비 변경사항:
#   1. taker_buy_base 컬럼 보존 → 캔들 레벨 CVD 계산 가능
#   2. UMFutures 선물 전용 클라이언트 추가 (OI, 펀딩비는 선물 API에만 존재)
#   3. 펀딩비 히스토리 수집 메서드 추가 (_fetch_funding_rate_history)
#   4. 미결제약정(OI) 히스토리 수집 메서드 추가 (_fetch_oi_history)
#   5. 모든 선물 데이터를 별도 CSV 파일로 저장 (기존 OHLCV와 분리)

import pandas as pd
from datetime import datetime, timezone
import time
from pathlib import Path
import logging
import shutil

# Binance 공식 커넥터: 현물(Spot)과 무기한 선물(UM Futures) 분리
from binance.spot import Spot as SpotClient
from binance.um_futures import UMFutures as FuturesClient
from binance.error import ClientError

from config import *


# ─── 선물 전용 상수 ────────────────────────────────────────────────────────────
# 바이낸스 무기한 선물 심볼 (현물 BTCUSD와 구분)
FUTURES_SYMBOL = "BTCUSDT"

# OI 히스토리 API는 타임프레임별 기간 제한이 있음
# period 파라미터가 kline interval과 별도로 존재 (5m/15m/30m/1h/2h/4h/6h/12h/1d)
OI_PERIOD_MAP = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# 펀딩비는 8시간마다 정산되므로 하루 3번. API limit 최대 1000
FUNDING_RATE_LIMIT = 1000
OI_HIST_LIMIT      = 500   # OI 히스토리 API 최대 limit

# 선물 데이터 저장 경로 (RAW_DATA_DIR는 config.py에서 가져옴)
# 기존 OHLCV CSV와 분리해서 혼용을 방지
FUTURES_DATA_FILES = {
    "funding_rate": "BTCUSDT_funding_rate.csv",
    "oi_1h":        "BTCUSDT_oi_1h.csv",
    "oi_4h":        "BTCUSDT_oi_4h.csv",
    "oi_1d":        "BTCUSDT_oi_1d.csv",
}

# ─── 숫자 반올림 설정 ────────────────────────────────────────────────────────────
# 각 데이터 타입별 소수점 자리수 (너무 큰 숫자 방지)
DECIMAL_PLACES = {
    "price": 2,              # 가격: 소수점 2자리 (예: 45123.45)
    "volume": 2,            # 거래량: 소수점 2자리
    "taker_buy": 2,         # taker_buy_base: 소수점 2자리
    "funding_rate": 8,      # 펀딩비: 소수점 8자리 (백분율 포함)
    "oi": 2,                # OI: 소수점 2자리
    "oi_change_pct": 4,     # OI 변화율: 소수점 4자리
}


class AdvancedBTCDataCollectorV2:
    """
    BTC 데이터 수집기 V2.
    - 기존 OHLCV 수집 기능 유지 (taker_buy_base 추가 보존)
    - 바이낸스 무기한 선물 전용 API로 펀딩비 / OI 히스토리 수집 추가
    """

    def __init__(self):
        self.logger = self._setup_logging()
        ensure_directories()
        
        # 현물 클라이언트: ping 테스트용 및 기존 klines 수집에 사용
        self.spot_client    = self._setup_spot_client()
        # 선물 클라이언트: OI, 펀딩비 등 선물 전용 엔드포인트에 사용
        self.futures_client = self._setup_futures_client()

    # ─── 클라이언트 초기화 ────────────────────────────────────────────────────

    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
        return logging.getLogger(__name__)

    def _setup_spot_client(self) -> SpotClient | None:
        try:
            client = SpotClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
            client.ping()
            self.logger.info("✅ Spot 클라이언트 연결 완료")
            return client
        except Exception as e:
            self.logger.error(f"❌ Spot 클라이언트 초기화 실패: {e}")
            return None

    def _setup_futures_client(self) -> FuturesClient | None:
        """
        선물 전용 클라이언트 초기화.
        OI, 펀딩비, 롱숏비율 등 선물 전용 엔드포인트는 이 클라이언트를 통해서만 접근 가능.
        """
        try:
            client = FuturesClient(
                key=BINANCE_API_KEY,
                secret=BINANCE_SECRET_KEY
            )
            client.ping()
            self.logger.info("✅ Futures 클라이언트 연결 완료")
            return client
        except Exception as e:
            self.logger.error(f"❌ Futures 클라이언트 초기화 실패: {e}")
            return None

    # ─── OHLCV + taker_buy_base 수집 ─────────────────────────────────────────

    def _fetch_klines(self, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
        """Spot klines 수집. 기존 로직과 동일하나 taker_buy_base를 보존."""
        if self.spot_client is None:
            self.logger.error("Spot 클라이언트가 초기화되지 않았습니다.")
            return None

        symbol   = BINANCE_SYMBOL
        interval = BINANCE_INTERVALS[timeframe]
        start_ts_ms = int(start_dt.timestamp() * 1000)
        end_ts_ms   = int(end_dt.timestamp() * 1000)

        self.logger.info(f"[OHLCV] {symbol} {interval} 수집 시작: {start_dt} ~ {end_dt}")
        all_klines = []
        fetch_start = start_ts_ms

        while True:
            try:
                klines = self.spot_client.klines(
                    symbol=symbol, interval=interval,
                    startTime=fetch_start, endTime=end_ts_ms, limit=MAX_LIMIT
                )
                if not klines:
                    break

                all_klines.extend(klines)
                fetch_start = klines[-1][0] + 1  # 마지막 open_time + 1ms
                self.logger.info(f"  {len(klines)}개 수집, 누적 {len(all_klines)}개")
                time.sleep(REQUEST_DELAY)

            except ClientError as e:
                self.logger.error(f"API 오류: {e.status_code} - {e.error_message}")
                return None
            except Exception as e:
                self.logger.error(f"수집 오류: {e}")
                return None

        if not all_klines:
            return pd.DataFrame()

        return self._format_klines(all_klines)

    def _format_klines(self, klines_data: list) -> pd.DataFrame:
        """
        Klines 데이터 포맷팅.
        핵심 변경: taker_buy_base를 최종 컬럼에 포함.
        CVD 계산: volume_delta = taker_buy_base - taker_sell_base
                               = taker_buy_base - (volume - taker_buy_base)
                               = 2 * taker_buy_base - volume
        (누적 CVD는 feature_extract 단계에서 candle_data를 순회하며 계산)
        """
        cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        df = pd.DataFrame(klines_data, columns=cols)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                        'quote_volume', 'taker_buy_base', 'taker_buy_quote']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        df['unix'] = df['open_time'] // 1000
        df['date'] = pd.to_datetime(df['unix'], unit='s', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')

        # volume_delta: 양수 = 매수 압력 우세, 음수 = 매도 압력 우세
        df['volume_delta'] = 2 * df['taker_buy_base'] - df['volume']

        df.rename(columns={'quote_volume': 'Volume USD'}, inplace=True)

        # taker_buy_base와 volume_delta를 기존 컬럼에 추가
        final_df = df[[
            'unix', 'date', 'open', 'high', 'low', 'close',
            'Volume USD', 'volume', 'taker_buy_base', 'volume_delta'
        ]].copy()

        # 숫자 반올림: 너무 큰 숫자 방지
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['Volume USD', 'volume', 'taker_buy_base', 'volume_delta']
        
        for col in price_cols:
            final_df[col] = final_df[col].round(DECIMAL_PLACES['price'])
        for col in volume_cols:
            final_df[col] = final_df[col].round(DECIMAL_PLACES['volume'])

        final_df['symbol'] = 'BTCUSD'

        # 기술적 지표 컬럼: 별도 indicator.py에서 채워짐
        for col in ['macd', 'macd_signal', 'macd_hist', 'rsi']:
            final_df[col] = pd.NA

        self.logger.info(f"OHLCV 포맷 완료: {len(final_df)}행")
        return final_df

    # ─── 펀딩비 히스토리 수집 ─────────────────────────────────────────────────

    def _fetch_funding_rate_history(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
        """
        바이낸스 선물 펀딩비 히스토리 수집.
        엔드포인트: GET /fapi/v1/fundingRate
        - 8시간마다 정산 (00:00, 08:00, 16:00 UTC)
        - 반환값: fundingTime(ms), fundingRate(float)
        - 사이클 분석에서 사용: 사이클 시작 시점의 펀딩비 → 시장 쏠림 정도 파악
          양수 높음 = 롱 과열 (숏 스퀴즈 위험), 음수 = 숏 과열 (롱 스퀴즈 위험)
        """
        if self.futures_client is None:
            self.logger.error("Futures 클라이언트가 없어 펀딩비 수집 불가")
            return None

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms   = int(end_dt.timestamp() * 1000)

        self.logger.info(f"[펀딩비] 수집 시작: {start_dt} ~ {end_dt}")
        all_rows = []
        fetch_start = start_ms

        while True:
            try:
                data = self.futures_client.funding_rate(
                    symbol=FUTURES_SYMBOL,
                    startTime=fetch_start,
                    endTime=end_ms,
                    limit=FUNDING_RATE_LIMIT
                )
                if not data:
                    break

                all_rows.extend(data)
                # 마지막 데이터 시간 + 1ms로 다음 시작점 설정
                fetch_start = data[-1]['fundingTime'] + 1
                self.logger.info(f"  {len(data)}개 수집, 누적 {len(all_rows)}개")

                if len(data) < FUNDING_RATE_LIMIT:
                    # 요청한 양보다 적게 왔으면 마지막 페이지
                    break
                time.sleep(REQUEST_DELAY)

            except ClientError as e:
                self.logger.error(f"펀딩비 API 오류: {e.status_code} - {e.error_message}")
                return None
            except Exception as e:
                self.logger.error(f"펀딩비 수집 오류: {e}")
                return None

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df['unix']         = df['fundingTime'] // 1000
        df['date']         = pd.to_datetime(df['unix'], unit='s', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        df['symbol']       = FUTURES_SYMBOL

        result = df[['unix', 'date', 'symbol', 'funding_rate']].copy()
        result = result.sort_values('unix').drop_duplicates(subset=['unix'])
        
        # 펀딩비 반올림
        result['funding_rate'] = result['funding_rate'].round(DECIMAL_PLACES['funding_rate'])
        
        self.logger.info(f"펀딩비 포맷 완료: {len(result)}행")
        return result

    # ─── OI 히스토리 수집 ────────────────────────────────────────────────────

    def _fetch_oi_history(self, period: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
        """
        바이낸스 선물 미결제약정(OI) 히스토리 수집.
        엔드포인트: GET /futures/data/openInterestHist
        - period: "1h", "4h", "1d" 등 (OI_PERIOD_MAP 참조)
        - 반환값: timestamp(ms), sumOpenInterest(계약수), sumOpenInterestValue(USDT 환산)
        
        사이클 분석에서의 활용:
        - oi_change (OI 변화량): 가격 상승 + OI 증가 → 새 롱 진입 (강한 신호)
                                  가격 상승 + OI 감소 → 숏 청산 (약한 신호)
        - 절대 OI 보다는 사이클 시작 시점 대비 변화율이 중요
        """
        if self.futures_client is None:
            self.logger.error("Futures 클라이언트가 없어 OI 수집 불가")
            return None

        # openInterestHist는 최대 30일치만 조회 가능 (바이낸스 제한)
        # 장기 히스토리가 필요하면 30일씩 분할 수집해야 함
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms   = int(end_dt.timestamp() * 1000)

        self.logger.info(f"[OI {period}] 수집 시작: {start_dt} ~ {end_dt}")
        all_rows = []
        fetch_start = start_ms

        while True:
            try:
                data = self.futures_client.open_interest_hist(
                    symbol=FUTURES_SYMBOL,
                    period=period,
                    startTime=fetch_start,
                    endTime=end_ms,
                    limit=OI_HIST_LIMIT
                )
                if not data:
                    break

                all_rows.extend(data)
                fetch_start = data[-1]['timestamp'] + 1
                self.logger.info(f"  {len(data)}개 수집, 누적 {len(all_rows)}개")

                if len(data) < OI_HIST_LIMIT:
                    break
                time.sleep(REQUEST_DELAY)

            except ClientError as e:
                self.logger.error(f"OI API 오류: {e.status_code} - {e.error_message}")
                return None
            except Exception as e:
                self.logger.error(f"OI 수집 오류: {e}")
                return None

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df['unix']   = df['timestamp'] // 1000
        df['date']   = pd.to_datetime(df['unix'], unit='s', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['oi']     = pd.to_numeric(df['sumOpenInterest'],      errors='coerce')  # 계약수
        df['oi_usd'] = pd.to_numeric(df['sumOpenInterestValue'], errors='coerce')  # USDT 환산
        df['symbol'] = FUTURES_SYMBOL
        df['period'] = period

        # OI 변화량 (절대값보다 변화율이 분석에 더 유용)
        df = df.sort_values('unix').reset_index(drop=True)
        df['oi_change']     = df['oi'].diff()
        df['oi_change_pct'] = df['oi'].pct_change() * 100

        result = df[['unix', 'date', 'symbol', 'period', 'oi', 'oi_usd',
                     'oi_change', 'oi_change_pct']].copy()
        result = result.drop_duplicates(subset=['unix'])
        
        # OI 데이터 반올림
        result['oi']            = result['oi'].round(DECIMAL_PLACES['oi'])
        result['oi_usd']        = result['oi_usd'].round(DECIMAL_PLACES['oi'])
        result['oi_change']     = result['oi_change'].round(DECIMAL_PLACES['oi'])
        result['oi_change_pct'] = result['oi_change_pct'].round(DECIMAL_PLACES['oi_change_pct'])
        
        self.logger.info(f"OI {period} 포맷 완료: {len(result)}행")
        return result

    # ─── 저장 공통 유틸 ───────────────────────────────────────────────────────

    def _merge_and_save(self, existing_df: pd.DataFrame, new_df: pd.DataFrame,
                        file_path: Path, key_col: str = 'unix'):
        """데이터 병합 + 중복 제거 + 백업 + 저장 (공통 로직)"""
        if new_df is None or new_df.empty:
            self.logger.info(f"추가할 데이터 없음: {file_path.name}")
            return

        # 백업
        if ENABLE_BACKUP and file_path.exists():
            ts          = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = BACKUP_DATA_DIR / f"{file_path.name}.backup_{ts}"
            try:
                shutil.copy2(file_path, backup_path)
                self.logger.info(f"백업: {backup_path.name}")
            except Exception as e:
                self.logger.error(f"백업 실패: {e}")

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        before   = len(combined)
        combined.drop_duplicates(subset=[key_col], keep='last', inplace=True)
        combined.sort_values(key_col, inplace=True)
        self.logger.info(f"중복 {before - len(combined)}개 제거 → {len(combined)}행")

        try:
            combined.to_csv(file_path, index=False)
            self.logger.info(f"✅ 저장: {file_path} ({len(combined)}행)")
        except Exception as e:
            self.logger.error(f"저장 실패: {e}")

    def _load_existing(self, file_path: Path) -> pd.DataFrame:
        """기존 CSV 로드. 없으면 빈 DataFrame 반환."""
        if file_path.exists():
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                self.logger.warning(f"기존 파일 로드 실패 ({file_path.name}): {e}")
        return pd.DataFrame()

    # ─── 업데이트 메서드 (퍼블릭 인터페이스) ─────────────────────────────────

    def update_ohlcv(self, timeframe: str):
        """OHLCV + taker_buy_base + volume_delta 업데이트"""
        self.logger.info(f"===== OHLCV {timeframe} 업데이트 =====")
        file_path   = RAW_DATA_DIR / DATA_FILES[timeframe]
        existing_df = self._load_existing(file_path)

        if not existing_df.empty:
            last_unix   = existing_df['unix'].iloc[-1]
            start_fetch = datetime.fromtimestamp(last_unix, tz=timezone.utc)
        else:
            start_fetch = datetime(2017, 1, 1, tzinfo=timezone.utc)

        new_df = self._fetch_klines(timeframe, start_fetch, datetime.now(timezone.utc))
        self._merge_and_save(existing_df, new_df, file_path)

    def update_funding_rate(self, start_dt: datetime = None):
        """펀딩비 히스토리 업데이트"""
        self.logger.info("===== 펀딩비 업데이트 =====")
        file_path   = RAW_DATA_DIR / FUTURES_DATA_FILES['funding_rate']
        existing_df = self._load_existing(file_path)

        if not existing_df.empty:
            last_unix   = existing_df['unix'].iloc[-1]
            start_fetch = datetime.fromtimestamp(last_unix, tz=timezone.utc)
        elif start_dt is not None:
            start_fetch = start_dt
        else:
            # 펀딩비 히스토리는 2019년 9월부터 존재 (BTCUSDT 무기한 선물 상장일)
            start_fetch = datetime(2019, 9, 13, tzinfo=timezone.utc)

        new_df = self._fetch_funding_rate_history(start_fetch, datetime.now(timezone.utc))
        self._merge_and_save(existing_df, new_df, file_path)

    def update_oi(self, period: str = "4h", start_dt: datetime = None):
        """
        미결제약정(OI) 히스토리 업데이트.
        주의: 바이낸스 OI 히스토리는 최근 30일치 데이터만 제공.
              따라서 장기 분석용으로는 정기적으로 수집해서 누적 저장해야 함.
        """
        if period not in OI_PERIOD_MAP:
            self.logger.error(f"지원하지 않는 period: {period}. 사용 가능: {list(OI_PERIOD_MAP.keys())}")
            return

        self.logger.info(f"===== OI {period} 업데이트 =====")
        file_key    = f"oi_{period}"
        file_path   = RAW_DATA_DIR / FUTURES_DATA_FILES[file_key]
        existing_df = self._load_existing(file_path)

        if not existing_df.empty:
            last_unix   = existing_df['unix'].iloc[-1]
            start_fetch = datetime.fromtimestamp(last_unix, tz=timezone.utc)
        elif start_dt is not None:
            start_fetch = start_dt
        else:
            # OI 히스토리는 최근 30일만 가능하므로 30일 전부터 시작
            from datetime import timedelta
            start_fetch = datetime.now(timezone.utc) - timedelta(days=30)

        new_df = self._fetch_oi_history(period, start_fetch, datetime.now(timezone.utc))
        self._merge_and_save(existing_df, new_df, file_path)

    def update_all_ohlcv(self):
        """모든 타임프레임 OHLCV 업데이트"""
        self.logger.info("🚀 전체 OHLCV 업데이트 시작")
        for tf in DATA_FILES.keys():
            self.update_ohlcv(tf)
        self.logger.info("🎉 전체 OHLCV 업데이트 완료")

    def update_all_futures_data(self):
        """펀딩비 + 주요 OI 타임프레임 일괄 업데이트"""
        self.logger.info("🚀 선물 데이터(펀딩비 + OI) 업데이트 시작")
        self.update_funding_rate()
        for period in OI_PERIOD_MAP.keys():
            self.update_oi(period)
        self.logger.info("🎉 선물 데이터 업데이트 완료")


# ─── 실행 진입점 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    collector = AdvancedBTCDataCollectorV2()

    print("\n🚀 BTC 데이터 수집기 V2")
    print("─" * 40)
    print("1. 전체 OHLCV 업데이트 (모든 타임프레임)")
    print("2. 특정 타임프레임 OHLCV 업데이트")
    print("3. 펀딩비 히스토리 업데이트")
    print("4. OI(미결제약정) 히스토리 업데이트")
    print("5. 선물 데이터 전체 업데이트 (펀딩비 + OI)")
    print("6. 전체 업데이트 (OHLCV + 선물 데이터)")

    choice = input("\n원하는 작업을 선택하세요 (1~6): ").strip()

    if choice == "1":
        collector.update_all_ohlcv()

    elif choice == "2":
        tf_list = list(DATA_FILES.keys())
        print(f"사용 가능한 타임프레임: {tf_list}")
        tf = input("타임프레임 입력: ").strip()
        if tf in tf_list:
            collector.update_ohlcv(tf)
        else:
            print("❌ 잘못된 타임프레임")

    elif choice == "3":
        collector.update_funding_rate()

    elif choice == "4":
        print(f"사용 가능한 period: {list(OI_PERIOD_MAP.keys())}")
        period = input("period 입력 (기본값 4h): ").strip() or "4h"
        collector.update_oi(period)

    elif choice == "5":
        collector.update_all_futures_data()

    elif choice == "6":
        collector.update_all_ohlcv()
        collector.update_all_futures_data()

    else:
        print("❌ 잘못된 선택")