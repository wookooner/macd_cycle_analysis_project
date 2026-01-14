"""
핵심 데이터 검증 모듈
데이터 분석에 필수적인 품질 검증만 수행

검증 항목:
1. NaN/무한값/이상값 검증
2. 핵심 가격 데이터 무결성 (음수, OHLC 관계)
3. 타임스탬프 중복/정렬 검증
4. Binance API 기반 지표 정확성 검증

필요한 라이브러리:
- pip install python-binance pandas numpy
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Binance API 관련 import (선택적)
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    print("⚠️  python-binance 라이브러리가 설치되지 않았습니다.")
    print("   Binance 기반 검증을 사용하려면 'pip install python-binance'를 실행하세요.")
    BINANCE_AVAILABLE = False
    Client = None
    BinanceAPIException = Exception

warnings.filterwarnings('ignore')

class DataValidator:
    def __init__(self, log_level="INFO", enable_binance_validation=True):
        self.setup_logging(log_level)
        self.enable_binance_validation = enable_binance_validation and BINANCE_AVAILABLE
        
        if self.enable_binance_validation:
            self.setup_binance_client()
        
    def setup_logging(self, log_level):
        """로깅 설정"""
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_binance_client(self):
        """Binance 클라이언트 초기화"""
        if not BINANCE_AVAILABLE:
            self.logger.warning("python-binance 라이브러리가 없어 Binance 검증을 비활성화합니다.")
            self.client = None
            self.enable_binance_validation = False
            return
            
        try:
            self.client = Client()  # Public API만 사용
            self.logger.info("Binance 클라이언트 초기화 완료")
        except Exception as e:
            self.logger.error(f"Binance 클라이언트 초기화 실패: {str(e)}")
            self.client = None
            self.enable_binance_validation = False

    def validate_data_quality(self, df, file_name):
        """핵심 데이터 품질 검증"""
        self.logger.info(f"=== {file_name} 데이터 품질 검증 ===")
        
        issues = []
        
        # 1. NaN/무한값 검증 (중요 컬럼만)
        critical_columns = ['unix', 'date', 'open', 'high', 'low', 'close']
        
        for col in critical_columns:
            if col not in df.columns:
                issues.append(f"필수 컬럼 누락: {col}")
                continue
                
            # NaN 값 확인
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                issues.append(f"{col}에 NaN 값: {nan_count}개")
            
            # 무한값 확인 (숫자 컬럼만)
            if col != 'date':
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    issues.append(f"{col}에 무한값: {inf_count}개")
        
        # 2. 핵심 가격 데이터 검증
        price_columns = ['open', 'high', 'low', 'close']
        
        if all(col in df.columns for col in price_columns):
            # 음수 또는 0 값 확인
            for col in price_columns:
                zero_or_negative = (df[col] <= 0).sum()
                if zero_or_negative > 0:
                    issues.append(f"{col}에 0 이하 값: {zero_or_negative}개")
            
            # OHLC 관계 검증
            invalid_high = (df['high'] < df[['open', 'close', 'low']].max(axis=1)).sum()
            if invalid_high > 0:
                issues.append(f"High가 다른 가격보다 낮은 경우: {invalid_high}개")
            
            invalid_low = (df['low'] > df[['open', 'close', 'high']].min(axis=1)).sum()
            if invalid_low > 0:
                issues.append(f"Low가 다른 가격보다 높은 경우: {invalid_low}개")
        
        # 3. 타임스탬프 기본 검증
        if 'unix' in df.columns:
            # 중복 타임스탬프
            duplicates = df['unix'].duplicated().sum()
            if duplicates > 0:
                issues.append(f"중복 타임스탬프: {duplicates}개")
            
            # 정렬 확인
            if not df['unix'].is_monotonic_increasing:
                issues.append("타임스탬프가 시간순으로 정렬되지 않음")
        
        if not issues:
            self.logger.info("✅ 데이터 품질 검증 통과")
        else:
            for issue in issues:
                self.logger.warning(f"⚠️  {issue}")
        
        return issues

    def fetch_binance_data_and_calculate(self, timeframe, limit=50):
        """Binance에서 데이터 가져와서 지표 계산 (정확한 정의 확인)"""
        if not self.enable_binance_validation or self.client is None:
            return None
            
        interval_mapping = {
            '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w', '1m': '1M'
        }
        
        if timeframe not in interval_mapping:
            return None
            
        try:
            # 데이터 가져오기
            klines = self.client.get_historical_klines(
                symbol="BTCUSDT",
                interval=interval_mapping[timeframe],
                limit=limit
            )
            
            if not klines:
                return None
            
            # 데이터 변환
            data_rows = []
            for kline in klines:
                unix_timestamp = int(kline[0]) // 1000
                row = {
                    'unix': unix_timestamp,
                    'close': float(kline[4])
                }
                data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            close_prices = df['close']
            
            self.logger.info(f"🔍 MACD 지표 계산 정의 확인:")
            self.logger.info(f"   1. EMA Fast (12기간) = close.ewm(span=12)")
            self.logger.info(f"   2. EMA Slow (26기간) = close.ewm(span=26)")
            self.logger.info(f"   3. MACD Line = EMA Fast - EMA Slow")
            self.logger.info(f"   4. Signal Line = MACD Line.ewm(span=9)")
            self.logger.info(f"   5. Histogram = MACD Line - Signal Line")
            
            # MACD 계산 (상세 로깅)
            ema_fast = close_prices.ewm(span=12, adjust=True).mean()
            ema_slow = close_prices.ewm(span=26, adjust=True).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=True).mean()
            histogram = macd_line - signal_line
            
            # 결과 저장 (순서 중요!)
            df['macd'] = macd_line.round(6)           # MACD Line
            df['macd_signal'] = signal_line.round(6)  # Signal Line  
            df['macd_hist'] = histogram.round(6)      # Histogram
            
            # 계산 결과 샘플 출력 (최근 3개)
            if len(df) >= 3:
                self.logger.info(f"📊 Binance 계산 결과 샘플 (최근 3개):")
                sample_df = df.tail(3)
                for idx, row in sample_df.iterrows():
                    self.logger.info(f"   Unix {int(row['unix'])}: "
                                   f"MACD={row['macd']:.6f}, "
                                   f"Signal={row['macd_signal']:.6f}, "
                                   f"Hist={row['macd_hist']:.6f}")
            
            # RSI 계산 (Wilder's smoothing)
            self.logger.info(f"\n🔍 RSI 지표 계산 정의 확인:")
            self.logger.info(f"   1. Delta = close.diff()")
            self.logger.info(f"   2. Gain = delta where delta > 0")
            self.logger.info(f"   3. Loss = -delta where delta < 0")
            self.logger.info(f"   4. Avg Gain = gain.ewm(alpha=1/14, adjust=False)")
            self.logger.info(f"   5. Avg Loss = loss.ewm(alpha=1/14, adjust=False)")
            self.logger.info(f"   6. RS = Avg Gain / Avg Loss")
            self.logger.info(f"   7. RSI = 100 - (100 / (1 + RS))")
            
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            df['rsi'] = rsi.round(2)
            
            # RSI 계산 결과 샘플 출력
            if len(df) >= 3:
                self.logger.info(f"📊 RSI 계산 결과 샘플 (최근 3개):")
                sample_df = df.tail(3)
                for idx, row in sample_df.iterrows():
                    self.logger.info(f"   Unix {int(row['unix'])}: RSI={row['rsi']:.2f}")
            
            self.logger.info(f"\n✅ Binance 데이터 및 지표 계산 완료: {len(df)}개")
            self.logger.info(f"⚠️  파일의 지표 정의와 일치하는지 확인 필요!")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Binance 데이터 가져오기 실패: {str(e)}")
            return None

    def verify_indicator_definitions(self, df, file_name):
        """파일의 지표 정의 확인 및 샘플 출력"""
        self.logger.info(f"\n🔍 {file_name} 파일 지표 정의 확인:")
        
        # 컬럼 순서 확인
        indicator_columns = []
        for col in df.columns:
            if col in ['macd', 'macd_signal', 'macd_hist', 'rsi']:
                indicator_columns.append(col)
        
        self.logger.info(f"📋 파일의 지표 컬럼 순서: {indicator_columns}")
        
        # 각 지표의 의미 추정
        self.logger.info(f"📝 추정되는 지표 의미:")
        self.logger.info(f"   - macd: MACD Line (EMA12 - EMA26)")
        self.logger.info(f"   - macd_signal: Signal Line (MACD의 EMA9)")
        self.logger.info(f"   - macd_hist: Histogram (MACD - Signal)")
        self.logger.info(f"   - rsi: RSI (14기간, Wilder's smoothing)")
        
        # 파일 데이터 샘플 출력 (최근 3개, NaN이 아닌 것들)
        if len(df) >= 3:
            # NaN이 아닌 최근 데이터 찾기
            valid_data = df[df[indicator_columns].notna().all(axis=1)]
            if len(valid_data) >= 3:
                self.logger.info(f"📊 파일 데이터 샘플 (최근 3개):")
                sample_df = valid_data.tail(3)
                for idx, row in sample_df.iterrows():
                    date_str = row.get('date', 'N/A')
                    unix_str = int(row['unix']) if 'unix' in row else 'N/A'
                    
                    macd_val = row.get('macd', 'N/A')
                    signal_val = row.get('macd_signal', 'N/A') 
                    hist_val = row.get('macd_hist', 'N/A')
                    rsi_val = row.get('rsi', 'N/A')
                    
                    self.logger.info(f"   {date_str} (Unix {unix_str}): "
                                   f"MACD={macd_val}, Signal={signal_val}, "
                                   f"Hist={hist_val}, RSI={rsi_val}")
                    
                    # Histogram = MACD - Signal 관계 확인
                    if (pd.notna(macd_val) and pd.notna(signal_val) and pd.notna(hist_val)):
                        calculated_hist = float(macd_val) - float(signal_val)
                        hist_diff = abs(calculated_hist - float(hist_val))
                        if hist_diff > 0.000001:
                            self.logger.warning(f"      ⚠️ Histogram 불일치: 계산값={calculated_hist:.6f}, 파일값={hist_val}, 차이={hist_diff:.6f}")
                        else:
                            self.logger.info(f"      ✅ Histogram 일치: {calculated_hist:.6f}")
        
        self.logger.info(f"\n❓ 위 정의가 맞다면 'y', 다르다면 'n'을 입력하세요.")
        return True

    def validate_indicators_with_binance(self, df, file_name, timeframe):
        """Binance 계산값과 파일 지표값 비교 (정확한 지표 정의 확인)"""
        self.logger.info(f"=== {file_name} Binance 지표 비교 검증 ===")
        
        if not self.enable_binance_validation:
            self.logger.info("Binance 검증 비활성화됨")
            return []
        
        issues = []
        
        try:
            # 1. 파일의 지표 정의 확인
            self.verify_indicator_definitions(df, file_name)
            
            # 2. Binance 데이터 및 지표 계산
            binance_df = self.fetch_binance_data_and_calculate(timeframe, limit=50)
            
            if binance_df is None:
                issues.append("Binance 데이터를 가져올 수 없음")
                return issues
            
            # 3. 지표 정의 일치 여부 사용자 확인
            self.logger.info(f"\n❗ 중요: 지표 정의 확인")
            self.logger.info(f"파일과 Binance 계산이 같은 정의를 사용하는지 위 샘플값들을 비교해보세요.")
            self.logger.info(f"특히 다음 사항들을 확인하세요:")
            self.logger.info(f"  1. MACD = EMA(12) - EMA(26) 인가?")
            self.logger.info(f"  2. Signal = MACD의 EMA(9) 인가?") 
            self.logger.info(f"  3. Histogram = MACD - Signal 인가?")
            self.logger.info(f"  4. RSI = 14기간 Wilder's smoothing 인가?")
            
            # 데이터 정렬 및 정리
            df_sorted = df.copy().sort_values('unix').reset_index(drop=True)
            binance_df_sorted = binance_df.copy().sort_values('unix').reset_index(drop=True)
            
            self.logger.info(f"\n📊 데이터 매칭:")
            self.logger.info(f"파일 데이터: {len(df_sorted)}개, Binance 데이터: {len(binance_df_sorted)}개")
            
            # Unix timestamp를 기준으로 정확한 매칭
            merged_df = pd.merge(
                df_sorted[['unix', 'date', 'macd', 'macd_signal', 'macd_hist', 'rsi']],
                binance_df_sorted[['unix', 'macd', 'macd_signal', 'macd_hist', 'rsi']],
                on='unix',
                how='inner',
                suffixes=('_file', '_binance')
            )
            
            if len(merged_df) < 5:
                issues.append(f"매칭된 공통 데이터가 부족함: {len(merged_df)}개")
                self.logger.warning(f"파일 unix 범위: {df_sorted['unix'].min()} ~ {df_sorted['unix'].max()}")
                self.logger.warning(f"Binance unix 범위: {binance_df_sorted['unix'].min()} ~ {binance_df_sorted['unix'].max()}")
                return issues
            
            # 최근 데이터만 비교 (최대 15개)
            recent_merged = merged_df.tail(15).copy()
            self.logger.info(f"매칭된 최근 {len(recent_merged)}개 데이터포인트 비교")
            
            # 매칭된 데이터의 샘플 비교 출력
            self.logger.info(f"\n📋 매칭된 데이터 샘플 비교 (최근 3개):")
            sample_merged = recent_merged.tail(3)
            for idx, row in sample_merged.iterrows():
                date_str = row['date'] if pd.notna(row['date']) else str(row['unix'])
                self.logger.info(f"📌 {date_str} (Unix: {int(row['unix'])}):")
                self.logger.info(f"   MACD     - 파일: {row['macd_file']:.6f}, Binance: {row['macd_binance']:.6f}, 차이: {abs(row['macd_file']-row['macd_binance']):.6f}")
                self.logger.info(f"   Signal   - 파일: {row['macd_signal_file']:.6f}, Binance: {row['macd_signal_binance']:.6f}, 차이: {abs(row['macd_signal_file']-row['macd_signal_binance']):.6f}")
                self.logger.info(f"   Hist     - 파일: {row['macd_hist_file']:.6f}, Binance: {row['macd_hist_binance']:.6f}, 차이: {abs(row['macd_hist_file']-row['macd_hist_binance']):.6f}")
                self.logger.info(f"   RSI      - 파일: {row['rsi_file']:.2f}, Binance: {row['rsi_binance']:.2f}, 차이: {abs(row['rsi_file']-row['rsi_binance']):.2f}")
            
            # 지표별 상세 비교
            indicators = ['macd', 'macd_signal', 'macd_hist', 'rsi']
            
            for indicator in indicators:
                file_col = f"{indicator}_file"
                binance_col = f"{indicator}_binance"
                
                if file_col not in recent_merged.columns or binance_col not in recent_merged.columns:
                    continue
                
                # NaN 값 제거
                valid_mask = recent_merged[file_col].notna() & recent_merged[binance_col].notna()
                if valid_mask.sum() == 0:
                    self.logger.warning(f"{indicator}: 비교 가능한 유효 데이터가 없음")
                    continue
                
                valid_data = recent_merged[valid_mask].copy()
                
                # 차이 계산
                valid_data['abs_diff'] = abs(valid_data[file_col] - valid_data[binance_col])
                valid_data['rel_diff'] = np.where(
                    abs(valid_data[binance_col]) > 1e-10,
                    valid_data['abs_diff'] / abs(valid_data[binance_col]) * 100,
                    0
                )
                
                # 통계 계산
                avg_abs_diff = valid_data['abs_diff'].mean()
                max_abs_diff = valid_data['abs_diff'].max()
                avg_rel_diff = valid_data['rel_diff'].mean()
                max_rel_diff = valid_data['rel_diff'].max()
                
                self.logger.info(f"\n📊 {indicator.upper()} 비교 통계 ({len(valid_data)}개 포인트):")
                self.logger.info(f"  평균 절대차이: {avg_abs_diff:.6f}, 최대 절대차이: {max_abs_diff:.6f}")
                self.logger.info(f"  평균 상대차이: {avg_rel_diff:.2f}%, 최대 상대차이: {max_rel_diff:.2f}%")
                
                # 임계값 설정 (더 관대하게 - 소수점 정밀도 고려)
                if indicator in ['macd', 'macd_signal', 'macd_hist']:
                    abs_threshold = 0.001      # 0.01 -> 0.001로 더 엄격하게
                    rel_threshold = 2.0        # 5.0 -> 2.0으로 더 엄격하게
                    max_abs_threshold = 0.01   # 0.05 -> 0.01로 더 엄격하게
                    max_rel_threshold = 10.0   # 15.0 -> 10.0으로 더 엄격하게
                else:  # RSI
                    abs_threshold = 0.5        # 1.0 -> 0.5로 더 엄격하게
                    rel_threshold = 1.0        # 2.0 -> 1.0으로 더 엄격하게
                    max_abs_threshold = 2.0    # 3.0 -> 2.0으로 더 엄격하게
                    max_rel_threshold = 3.0    # 5.0 -> 3.0으로 더 엄격하게
                
                # 문제가 있는 데이터포인트 찾기
                problematic_mask = (
                    (valid_data['abs_diff'] > abs_threshold) | 
                    (valid_data['rel_diff'] > rel_threshold)
                )
                problematic_points = valid_data[problematic_mask]
                
                # 임계값 초과 체크
                has_issues = False
                
                if avg_abs_diff > abs_threshold:
                    issues.append(f"{indicator} 평균 절대차이 초과: {avg_abs_diff:.6f} > {abs_threshold}")
                    has_issues = True
                
                if max_abs_diff > max_abs_threshold:
                    issues.append(f"{indicator} 최대 절대차이 초과: {max_abs_diff:.6f} > {max_abs_threshold}")
                    has_issues = True
                
                if avg_rel_diff > rel_threshold:
                    issues.append(f"{indicator} 평균 상대차이 초과: {avg_rel_diff:.2f}% > {rel_threshold}%")
                    has_issues = True
                
                if max_rel_diff > max_rel_threshold:
                    issues.append(f"{indicator} 최대 상대차이 초과: {max_rel_diff:.2f}% > {max_rel_threshold}%")
                    has_issues = True
                
                # 문제가 있거나 사용자가 확인하고 싶은 경우 상세 테이블 출력
                if has_issues or len(problematic_points) > 0:
                    self.print_comparison_table_v2(indicator, valid_data, problematic_points, abs_threshold, rel_threshold)
                elif len(valid_data) <= 5:  # 데이터가 적으면 항상 표시
                    self.print_comparison_table_v2(indicator, valid_data, problematic_points, abs_threshold, rel_threshold)
            
            if not issues:
                self.logger.info("✅ Binance 지표 비교 검증 통과")
                self.logger.info("💡 지표 정의가 일치하고 계산 정확도가 양호합니다.")
            else:
                self.logger.warning(f"⚠️  총 {len(issues)}개 지표 비교 이슈 발견")
                self.logger.warning(f"💡 지표 정의 불일치 또는 계산 정밀도 차이일 수 있습니다.")
                    
        except Exception as e:
            issues.append(f"Binance 비교 검증 오류: {str(e)}")
            self.logger.error(f"Binance 검증 실패: {str(e)}")
        
        return issues
    
    def print_comparison_table_v2(self, indicator, valid_data, problematic_points, abs_threshold, rel_threshold):
        """개선된 지표 비교 상세 테이블 출력"""
        
        # 차이가 큰 순으로 정렬
        sorted_data = valid_data.sort_values('abs_diff', ascending=False)
        
        # 표시할 데이터 선택 (문제 있는 것들 + 상위 차이)
        display_indices = set()
        
        # 1. 문제가 있는 포인트들 먼저 추가 (최대 5개)
        if len(problematic_points) > 0:
            display_indices.update(problematic_points.head(5).index)
        
        # 2. 나머지 상위 차이 데이터 추가 (총 최대 10개까지)
        for idx in sorted_data.index:
            if len(display_indices) >= 10:
                break
            display_indices.add(idx)
        
        display_data = valid_data.loc[sorted(display_indices)].sort_values('abs_diff', ascending=False)
        
        if len(display_data) > 0:
            self.logger.info(f"\n📊 {indicator.upper()} 상세 비교 테이블 (정확한 매칭):")
            self.logger.info("-" * 110)
            self.logger.info(f"{'시간':<20} {'Unix':<12} {'파일값':<12} {'Binance값':<12} {'절대차이':<12} {'상대차이':<10} {'상태'}")
            self.logger.info("-" * 110)
            
            for idx in display_data.index:
                row = display_data.loc[idx]
                
                # 상태 표시
                status = ""
                if row['abs_diff'] > abs_threshold:
                    status += "🔴"  # 절대차이 초과
                if row['rel_diff'] > rel_threshold:
                    status += "🟠"  # 상대차이 초과
                if not status:
                    status = "✅"   # 정상
                
                # 시간 표시 (간단하게)
                date_display = str(row['date'])[:19] if pd.notna(row['date']) else str(row['unix'])
                
                # Unix timestamp 표시 (뒤 6자리만)
                unix_display = str(int(row['unix']))[-6:]
                
                # 값 포맷팅
                file_col = f"{indicator}_file"
                binance_col = f"{indicator}_binance"
                
                if indicator == 'rsi':
                    file_val_str = f"{row[file_col]:.2f}"
                    binance_val_str = f"{row[binance_col]:.2f}"
                    diff_str = f"{row['abs_diff']:.2f}"
                else:
                    file_val_str = f"{row[file_col]:.6f}"
                    binance_val_str = f"{row[binance_col]:.6f}"
                    diff_str = f"{row['abs_diff']:.6f}"
                
                self.logger.info(f"{date_display:<20} {unix_display:<12} {file_val_str:<12} {binance_val_str:<12} "
                               f"{diff_str:<12} {row['rel_diff']:<9.2f}% {status}")
            
            self.logger.info("-" * 110)
            
            # 범례 및 요약
            if len(problematic_points) > 0:
                self.logger.info("🔴 절대차이 초과  🟠 상대차이 초과  ✅ 정상")
                self.logger.info(f"임계값: 절대차이 {abs_threshold}, 상대차이 {rel_threshold}%")
                
                self.logger.warning(f"⚠️  임계값 초과 데이터포인트: {len(problematic_points)}개 / {len(valid_data)}개")
                
                # 가장 문제가 심각한 포인트 강조
                worst_idx = problematic_points['abs_diff'].idxmax()
                worst_row = problematic_points.loc[worst_idx]
                worst_file_val = worst_row[f"{indicator}_file"]
                worst_binance_val = worst_row[f"{indicator}_binance"]
                
                if indicator == 'rsi':
                    self.logger.warning(f"   최악 케이스: {worst_row['date']} - "
                                      f"파일값 {worst_file_val:.2f}, "
                                      f"Binance값 {worst_binance_val:.2f}, "
                                      f"차이 {worst_row['abs_diff']:.2f} ({worst_row['rel_diff']:.2f}%)")
                else:
                    self.logger.warning(f"   최악 케이스: {worst_row['date']} - "
                                      f"파일값 {worst_file_val:.6f}, "
                                      f"Binance값 {worst_binance_val:.6f}, "
                                      f"차이 {worst_row['abs_diff']:.6f} ({worst_row['rel_diff']:.2f}%)")
            
            # 매칭 정확성 확인 정보
            first_row = display_data.iloc[0]
            last_row = display_data.iloc[-1]
            self.logger.info(f"📍 매칭 범위: {first_row['date']} ~ {last_row['date']}")
            self.logger.info(f"📍 Unix 범위: {int(first_row['unix'])} ~ {int(last_row['unix'])}")
            self.logger.info("")  # 빈 줄 추가

    def comprehensive_validation(self, file_path, timeframe=None):
        """핵심 검증 실행"""
        file_name = Path(file_path).name
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"핵심 데이터 검증: {file_name}")
        self.logger.info(f"{'='*50}")
        
        # 타임프레임 자동 추출
        if timeframe is None:
            for tf in ['1h', '4h', '1d', '1w', '1m']:
                if tf in file_name:
                    timeframe = tf
                    break
        
        try:
            # 데이터 로드
            df = pd.read_csv(file_path)
            self.logger.info(f"데이터 로드: {len(df)} rows")
            
            all_issues = []
            
            # 1. 핵심 데이터 품질 검증
            quality_issues = self.validate_data_quality(df, file_name)
            all_issues.extend(quality_issues)
            
            # 2. Binance 지표 비교 검증
            if timeframe and self.enable_binance_validation:
                binance_issues = self.validate_indicators_with_binance(df, file_name, timeframe)
                all_issues.extend(binance_issues)
            
            # 결과 요약
            self.logger.info(f"\n{'='*50}")
            if all_issues:
                self.logger.warning(f"⚠️  총 {len(all_issues)}개 이슈 발견:")
                for i, issue in enumerate(all_issues, 1):
                    self.logger.warning(f"  {i}. {issue}")
            else:
                self.logger.info("✅ 모든 핵심 검증 통과!")
            self.logger.info(f"{'='*50}\n")
            
            return {
                'file_name': file_name,
                'timeframe': timeframe,
                'total_rows': len(df),
                'validation_time': datetime.now(),
                'total_issues': len(all_issues),
                'is_valid': len(all_issues) == 0,
                'issues': all_issues
            }
            
        except Exception as e:
            self.logger.error(f"검증 중 오류: {str(e)}")
            return None

def validate_multiple_files(file_paths, enable_binance_validation=True):
    """여러 파일 일괄 검증"""
    validator = DataValidator(enable_binance_validation=enable_binance_validation)
    results = {}
    
    for file_path in file_paths:
        path_obj = Path(file_path)
        if not path_obj.exists():
            validator.logger.warning(f"파일이 존재하지 않습니다: {file_path}")
            continue
            
        result = validator.comprehensive_validation(file_path)
        results[file_path] = result
    
    # 전체 요약 출력
    validator.logger.info("\n" + "="*60)
    validator.logger.info("전체 검증 결과 요약")
    validator.logger.info("="*60)
    
    total_files = len(results)
    valid_files = sum(1 for r in results.values() if r and r['is_valid'])
    
    validator.logger.info(f"총 파일 수: {total_files}")
    validator.logger.info(f"검증 통과: {valid_files}")
    validator.logger.info(f"문제 파일: {total_files - valid_files}")
    
    for file_path, result in results.items():
        if result:
            status = "✅ PASS" if result['is_valid'] else f"⚠️  {result['total_issues']} issues"
            validator.logger.info(f"  {Path(file_path).name}: {status}")
    
    return results

if __name__ == "__main__":
    # 경로 확인 및 설정
    data_dir = Path("data/base_data")
    
    print("=== 경로 확인 ===")
    print(f"데이터 디렉토리: {data_dir.resolve()}")
    print(f"디렉토리 존재 여부: {data_dir.exists()}")
    
    if not data_dir.exists():
        print("❌ 데이터 디렉토리가 존재하지 않습니다.")
        possible_paths = [
            Path("data/base_data"),
            Path("./data/base_data"), 
            Path("../data/base_data"),
            Path("../../data/base_data")
        ]
        
        for path in possible_paths:
            print(f"  {path.resolve()}: {path.exists()}")
        
        current_dir = Path(".")
        csv_files = list(current_dir.glob("*.csv"))
        if csv_files:
            print(f"\n현재 디렉토리의 CSV 파일들:")
            for f in csv_files:
                print(f"  - {f.name}")
            
            use_current = input("\n현재 디렉토리의 파일들을 사용하시겠습니까? (y/n): ").lower()
            if use_current == 'y':
                data_dir = current_dir
        
        if not data_dir.exists() and not csv_files:
            print("검증할 파일을 찾을 수 없습니다.")
            exit(1)
    
    # 파일 목록
    files = ["BTCUSD_1h.csv", "BTCUSD_4h.csv", "BTCUSD_1d.csv", "BTCUSD_1w.csv", "BTCUSD_1m.csv"]
    
    # 실제 존재하는 파일만 필터링
    existing_files = []
    for f in files:
        file_path = data_dir / f
        if file_path.exists():
            existing_files.append(file_path)
            print(f"✅ 발견: {f}")
        else:
            print(f"❌ 없음: {f}")
    
    if not existing_files:
        print("검증할 CSV 파일이 없습니다.")
        exit(1)
    
    # Binance 검증 사용 여부 선택
    enable_binance = BINANCE_AVAILABLE
    
    if BINANCE_AVAILABLE:
        print(f"\n📊 Binance API 지표 검증:")
        print("1. 사용함 (추천)")
        print("2. 사용하지 않음")
        
        choice = input("선택 (1/2): ").strip()
        enable_binance = choice != "2"
    else:
        print(f"\n⚠️  python-binance 없음. 기본 검증만 실행")
    
    print(f"\n🔍 핵심 데이터 검증 시작... (총 {len(existing_files)}개 파일)")
    if enable_binance:
        print("   - 데이터 품질 + Binance 지표 비교")
    else:
        print("   - 데이터 품질 검증만")
    
    results = validate_multiple_files(existing_files, enable_binance_validation=enable_binance)