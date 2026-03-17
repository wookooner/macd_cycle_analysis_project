# C:\Users\Administrator\Desktop\macd_cycle_analysis_project\data_control\indicator.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import warnings
import shutil

warnings.filterwarnings('ignore')

class IndicatorCalculator:
    def __init__(self, log_level="INFO"):
        self.setup_logging(log_level)
        
    def setup_logging(self, log_level):
        """로깅 설정"""
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
    
    def check_existing_indicators(self, df, check_last_n=50):
        """기존 지표 계산 상태 확인"""
        self.logger.info("=== 기존 지표 상태 확인 ===")
        
        indicator_columns = ['macd', 'macd_signal', 'macd_hist', 'rsi', 'cvd', 'cvd_rolling']
        status = {}
        
        for col in indicator_columns:
            if col not in df.columns:
                status[col] = {'exists': False, 'calculated_rows': 0, 'missing_rows': len(df), 'last_n_calculated': 0, 'last_n_missing': min(check_last_n, len(df))}
                continue
                
            valid_count = df[col].notna().sum()
            missing_count = df[col].isna().sum()
            last_n_data = df[col].tail(check_last_n)
            last_n_valid = last_n_data.notna().sum()
            
            coverage = valid_count / len(df) if len(df) > 0 else 0
            
            status[col] = {
                'exists': True,
                'calculated_rows': valid_count,
                'missing_rows': missing_count,
                'coverage_ratio': coverage,
                'last_n_calculated': last_n_valid,
                'last_n_missing': len(last_n_data) - last_n_valid,
                'last_n_coverage': last_n_valid / len(last_n_data) if len(last_n_data) > 0 else 0
            }

        for col, info in status.items():
            if info['exists']:
                self.logger.info(f"{col}: 전체 {info['calculated_rows']}/{len(df)} 계산됨 ({info['coverage_ratio']:.1%}), 마지막 {check_last_n}개 중 {info['last_n_calculated']}개 계산됨 ({info['last_n_coverage']:.1%})")
            else:
                self.logger.info(f"{col}: 컬럼이 존재하지 않음")
        
        return status
    
    def calculate_ema(self, data, period, adjust=True):
        """지수이동평균(EMA) 계산"""
        return data.ewm(span=period, adjust=adjust).mean()
    
    def calculate_macd(self, df, fast_period=12, slow_period=26, signal_period=9):
        """MACD 지표 계산"""
        self.logger.info(f"MACD 계산 중... (Fast:{fast_period}, Slow:{slow_period}, Signal:{signal_period})")
        if 'close' not in df.columns:
            self.logger.error("close 컬럼이 없습니다.")
            return df
        
        close_prices = df['close'].copy()
        ema_fast = self.calculate_ema(close_prices, fast_period)
        ema_slow = self.calculate_ema(close_prices, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        df['macd'] = macd_line.round(6)
        df['macd_signal'] = signal_line.round(6)
        df['macd_hist'] = histogram.round(6)
        
        self.logger.info(f"MACD 계산 완료: {df['macd'].notna().sum()}개 값 생성")
        return df
    
    def calculate_rsi(self, df, period=14):
        """RSI 지표 계산"""
        self.logger.info(f"RSI 계산 중... (Period: {period})")
        if 'close' not in df.columns:
            self.logger.error("close 컬럼이 없습니다.")
            return df
            
        close_prices = df['close'].copy()
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df['rsi'] = rsi.round(2)
        
        self.logger.info(f"RSI 계산 완료: {df['rsi'].notna().sum()}개 값 생성")
        return df
    
    def calculate_cvd(self, df, rolling_period=20):
        """
        CVD(Cumulative Volume Delta) 계산.

        CVD란 무엇인가:
          각 캔들의 volume_delta(= 2*taker_buy_base - volume)를 누적 합산한 값.
          volume_delta가 양수면 그 캔들에서 공격적 매수(Taker Buy)가 우세했다는 의미이고,
          음수면 공격적 매도(Taker Sell)가 우세했다는 의미다.
          CVD가 지속적으로 상승하면 매수 압력이 누적되고 있는 상태이고,
          CVD가 하락하면 매도 압력이 누적되고 있는 상태다.

        두 가지 CVD를 계산하는 이유:
          - cvd (누적 CVD): 선물 상장일(2019-09-13) 이후부터의 전체 누적합.
            "지금까지 쌓인 순매수 압력의 총량"을 의미하며, 장기 추세 파악에 유용.
            단, 절대값이 매우 커지므로 기간 간 직접 비교는 어렵다.

          - cvd_rolling (롤링 CVD): 최근 N캔들의 volume_delta 합계.
            오실레이터처럼 동작해서 현재 시장의 단기 매수/매도 압력 균형을 보여준다.
            MACD와 함께 사이클 시작 시점의 자금 흐름 강도를 측정하는 데 적합하다.

        선물 상장일(2019-09-13) 이전 데이터는 taker_buy_base가 없으므로 NaN으로 유지.

        Parameters
        ----------
        rolling_period : int
            롤링 CVD 계산 윈도우 (기본값 20 — 타임프레임별로 다르게 쓸 수도 있음)
        """
        self.logger.info(f"CVD 계산 중... (rolling_period: {rolling_period})")

        if 'volume_delta' not in df.columns:
            self.logger.warning("volume_delta 컬럼이 없습니다. backfill_futures_columns.py 먼저 실행 필요.")
            df['cvd']         = pd.NA
            df['cvd_rolling'] = pd.NA
            return df

        # volume_delta가 NaN인 행(선물 상장 이전)은 CVD 계산에서 제외
        # → NaN 구간은 그대로 NaN으로 두고, 유효 구간만 누적합 계산
        vd = df['volume_delta'].copy()
        valid_mask = vd.notna()

        # ── 누적 CVD ──────────────────────────────────────────
        # 선물 데이터가 시작되는 첫 유효 행부터 누적합 계산
        # NaN 구간(이전)은 건드리지 않음
        cvd_series = pd.Series(np.nan, index=df.index, dtype='float64')
        if valid_mask.any():
            cvd_series[valid_mask] = vd[valid_mask].cumsum()

        # ── 롤링 CVD ──────────────────────────────────────────
        # min_periods=1: 롤링 윈도우가 채워지기 전에도 값을 계산
        # (초기 구간에서도 값이 끊기지 않도록)
        cvd_rolling_series = pd.Series(np.nan, index=df.index, dtype='float64')
        if valid_mask.any():
            cvd_rolling_series[valid_mask] = (
                vd[valid_mask]
                .rolling(window=rolling_period, min_periods=1)
                .sum()
            )

        df['cvd']         = cvd_series.round(4)
        df['cvd_rolling'] = cvd_rolling_series.round(4)

        valid_cvd = df['cvd'].notna().sum()
        self.logger.info(f"CVD 계산 완료: {valid_cvd}개 값 생성 (NaN {len(df) - valid_cvd}개는 선물 상장 이전)")
        return df

    def update_indicators(self, df, force_recalculate=False, recalc_last_n=50,
                          cvd_rolling_period=20):
        """
        지표 업데이트 (기존 계산 상태 고려).
        MACD, RSI와 함께 CVD도 계산한다.

        CVD는 MACD/RSI와 계산 조건이 다르기 때문에 별도로 판단한다.
          - MACD/RSI: close 가격만 있으면 항상 계산 가능
          - CVD: volume_delta(= taker_buy_base 기반)가 있어야 계산 가능
                 선물 상장일(2019-09-13) 이전 구간은 NaN이 정상이다.

        cvd_rolling_period는 타임프레임에 따라 조정할 수 있다.
        예를 들어 1d 데이터라면 20일 롤링, 4h 데이터라면 20*6=120 캔들이
        같은 "20거래일" 개념이 되므로 필요에 따라 변경해서 사용한다.
        """
        self.logger.info("=== 지표 업데이트 시작 ===")
        status = self.check_existing_indicators(df, recalc_last_n)

        full_recalc_needed = (
            force_recalculate or
            not status['macd']['exists'] or
            not status['rsi']['exists'] or
            status['macd']['coverage_ratio'] < 0.5 or
            status['rsi']['coverage_ratio'] < 0.5
        )

        # CVD는 volume_delta 존재 여부와 기존 cvd 컬럼 상태로 별도 판단
        cvd_recalc_needed = (
            force_recalculate or
            not status['cvd']['exists'] or
            status['cvd']['coverage_ratio'] < 0.01   # 거의 없으면 재계산
        )

        if full_recalc_needed:
            self.logger.info("전체 지표를 새로 계산합니다.")
            df = self.calculate_macd(df)
            df = self.calculate_rsi(df)
        else:
            self.logger.info(f"마지막 {recalc_last_n}개 데이터의 지표를 재계산합니다.")
            df = self.recalculate_last_indicators(df, recalc_last_n, cvd_rolling_period)

        # CVD는 누적합 특성상 항상 전체 재계산이 필요하다.
        # 중간 값 하나만 바꿔도 이후 모든 누적값이 달라지기 때문이다.
        if cvd_recalc_needed:
            df = self.calculate_cvd(df, rolling_period=cvd_rolling_period)
        else:
            self.logger.info("CVD: 기존 값 유지 (재계산 불필요)")

        return df

    def recalculate_last_indicators(self, df, last_n=50, cvd_rolling_period=20):
        """
        마지막 N개 데이터의 지표만 재계산.
        MACD/RSI는 부분 재계산이 가능하지만,
        CVD는 누적합이라 항상 전체를 다시 계산해야 한다.
        """
        total_rows = len(df)
        if total_rows <= last_n:
            df = self.calculate_macd(df)
            df = self.calculate_rsi(df)
            return df

        macd_recalc_start = max(0, total_rows - max(last_n, 60))
        rsi_recalc_start  = max(0, total_rows - max(last_n, 30))

        self.logger.info(f"MACD 재계산 구간: {macd_recalc_start}~{total_rows}")
        self.logger.info(f"RSI 재계산 구간: {rsi_recalc_start}~{total_rows}")

        df_temp = self.calculate_macd(self.calculate_rsi(df.copy()))

        df.loc[macd_recalc_start:, ['macd', 'macd_signal', 'macd_hist']] = \
            df_temp.loc[macd_recalc_start:, ['macd', 'macd_signal', 'macd_hist']]
        df.loc[rsi_recalc_start:, 'rsi'] = df_temp.loc[rsi_recalc_start:, 'rsi']

        # CVD rolling은 마지막 N행만 업데이트해도 되지만,
        # 누적 CVD는 전체를 재계산하지 않으면 정합성이 깨지므로 전체 재계산
        df = self.calculate_cvd(df, rolling_period=cvd_rolling_period)

        self.logger.info("마지막 구간 지표 재계산 완료")
        return df
    
    def process_file(self, file_path, output_path=None, force_recalculate=False,
                     recalc_last_n=50, cvd_rolling_period=20):
        """파일 단위로 지표 계산 처리"""
        self.logger.info(f"파일 처리 시작: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"데이터 로드 완료: {len(df)} rows, {len(df.columns)} columns")
            
            if 'unix' in df.columns:
                df = df.sort_values('unix').reset_index(drop=True)
                self.logger.info("데이터를 unix timestamp 기준으로 정렬했습니다.")
            
            df = self.update_indicators(df, force_recalculate, recalc_last_n,
                                         cvd_rolling_period=cvd_rolling_period)

            if output_path is None:
                output_path = file_path

            # --- 백업 로직 수정 ---
            if output_path == file_path:
                # 백업 디렉토리 경로 설정
                backup_dir = file_path.parent.parent / 'backup_data'
                # 백업 디렉토리 생성 (없으면)
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                backup_filename = f"{file_path.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = backup_dir / backup_filename
                
                shutil.copy2(file_path, backup_path)
                self.logger.info(f"백업 생성: {backup_path}")
            # --- 수정 끝 ---
            
            df.to_csv(output_path, index=False)
            self.logger.info(f"처리 완료: {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"파일 처리 실패: {str(e)}")
            return False

def process_multiple_files(file_paths, force_recalculate=False, recalc_last_n=50,
                           cvd_rolling_period=20):
    """여러 파일을 한 번에 처리"""
    calculator = IndicatorCalculator()
    results = {}
    
    calculator.logger.info(f"=== 배치 처리 시작: {len(file_paths)}개 파일 ===")
    
    for file_path in file_paths:
        result = calculator.process_file(
            file_path,
            force_recalculate=force_recalculate,
            recalc_last_n=recalc_last_n,
            cvd_rolling_period=cvd_rolling_period
        )
        results[file_path] = result
        
    return results

if __name__ == "__main__":
    data_dir = Path("data/base_data")
    
    print("=== 경로 확인 ===")
    print(f"데이터 디렉토리: {data_dir.resolve()}")
    print(f"디렉토리 존재 여부: {data_dir.exists()}")
    
    if not data_dir.exists():
        print("❌ 데이터 디렉토리 'data/base_data'를 찾을 수 없습니다. 스크립트를 프로젝트 루트에서 실행했는지 확인하세요.")
        exit(1)
    
    files = [ "BTCUSD_1h.csv", "BTCUSD_4h.csv", "BTCUSD_1d.csv", "BTCUSD_1w.csv", "BTCUSD_1m.csv" ]
    
    existing_files = [data_dir / f for f in files if (data_dir / f).exists()]
    
    print("\n--- 처리 대상 파일 ---")
    for f in existing_files:
        print(f"✅ {f.name}")
    if not existing_files:
        print("처리할 CSV 파일이 없습니다.")
        exit(1)
    
    print(f"\n지표 계산 설정:")
    print("1. 스마트 업데이트 (마지막 50개 데이터만 재계산)")
    print("2. 전체 재계산 (모든 지표 새로 계산)")
    
    mode_choice = input("모드를 선택하세요 (1/2): ").strip()
    
    if mode_choice == "1":
        force_recalc = False
        recalc_n = 50
    elif mode_choice == "2":
        force_recalc = True
        recalc_n = 50
    else:
        print("잘못된 선택입니다. 프로그램을 종료합니다.")
        exit(1)
    
    print(f"\n지표 계산 시작... (총 {len(existing_files)}개 파일)")
    results = process_multiple_files(existing_files, force_recalculate=force_recalc, recalc_last_n=recalc_n)
    
    print("\n--- 처리 결과 ---")
    for file_path, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {Path(file_path).name}: {status}")