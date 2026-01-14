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
        
        indicator_columns = ['macd', 'macd_signal', 'macd_hist', 'rsi']
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
    
    def update_indicators(self, df, force_recalculate=False, recalc_last_n=50):
        """지표 업데이트 (기존 계산 상태 고려)"""
        self.logger.info("=== 지표 업데이트 시작 ===")
        status = self.check_existing_indicators(df, recalc_last_n)
        
        full_recalc_needed = (
            force_recalculate or
            not status['macd']['exists'] or
            not status['rsi']['exists'] or
            status['macd']['coverage_ratio'] < 0.5 or
            status['rsi']['coverage_ratio'] < 0.5
        )
        
        if full_recalc_needed:
            self.logger.info("전체 지표를 새로 계산합니다.")
            df = self.calculate_macd(df)
            df = self.calculate_rsi(df)
        else:
            self.logger.info(f"마지막 {recalc_last_n}개 데이터의 지표를 재계산합니다.")
            df = self.recalculate_last_indicators(df, recalc_last_n)
        
        return df
    
    def recalculate_last_indicators(self, df, last_n=50):
        """마지막 N개 데이터의 지표만 재계산"""
        total_rows = len(df)
        if total_rows <= last_n:
            return self.calculate_macd(self.calculate_rsi(df))

        macd_recalc_start = max(0, total_rows - max(last_n, 60))
        rsi_recalc_start = max(0, total_rows - max(last_n, 30))
        
        self.logger.info(f"MACD 재계산 구간: {macd_recalc_start}~{total_rows}")
        self.logger.info(f"RSI 재계산 구간: {rsi_recalc_start}~{total_rows}")
        
        df_temp = self.calculate_macd(self.calculate_rsi(df.copy()))
        
        df.loc[macd_recalc_start:, ['macd', 'macd_signal', 'macd_hist']] = df_temp.loc[macd_recalc_start:, ['macd', 'macd_signal', 'macd_hist']]
        df.loc[rsi_recalc_start:, 'rsi'] = df_temp.loc[rsi_recalc_start:, 'rsi']
        
        self.logger.info("마지막 구간 지표 재계산 완료")
        return df
    
    def process_file(self, file_path, output_path=None, force_recalculate=False, recalc_last_n=50):
        """파일 단위로 지표 계산 처리"""
        self.logger.info(f"파일 처리 시작: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"데이터 로드 완료: {len(df)} rows, {len(df.columns)} columns")
            
            if 'unix' in df.columns:
                df = df.sort_values('unix').reset_index(drop=True)
                self.logger.info("데이터를 unix timestamp 기준으로 정렬했습니다.")
            
            df = self.update_indicators(df, force_recalculate, recalc_last_n)
            
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

def process_multiple_files(file_paths, force_recalculate=False, recalc_last_n=50):
    """여러 파일을 한 번에 처리"""
    calculator = IndicatorCalculator()
    results = {}
    
    calculator.logger.info(f"=== 배치 처리 시작: {len(file_paths)}개 파일 ===")
    
    for file_path in file_paths:
        result = calculator.process_file(
            file_path, 
            force_recalculate=force_recalculate,
            recalc_last_n=recalc_last_n
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