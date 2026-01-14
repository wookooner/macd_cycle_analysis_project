# final_predictor.py

import pandas as pd
import os
from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime

class IndicatorPredictor:
    """
    현재 진행중인 캔들(T+0)의 마감 상태와, 그를 기반으로 다음 캔들(T+1)을
    순차적으로 시뮬레이션하는 클래스.
    """

    def __init__(self, csv_path: str, fast: int = 12, slow: int = 26, signal: int = 9, rsi_period: int = 14):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.rsi_period = rsi_period

        self.df = self._load_data(csv_path)
        if len(self.df) < self.slow + self.signal:
            raise ValueError(f"데이터가 너무 적어({len(self.df)}개) 예측을 시작할 수 없습니다.")
            
        self._calculate_initial_indicators()

        # T-1 (마지막 확정 캔들) 상태 저장
        self.t_minus_1_state = self._extract_state(self.df.iloc[-2])
        self.current_candle_data = self.df.iloc[-1]
        
        # 시간 정보 계산
        self.t0_unix = self.df['unix'].iloc[-1]
        interval = self.t0_unix - self.df['unix'].iloc[-2]
        self.t1_unix = self.t0_unix + interval

    def _load_data(self, csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"오류: '{csv_path}' 파일을 찾을 수 없습니다.")
        df = pd.read_csv(csv_path).sort_values('unix').reset_index(drop=True)
        if 'close' not in df.columns or 'unix' not in df.columns:
            raise ValueError("오류: CSV 파일에 'close'와 'unix' 컬럼이 모두 존재해야 합니다.")
        return df

    def _calculate_initial_indicators(self):
        self.df['ema_fast'] = self.df['close'].ewm(span=self.fast, adjust=False).mean()
        self.df['ema_slow'] = self.df['close'].ewm(span=self.slow, adjust=False).mean()
        self.df['macd'] = self.df['ema_fast'] - self.df['ema_slow']
        self.df['signal'] = self.df['macd'].ewm(span=self.signal, adjust=False).mean()
        self.df['histogram'] = self.df['macd'] - self.df['signal']
        
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        self.df['avg_gain'] = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        self.df['avg_loss'] = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        rs = self.df['avg_gain'] / self.df['avg_loss']
        self.df['rsi'] = 100 - (100 / (1 + rs))

    def _extract_state(self, data_row: pd.Series) -> Dict:
        return {
            "close": data_row['close'], "ema_fast": data_row['ema_fast'],
            "ema_slow": data_row['ema_slow'], "signal": data_row['signal'],
            "avg_gain": data_row['avg_gain'], "avg_loss": data_row['avg_loss']
        }

    def project_indicators(self, target_price: float, prev_state: Dict) -> Tuple[Dict, Dict]:
        last_close = prev_state['close']
        last_ema_fast, last_ema_slow, last_signal = prev_state['ema_fast'], prev_state['ema_slow'], prev_state['signal']
        last_avg_gain, last_avg_loss = prev_state['avg_gain'], prev_state['avg_loss']

        mf, ms, msig = 2/(self.fast+1), 2/(self.slow+1), 2/(self.signal+1)
        next_ema_fast = (target_price * mf) + (last_ema_fast * (1 - mf))
        next_ema_slow = (target_price * ms) + (last_ema_slow * (1 - ms))
        next_macd = next_ema_fast - next_ema_slow
        next_signal = (next_macd * msig) + (last_signal * (1 - msig))
        
        alpha = 1 / self.rsi_period
        delta = target_price - last_close
        gain = delta if delta > 0 else 0
        loss = -delta if delta < 0 else 0
        next_avg_gain = (gain * alpha) + (last_avg_gain * (1 - alpha))
        next_avg_loss = (loss * alpha) + (last_avg_loss * (1 - alpha))
        rs = next_avg_gain / next_avg_loss if next_avg_loss > 0 else np.inf
        
        indicators = {
            "macd": next_macd, "signal": next_signal,
            "histogram": next_macd - next_signal, "rsi": 100 - (100 / (1 + rs))
        }
        next_state = {
            "close": target_price, "ema_fast": next_ema_fast, "ema_slow": next_ema_slow,
            "signal": next_signal, "avg_gain": next_avg_gain, "avg_loss": next_avg_loss
        }
        return indicators, next_state

    def calculate_price_for_target(self, target_value: float, target_type: str, prev_state: Dict) -> Optional[float]:
        mf, ms, msig = 2/(self.fast+1), 2/(self.slow+1), 2/(self.signal+1)
        
        if target_type == 'macd':
            numerator = target_value - (prev_state['ema_fast']*(1-mf)) + (prev_state['ema_slow']*(1-ms))
            denominator = mf - ms
        elif target_type == 'histogram':
            c1 = prev_state['ema_fast']*(1-mf) - prev_state['ema_slow']*(1-ms)
            c2 = prev_state['signal']*(1-msig)
            numerator = target_value - c1*(1-msig) + c2
            denominator = (mf-ms)*(1-msig)
        elif target_type == 'rsi':
            if not (0 < target_value < 100): return None
            target_rs = target_value / (100 - target_value)
            alpha = 1 / self.rsi_period
            
            price_if_gain = ((target_rs * (prev_state['avg_loss']*(1-alpha)) - (prev_state['avg_gain']*(1-alpha))) / alpha) + prev_state['close']
            if price_if_gain > prev_state['close']: return price_if_gain
            
            den_loss = (prev_state['avg_gain']*(1-alpha)) / target_rs if target_rs > 0 else np.inf
            price_if_loss = prev_state['close'] - ((den_loss - (prev_state['avg_loss']*(1-alpha))) / alpha)
            if price_if_loss < prev_state['close']: return price_if_loss
            return None
        else: return None
        return numerator / denominator if denominator != 0 else None

def run_prediction_step(predictor: IndicatorPredictor, step_name: str, prev_state: Dict, step_time: str):
    print("\n" + "="*20 + f" {step_name} 예측 " + "="*20)
    print(f"예측 대상 시간: {step_time}")
    
    while True:
        print("\n어떤 값을 기준으로 예측하시겠습니까? (1. 가격, 2. MACD, 3. Histogram, 4. RSI)")
        choice = input("선택: ")
        try:
            if choice == '1':
                price = float(input("예상 마감 가격: "))
                return predictor.project_indicators(price, prev_state)
            elif choice in ['2', '3', '4']:
                target_map = {'2': 'macd', '3': 'histogram', '4': 'rsi'}
                target_type = target_map[choice]
                value = float(input(f"목표 {target_type.upper()} 값: "))
                price = predictor.calculate_price_for_target(value, target_type, prev_state)
                if price is None:
                    print("\n--- ❗ 예측 실패: 해당 목표값을 달성하는 가격을 계산할 수 없습니다. ---")
                    return None, None
                print(f" -> 목표 달성을 위한 예상 가격: ${price:,.2f}")
                return predictor.project_indicators(price, prev_state)
            else: print("잘못된 선택입니다.")
        except ValueError: print("오류: 유효한 숫자를 입력하세요.")

if __name__ == "__main__":
    CSV_FILE_PATH = "data/base_data/BTCUSD_4h.csv"
    
    try:
        predictor = IndicatorPredictor(CSV_FILE_PATH)
        
        print("=" * 60)
        print("📊 현재 캔들(T+0) 및 다음 캔들(T+1) 예측기")
        print("=" * 60)
        
        while True:
            t_minus_1_data = predictor.df.iloc[-2]
            print(f"\n--- 기준 정보: 마지막 확정 캔들 (T-1) ---")
            print(f"시간: {datetime.fromtimestamp(t_minus_1_data['unix']).strftime('%Y-%m-%d %H:%M:%S')}, 종가: ${t_minus_1_data['close']:,.2f}")

            print(f"\n--- 참고 정보: 현재 진행중인 캔들 (T+0) ---")
            print(f"시간: {datetime.fromtimestamp(predictor.t0_unix).strftime('%Y-%m-%d %H:%M:%S')}, 현재가: ${predictor.current_candle_data['close']:,.2f}")
            
            # --- 1단계: 현재 캔들(T+0) 마감 예측 ---
            t0_indicators, t0_final_state = run_prediction_step(
                predictor, "현재 캔들(T+0) 마감", predictor.t_minus_1_state,
                datetime.fromtimestamp(predictor.t0_unix).strftime('%Y-%m-%d %H:%M:%S')
            )
            
            if t0_final_state is None: continue
            
            print("\n--- ✅ 현재 캔들(T+0) 마감 예측 결과 ---")
            print(f"예상 종가: ${t0_final_state['close']:,.2f}")
            print(f"  - MACD: {t0_indicators['macd']:.4f}, Histo: {t0_indicators['histogram']:.4f}, RSI: {t0_indicators['rsi']:.2f}")

            # --- 2단계: 다음 캔들(T+1) 예측 ---
            t1_indicators, t1_final_state = run_prediction_step(
                predictor, "다음 캔들(T+1)", t0_final_state,
                datetime.fromtimestamp(predictor.t1_unix).strftime('%Y-%m-%d %H:%M:%S')
            )

            if t1_final_state:
                print("\n--- ✅ 다음 캔들(T+1) 예측 결과 ---")
                print(f"예상 종가: ${t1_final_state['close']:,.2f}")
                print(f"  - MACD: {t1_indicators['macd']:.4f}, Histo: {t1_indicators['histogram']:.4f}, RSI: {t1_indicators['rsi']:.2f}")

            print("\n" + "="*50)
            if input("새로운 시나리오로 다시 예측하시겠습니까? (y/n): ").lower() != 'y':
                break
        
        print("프로그램을 종료합니다.")

    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"\n프로그램 실행 중 오류가 발생했습니다: {e}")