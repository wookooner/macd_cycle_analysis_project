"""
feature_extract.py
카테고리 기반 사이클 특징 추출기 (새로운 구조)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
from pathlib import Path
import sys
import json

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
print(project_root)

from macd_historgram_change_feature.config import DEFAULT_CONFIG

warnings.filterwarnings('ignore')

class CycleFeatureCalculator:
    """카테고리 기반 사이클 특징 계산기"""
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.name = "Categorized Cycle Feature Calculator"
        self.version = "2.0"
    
    def extract_features_from_candle_data(self, candle_data: List[Dict]) -> Dict[str, Dict]:
        """캔들 데이터로부터 카테고리별 특징 추출"""
        if candle_data is None or len(candle_data) == 0:
            return self.config.get_default_cycle_features_structure()
        
        # 캔들 데이터를 DataFrame으로 변환
        df = pd.DataFrame(candle_data)
        
        # 필요한 컬럼들 확인 및 기본값 설정
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'macd', 'macd_signal', 'macd_hist']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # RSI는 선택적
        if 'rsi' not in df.columns:
            df['rsi'] = 50.0
        
        # 특징 계산
        features = {}
        
        # 각 카테고리별로 특징 계산
        for category_name, category_data in self.config.FEATURE_CATEGORIES.items():
            features[category_name] = {}
            
            for feature_name, feature_config in category_data['features'].items():
                if feature_config['enabled']:
                    try:
                        calculator_method = getattr(self, feature_config['calculator'])
                        value = calculator_method(df)
                        
                        # 데이터 타입 변환
                        if feature_config['data_type'] == 'int':
                            value = int(value) if pd.notna(value) else feature_config['default_value']
                        elif feature_config['data_type'] == 'float':
                            value = float(value) if pd.notna(value) else feature_config['default_value']
                        
                        features[category_name][feature_name] = value
                        
                    except Exception as e:
                        print(f"특징 계산 오류 ({category_name}.{feature_name}): {e}")
                        features[category_name][feature_name] = feature_config['default_value']
        
        return features
    
    # ===== SHAPE 카테고리 계산 함수들 =====
    def calc_duration_candles(self, df: pd.DataFrame) -> int:
        """사이클의 전체 길이 (캔들 수)"""
        return len(df)
    
    def calc_core_count(self, df: pd.DataFrame) -> int:
        """사이클 방향과 일치하는 핵심 캔들 수 (히스토그램 변화량 기준)"""
        if len(df) < 2:
            return 0
        
        hist_changes = df['macd_hist'].diff().dropna()
        if len(hist_changes) == 0:
            return 0
        
        # 전체 추세 방향 결정
        overall_change = df['macd_hist'].iloc[-1] - df['macd_hist'].iloc[0]
        trend_direction = 1 if overall_change > 0 else -1
        
        # 추세와 같은 방향의 변화 개수
        same_direction_count = (np.sign(hist_changes) == trend_direction).sum()
        return int(same_direction_count)
    
    def calc_noise_count(self, df: pd.DataFrame) -> int:
        """허용된 노이즈(반대 방향) 캔들 수"""
        duration = self.calc_duration_candles(df)
        core_count = self.calc_core_count(df)
        return max(0, duration - core_count - 1)  # -1은 첫 번째 diff가 NaN이기 때문
    
    def calc_direction_change(self, df: pd.DataFrame) -> int:
        """사이클 내에서 모멘텀 방향이 전환된 횟수"""
        if len(df) < 2:
            return 0
        
        hist_changes = df['macd_hist'].diff().dropna()
        if len(hist_changes) == 0:
            return 0
        
        directions = np.sign(hist_changes)
        direction_changes = (directions.diff() != 0).sum()
        return int(direction_changes)
    
    def calc_peak_price_position(self, df: pd.DataFrame) -> float:
        """사이클 내에서 최고가(high)를 기록한 캔들의 위치를 전체 사이클 길이로 나눈 값"""
        if len(df) == 0:
            return 0.5
        
        # 모든 캔들의 high 값 추출
        high_prices = df['high'].values
        
        # 최고가를 기록한 캔들의 인덱스 찾기
        peak_index = np.argmax(high_prices)
        
        # 위치 비율 계산 (0~1 사이 값)
        if len(high_prices) == 1:
            position_ratio = 0.5
        else:
            position_ratio = peak_index / (len(high_prices) - 1)
        
        return round(position_ratio, 4)
    
    def calc_trough_price_position(self, df: pd.DataFrame) -> float:
        """사이클 내에서 최저가(low)를 기록한 캔들의 위치를 전체 사이클 길이로 나눈 값"""
        if len(df) == 0:
            return 0.5
        
        # 모든 캔들의 low 값 추출
        low_prices = df['low'].values
        
        # 최저가를 기록한 캔들의 인덱스 찾기
        trough_index = np.argmin(low_prices)
        
        # 위치 비율 계산 (0~1 사이 값)
        if len(low_prices) == 1:
            position_ratio = 0.5
        else:
            position_ratio = trough_index / (len(low_prices) - 1)
        
        return round(position_ratio, 4)


    # ===== STRENGTH 카테고리 계산 함수들 =====
    def calc_direction_pct(self, df: pd.DataFrame) -> float:
        """핵심 캔들의 비율 (core_count / duration_candles)"""
        duration = self.calc_duration_candles(df)
        core_count = self.calc_core_count(df)
        return (core_count / duration * 100) if duration > 0 else 0.0
    
    def calc_hist_positive_ratio(self, df: pd.DataFrame) -> float:
        """MACD 히스토그램이 양수였던 캔들의 비율"""
        if len(df) == 0:
            return 0.0
        
        positive_count = (df['macd_hist'] > 0).sum()
        return (positive_count / len(df) * 100)
    
    def calc_price_up_ratio(self, df: pd.DataFrame) -> float:
        """가격이 상승한 (양봉) 캔들의 비율"""
        if len(df) == 0:
            return 0.0
        
        up_count = (df['close'] > df['open']).sum()
        return (up_count / len(df) * 100)
    
    def calc_price_down_ratio(self, df: pd.DataFrame) -> float:
        """가격이 하락한 (음봉) 캔들의 비율"""
        if len(df) == 0:
            return 0.0
        
        down_count = (df['close'] < df['open']).sum()
        return (down_count / len(df) * 100)
    
    # ===== START 카테고리 계산 함수들 =====
    def calc_start_price(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 종가"""
        return df['close'].iloc[0] if len(df) > 0 else 0.0
    
    def calc_start_volume(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 거래량"""
        return df['volume'].iloc[0] if len(df) > 0 else 0.0
    
    def calc_start_rsi(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 RSI 값"""
        return df['rsi'].iloc[0] if len(df) > 0 else 50.0
    
    def calc_start_macd(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 MACD 값"""
        return df['macd'].iloc[0] if len(df) > 0 else 0.0
    
    def calc_start_macd_signal(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 MACD Signal 값"""
        return df['macd_signal'].iloc[0] if len(df) > 0 else 0.0
    
    def calc_start_hist(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 MACD Histogram 값"""
        return df['macd_hist'].iloc[0] if len(df) > 0 else 0.0
    
    # ===== END 카테고리 계산 함수들 =====
    def calc_end_price(self, df: pd.DataFrame) -> float:
        """사이클 마지막 캔들의 종가"""
        return df['close'].iloc[-1] if len(df) > 0 else 0.0
    
    def calc_end_volume(self, df: pd.DataFrame) -> float:
        """사이클 마지막 캔들의 거래량"""
        return df['volume'].iloc[-1] if len(df) > 0 else 0.0
    
    def calc_end_rsi(self, df: pd.DataFrame) -> float:
        """사이클 마지막 캔들의 RSI 값"""
        return df['rsi'].iloc[-1] if len(df) > 0 else 50.0
    
    def calc_end_macd(self, df: pd.DataFrame) -> float:
        """사이클 마지막 캔들의 MACD 값"""
        return df['macd'].iloc[-1] if len(df) > 0 else 0.0
    
    def calc_end_macd_signal(self, df: pd.DataFrame) -> float:
        """사이클 마지막 캔들의 MACD Signal 값"""
        return df['macd_signal'].iloc[-1] if len(df) > 0 else 0.0
    
    def calc_end_hist(self, df: pd.DataFrame) -> float:
        """사이클 마지막 캔들의 MACD Histogram 값"""
        return df['macd_hist'].iloc[-1] if len(df) > 0 else 0.0
    
    # ===== CHANGE 카테고리 계산 함수들 =====
    def calc_price_change_pct(self, df: pd.DataFrame) -> float:
        """시작가 대비 종료가의 등락률"""
        if len(df) == 0:
            return 0.0
        
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        
        if start_price == 0:
            return 0.0
        
        return ((end_price - start_price) / start_price) * 100
    
    def calc_rsi_change(self, df: pd.DataFrame) -> float:
        """RSI 값의 변화량"""
        if len(df) == 0:
            return 0.0
        
        return df['rsi'].iloc[-1] - df['rsi'].iloc[0]
    
    def calc_macd_change(self, df: pd.DataFrame) -> float:
        """MACD 값의 변화량"""
        if len(df) == 0:
            return 0.0
        
        return df['macd'].iloc[-1] - df['macd'].iloc[0]
    
    def calc_macd_signal_change(self, df: pd.DataFrame) -> float:
        """MACD Signal 값의 변화량"""
        if len(df) == 0:
            return 0.0
        
        return df['macd_signal'].iloc[-1] - df['macd_signal'].iloc[0]
    
    def calc_macd_histogram_change(self, df: pd.DataFrame) -> float:
        """MACD Histogram 값의 변화량"""
        if len(df) == 0:
            return 0.0
        
        return df['macd_hist'].iloc[-1] - df['macd_hist'].iloc[0]
    
    # ===== VOLATILITY 카테고리 계산 함수들 =====
    def calc_max_high_pct(self, df: pd.DataFrame) -> float:
        """시작가 대비 사이클 내 최고가의 상승률"""
        if len(df) == 0:
            return 0.0
        
        start_price = df['close'].iloc[0]
        max_high = df['high'].max()
        
        if start_price == 0:
            return 0.0
        
        return max(0.0, ((max_high - start_price) / start_price) * 100)
    
    def calc_max_loss_pct(self, df: pd.DataFrame) -> float:
        """시작가 대비 사이클 내 최저가의 하락률"""
        if len(df) == 0:
            return 0.0
        
        start_price = df['close'].iloc[0]
        min_low = df['low'].min()
        
        if start_price == 0:
            return 0.0
        
        return min(0.0, ((min_low - start_price) / start_price) * 100)
    
    def calc_max_intraday_high_pct(self, df: pd.DataFrame) -> float:
        """캔들 내에서 발생한 최대 상승 변동률 (시간 순서 지킴)"""
        if len(df) < 2:
            return 0.0
        
        max_gain = 0.0
        
        for i in range(len(df) - 1):
            for j in range(i + 1, len(df)):
                current_price = df['close'].iloc[i]
                future_high = df['high'].iloc[j]
                
                if current_price > 0:
                    gain_pct = ((future_high - current_price) / current_price) * 100
                    max_gain = max(max_gain, gain_pct)
        
        return max_gain
    
    def calc_max_intraday_loss_pct(self, df: pd.DataFrame) -> float:
        """캔들 내에서 발생한 최대 하락 변동률 (시간 순서 지킴)"""
        if len(df) < 2:
            return 0.0
        
        max_loss = 0.0
        
        for i in range(len(df) - 1):
            for j in range(i + 1, len(df)):
                current_price = df['close'].iloc[i]
                future_low = df['low'].iloc[j]
                
                if current_price > 0:
                    loss_pct = ((future_low - current_price) / current_price) * 100
                    max_loss = min(max_loss, loss_pct)
        
        return max_loss
    
    def calc_avg_true_range(self, df: pd.DataFrame) -> float:
        """사이클 내 캔들의 평균 ATR 값 (평균 변동폭)"""
        if len(df) == 0:
            return 0.0
        
        # True Range 계산: max(high-low, |high-prev_close|, |low-prev_close|)
        atr_values = []
        
        for i in range(len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            
            if i == 0:
                # 첫 번째 캔들은 단순히 high - low
                tr = high - low
            else:
                prev_close = df['close'].iloc[i-1]
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
            
            atr_values.append(tr)
        
        return np.mean(atr_values)
    
    def calc_price_change_deviation(self, df: pd.DataFrame) -> float:
        """사이클 내 가격 변동률의 표준편차"""
        if len(df) < 2:
            return 0.0
        
        # 각 캔들의 종가 변화율 계산
        price_changes = df['close'].pct_change().dropna()
        
        if len(price_changes) == 0:
            return 0.0
        
        # 변화율의 표준편차를 퍼센트로 변환
        return float(price_changes.std() * 100)
    
    # ===== AGGREGATE 카테고리 계산 함수들 =====
    def calc_all_volume(self, df: pd.DataFrame) -> float:
        """사이클 내 모든 캔들의 거래량 총합"""
        return df['volume'].sum() if len(df) > 0 else 0.0
    
    def convert_legacy_features_to_categorized(self, legacy_features: Dict[str, Any]) -> Dict[str, Dict]:
        """기존 flat 구조의 특징들을 새로운 카테고리 구조로 변환"""
        categorized_features = self.config.get_default_cycle_features_structure()
        
        # 매핑 테이블
        legacy_mapping = {
            # shape
            'duration_candles': ('shape', 'duration_candles'),
            'core_count': ('shape', 'core_count'),
            'noise_count': ('shape', 'noise_count'),
            'direction_change': ('shape', 'direction_change'),
            'peak_price_position': ('shape', 'peak_price_position'),  # 새로 추가
            'trough_price_position': ('shape', 'trough_price_position'),  # 새로 추가
            
            # strength  
            'direction_pct': ('strength', 'direction_pct'),
            'hist_positive_ratio': ('strength', 'hist_positive_ratio'),
            'price_up_ratio': ('strength', 'price_up_ratio'),
            'price_down_ratio': ('strength', 'price_down_ratio'),
            
            # start
            'start_price': ('start', 'price'),
            'start_volume': ('start', 'volume'),
            'start_rsi': ('start', 'rsi'),
            'start_macd': ('start', 'macd'),
            'start_macd_signal': ('start', 'macd_signal'),
            'start_hist': ('start', 'hist'),
            
            # end
            'end_price': ('end', 'price'),
            'end_volume': ('end', 'volume'),
            'end_rsi': ('end', 'rsi'),
            'end_macd': ('end', 'macd'),
            'end_macd_signal': ('end', 'macd_signal'),
            'end_hist': ('end', 'hist'),
            
            # change
            'price_change_pct': ('change', 'price_pct'),
            'rsi_change': ('change', 'rsi'),
            'macd_change': ('change', 'macd'),
            'macd_signal_change': ('change', 'macd_signal'),
            'macd_histogram_change': ('change', 'hist'),
            
            # volatility
            'max_high_pct': ('volatility', 'max_high_pct'),
            'max_loss_pct': ('volatility', 'max_loss_pct'),
            'max_high_change': ('volatility', 'max_intraday_high_pct'),
            'max_loss_change': ('volatility', 'max_intraday_loss_pct'),
            'avg_true_range': ('volatility', 'avg_true_range'),
            'price_change_deviation': ('volatility', 'price_change_deviation'),
            
            # aggregate
            'all_volume': ('aggregate', 'volume')
        }
        
        # 기존 특징들을 새로운 구조로 매핑
        for legacy_name, value in legacy_features.items():
            if legacy_name in legacy_mapping:
                category, feature = legacy_mapping[legacy_name]
                if category in categorized_features:
                    categorized_features[category][feature] = value
        
        return categorized_features


class StructuredCycleProcessor:
    """구조화된 사이클 처리기 - 기존 파케이 파일을 새로운 구조로 변환"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.calculator = CycleFeatureCalculator()
        
    def convert_existing_cycles_to_new_structure(self, output_path: Optional[Path] = None):
        """기존 사이클 데이터를 새로운 구조로 변환"""
        print(f"기존 사이클 데이터 구조 변환 시작: {self.data_path}")
        
        try:
            # 기존 데이터 로드
            df = pd.read_parquet(self.data_path)
            print(f"로드된 사이클: {len(df)}개")
            
            converted_cycles = []
            
            for idx, row in df.iterrows():
                print(f"변환 진행: {idx+1}/{len(df)} ({((idx+1)/len(df)*100):.1f}%)", end='\r')
                
                # 기존 cycle_features 추출
                if isinstance(row['cycle_features'], dict):
                    legacy_features = row['cycle_features']
                else:
                    # 빈 특징으로 처리
                    legacy_features = {}
                
                # 캔들 데이터도 이용해서 재계산
                candle_data = row['candle_data']
                if isinstance(candle_data, (list, np.ndarray)) and len(candle_data) > 0:
                    # 리스트 형태의 캔들 데이터를 dict 형태로 변환
                    if isinstance(candle_data[0], dict):
                        candle_dict_list = candle_data
                    else:
                        # numpy array인 경우 dict로 변환 필요
                        candle_dict_list = []
                        for candle in candle_data:
                            if hasattr(candle, 'item'):  # numpy scalar
                                candle_dict_list.append({'close': float(candle.item())})
                            elif isinstance(candle, dict):
                                candle_dict_list.append(candle)
                            else:
                                candle_dict_list.append({'close': float(candle)})
                    
                    # 새로운 구조로 특징 계산
                    new_features = self.calculator.extract_features_from_candle_data(candle_dict_list)
                else:
                    # 기존 특징만으로 변환
                    new_features = self.calculator.convert_legacy_features_to_categorized(legacy_features)
                
                # 새로운 레코드 생성
                new_record = {
                    'cycle_id': row['cycle_id'],
                    'timeframe': row['timeframe'],
                    'start_date': row['start_date'],
                    'end_date': row['end_date'],
                    'cycle_type': row['cycle_type'],
                    'duration_candles': row['duration_candles'],
                    'category': row['category'],
                    'algorithm_used': row['algorithm_used'],
                    'candle_data': row['candle_data'],
                    'cycle_features': new_features  # 새로운 카테고리 구조
                }
                
                converted_cycles.append(new_record)
            
            print(f"\n변환 완료: {len(converted_cycles)}개 사이클")
            
            # 새로운 DataFrame 생성
            new_df = pd.DataFrame(converted_cycles)
            
            # 저장 경로 결정
            if output_path is None:
                output_path = self.data_path.with_name(f"converted_{self.data_path.name}")
            
            # 저장
            new_df.to_parquet(output_path, index=False)
            print(f"새로운 구조로 저장: {output_path}")
            
            return output_path, len(converted_cycles)
            
        except Exception as e:
            print(f"변환 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return None, 0
    
    def process_and_enrich_cycles(self, output_path: Optional[Path] = None):
        """
        사이클 감지만 완료된 'raw' 파일을 받아, 각 사이클의 특징을 계산하고
        'enriched' 파일을 저장하는 후처리 함수.
        """
        print(f"🚀 특징 계산 및 보강 시작: {self.data_path.name}")
        
        try:
            df = pd.read_parquet(self.data_path)
            print(f"로드된 사이클: {len(df)}개")

            # 'cycle_features' 컬럼을 업데이트하기 위해 새로운 리스트를 생성
            new_features_list = []

            for idx, row in df.iterrows():
                print(f"특징 계산 진행: {idx+1}/{len(df)} ({((idx+1)/len(df)*100):.1f}%)", end='\r')
                
                candle_data = row['candle_data']
                
                # candle_data로부터 모든 특징을 새로 계산
                if isinstance(candle_data, (list, np.ndarray)) and len(candle_data) > 0:
                    enriched_features = self.calculator.extract_features_from_candle_data(list(candle_data))
                else:
                    # 캔들 데이터가 없으면 기본 빈 구조 사용
                    enriched_features = self.calculator.config.get_default_cycle_features_structure()
                
                new_features_list.append(enriched_features)
            
            # 기존 DataFrame에 계산된 특징 컬럼을 통째로 교체
            df['cycle_features'] = new_features_list
            print(f"\n✅ 특징 계산 완료: {len(df)}개 사이클 보강")

            # 저장 경로 결정
            if output_path is None:
                # 파일명에 'enriched' 추가
                output_name = self.data_path.name.replace('.parquet', '_enriched.parquet')
                output_path = self.data_path.with_name(output_name)
            
            # 저장
            df.to_parquet(output_path, index=False)
            print(f"💾 보강된 파일 저장 완료: {output_path.name}")
            
            self.validate_new_structure(output_path)
            return output_path, len(df)

        except Exception as e:
            print(f"\n❌ 특징 계산 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return None, 0

    def validate_new_structure(self, converted_file: Path):
        """새로운 구조가 올바르게 변환되었는지 검증"""
        try:
            df = pd.read_parquet(converted_file)
            print(f"\n검증 시작: {len(df)}개 사이클")
            
            # 몇 개 샘플 검사
            for i in range(min(3, len(df))):
                cycle = df.iloc[i]
                features = cycle['cycle_features']
                
                print(f"\n사이클 {i+1} 구조 검증:")
                print(f"  ID: {cycle['cycle_id']}")
                
                if isinstance(features, dict):
                    for category, feature_dict in features.items():
                        if isinstance(feature_dict, dict):
                            print(f"  {category}: {len(feature_dict)}개 특징")
                        else:
                            print(f"  {category}: 구조 오류")
                else:
                    print(f"  특징 구조 오류: {type(features)}")
            
            print("✅ 구조 검증 완료")
            return True
            
        except Exception as e:
            print(f"검증 실패: {e}")
            return False


def convert_all_timeframes():
    """모든 타임프레임의 기존 데이터를 새로운 구조로 변환"""
    print("🔄 전체 타임프레임 데이터 구조 변환 시작")
    print("="*60)
    
    # 기존 데이터 디렉토리
    structured_path = project_root.parent / "data" / "cycle_data" / "structured"
    
    if not structured_path.exists():
        print(f"❌ 구조화된 데이터 디렉토리를 찾을 수 없습니다: {structured_path}")
        return
    
    # 기존 파케이 파일들 찾기
    parquet_files = list(structured_path.glob("cycles_*.parquet"))
    
    if not parquet_files:
        print("❌ 변환할 파케이 파일을 찾을 수 없습니다")
        return
    
    print(f"📁 발견된 파일: {len(parquet_files)}개")
    
    conversion_results = {}
    
    for file_path in parquet_files:
        # [추가] 이미 변환된 파일(v2, converted)은 건너뛰기
        if "_v2" in file_path.name or "converted_" in file_path.name:
            print(f"\n⏭️  건너뛰기 (이미 변환된 파일): {file_path.name}")
            continue
        print(f"\n🔄 변환 중: {file_path.name}")
        
        try:
            processor = StructuredCycleProcessor(file_path)
            
            # 변환 실행
            converted_path, cycle_count = processor.convert_existing_cycles_to_new_structure()
            
            if converted_path and cycle_count > 0:
                # 검증
                if processor.validate_new_structure(converted_path):
                    conversion_results[file_path.name] = {
                        'original_path': file_path,
                        'converted_path': converted_path,
                        'cycle_count': cycle_count,
                        'status': 'success'
                    }
                else:
                    conversion_results[file_path.name] = {
                        'status': 'validation_failed'
                    }
            else:
                conversion_results[file_path.name] = {
                    'status': 'conversion_failed'
                }
        
        except Exception as e:
            print(f"❌ {file_path.name} 변환 실패: {e}")
            conversion_results[file_path.name] = {
                'status': 'error',
                'error': str(e)
            }
    
    # 결과 요약
    print("\n" + "="*60)
    print("🎉 변환 결과 요약")
    print("="*60)
    
    successful_conversions = [k for k, v in conversion_results.items() if v.get('status') == 'success']
    failed_conversions = [k for k, v in conversion_results.items() if v.get('status') != 'success']
    
    print(f"✅ 성공: {len(successful_conversions)}개")
    for file_name in successful_conversions:
        result = conversion_results[file_name]
        print(f"   {file_name} -> {result['converted_path'].name} ({result['cycle_count']}개 사이클)")
    
    if failed_conversions:
        print(f"\n❌ 실패: {len(failed_conversions)}개")
        for file_name in failed_conversions:
            result = conversion_results[file_name]
            print(f"   {file_name}: {result.get('status', 'unknown')}")
    
    print("="*60)
    return conversion_results

def process_all_timeframes_for_enrichment():
    """모든 타임프레임의 감지된 데이터를 보강"""
    print("🔄 전체 타임프레임 특징 계산(보강) 시작")
    print("="*60)
    
    # feature_extract.py 파일의 위치를 기준으로 프로젝트 루트를 찾습니다.
    project_root = Path(__file__).resolve().parent.parent.parent
    structured_path = project_root / "data" / "cycle_data" / "structured"
    print(project_root)
    # '_enriched'가 붙지 않은 파일들을 대상으로 함
    raw_files = [p for p in structured_path.glob("cycles_*.parquet") if "_enriched" not in p.name and "converted_" not in p.name]
    
    if not raw_files:
        print("❌ 특징을 계산할 'raw' 사이클 파일을 찾을 수 없습니다.")
        print("💡 먼저 'macd_histogram_change_detect.py'를 실행하여 사이클을 감지해주세요.")
        return

    print(f"📁 처리 대상 파일: {len(raw_files)}개")
    
    for file_path in raw_files:
        processor = StructuredCycleProcessor(file_path)
        processor.process_and_enrich_cycles()

def test_new_structure():
    """새로운 구조 테스트"""
    print("🧪 새로운 특징 구조 테스트")
    
    # 샘플 캔들 데이터 생성
    sample_candles = [
        {
            'timestamp': '2024-01-01 00:00:00',
            'open': 42000,
            'high': 42500,
            'low': 41800,
            'close': 42150,
            'volume': 120.5,
            'macd': 235.67,
            'macd_signal': 280.90,
            'macd_hist': -45.23,
            'rsi': 55.2
        },
        {
            'timestamp': '2024-01-01 01:00:00',
            'open': 42150,
            'high': 43200,
            'low': 42000,
            'close': 43000,
            'volume': 150.2,
            'macd': 400.89,
            'macd_signal': 350.15,
            'macd_hist': 50.74,
            'rsi': 62.8
        },
        {
            'timestamp': '2024-01-01 02:00:00',
            'open': 43000,
            'high': 44000,
            'low': 42800,
            'close': 43890,
            'volume': 180.8,
            'macd': 445.89,
            'macd_signal': 390.15,
            'macd_hist': 55.74,
            'rsi': 65.8
        }
    ]
    
    # 특징 계산
    calculator = CycleFeatureCalculator()
    features = calculator.extract_features_from_candle_data(sample_candles)
    
    print("\n계산된 특징 구조:")
    print(json.dumps(features, indent=2, ensure_ascii=False))
    
    return features


if __name__ == "__main__":
    print("🚀 카테고리 기반 특징 추출기 시작")
    
    print("\n선택하세요:")
    print("1: 새로운 구조 테스트")
    print("2: 기존 데이터 구조 변환")
    print("3: [신규] 감지된 사이클 파일에 특징 계산 및 저장")
    print("4: 설정 관리")
    
    choice = input("선택 (1-4): ").strip()
    
    if choice == "1":
        test_new_structure()
    elif choice == "2":
        convert_all_timeframes()
    elif choice == "3":
        # 새로 추가된 기능 호출
        process_all_timeframes_for_enrichment()
    elif choice == "4":
        from .config import main as config_main
        config_main()
    else:
        print("잘못된 선택입니다.")