"""
방향성 분석 특징 추출 시스템
================================
사이클 내에서 가격 변화의 방향성과 확실성을 분석하는 특징들
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class DirectionalFeature:
    """특징 정의 클래스"""
    name: str
    description: str
    calculator: callable
    category: str
    enabled: bool = True


class DirectionalFeatureCalculator:
    """방향성 분석 특징 계산기"""
    
    def __init__(self, cycle_type: str):
        self.cycle_type = cycle_type  # 'up' or 'down'
    
    def calculate_aligned_price_sum_pct(self, candle_data: List[Dict]) -> float:
        """방향 일치 누적 변화율
        
        상승 사이클: 양봉들의 변화율 합계
        하락 사이클: 음봉들의 변화율 합계
        """
        # NumPy 배열을 리스트로 변환
        if isinstance(candle_data, np.ndarray):
            candle_data = candle_data.tolist()
        
        if not candle_data or len(candle_data) == 0:
            return 0.0
        
        aligned_sum = 0.0
        for candle in candle_data:
            if not isinstance(candle, dict):
                continue
            
            close = candle.get('close', 0)
            open_price = candle.get('open', 0)
            
            if open_price == 0:
                continue
            
            price_change = (close - open_price) / open_price * 100
            
            if self.cycle_type == 'up' and price_change > 0:
                aligned_sum += price_change
            elif self.cycle_type == 'down' and price_change < 0:
                aligned_sum += abs(price_change)
        
        return float(aligned_sum)
    
    def calculate_misaligned_price_sum_pct(self, candle_data: List[Dict]) -> float:
        """방향 불일치 누적 변화율
        
        상승 사이클: 음봉들의 변화율 합계
        하락 사이클: 양봉들의 변화율 합계
        """
        if isinstance(candle_data, np.ndarray):
            candle_data = candle_data.tolist()
        
        if not candle_data or len(candle_data) == 0:
            return 0.0
        
        misaligned_sum = 0.0
        for candle in candle_data:
            if not isinstance(candle, dict):
                continue
            
            close = candle.get('close', 0)
            open_price = candle.get('open', 0)
            
            if open_price == 0:
                continue
            
            price_change = (close - open_price) / open_price * 100
            
            if self.cycle_type == 'up' and price_change < 0:
                misaligned_sum += abs(price_change)
            elif self.cycle_type == 'down' and price_change > 0:
                misaligned_sum += price_change
        
        return float(misaligned_sum)
    
    def calculate_direction_efficiency(self, candle_data: List[Dict]) -> float:
        """방향 효율성
        
        aligned / (aligned + misaligned)
        1에 가까울수록 깔끔한 추세
        """
        aligned = self.calculate_aligned_price_sum_pct(candle_data)
        misaligned = self.calculate_misaligned_price_sum_pct(candle_data)
        
        total = aligned + misaligned
        if total == 0:
            return 0.0
        
        return aligned / total
    
    def calculate_max_aligned_streak(self, candle_data: List[Dict]) -> int:
        """최대 연속 일치 구간 길이"""
        if isinstance(candle_data, np.ndarray):
            candle_data = candle_data.tolist()
        
        if not candle_data or len(candle_data) == 0:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for candle in candle_data:
            if not isinstance(candle, dict):
                continue
            
            close = candle.get('close', 0)
            open_price = candle.get('open', 0)
            price_change = close - open_price
            
            is_aligned = (self.cycle_type == 'up' and price_change > 0) or \
                        (self.cycle_type == 'down' and price_change < 0)
            
            if is_aligned:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return int(max_streak)
    
    def calculate_max_aligned_streak_change(self, candle_data: List[Dict]) -> float:
        """최대 연속 일치 구간의 누적 변화율"""
        if isinstance(candle_data, np.ndarray):
            candle_data = candle_data.tolist()
        
        if not candle_data or len(candle_data) == 0:
            return 0.0
        
        max_streak_change = 0.0
        current_streak_change = 0.0
        
        for candle in candle_data:
            if not isinstance(candle, dict):
                continue
            
            close = candle.get('close', 0)
            open_price = candle.get('open', 0)
            
            if open_price == 0:
                continue
            
            price_change_pct = (close - open_price) / open_price * 100
            
            is_aligned = (self.cycle_type == 'up' and price_change_pct > 0) or \
                        (self.cycle_type == 'down' and price_change_pct < 0)
            
            if is_aligned:
                current_streak_change += abs(price_change_pct)
                max_streak_change = max(max_streak_change, current_streak_change)
            else:
                current_streak_change = 0.0
        
        return float(max_streak_change)
    
    def calculate_aligned_candle_ratio(self, candle_data: List[Dict]) -> float:
        """방향 일치 캔들 비율"""
        if isinstance(candle_data, np.ndarray):
            candle_data = candle_data.tolist()
        
        if not candle_data or len(candle_data) == 0:
            return 0.0
        
        aligned_count = 0
        total_count = 0
        
        for candle in candle_data:
            if not isinstance(candle, dict):
                continue
            
            close = candle.get('close', 0)
            open_price = candle.get('open', 0)
            price_change = close - open_price
            
            is_aligned = (self.cycle_type == 'up' and price_change > 0) or \
                        (self.cycle_type == 'down' and price_change < 0)
            
            if is_aligned:
                aligned_count += 1
            total_count += 1
        
        if total_count == 0:
            return 0.0
        
        return float(aligned_count / total_count)
    
    def calculate_phase_alignment(self, candle_data: List[Dict]) -> Dict[str, float]:
        """구간별 방향 일치도
        
        Returns:
            early_phase_alignment: 초반 1/3 구간
            mid_phase_alignment: 중반 1/3 구간
            late_phase_alignment: 후반 1/3 구간
        """
        if isinstance(candle_data, np.ndarray):
            candle_data = candle_data.tolist()
        
        if not candle_data or len(candle_data) < 3:
            return {
                'early_phase_alignment': 0.0,
                'mid_phase_alignment': 0.0,
                'late_phase_alignment': 0.0
            }
        
        n = len(candle_data)
        third = n // 3
        
        phases = {
            'early': candle_data[:third] if third > 0 else candle_data[:1],
            'mid': candle_data[third:2*third] if third > 0 else candle_data[1:2] if n > 1 else [],
            'late': candle_data[2*third:] if third > 0 else candle_data[2:] if n > 2 else []
        }
        
        results = {}
        for phase_name, phase_data in phases.items():
            if not phase_data:
                results[f'{phase_name}_phase_alignment'] = 0.0
                continue
            
            aligned_count = 0
            total_count = 0
            
            for candle in phase_data:
                if not isinstance(candle, dict):
                    continue
                
                close = candle.get('close', 0)
                open_price = candle.get('open', 0)
                price_change = close - open_price
                
                is_aligned = (self.cycle_type == 'up' and price_change > 0) or \
                            (self.cycle_type == 'down' and price_change < 0)
                if is_aligned:
                    aligned_count += 1
                total_count += 1
            
            if total_count == 0:
                results[f'{phase_name}_phase_alignment'] = 0.0
            else:
                results[f'{phase_name}_phase_alignment'] = float(aligned_count / total_count)
        
        return results
    
    def calculate_dominant_phase(self, candle_data: List[Dict]) -> str:
        """가장 강한 방향 일치를 보인 구간
        
        Returns:
            'early', 'mid', 'late' 중 하나
        """
        phase_alignment = self.calculate_phase_alignment(candle_data)
        
        max_phase = max(phase_alignment.items(), key=lambda x: x[1])
        return max_phase[0].replace('_phase_alignment', '')
    
    def calculate_alignment_concentration(self, candle_data: List[Dict]) -> float:
        """방향 일치 변화의 집중도
        
        상위 50% 방향 일치 캔들이 전체 방향 일치 변화의 몇 %를 차지하는지
        """
        if isinstance(candle_data, np.ndarray):
            candle_data = candle_data.tolist()
        
        if not candle_data or len(candle_data) == 0:
            return 0.0
        
        aligned_changes = []
        for candle in candle_data:
            if not isinstance(candle, dict):
                continue
            
            close = candle.get('close', 0)
            open_price = candle.get('open', 0)
            
            if open_price == 0:
                continue
            
            price_change_pct = (close - open_price) / open_price * 100
            
            is_aligned = (self.cycle_type == 'up' and price_change_pct > 0) or \
                        (self.cycle_type == 'down' and price_change_pct < 0)
            
            if is_aligned:
                aligned_changes.append(abs(price_change_pct))
        
        if not aligned_changes:
            return 0.0
        
        aligned_changes.sort(reverse=True)
        top_half = aligned_changes[:len(aligned_changes)//2] if len(aligned_changes) > 1 else aligned_changes
        
        total_aligned = sum(aligned_changes)
        if total_aligned == 0:
            return 0.0
        
        return float(sum(top_half) / total_aligned)
    
    def calculate_directional_momentum_consistency(self, candle_data: List[Dict]) -> float:
        """방향성 모멘텀 일관성
        
        MACD 히스토그램과 가격 방향이 일치하는 비율
        """
        if isinstance(candle_data, np.ndarray):
            candle_data = candle_data.tolist()
        
        if not candle_data or len(candle_data) == 0:
            return 0.0
        
        consistent_count = 0
        total_count = 0
        
        for candle in candle_data:
            if not isinstance(candle, dict):
                continue
            
            close = candle.get('close', 0)
            open_price = candle.get('open', 0)
            macd_hist = candle.get('macd_hist', 0)
            
            price_change = close - open_price
            
            # 가격과 MACD 히스토그램 방향 일치 확인
            price_up = price_change > 0
            macd_up = macd_hist > 0
            
            if price_up == macd_up:
                consistent_count += 1
            total_count += 1
        
        if total_count == 0:
            return 0.0
        
        return float(consistent_count / total_count)


class DirectionalFeatureManager:
    """방향성 특징 관리자"""
    
    def __init__(self):
        self.features = self._initialize_features()
    
    def _initialize_features(self) -> List[DirectionalFeature]:
        """특징 목록 초기화"""
        return [
            DirectionalFeature(
                name="aligned_price_sum_pct",
                description="방향 일치 누적 변화율",
                calculator=lambda calc, data: calc.calculate_aligned_price_sum_pct(data),
                category="directional"
            ),
            DirectionalFeature(
                name="misaligned_price_sum_pct",
                description="방향 불일치 누적 변화율",
                calculator=lambda calc, data: calc.calculate_misaligned_price_sum_pct(data),
                category="directional"
            ),
            DirectionalFeature(
                name="direction_efficiency",
                description="방향 효율성 (0~1)",
                calculator=lambda calc, data: calc.calculate_direction_efficiency(data),
                category="directional"
            ),
            DirectionalFeature(
                name="max_aligned_streak",
                description="최대 연속 일치 구간 길이",
                calculator=lambda calc, data: calc.calculate_max_aligned_streak(data),
                category="directional"
            ),
            DirectionalFeature(
                name="max_aligned_streak_change",
                description="최대 연속 일치 구간의 누적 변화율",
                calculator=lambda calc, data: calc.calculate_max_aligned_streak_change(data),
                category="directional"
            ),
            DirectionalFeature(
                name="aligned_candle_ratio",
                description="방향 일치 캔들 비율",
                calculator=lambda calc, data: calc.calculate_aligned_candle_ratio(data),
                category="directional"
            ),
            DirectionalFeature(
                name="early_phase_alignment",
                description="초반 1/3 구간의 방향 일치도",
                calculator=lambda calc, data: calc.calculate_phase_alignment(data)['early_phase_alignment'],
                category="phase"
            ),
            DirectionalFeature(
                name="mid_phase_alignment",
                description="중반 1/3 구간의 방향 일치도",
                calculator=lambda calc, data: calc.calculate_phase_alignment(data)['mid_phase_alignment'],
                category="phase"
            ),
            DirectionalFeature(
                name="late_phase_alignment",
                description="후반 1/3 구간의 방향 일치도",
                calculator=lambda calc, data: calc.calculate_phase_alignment(data)['late_phase_alignment'],
                category="phase"
            ),
            DirectionalFeature(
                name="dominant_phase",
                description="가장 강한 방향 일치를 보인 구간",
                calculator=lambda calc, data: calc.calculate_dominant_phase(data),
                category="phase"
            ),
            DirectionalFeature(
                name="alignment_concentration",
                description="방향 일치 변화의 집중도",
                calculator=lambda calc, data: calc.calculate_alignment_concentration(data),
                category="directional"
            ),
            DirectionalFeature(
                name="directional_momentum_consistency",
                description="방향성 모멘텀 일관성",
                calculator=lambda calc, data: calc.calculate_directional_momentum_consistency(data),
                category="directional"
            )
        ]
    
    def calculate_all_features(self, cycle_data: Dict) -> Dict[str, Any]:
        """모든 활성화된 특징 계산"""
        cycle_type = cycle_data.get('cycle_type', 'up')
        candle_data = cycle_data.get('candle_data', [])
        
        calculator = DirectionalFeatureCalculator(cycle_type)
        results = {}
        
        for feature in self.features:
            if feature.enabled:
                try:
                    value = feature.calculator(calculator, candle_data)
                    results[f"new_{feature.name}"] = value
                except Exception as e:
                    print(f"Error calculating {feature.name}: {e}")
                    results[f"new_{feature.name}"] = None
        
        return results
    
    def batch_calculate(self, df: pd.DataFrame, candle_column: str = 'candle_data') -> pd.DataFrame:
        """배치 계산"""
        enriched_df = df.copy()
        
        # 새 특징 컬럼 초기화
        for feature in self.features:
            if feature.enabled:
                col_name = f"new_{feature.name}"
                # dominant_phase는 문자열, 나머지는 float
                if feature.name == "dominant_phase":
                    enriched_df[col_name] = ""
                else:
                    enriched_df[col_name] = 0.0
        
        # 각 사이클별 계산
        for idx, row in enriched_df.iterrows():
            cycle_data = {
                'cycle_type': row.get('cycle_type', 'up'),
                'candle_data': row.get(candle_column, [])
            }
            
            features = self.calculate_all_features(cycle_data)
            
            for key, value in features.items():
                if key in enriched_df.columns:
                    enriched_df.at[idx, key] = value if value is not None else (0.0 if key != "new_dominant_phase" else "")
        
        return enriched_df
    
    def get_feature_list(self) -> List[str]:
        """활성화된 특징 목록 반환"""
        return [f"new_{f.name}" for f in self.features if f.enabled]
    
    def get_feature_info(self) -> pd.DataFrame:
        """특징 정보 DataFrame 반환"""
        info = []
        for feature in self.features:
            info.append({
                'Feature': f"new_{feature.name}",
                'Description': feature.description,
                'Category': feature.category,
                'Enabled': feature.enabled
            })
        return pd.DataFrame(info)


def flatten_existing_features(df: pd.DataFrame) -> pd.DataFrame:
    """기존 cycle_features를 평면화"""
    enriched_df = df.copy()
    
    for idx, row in df.iterrows():
        if 'cycle_features' in row and isinstance(row['cycle_features'], dict):
            features = row['cycle_features']
            flat_features = _flatten_dict(features, prefix='existing')
            
            if idx == 0:
                for col in flat_features.keys():
                    if col not in enriched_df.columns:
                        enriched_df[col] = 0.0
            
            for col, value in flat_features.items():
                if col in enriched_df.columns:
                    enriched_df.at[idx, col] = value if value is not None else 0.0
    
    return enriched_df


def _flatten_dict(d: Dict, prefix: str = '') -> Dict:
    """중첩 딕셔너리 평면화"""
    flattened = {}
    for key, value in d.items():
        new_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, new_key))
        else:
            flattened[new_key] = value
    return flattened