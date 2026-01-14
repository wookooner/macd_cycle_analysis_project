"""
확장된 특징 관리 시스템 - 새로운 특징 추가
===============================================
max_true_change, true_change 특징 추가
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Callable, Type
from abc import ABC, abstractmethod
import warnings
import json
import os
import inspect
import importlib
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

class FeatureCalculator(ABC):
    """특징 계산기 추상 클래스"""
    
    def __init__(self, name: str, category: str, description: str, unit: str = "ratio"):
        self.name = name
        self.category = category
        self.description = description
        self.unit = unit
        
    @abstractmethod
    def calculate(self, candle_data: Union[List[Dict[str, Any]], np.ndarray]) -> float:
        """특징 계산 메서드"""
        pass
    
    def validate_data(self, candle_data: Union[List, np.ndarray], min_length: int = 1) -> Optional[List[Dict[str, Any]]]:
        """캔들 데이터 검증 및 변환"""
        if candle_data is None:
            return None
        
        try:
            data_length = len(candle_data)
        except:
            return None
            
        if data_length < min_length:
            return None
        
        if hasattr(candle_data, 'tolist'):
            candle_data = candle_data.tolist()
        
        if not candle_data:
            return None
        
        return candle_data
    
    def safe_calculate(self, candle_data: Union[List[Dict[str, Any]], np.ndarray], default_value: float = 0.5) -> float:
        """안전한 계산 (오류 시 기본값 반환)"""
        try:
            return self.calculate(candle_data)
        except Exception as e:
            print(f"Warning: Error calculating {self.name}: {e}")
            return default_value
    
    def to_dict(self) -> Dict[str, Any]:
        """특징 정보를 딕셔너리로 반환"""
        return {
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'unit': self.unit
        }


# =============================================================================
# 기존 특징 계산기들
# =============================================================================

class PeakPricePositionCalculator(FeatureCalculator):
    """최고가 위치 비율 계산기"""
    
    def __init__(self):
        super().__init__(
            name="peak_price_position",
            category="shape",
            description="사이클 내에서 최고가(high)를 기록한 캔들의 위치를 전체 사이클 길이로 나눈 값",
            unit="ratio"
        )
    
    def calculate(self, candle_data: Union[List[Dict[str, Any]], np.ndarray]) -> float:
        validated_data = self.validate_data(candle_data, min_length=1)
        if validated_data is None:
            return 0.5
        
        high_prices = np.array([candle['high'] for candle in validated_data])
        
        if len(high_prices) == 0:
            return 0.5
        
        peak_index = np.argmax(high_prices)
        
        if len(high_prices) == 1:
            position_ratio = 0.5
        else:
            position_ratio = peak_index / (len(high_prices) - 1)
        
        return round(position_ratio, 4)


class TroughPricePositionCalculator(FeatureCalculator):
    """최저가 위치 비율 계산기"""
    
    def __init__(self):
        super().__init__(
            name="trough_price_position",
            category="shape",
            description="사이클 내에서 최저가(low)를 기록한 캔들의 위치를 전체 사이클 길이로 나눈 값",
            unit="ratio"
        )
    
    def calculate(self, candle_data: Union[List[Dict[str, Any]], np.ndarray]) -> float:
        validated_data = self.validate_data(candle_data, min_length=1)
        if validated_data is None:
            return 0.5
        
        low_prices = np.array([candle['low'] for candle in validated_data])
        
        if len(low_prices) == 0:
            return 0.5
        
        trough_index = np.argmin(low_prices)
        
        if len(low_prices) == 1:
            position_ratio = 0.5
        else:
            position_ratio = trough_index / (len(low_prices) - 1)
        
        return round(position_ratio, 4)


# =============================================================================
# 새로운 특징 계산기들
# =============================================================================

class MaxTrueChangeCalculator(FeatureCalculator):
    """Max True Change 계산기"""
    
    def __init__(self):
        super().__init__(
            name="max_true_change",
            category="volatility",
            description="사이클 최종 가격변화율의 절댓값을 사이클 내 각 캔들의 high-low 변동률 절댓값 합으로 나눈 값",
            unit="ratio"
        )
    
    def calculate(self, candle_data: Union[List[Dict[str, Any]], np.ndarray]) -> float:
        validated_data = self.validate_data(candle_data, min_length=2)
        if validated_data is None:
            return 0.5
        
        try:
            # 사이클 최종 가격변화율의 절댓값
            start_price = validated_data[0]['close']
            end_price = validated_data[-1]['close']
            
            if start_price == 0:
                return 0.5
            
            final_change_abs = abs((end_price - start_price) / start_price)
            
            # 각 캔들의 high-low 변동률의 절댓값 합
            high_low_changes_sum = 0.0
            for candle in validated_data:
                close_price = candle['close']
                high_price = candle['high']
                low_price = candle['low']
                
                if close_price > 0:
                    high_low_change = abs((high_price - low_price) / close_price)
                    high_low_changes_sum += high_low_change
            
            # 분모가 0인 경우 처리
            if high_low_changes_sum == 0:
                return 0.5
            
            # 비율 계산
            max_true_change = final_change_abs / high_low_changes_sum
            
            return round(max_true_change, 6)
            
        except Exception as e:
            print(f"Error in MaxTrueChangeCalculator: {e}")
            return 0.5


class TrueChangeCalculator(FeatureCalculator):
    """True Change 계산기"""
    
    def __init__(self):
        super().__init__(
            name="true_change",
            category="volatility",
            description="사이클 최종 가격변화율의 절댓값을 사이클 내 각 캔들의 open-close 변동률 절댓값 합으로 나눈 값",
            unit="ratio"
        )
    
    def calculate(self, candle_data: Union[List[Dict[str, Any]], np.ndarray]) -> float:
        validated_data = self.validate_data(candle_data, min_length=2)
        if validated_data is None:
            return 0.5
        
        try:
            # 사이클 최종 가격변화율의 절댓값
            start_price = validated_data[0]['close']
            end_price = validated_data[-1]['close']
            
            if start_price == 0:
                return 0.5
            
            final_change_abs = abs((end_price - start_price) / start_price)
            
            # 각 캔들의 open-close 변동률의 절댓값 합
            open_close_changes_sum = 0.0
            for candle in validated_data:
                open_price = candle['open']
                close_price = candle['close']
                
                if open_price > 0:
                    open_close_change = abs((close_price - open_price) / open_price)
                    open_close_changes_sum += open_close_change
            
            # 분모가 0인 경우 처리
            if open_close_changes_sum == 0:
                return 0.5
            
            # 비율 계산
            true_change = final_change_abs / open_close_changes_sum
            
            return round(true_change, 6)
            
        except Exception as e:
            print(f"Error in TrueChangeCalculator: {e}")
            return 0.5


# =============================================================================
# 레지스트리 및 매니저 클래스들
# =============================================================================

class FeatureRegistry:
    """특징 계산기 레지스트리 - 동적 클래스 관리"""
    
    def __init__(self):
        self._calculators: Dict[str, Type[FeatureCalculator]] = {}
        self._discover_builtin_calculators()
    
    def _discover_builtin_calculators(self):
        """내장 계산기 자동 탐지"""
        current_module = inspect.getmodule(inspect.currentframe())
        
        for name, obj in inspect.getmembers(current_module):
            if (inspect.isclass(obj) and 
                issubclass(obj, FeatureCalculator) and 
                obj is not FeatureCalculator):
                try:
                    instance = obj()
                    self._calculators[instance.name] = obj
                    print(f"Auto-discovered calculator: {instance.name}")
                except Exception as e:
                    print(f"Failed to register {name}: {e}")
    
    def register_calculator_class(self, calculator_class: Type[FeatureCalculator]):
        """계산기 클래스 등록"""
        try:
            instance = calculator_class()
            self._calculators[instance.name] = calculator_class
            print(f"Registered calculator class: {instance.name}")
        except Exception as e:
            print(f"Failed to register {calculator_class.__name__}: {e}")
    
    def unregister_calculator(self, feature_name: str):
        """계산기 등록 해제"""
        if feature_name in self._calculators:
            del self._calculators[feature_name]
            print(f"Unregistered calculator: {feature_name}")
        else:
            print(f"Calculator '{feature_name}' not found in registry")
    
    def get_available_features(self) -> List[str]:
        """등록된 모든 특징 이름 반환"""
        return list(self._calculators.keys())
    
    def get_calculator_class(self, feature_name: str) -> Optional[Type[FeatureCalculator]]:
        """특징 이름으로 계산기 클래스 반환"""
        return self._calculators.get(feature_name)
    
    def create_calculator(self, feature_name: str) -> Optional[FeatureCalculator]:
        """특징 이름으로 계산기 인스턴스 생성"""
        calculator_class = self.get_calculator_class(feature_name)
        if calculator_class:
            try:
                return calculator_class()
            except Exception as e:
                print(f"Failed to create calculator for {feature_name}: {e}")
        return None
    
    def get_features_by_category(self) -> Dict[str, List[str]]:
        """카테고리별 특징 그룹핑"""
        categories = {}
        for feature_name, calc_class in self._calculators.items():
            try:
                instance = calc_class()
                if instance.category not in categories:
                    categories[instance.category] = []
                categories[instance.category].append(feature_name)
            except:
                continue
        return categories
    
    def print_registry(self):
        """레지스트리 내용 출력"""
        print("\n" + "="*60)
        print("FEATURE CALCULATOR REGISTRY")
        print("="*60)
        
        if not self._calculators:
            print("No calculators registered.")
            return
        
        categories = self.get_features_by_category()
        for category, features in categories.items():
            print(f"\n[{category.upper()}] ({len(features)} calculators)")
            for feature_name in features:
                calc_class = self._calculators[feature_name]
                try:
                    instance = calc_class()
                    print(f"  • {feature_name}: {instance.description} ({instance.unit})")
                except:
                    print(f"  • {feature_name}: (error getting info)")
        
        print("="*60)


class FeatureManager:
    """개선된 특징 관리자"""
    
    def __init__(self, registry: Optional[FeatureRegistry] = None):
        self.registry = registry or FeatureRegistry()
        self.active_calculators: Dict[str, FeatureCalculator] = {}
    
    def get_available_features(self) -> List[str]:
        """레지스트리에서 사용 가능한 모든 특징 반환"""
        return self.registry.get_available_features()
    
    def activate_feature(self, feature_name: str) -> bool:
        """특징 활성화"""
        if feature_name in self.active_calculators:
            print(f"Feature '{feature_name}' already active")
            return True
        
        calculator = self.registry.create_calculator(feature_name)
        if calculator:
            self.active_calculators[feature_name] = calculator
            print(f"Activated feature: {feature_name}")
            return True
        else:
            print(f"Failed to activate feature: {feature_name}")
            return False
    
    def activate_features(self, feature_names: List[str]) -> List[str]:
        """여러 특징 일괄 활성화"""
        activated = []
        for feature_name in feature_names:
            if self.activate_feature(feature_name):
                activated.append(feature_name)
        return activated
    
    def activate_all_features(self) -> List[str]:
        """모든 사용 가능한 특징 활성화"""
        return self.activate_features(self.get_available_features())
    
    def activate_features_by_category(self, category: str) -> List[str]:
        """카테고리별 특징 활성화"""
        categories = self.registry.get_features_by_category()
        features = categories.get(category, [])
        return self.activate_features(features)
    
    def deactivate_feature(self, feature_name: str) -> bool:
        """특징 비활성화"""
        if feature_name in self.active_calculators:
            del self.active_calculators[feature_name]
            print(f"Deactivated feature: {feature_name}")
            return True
        else:
            print(f"Feature '{feature_name}' not active")
            return False
    
    def deactivate_all_features(self):
        """모든 특징 비활성화"""
        feature_names = list(self.active_calculators.keys())
        for feature_name in feature_names:
            self.deactivate_feature(feature_name)
    
    def get_active_features(self) -> List[str]:
        """활성화된 특징 목록 반환"""
        return list(self.active_calculators.keys())
    
    def calculate_features(self, candle_data: Union[List[Dict[str, Any]], np.ndarray], 
                          feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """활성화된 특징들 계산"""
        if feature_names is None:
            feature_names = self.get_active_features()
        
        results = {}
        for feature_name in feature_names:
            if feature_name in self.active_calculators:
                calculator = self.active_calculators[feature_name]
                results[feature_name] = calculator.safe_calculate(candle_data)
            else:
                print(f"Warning: Feature '{feature_name}' not active")
        
        return results
    
    def batch_calculate(self, df: pd.DataFrame, candle_col: str = 'candle_data',
                       feature_names: Optional[List[str]] = None, 
                       progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """배치로 모든 사이클에 특징 계산 적용"""
        if feature_names is None:
            feature_names = self.get_active_features()
        
        if not feature_names:
            print("Warning: No features to calculate. Activate features first.")
            return df
        
        result_df = df.copy()
        
        # 특징 컬럼 초기화
        for feature_name in feature_names:
            result_df[f'new_{feature_name}'] = 0.0
        
        # 배치 계산
        total_rows = len(df)
        for idx, row in df.iterrows():
            try:
                candle_data = row[candle_col]
                features = self.calculate_features(candle_data, feature_names)
                
                for feature_name, value in features.items():
                    result_df.at[idx, f'new_{feature_name}'] = value
                
                # 진행률 콜백
                if progress_callback and idx % 50 == 0:
                    progress_callback(idx + 1, total_rows)
                    
            except Exception as e:
                print(f"Warning: Error processing row {idx}: {e}")
                continue
        
        return result_df
    
    def get_feature_info(self, feature_name: str = None) -> Union[Dict, List[Dict]]:
        """특징 정보 반환"""
        if feature_name:
            if feature_name in self.active_calculators:
                return self.active_calculators[feature_name].to_dict()
            else:
                calculator = self.registry.create_calculator(feature_name)
                return calculator.to_dict() if calculator else {}
        else:
            active_info = [calc.to_dict() for calc in self.active_calculators.values()]
            return active_info
    
    def print_status(self):
        """현재 상태 출력"""
        print("\n" + "="*60)
        print("FEATURE MANAGER STATUS")
        print("="*60)
        
        active = self.get_active_features()
        available = self.get_available_features()
        
        print(f"Available features: {len(available)}")
        print(f"Active features: {len(active)}")
        
        if active:
            print("\nActive Features:")
            categories = {}
            for feature_name in active:
                calc = self.active_calculators[feature_name]
                if calc.category not in categories:
                    categories[calc.category] = []
                categories[calc.category].append(feature_name)
            
            for category, features in categories.items():
                print(f"\n[{category.upper()}]")
                for feature in features:
                    calc = self.active_calculators[feature]
                    print(f"  • {feature}: {calc.description}")
        else:
            print("\nNo features currently active.")
            print("Use activate_feature() or activate_all_features() to activate features.")
        
        print("="*60)


# 사용 예시 및 테스트
if __name__ == "__main__":
    print("Enhanced Feature Manager with New Calculators Test")
    print("="*60)
    
    # 레지스트리 및 매니저 초기화
    registry = FeatureRegistry()
    manager = FeatureManager(registry)
    
    # 레지스트리 내용 확인
    registry.print_registry()
    
    # 새로운 특징들만 활성화 (테스트 목적)
    new_features = ["max_true_change", "true_change"]
    manager.activate_features(new_features)
    
    # 상태 확인
    manager.print_status()
    
    # 테스트 데이터 생성
    test_candle_data = [
        {'open': 100, 'high': 105, 'low': 95, 'close': 102},
        {'open': 102, 'high': 110, 'low': 99, 'close': 108},
        {'open': 108, 'high': 112, 'low': 105, 'close': 106},
        {'open': 106, 'high': 109, 'low': 103, 'close': 104},
        {'open': 104, 'high': 107, 'low': 101, 'close': 105},
    ]
    
    # 특징 계산 테스트
    print("\nTest calculation with sample data:")
    results = manager.calculate_features(test_candle_data)
    
    print("Results:")
    for feature_name, value in results.items():
        print(f"  {feature_name}: {value}")
    
    print("\n✓ Enhanced Feature Manager setup completed with new features!")