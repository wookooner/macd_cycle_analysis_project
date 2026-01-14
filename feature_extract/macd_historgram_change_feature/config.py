"""
config.py
카테고리 기반 특징 정의 및 관리 설정 (새로운 구조)
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import pyarrow.parquet as pq
import sys
import json
from pathlib import Path

class FeatureConfig:
    """카테고리 기반 특징 설정 관리 클래스"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        
        # 7개 카테고리로 구성된 특징 정의
        self.FEATURE_CATEGORIES = {
            'shape': {
                'description': '형태/구조 - 사이클의 기본적인 골격과 구조적 특성',
                'features': {
                    'duration_candles': {
                        'description': '사이클의 전체 길이 (캔들 수)',
                        'calculator': 'calc_duration_candles',
                        'enabled': True,
                        'default_value': 0,
                        'data_type': 'int'
                    },
                    'core_count': {
                        'description': '사이클 방향과 일치하는 핵심 캔들 수',
                        'calculator': 'calc_core_count',
                        'enabled': True,
                        'default_value': 0,
                        'data_type': 'int'
                    },
                    'noise_count': {
                        'description': '허용된 노이즈(반대 방향) 캔들 수',
                        'calculator': 'calc_noise_count',
                        'enabled': True,
                        'default_value': 0,
                        'data_type': 'int'
                    },
                    'direction_change': {
                        'description': '사이클 내에서 모멘텀 방향이 전환된 횟수',
                        'calculator': 'calc_direction_change',
                        'enabled': True,
                        'default_value': 0,
                        'data_type': 'int'
                    },
                    'peak_price_position': {
                        'description': '사이클 내에서 최고가(high)를 기록한 캔들의 위치를 전체 사이클 길이로 나눈 값',
                        'calculator': 'calc_peak_price_position',
                        'enabled': True,
                        'default_value': 0.5,
                        'data_type': 'float'
                    },
                    'trough_price_position': {
                        'description': '사이클 내에서 최저가(low)를 기록한 캔들의 위치를 전체 사이클 길이로 나눈 값',
                        'calculator': 'calc_trough_price_position',
                        'enabled': True,
                        'default_value': 0.5,
                        'data_type': 'float'
                    }
                }
            },
            
            'strength': {
                'description': '강도/견고함 - 사이클의 추세가 얼마나 일관되고 견고했는지',
                'features': {
                    'direction_pct': {
                        'description': '핵심 캔들의 비율 (core_count / duration_candles)',
                        'calculator': 'calc_direction_pct',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'hist_positive_ratio': {
                        'description': 'MACD 히스토그램이 양수였던 캔들의 비율',
                        'calculator': 'calc_hist_positive_ratio',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'price_up_ratio': {
                        'description': '가격이 상승한 (양봉) 캔들의 비율',
                        'calculator': 'calc_price_up_ratio',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'price_down_ratio': {
                        'description': '가격이 하락한 (음봉) 캔들의 비율',
                        'calculator': 'calc_price_down_ratio',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    }
                }
            },
            
            'start': {
                'description': '시작점 지표 - 사이클 시작 시점의 시장 상태',
                'features': {
                    'price': {
                        'description': '사이클 첫 캔들의 종가',
                        'calculator': 'calc_start_price',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'volume': {
                        'description': '사이클 첫 캔들의 거래량',
                        'calculator': 'calc_start_volume',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'rsi': {
                        'description': '사이클 첫 캔들의 RSI 값',
                        'calculator': 'calc_start_rsi',
                        'enabled': True,
                        'default_value': 50.0,
                        'data_type': 'float'
                    },
                    'macd': {
                        'description': '사이클 첫 캔들의 MACD 값',
                        'calculator': 'calc_start_macd',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'macd_signal': {
                        'description': '사이클 첫 캔들의 MACD Signal 값',
                        'calculator': 'calc_start_macd_signal',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'hist': {
                        'description': '사이클 첫 캔들의 MACD Histogram 값',
                        'calculator': 'calc_start_hist',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    }
                }
            },
            
            'end': {
                'description': '종료점 지표 - 사이클 종료 시점의 시장 상태',
                'features': {
                    'price': {
                        'description': '사이클 마지막 캔들의 종가',
                        'calculator': 'calc_end_price',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'volume': {
                        'description': '사이클 마지막 캔들의 거래량',
                        'calculator': 'calc_end_volume',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'rsi': {
                        'description': '사이클 마지막 캔들의 RSI 값',
                        'calculator': 'calc_end_rsi',
                        'enabled': True,
                        'default_value': 50.0,
                        'data_type': 'float'
                    },
                    'macd': {
                        'description': '사이클 마지막 캔들의 MACD 값',
                        'calculator': 'calc_end_macd',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'macd_signal': {
                        'description': '사이클 마지막 캔들의 MACD Signal 값',
                        'calculator': 'calc_end_macd_signal',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'hist': {
                        'description': '사이클 마지막 캔들의 MACD Histogram 값',
                        'calculator': 'calc_end_hist',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    }
                }
            },
            
            'change': {
                'description': '시작-종료 변화량 - 사이클 시작과 끝을 비교한 순수 변화량',
                'features': {
                    'price_pct': {
                        'description': '시작가 대비 종료가의 등락률',
                        'calculator': 'calc_price_change_pct',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'rsi': {
                        'description': 'RSI 값의 변화량',
                        'calculator': 'calc_rsi_change',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'macd': {
                        'description': 'MACD 값의 변화량',
                        'calculator': 'calc_macd_change',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'macd_signal': {
                        'description': 'MACD Signal 값의 변화량',
                        'calculator': 'calc_macd_signal_change',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'hist': {
                        'description': 'MACD Histogram 값의 변화량',
                        'calculator': 'calc_macd_histogram_change',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    }
                }
            },
            
            'volatility': {
                'description': '내부 변동성/극값 - 사이클 진행 중 발생한 가격 움직임의 크기 및 변동성',
                'features': {
                    'max_high_pct': {
                        'description': '시작가 대비 사이클 내 최고가의 상승률',
                        'calculator': 'calc_max_high_pct',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'max_loss_pct': {
                        'description': '시작가 대비 사이클 내 최저가의 하락률',
                        'calculator': 'calc_max_loss_pct',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'max_intraday_high_pct': {
                        'description': '캔들 내에서 발생한 최대 상승 변동률',
                        'calculator': 'calc_max_intraday_high_pct',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'max_intraday_loss_pct': {
                        'description': '캔들 내에서 발생한 최대 하락 변동률',
                        'calculator': 'calc_max_intraday_loss_pct',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'avg_true_range': {
                        'description': '사이클 내 캔들의 평균 ATR 값 (평균 변동폭)',
                        'calculator': 'calc_avg_true_range',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    },
                    'price_change_deviation': {
                        'description': '사이클 내 가격 변동률의 표준편차',
                        'calculator': 'calc_price_change_deviation',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    }
                }
            },
            
            'aggregate': {
                'description': '전체 집계 - 사이클 전체 기간의 데이터를 합산한 값',
                'features': {
                    'volume': {
                        'description': '사이클 내 모든 캔들의 거래량 총합',
                        'calculator': 'calc_all_volume',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float'
                    }
                }
            }
        }
        
        # 계산 설정
        self.CALCULATION_CONFIG = {
            'batch_size': 1000,
            'handle_errors': True,
            'use_default_on_error': True,
        }

        self.import_config()
    
    def get_all_features_flat(self) -> Dict[str, Dict[str, Any]]:
        """모든 특징을 flat 구조로 반환 (기존 호환성용)"""
        flat_features = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            for feature_name, feature_config in category_data['features'].items():
                # 카테고리 prefix 추가
                flat_name = f"{category_name}_{feature_name}"
                flat_features[flat_name] = {
                    'description': feature_config['description'],
                    'calculator': feature_config['calculator'],
                    'enabled': feature_config['enabled'],
                    'default_value': feature_config['default_value'],
                    'category': category_name,
                    'feature_key': feature_name
                }
        return flat_features
    
    def get_enabled_features_by_category(self) -> Dict[str, Dict[str, Dict]]:
        """카테고리별로 활성화된 특징들 반환"""
        enabled_by_category = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            enabled_features = {
                name: config for name, config in category_data['features'].items() 
                if config['enabled']
            }
            if enabled_features:
                enabled_by_category[category_name] = enabled_features
        return enabled_by_category
    
    def get_feature_names_by_category(self) -> Dict[str, List[str]]:
        """카테고리별 활성화된 특징 이름 리스트"""
        names_by_category = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            feature_names = [
                name for name, config in category_data['features'].items() 
                if config['enabled']
            ]
            if feature_names:
                names_by_category[category_name] = feature_names
        return names_by_category
    
    def get_all_calculator_names(self) -> List[str]:
        """모든 활성화된 계산함수 이름 리스트"""
        calculators = []
        for category_data in self.FEATURE_CATEGORIES.values():
            for feature_config in category_data['features'].values():
                if feature_config['enabled']:
                    calculators.append(feature_config['calculator'])
        return calculators
    
    def enable_feature(self, category_name: str, feature_name: str) -> bool:
        """특정 카테고리의 특징 활성화"""
        if category_name in self.FEATURE_CATEGORIES:
            if feature_name in self.FEATURE_CATEGORIES[category_name]['features']:
                self.FEATURE_CATEGORIES[category_name]['features'][feature_name]['enabled'] = True
                print(f"✅ 특징 '{category_name}.{feature_name}'을 활성화했습니다.")
                return True
        
        print(f"❌ 존재하지 않는 특징입니다: '{category_name}.{feature_name}'")
        return False
    
    def disable_feature(self, category_name: str, feature_name: str) -> bool:
        """특정 카테고리의 특징 비활성화"""
        if category_name in self.FEATURE_CATEGORIES:
            if feature_name in self.FEATURE_CATEGORIES[category_name]['features']:
                self.FEATURE_CATEGORIES[category_name]['features'][feature_name]['enabled'] = False
                print(f"❌ 특징 '{category_name}.{feature_name}'을 비활성화했습니다.")
                return True
        
        print(f"❌ 존재하지 않는 특징입니다: '{category_name}.{feature_name}'")
        return False
    
    def get_default_cycle_features_structure(self) -> Dict[str, Dict]:
        """기본 사이클 특징 구조 생성"""
        structure = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            structure[category_name] = {}
            for feature_name, feature_config in category_data['features'].items():
                if feature_config['enabled']:
                    structure[category_name][feature_name] = feature_config['default_value']
        return structure
    
    def validate_calculator_functions(self, calculator_module: Any) -> Dict[str, Dict[str, bool]]:
        """카테고리별로 계산함수가 존재하는지 검증"""
        validation_results = {}
        
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            validation_results[category_name] = {}
            for feature_name, feature_config in category_data['features'].items():
                if feature_config['enabled']:
                    calculator_name = feature_config['calculator']
                    validation_results[category_name][feature_name] = hasattr(calculator_module, calculator_name)
        
        return validation_results
    
    def print_feature_summary(self):
        """카테고리별 특징 요약 출력"""
        print("\n" + "="*80)
        print("📋 카테고리별 특징 구성 현황")
        print("="*80)
        
        total_features = 0
        enabled_features = 0
        
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            features = category_data['features']
            enabled_count = sum(1 for config in features.values() if config['enabled'])
            total_count = len(features)
            
            total_features += total_count
            enabled_features += enabled_count
            
            print(f"\n📁 {category_name.upper()} ({category_data['description']})")
            print(f"   활성화: {enabled_count}/{total_count}개")
            
            for feature_name, feature_config in features.items():
                status = "✅" if feature_config['enabled'] else "❌"
                print(f"   {status} {feature_name}: {feature_config['description']}")
        
        print("\n" + "="*80)
        print(f"📊 전체 요약: {enabled_features}/{total_features}개 특징 활성화")
        print("="*80)
    
    def export_config(self):
        """현재 설정을 JSON 파일로 내보내기"""
        export_data = {
            'feature_categories': self.FEATURE_CATEGORIES,
            'calculation_config': self.CALCULATION_CONFIG,
            'version': '2.0',
            'structure_type': 'categorized'
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def import_config(self):
        """JSON 파일에서 설정 가져오기"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
                
                if 'feature_categories' in import_data:
                    # 기존 설정과 merge
                    for category_name, category_data in import_data['feature_categories'].items():
                        if category_name in self.FEATURE_CATEGORIES:
                            for feature_name, feature_config in category_data['features'].items():
                                if feature_name in self.FEATURE_CATEGORIES[category_name]['features']:
                                    # enabled 상태만 업데이트
                                    self.FEATURE_CATEGORIES[category_name]['features'][feature_name]['enabled'] = feature_config['enabled']
                
                if 'calculation_config' in import_data:
                    self.CALCULATION_CONFIG = import_data['calculation_config']
                
                return True
                
            except Exception as e:
                print(f"❌ 설정 가져오기 실패: {e}")
                return False
        return False

# 전역 설정 인스턴스
DEFAULT_CONFIG_PATH = Path(__file__).parent / "features_config_v2.json"
DEFAULT_CONFIG = FeatureConfig(DEFAULT_CONFIG_PATH)

def main():
    """특징 관리 메인 함수 (카테고리 기반)"""
    print("🚀 카테고리 기반 특징 관리 스크립트 시작")
    
    while True:
        DEFAULT_CONFIG.print_feature_summary()
        
        print("\n[메뉴]")
        print("1: 카테고리별 특징 활성화/비활성화")
        print("2: 전체 특징 일괄 활성화")  
        print("3: 전체 특징 일괄 비활성화")
        print("4: 카테고리별 통계 보기")
        print("5: 종료")
        
        choice = input("명령을 입력하세요 (1-5): ").strip()
        
        if choice == '5':
            print("프로그램을 종료합니다.")
            break
        
        elif choice == '1':
            print("\n📁 사용 가능한 카테고리:")
            for i, (category_name, category_data) in enumerate(DEFAULT_CONFIG.FEATURE_CATEGORIES.items(), 1):
                enabled_count = sum(1 for config in category_data['features'].values() if config['enabled'])
                total_count = len(category_data['features'])
                print(f"   {i}. {category_name} ({enabled_count}/{total_count}개 활성화)")
            
            category_input = input("카테고리 이름을 입력하세요: ").strip()
            if category_input in DEFAULT_CONFIG.FEATURE_CATEGORIES:
                features = DEFAULT_CONFIG.FEATURE_CATEGORIES[category_input]['features']
                
                print(f"\n📋 {category_input} 카테고리 특징들:")
                for feature_name, feature_config in features.items():
                    status = "✅" if feature_config['enabled'] else "❌"
                    print(f"   {status} {feature_name}")
                
                feature_input = input("특징 이름을 입력하세요: ").strip()
                if feature_input in features:
                    current_status = features[feature_input]['enabled']
                    if current_status:
                        DEFAULT_CONFIG.disable_feature(category_input, feature_input)
                    else:
                        DEFAULT_CONFIG.enable_feature(category_input, feature_input)
                else:
                    print(f"❌ 존재하지 않는 특징입니다: {feature_input}")
            else:
                print(f"❌ 존재하지 않는 카테고리입니다: {category_input}")
        
        elif choice == '2':
            for category_name, category_data in DEFAULT_CONFIG.FEATURE_CATEGORIES.items():
                for feature_name in category_data['features'].keys():
                    DEFAULT_CONFIG.enable_feature(category_name, feature_name)
            print("✅ 모든 특징이 활성화되었습니다.")
        
        elif choice == '3':
            for category_name, category_data in DEFAULT_CONFIG.FEATURE_CATEGORIES.items():
                for feature_name in category_data['features'].keys():
                    DEFAULT_CONFIG.disable_feature(category_name, feature_name)
            print("❌ 모든 특징이 비활성화되었습니다.")
        
        elif choice == '4':
            enabled_by_category = DEFAULT_CONFIG.get_enabled_features_by_category()
            print("\n📊 카테고리별 활성화된 특징 통계:")
            for category_name, features in enabled_by_category.items():
                print(f"   {category_name}: {len(features)}개 특징")
        
        else:
            print("잘못된 명령입니다. 다시 시도해주세요.")
        
        DEFAULT_CONFIG.export_config()
        
if __name__ == "__main__":
    main()