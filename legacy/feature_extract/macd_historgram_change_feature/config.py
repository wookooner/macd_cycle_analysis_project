"""
config.py
카테고리 기반 특징 정의 및 관리 설정

변경 이력 (v3.0):
  - 신규 특징 추가: start.cvd_rolling, start.funding_rate, aggregate.cvd
  - 사후 특징(end.*, change.rsi/macd 등) enabled=False 처리
  - 통계적 유의성 낮은 특징(hist_positive_ratio 등) enabled=False 처리
  - peak_price_position, trough_price_position shape 카테고리에 추가
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import sys
import json
from pathlib import Path


class FeatureConfig:
    """카테고리 기반 특징 설정 관리 클래스"""

    def __init__(self, config_path: Path):
        self.config_path = config_path

        # ──────────────────────────────────────────────────────────────
        # 특징 카테고리 정의 (코드 내 기본값 — JSON이 있으면 덮어씌워짐)
        # enabled 기준:
        #   True  → 활성화: 진입 결정에 사용 가능하거나 타깃 변수
        #   False → 비활성: 사후 지표이거나 통계적 유의성 낮음
        # ──────────────────────────────────────────────────────────────
        self.FEATURE_CATEGORIES: Dict[str, Any] = {

            # ── Shape: 사이클의 구조적 골격 ──────────────────────────
            'shape': {
                'description': '형태/구조 - 사이클의 기본적인 골격과 구조적 특성',
                'features': {
                    'duration_candles': {
                        'description': '사이클의 전체 길이 (캔들 수)',
                        'calculator': 'calc_duration_candles',
                        'enabled': True,
                        'default_value': 0,
                        'data_type': 'int',
                    },
                    'core_count': {
                        'description': '사이클 방향과 일치하는 핵심 캔들 수',
                        'calculator': 'calc_core_count',
                        'enabled': True,
                        'default_value': 0,
                        'data_type': 'int',
                    },
                    'noise_count': {
                        'description': '허용된 노이즈(반대 방향) 캔들 수',
                        'calculator': 'calc_noise_count',
                        'enabled': True,
                        'default_value': 0,
                        'data_type': 'int',
                    },
                    'direction_change': {
                        'description': '사이클 내에서 모멘텀 방향이 전환된 횟수',
                        'calculator': 'calc_direction_change',
                        'enabled': True,
                        'default_value': 0,
                        'data_type': 'int',
                    },
                    'peak_price_position': {
                        'description': '사이클 내 최고가(high) 캔들의 위치 비율 (0~1)',
                        'calculator': 'calc_peak_price_position',
                        'enabled': True,
                        'default_value': 0.5,
                        'data_type': 'float',
                    },
                    'trough_price_position': {
                        'description': '사이클 내 최저가(low) 캔들의 위치 비율 (0~1)',
                        'calculator': 'calc_trough_price_position',
                        'enabled': True,
                        'default_value': 0.5,
                        'data_type': 'float',
                    },
                },
            },

            # ── Strength: 추세의 일관성 ───────────────────────────────
            'strength': {
                'description': '강도/견고함 - 사이클의 추세가 얼마나 일관되고 견고했는지',
                'features': {
                    'direction_pct': {
                        'description': '핵심 캔들의 비율 (core_count / duration_candles * 100)',
                        'calculator': 'calc_direction_pct',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    # 아래 세 특징은 이전 통계분석에서 p > 0.1로 유의성 없음
                    'hist_positive_ratio': {
                        'description': '[비활성] MACD 히스토그램 양수 비율 (p=0.83)',
                        'calculator': 'calc_hist_positive_ratio',
                        'enabled': False,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'price_up_ratio': {
                        'description': '[비활성] 양봉 캔들의 비율 (유의성 낮음)',
                        'calculator': 'calc_price_up_ratio',
                        'enabled': False,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'price_down_ratio': {
                        'description': '[비활성] 음봉 캔들의 비율 (유의성 낮음)',
                        'calculator': 'calc_price_down_ratio',
                        'enabled': False,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                },
            },

            # ── Start: 사이클 시작 시점 → 진입 결정에 직접 사용 ────────
            'start': {
                'description': '시작점 지표 - 사이클 시작 시점의 시장 상태 (진입 결정용)',
                'features': {
                    'price': {
                        'description': '사이클 첫 캔들의 종가',
                        'calculator': 'calc_start_price',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'volume': {
                        'description': '사이클 첫 캔들의 거래량',
                        'calculator': 'calc_start_volume',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'rsi': {
                        'description': '사이클 첫 캔들의 RSI 값',
                        'calculator': 'calc_start_rsi',
                        'enabled': True,
                        'default_value': 50.0,
                        'data_type': 'float',
                    },
                    'macd': {
                        'description': '사이클 첫 캔들의 MACD 값',
                        'calculator': 'calc_start_macd',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'macd_signal': {
                        'description': '[비활성] 사이클 첫 캔들의 MACD Signal 값 (p=0.44)',
                        'calculator': 'calc_start_macd_signal',
                        'enabled': False,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'hist': {
                        'description': '사이클 첫 캔들의 MACD Histogram 값 (핵심 진입 지표)',
                        'calculator': 'calc_start_hist',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    # ── 신규 특징 ─────────────────────────────────────
                    'cvd_rolling': {
                        'description': '[신규] 사이클 첫 캔들의 롤링 CVD. 단기 매수/매도 압력 균형.'
                                       ' candle_data에 cvd_rolling 컬럼 필요',
                        'calculator': 'calc_start_cvd_rolling',
                        'enabled': True,
                        'default_value': None,
                        'data_type': 'float',
                    },
                    'funding_rate': {
                        'description': '[신규] 사이클 시작 시점 가장 가까운 펀딩비.'
                                       ' 롱/숏 포지션 쏠림 측정. 외부 funding_rate CSV 필요',
                        'calculator': 'calc_start_funding_rate',
                        'enabled': True,
                        'default_value': None,
                        'data_type': 'float',
                    },
                },
            },

            # ── End: 사후 지표 → 진입 결정 불가 ─────────────────────
            'end': {
                'description': '[사후분석용] 종료점 지표 - 사이클 완료 후에만 확정',
                'features': {
                    'price':      {'description': '[비활성/사후] 마지막 캔들 종가',      'calculator': 'calc_end_price',       'enabled': False, 'default_value': 0.0,  'data_type': 'float'},
                    'volume':     {'description': '[비활성/사후] 마지막 캔들 거래량',     'calculator': 'calc_end_volume',      'enabled': False, 'default_value': 0.0,  'data_type': 'float'},
                    'rsi':        {'description': '[비활성/사후] 마지막 캔들 RSI',        'calculator': 'calc_end_rsi',         'enabled': False, 'default_value': 50.0, 'data_type': 'float'},
                    'macd':       {'description': '[비활성/사후] 마지막 캔들 MACD',       'calculator': 'calc_end_macd',        'enabled': False, 'default_value': 0.0,  'data_type': 'float'},
                    'macd_signal':{'description': '[비활성/사후] 마지막 캔들 MACD Signal','calculator': 'calc_end_macd_signal', 'enabled': False, 'default_value': 0.0,  'data_type': 'float'},
                    'hist':       {'description': '[비활성/사후] 마지막 캔들 MACD Hist',  'calculator': 'calc_end_hist',        'enabled': False, 'default_value': 0.0,  'data_type': 'float'},
                },
            },

            # ── Change: price_pct만 타깃 변수로 활성화 ───────────────
            'change': {
                'description': '시작-종료 변화량. price_pct는 분석 타깃(Y). 나머지는 사후 지표',
                'features': {
                    'price_pct': {
                        'description': '시작가 대비 종료가 등락률. 분석 타깃 변수(Y)',
                        'calculator': 'calc_price_change_pct',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'rsi':        {'description': '[비활성/사후] RSI 변화량',          'calculator': 'calc_rsi_change',           'enabled': False, 'default_value': 0.0, 'data_type': 'float'},
                    'macd':       {'description': '[비활성/사후] MACD 변화량',         'calculator': 'calc_macd_change',          'enabled': False, 'default_value': 0.0, 'data_type': 'float'},
                    'macd_signal':{'description': '[비활성/사후] MACD Signal 변화량',  'calculator': 'calc_macd_signal_change',   'enabled': False, 'default_value': 0.0, 'data_type': 'float'},
                    'hist':       {'description': '[비활성/사후] MACD Histogram 변화량','calculator': 'calc_macd_histogram_change','enabled': False, 'default_value': 0.0, 'data_type': 'float'},
                },
            },

            # ── Volatility: 통계 유의 특징만 활성화 ─────────────────
            'volatility': {
                'description': '내부 변동성/극값 - 사이클 진행 중 가격 움직임의 크기',
                'features': {
                    'max_high_pct': {
                        'description': '[비활성/사후] 시작가 대비 사이클 내 최고가 상승률',
                        'calculator': 'calc_max_high_pct',
                        'enabled': False,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'max_loss_pct': {
                        'description': '[비활성/사후] 시작가 대비 사이클 내 최저가 하락률',
                        'calculator': 'calc_max_loss_pct',
                        'enabled': False,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'max_intraday_high_pct': {
                        'description': '[비활성/사후] 사이클 내 최대 잠재 수익률. 사후+연산 비용 높음',
                        'calculator': 'calc_max_intraday_high_pct',
                        'enabled': False,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'max_intraday_loss_pct': {
                        'description': '[비활성/사후] 사이클 내 최대 잠재 손실률. 사후+연산 비용 높음',
                        'calculator': 'calc_max_intraday_loss_pct',
                        'enabled': False,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'avg_true_range': {
                        'description': '사이클 내 평균 ATR. 통계 유의성 확인됨 (p=0.037)',
                        'calculator': 'calc_avg_true_range',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    'price_change_deviation': {
                        'description': '사이클 내 캔들별 종가 변동률의 표준편차',
                        'calculator': 'calc_price_change_deviation',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                },
            },

            # ── Aggregate: 사이클 전체 집계 ──────────────────────────
            'aggregate': {
                'description': '전체 집계 - 사이클 전체 기간의 데이터를 합산한 값',
                'features': {
                    'volume': {
                        'description': '사이클 내 모든 캔들의 거래량 총합',
                        'calculator': 'calc_all_volume',
                        'enabled': True,
                        'default_value': 0.0,
                        'data_type': 'float',
                    },
                    # ── 신규 특징 ─────────────────────────────────────
                    'cvd': {
                        'description': '[신규] 사이클 전체 volume_delta 합산.'
                                       ' 사이클 기간 동안 누적된 순매수/매도 압력.'
                                       ' candle_data에 volume_delta 컬럼 필요',
                        'calculator': 'calc_aggregate_cvd',
                        'enabled': True,
                        'default_value': None,
                        'data_type': 'float',
                    },
                },
            },
        }

        # 계산 설정
        self.CALCULATION_CONFIG = {
            'batch_size': 1000,
            'handle_errors': True,
            'use_default_on_error': True,
        }

        # JSON 파일이 있으면 enabled 상태를 덮어씌움
        self.import_config()

    # ──────────────────────────────────────────────────────────────────
    # 조회 메서드들
    # ──────────────────────────────────────────────────────────────────

    def get_all_features_flat(self) -> Dict[str, Dict[str, Any]]:
        """모든 특징을 flat 구조로 반환 (기존 호환성용)"""
        flat_features = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            for feature_name, feature_config in category_data['features'].items():
                flat_name = f"{category_name}_{feature_name}"
                flat_features[flat_name] = {
                    'description': feature_config['description'],
                    'calculator': feature_config['calculator'],
                    'enabled': feature_config['enabled'],
                    'default_value': feature_config['default_value'],
                    'category': category_name,
                    'feature_key': feature_name,
                }
        return flat_features

    def get_enabled_features_by_category(self) -> Dict[str, Dict[str, Dict]]:
        """카테고리별로 활성화된 특징들 반환"""
        enabled_by_category = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            enabled_features = {
                name: config
                for name, config in category_data['features'].items()
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
                name
                for name, config in category_data['features'].items()
                if config['enabled']
            ]
            if feature_names:
                names_by_category[category_name] = feature_names
        return names_by_category

    def get_all_calculator_names(self) -> List[str]:
        """모든 활성화된 계산함수 이름 리스트"""
        return [
            feature_config['calculator']
            for category_data in self.FEATURE_CATEGORIES.values()
            for feature_config in category_data['features'].values()
            if feature_config['enabled']
        ]

    def get_default_cycle_features_structure(self) -> Dict[str, Dict]:
        """기본 사이클 특징 구조 생성 (활성화된 특징만)"""
        structure = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            structure[category_name] = {
                feature_name: feature_config['default_value']
                for feature_name, feature_config in category_data['features'].items()
                if feature_config['enabled']
            }
        return structure

    # ──────────────────────────────────────────────────────────────────
    # 활성화/비활성화
    # ──────────────────────────────────────────────────────────────────

    def enable_feature(self, category_name: str, feature_name: str) -> bool:
        """특정 카테고리의 특징 활성화"""
        try:
            self.FEATURE_CATEGORIES[category_name]['features'][feature_name]['enabled'] = True
            print(f"✅ 특징 '{category_name}.{feature_name}' 활성화")
            return True
        except KeyError:
            print(f"❌ 존재하지 않는 특징: '{category_name}.{feature_name}'")
            return False

    def disable_feature(self, category_name: str, feature_name: str) -> bool:
        """특정 카테고리의 특징 비활성화"""
        try:
            self.FEATURE_CATEGORIES[category_name]['features'][feature_name]['enabled'] = False
            print(f"❌ 특징 '{category_name}.{feature_name}' 비활성화")
            return True
        except KeyError:
            print(f"❌ 존재하지 않는 특징: '{category_name}.{feature_name}'")
            return False

    def validate_calculator_functions(self, calculator_module: Any) -> Dict[str, Dict[str, bool]]:
        """카테고리별로 계산함수가 존재하는지 검증"""
        validation_results = {}
        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            validation_results[category_name] = {
                feature_name: hasattr(calculator_module, feature_config['calculator'])
                for feature_name, feature_config in category_data['features'].items()
                if feature_config['enabled']
            }
        return validation_results

    # ──────────────────────────────────────────────────────────────────
    # JSON 가져오기/내보내기
    # ──────────────────────────────────────────────────────────────────

    def export_config(self):
        """현재 enabled 상태를 JSON 파일로 내보내기"""
        export_data = {
            'feature_categories': self.FEATURE_CATEGORIES,
            'calculation_config': self.CALCULATION_CONFIG,
            'version': '3.0',
            'structure_type': 'categorized',
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

    def import_config(self):
        """JSON 파일에서 enabled 상태를 가져와 코드 정의에 덮어씌움"""
        if not self.config_path.exists():
            return False
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            if 'feature_categories' not in import_data:
                return False

            for category_name, category_data in import_data['feature_categories'].items():
                if category_name not in self.FEATURE_CATEGORIES:
                    continue
                for feature_name, feature_config in category_data.get('features', {}).items():
                    if feature_name in self.FEATURE_CATEGORIES[category_name]['features']:
                        # enabled 상태만 JSON에서 읽어옴
                        # (description, calculator 등 구조 정의는 코드가 우선)
                        if 'enabled' in feature_config:
                            self.FEATURE_CATEGORIES[category_name]['features'][feature_name]['enabled'] = \
                                feature_config['enabled']

            if 'calculation_config' in import_data:
                self.CALCULATION_CONFIG = import_data['calculation_config']

            return True

        except Exception as e:
            print(f"⚠️  설정 가져오기 실패 (기본값 사용): {e}")
            return False

    # ──────────────────────────────────────────────────────────────────
    # 출력
    # ──────────────────────────────────────────────────────────────────

    def print_feature_summary(self):
        """카테고리별 특징 요약 출력"""
        print("\n" + "=" * 80)
        print("📋 카테고리별 특징 구성 현황")
        print("=" * 80)

        total_features = 0
        enabled_features = 0

        for category_name, category_data in self.FEATURE_CATEGORIES.items():
            features = category_data['features']
            enabled_count = sum(1 for cfg in features.values() if cfg['enabled'])
            total_count = len(features)

            total_features += total_count
            enabled_features += enabled_count

            print(f"\n📁 {category_name.upper()} ({category_data['description']})")
            print(f"   활성화: {enabled_count}/{total_count}개")

            for feature_name, feature_config in features.items():
                status = "✅" if feature_config['enabled'] else "❌"
                print(f"   {status} {feature_name}: {feature_config['description']}")

        print("\n" + "=" * 80)
        print(f"📊 전체 요약: {enabled_features}/{total_features}개 특징 활성화")
        print("=" * 80)


# ──────────────────────────────────────────────────────────────────────
# 전역 싱글톤 설정 인스턴스
# ──────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG_PATH = Path(__file__).parent / "features_config_v2.json"
DEFAULT_CONFIG = FeatureConfig(DEFAULT_CONFIG_PATH)


def main():
    """특징 관리 메인 함수"""
    print("🚀 카테고리 기반 특징 관리 스크립트 시작")

    while True:
        DEFAULT_CONFIG.print_feature_summary()

        print("\n[메뉴]")
        print("1: 특징 활성화/비활성화")
        print("2: 전체 활성화")
        print("3: 전체 비활성화")
        print("4: 카테고리별 통계")
        print("5: 종료")

        choice = input("명령을 입력하세요 (1-5): ").strip()

        if choice == '5':
            print("프로그램을 종료합니다.")
            break

        elif choice == '1':
            print("\n📁 사용 가능한 카테고리:")
            for i, (category_name, category_data) in enumerate(DEFAULT_CONFIG.FEATURE_CATEGORIES.items(), 1):
                enabled_count = sum(1 for cfg in category_data['features'].values() if cfg['enabled'])
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
                    print(f"❌ 존재하지 않는 특징: {feature_input}")
            else:
                print(f"❌ 존재하지 않는 카테고리: {category_input}")

        elif choice == '2':
            for category_name, category_data in DEFAULT_CONFIG.FEATURE_CATEGORIES.items():
                for feature_name in category_data['features']:
                    DEFAULT_CONFIG.enable_feature(category_name, feature_name)
            print("✅ 모든 특징이 활성화되었습니다.")

        elif choice == '3':
            for category_name, category_data in DEFAULT_CONFIG.FEATURE_CATEGORIES.items():
                for feature_name in category_data['features']:
                    DEFAULT_CONFIG.disable_feature(category_name, feature_name)
            print("❌ 모든 특징이 비활성화되었습니다.")

        elif choice == '4':
            enabled_by_category = DEFAULT_CONFIG.get_enabled_features_by_category()
            print("\n📊 카테고리별 활성화된 특징 수:")
            for category_name, features in enabled_by_category.items():
                print(f"   {category_name}: {len(features)}개")

        else:
            print("잘못된 명령입니다.")

        DEFAULT_CONFIG.export_config()


if __name__ == "__main__":
    main()