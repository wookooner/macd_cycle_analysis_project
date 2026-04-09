"""
조건부 패턴 자동 발견기
=====================
특정 조건에서 상승/하락 비율이 얼마나 치우쳐 있는지 자동으로 탐색

예시:
- "start_rsi < 30일 때 상승 확률 80%"
- "duration < 15이고 macd > 100일 때 평균 수익 +5%"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json  # ← 추가!
from pathlib import Path
from datetime import datetime
from itertools import combinations, product
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ConditionalPatternFinder:
    """조건부 패턴 자동 발견 시스템"""
    
    def __init__(self, data_path: str = None, output_dir: str = None):
        """초기화"""
        if data_path is None:
            self.data_path = "data/cycle_data/structured/cycles_4h.parquet"
        else:
            self.data_path = data_path
        
        if output_dir is None:
            self.output_dir = Path("pattern_discovery_results")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.features_df = None
        self.discovered_patterns = []
        
        print(f"✅ 조건부 패턴 발견기 초기화")
    
    def load_data(self):
        """데이터 로드 및 전처리"""
        print("\n📊 데이터 로드 중...")
        
        self.df = pd.read_parquet(self.data_path)
        
        # 특징 평탄화
        records = []
        for _, row in self.df.iterrows():
            record = {
                'cycle_id': row['cycle_id'],
                'cycle_type': row['cycle_type'],
                'duration_candles': row['duration_candles']
            }
            
            if isinstance(row['cycle_features'], dict):
                for category, features in row['cycle_features'].items():
                    if isinstance(features, dict):
                        for fname, value in features.items():
                            record[f"{category}_{fname}"] = value
            
            records.append(record)
        
        self.features_df = pd.DataFrame(records)
        
        # 결측값 처리
        for col in self.features_df.columns:
            if self.features_df[col].dtype in ['float64', 'int64']:
                self.features_df[col].fillna(self.features_df[col].median(), inplace=True)
        
        print(f"✅ 데이터 로드 완료: {len(self.features_df)} 사이클")
    
    def analyze_single_condition(self,
                                 feature: str,
                                 operator: str,
                                 threshold: float,
                                 min_samples: int = 10) -> Dict:
        """
        단일 조건 분석
        
        Args:
            feature: 특징 이름
            operator: '>', '<', '>=', '<='
            threshold: 임계값
            min_samples: 최소 샘플 수
            
        Returns:
            분석 결과
        """
        if feature not in self.features_df.columns:
            return None
        
        # 조건 적용
        if operator == '>':
            mask = self.features_df[feature] > threshold
        elif operator == '<':
            mask = self.features_df[feature] < threshold
        elif operator == '>=':
            mask = self.features_df[feature] >= threshold
        elif operator == '<=':
            mask = self.features_df[feature] <= threshold
        else:
            return None
        
        filtered_df = self.features_df[mask]
        
        if len(filtered_df) < min_samples:
            return None
        
        # 상승/하락 비율
        up_count = (filtered_df['cycle_type'] == 'up').sum()
        down_count = (filtered_df['cycle_type'] == 'down').sum()
        total = len(filtered_df)
        
        up_ratio = up_count / total * 100
        down_ratio = down_count / total * 100
        
        # 평균 가격 변화
        if 'change_price_pct' in filtered_df.columns:
            avg_price_change = filtered_df['change_price_pct'].mean()
            median_price_change = filtered_df['change_price_pct'].median()
            
            # 수익/손실 비율
            profit_count = (filtered_df['change_price_pct'] > 0).sum()
            loss_count = (filtered_df['change_price_pct'] <= 0).sum()
            profit_ratio = profit_count / total * 100
        else:
            avg_price_change = None
            median_price_change = None
            profit_ratio = None
        
        result = {
            'condition': f"{feature} {operator} {threshold}",
            'feature': feature,
            'operator': operator,
            'threshold': threshold,
            'sample_count': total,
            'up_count': int(up_count),
            'down_count': int(down_count),
            'up_ratio': up_ratio,
            'down_ratio': down_ratio,
            'avg_price_change': avg_price_change,
            'median_price_change': median_price_change,
            'profit_ratio': profit_ratio,
            'bias_score': abs(up_ratio - 50)  # 50%에서 얼마나 치우쳐 있는지
        }
        
        return result
    
    def search_high_bias_patterns(self,
                                  features: List[str] = None,
                                  min_bias: float = 20.0,
                                  min_samples: int = 15,
                                  top_n: int = 50) -> List[Dict]:
        """
        치우침이 큰 패턴 자동 탐색
        
        Args:
            features: 탐색할 특징 리스트
            min_bias: 최소 치우침 정도 (예: 20 = 70%:30% 이상)
            min_samples: 최소 샘플 수
            top_n: 상위 N개 반환
            
        Returns:
            발견된 패턴 리스트
        """
        print(f"\n🔍 치우침 큰 패턴 탐색 중...")
        print(f"  조건: 치우침 >= {min_bias}%, 샘플 >= {min_samples}개")
        
        if features is None:
            # 수치형 특징만 선택
            features = [col for col in self.features_df.columns 
                       if self.features_df[col].dtype in ['float64', 'int64']
                       and col not in ['cycle_id']]
        
        patterns = []
        total_tests = 0
        
        for feature in features:
            # 특징의 분포를 기반으로 임계값 후보 생성
            values = self.features_df[feature].dropna()
            
            if len(values) < min_samples:
                continue
            
            # 분위수 기반 임계값
            percentiles = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
            thresholds = [np.percentile(values, p) for p in percentiles]
            
            # 각 임계값과 연산자 조합 테스트
            for threshold in thresholds:
                for operator in ['<', '>']:
                    total_tests += 1
                    
                    result = self.analyze_single_condition(
                        feature, operator, threshold, min_samples
                    )
                    
                    if result and result['bias_score'] >= min_bias:
                        patterns.append(result)
        
        print(f"  총 {total_tests}개 조건 테스트 완료")
        print(f"  발견된 패턴: {len(patterns)}개")
        
        # 치우침이 큰 순서로 정렬
        patterns.sort(key=lambda x: x['bias_score'], reverse=True)
        
        self.discovered_patterns = patterns[:top_n]
        
        return self.discovered_patterns
    
    def search_combined_conditions(self,
                                   feature_pairs: List[Tuple[str, str]] = None,
                                   min_bias: float = 25.0,
                                   min_samples: int = 10,
                                   top_n: int = 30) -> List[Dict]:
        """
        2개 특징 조합 조건 탐색
        
        예: "start_rsi < 30 AND duration < 20"
        
        Args:
            feature_pairs: 탐색할 특징 쌍 리스트
            min_bias: 최소 치우침
            min_samples: 최소 샘플 수
            top_n: 상위 N개 반환
            
        Returns:
            발견된 조합 패턴
        """
        print(f"\n🔍 조합 조건 패턴 탐색 중...")
        
        if feature_pairs is None:
            # 주요 특징들 선택
            key_features = [
                'duration_candles',
                'start_rsi', 'end_rsi',
                'start_macd', 'end_macd',
                'start_hist', 'end_hist',
                'strength_direction_pct',
                'change_price_pct'
            ]
            
            # 존재하는 특징만 필터링
            key_features = [f for f in key_features if f in self.features_df.columns]
            
            # 모든 2개 조합
            feature_pairs = list(combinations(key_features, 2))
            
            print(f"  탐색할 특징 쌍: {len(feature_pairs)}개")
        
        patterns = []
        total_tests = 0
        
        for feat1, feat2 in feature_pairs:
            # 각 특징의 임계값 후보
            vals1 = self.features_df[feat1].dropna()
            vals2 = self.features_df[feat2].dropna()
            
            if len(vals1) < min_samples or len(vals2) < min_samples:
                continue
            
            thresholds1 = [np.percentile(vals1, p) for p in [25, 50, 75]]
            thresholds2 = [np.percentile(vals2, p) for p in [25, 50, 75]]
            
            # 조합 테스트
            for t1, t2 in product(thresholds1, thresholds2):
                for op1, op2 in [('<', '<'), ('<', '>'), ('>', '<'), ('>', '>')]:
                    total_tests += 1
                    
                    # 조건 적용
                    if op1 == '<':
                        mask1 = self.features_df[feat1] < t1
                    else:
                        mask1 = self.features_df[feat1] > t1
                    
                    if op2 == '<':
                        mask2 = self.features_df[feat2] < t2
                    else:
                        mask2 = self.features_df[feat2] > t2
                    
                    filtered_df = self.features_df[mask1 & mask2]
                    
                    if len(filtered_df) < min_samples:
                        continue
                    
                    # 분석
                    up_count = (filtered_df['cycle_type'] == 'up').sum()
                    total = len(filtered_df)
                    up_ratio = up_count / total * 100
                    
                    bias_score = abs(up_ratio - 50)
                    
                    if bias_score >= min_bias:
                        result = {
                            'condition': f"{feat1} {op1} {t1:.2f} AND {feat2} {op2} {t2:.2f}",
                            'features': [feat1, feat2],
                            'sample_count': total,
                            'up_ratio': up_ratio,
                            'down_ratio': 100 - up_ratio,
                            'bias_score': bias_score
                        }
                        
                        if 'change_price_pct' in filtered_df.columns:
                            result['avg_price_change'] = filtered_df['change_price_pct'].mean()
                        
                        patterns.append(result)
        
        print(f"  총 {total_tests}개 조합 테스트 완료")
        print(f"  발견된 조합 패턴: {len(patterns)}개")
        
        patterns.sort(key=lambda x: x['bias_score'], reverse=True)
        
        return patterns[:top_n]
    
    def print_pattern_report(self, patterns: List[Dict], title: str = "발견된 패턴"):
        """패턴 리포트 출력"""
        
        print(f"\n{'='*80}")
        print(f"📊 {title}")
        print(f"{'='*80}")
        
        for idx, pattern in enumerate(patterns, 1):
            print(f"\n{idx}. {pattern['condition']}")
            print(f"   샘플: {pattern['sample_count']}개")
            print(f"   상승: {pattern['up_ratio']:.1f}% / 하락: {pattern['down_ratio']:.1f}%")
            print(f"   치우침: {pattern['bias_score']:.1f}%")
            
            if 'avg_price_change' in pattern and pattern['avg_price_change'] is not None:
                print(f"   평균 가격변화: {pattern['avg_price_change']:+.2f}%")
            
            if 'profit_ratio' in pattern and pattern['profit_ratio'] is not None:
                print(f"   수익 비율: {pattern['profit_ratio']:.1f}%")
    
    def visualize_top_patterns(self, patterns: List[Dict], top_n: int = 15):
        """상위 패턴 시각화"""
        
        if not patterns:
            print("시각화할 패턴이 없습니다.")
            return
        
        patterns_to_plot = patterns[:top_n]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. 치우침 정도 (Bias Score)
        conditions = [p['condition'][:50] + '...' if len(p['condition']) > 50 else p['condition'] 
                     for p in patterns_to_plot]
        bias_scores = [p['bias_score'] for p in patterns_to_plot]
        
        colors = ['#2ecc71' if p['up_ratio'] > 50 else '#e74c3c' for p in patterns_to_plot]
        
        axes[0].barh(range(len(patterns_to_plot)), bias_scores, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_yticks(range(len(patterns_to_plot)))
        axes[0].set_yticklabels(conditions, fontsize=8)
        axes[0].set_xlabel('치우침 정도 (%)', fontsize=10)
        axes[0].set_title('상위 패턴의 치우침 정도', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].invert_yaxis()
        
        # 2. 상승 비율
        up_ratios = [p['up_ratio'] for p in patterns_to_plot]
        sample_counts = [p['sample_count'] for p in patterns_to_plot]
        
        scatter = axes[1].scatter(up_ratios, range(len(patterns_to_plot)), 
                                 s=[s*2 for s in sample_counts],
                                 c=up_ratios, cmap='RdYlGn', 
                                 alpha=0.6, edgecolors='black', linewidth=1)
        axes[1].axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% (중립)')
        axes[1].set_yticks(range(len(patterns_to_plot)))
        axes[1].set_yticklabels(conditions, fontsize=8)
        axes[1].set_xlabel('상승 비율 (%)', fontsize=10)
        axes[1].set_title('상위 패턴의 상승 비율 (크기 = 샘플 수)', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='x')
        axes[1].invert_yaxis()
        
        plt.colorbar(scatter, ax=axes[1], label='상승 비율 (%)')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"top_patterns_{timestamp}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\n💾 저장: {filepath}")
        
        plt.show()
    
    def export_patterns_to_json(self, patterns: List[Dict], filename: str = None):
        """패턴을 JSON 파일로 저장"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"discovered_patterns_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=2, ensure_ascii=False)
        
        print(f"💾 패턴 저장: {filepath}")
    
    def run_full_analysis(self,
                         single_min_bias: float = 20.0,
                         combined_min_bias: float = 25.0,
                         min_samples: int = 15):
        """전체 분석 실행"""
        
        print("\n" + "="*80)
        print("🚀 조건부 패턴 전체 분석 시작")
        print("="*80)
        
        # 1. 단일 조건 패턴
        print("\n1️⃣ 단일 조건 패턴 탐색")
        single_patterns = self.search_high_bias_patterns(
            min_bias=single_min_bias,
            min_samples=min_samples,
            top_n=30
        )
        
        if single_patterns:
            self.print_pattern_report(single_patterns[:15], "상위 15개 단일 조건 패턴")
            self.visualize_top_patterns(single_patterns, top_n=15)
            self.export_patterns_to_json(single_patterns, "single_condition_patterns.json")
        
        # 2. 조합 조건 패턴
        print("\n2️⃣ 조합 조건 패턴 탐색")
        combined_patterns = self.search_combined_conditions(
            min_bias=combined_min_bias,
            min_samples=min_samples,
            top_n=20
        )
        
        if combined_patterns:
            self.print_pattern_report(combined_patterns[:10], "상위 10개 조합 조건 패턴")
            self.visualize_top_patterns(combined_patterns, top_n=10)
            self.export_patterns_to_json(combined_patterns, "combined_condition_patterns.json")
        
        print("\n" + "="*80)
        print("✅ 전체 분석 완료!")
        print(f"📁 결과 저장 위치: {self.output_dir}")
        print("="*80)
        
        return {
            'single_patterns': single_patterns,
            'combined_patterns': combined_patterns
        }


def main():
    """메인 실행"""
    
    print("\n" + "="*80)
    print("🎯 조건부 패턴 자동 발견 시스템")
    print("="*80)
    print("\n이 도구는 다음을 자동으로 찾아줍니다:")
    print("  - 특정 조건에서 상승/하락이 치우쳐진 패턴")
    print("  - 예: 'start_rsi < 30일 때 상승 확률 80%'")
    print("  - 예: 'duration < 15 AND macd > 100일 때 평균 수익 +5%'")
    
    # 초기화
    finder = ConditionalPatternFinder()
    
    # 데이터 로드
    finder.load_data()
    
    # 파라미터 설정
    print("\n" + "="*60)
    print("분석 파라미터 설정:")
    print("="*60)
    
    single_bias = float(input("단일 조건 최소 치우침 (%, 기본 20): ") or "20")
    combined_bias = float(input("조합 조건 최소 치우침 (%, 기본 25): ") or "25")
    min_samples = int(input("최소 샘플 수 (기본 15): ") or "15")
    
    # 전체 분석 실행
    results = finder.run_full_analysis(
        single_min_bias=single_bias,
        combined_min_bias=combined_bias,
        min_samples=min_samples
    )
    
    print("\n✅ 완료!")


if __name__ == "__main__":
    main()