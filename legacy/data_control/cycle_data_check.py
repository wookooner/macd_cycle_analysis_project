"""
카테고리 기반 Cycle Feature 검증 시스템 (v2.0)
새로운 nested 구조를 지원하는 개선된 검증기
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class CategorizedFeatureValidator:
    """
    카테고리 기반 Cycle Feature 검증 시스템
    - 7개 카테고리 구조 지원: shape, strength, start, end, change, volatility, aggregate
    - 카테고리별 특화 분석
    - 카테고리 간 관계 분석
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data = None
        self.flat_data = None  # 분석용 flat 데이터
        self.available_categories = []
        self.available_features = {}  # 카테고리별 특징 목록
        self.target_feature = 'change_price_pct'  # 기본 타겟
        
        # 분석 결과 저장
        self.validation_results = {}
        self.category_analysis = {}
        
        # 설정
        self._setup_korean_font()
        self._load_and_prepare_data()
        
    def _setup_korean_font(self):
        """한글 폰트 설정"""
        try:
            import matplotlib.font_manager as fm
            font_candidates = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']
            
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            selected_font = None
            
            for font in font_candidates:
                if font in available_fonts:
                    selected_font = font
                    break
            
            if selected_font:
                plt.rcParams['font.family'] = selected_font
                plt.rcParams['axes.unicode_minus'] = False
                print(f"폰트 설정: {selected_font}")
        except Exception as e:
            print(f"폰트 설정 오류: {e}")
    
    def _load_and_prepare_data(self):
        """카테고리 구조 데이터 로드 및 flat 변환"""
        try:
            print("카테고리 구조 데이터 로딩 및 분석 준비 중...")
            self.data = pd.read_parquet(self.data_path)
            print(f"총 {len(self.data):,}개 사이클 로드됨")
            
            # 새로운 구조인지 확인
            if len(self.data) > 0:
                sample_features = self.data.iloc[0]['cycle_features']
                
                if isinstance(sample_features, dict) and any(
                    isinstance(v, dict) for v in sample_features.values()
                ):
                    print("✅ 카테고리 구조 데이터 감지됨")
                    self._extract_categorized_features()
                else:
                    print("⚠️  기존 flat 구조 데이터입니다. 먼저 구조 변환을 실행하세요.")
                    return
            
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            raise
    
    def _extract_categorized_features(self):
        """카테고리 구조에서 특징들을 flat 형태로 추출하여 분석용 데이터 생성"""
        print("카테고리 구조 특징 추출 중...")
        
        flat_records = []
        self.available_categories = set()
        category_features = {}
        
        for idx, row in self.data.iterrows():
            flat_record = {
                'cycle_id': row['cycle_id'],
                'timeframe': row['timeframe'],
                'cycle_type': row['cycle_type'],
                'duration_candles': row['duration_candles']
            }
            
            # cycle_features에서 특징 추출
            features = row['cycle_features']
            if isinstance(features, dict):
                for category_name, category_data in features.items():
                    self.available_categories.add(category_name)
                    
                    if category_name not in category_features:
                        category_features[category_name] = set()
                    
                    if isinstance(category_data, dict):
                        for feature_name, feature_value in category_data.items():
                            category_features[category_name].add(feature_name)
                            
                            # flat 형태로 저장 (분석용)
                            flat_key = f"{category_name}_{feature_name}"
                            flat_record[flat_key] = feature_value
            
            flat_records.append(flat_record)
        
        # 분석용 DataFrame 생성
        self.flat_data = pd.DataFrame(flat_records)
        self.available_categories = list(self.available_categories)
        self.available_features = {cat: list(feats) for cat, feats in category_features.items()}
        
        # 수치형 특징만 선별
        numeric_columns = self.flat_data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_columns = ['duration_candles']
        self.numeric_features = [col for col in numeric_columns 
                               if col not in exclude_columns 
                               and self.flat_data[col].notna().sum() > 10]
        
        print(f"발견된 카테고리: {len(self.available_categories)}개")
        print(f"분석 가능한 특징: {len(self.numeric_features)}개")
        
        # 카테고리별 요약
        self._summarize_categories()
    
    def _summarize_categories(self):
        """카테고리별 특징 요약"""
        print(f"\n📊 카테고리별 특징 구성:")
        
        for category in self.available_categories:
            features = self.available_features.get(category, [])
            flat_features = [f"{category}_{feat}" for feat in features if f"{category}_{feat}" in self.numeric_features]
            
            print(f"  📁 {category}: {len(flat_features)}개 특징")
            
            if len(flat_features) <= 4:
                for feat in flat_features:
                    print(f"    - {feat.replace(category+'_', '')}")
            else:
                for feat in flat_features[:3]:
                    print(f"    - {feat.replace(category+'_', '')}")
                print(f"    - ... 외 {len(flat_features)-3}개")
    
    def validate_category(self, category_name: str, show_plots: bool = True) -> Dict[str, Any]:
        """특정 카테고리의 모든 특징에 대한 일괄 검증"""
        if category_name not in self.available_categories:
            print(f"오류: '{category_name}' 카테고리를 찾을 수 없습니다.")
            return {}
        
        print(f"\n{'='*80}")
        print(f"카테고리 검증: {category_name}")
        print(f"{'='*80}")
        
        # 해당 카테고리의 특징들
        category_features = [f for f in self.numeric_features if f.startswith(f"{category_name}_")]
        
        if not category_features:
            print(f"❌ '{category_name}' 카테고리에 분석 가능한 특징이 없습니다.")
            return {}
        
        print(f"분석 대상 특징: {len(category_features)}개")
        
        category_result = {
            'category_name': category_name,
            'feature_count': len(category_features),
            'validation_timestamp': datetime.now().isoformat(),
            'feature_results': {},
            'category_summary': {},
            'inter_feature_analysis': {}
        }
        
        # 각 특징별 개별 검증
        feature_scores = {}
        
        for feature in category_features:
            try:
                result = self.validate_single_feature(feature, show_individual_plots=False)
                category_result['feature_results'][feature] = result
                
                if 'final_recommendation' in result:
                    score = result['final_recommendation'].get('overall_score', 0)
                    feature_scores[feature] = score
                
            except Exception as e:
                print(f"특징 '{feature}' 검증 중 오류: {e}")
                continue
        
        # 카테고리 내 특징 간 관계 분석
        category_result['inter_feature_analysis'] = self._analyze_intra_category_relationships(
            category_features, show_plots
        )
        
        # 카테고리 요약
        category_result['category_summary'] = self._generate_category_summary(
            category_name, feature_scores, category_result['inter_feature_analysis']
        )
        
        self.category_analysis[category_name] = category_result
        return category_result
    
    def validate_single_feature(self, feature_name: str, show_individual_plots: bool = True) -> Dict[str, Any]:
        """단일 특징에 대한 기본 검증 (기존 3단계 분석 간소화 버전)"""
        if feature_name not in self.numeric_features:
            return {'error': f"특징 '{feature_name}'을 찾을 수 없습니다."}
        
        data_series = self.flat_data[feature_name].dropna()
        target_series = self.flat_data[self.target_feature].dropna() if self.target_feature in self.flat_data.columns else None
        
        result = {
            'feature_name': feature_name,
            'basic_stats': {},
            'target_correlation': {},
            'quality_score': 0
        }
        
        # 기본 통계
        desc_stats = data_series.describe()
        result['basic_stats'] = {k: float(v) for k, v in desc_stats.to_dict().items()}
        
        # 결측치 및 변별력
        missing_pct = (len(self.flat_data) - len(data_series)) / len(self.flat_data) * 100
        result['basic_stats']['missing_pct'] = float(missing_pct)
        
        # 타겟과의 상관관계 (타겟이 있는 경우)
        if target_series is not None and len(target_series) > 10:
            # 공통 인덱스 찾기
            common_data = pd.concat([data_series, target_series], axis=1, join='inner').dropna()
            
            if len(common_data) > 10:
                try:
                    corr_coef, p_value = stats.pearsonr(common_data.iloc[:, 0], common_data.iloc[:, 1])
                    result['target_correlation'] = {
                        'correlation': float(corr_coef),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05)
                    }
                except:
                    result['target_correlation'] = {'error': '상관관계 계산 실패'}
        
        # 간단한 품질 점수 계산
        score = 0
        if missing_pct < 5:
            score += 30
        elif missing_pct < 20:
            score += 20
        
        if 'target_correlation' in result and 'correlation' in result['target_correlation']:
            corr_val = abs(result['target_correlation']['correlation'])
            p_val = result['target_correlation']['p_value']
            
            if corr_val >= 0.5 and p_val < 0.05:
                score += 40
            elif corr_val >= 0.3 and p_val < 0.05:
                score += 30
            elif corr_val >= 0.1 and p_val < 0.05:
                score += 20
        
        # 변별력 점수
        cv = desc_stats['std'] / abs(desc_stats['mean']) if desc_stats['mean'] != 0 else 0
        if cv > 0.1:
            score += 30
        elif cv > 0.05:
            score += 20
        
        result['quality_score'] = min(100, score)
        
        # 간단한 추천
        if score >= 80:
            result['recommendation'] = "A급 특징 - 즉시 모델 사용 권장"
        elif score >= 60:
            result['recommendation'] = "B급 특징 - 모델 사용 권장"  
        elif score >= 40:
            result['recommendation'] = "C급 특징 - 조건부 사용"
        else:
            result['recommendation'] = "D급 특징 - 사용 신중 검토"
        
        return result
    
    def _analyze_intra_category_relationships(self, category_features: List[str], show_plots: bool) -> Dict[str, Any]:
        """카테고리 내 특징 간 관계 분석"""
        if len(category_features) < 2:
            return {'error': '분석할 특징이 부족합니다 (2개 이상 필요)'}
        
        print(f"\n카테고리 내 특징 간 관계 분석 ({len(category_features)}개 특징):")
        
        # 상관관계 매트릭스 계산
        feature_data = self.flat_data[category_features].corr()
        
        # 높은 상관관계 찾기
        high_correlations = []
        for i in range(len(category_features)):
            for j in range(i+1, len(category_features)):
                corr_val = feature_data.iloc[i, j]
                if abs(corr_val) >= 0.8:
                    high_correlations.append({
                        'feature1': category_features[i],
                        'feature2': category_features[j], 
                        'correlation': float(corr_val)
                    })
        
        # 결과 정리
        result = {
            'correlation_matrix': feature_data.to_dict(),
            'high_correlations': high_correlations,
            'redundancy_level': 'low'
        }
        
        if len(high_correlations) > 0:
            if len(high_correlations) >= len(category_features) // 2:
                result['redundancy_level'] = 'high'
            else:
                result['redundancy_level'] = 'medium'
        
        print(f"  높은 상관관계(|r|≥0.8): {len(high_correlations)}개")
        print(f"  중복성 수준: {result['redundancy_level']}")
        
        # 상관관계 히트맵 생성
        if show_plots and len(category_features) > 1:
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(feature_data, dtype=bool))
            sns.heatmap(feature_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.3f', cbar_kws={"shrink": .8})
            plt.title(f'{category_features[0].split("_")[0]} 카테고리 특징 간 상관관계')
            plt.tight_layout()
            plt.show()
        
        return result
    
    def _generate_category_summary(self, category_name: str, feature_scores: Dict[str, int], 
                                 inter_analysis: Dict) -> Dict[str, Any]:
        """카테고리 요약 생성"""
        if not feature_scores:
            return {'error': '분석할 특징이 없습니다'}
        
        scores = list(feature_scores.values())
        
        summary = {
            'category_name': category_name,
            'total_features': len(feature_scores),
            'avg_score': float(np.mean(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores)),
            'score_std': float(np.std(scores)),
            'redundancy_level': inter_analysis.get('redundancy_level', 'unknown'),
            'high_correlation_count': len(inter_analysis.get('high_correlations', []))
        }
        
        # 카테고리 등급
        avg_score = summary['avg_score']
        if avg_score >= 80:
            summary['category_grade'] = 'A'
            summary['category_recommendation'] = "우수한 카테고리 - 모든 특징 적극 활용"
        elif avg_score >= 65:
            summary['category_grade'] = 'B' 
            summary['category_recommendation'] = "양호한 카테고리 - 상위 특징들 우선 사용"
        elif avg_score >= 50:
            summary['category_grade'] = 'C'
            summary['category_recommendation'] = "보통 카테고리 - 선별적 사용"
        else:
            summary['category_grade'] = 'D'
            summary['category_recommendation'] = "개선 필요 카테고리 - 신중한 검토 후 사용"
        
        print(f"\n📋 {category_name} 카테고리 요약:")
        print(f"  평균 점수: {avg_score:.1f}/100 ({summary['category_grade']}급)")
        print(f"  추천 사항: {summary['category_recommendation']}")
        print(f"  중복성: {summary['redundancy_level']} (높은 상관관계 {summary['high_correlation_count']}개)")
        
        return summary
    
    def validate_all_categories(self, show_plots: bool = True) -> Dict[str, Dict]:
        """모든 카테고리에 대한 일괄 검증"""
        print(f"\n{'='*80}")
        print(f"전체 카테고리 일괄 검증 시작 ({len(self.available_categories)}개 카테고리)")
        print(f"{'='*80}")
        
        all_results = {}
        
        for i, category_name in enumerate(self.available_categories, 1):
            print(f"\n진행률: {i}/{len(self.available_categories)} ({i/len(self.available_categories)*100:.1f}%)")
            
            try:
                result = self.validate_category(category_name, show_plots=show_plots)
                all_results[category_name] = result
            except Exception as e:
                print(f"카테고리 '{category_name}' 검증 중 오류: {e}")
                continue
        
        # 전체 요약 리포트 생성
        self._generate_overall_summary(all_results)
        
        # 결과 저장
        self._save_categorized_validation_report(all_results)
        
        return all_results
    
    def _generate_overall_summary(self, all_results: Dict[str, Dict]):
        """전체 카테고리 검증 요약"""
        print(f"\n{'='*80}")
        print("전체 카테고리 검증 요약 리포트")
        print(f"{'='*80}")
        
        if not all_results:
            print("검증된 카테고리가 없습니다.")
            return
        
        # 카테고리별 성과 수집
        category_scores = []
        category_grades = {'A': [], 'B': [], 'C': [], 'D': []}
        
        for category_name, result in all_results.items():
            if 'category_summary' in result:
                summary = result['category_summary']
                score = summary.get('avg_score', 0)
                grade = summary.get('category_grade', 'D')
                
                category_scores.append((category_name, score, grade))
                category_grades[grade].append((category_name, score))
        
        # 점수순 정렬
        category_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"카테고리 성과 순위:")
        for i, (category_name, score, grade) in enumerate(category_scores, 1):
            redundancy = all_results[category_name]['category_summary'].get('redundancy_level', 'unknown')
            print(f"   {i}. {category_name:<12} {score:>5.1f}점 ({grade}급) [중복성: {redundancy}]")
        
        # 등급별 분포
        print(f"\n등급별 카테고리 분포:")
        for grade, categories in category_grades.items():
            if categories:
                print(f"   {grade}급: {len(categories)}개 카테고리")
                for cat_name, score in categories:
                    print(f"      {cat_name} ({score:.1f}점)")
        
        # 종합 추천사항
        print(f"\n📋 종합 추천사항:")
        
        a_grade_categories = [cat for cat, _ in category_grades['A']]
        b_grade_categories = [cat for cat, _ in category_grades['B']]
        
        if a_grade_categories:
            print(f"  🌟 우선 사용 권장: {', '.join(a_grade_categories)}")
        
        if b_grade_categories:
            print(f"  👍 보조 사용 권장: {', '.join(b_grade_categories)}")
        
        # 중복성이 높은 카테고리 경고
        high_redundancy = [cat for cat, result in all_results.items() 
                          if result.get('category_summary', {}).get('redundancy_level') == 'high']
        
        if high_redundancy:
            print(f"  ⚠️  높은 중복성 주의: {', '.join(high_redundancy)}")
    
    def analyze_cross_category_relationships(self, show_plots: bool = True):
        """카테고리 간 관계 분석"""
        print(f"\n{'='*80}")
        print("카테고리 간 관계 분석")
        print(f"{'='*80}")
        
        if len(self.available_categories) < 2:
            print("분석할 카테고리가 부족합니다 (2개 이상 필요)")
            return {}
        
        # 각 카테고리의 대표 특징 선정 (가장 높은 점수)
        category_representatives = {}
        
        for category_name in self.available_categories:
            category_features = [f for f in self.numeric_features if f.startswith(f"{category_name}_")]
            
            if category_features:
                # 간단한 점수 계산으로 대표 특징 선정
                best_feature = None
                best_score = -1
                
                for feature in category_features:
                    # 간단한 품질 점수 계산
                    data_series = self.flat_data[feature].dropna()
                    missing_pct = (len(self.flat_data) - len(data_series)) / len(self.flat_data) * 100
                    
                    score = 0
                    if missing_pct < 5:
                        score += 30
                    elif missing_pct < 20:
                        score += 20
                    
                    # 변별력
                    cv = data_series.std() / abs(data_series.mean()) if data_series.mean() != 0 else 0
                    if cv > 0.1:
                        score += 30
                    
                    if score > best_score:
                        best_score = score
                        best_feature = feature
                
                if best_feature:
                    category_representatives[category_name] = best_feature
        
        print(f"카테고리별 대표 특징:")
        for category, feature in category_representatives.items():
            print(f"  {category}: {feature.replace(category+'_', '')}")
        
        # 카테고리 간 상관관계 분석
        if len(category_representatives) >= 2 and show_plots:
            rep_features = list(category_representatives.values())
            rep_data = self.flat_data[rep_features].corr()
            
            plt.figure(figsize=(10, 8))
            labels = [f"{cat}\n({feat.split('_')[-1]})" for cat, feat in category_representatives.items()]
            
            sns.heatmap(rep_data, annot=True, cmap='RdBu_r', center=0,
                       xticklabels=labels, yticklabels=labels,
                       square=True, fmt='.3f')
            plt.title('카테고리 간 관계 (대표 특징 기준)')
            plt.tight_layout()
            plt.show()
        
        return category_representatives
    
    def generate_feature_recommendations(self) -> Dict[str, List[str]]:
        """카테고리별 추천 특징 리스트 생성"""
        recommendations = {}
        
        for category_name in self.available_categories:
            if category_name in self.category_analysis:
                category_result = self.category_analysis[category_name]
                feature_results = category_result.get('feature_results', {})
                
                # 점수 기준으로 정렬
                scored_features = []
                for feature_name, result in feature_results.items():
                    score = result.get('quality_score', 0)
                    scored_features.append((feature_name, score))
                
                scored_features.sort(key=lambda x: x[1], reverse=True)
                
                # 상위 특징들만 추천
                top_features = [feat for feat, score in scored_features if score >= 60]
                recommendations[category_name] = top_features
        
        print(f"\n📋 카테고리별 추천 특징 목록:")
        for category, features in recommendations.items():
            print(f"  📁 {category}: {len(features)}개 추천")
            for feat in features[:3]:  # 상위 3개만 표시
                clean_name = feat.replace(f"{category}_", "")
                print(f"    - {clean_name}")
            if len(features) > 3:
                print(f"    - ... 외 {len(features)-3}개")
        
        return recommendations
    
    def _save_categorized_validation_report(self, all_results: Dict[str, Dict]):
        """카테고리 기반 검증 결과를 JSON 파일로 저장"""
        try:
            report_dir = Path("./validation_reports")
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"categorized_feature_validation_{timestamp}.json"
            
            # JSON 직렬화 가능하도록 변환
            serializable_results = {}
            for category_name, result in all_results.items():
                serializable_results[category_name] = self._make_json_serializable(result)
            
            # 추가 메타데이터
            report_data = {
                'validation_type': 'categorized_features',
                'structure_version': '2.0',
                'timestamp': timestamp,
                'data_path': str(self.data_path),
                'total_cycles': len(self.data),
                'categories_analyzed': list(all_results.keys()),
                'results': serializable_results
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n카테고리 검증 리포트 저장됨: {report_file}")
            
        except Exception as e:
            print(f"리포트 저장 오류: {e}")
    
    def _make_json_serializable(self, obj):
        """JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        else:
            return obj


# 마이그레이션 및 테스트 함수들
def migrate_to_categorized_structure():
    """기존 데이터를 카테고리 구조로 마이그레이션"""
    print("🔄 카테고리 구조 마이그레이션 시작")
    print("="*60)
    
    # 기존 파일들 찾기
    structured_path = Path("./data/cycle_data/structured")
    if not structured_path.exists():
        print("❌ 구조화된 데이터 디렉토리를 찾을 수 없습니다")
        return
    
    existing_files = list(structured_path.glob("cycles_*.parquet"))
    v2_files = [f for f in existing_files if '_v2' in f.name]
    original_files = [f for f in existing_files if '_v2' not in f.name]
    
    print(f"기존 파일: {len(original_files)}개")
    print(f"v2 파일: {len(v2_files)}개")
    
    if not original_files:
        print("❌ 마이그레이션할 원본 파일이 없습니다")
        return
    
    # feature_extract 모듈에서 변환 함수 호출
    try:
        from feature_extract.macd_historgram_change_feature.feature_extract import StructuredCycleProcessor
        
        migration_results = {}
        
        for file_path in original_files:
            print(f"\n🔄 마이그레이션: {file_path.name}")
            
            try:
                processor = StructuredCycleProcessor(file_path)
                converted_path, cycle_count = processor.convert_existing_cycles_to_new_structure()
                
                if converted_path and cycle_count > 0:
                    if processor.validate_new_structure(converted_path):
                        migration_results[file_path.name] = {
                            'status': 'success',
                            'cycle_count': cycle_count,
                            'converted_file': converted_path.name
                        }
                    else:
                        migration_results[file_path.name] = {'status': 'validation_failed'}
                else:
                    migration_results[file_path.name] = {'status': 'conversion_failed'}
                    
            except Exception as e:
                print(f"❌ 마이그레이션 실패: {e}")
                migration_results[file_path.name] = {'status': 'error', 'error': str(e)}
        
        # 결과 요약
        print(f"\n{'='*60}")
        print("마이그레이션 결과 요약")
        print("="*60)
        
        successful = [k for k, v in migration_results.items() if v.get('status') == 'success']
        failed = [k for k, v in migration_results.items() if v.get('status') != 'success']
        
        print(f"✅ 성공: {len(successful)}개")
        print(f"❌ 실패: {len(failed)}개")
        
        if successful:
            total_cycles = sum(migration_results[k]['cycle_count'] for k in successful)
            print(f"📈 총 마이그레이션된 사이클: {total_cycles:,}개")
        
        return migration_results
        
    except ImportError as e:
        print(f"❌ 마이그레이션 모듈 로드 실패: {e}")
        return {}


def test_categorized_validation():
    """카테고리 구조 검증 테스트"""
    print("🧪 카테고리 구조 검증 시스템 테스트")
    
    # v2 파일 찾기
    structured_path = Path("./data/cycle_data/structured")
    v2_files = list(structured_path.glob("cycles_*_v2.parquet"))
    
    if not v2_files:
        print("❌ 테스트할 v2 파일을 찾을 수 없습니다. 먼저 마이그레이션을 실행하세요.")
        return
    
    print(f"테스트 가능한 파일: {len(v2_files)}개")
    for i, f in enumerate(v2_files, 1):
        print(f"  {i}. {f.name}")
    
    # 첫 번째 파일로 테스트
    test_file = v2_files[0]
    print(f"\n테스트 파일: {test_file.name}")
    
    try:
        validator = CategorizedFeatureValidator(test_file)
        
        print("\n1. 단일 카테고리 검증 테스트:")
        if validator.available_categories:
            test_category = validator.available_categories[0]
            validator.validate_category(test_category, show_plots=True)
        
        print(f"\n2. 카테고리 간 관계 분석 테스트:")
        validator.analyze_cross_category_relationships(show_plots=True)
        
        print(f"\n3. 추천 특징 생성 테스트:")
        recommendations = validator.generate_feature_recommendations()
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False


# 메인 실행 함수
def main():
    """카테고리 기반 특징 검증 메인 함수"""
    print("🚀 카테고리 기반 특징 검증 시스템 v2.0")
    
    print("\n선택하세요:")
    print("1: 기존 데이터를 카테고리 구조로 마이그레이션")
    print("2: 카테고리 구조 검증 시스템 테스트")  
    print("3: 특정 카테고리 검증")
    print("4: 전체 카테고리 일괄 검증")
    print("5: 카테고리 간 관계 분석")
    
    choice = input("선택 (1-5): ").strip()
    
    if choice == "1":
        migrate_to_categorized_structure()
    
    elif choice == "2":
        test_categorized_validation()
    
    elif choice == "3":
        # v2 파일 선택
        structured_path = Path("./data/cycle_data/structured")
        v2_files = list(structured_path.glob("cycles_*_v2.parquet"))
        
        if not v2_files:
            print("❌ v2 파일을 찾을 수 없습니다. 먼저 마이그레이션을 실행하세요.")
            return
        
        print("사용 가능한 v2 파일:")
        for i, f in enumerate(v2_files, 1):
            print(f"  {i}. {f.name}")
        
        try:
            file_idx = int(input("파일 번호를 선택하세요: ")) - 1
            selected_file = v2_files[file_idx]
            
            validator = CategorizedFeatureValidator(selected_file)
            
            print("사용 가능한 카테고리:")
            for i, cat in enumerate(validator.available_categories, 1):
                print(f"  {i}. {cat}")
            
            category = input("검증할 카테고리명을 입력하세요: ").strip()
            validator.validate_category(category, show_plots=True)
            
        except (ValueError, IndexError):
            print("❌ 잘못된 선택입니다")
        except Exception as e:
            print(f"❌ 카테고리 검증 실패: {e}")
    
    elif choice == "4":
        # 전체 검증
        structured_path = Path("./data/cycle_data/structured")
        v2_files = list(structured_path.glob("cycles_*_v2.parquet"))
        
        if v2_files:
            print("검증할 파일을 선택하세요:")
            for i, f in enumerate(v2_files, 1):
                print(f"  {i}. {f.name}")
            
            try:
                file_idx = int(input("파일 번호: ")) - 1
                selected_file = v2_files[file_idx]
                
                validator = CategorizedFeatureValidator(selected_file)
                validator.validate_all_categories(show_plots=True)
                
            except (ValueError, IndexError):
                print("❌ 잘못된 선택입니다")
            except Exception as e:
                print(f"❌ 전체 검증 실패: {e}")
        else:
            print("❌ v2 파일을 찾을 수 없습니다")
    
    elif choice == "5":
        # 카테고리 간 관계 분석
        structured_path = Path("./data/cycle_data/structured") 
        v2_files = list(structured_path.glob("cycles_*_v2.parquet"))
        
        if v2_files:
            selected_file = v2_files[0]  # 첫 번째 파일 사용
            print(f"분석 파일: {selected_file.name}")
            
            try:
                validator = CategorizedFeatureValidator(selected_file)
                validator.analyze_cross_category_relationships(show_plots=True)
                validator.generate_feature_recommendations()
                
            except Exception as e:
                print(f"❌ 관계 분석 실패: {e}")
        else:
            print("❌ v2 파일을 찾을 수 없습니다")
    
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()