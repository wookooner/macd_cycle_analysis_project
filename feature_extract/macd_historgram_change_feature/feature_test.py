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

class CycleFeatureValidator:
    """
    Cycle Feature 3-Stage 검증 시스템 (개선된 버전)
    - Stage 1: 단변량 분석 (Univariate Analysis)  
    - Stage 2: 이변량 분석 (Bivariate Analysis)
    - Stage 3: 다변량 분석 (Multivariate Analysis)
    
    주요 개선사항:
    - 강화된 그룹 분석 오류 처리
    - 정확한 왜도 계산 (scipy.stats.skew)
    - JSON 직렬화 오류 해결
    - 확장성을 고려한 자동 특징 감지
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data = None
        self.available_features = []
        self.target_feature = 'price_change_pct'
        
        # 분석 결과 저장
        self.validation_results = {}
        self.feature_recommendations = {}
        
        # 한글 폰트 설정
        self._setup_korean_font()
        
        # 데이터 로드
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
            else:
                print("한글 폰트를 찾을 수 없습니다. 영문으로 표시됩니다.")
        except Exception as e:
            print(f"폰트 설정 오류: {e}")
    
    def _load_and_prepare_data(self):
        """데이터 로드 및 특징 자동 감지"""
        try:
            print("데이터 로딩 및 특징 감지 중...")
            self.data = pd.read_parquet(self.data_path)
            print(f"총 {len(self.data):,}개 사이클 로드됨")
            
            # cycle_features에서 특징 추출
            if 'cycle_features' in self.data.columns:
                # cycle_features를 DataFrame으로 변환
                features_list = []
                for idx, features in self.data['cycle_features'].items():
                    if isinstance(features, dict):
                        features_list.append(features)
                    elif hasattr(features, '__iter__') and len(features) > 0:
                        if isinstance(features[0], dict):
                            features_list.append(features[0])
                        else:
                            features_list.append({})
                    else:
                        features_list.append({})
                
                features_df = pd.DataFrame(features_list, index=self.data.index)
                
                # 기존 데이터와 병합
                self.data = pd.concat([
                    self.data.drop('cycle_features', axis=1),
                    features_df
                ], axis=1)
            
            # 수치형 특징만 자동 감지
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            # 제외할 컬럼들 (ID나 인덱스 성격의 컬럼)
            exclude_columns = ['cycle_id', 'duration_candles'] # duration_candles는 범주형에 가까움
            
            self.available_features = [col for col in numeric_columns 
                                     if col not in exclude_columns 
                                     and self.data[col].notna().sum() > 10]  # 최소 10개 이상 유효값
            
            print(f"분석 가능한 특징 {len(self.available_features)}개 발견:")
            
            # 특징을 카테고리별로 분류하여 표시
            self._categorize_and_display_features()
            
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            raise
    
    def _categorize_and_display_features(self):
        """특징을 카테고리별로 분류하여 표시"""
        feature_categories = {
            "시작점 특징": [f for f in self.available_features if f.startswith('start_')],
            "종료점 특징": [f for f in self.available_features if f.startswith('end_')],
            "변화량 특징": [f for f in self.available_features if 'change' in f or 'pct' in f],
            "극값 특징": [f for f in self.available_features if 'max_' in f or 'min_' in f],
            "노이즈/방향성": [f for f in self.available_features if any(x in f for x in ['noise', 'direction', 'core'])],
            "거래량 특징": [f for f in self.available_features if 'volume' in f],
            "기타 특징": []
        }
        
        # 분류되지 않은 특징들은 기타에 추가
        categorized_features = set()
        for features in feature_categories.values():
            categorized_features.update(features)
        
        feature_categories["기타 특징"] = [f for f in self.available_features 
                                        if f not in categorized_features]
        
        for category, features in feature_categories.items():
            if features:
                print(f"  {category}: {len(features)}개")
                for i, feature in enumerate(features):
                    if i < 3:  # 처음 3개만 표시
                        print(f"    - {feature}")
                    elif i == 3:
                        print(f"    - ... 외 {len(features)-3}개")
                        break
    
    def validate_feature(self, feature_name: str, show_plots: bool = True) -> Dict[str, Any]:
        """단일 특징에 대한 3단계 검증 수행"""
        if feature_name not in self.available_features:
            print(f"오류: '{feature_name}' 특징을 찾을 수 없습니다.")
            return {}
        
        if self.target_feature not in self.available_features:
            print(f"오류: 타겟 특징 '{self.target_feature}'를 찾을 수 없습니다.")
            return {}
        
        print(f"\n{'='*80}")
        print(f"특징 검증: {feature_name}")
        print(f"{'='*80}")
        
        result = {
            'feature_name': feature_name,
            'validation_timestamp': datetime.now().isoformat(),
            'stage1': {},
            'stage2': {},
            'stage3': {},
            'final_recommendation': {}
        }
        
        # Stage 1: 단변량 분석
        result['stage1'] = self._stage1_univariate_analysis(feature_name, show_plots)
        
        # Stage 2: 이변량 분석  
        result['stage2'] = self._stage2_bivariate_analysis(feature_name, show_plots)
        
        # Stage 3: 다변량 분석
        result['stage3'] = self._stage3_multivariate_analysis(feature_name, show_plots)
        
        # 최종 추천
        result['final_recommendation'] = self._generate_final_recommendation(result)
        
        self.validation_results[feature_name] = result
        return result
    
    def _stage1_univariate_analysis(self, feature_name: str, show_plots: bool) -> Dict[str, Any]:
        """Stage 1: 단변량 분석 (개선된 왜도 계산 포함)"""
        print(f"\n분석 Stage 1: 단변량 분석 - '{feature_name}' 특징 자체 깊이 이해하기")
        print("-" * 60)
        
        data_series = self.data[feature_name].dropna()
        
        # Action 1-1: 기술 통계량 계산
        print("\n통계량 계산:")
        desc_stats = data_series.describe()
        print(desc_stats)
        
        # Check Point 1-1: 통계량 해석
        print("\n통계량 해석:")
        
        result = {
            'data_count': int(len(data_series)),
            'missing_count': int(len(self.data) - len(data_series)),
            'basic_stats': {k: float(v) for k, v in desc_stats.to_dict().items()},
            'checks': {}
        }
        
        # 결측치 확인
        missing_pct = (len(self.data) - len(data_series)) / len(self.data) * 100
        print(f"  결측치: {len(self.data) - len(data_series):,}개 ({missing_pct:.1f}%)")
        result['checks']['missing_rate'] = float(missing_pct)
        
        # 개선된 왜도 계산 (scipy.stats.skew 사용)
        try:
            skewness_value = float(stats.skew(data_series))
            
            if abs(skewness_value) < 0.5:
                skew_interpretation = "대칭적"
            elif abs(skewness_value) < 1:
                skew_interpretation = "중간 편향"
            else:
                skew_interpretation = "강한 편향"
            
            skew_direction = "우편향" if skewness_value > 0 else "좌편향" if skewness_value < 0 else "대칭"
            
            print(f"  분포 편향: {skew_interpretation} ({skew_direction}, 왜도={skewness_value:.3f})")
            result['checks']['skewness'] = {
                'value': float(skewness_value), 
                'interpretation': skew_interpretation,
                'direction': skew_direction
            }
            
        except Exception as e:
            print(f"  분포 편향: 계산 오류 ({str(e)})")
            result['checks']['skewness'] = {'error': str(e)}
        
        # 변별력 확인
        mean_val = desc_stats['mean']
        std_val = desc_stats['std']
        
        if std_val == 0:
            variability = "변별력 없음"
        elif abs(mean_val) > 0 and std_val < abs(mean_val) * 0.01:
            variability = "매우 낮음"
        elif abs(mean_val) > 0 and std_val < abs(mean_val) * 0.1:
            variability = "낮음"
        else:
            variability = "충분함"
        
        cv = float(std_val / abs(mean_val)) if mean_val != 0 else float('inf')
        print(f"  변별력: {variability} (변동계수={cv:.3f})")
        result['checks']['variability'] = {
            'interpretation': variability,
            'coefficient_of_variation': float(cv) if cv != float('inf') else None
        }
        
        # 이상치 감지 (IQR 방법)
        Q1, Q3 = desc_stats['25%'], desc_stats['75%']
        IQR = Q3 - Q1
        outlier_threshold_lower = Q1 - 1.5 * IQR
        outlier_threshold_upper = Q3 + 1.5 * IQR
        
        outliers = data_series[(data_series < outlier_threshold_lower) | 
                              (data_series > outlier_threshold_upper)]
        outlier_pct = len(outliers) / len(data_series) * 100
        
        print(f"  이상치: {len(outliers):,}개 ({outlier_pct:.1f}%)")
        result['checks']['outliers'] = {
            'count': int(len(outliers)), 
            'percentage': float(outlier_pct),
            'lower_threshold': float(outlier_threshold_lower),
            'upper_threshold': float(outlier_threshold_upper)
        }
        
        # Action 1-2: 분포 시각화
        if show_plots:
            print("\n분포 시각화 생성 중...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 히스토그램 + KDE
            sns.histplot(data=data_series, kde=True, ax=ax1, alpha=0.7)
            ax1.axvline(desc_stats['mean'], color='red', linestyle='--', alpha=0.8, 
                       label=f'평균: {desc_stats["mean"]:.3f}')
            ax1.axvline(desc_stats['50%'], color='orange', linestyle='--', alpha=0.8, 
                       label=f'중앙값: {desc_stats["50%"]:.3f}')
            ax1.set_title(f'{feature_name} 분포')
            ax1.legend()
            
            # 박스플롯
            sns.boxplot(y=data_series, ax=ax2)
            ax2.set_title(f'{feature_name} 박스플롯')
            
            # Q-Q 플롯 (정규성 확인)
            stats.probplot(data_series, dist="norm", plot=ax3)
            ax3.set_title('정규성 검증 (Q-Q 플롯)')
            
            # 로그 변환 분포 (양수인 경우만)
            if data_series.min() > 0:
                log_data = np.log(data_series)
                sns.histplot(data=log_data, kde=True, ax=ax4, alpha=0.7, color='green')
                ax4.set_title(f'{feature_name} 로그 변환 분포')
            else:
                ax4.text(0.5, 0.5, '로그 변환 불가\n(0 이하 값 존재)', 
                        transform=ax4.transAxes, ha='center', va='center')
                ax4.set_title('로그 변환 분포')
            
            plt.suptitle(f'Stage 1: {feature_name} 단변량 분석', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        return result
    
    def _stage2_bivariate_analysis(self, feature_name: str, show_plots: bool) -> Dict[str, Any]:
        """Stage 2: 이변량 분석 (강화된 그룹 분석 오류 처리 포함)"""
        print(f"\n분석 Stage 2: 이변량 분석 - '{feature_name}'과 타겟 관계 파헤치기")
        print("-" * 60)
        
        # 유효한 데이터만 선택
        valid_data = self.data[[feature_name, self.target_feature]].dropna()
        
        if len(valid_data) < 10:
            print("분석할 데이터가 부족합니다.")
            return {'error': '데이터 부족'}
        
        feature_values = valid_data[feature_name]
        target_values = valid_data[self.target_feature]
        
        result = {
            'valid_samples': int(len(valid_data)),
            'correlations': {},
            'group_analysis': {},
            'relationship_strength': {}
        }
        
        # Action 2-1 & 2-2: 통계적 관계 측정
        print("\n통계적 관계 측정:")
        
        # 피어슨 상관계수
        try:
            pearson_corr, pearson_p = stats.pearsonr(feature_values, target_values)
            print(f"  피어슨 상관계수: {pearson_corr:.4f} (p-value: {pearson_p:.6f})")
            result['correlations']['pearson'] = {
                'corr': float(pearson_corr), 
                'p_value': float(pearson_p)
            }
        except Exception as e:
            print(f"  피어슨 상관계수 계산 오류: {e}")
            result['correlations']['pearson'] = {'error': str(e)}
        
        # 스피어만 상관계수
        try:
            spearman_corr, spearman_p = stats.spearmanr(feature_values, target_values)
            print(f"  스피어만 상관계수: {spearman_corr:.4f} (p-value: {spearman_p:.6f})")
            result['correlations']['spearman'] = {
                'corr': float(spearman_corr), 
                'p_value': float(spearman_p)
            }
        except Exception as e:
            print(f"  스피어만 상관계수 계산 오류: {e}")
            result['correlations']['spearman'] = {'error': str(e)}
        
        # Check Point 2-2: 상관계수 해석
        print("\n상관계수 해석:")
        
        if 'pearson' in result['correlations'] and 'corr' in result['correlations']['pearson']:
            corr_val = abs(result['correlations']['pearson']['corr'])
            p_val = result['correlations']['pearson']['p_value']
            
            if corr_val >= 0.7:
                strength = "매우 강한"
            elif corr_val >= 0.5:
                strength = "강한"
            elif corr_val >= 0.3:
                strength = "중간"
            elif corr_val >= 0.1:
                strength = "약한"
            else:
                strength = "거의 없는"
            
            significance = "유의미함" if p_val < 0.05 else "유의미하지 않음"
            print(f"  관계 강도: {strength} 선형관계")
            print(f"  통계적 유의성: {significance}")
            
            result['relationship_strength']['linear'] = {
                'strength': strength,
                'significant': bool(p_val < 0.05)
            }
        
        # Action 2-3: 강화된 그룹 분석
        print("\n그룹 분석 (강화된 오류 처리):")
        
        group_analysis_result = self._robust_group_analysis(feature_values, target_values)
        result['group_analysis'] = group_analysis_result
        
        # 시각화
        if show_plots:
            self._create_bivariate_plots(feature_name, feature_values, target_values, 
                                       group_analysis_result, result)
        
        return result
    
    def _robust_group_analysis(self, feature_values: pd.Series, target_values: pd.Series) -> Dict[str, Any]:
        """강화된 그룹 분석 - 다양한 분위수로 재시도"""
        
        # 먼저 데이터의 유니크 값 개수 확인
        unique_values = feature_values.nunique()
        print(f"  유니크 값 개수: {unique_values}")
        
        if unique_values < 3:
            return {
                'error': f'유니크 값이 너무 적음 ({unique_values}개)',
                'recommendation': '그룹 분석 불가능'
            }
        
        # 분위수 후보들 (많은 것부터 적은 것 순으로 시도)
        quantile_options = [
            (5, ['매우낮음', '낮음', '중간', '높음', '매우높음']),
            (4, ['낮음', '중하', '중상', '높음']),
            (3, ['낮음', '중간', '높음']),
        ]
        
        for q_num, labels in quantile_options:
            if unique_values < q_num:
                continue
                
            try:
                print(f"  {q_num}분위 그룹 분석 시도...")
                
                # 분위수 그룹 생성
                feature_quantiles = pd.qcut(feature_values, q=q_num, labels=labels, duplicates='drop')
                
                # 실제 생성된 그룹 수 확인
                actual_groups = feature_quantiles.nunique()
                print(f"  실제 생성된 그룹 수: {actual_groups}")
                
                if actual_groups < 2:
                    print(f"  그룹이 너무 적음, 다음 옵션 시도...")
                    continue
                
                # 그룹별 통계 계산
                valid_data_temp = pd.DataFrame({
                    'feature_group': feature_quantiles,
                    'target': target_values
                }).dropna()
                
                group_stats = valid_data_temp.groupby('feature_group')['target'].agg([
                    'count', 'mean', 'std', 'median'
                ]).round(4)
                
                print("  그룹별 타겟값 통계:")
                print(group_stats)
                
                # 추세 분석
                group_means = group_stats['mean'].values
                trend_analysis = self._analyze_trend(group_means)
                
                print(f"  추세 패턴: {trend_analysis['pattern']}")
                
                return {
                    'quantiles_used': int(q_num),
                    'labels_used': labels[:actual_groups],
                    'group_stats': {
                        index: {
                            'count': int(row['count']),
                            'mean': float(row['mean']),
                            'std': float(row['std']) if not pd.isna(row['std']) else None,
                            'median': float(row['median'])
                        } for index, row in group_stats.iterrows()
                    },
                    'trend_analysis': trend_analysis,
                    'successful': True
                }
                
            except Exception as e:
                print(f"  {q_num}분위 분석 실패: {str(e)}")
                continue
        
        # 모든 분위수 시도 실패
        print("  모든 그룹 분석 시도 실패")
        return {
            'error': '그룹 분석 불가능',
            'reason': '데이터 분포가 그룹 분석에 적합하지 않음 (대부분의 값이 동일하거나 극단적 분포)',
            'recommendation': '다른 분석 방법 고려 필요',
            'successful': False
        }
    
    def _analyze_trend(self, group_means: np.ndarray) -> Dict[str, Any]:
        """그룹별 평균값의 추세 분석"""
        if len(group_means) < 2:
            return {'pattern': '분석 불가', 'monotonic': False}
        
        # 단조성 확인
        increasing = all(group_means[i] <= group_means[i+1] for i in range(len(group_means)-1))
        decreasing = all(group_means[i] >= group_means[i+1] for i in range(len(group_means)-1))
        
        # 추세 강도 계산 (선형 회귀의 기울기)
        x = np.arange(len(group_means))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, group_means)
        
        if increasing:
            pattern = "단조 증가"
        elif decreasing:
            pattern = "단조 감소"
        elif abs(r_value) > 0.5:
            pattern = "선형 관계" if slope > 0 else "음의 선형 관계"
        else:
            pattern = "비선형/불규칙"
        
        return {
            'pattern': pattern,
            'monotonic': bool(increasing or decreasing),
            'slope': float(slope),
            'r_squared': float(r_value**2),
            'trend_strength': "강함" if abs(r_value) > 0.7 else "중간" if abs(r_value) > 0.4 else "약함"
        }
    
    def _create_bivariate_plots(self, feature_name: str, feature_values: pd.Series, 
                               target_values: pd.Series, group_analysis: Dict, 
                               correlation_results: Dict):
        """이변량 분석 시각화"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 산점도
        sns.scatterplot(x=feature_values, y=target_values, ax=ax1, alpha=0.6)
        ax1.set_title(f'{feature_name} vs {self.target_feature}')
        ax1.set_xlabel(feature_name)
        ax1.set_ylabel(self.target_feature)
        
        # 회귀선 추가
        sns.regplot(x=feature_values, y=target_values, ax=ax1, scatter=False, color='red')
        
        # 그룹별 박스플롯 (그룹 분석이 성공한 경우)
        if group_analysis.get('successful', False):
            try:
                # 그룹 분석 결과 재구성
                q_num = group_analysis['quantiles_used']
                labels = group_analysis['labels_used']
                
                temp_quantiles = pd.qcut(feature_values, q=q_num, labels=labels, duplicates='drop')
                valid_data_with_groups = pd.DataFrame({
                    'feature_groups': temp_quantiles,
                    self.target_feature: target_values
                }).dropna()
                
                sns.boxplot(data=valid_data_with_groups, x='feature_groups', y=self.target_feature, ax=ax2)
                ax2.set_title(f'{feature_name} 그룹별 {self.target_feature} 분포')
                ax2.tick_params(axis='x', rotation=45)
                
                # 그룹별 평균값 바차트
                group_stats = group_analysis['group_stats']
                group_names = list(group_stats.keys())
                group_means = [group_stats[name]['mean'] for name in group_names]
                
                bars = ax3.bar(range(len(group_means)), group_means)
                ax3.set_xticks(range(len(group_means)))
                ax3.set_xticklabels(group_names, rotation=45)
                ax3.set_title(f'{feature_name} 그룹별 평균 {self.target_feature}')
                ax3.set_ylabel(f'평균 {self.target_feature}')
                
                # 막대 위에 값 표시
                for bar, val in zip(bars, group_means):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                            f'{val:.2f}', ha='center', va='bottom')
                
            except Exception as e:
                ax2.text(0.5, 0.5, f'그룹별 시각화 오류\n{str(e)}', 
                        transform=ax2.transAxes, ha='center', va='center')
                ax3.text(0.5, 0.5, f'바차트 오류\n{str(e)}', 
                        transform=ax3.transAxes, ha='center', va='center')
        else:
            ax2.text(0.5, 0.5, '그룹 분석 실패\n데이터 분포가 부적합', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax3.text(0.5, 0.5, '그룹 분석 실패\n데이터 분포가 부적합', 
                    transform=ax3.transAxes, ha='center', va='center')
        
        # 상관계수 히트맵
        if 'pearson' in correlation_results['correlations']:
            corr_data = pd.DataFrame({
                'Feature': [feature_name],
                'Target': [correlation_results['correlations']['pearson']['corr']]
            }).set_index('Feature')
            
            sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, ax=ax4, 
                       vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
            ax4.set_title('상관계수')
        
        plt.suptitle(f'Stage 2: {feature_name} 이변량 분석', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _stage3_multivariate_analysis(self, feature_name: str, show_plots: bool) -> Dict[str, Any]:
        """Stage 3: 다변량 분석"""
        print(f"\n분석 Stage 3: 다변량 분석 - '{feature_name}'과 기존 특징들의 중복성 확인")
        print("-" * 60)
        
        # 현재 특징을 제외한 다른 모든 특징들과의 상관관계 계산
        other_features = [f for f in self.available_features if f != feature_name]
        
        if len(other_features) < 2:
            print("비교할 다른 특징이 부족합니다.")
            return {'error': '특징 부족'}
        
        result = {
            'compared_features': int(len(other_features)),
            'high_correlations': [],
            'redundancy_analysis': {}
        }
        
        # Action 3-1: 중복성 확인 (상관관계 행렬)
        print("\n기존 특징들과의 상관관계 분석:")
        
        correlation_results = []
        
        for other_feature in other_features:
            try:
                # 두 특징 모두 유효한 데이터만 선택
                valid_data = self.data[[feature_name, other_feature]].dropna()
                
                if len(valid_data) < 10:
                    continue
                
                corr_coef, p_value = stats.pearsonr(valid_data[feature_name], valid_data[other_feature])
                correlation_results.append({
                    'feature': other_feature,
                    'correlation': float(corr_coef),
                    'abs_correlation': float(abs(corr_coef)),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                })
                
            except Exception as e:
                continue
        
        # 상관계수 기준으로 정렬
        correlation_results.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        # 상위 10개 출력
        print(f"\n  상위 10개 상관관계:")
        for i, corr_result in enumerate(correlation_results[:10]):
            significance = "***" if corr_result['p_value'] < 0.001 else "**" if corr_result['p_value'] < 0.01 else "*" if corr_result['significant'] else ""
            print(f"  {i+1:2d}. {corr_result['feature']:<25} r={corr_result['correlation']:>7.4f} {significance}")
        
        # Check Point 3-1: 높은 상관관계 식별
        print("\n높은 상관관계 특징 식별:")
        
        high_corr_features = [r for r in correlation_results if r['abs_correlation'] >= 0.8]
        very_high_corr_features = [r for r in correlation_results if r['abs_correlation'] >= 0.9]
        
        if very_high_corr_features:
            print(f"  매우 높은 상관관계 (r≥0.9): {len(very_high_corr_features)}개")
            for corr_result in very_high_corr_features:
                print(f"    - {corr_result['feature']}: r={corr_result['correlation']:.4f}")
        
        if high_corr_features:
            remaining_high = len(high_corr_features) - len(very_high_corr_features)
            if remaining_high > 0:
                print(f"  높은 상관관계 (0.8≤r<0.9): {remaining_high}개")
                for corr_result in high_corr_features:
                    if corr_result not in very_high_corr_features:
                        print(f"    - {corr_result['feature']}: r={corr_result['correlation']:.4f}")
        else:
            print("  높은 상관관계(r≥0.8)를 가진 특징이 없습니다. 중복성이 낮습니다.")
        
        result['high_correlations'] = high_corr_features
        result['correlation_analysis'] = correlation_results[:20]  # 상위 20개만 저장
        
        # Action 3-2: 중복 관계 시각화
        if show_plots and high_corr_features:
            self._create_multivariate_plots(feature_name, high_corr_features[:4])
        
        # 중복성 평가
        result['redundancy_analysis'] = self._evaluate_redundancy(very_high_corr_features, 
                                                                 high_corr_features, 
                                                                 correlation_results)
        
        return result
    
    def _create_multivariate_plots(self, feature_name: str, top_corr_features: List[Dict]):
        """다변량 분석 시각화"""
        print("\n높은 상관관계 특징들과의 시각화:")
        
        if len(top_corr_features) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        elif len(top_corr_features) == 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        elif len(top_corr_features) in [3, 4]:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
        
        for i, corr_result in enumerate(top_corr_features):
            other_feature = corr_result['feature']
            valid_data = self.data[[feature_name, other_feature]].dropna()
            
            if len(valid_data) > 0:
                sns.scatterplot(data=valid_data, x=feature_name, y=other_feature, ax=axes[i], alpha=0.6)
                sns.regplot(data=valid_data, x=feature_name, y=other_feature, ax=axes[i], 
                           scatter=False, color='red')
                axes[i].set_title(f'{feature_name} vs {other_feature}\nr = {corr_result["correlation"]:.4f}')
        
        # 사용하지 않는 subplot 숨기기
        for j in range(len(top_corr_features), len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f'Stage 3: {feature_name} 중복성 분석', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _evaluate_redundancy(self, very_high_corr_features: List[Dict], 
                           high_corr_features: List[Dict], 
                           all_correlations: List[Dict]) -> Dict[str, Any]:
        """중복성 평가"""
        
        if very_high_corr_features:
            redundancy_level = "매우 높음"
            redundancy_recommendation = "중복 제거 필요"
        elif high_corr_features:
            redundancy_level = "높음"
            redundancy_recommendation = "중복성 검토 권장"
        elif len([r for r in all_correlations if r['abs_correlation'] >= 0.6]) > 5:
            redundancy_level = "중간"
            redundancy_recommendation = "일부 중복성 존재"
        else:
            redundancy_level = "낮음"
            redundancy_recommendation = "독립적 특징"
        
        result = {
            'level': redundancy_level,
            'recommendation': redundancy_recommendation,
            'high_corr_count': int(len(high_corr_features)),
            'very_high_corr_count': int(len(very_high_corr_features))
        }
        
        print(f"\n중복성 평가: {redundancy_level} - {redundancy_recommendation}")
        
        return result
    
    def _generate_final_recommendation(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """3단계 분석 결과를 종합하여 최종 추천 생성"""
        print(f"\n최종 추천: {validation_result['feature_name']}")
        print("="*60)
        
        feature_name = validation_result['feature_name']
        stage1 = validation_result.get('stage1', {})
        stage2 = validation_result.get('stage2', {})
        stage3 = validation_result.get('stage3', {})
        
        recommendation = {
            'feature_name': feature_name,
            'overall_score': 0,  # 0-100 점수
            'grade': 'F',  # A, B, C, D, F
            'recommendation': '',
            'reasons': [],
            'concerns': [],
            'action_items': []
        }
        
        score = 0
        reasons = []
        concerns = []
        action_items = []
        
        # Stage 1 평가 (30점 만점)
        if 'checks' in stage1:
            checks = stage1['checks']
            
            # 결측치 평가 (10점)
            missing_rate = checks.get('missing_rate', 100)
            if missing_rate < 5:
                score += 10
                reasons.append("결측치가 매우 적음 (<5%)")
            elif missing_rate < 20:
                score += 7
                reasons.append("결측치가 적음 (<20%)")
            elif missing_rate < 50:
                score += 4
                concerns.append(f"결측치가 다소 많음 ({missing_rate:.1f}%)")
            else:
                concerns.append(f"결측치가 매우 많음 ({missing_rate:.1f}%)")
            
            # 변별력 평가 (10점)
            variability = checks.get('variability', {}).get('interpretation', '')
            if variability == '충분함':
                score += 10
                reasons.append("충분한 변별력 보유")
            elif variability == '낮음':
                score += 5
                concerns.append("변별력이 다소 낮음")
            else:
                concerns.append("변별력이 매우 낮음")
            
            # 이상치 평가 (10점)
            outlier_pct = checks.get('outliers', {}).get('percentage', 100)
            if outlier_pct < 5:
                score += 10
                reasons.append("이상치가 적음")
            elif outlier_pct < 15:
                score += 7
            elif outlier_pct < 30:
                score += 4
                concerns.append("이상치가 다소 많음")
            else:
                concerns.append("이상치가 매우 많음")
                action_items.append("이상치 처리 검토 필요")
        
        # Stage 2 평가 (40점 만점)
        if 'correlations' in stage2 and 'pearson' in stage2['correlations']:
            pearson_data = stage2['correlations']['pearson']
            if 'corr' in pearson_data:
                corr_val = abs(pearson_data['corr'])
                p_val = pearson_data.get('p_value', 1)
                
                # 상관관계 강도 평가 (25점)
                if corr_val >= 0.7 and p_val < 0.05:
                    score += 25
                    reasons.append(f"매우 강한 타겟 관계 (r={pearson_data['corr']:.3f})")
                elif corr_val >= 0.5 and p_val < 0.05:
                    score += 20
                    reasons.append(f"강한 타겟 관계 (r={pearson_data['corr']:.3f})")
                elif corr_val >= 0.3 and p_val < 0.05:
                    score += 15
                    reasons.append(f"중간 타겟 관계 (r={pearson_data['corr']:.3f})")
                elif corr_val >= 0.1 and p_val < 0.05:
                    score += 10
                    reasons.append(f"약한 타겟 관계 (r={pearson_data['corr']:.3f})")
                else:
                    concerns.append("타겟과의 관계가 약하거나 유의미하지 않음")
                
                # 통계적 유의성 평가 (15점)
                if p_val < 0.001:
                    score += 15
                    reasons.append("매우 높은 통계적 유의성")
                elif p_val < 0.01:
                    score += 12
                    reasons.append("높은 통계적 유의성")
                elif p_val < 0.05:
                    score += 8
                    reasons.append("통계적으로 유의미함")
                else:
                    concerns.append("통계적으로 유의미하지 않음")
        
        # 그룹 분석 평가 (추가점수 최대 10점)
        if 'group_analysis' in stage2:
            group_data = stage2['group_analysis']
            if group_data.get('successful', False) and 'trend_analysis' in group_data:
                trend = group_data['trend_analysis']['pattern']
                if trend in ['단조 증가', '단조 감소']:
                    score += 10
                    reasons.append(f"명확한 추세 패턴 ({trend})")
                elif "선형" in trend:
                    score += 7
                    reasons.append("선형 관계 존재")
                else:
                    score += 3
                    reasons.append("비선형 관계 존재")
        
        # Stage 3 평가 (30점 만점)
        if 'redundancy_analysis' in stage3:
            redundancy = stage3['redundancy_analysis']
            level = redundancy.get('level', '')
            
            if level == '낮음':
                score += 30
                reasons.append("독립적 특징, 중복성 낮음")
            elif level == '중간':
                score += 20
                reasons.append("일부 중복성 존재하나 허용 수준")
            elif level == '높음':
                score += 10
                concerns.append("기존 특징과 높은 중복성")
                action_items.append("중복 특징과의 비교 분석 필요")
            else:
                score += 0
                concerns.append("매우 높은 중복성, 중복 제거 검토 필요")
                action_items.append("대체 특징 선택 권장")
        
        # 점수 정규화 (최대 100점)
        recommendation['overall_score'] = int(min(100, max(0, score)))
        
        # 등급 부여
        if score >= 80:
            recommendation['grade'] = 'A'
            recommendation['recommendation'] = "A급 특징 - 즉시 모델에 사용 권장"
        elif score >= 65:
            recommendation['grade'] = 'B'
            recommendation['recommendation'] = "B급 특징 - 모델 사용 권장 (일부 개선 고려)"
        elif score >= 50:
            recommendation['grade'] = 'C'
            recommendation['recommendation'] = "C급 특징 - 조건부 사용 권장 (전처리 후)"
        elif score >= 35:
            recommendation['grade'] = 'D'
            recommendation['recommendation'] = "D급 특징 - 신중한 검토 후 사용 여부 결정"
        else:
            recommendation['grade'] = 'F'
            recommendation['recommendation'] = "F급 특징 - 사용 비권장"
        
        recommendation['reasons'] = reasons
        recommendation['concerns'] = concerns  
        recommendation['action_items'] = action_items
        
        # 결과 출력
        print(f"종합 점수: {recommendation['overall_score']}/100 ({recommendation['grade']}등급)")
        print(f"최종 추천: {recommendation['recommendation']}")
        
        if reasons:
            print(f"\n긍정적 요소:")
            for reason in reasons:
                print(f"  • {reason}")
        
        if concerns:
            print(f"\n우려사항:")
            for concern in concerns:
                print(f"  • {concern}")
        
        if action_items:
            print(f"\n개선사항:")
            for action in action_items:
                print(f"  • {action}")
        
        return recommendation
    
    def validate_all_features(self, show_plots: bool = False, save_report: bool = True) -> Dict[str, Dict]:
        """모든 특징에 대한 일괄 검증"""
        print(f"\n{'='*80}")
        print(f"전체 특징 일괄 검증 시작 ({len(self.available_features)}개 특징)")
        print(f"{'='*80}")
        
        if self.target_feature not in self.available_features:
            print(f"오류: 타겟 특징 '{self.target_feature}'를 찾을 수 없습니다.")
            return {}
        
        all_results = {}
        
        for i, feature_name in enumerate(self.available_features, 1):
            if feature_name == self.target_feature:
                continue
                
            print(f"\n진행률: {i-1}/{len(self.available_features)-1} ({(i-1)/(len(self.available_features)-1)*100:.1f}%)")
            
            try:
                result = self.validate_feature(feature_name, show_plots=show_plots)
                all_results[feature_name] = result
            except Exception as e:
                print(f"특징 '{feature_name}' 검증 중 오류: {e}")
                continue
        
        # 결과 요약
        self._generate_summary_report(all_results)
        
        # 결과 저장
        if save_report:
            self._save_validation_report(all_results)
        
        return all_results
    
    def _generate_summary_report(self, all_results: Dict[str, Dict]):
        """전체 검증 결과 요약 리포트 생성"""
        print(f"\n{'='*80}")
        print("전체 특징 검증 요약 리포트")
        print(f"{'='*80}")
        
        if not all_results:
            print("검증된 특징이 없습니다.")
            return
        
        # 등급별 분류
        grade_counts = {'A': [], 'B': [], 'C': [], 'D': [], 'F': []}
        
        for feature_name, result in all_results.items():
            if 'final_recommendation' in result:
                grade = result['final_recommendation'].get('grade', 'F')
                score = result['final_recommendation'].get('overall_score', 0)
                grade_counts[grade].append((feature_name, score))
        
        # 점수순으로 정렬
        for grade in grade_counts:
            grade_counts[grade].sort(key=lambda x: x[1], reverse=True)
        
        # 결과 출력
        total_features = sum(len(features) for features in grade_counts.values())
        
        print(f"등급별 특징 분포 (총 {total_features}개 특징):")
        for grade, features in grade_counts.items():
            count = len(features)
            percentage = count / total_features * 100 if total_features > 0 else 0
            
            grade_descriptions = {
                'A': "즉시 사용 권장",
                'B': "사용 권장", 
                'C': "조건부 사용",
                'D': "신중 검토",
                'F': "사용 비권장"
            }
            
            description = grade_descriptions.get(grade, "")
            print(f"\n{grade}등급: {count}개 ({percentage:.1f}%) - {description}")
            
            if features:
                print(f"   상위 3개:")
                for i, (feature_name, score) in enumerate(features[:3]):
                    print(f"   {i+1}. {feature_name:<30} (점수: {score})")
                
                if len(features) > 3:
                    print(f"   ... 외 {len(features)-3}개")
        
        # 상위 10개 특징 추천
        print(f"\n상위 추천 특징 (Top 10):")
        
        all_features_with_scores = []
        for features in grade_counts.values():
            all_features_with_scores.extend(features)
        
        all_features_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature_name, score) in enumerate(all_features_with_scores[:10], 1):
            grade = 'A' if score >= 80 else 'B' if score >= 65 else 'C' if score >= 50 else 'D' if score >= 35 else 'F'
            print(f"   {i:2d}. {feature_name:<30} {score:>3d}점 ({grade}등급)")
        
        # 카테고리별 성과 분석
        self._analyze_feature_categories(all_results)
    
    def _analyze_feature_categories(self, all_results: Dict[str, Dict]):
        """특징 카테고리별 성과 분석"""
        print(f"\n카테고리별 성과 분석:")
        
        # 카테고리 정의
        categories = {
            "시작점 특징": [f for f in all_results.keys() if f.startswith('start_')],
            "종료점 특징": [f for f in all_results.keys() if f.startswith('end_')],
            "변화량 특징": [f for f in all_results.keys() if 'change' in f or 'pct' in f],
            "극값 특징": [f for f in all_results.keys() if 'max_' in f or 'min_' in f],
            "노이즈/방향성": [f for f in all_results.keys() if any(x in f for x in ['noise', 'direction', 'core'])],
            "거래량 특징": [f for f in all_results.keys() if 'volume' in f]
        }
        
        for category, features in categories.items():
            if not features:
                continue
                
            scores = []
            for feature in features:
                if feature in all_results and 'final_recommendation' in all_results[feature]:
                    score = all_results[feature]['final_recommendation'].get('overall_score', 0)
                    scores.append(score)
            
            if scores:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                print(f"   {category:<15}: 평균 {avg_score:>5.1f}점 | 최고 {max_score:>3.0f}점 | {len(features)}개 특징")
    
    def _save_validation_report(self, all_results: Dict[str, Dict]):
        """검증 결과를 JSON 파일로 저장 (개선된 직렬화)"""
        try:
            report_dir = Path("./validation_reports")
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"feature_validation_{timestamp}.json"
            
            # JSON 직렬화 가능하도록 변환
            serializable_results = {}
            for feature_name, result in all_results.items():
                serializable_results[feature_name] = self._make_json_serializable(result)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n검증 리포트 저장됨: {report_file}")
            
        except Exception as e:
            print(f"리포트 저장 오류: {e}")
    
    def _make_json_serializable(self, obj):
        """JSON 직렬화 가능한 형태로 변환 (개선된 버전)"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif obj is None:
            return None
        elif isinstance(obj, (str, int, float)):
            return obj
        else:
            # 알 수 없는 타입은 문자열로 변환
            try:
                return str(obj)
            except:
                return None

# 사용 예시
def main():
    """메인 실행 함수"""
    try:
        # 데이터 경로 설정
        data_path = Path("./data/cycle_data/structured/cycles_1h.parquet")
        
        if not data_path.exists():
            print(f"데이터 파일을 찾을 수 없습니다: {data_path}")
            return
        
        # 검증기 초기화
        validator = CycleFeatureValidator(data_path)
        
        print("\n특징 검증 모드를 선택하세요:")
        print("1. 단일 특징 검증 (시각화 포함)")
        print("2. 전체 특징 일괄 검증 (요약 리포트)")
        print("3. 상위 5개 특징만 상세 검증")
        
        choice = input("\n선택하세요 (1-3): ").strip()
        
        if choice == "1":
            # 사용 가능한 특징 목록 표시
            print(f"\n사용 가능한 특징들:")
            for i, feature in enumerate(validator.available_features, 1):
                print(f"  {i:2d}. {feature}")
            
            feature_input = input(f"\n검증할 특징명을 입력하세요: ").strip()
            
            if feature_input in validator.available_features:
                result = validator.validate_feature(feature_input, show_plots=True)
                print(f"\n검증 완료: {feature_input}")
            else:
                print(f"특징을 찾을 수 없습니다: {feature_input}")
        
        elif choice == "2":
            print("\n전체 특징 일괄 검증을 시작합니다...")
            all_results = validator.validate_all_features(show_plots=False, save_report=True)
            print(f"\n전체 검증 완료: {len(all_results)}개 특징")
        
        elif choice == "3":
            print("\n상위 특징 상세 검증을 위해 먼저 전체 특징을 빠르게 평가합니다...")
            all_results = validator.validate_all_features(show_plots=False, save_report=False)
            
            # 상위 5개 특징 추출
            feature_scores = []
            for feature_name, result in all_results.items():
                if 'final_recommendation' in result:
                    score = result['final_recommendation'].get('overall_score', 0)
                    feature_scores.append((feature_name, score))
            
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            top_5_features = [feature for feature, score in feature_scores[:5]]
            
            print(f"\n상위 5개 특징 상세 검증:")
            for feature_name in top_5_features:
                print(f"\n{'-'*60}")
                validator.validate_feature(feature_name, show_plots=True)
        
        else:
            print("잘못된 선택입니다.")
    
    except KeyboardInterrupt:
        print("\n\n검증이 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()