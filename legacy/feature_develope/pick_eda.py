import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

warnings.filterwarnings('ignore')

# Python 3.13 호환 라이브러리 체크
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import sweetviz as sv
    SWEETVIZ_AVAILABLE = True
except ImportError:
    SWEETVIZ_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class SelectiveFeatureEDAAnalyzer:
    """특징 선택 및 조건부 필터링 EDA 분석기"""
    
    def __init__(self, data_path: str, output_dir: Optional[str] = None):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else Path("selective_eda_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.flattened_features = {}
        self.selected_features = []
        self.filtered_data = None
        self.analysis_results = {}
        
        # 설정
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self) -> pd.DataFrame:
        """사이클 데이터 로드 및 기본 전처리"""
        print(f"📁 데이터 로딩: {self.data_path}")
        
        self.df = pd.read_parquet(self.data_path)
        print(f"✅ 로드 완료: {len(self.df)}개 사이클")
        
        # 날짜 변환
        try:
            self.df['start_date'] = pd.to_datetime(pd.to_numeric(self.df['start_date']), unit='s')
            self.df['end_date'] = pd.to_datetime(pd.to_numeric(self.df['end_date']), unit='s')
            print("✅ 날짜 변환 완료")
        except:
            print("⚠️ 날짜 변환 실패 - 원본 유지")
        
        # 기본 정보
        print(f"📊 사이클 타입 분포: {dict(self.df['cycle_type'].value_counts())}")
        print(f"📅 기간: {self.df['start_date'].min()} ~ {self.df['start_date'].max()}")
        
        return self.df
    
    def extract_and_flatten_features(self) -> Dict[str, List[str]]:
        """기존 특징들을 평면화하고 카테고리별로 분류"""
        print("\n🔧 특징 추출 및 평면화 중...")
        
        # 기본 컬럼들
        basic_columns = ['cycle_id', 'timeframe', 'start_date', 'end_date', 
                        'cycle_type', 'duration_candles', 'category', 'algorithm_used']
        
        # cycle_features 평면화
        feature_columns = []
        for idx, row in self.df.iterrows():
            features = row['cycle_features']
            if isinstance(features, dict):
                flat_features = self._flatten_dict(features)
                
                if not feature_columns:
                    feature_columns = list(flat_features.keys())
                    for col in feature_columns:
                        self.df[col] = 0.0
                
                for col, value in flat_features.items():
                    if col in self.df.columns:
                        self.df.at[idx, col] = value if value is not None else 0.0
        
        # 특징 카테고리 분류
        self.flattened_features = self._categorize_features(feature_columns)
        
        print(f"✅ 특징 평면화 완료: {len(feature_columns)}개 특징")
        for category, features in self.flattened_features.items():
            print(f"   📂 {category}: {len(features)}개")
        
        return self.flattened_features
    
    def _flatten_dict(self, d: Dict, prefix: str = '') -> Dict:
        """중첩 딕셔너리 평면화"""
        flattened = {}
        for key, value in d.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_dict(value, new_key))
            else:
                flattened[new_key] = value
        return flattened
    
    def _categorize_features(self, feature_columns: List[str]) -> Dict[str, List[str]]:
        """특징들을 카테고리별로 분류"""
        categories = {
            'shape': [],      # 사이클 형태
            'strength': [],   # 강도 관련
            'change': [],     # 변화량
            'start': [],      # 시작점
            'end': [],        # 종료점
            'volatility': [], # 변동성
            'aggregate': [],  # 집계
            'other': []       # 기타
        }
        
        for feature in feature_columns:
            categorized = False
            for category in categories.keys():
                if category in feature.lower():
                    categories[category].append(feature)
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(feature)
        
        # 빈 카테고리 제거
        return {k: v for k, v in categories.items() if v}
    
    def show_feature_selection_interface(self) -> List[str]:
        """대화형 특징 선택 인터페이스"""
        print("\n" + "="*60)
        print("🎯 특징 선택 인터페이스")
        print("="*60)
        
        print("사용 가능한 특징 카테고리:")
        for i, (category, features) in enumerate(self.flattened_features.items(), 1):
            print(f"{i}. {category.upper()}: {len(features)}개 특징")
            print(f"   예시: {', '.join(features[:3])}{'...' if len(features) > 3 else ''}")
        
        selected_features = []
        
        # 1. 카테고리 선택
        print(f"\n📋 특징 선택 방법:")
        print("1. 전체 선택")
        print("2. 카테고리별 선택") 
        print("3. 개별 특징 선택")
        print("4. 자동 추천 (상관관계 기반)")
        
        try:
            choice = input("\n선택 방법을 입력하세요 (1-4): ").strip()
            
            if choice == "1":
                # 전체 선택
                for features in self.flattened_features.values():
                    selected_features.extend(features)
                print(f"✅ 전체 {len(selected_features)}개 특징 선택됨")
                
            elif choice == "2":
                # 카테고리별 선택
                print("\n카테고리를 선택하세요 (쉼표로 구분, 예: 1,3,5):")
                for i, category in enumerate(self.flattened_features.keys(), 1):
                    print(f"{i}. {category}")
                
                cat_input = input("카테고리 번호: ").strip()
                cat_indices = [int(x.strip()) - 1 for x in cat_input.split(',') if x.strip().isdigit()]
                
                categories = list(self.flattened_features.keys())
                for idx in cat_indices:
                    if 0 <= idx < len(categories):
                        selected_features.extend(self.flattened_features[categories[idx]])
                
                print(f"✅ {len(selected_features)}개 특징 선택됨")
                
            elif choice == "3":
                # 개별 특징 선택 (간소화)
                print("\n⚠️ 개별 선택은 복잡하므로 상위 20개 중요 특징으로 자동 선택합니다.")
                selected_features = self._get_top_features(20)
                
            elif choice == "4":
                # 자동 추천
                selected_features = self._auto_recommend_features()
            
            else:
                print("⚠️ 잘못된 선택, 자동 추천으로 진행합니다.")
                selected_features = self._auto_recommend_features()
        
        except (ValueError, KeyboardInterrupt):
            print("⚠️ 입력 오류, 자동 추천으로 진행합니다.")
            selected_features = self._auto_recommend_features()
        
        self.selected_features = selected_features[:50]  # 최대 50개로 제한
        print(f"\n✅ 최종 선택된 특징: {len(self.selected_features)}개")
        
        return self.selected_features
    
    def _get_top_features(self, n: int = 20) -> List[str]:
        """상위 n개 중요 특징 자동 선택"""
        all_features = []
        for features in self.flattened_features.values():
            all_features.extend(features)
        
        # 결측값이 적고 분산이 큰 특징 우선
        feature_scores = []
        for feature in all_features:
            if feature in self.df.columns:
                data = self.df[feature]
                null_ratio = data.isnull().sum() / len(data)
                variance = data.var() if data.dtype in ['float64', 'int64'] else 0
                
                # 점수: (1 - 결측비율) * 분산
                score = (1 - null_ratio) * variance if variance > 0 else 0
                feature_scores.append((feature, score))
        
        # 점수 기준 정렬
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in feature_scores[:n]]
    
    def _auto_recommend_features(self) -> List[str]:
        """자동 특징 추천 (균형 잡힌 선택)"""
        print("🤖 자동 추천 중... (각 카테고리에서 균등하게 선택)")
        
        selected = []
        features_per_category = max(3, 20 // len(self.flattened_features))
        
        for category, features in self.flattened_features.items():
            # 각 카테고리에서 상위 특징들 선택
            category_top = self._get_category_top_features(features, features_per_category)
            selected.extend(category_top)
            print(f"   {category}: {len(category_top)}개 선택")
        
        return selected[:25]  # 총 25개로 제한
    
    def _get_category_top_features(self, features: List[str], n: int) -> List[str]:
        """카테고리 내 상위 특징 선택"""
        valid_features = [f for f in features if f in self.df.columns]
        
        if not valid_features:
            return []
        
        # 분산 기준으로 정렬
        feature_vars = []
        for feature in valid_features:
            data = self.df[feature]
            if data.dtype in ['float64', 'int64']:
                var_score = data.var() * (1 - data.isnull().sum() / len(data))
                feature_vars.append((feature, var_score))
        
        feature_vars.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in feature_vars[:n]]
    
    def show_cycle_filter_interface(self) -> pd.DataFrame:
        """사이클 필터링 인터페이스"""
        print("\n" + "="*60)
        print("🔍 사이클 필터링 인터페이스")
        print("="*60)
        
        print("필터링 옵션:")
        print("1. 전체 사이클 사용")
        print("2. 사이클 타입별 필터링 (up/down)")
        print("3. 기간별 필터링")
        print("4. 길이별 필터링")
        print("5. 성과별 필터링 (가격 변화 기준)")
        print("6. 복합 필터링")
        
        try:
            choice = input("\n필터링 방법을 선택하세요 (1-6): ").strip()
            
            if choice == "1":
                self.filtered_data = self.df.copy()
                print(f"✅ 전체 {len(self.filtered_data)}개 사이클 사용")
                
            elif choice == "2":
                self.filtered_data = self._filter_by_cycle_type()
                
            elif choice == "3":
                self.filtered_data = self._filter_by_period()
                
            elif choice == "4":
                self.filtered_data = self._filter_by_duration()
                
            elif choice == "5":
                self.filtered_data = self._filter_by_performance()
                
            elif choice == "6":
                self.filtered_data = self._filter_complex()
                
            else:
                print("⚠️ 잘못된 선택, 전체 사이클 사용")
                self.filtered_data = self.df.copy()
        
        except (ValueError, KeyboardInterrupt):
            print("⚠️ 입력 오류, 전체 사이클 사용")
            self.filtered_data = self.df.copy()
        
        print(f"🎯 필터링 결과: {len(self.filtered_data)}개 사이클")
        print(f"   사이클 타입 분포: {dict(self.filtered_data['cycle_type'].value_counts())}")
        
        return self.filtered_data
    
    def _filter_by_cycle_type(self) -> pd.DataFrame:
        """사이클 타입별 필터링"""
        cycle_counts = dict(self.df['cycle_type'].value_counts())
        print(f"\n현재 분포: {cycle_counts}")
        
        try:
            cycle_type = input("선택할 사이클 타입 (up/down/both): ").strip().lower()
            
            if cycle_type in ['up', 'down']:
                filtered = self.df[self.df['cycle_type'] == cycle_type].copy()
                print(f"✅ {cycle_type} 사이클 {len(filtered)}개 선택")
                return filtered
            else:
                print("✅ 전체 사이클 사용")
                return self.df.copy()
        except:
            return self.df.copy()
    
    def _filter_by_duration(self) -> pd.DataFrame:
        """길이별 필터링"""
        duration_stats = self.df['duration_candles'].describe()
        print(f"\n사이클 길이 통계:")
        print(f"   평균: {duration_stats['mean']:.1f}")
        print(f"   중앙값: {duration_stats['50%']:.1f}")
        print(f"   범위: {duration_stats['min']:.0f} ~ {duration_stats['max']:.0f}")
        
        try:
            min_dur = input(f"최소 길이 (기본값: {duration_stats['25%']:.0f}): ").strip()
            max_dur = input(f"최대 길이 (기본값: {duration_stats['75%']:.0f}): ").strip()
            
            min_dur = float(min_dur) if min_dur else duration_stats['25%']
            max_dur = float(max_dur) if max_dur else duration_stats['75%']
            
            filtered = self.df[
                (self.df['duration_candles'] >= min_dur) & 
                (self.df['duration_candles'] <= max_dur)
            ].copy()
            
            print(f"✅ 길이 {min_dur}~{max_dur} 사이클 {len(filtered)}개 선택")
            return filtered
            
        except:
            return self.df.copy()
    
    def _filter_by_performance(self) -> pd.DataFrame:
        """성과별 필터링"""
        if 'change_price_pct' not in self.df.columns:
            print("⚠️ 가격 변화 데이터가 없어 전체 사이클 사용")
            return self.df.copy()
        
        perf_stats = self.df['change_price_pct'].describe()
        print(f"\n가격 변화율 통계:")
        print(f"   평균: {perf_stats['mean']:.2f}%")
        print(f"   중앙값: {perf_stats['50%']:.2f}%") 
        print(f"   범위: {perf_stats['min']:.2f}% ~ {perf_stats['max']:.2f}%")
        
        try:
            filter_type = input("필터 타입 (top/bottom/range): ").strip().lower()
            
            if filter_type == "top":
                pct = float(input("상위 몇 %? (예: 20): ") or "20")
                threshold = self.df['change_price_pct'].quantile(1 - pct/100)
                filtered = self.df[self.df['change_price_pct'] >= threshold].copy()
                print(f"✅ 상위 {pct}% 성과 사이클 {len(filtered)}개 선택")
                
            elif filter_type == "bottom":
                pct = float(input("하위 몇 %? (예: 20): ") or "20")
                threshold = self.df['change_price_pct'].quantile(pct/100)
                filtered = self.df[self.df['change_price_pct'] <= threshold].copy()
                print(f"✅ 하위 {pct}% 성과 사이클 {len(filtered)}개 선택")
                
            else:  # range
                min_pct = float(input("최소 변화율% (예: -5): ") or "-5")
                max_pct = float(input("최대 변화율% (예: 10): ") or "10")
                filtered = self.df[
                    (self.df['change_price_pct'] >= min_pct) & 
                    (self.df['change_price_pct'] <= max_pct)
                ].copy()
                print(f"✅ {min_pct}%~{max_pct}% 범위 사이클 {len(filtered)}개 선택")
            
            return filtered
            
        except:
            return self.df.copy()
    
    def _filter_by_period(self) -> pd.DataFrame:
        """기간별 필터링"""
        try:
            date_range = pd.to_datetime(self.df['start_date'])
            print(f"\n데이터 기간: {date_range.min().date()} ~ {date_range.max().date()}")
            
            start_date = input("시작 날짜 (YYYY-MM-DD, 엔터시 전체): ").strip()
            end_date = input("종료 날짜 (YYYY-MM-DD, 엔터시 전체): ").strip()
            
            filtered = self.df.copy()
            
            if start_date:
                start_date = pd.to_datetime(start_date)
                filtered = filtered[pd.to_datetime(filtered['start_date']) >= start_date]
                
            if end_date:
                end_date = pd.to_datetime(end_date)
                filtered = filtered[pd.to_datetime(filtered['start_date']) <= end_date]
            
            print(f"✅ 기간 필터링 결과: {len(filtered)}개 사이클")
            return filtered
            
        except:
            return self.df.copy()
    
    def _filter_complex(self) -> pd.DataFrame:
        """복합 필터링"""
        print("\n🔧 복합 필터링 (단계별 적용)")
        
        filtered = self.df.copy()
        print(f"시작: {len(filtered)}개 사이클")
        
        # 1단계: 사이클 타입
        cycle_type = input("1. 사이클 타입 (up/down/both): ").strip().lower()
        if cycle_type in ['up', 'down']:
            filtered = filtered[filtered['cycle_type'] == cycle_type]
            print(f"   타입 필터링 후: {len(filtered)}개")
        
        # 2단계: 길이
        try:
            min_dur = input("2. 최소 길이 (엔터시 스킵): ").strip()
            if min_dur:
                filtered = filtered[filtered['duration_candles'] >= float(min_dur)]
                print(f"   길이 필터링 후: {len(filtered)}개")
        except:
            pass
        
        # 3단계: 성과
        if 'change_price_pct' in filtered.columns:
            try:
                min_perf = input("3. 최소 성과% (엔터시 스킵): ").strip()
                if min_perf:
                    filtered = filtered[filtered['change_price_pct'] >= float(min_perf)]
                    print(f"   성과 필터링 후: {len(filtered)}개")
            except:
                pass
        
        return filtered
    
    def run_python313_eda_analysis(self):
        """Python 3.13 호환 EDA 분석 실행"""
        if self.filtered_data is None or not self.selected_features:
            print("⚠️ 먼저 특징 선택과 데이터 필터링을 수행해주세요")
            return
        
        print("\n" + "="*60)
        print("📊 Python 3.13 호환 EDA 분석 시작")
        print("="*60)
        
        # 분석 데이터 준비
        analysis_df = self._prepare_analysis_data()
        
        # 분석 실행
        self._run_basic_statistics(analysis_df)
        self._run_distribution_analysis(analysis_df)
        self._run_correlation_analysis(analysis_df)
        self._run_target_analysis(analysis_df)
        self._run_interactive_dashboard(analysis_df)
        
        if SWEETVIZ_AVAILABLE:
            self._run_sweetviz_analysis(analysis_df)
        
        if SKLEARN_AVAILABLE:
            self._run_advanced_analysis(analysis_df)
        
        # 결과 요약
        self._generate_analysis_summary()
        
        print(f"\n🎉 EDA 분석 완료! 결과는 {self.output_dir}에서 확인하세요.")
    
    def _prepare_analysis_data(self) -> pd.DataFrame:
        """분석용 데이터 준비"""
        print("🔧 분석용 데이터 준비 중...")
        
        # 선택된 특징 + 기본 컬럼
        analysis_columns = ['cycle_type', 'duration_candles'] + self.selected_features
        analysis_columns = [col for col in analysis_columns if col in self.filtered_data.columns]
        
        analysis_df = self.filtered_data[analysis_columns].copy()
        
        # 결측값 처리
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
        analysis_df[numeric_cols] = analysis_df[numeric_cols].fillna(0)
        
        # 무한대 값 처리
        analysis_df = analysis_df.replace([np.inf, -np.inf], 0)
        
        print(f"✅ 분석 데이터 준비 완료: {len(analysis_df)}행 × {len(analysis_df.columns)}컬럼")
        
        return analysis_df
    
    def _run_basic_statistics(self, df: pd.DataFrame):
        """기본 통계 분석"""
        print("\n📈 기본 통계 분석...")
        
        # 수치형 컬럼 통계
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe().round(4)
            
            # 추가 통계
            stats_df.loc['skewness'] = df[numeric_cols].skew()
            stats_df.loc['kurtosis'] = df[numeric_cols].kurtosis()
            stats_df.loc['nulls'] = df[numeric_cols].isnull().sum()
            
            # 저장
            stats_path = self.output_dir / "basic_statistics.csv"
            stats_df.to_csv(stats_path)
            
            print(f"   📊 {len(numeric_cols)}개 수치형 특징 통계 저장: {stats_path}")
            
            # 상위 5개 특징만 출력
            print(f"\n   주요 특징 요약 (상위 5개):")
            for col in numeric_cols[:5]:
                data = df[col]
                print(f"   • {col}: 평균={data.mean():.3f}, 표준편차={data.std():.3f}, 범위=[{data.min():.3f}, {data.max():.3f}]")
        
        # 범주형 컬럼 분포
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\n   📋 범주형 특징 분포:")
            for col in categorical_cols:
                dist = dict(df[col].value_counts().head(5))
                print(f"   • {col}: {dist}")
    
    def _run_distribution_analysis(self, df: pd.DataFrame):
        """분포 분석 및 시각화"""
        print("\n📊 분포 분석 및 시각화...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_features = min(12, len(numeric_cols))  # 최대 12개만
        
        if n_features == 0:
            print("   ⚠️ 분석할 수치형 특징이 없습니다")
            return
        
        # 그리드 크기 계산
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numeric_cols[:n_features]):
            row, col_idx = i // n_cols, i % n_cols
            ax = axes[row, col_idx]
            
            data = df[col].dropna()
            
            if len(data) > 0:
                # 히스토그램
                ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(data.mean(), color='red', linestyle='--', alpha=0.8, label=f'Mean: {data.mean():.2f}')
                ax.axvline(data.median(), color='green', linestyle='--', alpha=0.8, label=f'Median: {data.median():.2f}')
                
                ax.set_title(col, fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # 빈 subplot 숨기기
        for i in range(n_features, n_rows * n_cols):
            row, col_idx = i // n_cols, i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        dist_path = self.output_dir / "feature_distributions.png"
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 분포 차트 저장: {dist_path}")
    
    def _run_correlation_analysis(self, df: pd.DataFrame):
        """상관관계 분석"""
        print("\n🔗 상관관계 분석...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            print("   ⚠️ 상관관계 분석할 특징이 부족합니다")
            return
        
        # 상관관계 매트릭스
        corr_matrix = df[numeric_cols].corr()
        
        # 시각화
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix', fontsize=14)
        plt.tight_layout()
        
        corr_path = self.output_dir / "correlation_matrix.png"
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 높은 상관관계 쌍 찾기
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        print(f"   🔗 상관관계 매트릭스 저장: {corr_path}")
        if high_corr_pairs:
            print(f"   ⚠️ 높은 상관관계 특징 쌍 ({len(high_corr_pairs)}개):")
            for feat1, feat2, corr in high_corr_pairs[:5]:  # 상위 5개만
                print(f"      • {feat1} ↔ {feat2}: r = {corr:.3f}")
        else:
            print("   ✅ 높은 상관관계 특징 쌍 없음 (다중공선성 문제 없음)")
        
        # 상관관계 매트릭스 저장
        corr_matrix.to_csv(self.output_dir / "correlation_matrix.csv")
    
    def _run_target_analysis(self, df: pd.DataFrame):
        """타겟 변수 관계 분석"""
        print("\n🎯 사이클 타입별 특징 분석...")
        
        if 'cycle_type' not in df.columns:
            print("   ⚠️ 사이클 타입 정보가 없습니다")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_features = min(8, len(numeric_cols))
        
        if n_features == 0:
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:n_features]):
            ax = axes[i]
            
            # 박스플롯
            df.boxplot(column=col, by='cycle_type', ax=ax)
            ax.set_title(f'{col} by Cycle Type')
            ax.set_xlabel('Cycle Type')
            
        # 빈 subplot 숨기기
        for i in range(n_features, 8):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        target_path = self.output_dir / "target_analysis.png"
        plt.savefig(target_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 통계적 차이 검정
        print(f"   📊 사이클 타입별 특징 차이 분석:")
        up_cycles = df[df['cycle_type'] == 'up']
        down_cycles = df[df['cycle_type'] == 'down']
        
        significant_diffs = []
        for col in numeric_cols[:10]:  # 상위 10개만
            if len(up_cycles[col]) > 0 and len(down_cycles[col]) > 0:
                try:
                    stat, p_value = stats.ttest_ind(up_cycles[col].dropna(), down_cycles[col].dropna())
                    if p_value < 0.05:
                        significant_diffs.append((col, p_value, 
                                               up_cycles[col].mean(), down_cycles[col].mean()))
                except:
                    continue
        
        if significant_diffs:
            print(f"   🔍 통계적으로 유의한 차이를 보이는 특징 ({len(significant_diffs)}개):")
            for col, p_val, up_mean, down_mean in significant_diffs[:5]:
                print(f"      • {col}: p={p_val:.4f}, 상승평균={up_mean:.3f}, 하락평균={down_mean:.3f}")
        
        print(f"   📈 타겟 분석 차트 저장: {target_path}")
    
    def _run_interactive_dashboard(self, df: pd.DataFrame):
        """인터랙티브 대시보드 생성"""
        print("\n🌐 인터랙티브 대시보드 생성...")
        
        try:
            # 메인 대시보드
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('사이클 타입 분포', '기간별 패턴', '특징 관계', '성과 분포'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "histogram"}]]
            )
            
            # 1. 사이클 타입 분포
            type_counts = df['cycle_type'].value_counts()
            fig.add_trace(
                go.Bar(x=type_counts.index, y=type_counts.values, name="사이클 개수",
                       marker_color=['#FF6B6B', '#4ECDC4']),
                row=1, col=1
            )
            
            # 2. 기간별 패턴 (기간이 있다면)
            if 'start_date' in self.filtered_data.columns:
                monthly_data = self.filtered_data.copy()
                monthly_data['month'] = pd.to_datetime(monthly_data['start_date']).dt.to_period('M')
                monthly_counts = monthly_data.groupby(['month', 'cycle_type']).size().unstack(fill_value=0)
                
                for cycle_type in monthly_counts.columns:
                    fig.add_trace(
                        go.Scatter(x=monthly_counts.index.astype(str), y=monthly_counts[cycle_type],
                                 mode='lines+markers', name=f'{cycle_type} cycles'),
                        row=1, col=2
                    )
            
            # 3. 특징 관계 (상위 2개 특징)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                feat1, feat2 = numeric_cols[0], numeric_cols[1]
                for cycle_type in df['cycle_type'].unique():
                    cycle_data = df[df['cycle_type'] == cycle_type]
                    fig.add_trace(
                        go.Scatter(x=cycle_data[feat1], y=cycle_data[feat2],
                                 mode='markers', name=f'{cycle_type}',
                                 opacity=0.6),
                        row=2, col=1
                    )
            
            # 4. 성과 분포 (가격 변화가 있다면)
            if 'change_price_pct' in df.columns:
                fig.add_trace(
                    go.Histogram(x=df['change_price_pct'], name="가격 변화율",
                               opacity=0.7),
                    row=2, col=2
                )
            else:
                # 대체: duration 분포
                fig.add_trace(
                    go.Histogram(x=df['duration_candles'], name="사이클 길이",
                               opacity=0.7),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, showlegend=True, 
                            title_text="📊 선택적 특징 EDA 인터랙티브 대시보드")
            
            dashboard_path = self.output_dir / "interactive_dashboard.html"
            fig.write_html(str(dashboard_path))
            print(f"   🌐 인터랙티브 대시보드 저장: {dashboard_path}")
            
        except Exception as e:
            print(f"   ⚠️ 대시보드 생성 중 오류: {e}")
    
    def _run_sweetviz_analysis(self, df: pd.DataFrame):
        """Sweetviz 분석 (사용 가능한 경우)"""
        print("\n🍭 Sweetviz 분석...")
        
        try:
            # cycle_type을 수치형으로 변환
            df_sweetviz = df.copy()
            df_sweetviz['cycle_type_numeric'] = df_sweetviz['cycle_type'].map({'up': 1, 'down': 0})
            
            report = sv.analyze(df_sweetviz, target_feat='cycle_type_numeric')
            
            sweetviz_path = self.output_dir / "sweetviz_report.html"
            report.show_html(str(sweetviz_path))
            
            print(f"   🍭 Sweetviz 리포트 저장: {sweetviz_path}")
            
        except Exception as e:
            print(f"   ⚠️ Sweetviz 분석 중 오류: {e}")
    
    def _run_advanced_analysis(self, df: pd.DataFrame):
        """고급 분석 (scikit-learn 사용)"""
        print("\n🧠 고급 분석 (차원축소, 이상치 탐지)...")
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 3:
                return
            
            X = df[numeric_cols].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 1. PCA 분석
            pca = PCA(n_components=min(3, X_scaled.shape[1]))
            X_pca = pca.fit_transform(X_scaled)
            
            # PCA 시각화
            fig = plt.figure(figsize=(15, 5))
            
            # 2D PCA
            plt.subplot(1, 3, 1)
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cycle_type'].map({'up': 1, 'down': 0}), 
                                cmap='RdYlBu', alpha=0.6)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.title('PCA Analysis')
            plt.colorbar(scatter)
            
            # 특징 중요도 (분산 기준)
            plt.subplot(1, 3, 2)
            feature_importance = np.var(X_scaled, axis=0)
            top_features = np.argsort(feature_importance)[-10:]
            
            plt.barh(range(len(top_features)), feature_importance[top_features])
            plt.yticks(range(len(top_features)), [numeric_cols[i] for i in top_features])
            plt.xlabel('Variance')
            plt.title('Feature Importance (Variance)')
            
            # 3. 이상치 탐지
            plt.subplot(1, 3, 3)
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(X_scaled)
            
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=outliers, cmap='RdYlBu', alpha=0.6)
            plt.xlabel(f'PC1')
            plt.ylabel(f'PC2')
            plt.title('Outlier Detection')
            
            plt.tight_layout()
            advanced_path = self.output_dir / "advanced_analysis.png"
            plt.savefig(advanced_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   🧠 고급 분석 결과 저장: {advanced_path}")
            print(f"   📊 PCA 설명 분산: {pca.explained_variance_ratio_[:2]} (상위 2개 성분)")
            print(f"   🚨 이상치 비율: {(outliers == -1).sum() / len(outliers):.1%}")
            
        except Exception as e:
            print(f"   ⚠️ 고급 분석 중 오류: {e}")
    
    def _generate_analysis_summary(self):
        """분석 요약 리포트 생성"""
        print("\n📋 분석 요약 리포트 생성...")
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_info': {
                'total_cycles': len(self.df),
                'filtered_cycles': len(self.filtered_data) if self.filtered_data is not None else 0,
                'selected_features': len(self.selected_features),
                'cycle_type_distribution': dict(self.filtered_data['cycle_type'].value_counts()) if self.filtered_data is not None else {}
            },
            'selected_features': self.selected_features,
            'feature_categories': {k: len(v) for k, v in self.flattened_features.items()},
            'recommendations': [
                "✅ 선택된 특징들이 분석에 적합합니다" if len(self.selected_features) >= 5 else "⚠️ 더 많은 특징 선택을 권장합니다",
                "✅ 데이터 품질이 양호합니다" if self.filtered_data is not None and len(self.filtered_data) > 100 else "⚠️ 데이터 양이 부족할 수 있습니다",
                "📊 생성된 시각화와 리포트를 통해 특징들의 패턴을 분석해보세요",
                "🔍 상관관계가 높은 특징들은 선택적으로 사용을 고려하세요"
            ]
        }
        
        summary_path = self.output_dir / "analysis_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   📋 요약 리포트 저장: {summary_path}")
        
        # 콘솔 출력
        print("\n🎯 분석 요약:")
        print(f"   • 전체 사이클: {summary['data_info']['total_cycles']}개")
        print(f"   • 필터링된 사이클: {summary['data_info']['filtered_cycles']}개")
        print(f"   • 선택된 특징: {summary['data_info']['selected_features']}개")
        print(f"   • 사이클 분포: {summary['data_info']['cycle_type_distribution']}")

def main():
    """메인 실행 함수"""
    print("🚀 Python 3.13 호환 선택적 특징 EDA 분석기")
    print("=" * 60)
    
    # 설정
    DATA_PATH = "data/cycle_data/structured/cycles_4h.parquet"
    OUTPUT_DIR = "selective_eda_results"
    
    try:
        # 분석기 초기화
        analyzer = SelectiveFeatureEDAAnalyzer(DATA_PATH, OUTPUT_DIR)
        
        # 1. 데이터 로드 및 특징 추출
        analyzer.load_data()
        analyzer.extract_and_flatten_features()
        
        # 2. 특징 선택 인터페이스
        selected_features = analyzer.show_feature_selection_interface()
        
        # 3. 사이클 필터링 인터페이스
        filtered_data = analyzer.show_cycle_filter_interface()
        
        # 4. EDA 분석 실행
        analyzer.run_python313_eda_analysis()
        
        print(f"\n🎉 모든 분석 완료!")
        print(f"📂 결과 확인: {analyzer.output_dir}")
        print("📊 주요 결과 파일:")
        print("   • interactive_dashboard.html - 인터랙티브 대시보드")
        print("   • correlation_matrix.png - 특징 상관관계")
        print("   • feature_distributions.png - 특징 분포")
        print("   • analysis_summary.json - 분석 요약")
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {DATA_PATH}")
        print("다음 경로들을 확인해보세요:")
        print("  - data/cycle_data/structured/cycles_4h.parquet")
        print("  - ../data/cycle_data/structured/cycles_4h.parquet")
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()