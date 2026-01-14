"""
방향성 특징 전문 분석 시스템
============================
사이클 방향성 특징에 대한 종합적인 분석 및 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr, normaltest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", palette="husl")


class DirectionalFeatureAnalyzer:
    """방향성 특징 분석기"""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "directional_analysis_results"):
        self.df = df
        self.original_df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 서브 디렉토리
        self.plots_dir = self.output_dir / "plots"
        self.stats_dir = self.output_dir / "statistics"
        self.plots_dir.mkdir(exist_ok=True)
        self.stats_dir.mkdir(exist_ok=True)
        
        # 특징 구분
        self.new_features = [col for col in df.columns if col.startswith('new_')]
        self.existing_features = [col for col in df.columns if col.startswith('existing_')]
        
        print(f"분석 시스템 초기화 완료")
        print(f"  - 새 특징: {len(self.new_features)}개")
        print(f"  - 기존 특징: {len(self.existing_features)}개")
        print(f"  - 전체 사이클: {len(self.df)}개")
    
    def apply_filters(self, **filter_kwargs):
        """필터링 적용
        
        Args:
            cycle_type: List[str] - 사이클 타입 ('up', 'down')
            min_duration: int - 최소 길이
            max_duration: int - 최대 길이
            min_price_change: float - 최소 가격 변화율
            max_price_change: float - 최대 가격 변화율
        """
        filtered = self.original_df.copy()
        
        if 'cycle_type' in filter_kwargs:
            filtered = filtered[filtered['cycle_type'].isin(filter_kwargs['cycle_type'])]
            print(f"사이클 타입 필터: {filter_kwargs['cycle_type']} -> {len(filtered)}개")
        
        if 'min_duration' in filter_kwargs:
            filtered = filtered[filtered['duration_candles'] >= filter_kwargs['min_duration']]
            print(f"최소 길이 필터: >={filter_kwargs['min_duration']} -> {len(filtered)}개")
        
        if 'max_duration' in filter_kwargs:
            filtered = filtered[filtered['duration_candles'] <= filter_kwargs['max_duration']]
            print(f"최대 길이 필터: <={filter_kwargs['max_duration']} -> {len(filtered)}개")
        
        if 'existing_change_price_pct' in filtered.columns:
            if 'min_price_change' in filter_kwargs:
                filtered = filtered[filtered['existing_change_price_pct'] >= filter_kwargs['min_price_change']]
                print(f"최소 가격변화 필터: >={filter_kwargs['min_price_change']}% -> {len(filtered)}개")
            
            if 'max_price_change' in filter_kwargs:
                filtered = filtered[filtered['existing_change_price_pct'] <= filter_kwargs['max_price_change']]
                print(f"최대 가격변화 필터: <={filter_kwargs['max_price_change']}% -> {len(filtered)}개")
        
        self.df = filtered
        self.new_features = [col for col in self.df.columns if col.startswith('new_')]
        self.existing_features = [col for col in self.df.columns if col.startswith('existing_')]
        
        print(f"\n최종 필터링 결과: {len(self.df)}개 사이클")
        return self.df
    
    def reset_filters(self):
        """필터 초기화"""
        self.df = self.original_df.copy()
        print(f"필터 초기화: {len(self.df)}개 사이클")
    
    def analyze_temporal_distribution(self):
        """1. 시간에 따른 데이터 분포 분석"""
        print("\n[1] 시간에 따른 데이터 분포 분석...")
        
        if 'start_date' not in self.df.columns:
            print("  경고: start_date 컬럼이 없어 건너뜁니다.")
            return
        
        # 날짜 변환 (Unix timestamp 처리)
        df_temp = self.df.copy()
        try:
            # Unix timestamp인 경우 (숫자형)
            if pd.api.types.is_numeric_dtype(df_temp['start_date']):
                df_temp['date'] = pd.to_datetime(df_temp['start_date'], unit='s')
            else:
                # 문자열인 경우
                df_temp['date'] = pd.to_datetime(df_temp['start_date'])
        except Exception as e:
            print(f"  경고: 날짜 변환 실패 - {e}")
            print("  시간별 분포 분석을 건너뜁니다.")
            return
        
        df_temp['year_month'] = df_temp['date'].dt.to_period('M')
        
        # 새 특징 중 주요 3개 선택
        key_features = self.new_features[:3] if len(self.new_features) >= 3 else self.new_features
        
        fig, axes = plt.subplots(len(key_features), 1, figsize=(15, 5*len(key_features)))
        if len(key_features) == 1:
            axes = [axes]
        
        for idx, feature in enumerate(key_features):
            ax = axes[idx]
            
            # 월별 평균값 계산
            monthly_data = df_temp.groupby(['year_month', 'cycle_type'])[feature].mean().unstack(fill_value=0)
            
            # 플롯
            monthly_data.plot(ax=ax, marker='o', linewidth=2)
            ax.set_title(f'{feature.replace("new_", "")} - 시간별 추이', fontsize=14, fontweight='bold')
            ax.set_xlabel('기간', fontsize=12)
            ax.set_ylabel('평균값', fontsize=12)
            ax.legend(title='사이클 타입', labels=['상승', '하락'])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.plots_dir / "temporal_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 시간별 분포 그래프 저장: {save_path}")
    
    def analyze_normalized_features(self):
        """2. 데이터 정규화 분석"""
        print("\n[2] 정규화 데이터 분석...")
        
        numeric_new_features = [f for f in self.new_features 
                               if self.df[f].dtype in ['float64', 'int64']]
        
        if not numeric_new_features:
            print("  경고: 분석할 수치형 특징이 없습니다.")
            return
        
        # StandardScaler와 MinMaxScaler 비교
        scaler_std = StandardScaler()
        scaler_minmax = MinMaxScaler()
        
        data = self.df[numeric_new_features].fillna(0)
        data_std = scaler_std.fit_transform(data)
        data_minmax = scaler_minmax.fit_transform(data)
        
        # 시각화
        n_features = min(6, len(numeric_new_features))
        fig, axes = plt.subplots(3, n_features, figsize=(4*n_features, 12))
        
        if n_features == 1:
            axes = axes.reshape(-1, 1)
        
        for i, feature in enumerate(numeric_new_features[:n_features]):
            # 원본
            ax1 = axes[0, i]
            ax1.hist(data[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'원본: {feature.replace("new_", "")}', fontsize=10)
            ax1.axvline(data[feature].mean(), color='red', linestyle='--', linewidth=2)
            
            # StandardScaler
            ax2 = axes[1, i]
            ax2.hist(data_std[:, i], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_title('Standard 정규화', fontsize=10)
            ax2.axvline(data_std[:, i].mean(), color='red', linestyle='--', linewidth=2)
            
            # MinMaxScaler
            ax3 = axes[2, i]
            ax3.hist(data_minmax[:, i], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax3.set_title('MinMax 정규화 (0-1)', fontsize=10)
            ax3.axvline(data_minmax[:, i].mean(), color='red', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        save_path = self.plots_dir / "normalized_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 정규화 분석 그래프 저장: {save_path}")
        
        # 정규화 통계 저장
        norm_stats = pd.DataFrame({
            'Feature': [f.replace('new_', '') for f in numeric_new_features[:n_features]],
            'Original_Mean': data.iloc[:, :n_features].mean().values,
            'Original_Std': data.iloc[:, :n_features].std().values,
            'Std_Norm_Mean': data_std[:, :n_features].mean(axis=0),
            'Std_Norm_Std': data_std[:, :n_features].std(axis=0),
            'MinMax_Min': data_minmax[:, :n_features].min(axis=0),
            'MinMax_Max': data_minmax[:, :n_features].max(axis=0)
        })
        
        norm_stats.to_csv(self.stats_dir / "normalization_stats.csv", index=False)
        print(f"  ✅ 정규화 통계 저장: {self.stats_dir / 'normalization_stats.csv'}")
    
    def analyze_high_correlations(self, threshold: float = 0.6):
        """3. 높은 상관관계 분석 (>= 0.6)"""
        print(f"\n[3] 높은 상관관계 분석 (임계값: {threshold})...")
        
        high_corr_pairs = []
        
        for new_feat in self.new_features:
            if self.df[new_feat].dtype not in ['float64', 'int64']:
                continue
            
            for exist_feat in self.existing_features:
                if self.df[exist_feat].dtype not in ['float64', 'int64']:
                    continue
                
                # 공통 인덱스
                common_idx = self.df[[new_feat, exist_feat]].dropna().index
                if len(common_idx) < 10:
                    continue
                
                x = self.df.loc[common_idx, new_feat]
                y = self.df.loc[common_idx, exist_feat]
                
                try:
                    corr, p_value = pearsonr(x, y)
                    if abs(corr) >= threshold:
                        high_corr_pairs.append({
                            'new_feature': new_feat,
                            'existing_feature': exist_feat,
                            'correlation': corr,
                            'p_value': p_value,
                            'sample_size': len(common_idx)
                        })
                except:
                    continue
        
        if not high_corr_pairs:
            print(f"  ⚠️ 상관관계 {threshold} 이상인 쌍이 없습니다.")
            return
        
        # 상위 12개만 시각화
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        top_pairs = high_corr_pairs[:12]
        
        # 산점도 그리드
        n_pairs = len(top_pairs)
        n_cols = 4
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, pair in enumerate(top_pairs):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            new_f = pair['new_feature']
            exist_f = pair['existing_feature']
            
            common_idx = self.df[[new_f, exist_f]].dropna().index
            x = self.df.loc[common_idx, exist_f]
            y = self.df.loc[common_idx, new_f]
            
            # 사이클 타입별 색상
            colors = self.df.loc[common_idx, 'cycle_type'].map({'up': 'green', 'down': 'red'})
            
            ax.scatter(x, y, c=colors, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
            
            # 회귀선
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel(exist_f.replace('existing_', ''), fontsize=9)
            ax.set_ylabel(new_f.replace('new_', ''), fontsize=9)
            ax.set_title(f'r = {pair["correlation"]:.3f}, p = {pair["p_value"]:.3f}', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 빈 서브플롯 숨기기
        for idx in range(n_pairs, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        save_path = self.plots_dir / f"high_correlation_scatter_{threshold}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 높은 상관관계 산점도 저장: {save_path}")
        
        # 상관관계 데이터 저장
        corr_df = pd.DataFrame(high_corr_pairs)
        corr_df.to_csv(self.stats_dir / f"high_correlations_{threshold}.csv", index=False)
        print(f"  ✅ 상관관계 데이터 저장: {len(high_corr_pairs)}개 쌍")
    
    def analyze_basic_statistics(self):
        """4. 기본 통계 분석"""
        print("\n[4] 기본 통계 분석...")
        
        stats_results = []
        
        for feature in self.new_features:
            if self.df[feature].dtype not in ['float64', 'int64']:
                continue
            
            data = self.df[feature].dropna()
            if len(data) == 0:
                continue
            
            # 통계량 계산
            stat_dict = {
                'Feature': feature.replace('new_', ''),
                'Count': len(data),
                'Mean': data.mean(),
                'Std': data.std(),
                'Min': data.min(),
                'Q25': data.quantile(0.25),
                'Median': data.median(),
                'Q75': data.quantile(0.75),
                'Max': data.max(),
                'Skewness': data.skew(),
                'Kurtosis': data.kurtosis(),
                'CV': data.std() / data.mean() if data.mean() != 0 else 0
            }
            
            # 정규성 검정
            try:
                norm_stat, norm_p = normaltest(data)
                stat_dict['Normality_Stat'] = norm_stat
                stat_dict['Normality_P'] = norm_p
                stat_dict['Is_Normal'] = norm_p > 0.05
            except:
                stat_dict['Normality_Stat'] = np.nan
                stat_dict['Normality_P'] = np.nan
                stat_dict['Is_Normal'] = False
            
            stats_results.append(stat_dict)
        
        stats_df = pd.DataFrame(stats_results)
        
        # 저장
        save_path = self.stats_dir / "basic_statistics.csv"
        stats_df.to_csv(save_path, index=False)
        print(f"  ✅ 기본 통계 저장: {save_path}")
        
        # 요약 출력
        print(f"\n  📊 통계 요약 (상위 5개 특징):")
        for idx, row in stats_df.head(5).iterrows():
            print(f"    • {row['Feature']}: "
                  f"평균={row['Mean']:.4f}, 표준편차={row['Std']:.4f}, "
                  f"CV={row['CV']:.4f}, 정규성={'O' if row['Is_Normal'] else 'X'}")
        
        return stats_df
    
    def analyze_price_change_relationship(self):
        """5. 가격변화율과의 관계 분석"""
        print("\n[5] 가격변화율과의 관계 분석...")
        
        if 'existing_change_price_pct' not in self.df.columns:
            print("  ⚠️ existing_change_price_pct 컬럼이 없어 건너뜁니다.")
            return
        
        price_change_col = 'existing_change_price_pct'
        
        # 주요 새 특징들과의 관계
        key_features = [f for f in self.new_features 
                       if self.df[f].dtype in ['float64', 'int64']][:6]
        
        if not key_features:
            print("  ⚠️ 분석할 수치형 특징이 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(key_features):
            ax = axes[idx]
            
            # 데이터 준비
            valid_idx = self.df[[feature, price_change_col]].dropna().index
            x = self.df.loc[valid_idx, feature]
            y = self.df.loc[valid_idx, price_change_col]
            
            # 사이클 타입별 색상
            colors = self.df.loc[valid_idx, 'cycle_type'].map({'up': 'green', 'down': 'red'})
            
            # 산점도
            ax.scatter(x, y, c=colors, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
            
            # 회귀선
            if len(x) > 5:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "b--", alpha=0.8, linewidth=2)
            
            # 상관계수
            try:
                corr, p_val = pearsonr(x, y)
                ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass
            
            ax.set_xlabel(feature.replace('new_', ''), fontsize=11)
            ax.set_ylabel('가격변화율 (%)', fontsize=11)
            ax.set_title(f'{feature.replace("new_", "")} vs 가격변화율', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
            ax.axvline(x.median(), color='orange', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        save_path = self.plots_dir / "price_change_relationship.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 가격변화 관계 그래프 저장: {save_path}")
    
    def analyze_advanced_insights(self):
        """6. 추가 심층 분석"""
        print("\n[6] 추가 심층 분석...")
        
        # 6-1. 사이클 타입별 특징 분포 비교
        self._analyze_cycle_type_distribution()
        
        # 6-2. 특징 간 상호작용
        self._analyze_feature_interactions()
        
        # 6-3. 극값 분석
        self._analyze_extreme_values()
    
    def _analyze_cycle_type_distribution(self):
        """사이클 타입별 분포 비교"""
        numeric_features = [f for f in self.new_features 
                           if self.df[f].dtype in ['float64', 'int64']][:6]
        
        if not numeric_features:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(numeric_features):
            ax = axes[idx]
            
            up_data = self.df[self.df['cycle_type'] == 'up'][feature].dropna()
            down_data = self.df[self.df['cycle_type'] == 'down'][feature].dropna()
            
            # Violin plot
            parts = ax.violinplot([up_data, down_data], positions=[1, 2], 
                                 showmeans=True, showmedians=True)
            
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['상승', '하락'])
            ax.set_ylabel(feature.replace('new_', ''), fontsize=11)
            ax.set_title(f'{feature.replace("new_", "")} 분포 비교', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 통계적 차이 검정
            try:
                t_stat, p_val = stats.ttest_ind(up_data, down_data)
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                ax.text(0.5, 0.95, f'p = {p_val:.4f} {sig}', transform=ax.transAxes,
                       ha='center', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            except:
                pass
        
        plt.tight_layout()
        save_path = self.plots_dir / "cycle_type_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 사이클 타입별 분포 저장: {save_path}")
    
    def _analyze_feature_interactions(self):
        """특징 간 상호작용 분석"""
        numeric_features = [f for f in self.new_features 
                           if self.df[f].dtype in ['float64', 'int64']][:4]
        
        if len(numeric_features) < 2:
            return
        
        # 페어플롯
        plot_data = self.df[numeric_features + ['cycle_type']].copy()
        plot_data.columns = [c.replace('new_', '') for c in plot_data.columns]
        
        g = sns.pairplot(plot_data, hue='cycle_type', 
                        palette={'up': 'green', 'down': 'red'},
                        diag_kind='kde', plot_kws={'alpha': 0.6})
        
        save_path = self.plots_dir / "feature_interactions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 특징 상호작용 분석 저장: {save_path}")
    
    def _analyze_extreme_values(self):
        """극값 분석"""
        numeric_features = [f for f in self.new_features 
                           if self.df[f].dtype in ['float64', 'int64']][:6]
        
        if not numeric_features:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(numeric_features):
            ax = axes[idx]
            
            data = self.df[feature].dropna()
            
            # 상위/하위 10% 표시
            p10 = data.quantile(0.10)
            p90 = data.quantile(0.90)
            
            ax.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(p10, color='red', linestyle='--', linewidth=2, label=f'10%: {p10:.3f}')
            ax.axvline(p90, color='green', linestyle='--', linewidth=2, label=f'90%: {p90:.3f}')
            ax.axvline(data.median(), color='orange', linestyle='-', linewidth=2, label=f'중앙값: {data.median():.3f}')
            
            ax.set_xlabel(feature.replace('new_', ''), fontsize=11)
            ax.set_ylabel('빈도', fontsize=11)
            ax.set_title(f'{feature.replace("new_", "")} 극값 분석', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.plots_dir / "extreme_values_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 극값 분석 저장: {save_path}")
    
    def generate_comprehensive_report(self):
        """종합 분석 보고서 생성"""
        print("\n[보고서] 종합 분석 보고서 생성...")
        
        report_lines = [
            "=" * 100,
            "방향성 특징 종합 분석 보고서",
            "=" * 100,
            f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"전체 사이클: {len(self.original_df)}개",
            f"분석 사이클: {len(self.df)}개",
            f"분석 특징: {len(self.new_features)}개",
            "",
            "📊 분석된 특징 목록:",
            "-" * 100
        ]
        
        for feature in self.new_features:
            report_lines.append(f"  • {feature.replace('new_', '')}")
        
        report_lines.extend([
            "",
            "📈 생성된 분석 파일:",
            "-" * 100,
            f"  • 시간별 분포: {self.plots_dir / 'temporal_distribution.png'}",
            f"  • 정규화 분석: {self.plots_dir / 'normalized_distribution.png'}",
            f"  • 높은 상관관계: {self.plots_dir / 'high_correlation_scatter_0.6.png'}",
            f"  • 가격변화 관계: {self.plots_dir / 'price_change_relationship.png'}",
            f"  • 사이클 타입 분포: {self.plots_dir / 'cycle_type_distribution.png'}",
            f"  • 특징 상호작용: {self.plots_dir / 'feature_interactions.png'}",
            f"  • 극값 분석: {self.plots_dir / 'extreme_values_analysis.png'}",
            "",
            f"  • 기본 통계: {self.stats_dir / 'basic_statistics.csv'}",
            f"  • 정규화 통계: {self.stats_dir / 'normalization_stats.csv'}",
            f"  • 높은 상관관계 데이터: {self.stats_dir / 'high_correlations_0.6.csv'}",
            "",
            "=" * 100
        ])
        
        report_text = '\n'.join(report_lines)
        print(report_text)
        
        # 보고서 저장
        report_path = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✅ 종합 보고서 저장: {report_path}")
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("\n" + "=" * 100)
        print("방향성 특징 전문 분석 시스템 시작")
        print("=" * 100)
        
        self.analyze_temporal_distribution()
        self.analyze_normalized_features()
        self.analyze_high_correlations(threshold=0.6)
        self.analyze_basic_statistics()
        self.analyze_price_change_relationship()
        self.analyze_advanced_insights()
        self.generate_comprehensive_report()
        
        print("\n" + "=" * 100)
        print(f"✅ 전체 분석 완료! 결과 확인: {self.output_dir}")
        print("=" * 100)


# 메인 실행
if __name__ == "__main__":
    import sys
    sys.path.append('feature_develope')
    from new_feature import DirectionalFeatureManager, flatten_existing_features
    
    # 데이터 로드
    DATA_PATH = "data/cycle_data/structured/cycles_4h.parquet"
    df = pd.read_parquet(DATA_PATH)
    
    print(f"데이터 로드: {len(df)}개 사이클")
    
    # 기존 특징 평면화
    df = flatten_existing_features(df)
    
    # 새 특징 계산
    manager = DirectionalFeatureManager()
    df = manager.batch_calculate(df)
    
    print(f"특징 계산 완료")
    print(f"  - 새 특징: {len(manager.get_feature_list())}개")
    
    # 분석기 초기화
    analyzer = DirectionalFeatureAnalyzer(df)
    
    # 필터링 예시 (선택 사항)
    print("\n필터링 옵션:")
    print("1. 전체 사이클 분석")
    print("2. 상승 사이클만")
    print("3. 하락 사이클만")
    print("4. 맞춤 필터")
    
    choice = input("선택 (1-4): ").strip()
    
    if choice == "2":
        analyzer.apply_filters(cycle_type=['up'])
    elif choice == "3":
        analyzer.apply_filters(cycle_type=['down'])
    elif choice == "4":
        analyzer.apply_filters(
            cycle_type=['up', 'down'],
            min_duration=5,
            max_duration=30
        )
    
    # 전체 분석 실행
    analyzer.run_full_analysis()