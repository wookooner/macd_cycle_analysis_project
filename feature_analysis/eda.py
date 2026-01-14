#!/usr/bin/env python3
"""
고급 커스텀 EDA 리포트 생성기 (Python 3.13 호환)
matplotlib, seaborn, plotly를 활용한 전문적인 분석 리포트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import os
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class AdvancedCycleEDAReporter:
    """고급 사이클 데이터 EDA 리포트 생성 클래스"""
    
    def __init__(self, data_path: str, output_dir: str = "eda_reports"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # 색상 팔레트
        self.colors = {
            'up': '#2E8B57',      # SeaGreen
            'down': '#DC143C',    # Crimson  
            'primary': '#1f77b4', # Blue
            'secondary': '#ff7f0e', # Orange
            'accent': '#2ca02c'    # Green
        }
    
    def ensure_output_dir(self):
        """출력 디렉토리 생성"""
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.output_dir + "/plots").mkdir(exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Parquet 파일 로드"""
        print(f"📁 데이터 로딩 중: {self.data_path}")
        return pd.read_parquet(self.data_path)
    
    def flatten_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """cycle_features 딕셔너리를 플랫한 컬럼으로 변환"""
        print("🔄 cycle_features 플래튼화 중...")
        
        flattened_rows = []
        for _, row in df.iterrows():
            flat_row = {
                'cycle_id': row['cycle_id'],
                'timeframe': row['timeframe'],
                'start_date': row['start_date'],
                'end_date': row['end_date'],
                'cycle_type': row['cycle_type'],
                'duration_candles': row['duration_candles'],
                'category': row['category'],
                'algorithm_used': row['algorithm_used']
            }
            
            features = row['cycle_features']
            if features:
                for category_name, category_data in features.items():
                    if isinstance(category_data, dict):
                        for feature_name, feature_value in category_data.items():
                            column_name = f"{category_name}_{feature_name}"
                            flat_row[column_name] = feature_value
            
            flattened_rows.append(flat_row)
        
        return pd.DataFrame(flattened_rows)
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """파생 특징 추가"""
        print("📊 파생 특징 생성 중...")
        
        # 날짜 관련 - 유닉스 타임스탬프 처리
        if 'start_date' in df.columns:
            try:
                # 먼저 샘플 데이터 확인
                sample_date = str(df['start_date'].iloc[0])
                print(f"🔍 날짜 형식 샘플: {sample_date}")
                
                # 숫자로만 구성되어 있으면 유닉스 타임스탬프로 가정
                if sample_date.isdigit() and len(sample_date) == 10:
                    print("📅 유닉스 타임스탬프로 인식, 변환 중...")
                    df['start_date_dt'] = pd.to_datetime(df['start_date'].astype(int), unit='s')
                else:
                    print("📅 일반 날짜 형식으로 변환 중...")
                    df['start_date_dt'] = pd.to_datetime(df['start_date'])
                
                # 날짜 파생 특징 생성
                df['start_year'] = df['start_date_dt'].dt.year
                df['start_month'] = df['start_date_dt'].dt.month
                df['start_day_of_week'] = df['start_date_dt'].dt.dayofweek
                print(f"✅ 날짜 변환 완료 - 범위: {df['start_date_dt'].min()} ~ {df['start_date_dt'].max()}")
                
            except Exception as e:
                print(f"⚠️  날짜 변환 실패: {e}")
                print("🔄 날짜 관련 파생 특징 건너뜀...")
        
        # 비율 계산
        if 'shape_core_count' in df.columns and 'shape_noise_count' in df.columns:
            df['core_to_noise_ratio'] = df['shape_core_count'] / (df['shape_noise_count'] + 1)
            df['noise_ratio'] = df['shape_noise_count'] / df['duration_candles']
        
        # 강도 카테고리
        if 'strength_direction_pct' in df.columns:
            df['strength_category'] = pd.cut(
                df['strength_direction_pct'], 
                bins=[0, 0.6, 0.8, 1.0], 
                labels=['Weak', 'Medium', 'Strong'],
                include_lowest=True
            )
        
        # 수익성 분석
        if 'change_price_pct' in df.columns:
            df['profitability'] = df['change_price_pct'].apply(
                lambda x: 'Highly Profitable' if x > 5 
                         else 'Profitable' if x > 0
                         else 'Loss' if x < -5
                         else 'Small Loss'
            )
        
        return df
    
    def plot_to_base64(self, fig) -> str:
        """matplotlib 그래프를 base64로 변환"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def generate_basic_stats_table(self, df: pd.DataFrame) -> str:
        """기본 통계 테이블 생성"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_html = """
        <table class="stats-table">
            <tr>
                <th>변수명</th><th>평균</th><th>표준편차</th><th>최솟값</th><th>Q1</th><th>중앙값</th><th>Q3</th><th>최댓값</th><th>결측치</th>
            </tr>
        """
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:  # 데이터가 있는 컬럼만
                stats = df[col].describe()
                missing = df[col].isnull().sum()
                stats_html += f"""
                <tr>
                    <td><strong>{col}</strong></td>
                    <td>{stats['mean']:.4f}</td>
                    <td>{stats['std']:.4f}</td>
                    <td>{stats['min']:.4f}</td>
                    <td>{stats['25%']:.4f}</td>
                    <td>{stats['50%']:.4f}</td>
                    <td>{stats['75%']:.4f}</td>
                    <td>{stats['max']:.4f}</td>
                    <td>{missing}</td>
                </tr>
                """
        
        stats_html += "</table>"
        return stats_html
    
    def create_distribution_plots(self, df: pd.DataFrame) -> list:
        """분포 차트 생성"""
        print("📊 분포 차트 생성 중...")
        plots = []
        
        # 1. 사이클 타입 분포 (파이 차트)
        if 'cycle_type' in df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            cycle_counts = df['cycle_type'].value_counts()
            colors = [self.colors['up'] if x == 'up' else self.colors['down'] for x in cycle_counts.index]
            ax.pie(cycle_counts.values, labels=cycle_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            ax.set_title('Cycle Type Distribution', fontsize=14, fontweight='bold')
            plots.append(('cycle_type_pie', self.plot_to_base64(fig)))
        
        # 2. Duration 분포
        if 'duration_candles' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 히스토그램
            ax1.hist(df['duration_candles'], bins=30, alpha=0.7, color=self.colors['primary'])
            ax1.set_xlabel('Duration (Candles)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Duration Distribution')
            ax1.grid(True, alpha=0.3)
            
            # 박스플롯 (사이클 타입별)
            if 'cycle_type' in df.columns:
                df.boxplot(column='duration_candles', by='cycle_type', ax=ax2)
                ax2.set_title('Duration by Cycle Type')
                ax2.set_xlabel('Cycle Type')
            else:
                ax2.boxplot(df['duration_candles'])
                ax2.set_title('Duration Box Plot')
            
            plt.tight_layout()
            plots.append(('duration_dist', self.plot_to_base64(fig)))
        
        # 3. 주요 강도 지표들 히스토그램
        strength_cols = [col for col in df.columns if 'strength_' in col or 'direction_pct' in col]
        if strength_cols:
            n_cols = len(strength_cols)
            n_rows = (n_cols + 2) // 3
            fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes]
            axes = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]
            
            for i, col in enumerate(strength_cols):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=20, alpha=0.7, color=self.colors['accent'])
                    axes[i].set_title(col.replace('_', ' ').title())
                    axes[i].grid(True, alpha=0.3)
            
            # 남는 subplot 숨기기
            for i in range(len(strength_cols), len(axes)):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            plots.append(('strength_distributions', self.plot_to_base64(fig)))
        
        return plots
    
    def create_correlation_analysis(self, df: pd.DataFrame) -> str:
        """상관관계 분석"""
        print("🔗 상관관계 분석 중...")
        
        # 수치형 컬럼만 선택
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return ""
        
        # 상관계수 계산
        correlation_matrix = numeric_df.corr()
        
        # 히트맵 생성
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .5}, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return self.plot_to_base64(fig)
    
    def create_cycle_type_comparison(self, df: pd.DataFrame) -> list:
        """사이클 타입별 비교 분석"""
        print("⚖️  사이클 타입별 비교 분석 중...")
        plots = []
        
        if 'cycle_type' not in df.columns:
            return plots
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        key_features = [col for col in numeric_cols 
                       if any(keyword in col.lower() for keyword in 
                             ['duration', 'change', 'strength', 'direction'])][:6]  # 상위 6개만
        
        if not key_features:
            return plots
        
        # 그룹별 비교 차트
        n_features = len(key_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        else:
            axes = [axes] if n_rows == 1 else axes
        
        axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]
        
        for i, feature in enumerate(key_features):
            if i < len(axes_flat):
                ax = axes_flat[i]
                df_clean = df[df[feature].notna()]
                if len(df_clean) > 0:
                    df_clean.boxplot(column=feature, by='cycle_type', ax=ax)
                    ax.set_title(f'{feature.replace("_", " ").title()}')
                    ax.set_xlabel('Cycle Type')
        
        # 남는 subplot 숨기기
        for i in range(len(key_features), len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.suptitle('Key Features by Cycle Type', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plots.append(('cycle_comparison', self.plot_to_base64(fig)))
        
        return plots
    
    def create_time_series_analysis(self, df: pd.DataFrame) -> str:
        """시계열 분석"""
        print("📅 시계열 분석 중...")
        
        if 'start_date_dt' not in df.columns:
            print("⚠️  start_date_dt 컬럼이 없어 시계열 분석을 건너뜀")
            return ""
        
        # 유효한 날짜 데이터가 있는지 확인
        if df['start_date_dt'].isnull().all():
            print("⚠️  유효한 날짜 데이터가 없어 시계열 분석을 건너뜀")
            return ""
        
        try:
            # 월별 사이클 개수
            df['year_month'] = df['start_date_dt'].dt.to_period('M')
            monthly_counts = df.groupby(['year_month', 'cycle_type']).size().unstack(fill_value=0)
            
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # 월별 사이클 수 추이
            if not monthly_counts.empty:
                if 'up' in monthly_counts.columns:
                    axes[0].plot(monthly_counts.index.astype(str), monthly_counts['up'], 
                            marker='o', color=self.colors['up'], label='Up Cycles', linewidth=2)
                if 'down' in monthly_counts.columns:
                    axes[0].plot(monthly_counts.index.astype(str), monthly_counts['down'], 
                            marker='s', color=self.colors['down'], label='Down Cycles', linewidth=2)
                
                axes[0].set_title('Monthly Cycle Counts Over Time', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Month')
                axes[0].set_ylabel('Number of Cycles')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[0].tick_params(axis='x', rotation=45)
            
            # 연도별 집계
            if 'start_year' in df.columns:
                yearly_stats = df.groupby(['start_year', 'cycle_type']).size().unstack(fill_value=0)
                if len(yearly_stats) > 0:
                    yearly_stats.plot(kind='bar', ax=axes[1], color=[self.colors['up'], self.colors['down']])
                    axes[1].set_title('Yearly Cycle Distribution', fontsize=14, fontweight='bold')
                    axes[1].set_xlabel('Year')
                    axes[1].set_ylabel('Number of Cycles')
                    axes[1].legend()
                    axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return self.plot_to_base64(fig)
            
        except Exception as e:
            print(f"⚠️  시계열 분석 중 오류 발생: {e}")
            return ""
    
    def generate_advanced_report(self, df: pd.DataFrame, report_name: str = None) -> str:
        """고급 EDA 리포트 생성"""
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timeframe = df['timeframe'].iloc[0] if 'timeframe' in df.columns else 'unknown'
            report_name = f"advanced_cycle_eda_{timeframe}_{timestamp}"
        
        print(f"📈 고급 EDA 리포트 생성 중: {report_name}")
        
        # 기본 통계
        stats_table = self.generate_basic_stats_table(df)
        
        # 차트들 생성
        distribution_plots = self.create_distribution_plots(df)
        correlation_plot = self.create_correlation_analysis(df)
        comparison_plots = self.create_cycle_type_comparison(df)
        timeseries_plot = self.create_time_series_analysis(df)
        
        # HTML 생성
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Advanced Cycle EDA Report - {report_name}</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f8f9fa;
                    color: #333;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white; 
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    padding-bottom: 20px;
                    border-bottom: 3px solid #1f77b4;
                }}
                .section {{ 
                    margin: 40px 0; 
                    padding: 20px;
                    background: #fafafa;
                    border-radius: 8px;
                    border-left: 5px solid #1f77b4;
                }}
                .stats-table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0;
                    background: white;
                }}
                .stats-table th, .stats-table td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px 8px; 
                    text-align: center;
                }}
                .stats-table th {{ 
                    background-color: #1f77b4; 
                    color: white;
                    font-weight: bold;
                }}
                .stats-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .plot-container {{
                    text-align: center;
                    margin: 20px 0;
                    padding: 15px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .plot-container img {{ 
                    max-width: 100%; 
                    height: auto;
                    border-radius: 5px;
                }}
                h1 {{ color: #1f77b4; margin: 0; }}
                h2 {{ 
                    color: #2c3e50; 
                    border-bottom: 2px solid #ecf0f1; 
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                h3 {{ color: #34495e; }}
                .summary-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .summary-card h4 {{ margin: 0 0 10px 0; }}
                .summary-card .value {{ font-size: 24px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Advanced Cycle EDA Report</h1>
                    <p><strong>생성 시간:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>리포트명:</strong> {report_name}</p>
                </div>
                
                <div class="summary-stats">
                    <div class="summary-card">
                        <h4>총 사이클 수</h4>
                        <div class="value">{len(df):,}</div>
                    </div>
                    <div class="summary-card">
                        <h4>특징 개수</h4>
                        <div class="value">{len(df.select_dtypes(include=[np.number]).columns)}</div>
                    </div>
                    <div class="summary-card">
                        <h4>상승 사이클</h4>
                        <div class="value">{len(df[df['cycle_type'] == 'up']) if 'cycle_type' in df.columns else 'N/A'}</div>
                    </div>
                    <div class="summary-card">
                        <h4>하락 사이클</h4>
                        <div class="value">{len(df[df['cycle_type'] == 'down']) if 'cycle_type' in df.columns else 'N/A'}</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 기본 통계 분석</h2>
                    {stats_table}
                </div>
        """
        
        # 분포 차트들 추가
        if distribution_plots:
            html_content += '<div class="section"><h2>📈 분포 분석</h2>'
            for plot_name, plot_data in distribution_plots:
                html_content += f'''
                <div class="plot-container">
                    <h3>{plot_name.replace('_', ' ').title()}</h3>
                    <img src="{plot_data}" alt="{plot_name}">
                </div>
                '''
            html_content += '</div>'
        
        # 상관관계 차트
        if correlation_plot:
            html_content += f'''
            <div class="section">
                <h2>🔗 상관관계 분석</h2>
                <div class="plot-container">
                    <img src="{correlation_plot}" alt="Correlation Matrix">
                </div>
            </div>
            '''
        
        # 비교 분석
        if comparison_plots:
            html_content += '<div class="section"><h2>⚖️ 사이클 타입별 비교</h2>'
            for plot_name, plot_data in comparison_plots:
                html_content += f'''
                <div class="plot-container">
                    <img src="{plot_data}" alt="{plot_name}">
                </div>
                '''
            html_content += '</div>'
        
        # 시계열 분석
        if timeseries_plot:
            html_content += f'''
            <div class="section">
                <h2>📅 시계열 분석</h2>
                <div class="plot-container">
                    <img src="{timeseries_plot}" alt="Time Series Analysis">
                </div>
            </div>
            '''
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # 파일 저장
        output_path = os.path.join(self.output_dir, f"{report_name}.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def run_full_analysis(self, include_derived_features: bool = True) -> str:
        """전체 분석 실행"""
        print("🚀 고급 EDA 리포트 생성 시작")
        print("=" * 60)
        
        # 데이터 로드 및 전처리
        df = self.load_data()
        print(f"📊 원본 데이터 크기: {df.shape}")
        
        df_flat = self.flatten_cycle_features(df)
        print(f"📊 플래튼화 후 데이터 크기: {df_flat.shape}")
        
        if include_derived_features:
            df_flat = self.add_derived_features(df_flat)
            print(f"📊 파생 특징 추가 후 크기: {df_flat.shape}")
        
        # 리포트 생성
        output_path = self.generate_advanced_report(df_flat)
        
        print("=" * 60)
        print(f"✅ 고급 EDA 리포트 생성 완료!")
        print(f"📄 파일 위치: {output_path}")
        print(f"🌐 브라우저에서 열어보세요: {output_path}")
        print("\n📋 포함된 분석:")
        print("   ✓ 기본 통계 및 분포 분석")
        print("   ✓ 상관관계 히트맵") 
        print("   ✓ 사이클 타입별 비교 분석")
        print("   ✓ 시계열 추이 분석")
        print("   ✓ 전문적인 시각화 차트")
        
        return output_path


def main():
    """메인 실행 함수"""
    DATA_PATH = "data/cycle_data/structured/cycles_4h.parquet"
    OUTPUT_DIR = "eda_reports"
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        return
    
    try:
        reporter = AdvancedCycleEDAReporter(DATA_PATH, OUTPUT_DIR)
        report_path = reporter.run_full_analysis(include_derived_features=True)
        print(f"\n🎉 성공! 리포트가 생성되었습니다: {report_path}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()