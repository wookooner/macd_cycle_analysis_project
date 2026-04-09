"""
계층적 사이클 분석기 (Hierarchical Cycle Analyzer) - 시각화 기능 추가
=====================================================================
상위/하위 시간대 사이클 관계 분석 도구
성능 최적화 및 유연한 필터링 기능 추가
가격 상승/하락별 분리 통계 출력
특징 분포 시각화 기능 추가

주요 개선사항:
1. 캐싱 메커니즘으로 성능 향상
2. 유연한 필터링 시스템
3. Parent/Child 사이클 필터링 (계층 관계 유지)
4. 개선된 출력 형식
5. Parent 사이클에서의 위치 정보 추가
6. Peak/Trough Position 특징 추가
7. 가격변화율 +/- 비율 통계 추가
8. IndexError 해결 - 안전한 특징 데이터 접근
9. 통계 출력 형식 통일
10. 가격 상승/하락별 특징 통계 분리 출력
11. 특징 분포 시각화 기능 추가
12. 계층별 비교 시각화 기능 추가
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from scipy import stats
from collections import defaultdict
from functools import lru_cache
import re
warnings.filterwarnings('ignore')

# 한글 폰트 설정
def setup_korean_font():

# ============================================
# 개선된 특징 시각화 클래스
# ============================================
class ImprovedFeatureVisualizer:
    """데이터 타입에 맞는 시각화를 제공하는 클래스"""
    
    def __init__(self):
        # 특징 타입 분류
        self.feature_types = {
            # 정수형 특징 (히스토그램/막대그래프)
            'integer': [
                'duration_candles',
                'shape_core_count',
                'shape_noise_count', 
                'shape_direction_change'
            ],
            # 비율형 특징 0-1 (스택 바/파이 차트)
            'ratio': [
                'strength_direction_pct',
                'strength_hist_positive_ratio',
                'strength_price_up_ratio',
                'strength_price_down_ratio',
                'shape_peak_price_position',
                'shape_trough_price_position'
            ],
            # 연속형 특징 (박스플롯/바이올린)
            'continuous': [
                'change_price_pct',
                'start_rsi', 'end_rsi', 'change_rsi',
                'start_macd', 'end_macd', 'change_macd',
                'start_hist', 'end_hist', 'change_hist',
                'start_macd_signal', 'end_macd_signal', 'change_macd_signal',
                'volatility_max_high_pct', 'volatility_max_loss_pct',
                'volatility_max_intraday_high_pct', 'volatility_max_intraday_loss_pct',
                'volatility_avg_true_range', 'volatility_price_change_deviation'
            ]
        }
        
        # 특징 한글 이름 매핑
        self.feature_names_kr = {
            'duration_candles': '지속 기간',
            'shape_core_count': '핵심 캔들 수',
            'shape_noise_count': '노이즈 캔들 수',
            'shape_direction_change': '방향 전환 횟수',
            'strength_direction_pct': '방향성 비율',
            'strength_hist_positive_ratio': 'Hist 양수 비율',
            'strength_price_up_ratio': '가격 상승 비율',
            'strength_price_down_ratio': '가격 하락 비율',
            'shape_peak_price_position': 'Peak 위치',
            'shape_trough_price_position': 'Trough 위치',
            'change_price_pct': '가격 변화율',
            'start_rsi': 'RSI 시작', 'end_rsi': 'RSI 종료', 'change_rsi': 'RSI 변화',
            'start_macd': 'MACD 시작', 'end_macd': 'MACD 종료', 'change_macd': 'MACD 변화',
            'start_hist': 'Hist 시작', 'end_hist': 'Hist 종료', 'change_hist': 'Hist 변화',
            'start_macd_signal': 'Signal 시작', 'end_macd_signal': 'Signal 종료', 
            'change_macd_signal': 'Signal 변화'
        }
    
    def get_feature_type(self, column_name):
        """컬럼 이름으로 특징 타입 판별"""
        for ftype, features in self.feature_types.items():
            if column_name in features:
                return ftype
        return 'continuous'  # 기본값
    
    def plot_integer_feature(self, ax, data_dict, feature_name, title):
        """정수형 특징 시각화 - 히스토그램/막대그래프"""
        if not data_dict:
            ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center')
            ax.set_title(title)
            return
        
        # 모든 데이터를 합쳐서 unique 값 범위 확인
        all_values = pd.concat([pd.Series(data) for data in data_dict.values()])
        unique_count = all_values.nunique()
        
        if unique_count <= 30:  # 범주가 적으면 막대그래프
            # 각 타입별 값의 빈도수 계산
            value_range = sorted(all_values.unique())
            
            bar_width = 0.8 / len(data_dict)
            x = np.arange(len(value_range))
            
            colors = ['#ff6b6b', '#4ecdc4', '#95e1d3', '#f38181']
            
            for i, (label, data) in enumerate(data_dict.items()):
                counts = pd.Series(data).value_counts().reindex(value_range, fill_value=0)
                offset = (i - len(data_dict)/2 + 0.5) * bar_width
                ax.bar(x + offset, counts.values, bar_width, 
                      label=label, alpha=0.8, color=colors[i % len(colors)])
            
            ax.set_xlabel('값')
            ax.set_ylabel('빈도')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{int(v)}' for v in value_range], rotation=45)
            ax.legend()
            
        else:  # 범주가 많으면 히스토그램
            colors = ['#ff6b6b', '#4ecdc4', '#95e1d3', '#f38181']
            
            for i, (label, data) in enumerate(data_dict.items()):
                ax.hist(data, bins=30, alpha=0.6, label=label, 
                       color=colors[i % len(colors)], edgecolor='black')
            
            ax.set_xlabel('값')
            ax.set_ylabel('빈도')
            ax.legend()
        
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 통계 정보 추가
        stats_text = []
        for label, data in data_dict.items():
            data_series = pd.Series(data)
            stats_text.append(f'{label}: μ={data_series.mean():.1f}, M={data_series.median():.0f}')
        
        ax.text(0.02, 0.98, '\n'.join(stats_text), 
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def plot_ratio_feature(self, ax, data_dict, feature_name, title):
        """비율형 특징 시각화 - 스택 바 차트"""
        if not data_dict:
            ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center')
            ax.set_title(title)
            return
        
        # 비율 구간별 분포 계산 (0-0.2, 0.2-0.4, ...)
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        
        distribution = {}
        for label, data in data_dict.items():
            data_series = pd.Series(data)
            counts = pd.cut(data_series, bins=bins, labels=bin_labels, include_lowest=True).value_counts()
            distribution[label] = counts.reindex(bin_labels, fill_value=0)
        
        # 스택 바 차트
        df_dist = pd.DataFrame(distribution)
        
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db']
        df_dist.T.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7)
        
        ax.set_ylabel('빈도')
        ax.set_xlabel('')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.legend(title='비율 구간', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 평균 비율 표시
        stats_text = []
        for label, data in data_dict.items():
            data_series = pd.Series(data)
            stats_text.append(f'{label}: μ={data_series.mean():.2%}')
        
        ax.text(0.02, 0.98, '\n'.join(stats_text), 
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def plot_continuous_feature(self, ax, data_dict, feature_name, title):
        """연속형 특징 시각화 - 박스플롯 + 바이올린"""
        if not data_dict:
            ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center')
            ax.set_title(title)
            return
        
        data_list = list(data_dict.values())
        labels = list(data_dict.keys())
        positions = range(len(data_list))
        
        colors = ['#ff6b6b', '#4ecdc4', '#95e1d3', '#f38181']
        
        # 바이올린 플롯
        parts = ax.violinplot(data_list, positions=positions, 
                             showmeans=True, showmedians=True, widths=0.7)
        
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
        
        # 박스플롯 오버레이
        bp = ax.boxplot(data_list, positions=positions,
                       widths=0.3, patch_artist=True,
                       boxprops=dict(alpha=0.6),
                       medianprops=dict(color='red', linewidth=2))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 통계 정보
        stats_text = []
        for label, data in zip(labels, data_list):
            data_series = pd.Series(data)
            stats_text.append(f'{label}: μ={data_series.mean():.2f}, M={data_series.median():.2f}')
        
        ax.text(0.02, 0.98, '\n'.join(stats_text), 
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def visualize_all_features(self, feature_df, timeframe, output_path=None, title_prefix=""):
        """
        모든 특징을 데이터 타입에 맞게 시각화
        
        Args:
            feature_df: 특징 데이터프레임 (cycle_type 컬럼 포함)
            timeframe: 시간대
            output_path: 저장 경로
            title_prefix: 제목 접두사
        """
        if feature_df is None or feature_df.empty:
            print("⚠️ 시각화할 데이터가 없습니다.")
            return
        
        # up/down 사이클 분리
        up_cycles = feature_df[feature_df['cycle_type'] == 'up']
        down_cycles = feature_df[feature_df['cycle_type'] == 'down']
        
        # 시각화할 특징 수집
        all_features = (self.feature_types['integer'] + 
                       self.feature_types['ratio'] + 
                       self.feature_types['continuous'])
        
        available_features = [col for col in all_features if col in feature_df.columns]
        
        if not available_features:
            print("⚠️ 시각화할 특징이 없습니다.")
            return
        
        # 그리드 설정
        n_features = len(available_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Figure 생성
        fig = plt.figure(figsize=(20, 5 * n_rows))
        fig.suptitle(f'{title_prefix} {timeframe.upper()} 특징 분포 (총 {len(feature_df)}개: 상승 {len(up_cycles)}, 하락 {len(down_cycles)})', 
                     fontsize=16, fontweight='bold', y=0.998)
        
        for idx, col in enumerate(available_features, 1):
            ax = plt.subplot(n_rows, n_cols, idx)
            
            # 데이터 준비
            data_dict = {}
            
            if not up_cycles.empty and col in up_cycles.columns:
                up_data = up_cycles[col].dropna()
                if len(up_data) > 0:
                    data_dict[f'상승 (n={len(up_data)})'] = up_data.values
            
            if not down_cycles.empty and col in down_cycles.columns:
                down_data = down_cycles[col].dropna()
                if len(down_data) > 0:
                    data_dict[f'하락 (n={len(down_data)})'] = down_data.values
            
            # 특징명
            feature_name_kr = self.feature_names_kr.get(col, col)
            title = f'{feature_name_kr}\n({col})'
            
            # 타입별로 적절한 시각화
            feature_type = self.get_feature_type(col)
            
            if feature_type == 'integer':
                self.plot_integer_feature(ax, data_dict, col, title)
            elif feature_type == 'ratio':
                self.plot_ratio_feature(ax, data_dict, col, title)
            else:  # continuous
                self.plot_continuous_feature(ax, data_dict, col, title)
        
        plt.tight_layout()
        
        # 저장
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ 시각화 저장: {output_path}")
        
        plt.show()
        plt.close()


    """한글 폰트 설정 함수"""
    try:
        font_path = 'C:/Windows/Fonts/malgun.ttf'
        if Path(font_path).exists():
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
        else:
            korean_fonts = [font for font in fm.findSystemFonts() 
                          if any(name in font.lower() for name in ['malgun', 'nanum', 'gulim', 'dotum'])]
            if korean_fonts:
                font_prop = fm.FontProperties(fname=korean_fonts[0])
                plt.rcParams['font.family'] = font_prop.get_name()
            else:
                plt.rcParams['font.family'] = 'DejaVu Sans'
                print("⚠️ 한글 폰트를 찾을 수 없어 영문 폰트를 사용합니다.")
        
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        print(f"⚠️ 폰트 설정 실패: {e}")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        return False

setup_korean_font()

class ImprovedHierarchicalCycleAnalyzer:
    def __init__(self, base_path="C:/Users/Administrator/Desktop/macd_cycle_analysis_project"):
        """
        개선된 계층적 사이클 분석기 초기화
        
        Args:
            base_path (str): 프로젝트 루트 경로
        """
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data" / "cycle_data" / "structured"
        self.hierarchy_map_path = self.data_path / "cycle_hierarchy_map.json"
        
        # 시간대 정의
        self.timeframes = ['1m', '1w', '1d', '4h', '1h']
        self.timeframe_hierarchy = {
            '1m': {'parent': None, 'children': ['1w', '1d', '4h', '1h']},
            '1w': {'parent': '1m', 'children': ['1d', '4h', '1h']},
            '1d': {'parent': ['1m', '1w'], 'children': ['4h', '1h']},
            '4h': {'parent': ['1m', '1w', '1d'], 'children': ['1h']},
            '1h': {'parent': ['1m', '1w', '1d', '4h'], 'children': None}
        }
        
        # 데이터 캐시
        self.cycle_data = {}
        self.hierarchy_map = None
        self.available_timeframes = []
        self._feature_cache = {}  # 특징 추출 캐시
        self._child_cache = {}    # 하위 사이클 캐시
        self._position_cache = {} # 위치 정보 캐시
        
        # 출력할 특징 정의 (설정 가능)
        self.display_features = {
            'basic': ['cycle_id', 'start_date', 'end_date', 'cycle_type'],
            'indicators': ['start_rsi', 'start_macd', 'start_macd_signal', 'start_hist',
                          'end_rsi', 'end_macd', 'end_macd_signal', 'end_hist',
                          'change_rsi', 'change_macd', 'change_macd_signal', 'change_hist'],
            'shape': ['duration_candles', 'peak_price_position', 'trough_price_position'],
            'strength': ['direction_pct', 'hist_positive_ratio', 'price_up_ratio', 'price_down_ratio']
        }
        
        self._load_hierarchy_map()
        self._check_available_data()
    
    def _load_hierarchy_map(self):
        """사이클 계층 관계 맵 로드"""
        try:
            with open(self.hierarchy_map_path, 'r', encoding='utf-8') as f:
                self.hierarchy_map = json.load(f)
            print(f"✅ 계층 관계 맵 로드 완료: {len(self.hierarchy_map)} 시간대")
            return True
        except FileNotFoundError:
            print(f"❌ 계층 관계 맵 파일을 찾을 수 없습니다: {self.hierarchy_map_path}")
            return False
        except Exception as e:
            print(f"❌ 계층 관계 맵 로드 실패: {e}")
            return False
    
    def _check_available_data(self):
        """사용 가능한 데이터 파일 확인"""
        self.available_timeframes = []
        for tf in self.timeframes:
            file_path = self.data_path / f"cycles_{tf}.parquet"
            if file_path.exists():
                self.available_timeframes.append(tf)
        print(f"📂 사용 가능한 시간대: {self.available_timeframes}")
    
    @lru_cache(maxsize=32)
    def load_timeframe_data(self, timeframe):
        """특정 시간대의 사이클 데이터 로드 (캐싱)"""
        if timeframe in self.cycle_data:
            return self.cycle_data[timeframe]
        
        if timeframe not in self.available_timeframes:
            print(f"❌ {timeframe} 데이터가 존재하지 않습니다.")
            return None
        
        file_path = self.data_path / f"cycles_{timeframe}.parquet"
        
        try:
            df = pd.read_parquet(file_path)
            self.cycle_data[timeframe] = df
            print(f"✅ {timeframe} 데이터 로드 완료: {len(df)} 사이클")
            return df
        except Exception as e:
            print(f"❌ {timeframe} 데이터 로드 실패: {e}")
            return None
    
    def get_cycle_position_in_parent(self, timeframe, cycle_id, parent_timeframe, parent_cycle_id):
        """Parent 사이클 내에서 현재 사이클의 위치 계산"""
        cache_key = f"{timeframe}_{cycle_id}_{parent_timeframe}_{parent_cycle_id}"
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]
        
        # Parent 사이클의 모든 child 사이클들을 가져오기
        parent_child_cycles = self.get_child_cycles(parent_timeframe, parent_cycle_id)
        
        if timeframe not in parent_child_cycles:
            return None
        
        # 해당 시간대의 child 사이클들을 시간순으로 정렬
        child_df = parent_child_cycles[timeframe].copy()
        child_df = child_df.sort_values('start_date')
        
        # 현재 사이클의 위치 찾기
        cycle_ids = child_df['cycle_id'].tolist()
        if cycle_id in cycle_ids:
            position = cycle_ids.index(cycle_id) + 1
            total = len(cycle_ids)
            result = f"{position}/{total}"
            
            # 캐시에 저장
            self._position_cache[cache_key] = result
            return result
        
        return None
    
    def extract_all_cycle_features(self, df):
        """사이클 특징 추출 (캐싱 적용)"""
        # 캐시 키 생성
        cache_key = id(df)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        feature_data = []
        
        for _, row in df.iterrows():
            cycle_features = row['cycle_features']
            flat_features = {'cycle_id': row['cycle_id'], 'cycle_type': row['cycle_type']}
            
            # 날짜 정보 추가
            if 'start_date' in row:
                flat_features['start_date'] = row['start_date']
            if 'end_date' in row:
                flat_features['end_date'] = row['end_date']
            
            # duration_candles 추가 (원본 데이터에서)
            if 'duration_candles' in row:
                flat_features['duration_candles'] = row['duration_candles']
            
            # cycle_features가 이미 평면화된 dict인지 확인
            if isinstance(cycle_features, dict):
                # 중첩된 구조인지 확인
                has_nested = any(isinstance(v, dict) for v in cycle_features.values())
                
                if has_nested:
                    # 중첩된 구조: {category: {feature: value}}
                    for category, features in cycle_features.items():
                        if isinstance(features, dict):
                            for feature_name, value in features.items():
                                if isinstance(value, (int, float)) and not np.isnan(value):
                                    flat_features[f"{category}_{feature_name}"] = value
                        else:
                            if isinstance(features, (int, float)) and not np.isnan(features):
                                flat_features[category] = features
                else:
                    # 이미 평면화된 구조: {feature: value}
                    for key, value in cycle_features.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            flat_features[key] = value
            
            feature_data.append(flat_features)
        
        result = pd.DataFrame(feature_data)
        
        # 캐시에 저장
        self._feature_cache[cache_key] = result
        return result
    
    def get_safe_feature_value(self, feature_df, cycle_id, feature_name, default_value=0):
        """안전한 특징값 조회 - IndexError 방지"""
        try:
            matching_rows = feature_df[feature_df['cycle_id'] == cycle_id]
            if matching_rows.empty:
                print(f"⚠️ {cycle_id}의 특징 데이터를 찾을 수 없습니다. 기본값 {default_value} 사용")
                return default_value
            
            feature_row = matching_rows.iloc[0]
            return feature_row.get(feature_name, default_value)
        except Exception as e:
            print(f"⚠️ {cycle_id}의 {feature_name} 특징값 조회 실패: {e}. 기본값 {default_value} 사용")
            return default_value
    
    def parse_filter_expression(self, filter_str):
        """필터 표현식 파싱 개선"""
        filters = []
        
        if not filter_str or not filter_str.strip():
            return filters
        
        print(f"🔍 파싱할 필터 문자열: '{filter_str}'")
        
        # 쉼표로 구분된 조건들 분리 (더 정확한 분리)
        parts = [part.strip() for part in filter_str.split(',') if part.strip()]
        
        print(f"🔍 분리된 조건들: {parts}")
        
        for part in parts:
            if not part:
                continue
            
            # cycle_type 필터
            if part.lower() in ['up', 'down']:
                filters.append(('cycle_type', '==', part.lower()))
                print(f"  ✅ cycle_type 필터: {part.lower()}")
                continue
            
            # 범위 표현식: min<feature<max
            range_match = re.match(r'([+-]?\d*\.?\d+)\s*([<>]=?)\s*(\w+)\s*([<>]=?)\s*([+-]?\d*\.?\d+)', part)
            if range_match:
                val1, op1, feature, op2, val2 = range_match.groups()
                val1, val2 = float(val1), float(val2)
                
                if '<' in op1 and '<' in op2:
                    filters.append((feature, '>', val1))
                    filters.append((feature, '<', val2))
                elif '>' in op1 and '>' in op2:
                    filters.append((feature, '>', val2))
                    filters.append((feature, '<', val1))
                print(f"  ✅ 범위 필터: {feature} {op1} {val1} {op2} {val2}")
                continue
            
            # 단일 비교 표현식: feature>value 또는 value<feature
            comp_match = re.match(r'(\w+)\s*([<>]=?|==|!=)\s*([+-]?\d*\.?\d+)', part)
            if comp_match:
                feature, op, value = comp_match.groups()
                filters.append((feature, op, float(value)))
                print(f"  ✅ 비교 필터: {feature} {op} {value}")
                continue
            
            comp_match2 = re.match(r'([+-]?\d*\.?\d+)\s*([<>]=?)\s*(\w+)', part)
            if comp_match2:
                value, op, feature = comp_match2.groups()
                # 연산자 반전
                if '<' in op:
                    op = '>' if '<' == op else '>='
                elif '>' in op:
                    op = '<' if '>' == op else '<='
                filters.append((feature, op, float(value)))
                print(f"  ✅ 반전 비교 필터: {feature} {op} {value}")
                continue
        
            print(f"  ❌ 파싱 실패: '{part}'")
        
        print(f"🔍 최종 파싱된 필터: {filters}")
        return filters
    
    def apply_filters(self, df, filters):
        """필터 적용"""
        if not filters:
            return df
    
        # 특징 데이터 추출
        feature_df = self.extract_all_cycle_features(df)
    
        # 기본 컬럼 병합
        result_df = df.copy()
        for col in feature_df.columns:
            if col not in result_df.columns:
                result_df[col] = feature_df[col]
    
        # 필터 적용
        for feature, op, value in filters:
            if feature not in result_df.columns:
                # 특징 컬럼 이름 변환 시도
                possible_names = [col for col in result_df.columns if feature in col]
                if possible_names:
                    feature = possible_names[0]
                else:
                    continue
        
            # 필터 적용
            if op == '==':
                result_df = result_df[result_df[feature] == value]
            elif op == '!=':
                result_df = result_df[result_df[feature] != value]
            elif op == '>':
                result_df = result_df[result_df[feature] > value]
            elif op == '>=':
                result_df = result_df[result_df[feature] >= value]
            elif op == '<':
                result_df = result_df[result_df[feature] < value]
            elif op == '<=':
                result_df = result_df[result_df[feature] <= value]
    
        return result_df
    
    def _filter_cycles_by_parent_conditions(self, timeframe, cycles_df, parent_filters_map):
        """
        Parent 조건을 만족하는 사이클들만 필터링 (계층 관계 유지)
        """
        valid_cycle_ids = []
        
        print(f"🔍 Parent 조건 검사 시작: {len(cycles_df)} 사이클")
        
        for _, row in cycles_df.iterrows():
            cycle_id = row['cycle_id']
            
            # 이 사이클의 parent들을 확인
            parent_cycles = self.get_parent_cycles(timeframe, cycle_id)
            
            # 모든 parent 필터 조건을 만족하는지 확인
            satisfies_all_parent_conditions = True
            
            for parent_tf, parent_filters in parent_filters_map.items():
                if not parent_filters:  # 필터가 없으면 통과
                    continue
                
                # 해당 시간대의 parent가 있는지 확인
                if parent_tf not in parent_cycles:
                    satisfies_all_parent_conditions = False
                    break
                
                parent_df = parent_cycles[parent_tf]
                
                # parent 필터 조건 적용
                filtered_parent = self.apply_filters(parent_df, parent_filters)
                
                if filtered_parent.empty:
                    satisfies_all_parent_conditions = False
                    break
            
            if satisfies_all_parent_conditions:
                valid_cycle_ids.append(cycle_id)
        
        result_df = cycles_df[cycles_df['cycle_id'].isin(valid_cycle_ids)]
        
        return result_df
    
    def _filter_cycles_by_child_conditions(self, timeframe, cycles_df, child_filters_map):
        """
        Child 조건을 만족하는 사이클들만 필터링 (계층 관계 유지)
        """
        valid_cycle_ids = []
        
        print(f"🔍 Child 조건 검사 시작: {len(cycles_df)} 사이클")
        
        for _, row in cycles_df.iterrows():
            cycle_id = row['cycle_id']
            
            # 이 사이클의 child들을 확인
            child_cycles = self.get_child_cycles(timeframe, cycle_id)
            
            # 모든 child 필터 조건을 만족하는지 확인
            satisfies_all_child_conditions = True
            
            for child_tf, child_filters in child_filters_map.items():
                if not child_filters:  # 필터가 없으면 통과
                    continue
                
                # 해당 시간대의 child가 있는지 확인
                if child_tf not in child_cycles:
                    satisfies_all_child_conditions = False
                    print(f"  ❌ {cycle_id}: {child_tf} child 없음")
                    break
                
                child_df = child_cycles[child_tf]
                
                # child 필터 조건 적용
                filtered_child = self.apply_filters(child_df, child_filters)
                
                if filtered_child.empty:
                    satisfies_all_child_conditions = False
                    break
            
            if satisfies_all_child_conditions:
                valid_cycle_ids.append(cycle_id)
        
        result_df = cycles_df[cycles_df['cycle_id'].isin(valid_cycle_ids)]
        
        return result_df
    
    def _get_related_parent_cycles(self, timeframe, cycle_ids):
        """
        주어진 사이클들과 실제 관계가 있는 parent 사이클들만 수집
        """
        related_parents = {}
        
        for cycle_id in cycle_ids:
            parent_cycles = self.get_parent_cycles(timeframe, cycle_id)
            
            for parent_tf, parent_df in parent_cycles.items():
                if parent_tf not in related_parents:
                    related_parents[parent_tf] = []
                
                # 중복 방지를 위해 cycle_id로 확인
                existing_ids = [df['cycle_id'].iloc[0] for df in related_parents[parent_tf] 
                              if not df.empty]
                
                current_parent_id = parent_df['cycle_id'].iloc[0] if not parent_df.empty else None
                
                if current_parent_id and current_parent_id not in existing_ids:
                    related_parents[parent_tf].append(parent_df)
        
        # DataFrame으로 병합
        final_parents = {}
        for parent_tf, parent_dfs in related_parents.items():
            if parent_dfs:
                combined_df = pd.concat(parent_dfs).drop_duplicates(subset=['cycle_id'])
                final_parents[parent_tf] = combined_df
        
        return final_parents
    
    def _get_related_child_cycles(self, timeframe, cycle_ids):
        """
        주어진 사이클들과 실제 관계가 있는 child 사이클들만 수집
        """
        related_children = {}
        
        for cycle_id in cycle_ids:
            child_cycles = self.get_child_cycles(timeframe, cycle_id)
            
            for child_tf, child_df in child_cycles.items():
                if child_tf not in related_children:
                    related_children[child_tf] = []
                related_children[child_tf].append(child_df)
        
        # DataFrame으로 병합
        final_children = {}
        for child_tf, child_dfs in related_children.items():
            if child_dfs:
                combined_df = pd.concat(child_dfs).drop_duplicates(subset=['cycle_id'])
                final_children[child_tf] = combined_df
        
        return final_children
    
    def analyze_filtered_cycles_with_hierarchy(self, timeframe, filters=None, parent_filters_map=None, child_filters_map=None):
        """
        계층 관계를 유지하면서 필터링된 사이클 분석 (수정된 메인 로직)
        
        Args:
            timeframe: 기준 시간대
            filters: 기준 시간대 필터 조건
            parent_filters_map: parent 시간대별 필터 조건 {timeframe: filters}
            child_filters_map: child 시간대별 필터 조건 {timeframe: filters}
        
        Returns:
            tuple: (필터링된_기준_사이클, 관련_parent_사이클, 관련_child_사이클)
        """
        # 1. 기준 시간대 데이터 로드
        df = self.load_timeframe_data(timeframe)
        if df is None:
            print(f"❌ {timeframe} 데이터를 로드할 수 없습니다.")
            return None, None, None
        
        # 2. 기준 시간대 초기 필터링
        main_filtered_df = self.apply_filters(df, filters) if filters else df
        
        if main_filtered_df.empty:
            print("❌ 기준 시간대 필터 조건에 맞는 사이클이 없습니다.")
            return None, None, None
        
        print(f"✅ 기준 시간대 초기 필터링 결과: {len(main_filtered_df)} 사이클")
        
        # 3. Parent 필터링이 있는 경우 계층 관계 고려하여 필터링
        if parent_filters_map:
            main_filtered_df = self._filter_cycles_by_parent_conditions(
                timeframe, main_filtered_df, parent_filters_map
            )
            
            if main_filtered_df.empty:
                return None, None, None
            
            print(f"✅ Parent 조건 적용 후: {len(main_filtered_df)} 사이클")
        
        # 4. Child 필터링이 있는 경우 계층 관계 고려하여 필터링
        if child_filters_map:
            main_filtered_df = self._filter_cycles_by_child_conditions(
                timeframe, main_filtered_df, child_filters_map
            )
            
            if main_filtered_df.empty:
                print("❌ Child 필터 조건을 만족하는 사이클이 없습니다.")
                return None, None, None
            
            print(f"✅ Child 조건 적용 후: {len(main_filtered_df)} 사이클")
        
        # 5. 최종 필터링된 사이클들의 실제 parent/child 수집
        final_cycle_ids = main_filtered_df['cycle_id'].tolist()
        
        # Parent 사이클들 수집 (실제 관계가 있는 것만)
        related_parent_cycles = self._get_related_parent_cycles(timeframe, final_cycle_ids)
        
        # Child 사이클들 수집 (실제 관계가 있는 것만)  
        related_child_cycles = self._get_related_child_cycles(timeframe, final_cycle_ids)
        
        print(f"📊 최종 결과:")
        print(f"   - 기준 {timeframe}: {len(main_filtered_df)} 사이클")
        if related_parent_cycles:
            for tf, df in related_parent_cycles.items():
                print(f"   - Parent {tf}: {len(df)} 사이클")
        if related_child_cycles:
            for tf, df in related_child_cycles.items():
                print(f"   - Child {tf}: {len(df)} 사이클")
        
        return main_filtered_df, related_parent_cycles, related_child_cycles
    
    def format_datetime_kst(self, date_str):
        """UTC 시간을 KST로 변환 (UTC + 9시간)"""
        try:
            dt = pd.to_datetime(date_str)
            kst_dt = dt + timedelta(hours=9)
            return kst_dt.strftime('%Y-%m-%d %H:%M KST')
        except:
            return date_str
    
    def get_cycle_position(self, parent_df, cycle_id):
        """부모 사이클에서 특정 사이클의 위치 계산"""
        if parent_df is None or parent_df.empty:
            return None
        
        # 시간순 정렬
        sorted_df = parent_df.sort_values('start_date')
        
        # 해당 사이클의 인덱스 찾기
        try:
            idx = sorted_df[sorted_df['cycle_id'] == cycle_id].index[0]
            position = sorted_df.index.get_loc(idx)
            total = len(sorted_df)
            return f"{position + 1}/{total} ({(position + 1) / total * 100:.1f}%)"
        except:
            return None
    
    def get_parent_cycles(self, timeframe, cycle_id):
        """특정 사이클의 부모 사이클들 찾기"""
        parent_cycles = {}
        
        if not self.hierarchy_map or timeframe not in self.hierarchy_map:
            return parent_cycles
        
        if cycle_id not in self.hierarchy_map[timeframe]:
            return parent_cycles
        
        cycle_info = self.hierarchy_map[timeframe][cycle_id]
        parent_ids = cycle_info.get('parent_cycle_ids', {})
        
        for parent_tf, parent_id_list in parent_ids.items():
            if parent_id_list and parent_tf in self.available_timeframes:
                parent_df = self.load_timeframe_data(parent_tf)
                if parent_df is not None:
                    for pid in parent_id_list:
                        parent_cycle = parent_df[parent_df['cycle_id'] == pid]
                        if not parent_cycle.empty:
                            parent_cycles[parent_tf] = parent_cycle
                            break
        
        return parent_cycles
    
    def get_child_cycles(self, timeframe, cycle_id):
        """특정 사이클의 하위 사이클들 찾기 (캐싱)"""
        cache_key = f"{timeframe}_{cycle_id}"
        if cache_key in self._child_cache:
            return self._child_cache[cache_key]
        
        child_cycles = {}
        
        if not self.hierarchy_map or timeframe not in self.hierarchy_map:
            return child_cycles
        
        if cycle_id not in self.hierarchy_map[timeframe]:
            return child_cycles
        
        cycle_info = self.hierarchy_map[timeframe][cycle_id]
        child_ids = cycle_info.get('child_cycle_ids', {})
        
        for child_tf, child_id_list in child_ids.items():
            if child_id_list and child_tf in self.available_timeframes:
                child_df = self.load_timeframe_data(child_tf)
                if child_df is not None:
                    existing_cycles = child_df[child_df['cycle_id'].isin(child_id_list)]
                    if not existing_cycles.empty:
                        child_cycles[child_tf] = existing_cycles
        
        # 캐시에 저장
        self._child_cache[cache_key] = child_cycles
        return child_cycles
    
    def print_cycle_details(self, cycle_df, label="사이클", show_stats=False):
        """사이클 상세 정보 출력"""
        if cycle_df is None or cycle_df.empty:
            print(f"\n{label} 데이터가 없습니다.")
            return
        
        print(f"\n{'='*80}")
        print(f"### {label} ###")
        print('='*80)
        
        if show_stats and len(cycle_df) > 1:
            # 통계 모드
            feature_df = self.extract_all_cycle_features(cycle_df)
            
            print(f"이 사이클 개수: {len(cycle_df)}")
            print(f"사이클 타입 분포: {cycle_df['cycle_type'].value_counts().to_dict()}")
            
            # 주요 특징 통계
            stat_features = []
            for category_features in self.display_features.values():
                stat_features.extend(category_features)
            
            print("\n주요 특징 통계:")
            for feature in stat_features:
                if feature in ['cycle_id', 'start_date', 'end_date', 'cycle_type']:
                    continue
                
                # 특징 컬럼 찾기
                matching_cols = [col for col in feature_df.columns if feature in col]
                if matching_cols:
                    col = matching_cols[0]
                    if col in feature_df.columns:
                        values = feature_df[col].dropna()
                        if len(values) > 0:
                            print(f"  {feature}:")
                            print(f"    평균: {values.mean():.4f}, 중앙값: {values.median():.4f}")
                            print(f"    Q1: {values.quantile(0.25):.4f}, Q3: {values.quantile(0.75):.4f}")
        else:
            # 개별 데이터 모드
            for idx, row in cycle_df.iterrows():
                feature_dict = self.extract_single_cycle_features(row)
                
                # 사이클 ID와 타입
                cycle_id = feature_dict.get('cycle_id', 'Unknown')
                cycle_type = feature_dict.get('cycle_type', '')
                type_emoji = '🔴' if cycle_type == 'up' else '🔵'
                
                print(f"\n📊 {cycle_id} {type_emoji} ({cycle_type.upper()})")
                print("─" * 60)
                
                # 시간 정보
                start_kst = self.format_datetime_kst(feature_dict.get('start_date', ''))
                end_kst = self.format_datetime_kst(feature_dict.get('end_date', ''))
                print(f"⏰ 기간: {start_kst} ~ {end_kst}")
                
                # 기본 정보
                duration = feature_dict.get('duration_candles', 0)
                print(f"📈 지속시간: {duration} 캔들")
                
                # 지표 정보 (시작/종료)
                print(f"\n📋 지표 정보:")
                print(f"  시작 RSI: {feature_dict.get('start_rsi', 0):.2f} → 종료 RSI: {feature_dict.get('end_rsi', 0):.2f} (변화: {feature_dict.get('change_rsi', 0):+.2f})")
                print(f"  시작 MACD: {feature_dict.get('start_macd', 0):.2f} → 종료 MACD: {feature_dict.get('end_macd', 0):.2f} (변화: {feature_dict.get('change_macd', 0):+.2f})")
                print(f"  시작 Signal: {feature_dict.get('start_macd_signal', 0):.2f} → 종료 Signal: {feature_dict.get('end_macd_signal', 0):.2f} (변화: {feature_dict.get('change_macd_signal', 0):+.2f})")
                print(f"  시작 Hist: {feature_dict.get('start_hist', 0):.2f} → 종료 Hist: {feature_dict.get('end_hist', 0):.2f} (변화: {feature_dict.get('change_hist', 0):+.2f})")
                
                # 가격 정보
                price_change = feature_dict.get('change_price_pct', 0)
                print(f"\n💰 가격 변화: {price_change:+.2f}%")
                
                # 형태 정보
                peak_pos = feature_dict.get('shape_peak_price_position', 0)
                trough_pos = feature_dict.get('shape_trough_price_position', 0)
                print(f"\n🔍 형태 정보:")
                print(f"  피크 위치: {peak_pos:.2f}, 골 위치: {trough_pos:.2f}")
                
                # 강도 정보
                direction_pct = feature_dict.get('strength_direction_pct', 0)
                hist_positive = feature_dict.get('strength_hist_positive_ratio', 0)
                price_up = feature_dict.get('strength_price_up_ratio', 0)
                price_down = feature_dict.get('strength_price_down_ratio', 0)
                print(f"\n💪 강도 정보:")
                print(f"  방향성: {direction_pct:.1f}%, 히스토그램 양수: {hist_positive:.1f}%")
                print(f"  상승 비율: {price_up:.1f}%, 하락 비율: {price_down:.1f}%")
    
    def extract_single_cycle_features(self, row):
        """단일 사이클의 특징 추출"""
        cycle_features = row['cycle_features']
        flat_features = {
            'cycle_id': row['cycle_id'],
            'cycle_type': row['cycle_type'],
            'start_date': row.get('start_date', ''),
            'end_date': row.get('end_date', ''),
            'duration_candles': row.get('duration_candles', 0)
        }
        
        # cycle_features가 이미 평면화된 dict인지 확인
        if isinstance(cycle_features, dict):
            # 중첩된 구조인지 확인
            has_nested = any(isinstance(v, dict) for v in cycle_features.values())
            
            if has_nested:
                # 중첩된 구조: {category: {feature: value}}
                for category, features in cycle_features.items():
                    if isinstance(features, dict):
                        for feature_name, value in features.items():
                            flat_features[f"{category}_{feature_name}"] = value
                    else:
                        flat_features[category] = features
            else:
                # 이미 평면화된 구조: {feature: value}
                flat_features.update(cycle_features)
        
        return flat_features
    
    def print_cycle_list(self, df, timeframe):
        """개선된 사이클 목록 출력 - parent 위치 정보 및 특징 추가"""
        print(f"\n{'='*120}")
        print(f"📋 필터링된 {timeframe} 사이클 목록 (위치 정보 포함)")
        print('='*120)
        
        feature_df = self.extract_all_cycle_features(df)
        
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            cycle_id = row['cycle_id']
            cycle_type = row['cycle_type']
            duration = row.get('duration_candles', 0)
            
            # 특징 값 가져오기
            feature_row = feature_df[feature_df['cycle_id'] == cycle_id].iloc[0] if not feature_df[feature_df['cycle_id'] == cycle_id].empty else {}
            
            # 가격 변화율
            price_change = feature_row.get('change_price_pct', 0) if isinstance(feature_row, dict) else feature_row.get('change_price_pct', 0)
            
            # Peak/Trough Position 추가
            peak_pos = feature_row.get('shape_peak_price_position', 0) if isinstance(feature_row, dict) else feature_row.get('shape_peak_price_position', 0)
            trough_pos = feature_row.get('shape_trough_price_position', 0) if isinstance(feature_row, dict) else feature_row.get('shape_trough_price_position', 0)
            
            # Parent 사이클에서의 위치 정보
            parent_cycles = self.get_parent_cycles(timeframe, cycle_id)
            position_info_parts = []
            
            for parent_tf, parent_df in parent_cycles.items():
                if not parent_df.empty:
                    parent_cycle_id = parent_df['cycle_id'].iloc[0]
                    position = self.get_cycle_position_in_parent(timeframe, cycle_id, parent_tf, parent_cycle_id)
                    if position:
                        position_info_parts.append(f"{parent_tf}:{position}")
            
            # 위치 정보 문자열
            if position_info_parts:
                position_info_str = " | ".join(position_info_parts)
            else:
                position_info_str = "최상위"
            
            # 하위 사이클 정보 (시간대별로 개수 표시)
            child_cycles = self.get_child_cycles(timeframe, cycle_id)
            child_info_parts = []
            total_children = 0
            
            for child_tf, child_df in child_cycles.items():
                count = len(child_df)
                child_info_parts.append(f"{child_tf}({count}개)")
                total_children += count
            
            # 하위 사이클이 없는 경우
            if not child_info_parts:
                child_info_str = "없음"
            else:
                child_info_str = ", ".join(child_info_parts)
            
            type_emoji = '🔴' if cycle_type == 'up' else '🔵'
            
            print(f"{idx:3d}. {cycle_id} {type_emoji} "
                  f"({price_change:+.2f}% | {duration}캔들 | "
                  f"Peak:{peak_pos:.2f} Trough:{trough_pos:.2f} | "
                  f"위치: {position_info_str} | 하위: {child_info_str})")
    
    def print_cycle_summary_table(self, cycles_dict, title="사이클 요약"):
        """사이클들을 테이블 형태로 요약 출력 (안전한 특징 값 접근으로 IndexError 방지)"""
        if not cycles_dict:
            print(f"\n{title}: 없음")
            return
    
        print(f"\n{'='*80}")
        print(f"📊 {title}")
        print('='*80)
    
        for tf, df in cycles_dict.items():
            if df.empty:
                continue
            
            print(f"\n📈 {tf.upper()} 사이클 ({len(df)}개):")
        
            # 1. 모든 행의 Position과 Children 정보를 미리 계산하여 리스트에 저장
            table_rows_data = []
            for _, row in df.iterrows():
                cycle_id = row['cycle_id']
                # Position 정보 계산
                parent_cycles = self.get_parent_cycles(tf, cycle_id)
                position_info_parts = []
                for parent_tf, parent_df in parent_cycles.items():
                    if not parent_df.empty:
                        parent_row = parent_df.iloc[0]
                        parent_cycle_id, parent_cycle_type = parent_row['cycle_id'], parent_row['cycle_type']
                        parent_type_emoji = '🔴' if parent_cycle_type == 'up' else '🔵'
                        position = self.get_cycle_position_in_parent(tf, cycle_id, parent_tf, parent_cycle_id)
                        if position:
                            position_info_parts.append(f"{parent_type_emoji}{parent_tf}:{position}")
                position_info_str = " | ".join(position_info_parts) if position_info_parts else "최상위"

                # Children 정보 계산
                child_cycles = self.get_child_cycles(tf, cycle_id)
                child_info_parts = []
                for child_tf, child_df in child_cycles.items():
                    child_info_parts.append(f"{child_tf}({len(child_df)})")
                child_info_str = ", ".join(child_info_parts) if child_info_parts else "없음"

                table_rows_data.append({
                    'position': position_info_str,
                    'children': child_info_str
                })

            # 2. 계산된 정보들 중 가장 긴 길이를 찾아 동적 너비 결정 (최소 너비 + 여백 포함)
            pos_width = max([len(d['position']) for d in table_rows_data] + [10]) + 2
            child_width = max([len(d['children']) for d in table_rows_data] + [10]) + 2

            feature_df = self.extract_all_cycle_features(df)

            # 3. 동적으로 계산된 너비로 헤더 생성
            header = (f"{'No':<3} {'Cycle ID':<27} {'Type':<6} {'Dur':>3} {'%':>6}   "
                      f"{'Position':<{pos_width}} {'Children':<{child_width}} "
                      f"{'Peak':>6} {'Trough':>7} {'RSI':^15} {'Macd':^15} {'Macd Signal':^15} {'Hist':^15}")
            print("-" * len(header))
            print(header)
            print("-" * len(header))

            for idx, (_, row) in enumerate(df.iterrows(), 1):
                cycle_id = row['cycle_id']
                cycle_type = row['cycle_type']
                duration = row.get('duration_candles', 0)

                # 안전한 특징 값 가져오기 (IndexError 방지)
                price_change = self.get_safe_feature_value(feature_df, cycle_id, 'change_price_pct', 0)
                peak_pos = self.get_safe_feature_value(feature_df, cycle_id, 'shape_peak_price_position', 0)
                trough_pos = self.get_safe_feature_value(feature_df, cycle_id, 'shape_trough_price_position', 0)
                start_rsi = self.get_safe_feature_value(feature_df, cycle_id, 'start_rsi', 0)
                end_rsi = self.get_safe_feature_value(feature_df, cycle_id, 'end_rsi', 0)
                start_hist = self.get_safe_feature_value(feature_df, cycle_id, 'start_hist', 0)
                end_hist = self.get_safe_feature_value(feature_df, cycle_id, 'end_hist', 0)
                start_macd = self.get_safe_feature_value(feature_df, cycle_id, 'start_macd', 0)
                end_macd = self.get_safe_feature_value(feature_df, cycle_id, 'end_macd', 0)
                start_macd_signal = self.get_safe_feature_value(feature_df, cycle_id, 'start_macd_signal', 0)
                end_macd_signal = self.get_safe_feature_value(feature_df, cycle_id, 'end_macd_signal', 0)

                # 미리 계산해 둔 Position, Children 정보 가져오기
                position_info_str = table_rows_data[idx-1]['position']
                child_info_str = table_rows_data[idx-1]['children']

                type_emoji = '🔴' if cycle_type == 'up' else '🔵'
                rsi_str = f"{start_rsi: >6.1f}→{end_rsi: <.1f}"
                macd_str = f"{start_macd: >6.1f}→{end_macd: <.1f}"
                macd_signal_str = f"{start_macd_signal: >6.1f}→{end_macd_signal: <.1f}"
                hist_str = f"{start_hist: >6.1f}→{end_hist: <.1f}"

                # 4. 동적으로 계산된 너비로 데이터 행 출력
                print(f"{idx:<3} {cycle_id:<27} {type_emoji}{cycle_type:<4} {duration:>3} {price_change:>+9.2f}%   "
                      f"{position_info_str:<{pos_width}} {child_info_str:<{child_width}} "
                      f"{peak_pos:>6.2f} {trough_pos:>7.2f} "
                      f"{rsi_str:^15} {macd_str:^15} {macd_signal_str:^15} {hist_str:^15}")

            self.print_cycle_statistics(feature_df, tf)

    def print_cycle_statistics(self, feature_df, timeframe):
        """개선된 사이클별 통계 출력 - 가격변화율 +/- 비율 추가"""
        if feature_df.empty:
            return
        
        # 상승/하락 사이클 분리
        up_cycles = feature_df[feature_df['cycle_type'] == 'up']
        down_cycles = feature_df[feature_df['cycle_type'] == 'down']
        
        print(f"\n📊 {timeframe.upper()} 사이클 통계:")
        print("─" * 80)
        
        # 전체 통계
        total_count = len(feature_df)
        up_count = len(up_cycles)
        down_count = len(down_cycles)
        
        print(f"📈 전체: {total_count}개 (상승: {up_count}개, 하락: {down_count}개)")
        
        # 전체 가격변화율 +/- 분포 추가
        if 'change_price_pct' in feature_df.columns:
            price_changes = feature_df['change_price_pct']
            positive_count = (price_changes > 0).sum()
            negative_count = (price_changes < 0).sum()
            zero_count = (price_changes == 0).sum()
            
            positive_ratio = positive_count / total_count * 100
            negative_ratio = negative_count / total_count * 100
            zero_ratio = zero_count / total_count * 100
            
            print(f"💰 가격변화 분포: 상승 {positive_count}개({positive_ratio:.1f}%), "
                  f"하락 {negative_count}개({negative_ratio:.1f}%), "
                  f"보합 {zero_count}개({zero_ratio:.1f}%)")
        
        # 상승 사이클 통계
        if not up_cycles.empty:
            print(f"\n🔴 상승 사이클 ({up_count}개):")
            self._print_cycle_type_stats(up_cycles, "상승")
        
        # 하락 사이클 통계
        if not down_cycles.empty:
            print(f"\n🔵 하락 사이클 ({down_count}개):")
            self._print_cycle_type_stats(down_cycles, "하락")
    
    def _print_cycle_type_stats(self, cycles_df, cycle_type_name):
        """특정 타입의 사이클 통계 출력 - 가격 상승/하락별 분리"""
        # 사용 가능한 컬럼 확인
        available_cols = cycles_df.columns.tolist()
    
        # change_price_pct로 가격 상승/하락 분리
        if 'change_price_pct' not in available_cols:
            print("   ⚠️ 가격 변화 데이터가 없습니다.")
            return
        
        price_changes = cycles_df['change_price_pct']
        price_positive_cycles = cycles_df[price_changes > 0]
        price_negative_cycles = cycles_df[price_changes < 0]
        
        positive_count = len(price_positive_cycles)
        negative_count = len(price_negative_cycles)
        total_count = len(cycles_df)
        
        print(f"   💰 가격 실제 변화: 상승 {positive_count}개 ({positive_count/total_count*100:.1f}%), "
              f"하락 {negative_count}개 ({negative_count/total_count*100:.1f}%)")
        
        # 가격 상승한 사이클들의 통계
        if not price_positive_cycles.empty:
            print(f"\n   📈 가격 상승한 사이클 ({positive_count}개):")
            self._print_detailed_stats(price_positive_cycles, "가격 상승", available_cols)
        
        # 가격 하락한 사이클들의 통계
        if not price_negative_cycles.empty:
            print(f"\n   📉 가격 하락한 사이클 ({negative_count}개):")
            self._print_detailed_stats(price_negative_cycles, "가격 하락", available_cols)
    
    def _print_detailed_stats(self, cycles_df, label, available_cols):
        """상세 통계 출력 헬퍼 함수"""
        stats = {}
        
        # duration_candles 처리
        if 'duration_candles' in available_cols:
            stats['duration_mean'] = cycles_df['duration_candles'].mean()
            stats['duration_median'] = cycles_df['duration_candles'].median()
        elif 'shape_duration_candles' in available_cols:
            stats['duration_mean'] = cycles_df['shape_duration_candles'].mean()
            stats['duration_median'] = cycles_df['shape_duration_candles'].median()
        else:
            stats['duration_mean'] = stats['duration_median'] = 0

        # 가격 변화
        if 'change_price_pct' in available_cols:
            price_changes = cycles_df['change_price_pct']
            stats['price_change_mean'] = price_changes.mean()
            stats['price_change_median'] = price_changes.median()
            price_min = price_changes.min()
            price_max = price_changes.max()
        else:
            stats['price_change_mean'] = stats['price_change_median'] = 0
            price_min = price_max = 0

        # Peak/Trough Position
        if 'shape_peak_price_position' in available_cols:
            stats['peak_pos_mean'] = cycles_df['shape_peak_price_position'].mean()
            stats['peak_pos_median'] = cycles_df['shape_peak_price_position'].median()
        else:
            stats['peak_pos_mean'] = stats['peak_pos_median'] = 0

        if 'shape_trough_price_position' in available_cols:
            stats['trough_pos_mean'] = cycles_df['shape_trough_price_position'].mean()
            stats['trough_pos_median'] = cycles_df['shape_trough_price_position'].median()
        else:
            stats['trough_pos_mean'] = stats['trough_pos_median'] = 0

        # RSI
        if 'start_rsi' in available_cols and 'end_rsi' in available_cols:
            stats['start_rsi_mean'] = cycles_df['start_rsi'].mean()
            stats['start_rsi_median'] = cycles_df['start_rsi'].median()
            stats['end_rsi_mean'] = cycles_df['end_rsi'].mean()
            stats['end_rsi_median'] = cycles_df['end_rsi'].median()
            if 'change_rsi' in available_cols:
                stats['rsi_change_mean'] = cycles_df['change_rsi'].mean()
                stats['rsi_change_median'] = cycles_df['change_rsi'].median()
            else:
                stats['rsi_change_mean'] = stats['rsi_change_median'] = 0
        else:
            stats['start_rsi_mean'] = stats['start_rsi_median'] = 0
            stats['end_rsi_mean'] = stats['end_rsi_median'] = 0
            stats['rsi_change_mean'] = stats['rsi_change_median'] = 0

        # MACD
        if 'start_macd' in available_cols and 'end_macd' in available_cols:
            stats['start_macd_mean'] = cycles_df['start_macd'].mean()
            stats['start_macd_median'] = cycles_df['start_macd'].median()
            stats['end_macd_mean'] = cycles_df['end_macd'].mean()
            stats['end_macd_median'] = cycles_df['end_macd'].median()
            if 'change_macd' in available_cols:
                stats['macd_change_mean'] = cycles_df['change_macd'].mean()
                stats['macd_change_median'] = cycles_df['change_macd'].median()
            else:
                stats['macd_change_mean'] = stats['macd_change_median'] = 0
        else:
            stats['start_macd_mean'] = stats['start_macd_median'] = 0
            stats['end_macd_mean'] = stats['end_macd_median'] = 0
            stats['macd_change_mean'] = stats['macd_change_median'] = 0

        # Signal
        if 'start_macd_signal' in available_cols and 'end_macd_signal' in available_cols:
            stats['start_signal_mean'] = cycles_df['start_macd_signal'].mean()
            stats['start_signal_median'] = cycles_df['start_macd_signal'].median()
            stats['end_signal_mean'] = cycles_df['end_macd_signal'].mean()
            stats['end_signal_median'] = cycles_df['end_macd_signal'].median()
            if 'change_macd_signal' in available_cols:
                stats['signal_change_mean'] = cycles_df['change_macd_signal'].mean()
                stats['signal_change_median'] = cycles_df['change_macd_signal'].median()
            else:
                stats['signal_change_mean'] = stats['signal_change_median'] = 0
        else:
            stats['start_signal_mean'] = stats['start_signal_median'] = 0
            stats['end_signal_mean'] = stats['end_signal_median'] = 0
            stats['signal_change_mean'] = stats['signal_change_median'] = 0

        # Hist
        if 'start_hist' in available_cols and 'end_hist' in available_cols:
            stats['start_hist_mean'] = cycles_df['start_hist'].mean()
            stats['start_hist_median'] = cycles_df['start_hist'].median()
            stats['end_hist_mean'] = cycles_df['end_hist'].mean()
            stats['end_hist_median'] = cycles_df['end_hist'].median()
            if 'change_hist' in available_cols:
                stats['hist_change_mean'] = cycles_df['change_hist'].mean()
                stats['hist_change_median'] = cycles_df['change_hist'].median()
            else:
                stats['hist_change_mean'] = stats['hist_change_median'] = 0
        else:
            stats['start_hist_mean'] = stats['start_hist_median'] = 0
            stats['end_hist_mean'] = stats['end_hist_median'] = 0
            stats['hist_change_mean'] = stats['hist_change_median'] = 0
        
        # Strength 특징들 추가
        if 'strength_direction_pct' in available_cols:
            stats['direction_pct_mean'] = cycles_df['strength_direction_pct'].mean()
            stats['direction_pct_median'] = cycles_df['strength_direction_pct'].median()
        else:
            stats['direction_pct_mean'] = stats['direction_pct_median'] = 0

        if 'strength_hist_positive_ratio' in available_cols:
            stats['hist_positive_ratio_mean'] = cycles_df['strength_hist_positive_ratio'].mean()
            stats['hist_positive_ratio_median'] = cycles_df['strength_hist_positive_ratio'].median()
        else:
            stats['hist_positive_ratio_mean'] = stats['hist_positive_ratio_median'] = 0

        if 'strength_price_up_ratio' in available_cols:
            stats['price_up_ratio_mean'] = cycles_df['strength_price_up_ratio'].mean()
            stats['price_up_ratio_median'] = cycles_df['strength_price_up_ratio'].median()
        else:
            stats['price_up_ratio_mean'] = stats['price_up_ratio_median'] = 0

        if 'strength_price_down_ratio' in available_cols:
            stats['price_down_ratio_mean'] = cycles_df['strength_price_down_ratio'].mean()
            stats['price_down_ratio_median'] = cycles_df['strength_price_down_ratio'].median()
        else:
            stats['price_down_ratio_mean'] = stats['price_down_ratio_median'] = 0
        
        # core count
        if 'shape_noise_count' in available_cols:
            stats['noise_count_mean']=cycles_df['shape_noise_count'].mean()
            stats['noise_count_median']=cycles_df['shape_noise_count'].median()
        else:
            stats['noise_count_mean'] = stats['noise_count_median'] = 0

        # 출력
        print(f"      가격 변화: 평균 {stats['price_change_mean']:+.2f}%, 중앙값 {stats['price_change_median']:+.2f}% "
              f"(범위: {price_min:+.2f}% ~ {price_max:+.2f}%)")
        print(f"      지속시간: 평균 {stats['duration_mean']:.1f} 캔들, 중앙값 {stats['duration_median']:.1f} 캔들")
        print(f"      Peak 위치: 평균 {stats['peak_pos_mean']:.2f}, 중앙값 {stats['peak_pos_median']:.2f}")
        print(f"      Trough 위치: 평균 {stats['trough_pos_mean']:.2f}, 중앙값 {stats['trough_pos_median']:.2f}")
        print(f"      RSI: {stats['start_rsi_mean']:.1f} (중앙값 {stats['start_rsi_median']:.1f}) → "
              f"{stats['end_rsi_mean']:.1f} (중앙값 {stats['end_rsi_median']:.1f}) "
              f"(변화: 평균 {stats['rsi_change_mean']:+.1f}, 중앙값 {stats['rsi_change_median']:+.1f})")
        print(f"      MACD: {stats['start_macd_mean']:.1f} (중앙값 {stats['start_macd_median']:.1f}) → "
              f"{stats['end_macd_mean']:.1f} (중앙값 {stats['end_macd_median']:.1f}) "
              f"(변화: 평균 {stats['macd_change_mean']:+.1f}, 중앙값 {stats['macd_change_median']:+.1f})")
        print(f"      Signal: {stats['start_signal_mean']:.1f} (중앙값 {stats['start_signal_median']:.1f}) → "
              f"{stats['end_signal_mean']:.1f} (중앙값 {stats['end_signal_median']:.1f}) "
              f"(변화: 평균 {stats['signal_change_mean']:+.1f}, 중앙값 {stats['signal_change_median']:+.1f})")
        print(f"      Hist: {stats['start_hist_mean']:.1f} (중앙값 {stats['start_hist_median']:.1f}) → "
              f"{stats['end_hist_mean']:.1f} (중앙값 {stats['end_hist_median']:.1f}) "
              f"(변화: 평균 {stats['hist_change_mean']:+.1f}, 중앙값 {stats['hist_change_median']:+.1f})")
        print(f"      방향성: 평균 {stats['direction_pct_mean']:.1f}%, 중앙값 {stats['direction_pct_median']:.1f}%")
        print(f"      히스토그램 양수 비율: 평균 {stats['hist_positive_ratio_mean']:.1f}%, 중앙값 {stats['hist_positive_ratio_median']:.1f}%")
        print(f"      가격 상승 비율: 평균 {stats['price_up_ratio_mean']:.1f}%, 중앙값 {stats['price_up_ratio_median']:.1f}%")
        print(f"      가격 하락 비율: 평균 {stats['price_down_ratio_mean']:.1f}%, 중앙값 {stats['price_down_ratio_median']:.1f}%")
        print(f"      노이즈 개수: 평균 {stats['noise_count_mean']:.1f}개, 중앙값 {stats['noise_count_median']:.1f}개")
    
    def analyze_single_cycle(self, timeframe, cycle_id):
        """개별 사이클 상세 분석"""
        # 선택된 사이클 로드
        df = self.load_timeframe_data(timeframe)
        cycle_df = df[df['cycle_id'] == cycle_id]
        
        if cycle_df.empty:
            print(f"❌ {cycle_id} 사이클을 찾을 수 없습니다.")
            return
        
        print(f"\n{'='*100}")
        print(f"🔍 개별 사이클 상세 분석: {cycle_id}")
        print('='*100)
        
        # Parent 사이클 정보
        parent_cycles = self.get_parent_cycles(timeframe, cycle_id)
        if parent_cycles:
            print(f"\n📈 상위 사이클 정보:")
            for parent_tf, parent_df in parent_cycles.items():
                # cycle_position 계산
                parent_timeframe_df = self.load_timeframe_data(parent_tf)
                if parent_timeframe_df is not None:
                    position = self.get_cycle_position(parent_timeframe_df, parent_df.iloc[0]['cycle_id'])
                    if position:
                        print(f"  📍 {parent_tf}에서의 위치: {position}")
            
            # 상위 사이클들을 테이블로 출력
            self.print_cycle_summary_table(parent_cycles, "상위 사이클 요약")
        else:
            print(f"\n📈 상위 사이클: 없음")
        
        # 선택된 사이클 정보 (상세)
        print(f"\n🎯 분석 대상 사이클 ({timeframe}):")
        self.print_cycle_details(cycle_df, f"선택된 사이클 ({timeframe})", show_stats=False)
        
        # Child 사이클 정보
        child_cycles = self.get_child_cycles(timeframe, cycle_id)
        if child_cycles:
            print(f"\n📉 하위 사이클 정보:")
            # 하위 사이클들을 테이블로 출력
            self.print_cycle_summary_table(child_cycles, "하위 사이클 요약")
            
            # 개별 하위 사이클 상세 정보 (선택사항)
            while True:
                choice = input(f"\n하위 사이클 상세 정보를 보시겠습니까? (y/n): ").strip().lower()
                if choice in ['y', 'yes', '예']:
                    print(f"\n하위 사이클 상세 정보:")
                    for child_tf, child_df in child_cycles.items():
                        if not child_df.empty:
                            print(f"\n📊 {child_tf} 사이클 상세 ({len(child_df)}개):")
                            self.print_cycle_details(child_df, f"Child ({child_tf})", show_stats=False)
                    break
                elif choice in ['n', 'no', '아니오']:
                    break
                else:
                    print("y 또는 n을 입력하세요.")
        else:
            print(f"\n📉 하위 사이클: 없음")
    
    def _verify_cycle_relationships(self, base_timeframe, main_cycles, parent_cycles, child_cycles):
        """
        필터링된 사이클들 간의 실제 관계를 검증하고 출력
        """
        print(f"📋 관계 검증 결과:")
        
        # 샘플 사이클 몇 개를 선택해서 관계 확인
        sample_count = min(3, len(main_cycles))
        sample_cycles = main_cycles.head(sample_count)
        
        for idx, (_, row) in enumerate(sample_cycles.iterrows(), 1):
            cycle_id = row['cycle_id']
            cycle_type = row['cycle_type']
            
            print(f"\n{idx}. {cycle_id} ({cycle_type})")
            
            # Parent 관계 확인
            actual_parents = self.get_parent_cycles(base_timeframe, cycle_id)
            for parent_tf, parent_df in actual_parents.items():
                if parent_cycles and parent_tf in parent_cycles:
                    parent_id = parent_df['cycle_id'].iloc[0]
                    parent_type = parent_df['cycle_type'].iloc[0]
                    
                    # 필터링된 parent 목록에 있는지 확인
                    is_in_filtered = parent_id in parent_cycles[parent_tf]['cycle_id'].values
                    status = "✅ 포함됨" if is_in_filtered else "❌ 누락됨"
                    
                    print(f"   📈 Parent {parent_tf}: {parent_id} ({parent_type}) - {status}")
            
            # Child 관계 확인
            actual_children = self.get_child_cycles(base_timeframe, cycle_id)
            for child_tf, child_df in actual_children.items():
                if child_cycles and child_tf in child_cycles:
                    child_count = len(child_df)
                    filtered_child_ids = set(child_cycles[child_tf]['cycle_id'].values)
                    actual_child_ids = set(child_df['cycle_id'].values)
                    
                    overlap_count = len(actual_child_ids.intersection(filtered_child_ids))
                    
                    print(f"   📉 Child {child_tf}: {overlap_count}/{child_count} 사이클 포함됨")
    
    def print_cycles_statistics_only(self, cycles_dict, title="사이클 통계"):
        """사이클들의 통계만 출력 (테이블 없이)"""
        if not cycles_dict:
            print(f"\n{title}: 없음")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 {title}")
        print('='*80)
        
        for tf, df in cycles_dict.items():
            if df.empty:
                continue
            
            feature_df = self.extract_all_cycle_features(df)
            
            # 시간대 헤더 출력
            print(f"\n📈 {tf.upper()} 사이클 ({len(df)}개)")
            print("─" * 80)
            
            # 통계 출력
            self.print_cycle_statistics(feature_df, tf)
    
    def visualize_cycle_features(self, df, timeframe, label="사이클"):
        """개선된 특징 분포 시각화 - 데이터 타입별 적절한 차트"""
        if df is None or df.empty:
            print(f"\n{label} 데이터가 없어 시각화를 건너뜁니다.")
            return
        
        print(f"\n📊 {label} 특징 분포 시각화 생성 중...")
        
        # 특징 데이터 추출
        feature_df = self.extract_all_cycle_features(df)
        
        # ImprovedFeatureVisualizer 인스턴스 생성
        visualizer = ImprovedFeatureVisualizer()
        
        # 저장 경로 설정
        output_dir = self.base_path / "feature_analysis" / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"improved_features_{timeframe}_{timestamp}.png"
        filepath = output_dir / filename
        
        # 시각화 실행
        visualizer.visualize_all_features(
            feature_df, 
            timeframe, 
            output_path=filepath,
            title_prefix=label
        )
    
    def visualize_feature_comparison(self, main_df, parent_dict, child_dict, timeframe):
        """메인-Parent-Child 사이클 특징 비교 시각화"""
        print(f"\n📊 계층별 사이클 특징 비교 시각화 생성 중...")
        
        # 비교할 주요 특징들
        key_features = [
            ('change_price_pct', '가격 변화율 (%)'),
            ('duration_candles', '지속시간 (캔들)'),
            ('start_rsi', 'RSI (시작)'),
            ('end_rsi', 'RSI (종료)'),
            ('strength_direction_pct', '방향성 비율 (%)'),
            ('shape_peak_price_position', 'Peak 위치'),
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'계층별 사이클 특징 비교 - {timeframe.upper()}', 
                     fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, (col, title) in enumerate(key_features):
            ax = axes[idx]
            
            data_dict = {}
            
            # 메인 사이클 데이터
            main_feature_df = self.extract_all_cycle_features(main_df)
            if col in main_feature_df.columns:
                main_data = main_feature_df[col].dropna()
                if len(main_data) > 0:
                    data_dict[f'메인\n{timeframe}\n(n={len(main_data)})'] = main_data
            
            # Parent 사이클 데이터
            if parent_dict:
                for tf, df in parent_dict.items():
                    feature_df = self.extract_all_cycle_features(df)
                    if col in feature_df.columns:
                        data = feature_df[col].dropna()
                        if len(data) > 0:
                            data_dict[f'Parent\n{tf}\n(n={len(data)})'] = data
            
            # Child 사이클 데이터
            if child_dict:
                for tf, df in child_dict.items():
                    feature_df = self.extract_all_cycle_features(df)
                    if col in feature_df.columns:
                        data = feature_df[col].dropna()
                        if len(data) > 0:
                            data_dict[f'Child\n{tf}\n(n={len(data)})'] = data
            
            if not data_dict:
                ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center')
                ax.set_title(title)
                continue
            
            # 박스플롯
            positions = range(len(data_dict))
            data_list = list(data_dict.values())
            labels = list(data_dict.keys())
            
            bp = ax.boxplot(data_list, positions=positions,
                           patch_artist=True, widths=0.6,
                           medianprops=dict(color='red', linewidth=2))
            
            # 색상 설정 (메인=파랑, Parent=초록, Child=주황)
            colors = []
            for label in labels:
                if '메인' in label:
                    colors.append('#3498db')
                elif 'Parent' in label:
                    colors.append('#2ecc71')
                else:
                    colors.append('#e74c3c')
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 저장
        output_dir = self.base_path / "feature_analysis" / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hierarchy_comparison_{timeframe}_{timestamp}.png"
        filepath = output_dir / filename
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✅ 계층 비교 시각화 저장: {filepath}")
        
        plt.show()
        plt.close()
    
    def run_interactive_analysis(self):
        """개선된 대화형 분석 실행"""
        print("\n🚀 개선된 계층적 사이클 분석기 (정확한 필터링)")
        print("="*80)
        
        # 1. 시간대 선택
        print("\n사용 가능한 시간대:")
        for i, tf in enumerate(self.available_timeframes, 1):
            print(f"  {i}. {tf}")
        
        while True:
            try:
                choice = input(f"\n분석할 시간대 선택 (1-{len(self.available_timeframes)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(self.available_timeframes):
                    selected_timeframe = self.available_timeframes[idx]
                    print(f"✅ 선택된 시간대: {selected_timeframe}")
                    break
                else:
                    print("❌ 올바른 번호를 입력하세요.")
            except ValueError:
                print("❌ 숫자를 입력하세요.")
        
        # 2. 필터 입력
        print("\n🔍 필터 설정 (Enter=전체)")
        print("예시: up, change_price_pct>0, 500>start_macd>300")
        
        # 2-1) 기준 시간대 필터
        main_filter_str = input(f"{selected_timeframe} (기준) 필터: ").strip()
        main_filters = self.parse_filter_expression(main_filter_str) if main_filter_str else None
        
        # 2-2) Parent 시간대별 필터
        parent_filters_map = {}
        parent_tfs = [tf for tf in (self.timeframe_hierarchy.get(selected_timeframe, {}).get('parent') or []) 
                     if tf in self.available_timeframes]
        if parent_tfs:
            print(f"\n📈 Parent 시간대별 필터 (Enter=조건없음):")
            print("⚠️  주의: Parent 필터는 기준 사이클과의 관계를 고려하여 적용됩니다.")
            for tf in parent_tfs:
                s = input(f"  - {tf}: ").strip()
                if s:
                    parent_filters_map[tf] = self.parse_filter_expression(s)
        
        # 2-3) Child 시간대별 필터
        child_filters_map = {}
        child_tfs = [tf for tf in (self.timeframe_hierarchy.get(selected_timeframe, {}).get('children') or []) 
                    if tf in self.available_timeframes]
        if child_tfs:
            print(f"\n📉 Child 시간대별 필터 (Enter=조건없음):")
            print("⚠️  주의: Child 필터는 기준 사이클과의 관계를 고려하여 적용됩니다.")
            for tf in child_tfs:
                s = input(f"  - {tf}: ").strip()
                if s:
                    child_filters_map[tf] = self.parse_filter_expression(s)
        
        # 3. 개선된 필터링 및 분석
        print(f"\n🔄 계층 관계를 고려한 필터링 실행 중...")
        
        main_cycles, parent_cycles, child_cycles = self.analyze_filtered_cycles_with_hierarchy(
            selected_timeframe,
            filters=main_filters,
            parent_filters_map=parent_filters_map if parent_filters_map else None,
            child_filters_map=child_filters_map if child_filters_map else None
        )
        
        if main_cycles is None:
            print("❌ 조건을 만족하는 사이클이 없습니다.")
            return
        
        # 4. 통계 출력
        print("\n" + "="*80)
        print("📊 필터링된 사이클 통계")
        print("="*80)
        
        # Parent 통계 (테이블 없이 통계만)
        if parent_cycles:
            self.print_cycles_statistics_only(parent_cycles, "Parent 사이클 통계")
        
        # Child 통계 (테이블 없이 통계만)
        if child_cycles:
            self.print_cycles_statistics_only(child_cycles, "Child 사이클 통계")
        
        # 메인 사이클 상세 테이블 + 통계
        main_dict = {selected_timeframe: main_cycles}
        self.print_cycle_summary_table(main_dict, f"메인 ({selected_timeframe}) 사이클")
        
        # 5. 관계 검증
        print(f"\n🔍 관계 검증")
        print("="*80)
        self._verify_cycle_relationships(selected_timeframe, main_cycles, parent_cycles, child_cycles)
        
        # 6. 메인 사이클 특징 분포 시각화
        print("\n" + "="*80)
        print("📊 특징 분포 시각화")
        print("="*80)
        
        viz_choice = input(f"\n메인 ({selected_timeframe}) 사이클 특징 분포를 시각화하시겠습니까? (y/n): ").strip().lower()
        if viz_choice in ['y', 'yes', '예']:
            self.visualize_cycle_features(main_cycles, selected_timeframe, 
                                         f"메인 {selected_timeframe} 사이클")
        
        # 7. Parent 사이클 시각화 옵션
        if parent_cycles:
            print("\n📈 Parent 사이클 시각화 옵션:")
            for tf in parent_cycles.keys():
                viz_choice = input(f"  - Parent {tf} 사이클을 시각화하시겠습니까? (y/n): ").strip().lower()
                if viz_choice in ['y', 'yes', '예']:
                    self.visualize_cycle_features(parent_cycles[tf], tf, f"Parent {tf} 사이클")
        
        # 8. Child 사이클 시각화 옵션
        if child_cycles:
            print("\n📉 Child 사이클 시각화 옵션:")
            for tf in child_cycles.keys():
                viz_choice = input(f"  - Child {tf} 사이클을 시각화하시겠습니까? (y/n): ").strip().lower()
                if viz_choice in ['y', 'yes', '예']:
                    self.visualize_cycle_features(child_cycles[tf], tf, f"Child {tf} 사이클")
        
        # 9. 계층별 비교 시각화
        if parent_cycles or child_cycles:
            viz_choice = input(f"\n메인-Parent-Child 계층별 특징 비교 시각화를 생성하시겠습니까? (y/n): ").strip().lower()
            if viz_choice in ['y', 'yes', '예']:
                self.visualize_feature_comparison(main_cycles, parent_cycles, child_cycles, 
                                                 selected_timeframe)
        
        # 10. 개별 분석 선택
        print("\n" + "="*80)
        print("🔍 개별 사이클 상세 분석")
        print("="*80)
        
        while True:
            choice = input(f"\n개별 분석할 사이클 번호 (1-{len(main_cycles)}, q:종료): ").strip()
            
            if choice.lower() == 'q':
                print("분석을 종료합니다.")
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(main_cycles):
                    cycle_id = main_cycles.iloc[idx]['cycle_id']
                    self.analyze_single_cycle(selected_timeframe, cycle_id)
                else:
                    print("❌ 올바른 번호를 입력하세요.")
            except ValueError:
                print("❌ 숫자를 입력하세요.")


def main():
    """메인 실행 함수"""
    analyzer = ImprovedHierarchicalCycleAnalyzer()
    analyzer.run_interactive_analysis()

if __name__ == "__main__":
    main()