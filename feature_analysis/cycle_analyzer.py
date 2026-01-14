# feature_analysis/cycle_analyzer.py
"""
사이클 특징 분석기
MACD 사이클 예측과 실제 가격 변화의 관계를 분석하고 시각화
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
import ast
from typing import Dict, Tuple, Optional
from config import DATA_PATHS, AVAILABLE_TIMEFRAMES, ANALYSIS_CATEGORIES, PLOT_CONFIG, OUTPUT_CONFIG

# 한국어 폰트 설정
def setup_korean_font():
    """한국어 폰트 설정"""
    try:
        # Windows 환경에서 사용 가능한 한국어 폰트 찾기
        font_candidates = [
            'Malgun Gothic',  # Windows 기본 한글 폰트
            'NanumGothic',    # 나눔고딕
            'AppleGothic',    # Mac 한글 폰트
            'Noto Sans CJK KR', # 구글 Noto 폰트
            'DejaVu Sans'     # 폴백 폰트
        ]
        
        available_fonts = [font.name for font in fm.fontManager.ttflist]
        
        for font_name in font_candidates:
            if font_name in available_fonts:
                plt.rcParams['font.family'] = font_name
                print(f"✅ 한국어 폰트 설정: {font_name}")
                return
        
        # 폰트를 찾지 못한 경우 영어 라벨 사용
        print("⚠️ 한국어 폰트를 찾을 수 없습니다. 영어 라벨을 사용합니다.")
        return False
        
    except Exception as e:
        print(f"⚠️ 폰트 설정 오류: {str(e)}")
        return False

# 한국어 폰트 설정 실행
setup_korean_font()

# 음수 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

class CycleFeatureAnalyzer:
    """사이클 특징 분석 클래스"""
    
    def __init__(self):
        """분석기 초기화"""
        self.data = None
        self.timeframe = None
        self.categories_data = {}
        
        # 한국어 폰트 설정 및 언어 결정
        self.use_korean = setup_korean_font()
        if self.use_korean is False:
            self.use_korean = False
            print("📝 Using English labels due to font issue")
        else:
            self.use_korean = True
        
        # 출력 디렉토리 생성
        self.output_dir = DATA_PATHS['output']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 스타일 설정
        plt.style.use('default')
        sns.set_palette("husl")
    
    def get_category_name(self, category_key: str) -> str:
        """카테고리 이름 반환 (언어에 따라)"""
        category_info = ANALYSIS_CATEGORIES[category_key]
        if self.use_korean:
            return category_info['name_ko']
        else:
            return category_info['name_en']
    
    def get_category_description(self, category_key: str) -> str:
        """카테고리 설명 반환 (언어에 따라)"""
        category_info = ANALYSIS_CATEGORIES[category_key]
        if self.use_korean:
            return category_info['description_ko']
        else:
            return category_info['description_en']
    
    def load_data(self, timeframe: str) -> bool:
        """
        지정된 타임프레임의 사이클 데이터 로드
        
        Args:
            timeframe (str): 타임프레임 (1m, 1h, 4h, 1d, 1w)
            
        Returns:
            bool: 로드 성공 여부
        """
        if timeframe not in AVAILABLE_TIMEFRAMES:
            print(f"❌ 지원하지 않는 타임프레임: {timeframe}")
            print(f"📋 사용 가능한 타임프레임: {AVAILABLE_TIMEFRAMES}")
            return False
        
        file_path = DATA_PATHS['structured_data'] / f'cycles_{timeframe}.parquet'
        
        if not file_path.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return False
        
        try:
            print(f"📊 데이터 로딩 중: {file_path}")
            self.data = pd.read_parquet(file_path)
            self.timeframe = timeframe
            
            print(f"✅ 데이터 로드 완료")
            print(f"📈 총 사이클 수: {len(self.data):,}개")
            print(f"📅 타임프레임: {timeframe}")
            print(f"📋 데이터 컬럼: {list(self.data.columns)}")
            
            # 데이터 샘플 출력
            if len(self.data) > 0:
                print(f"📊 데이터 샘플:")
                sample = self.data.head(2)
                for idx, row in sample.iterrows():
                    print(f"   행 {idx}: cycle_type={row['cycle_type']}, duration={row['duration_candles']}")
                    print(f"           cycle_features 타입: {type(row['cycle_features'])}")
                    
                    # cycle_features의 일부 내용 확인
                    if isinstance(row['cycle_features'], dict):
                        features = row['cycle_features']
                        print(f"           price_change_pct: {features.get('price_change_pct', 'N/A')}")
                    else:
                        print(f"           cycle_features 문자열 길이: {len(str(row['cycle_features']))}")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {str(e)}")
            return False
    
    def extract_cycle_features(self) -> pd.DataFrame:
        """
        사이클 특징 데이터 추출 및 정리
        
        Returns:
            pd.DataFrame: 정리된 특징 데이터
        """
        if self.data is None:
            print("❌ 먼저 데이터를 로드해주세요.")
            return None
        
        print("🔄 사이클 특징 추출 중...")
        print(f"📊 원본 데이터 샘플:")
        print(f"   - 컬럼: {list(self.data.columns)}")
        print(f"   - cycle_type 샘플: {self.data['cycle_type'].value_counts().head()}")
        
        # cycle_features가 어떤 형태인지 먼저 확인
        first_row = self.data.iloc[0]
        print(f"   - cycle_features 타입: {type(first_row['cycle_features'])}")
        print(f"   - cycle_features 샘플: {str(first_row['cycle_features'])[:200]}...")
        
        # cycle_features 딕셔너리 파싱
        features_list = []
        parse_errors = 0
        
        for idx, row in self.data.iterrows():
            try:
                # cycle_features가 딕셔너리인지 문자열인지 확인
                if isinstance(row['cycle_features'], str):
                    features = ast.literal_eval(row['cycle_features'])
                elif isinstance(row['cycle_features'], dict):
                    features = row['cycle_features']
                else:
                    print(f"⚠️ 행 {idx}: 예상하지 못한 cycle_features 타입: {type(row['cycle_features'])}")
                    continue
                
                # 기본 정보와 특징 합치기
                cycle_info = {
                    'cycle_id': row['cycle_id'],
                    'timeframe': row['timeframe'],
                    'cycle_type': row['cycle_type'],
                    'duration_candles': row['duration_candles'],
                    'start_date': row['start_date'],
                    'end_date': row['end_date']
                }
                
                # 특징 정보 추가
                cycle_info.update(features)
                features_list.append(cycle_info)
                
                # 첫 몇개 행의 데이터 샘플 출력
                if idx < 3:
                    print(f"   - 행 {idx}: cycle_type={row['cycle_type']}, price_change_pct={features.get('price_change_pct', 'N/A')}")
                
            except Exception as e:
                parse_errors += 1
                if parse_errors <= 5:  # 처음 5개 오류만 출력
                    print(f"⚠️ 행 {idx} 파싱 오류: {str(e)}")
                continue
        
        df = pd.DataFrame(features_list)
        
        if len(df) > 0:
            print(f"✅ 특징 추출 완료: {len(df):,}개 사이클")
            print(f"📊 추출된 데이터 정보:")
            print(f"   - price_change_pct가 있는 행: {df['price_change_pct'].notna().sum()}개")
            if 'price_change_pct' in df.columns:
                print(f"   - price_change_pct 범위: {df['price_change_pct'].min():.3f} ~ {df['price_change_pct'].max():.3f}")
                print(f"   - price_change_pct > 0: {(df['price_change_pct'] > 0).sum()}개")
                print(f"   - price_change_pct < 0: {(df['price_change_pct'] < 0).sum()}개")
        else:
            print("❌ 특징 추출 실패: 추출된 데이터가 없습니다.")
        
        if parse_errors > 5:
            print(f"⚠️ 총 {parse_errors}개 행에서 파싱 오류 발생")
        
        return df
    
    def categorize_cycles(self, df: pd.DataFrame) -> Dict:
        """
        사이클을 4개 카테고리로 분류
        
        Args:
            df (pd.DataFrame): 추출된 특징 데이터
            
        Returns:
            Dict: 카테고리별 데이터
        """
        print("📊 사이클 카테고리 분류 중...")
        
        if len(df) == 0:
            print("❌ 분류할 데이터가 없습니다.")
            return {}
        
        print(f"🔍 분류 전 데이터 상태:")
        print(f"   - 총 사이클: {len(df)}개")
        print(f"   - cycle_type 분포: {df['cycle_type'].value_counts().to_dict()}")
        
        # price_change_pct 컬럼이 있는지 확인
        if 'price_change_pct' not in df.columns:
            print("❌ price_change_pct 컬럼이 없습니다!")
            print(f"📋 사용 가능한 컬럼: {list(df.columns)}")
            return {}
        
        categories_data = {}
        
        for category_key, category_info in ANALYSIS_CATEGORIES.items():
            condition = category_info['condition']
            
            # 조건에 맞는 데이터 필터링 - 디버깅 추가
            filtered_data = []
            for idx, row in df.iterrows():
                try:
                    cycle_type = row['cycle_type']
                    price_change = row.get('price_change_pct', 0)
                    
                    # 값이 유효한지 확인
                    if pd.isna(price_change):
                        continue
                        
                    if condition(cycle_type, price_change):
                        filtered_data.append(row)
                        
                    # 처음 몇 개의 조건 체크 결과 출력
                    if idx < 5:
                        result = condition(cycle_type, price_change)
                        print(f"   - 행 {idx}: {category_key} -> cycle_type={cycle_type}, price_change={price_change:.3f}, 조건={result}")
                        
                except Exception as e:
                    print(f"⚠️ 행 {idx} 분류 오류 ({category_key}): {str(e)}")
                    continue
            
            category_data = pd.DataFrame(filtered_data) if filtered_data else pd.DataFrame()
            categories_data[category_key] = category_data
            
            category_name = self.get_category_name(category_key)
            print(f"📈 {category_name}: {len(category_data):,}개 ({len(category_data)/len(df)*100:.1f}%)")
        
        self.categories_data = categories_data
        return categories_data
    
    def create_scatter_plot(self, save_plot: bool = True) -> None:
        """
        4개 카테고리의 산점도 생성
        
        Args:
            save_plot (bool): 플롯 저장 여부
        """
        if not self.categories_data:
            print("❌ 먼저 카테고리를 분류해주세요.")
            return
        
        print("🎨 산점도 생성 중...")
        
        # 플롯 설정
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
        
        # 카테고리별 데이터 플롯
        for category_key, category_data in self.categories_data.items():
            if len(category_data) == 0:
                continue
                
            category_info = ANALYSIS_CATEGORIES[category_key]
            
            # X축: 카테고리별 위치 (약간의 지터 추가)
            x_base = {'up': 0, 'down': 1}  # 'rising' -> 'up', 'falling' -> 'down'으로 수정
            x_positions = []
            y_values = []
            
            for _, row in category_data.iterrows():
                cycle_type = row['cycle_type']
                price_change = row.get('price_change_pct', 0)
                
                x_pos = x_base[cycle_type] + np.random.uniform(-0.1, 0.1)  # 지터 추가
                x_positions.append(x_pos)
                y_values.append(price_change)
            
            # 산점도 그리기
            category_name = self.get_category_name(category_key)
            scatter = ax.scatter(
                x_positions, 
                y_values,
                c=category_info['color'],
                marker=category_info['marker'],
                s=PLOT_CONFIG['markersize'],
                alpha=PLOT_CONFIG['alpha'],
                label=f"{category_name}\n({len(category_data):,}개)",  # 언어에 맞는 이름 사용
                edgecolors='black',
                linewidth=0.5
            )
        
        # 축 설정
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Rising Cycles', 'Falling Cycles'])
        ax.set_ylabel('Price Change Percentage (%)')
        ax.set_title(f'MACD Cycle Prediction vs Actual Price Change\nTimeframe: {self.timeframe.upper()}', 
                    fontsize=14, fontweight='bold')
        
        # 0% 라인 추가
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # 범례 설정 - 위치와 스타일 개선
        legend_title = "사이클 카테고리" if self.use_korean else "Cycle Categories"
        legend = ax.legend(
            bbox_to_anchor=(1.02, 1), 
            loc='upper left',
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=10,
            title=legend_title,
            title_fontsize=12
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # 그리드 설정
        ax.grid(True, alpha=0.3)
        
        # 통계 정보 텍스트 추가
        total_cycles = sum(len(data) for data in self.categories_data.values())
        stats_text = f"Total Cycles: {total_cycles:,}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 레이아웃 조정 - 범례가 잘리지 않도록
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # 범례를 위한 공간 확보
        
        # 파일 저장
        if save_plot:
            filename = f"cycle_prediction_scatter_{self.timeframe}.{OUTPUT_CONFIG['image_format']}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=OUTPUT_CONFIG['save_dpi'], 
                       bbox_inches=OUTPUT_CONFIG['bbox_inches'])
            print(f"💾 플롯 저장: {filepath}")
        
        plt.show()
    
    def print_category_statistics(self) -> None:
        """카테고리별 통계 출력"""
        if not self.categories_data:
            print("❌ 먼저 카테고리를 분류해주세요.")
            return
        
        print("\n" + "="*60)
        print(f"📊 {self.timeframe.upper()} 타임프레임 사이클 분석 결과")
        print("="*60)
        
        total_cycles = sum(len(data) for data in self.categories_data.values())
        
        for category_key, category_data in self.categories_data.items():
            category_name = self.get_category_name(category_key)
            category_desc = self.get_category_description(category_key)
            count = len(category_data)
            percentage = (count / total_cycles * 100) if total_cycles > 0 else 0
            
            print(f"\n🎯 {category_name}")
            print(f"   📝 {category_desc}")
            print(f"   📊 개수: {count:,}개 ({percentage:.1f}%)")
            
            if count > 0:
                price_changes = category_data['price_change_pct']
                print(f"   📈 평균 가격변화: {price_changes.mean():.2f}%")
                print(f"   📉 중간값 가격변화: {price_changes.median():.2f}%")
                print(f"   📊 표준편차: {price_changes.std():.2f}%")
        
        print(f"\n📈 총 사이클: {total_cycles:,}개")
        print("="*60)
    
    def analyze_timeframe(self, timeframe: str, save_plot: bool = True) -> bool:
        """
        특정 타임프레임의 전체 분석 수행
        
        Args:
            timeframe (str): 분석할 타임프레임
            save_plot (bool): 플롯 저장 여부
            
        Returns:
            bool: 분석 성공 여부
        """
        print(f"\n🚀 {timeframe.upper()} 타임프레임 분석 시작")
        print("-" * 40)
        
        # 1. 데이터 로드
        if not self.load_data(timeframe):
            return False
        
        # 2. 특징 추출
        df = self.extract_cycle_features()
        if df is None:
            return False
        
        # 3. 카테고리 분류
        self.categorize_cycles(df)
        
        # 4. 통계 출력
        self.print_category_statistics()
        
        # 5. 시각화
        self.create_scatter_plot(save_plot)
        
        print(f"\n✅ {timeframe.upper()} 타임프레임 분석 완료!")
        return True

# 사용 예제
if __name__ == "__main__":
    analyzer = CycleFeatureAnalyzer()
    
    # 1일 타임프레임 분석 예제
    analyzer.analyze_timeframe('1d')