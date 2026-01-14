import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import numpy as np
import shlex
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import gc
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

class AdvancedCycleAnalyzer:
    """
    고급 사이클 특징 분석기 (최적화된 버전)
    - 한글 폰트 지원
    - 고속 데이터 로딩
    - 계층적 시각화 시스템
    - 전문적 UI 메뉴
    - 새로운 중첩 데이터 구조 지원
    - 날짜 분석 및 가격변화율 시각화 추가
    - 개별 시점 기반 상위 타임프레임 매칭 (수정됨)
    """
    
    def __init__(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        
        # 한글 폰트 설정
        self._setup_korean_font()
        
        print("📄 데이터 로딩 및 전처리 중...")
        try:
            self._fast_load_and_prepare_data(file_path)
            print("✅ 데이터 준비 완료.")
            
            # 상위 타임프레임 데이터 경로 설정
            self._setup_higher_timeframe_paths(file_path)
            
        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {e}")
            raise
            
    def _setup_korean_font(self):
        """한글 폰트 설정"""
        try:
            # Windows의 경우
            font_candidates = [
                'Malgun Gothic',  # 맑은 고딕
                'NanumGothic',    # 나눔고딕
                'AppleGothic',    # Mac용
                'Noto Sans CJK KR',  # Linux용
                'DejaVu Sans'     # 기본 대체
            ]
            
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            selected_font = None
            
            for font in font_candidates:
                if font in available_fonts:
                    selected_font = font
                    break
            
            if selected_font:
                plt.rcParams['font.family'] = selected_font
                plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
                print(f"  🎨 한글 폰트 설정: {selected_font}")
            else:
                print("  ⚠️ 한글 폰트를 찾을 수 없습니다. 영문으로 표시됩니다.")
                
        except Exception as e:
            print(f"  ⚠️ 폰트 설정 오류: {e}")
    
    def _setup_higher_timeframe_paths(self, current_file_path: Path):
        """상위 타임프레임 데이터 파일 경로 설정"""
        self.current_timeframe = None
        self.higher_timeframe_paths = {}
        
        # 현재 타임프레임 추출
        file_name = current_file_path.name
        if 'cycles_1h' in file_name:
            self.current_timeframe = '1h'
            timeframe_hierarchy = ['4h', '1d', '1w']
        elif 'cycles_4h' in file_name:
            self.current_timeframe = '4h'
            timeframe_hierarchy = ['1d', '1w']
        elif 'cycles_1d' in file_name:
            self.current_timeframe = '1d'
            timeframe_hierarchy = ['1w']
        elif 'cycles_1w' in file_name:
            self.current_timeframe = '1w'
            timeframe_hierarchy = []
        else:
            print("   ⚠️ 타임프레임을 식별할 수 없습니다.")
            return
        
        # 상위 타임프레임 파일 경로 구성
        base_dir = current_file_path.parent
        
        for timeframe in timeframe_hierarchy:
            higher_file_path = base_dir / f"cycles_{timeframe}.parquet"
            if higher_file_path.exists():
                self.higher_timeframe_paths[timeframe] = higher_file_path
                print(f"  🔍 상위 타임프레임 발견: {timeframe} → {higher_file_path.name}")
        
        if self.higher_timeframe_paths:
            print(f"  ✅ {len(self.higher_timeframe_paths)}개 상위 타임프레임 사용 가능")
        else:
            print("  ℹ️ 상위 타임프레임 파일이 없습니다.")
        
    
    def _fast_load_and_prepare_data(self, file_path: Path):
        """고속 데이터 로딩 및 전처리 (캔들 데이터 지연 로딩)"""
        # 1. 파일 읽기 (캔들 데이터 제외)
        try:
            print("  📊 메타데이터 로딩...")
            df = pd.read_parquet(file_path, engine='pyarrow')
            print(f"     로딩 완료: {len(df):,}개 행, {len(df.columns)}개 컬럼")
        except Exception as e:
            print(f"  ❌ 파일 읽기 실패: {e}")
            raise
        
        # 2. 필수 컬럼 확인
        required_cols = ['cycle_id', 'cycle_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")
        
        # 3. cycle_features 고속 처리 (새로운 중첩 구조 지원)
        print("  🔧 features 처리...")
        if 'cycle_features' in df.columns:
            features_df = self._fast_nested_json_normalize(df['cycle_features'])
        else:
            features_df = pd.DataFrame(index=df.index)
        
        # 4. 날짜 컬럼 처리 (벡터화)
        print("  📅 날짜 변환...")
        date_columns = ['start_date', 'end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = self._fast_datetime_convert(df[col])
        
        # 5. 마스터 DataFrame 생성 (캔들 데이터 제외)
        cols_to_drop = ['cycle_features']
        
        self.master_df = pd.concat([
            df.drop(columns=cols_to_drop, errors='ignore'), 
            features_df
        ], axis=1)
        
        # 6. 캔들 데이터는 파일 경로와 인덱스만 저장 (지연 로딩)
        self.file_path = file_path
        self.candle_data_loaded = False
        self.candle_data_dict = {}
        
        if 'candle_data' in df.columns:
            # 캔들 데이터가 있는 사이클 ID만 저장
            self.available_candle_ids = set(df['cycle_id'].tolist())
            self.raw_candle_data = df[['cycle_id', 'candle_data']].set_index('cycle_id')['candle_data'].to_dict()
        else:
            self.available_candle_ids = set()
            self.raw_candle_data = {}
        
        # 7. 특징 분류 (고속화)
        self.numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = [col for col in ['cycle_type', 'category', 'algorithm_used'] 
                                   if col in self.master_df.columns]
        
        # 8. 메모리 최적화
        self._optimize_memory_fast()
        
        # 9. 인덱스 최적화
        if 'cycle_id' in self.master_df.columns:
            self.master_df.set_index('cycle_id', inplace=True)
        
        # 메모리 정리
        del df, features_df
        gc.collect()
        
        print(f"  🚀 최적화 완료: {len(self.master_df):,}개 사이클, {len(self.numeric_features)}개 특징")

    def _fast_nested_json_normalize(self, features_col: pd.Series) -> pd.DataFrame:
        """중첩된 JSON 구조를 평면화하여 처리"""
        try:
            if len(features_col) == 0:
                return pd.DataFrame()
            
            print("    📄 중첩 구조 평면화 중...")
            
            # 첫 번째 아이템으로 구조 파악
            first_item = features_col.iloc[0]
            
            if isinstance(first_item, dict):
                # 중첩된 딕셔너리를 평면화
                all_flattened = []
                for item in features_col:
                    if isinstance(item, dict):
                        flattened = self._flatten_nested_dict(item)
                        all_flattened.append(flattened)
                    else:
                        all_flattened.append({})
                
                result_df = pd.DataFrame(all_flattened)
                print(f"    ✅ 평면화 완료: {len(result_df.columns)}개 특징 추출")
                return result_df
                
            # numpy array 처리
            elif isinstance(first_item, np.ndarray) and len(first_item) > 0:
                if isinstance(first_item[0], dict):
                    all_flattened = []
                    for item in features_col:
                        if isinstance(item, np.ndarray) and len(item) > 0 and isinstance(item[0], dict):
                            flattened = self._flatten_nested_dict(item[0])
                            all_flattened.append(flattened)
                        else:
                            all_flattened.append({})
                    result_df = pd.DataFrame(all_flattened)
                    print(f"    ✅ numpy array 평면화 완료: {len(result_df.columns)}개 특징")
                    return result_df
            
            # 문자열 JSON 처리
            elif isinstance(first_item, str):
                all_flattened = []
                for item in features_col:
                    try:
                        parsed = json.loads(item) if item else {}
                        flattened = self._flatten_nested_dict(parsed)
                        all_flattened.append(flattened)
                    except:
                        all_flattened.append({})
                result_df = pd.DataFrame(all_flattened)
                print(f"    ✅ JSON 문자열 평면화 완료: {len(result_df.columns)}개 특징")
                return result_df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"    ⚠️ 중첩 구조 처리 오류: {e}")
            return pd.DataFrame()

    def _flatten_nested_dict(self, nested_dict: dict, parent_key: str = '', sep: str = '.') -> dict:
        """중첩된 딕셔너리를 평면화"""
        items = []
        
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(self._flatten_nested_dict(value, new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        
        return dict(items)
    
    def _fast_datetime_convert(self, date_col: pd.Series) -> pd.Series:
        """고속 날짜 변환"""
        try:
            # 숫자형인 경우 (Unix timestamp)
            if pd.api.types.is_numeric_dtype(date_col):
                return pd.to_datetime(date_col, unit='s')
            
            # 문자열 숫자인 경우
            try:
                numeric_values = pd.to_numeric(date_col, errors='coerce')
                if not numeric_values.isna().all():
                    return pd.to_datetime(numeric_values, unit='s')
            except:
                pass
            
            # 일반 날짜 문자열
            return pd.to_datetime(date_col, errors='coerce')
            
        except Exception:
            return pd.NaT
    
    def _optimize_memory_fast(self):
        """고속 메모리 최적화"""
        # 카테고리형 변환
        for col in self.categorical_features:
            if col in self.master_df.columns:
                self.master_df[col] = self.master_df[col].astype('category')
        
        # 숫자형 다운캐스팅 (벡터화)
        for col in self.numeric_features:
            if col in self.master_df.columns:
                col_data = self.master_df[col]
                if col_data.dtype == 'float64':
                    self.master_df[col] = pd.to_numeric(col_data, downcast='float')
                elif col_data.dtype == 'int64':
                    self.master_df[col] = pd.to_numeric(col_data, downcast='integer')

    def _load_candle_data_on_demand(self, cycle_ids: List[str]):
        """필요시 캔들 데이터 로딩"""
        if not cycle_ids:
            return
            
        new_ids = [cid for cid in cycle_ids if cid not in self.candle_data_dict and cid in self.available_candle_ids]
        
        if new_ids:
            print(f"  🕯️ {len(new_ids)}개 사이클의 캔들 데이터 로딩 중...")
            for cycle_id in new_ids:
                if cycle_id in self.raw_candle_data:
                    candle_data = self.raw_candle_data[cycle_id]
                    if isinstance(candle_data, np.ndarray):
                        self.candle_data_dict[cycle_id] = candle_data
                    elif isinstance(candle_data, list):
                        self.candle_data_dict[cycle_id] = np.array(candle_data)

    def _get_user_conditions(self):
        """개선된 사용자 조건 입력 (새로운 특징명 반영)"""
        print("\n" + "="*80)
        print("📊 고급 사이클 분석 - 조건 설정")
        print("="*80)
        
        # 특징 분류 (새로운 중첩 구조에 맞게 업데이트)
        feature_categories = {
            "🎯 시작/종료": [f for f in self.numeric_features if any(x in f for x in ['start.', 'end.'])],
            "📈 성과/변화": [f for f in self.numeric_features if any(x in f for x in ['change.', '.pct', 'ratio'])],
            "⏱️ 지속성": [f for f in self.numeric_features if any(x in f for x in ['duration', 'count', 'core', 'shape.'])],
            "🔧 기술지표": [f for f in self.numeric_features if any(x in f.lower() for x in ['macd', 'rsi', 'hist', 'volume'])],
            "🎢 변동성": [f for f in self.numeric_features if any(x in f for x in ['volatility.', 'max_', 'deviation'])],
            "💪 강도": [f for f in self.numeric_features if any(x in f for x in ['strength.', 'direction'])],
        }
        
        # 간단한 카테고리 표시
        print("💡 주요 특징 카테고리:")
        for category, features in feature_categories.items():
            if features:
                sample_features = features[:3]  # 처음 3개만 표시
                more_count = len(features) - 3
                display_text = ", ".join([f.split('.')[-1] for f in sample_features])  # 중첩 경로의 마지막 부분만
                if more_count > 0:
                    display_text += f" 외 {more_count}개"
                print(f"   {category}: {display_text}")
        
        print(f"\n📊 사용 가능한 모든 특징 ({len(self.numeric_features)}개):")
        # 4열로 간단히 표시 (중첩 경로는 축약)
        all_features = sorted(self.numeric_features)
        for i in range(0, len(all_features), 4):
            row = all_features[i:i+4]
            # 중첩 경로를 간단히 표시 (예: shape.duration_candles -> shape.duration)
            simplified_row = []
            for f in row:
                if '.' in f:
                    parts = f.split('.')
                    if len(parts) == 2:
                        simplified = f"{parts[0]}.{parts[1].split('_')[0]}"
                    else:
                        simplified = f
                else:
                    simplified = f
                simplified_row.append(f"{simplified:<18}")
            print("   " + "  ".join(simplified_row))
        
        print("\n" + "="*80)
        print("🚀 빠른 조건 입력 가이드:")
        print("   조건 형태: 특징명 연산자 값")
        print("   연산자: > < >= <= == !=")  
        print("   복합: shape.duration_candles < 10 && change.price_pct > 5")
        print("   명령어: 'done' (완료), 'help' (도움말), 'list' (전체 특징)")
        print("="*80)

        conditions = []
        while True:
            condition_str = input(f"\n[조건 {len(conditions)+1}] 입력하세요: ").strip()

            if condition_str.lower() == 'done':
                break
            elif condition_str.lower() == 'help':
                self._show_quick_help()
                continue
            elif condition_str.lower() == 'list':
                self._show_all_features()
                continue

            parsed = self._parse_condition_fast(condition_str)
            if parsed:
                conditions.extend(parsed)

        return conditions
    
    def _show_quick_help(self):
        """빠른 도움말 (새로운 특징명 예시 포함)"""
        print("\n" + "─" * 60)
        print("📚 빠른 사용법")
        print("─" * 60)
        print("기본: 특징명 연산자 값")
        print("예시: change.price_pct > 5.0")
        print("복합: shape.duration_candles < 10 && strength.direction_pct > 80")
        print("유용한 조건들:")
        print("  • 강한 상승: change.price_pct > 10 && shape.noise_count < 3")
        print("  • RSI 반전: start.rsi < 30 && end.rsi > 50") 
        print("  • 긴 추세: shape.duration_candles > 15 && strength.direction_pct > 80")
        print("  • 낮은 변동성: volatility.price_change_deviation < 2")
        print("─" * 60)
    
    def _show_all_features(self):
        """전체 특징 목록 표시"""
        print("\n" + "─" * 80)
        print("📋 전체 특징 목록")
        print("─" * 80)
        
        # 카테고리별로 분류해서 표시
        feature_categories = {
            "Shape (형태)": [f for f in self.numeric_features if f.startswith('shape.')],
            "Strength (강도)": [f for f in self.numeric_features if f.startswith('strength.')],
            "Start (시작점)": [f for f in self.numeric_features if f.startswith('start.')],
            "End (종료점)": [f for f in self.numeric_features if f.startswith('end.')],
            "Change (변화량)": [f for f in self.numeric_features if f.startswith('change.')],
            "Volatility (변동성)": [f for f in self.numeric_features if f.startswith('volatility.')],
            "Aggregate (집계)": [f for f in self.numeric_features if f.startswith('aggregate.')],
            "기타": [f for f in self.numeric_features if '.' not in f]
        }
        
        for category, features in feature_categories.items():
            if features:
                print(f"\n{category}:")
                for i, feature in enumerate(features, 1):
                    print(f"  {i:2d}. {feature}")
        
        print("─" * 80)

    def _parse_condition_fast(self, condition_str: str) -> List[Dict[str, Any]]:
        """고속 조건 파싱"""
        conditions = []
        
        try:
            # 복합 조건 분리
            if '&&' in condition_str or ' and ' in condition_str.lower():
                sub_conditions = condition_str.replace('&&', '|').replace(' and ', '|').replace(' AND ', '|').split('|')
                for sub_cond in sub_conditions:
                    parsed = self._parse_single_condition_fast(sub_cond.strip())
                    if parsed:
                        conditions.append(parsed)
            else:
                parsed = self._parse_single_condition_fast(condition_str)
                if parsed:
                    conditions.append(parsed)
                    
        except Exception as e:
            print(f"❌ 조건 파싱 오류: {e}")
            
        return conditions
    
    def _parse_single_condition_fast(self, condition_str: str) -> Optional[Dict[str, Any]]:
        """단일 조건 고속 파싱"""
        try:
            parts = shlex.split(condition_str)
            if len(parts) != 3:
                print("❌ 형식: 특징명 연산자 값")
                return None

            feature, operator, value = parts
            
            # 특징 검증 (빠른 검색)
            if feature not in self.numeric_features:
                # 부분 매칭 시도 (더 관대한 매칭)
                matches = [f for f in self.numeric_features if feature.lower() in f.lower() or f.lower() in feature.lower()]
                if matches:
                    print(f"💡 '{feature}' 대신 이것들 중 하나인가요? {matches[:5]}")
                else:
                    print(f"❌ '{feature}' 특징을 찾을 수 없습니다.")
                    # 더 친화적인 제안
                    if '.' not in feature:
                        suggestions = [f for f in self.numeric_features if feature.lower() in f.split('.')[-1].lower()]
                        if suggestions:
                            print(f"   혹시 이런 특징을 찾으셨나요? {suggestions[:3]}")
                return None
            
            # 연산자 검증
            if operator not in ['>', '<', '>=', '<=', '==', '!=']:
                print(f"❌ 잘못된 연산자: {operator}")
                return None
            
            # 값 변환
            try:
                value = float(value)
            except ValueError:
                print(f"❌ 숫자가 아님: {value}")
                return None
            
            print(f"   ✅ {feature} {operator} {value}")
            return {'feature': feature, 'operator': operator, 'value': value}
            
        except Exception as e:
            print(f"❌ 파싱 오류: {e}")
            return None

    def _filter_cycles(self, conditions: List[Dict[str, Any]]) -> pd.DataFrame:
        """고속 필터링 - Series boolean 문제 수정"""
        if not conditions:
            return self.master_df.copy()
        
        print(f"\n🔍 {len(conditions)}개 조건으로 필터링...")
        
        # 벡터화된 마스크 생성
        mask = pd.Series(True, index=self.master_df.index, dtype=bool)
        
        for i, cond in enumerate(conditions, 1):
            feature, op, val = cond['feature'], cond['operator'], cond['value']
            
            if feature not in self.master_df.columns:
                continue
            
            feature_data = self.master_df[feature].fillna(0)
            prev_count = mask.sum()  # Series의 sum() 사용
            
            # 조건 적용 (벡터화) - 각 조건별로 boolean Series 생성
            if op == '>': 
                condition_mask = feature_data > val
            elif op == '<': 
                condition_mask = feature_data < val
            elif op == '>=': 
                condition_mask = feature_data >= val
            elif op == '<=': 
                condition_mask = feature_data <= val
            elif op == '==': 
                condition_mask = feature_data == val
            elif op == '!=': 
                condition_mask = feature_data != val
            else: 
                continue
            
            # boolean Series끼리 AND 연산
            mask = mask & condition_mask
            new_count = mask.sum()  # Series의 sum() 사용
            
            reduction_pct = (prev_count - new_count) / prev_count * 100 if prev_count > 0 else 0
            print(f"   {i}. {feature} {op} {val}: {prev_count:,} → {new_count:,} (-{reduction_pct:.1f}%)")
        
        return self.master_df[mask].copy()

    def _show_cycle_timeframe(self, filtered_df: pd.DataFrame):
        """간단한 사이클 시간대 정보 표시 및 날짜 목록 반환"""
        print(f"\n⏰ 사이클 시간대 정보:")
        
        if 'start_date' in filtered_df.columns:
            start_dates = pd.to_datetime(filtered_df['start_date'], errors='coerce').dropna()
            if len(start_dates) > 0:
                print(f"   첫 사이클: {start_dates.min().strftime('%Y-%m-%d %H:%M')}")
                print(f"   마지막: {start_dates.max().strftime('%Y-%m-%d %H:%M')}")
                print(f"   기간: {(start_dates.max() - start_dates.min()).days}일")
                
                # 날짜 목록 생성 및 출력
                filtered_dates = self._show_filtered_cycle_dates(filtered_df)
                return filtered_dates
            else:
                print("   날짜 정보 없음")
                return []
        else:
            print("   날짜 컬럼 없음")
            return []

    def _show_filtered_cycle_dates(self, filtered_df: pd.DataFrame):
        """필터링된 사이클들의 날짜를 순서대로 출력"""
        print(f"\n📅 필터링된 사이클 날짜 목록 ({len(filtered_df)}개):")
        print("─" * 60)
        if 'start_date' not in filtered_df.columns:
            print("   ⚠️ 시작 날짜 정보가 없습니다.")
            return []
        
        # 날짜 데이터 준비 및 정렬
        cycles_with_dates = filtered_df[['start_date', 'end_date']].copy()
        cycles_with_dates['start_date'] = pd.to_datetime(cycles_with_dates['start_date'], errors='coerce')
        cycles_with_dates['end_date'] = pd.to_datetime(cycles_with_dates['end_date'], errors='coerce')
        
        # 시작 날짜 기준으로 정렬
        cycles_with_dates = cycles_with_dates.sort_values('start_date')
        
        dates_list = []
        print("   순번  시작일시           종료일시           기간")
        print("   " + "─" * 54)
        
        for i, (cycle_id, row) in enumerate(cycles_with_dates.iterrows(), 1):
            start_date = row['start_date']
            end_date = row['end_date']
            
            if pd.notna(start_date) and pd.notna(end_date):
                duration = (end_date - start_date).total_seconds() / 3600  # 시간 단위
                
                start_str = start_date.strftime('%Y-%m-%d %H:%M')
                end_str = end_date.strftime('%Y-%m-%d %H:%M')
                
                print(f"   {i:3d}   {start_str}   {end_str}   {duration:5.1f}h")
                dates_list.append({
                    'cycle_id': cycle_id,
                    'start_date': start_date,
                    'end_date': end_date
                })
        
        if dates_list:
            min_date = min(d['start_date'] for d in dates_list)
            max_date = max(d['end_date'] for d in dates_list)
            total_span = (max_date - min_date).days
            
            print(f"\n📊 전체 날짜 범위:")
            print(f"   시작: {min_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"   종료: {max_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"   총 기간: {total_span}일")
            
            return dates_list
        else:
            print("   유효한 날짜 데이터가 없습니다.")
            return []

    def _load_higher_timeframe_data(self, timeframe: str) -> pd.DataFrame:
        """상위 타임프레임 데이터 로딩"""
        if timeframe not in self.higher_timeframe_paths:
            return pd.DataFrame()
        
        file_path = self.higher_timeframe_paths[timeframe]
        print(f"  📊 {timeframe} 타임프레임 데이터 로딩 중...")
        
        try:
            # 같은 방식으로 데이터 로딩
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            # features 처리
            if 'cycle_features' in df.columns:
                features_df = self._fast_nested_json_normalize(df['cycle_features'])
            else:
                features_df = pd.DataFrame(index=df.index)
            
            # 날짜 변환
            date_columns = ['start_date', 'end_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = self._fast_datetime_convert(df[col])
            
            # 통합 DataFrame 생성
            cols_to_drop = ['cycle_features', 'candle_data']  # 캔들 데이터는 제외
            higher_df = pd.concat([
                df.drop(columns=cols_to_drop, errors='ignore'), 
                features_df
            ], axis=1)
            
            print(f"     ✅ {timeframe} 로딩 완료: {len(higher_df):,}개 사이클")
            return higher_df
            
        except Exception as e:
            print(f"     ❌ {timeframe} 로딩 실패: {e}")
            return pd.DataFrame()

    def _find_cycle_containing_timestamp(self, higher_df: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        """특정 시점이 포함된 상위 타임프레임 사이클 찾기 (개별 시점 기준)"""
        if higher_df.empty:
            return pd.DataFrame()
        
        if 'start_date' not in higher_df.columns or 'end_date' not in higher_df.columns:
            return pd.DataFrame()
        
        # 날짜 변환
        higher_df['start_date'] = pd.to_datetime(higher_df['start_date'], errors='coerce')
        higher_df['end_date'] = pd.to_datetime(higher_df['end_date'], errors='coerce')
        
        # 특정 시점이 포함된 사이클 찾기: start_date <= timestamp <= end_date
        mask = (
            (higher_df['start_date'] <= timestamp) & 
            (higher_df['end_date'] >= timestamp)
        )
        
        matching_cycles = higher_df[mask].copy()
        
        return matching_cycles

    def _analyze_higher_timeframe_cycles(self, filtered_dates: List[Dict]):
        """상위 타임프레임에서 각 시점별로 개별 사이클 분석 (수정된 버전)"""
        if not filtered_dates:
            print("\n   ⚠️ 분석할 날짜 데이터가 없습니다.")
            return
        
        if not self.higher_timeframe_paths:
            print(f"\n   ℹ️ {self.current_timeframe}보다 상위 타임프레임이 없습니다.")
            return
        
        print(f"\n🔍 상위 타임프레임 개별 시점 분석 (필터링된 {len(filtered_dates)}개 사이클)")
        print("=" * 90)
        print(f"📊 기준 타임프레임: {self.current_timeframe}")
        print("🎯 분석 방식: 각 사이클의 시작 시점이 포함된 상위 사이클 개별 매칭")
        
        # 각 상위 타임프레임별 분석
        for timeframe in sorted(self.higher_timeframe_paths.keys()):
            print(f"\n📈 {timeframe} 타임프레임 분석:")
            print("─" * 60)
            
            # 상위 타임프레임 데이터 로딩
            higher_df = self._load_higher_timeframe_data(timeframe)
            
            if len(higher_df) == 0:
                print(f"     ❌ {timeframe} 데이터를 로딩할 수 없습니다.")
                continue
            
            # 각 필터링된 사이클의 시작 시점별로 개별 매칭
            matched_cycles = []
            matching_details = []
            
            for i, cycle_info in enumerate(filtered_dates, 1):
                cycle_id = cycle_info['cycle_id']
                start_timestamp = cycle_info['start_date']
                
                # 해당 시점이 포함된 상위 사이클 찾기
                containing_cycle = self._find_cycle_containing_timestamp(higher_df, start_timestamp)
                
                if len(containing_cycle) > 0:
                    # 첫 번째 매칭 사이클 선택 (보통 하나만 있어야 함)
                    matched_cycle = containing_cycle.iloc[0]
                    matched_cycles.append(matched_cycle)
                    
                    matching_details.append({
                        'lower_cycle_id': cycle_id,
                        'lower_start': start_timestamp,
                        'higher_cycle_id': matched_cycle.name,  # index가 cycle_id
                        'higher_start': matched_cycle['start_date'],
                        'higher_end': matched_cycle['end_date'],
                        'cycle_data': matched_cycle
                    })
                    
                    print(f"     ✅ {cycle_id} ({start_timestamp.strftime('%m-%d %H:%M')}) "
                          f"→ {matched_cycle.name} ({matched_cycle['start_date'].strftime('%m-%d %H:%M')} ~ "
                          f"{matched_cycle['end_date'].strftime('%m-%d %H:%M')})")
                else:
                    print(f"     ❌ {cycle_id} ({start_timestamp.strftime('%m-%d %H:%M')}) → 매칭되는 {timeframe} 사이클 없음")
            
            if matched_cycles:
                # 매칭된 사이클들로 DataFrame 생성
                matched_df = pd.DataFrame(matched_cycles)
                matched_df.index = [detail['higher_cycle_id'] for detail in matching_details]
                
                print(f"\n     📊 매칭 결과: {len(matched_df)}개 {timeframe} 사이클")
                
                # 통계 분석
                self._analyze_individual_higher_cycles(matched_df, timeframe, matching_details)
                
                # 시각화 옵션
                if input(f"     🎨 {timeframe} 매칭 사이클 시각화를 보시겠습니까? (y/n): ").lower() == 'y':
                    self._visualize_matched_higher_cycles(matched_df, timeframe, matching_details)
            else:
                print(f"     ❌ 매칭되는 {timeframe} 사이클이 없습니다.")

    def _analyze_individual_higher_cycles(self, matched_df: pd.DataFrame, timeframe: str, matching_details: List[Dict]):
        """매칭된 상위 사이클들의 개별 성과 분석"""
        print(f"\n    📈 {timeframe} 매칭 사이클 성과 분석:")
        
        # 기본 정보
        total_matched = len(matched_df)
        up_cycles = len(matched_df[matched_df['cycle_type'] == 'up']) if 'cycle_type' in matched_df.columns else 0
        down_cycles = total_matched - up_cycles
        
        print(f"       총 매칭 사이클: {total_matched}개")
        print(f"       상승 사이클: {up_cycles}개 ({up_cycles/total_matched*100:.1f}%)")
        print(f"       하락 사이클: {down_cycles}개 ({down_cycles/total_matched*100:.1f}%)")
        
        # 가격변화율 분석
        price_change_col = 'change.price_pct'
        if price_change_col not in matched_df.columns:
            alt_cols = [col for col in matched_df.columns if 'price' in col and ('change' in col or 'pct' in col)]
            price_change_col = alt_cols[0] if alt_cols else None
        
        if price_change_col and price_change_col in matched_df.columns:
            price_data = matched_df[price_change_col].dropna()
            
            if not price_data.empty:
                print(f"\n    💰 {timeframe} 가격 성과:")
                print(f"       평균 수익률: {price_data.mean():>8.2f}%")
                print(f"       중앙값:     {price_data.median():>8.2f}%")
                print(f"       표준편차:   {price_data.std():>8.2f}%")
                print(f"       최대:       {price_data.max():>8.2f}%")
                print(f"       최소:       {price_data.min():>8.2f}%")
                print(f"       승률:       {(price_data > 0).mean()*100:>8.1f}%")
                
                # 개별 사이클 매칭 세부사항
                print(f"\n    🔍 개별 매칭 세부사항:")
                print("       하위사이클ID              시작         →  상위사이클ID              기간                 수익률")
                print("       " + "─" * 85)
                
                for detail in matching_details[:10]:  # 상위 10개만 표시
                    lower_id = detail['lower_cycle_id']
                    lower_start = detail['lower_start'].strftime('%m-%d %H:%M')
                    higher_id = detail['higher_cycle_id']
                    higher_start = detail['higher_start'].strftime('%m-%d %H:%M')
                    higher_end = detail['higher_end'].strftime('%m-%d %H:%M')
                    
                    # 해당 상위 사이클의 수익률
                    if higher_id in matched_df.index and price_change_col in matched_df.columns:
                        return_pct = matched_df.loc[higher_id, price_change_col]
                        return_str = f"{return_pct:>7.2f}%" if pd.notna(return_pct) else "    N/A"
                    else:
                        return_str = "    N/A"
                    
                    print(f"       {lower_id:<25} {lower_start} → {higher_id:<25} {higher_start}~{higher_end} {return_str}")
                
                if len(matching_details) > 10:
                    print(f"       ... (총 {len(matching_details)}개 중 10개만 표시)")
                
                # 매칭 상세 통계
                self._analyze_matching_details(matching_details, matched_df, timeframe)

    def _analyze_matching_details(self, matching_details: List[Dict], matched_df: pd.DataFrame, timeframe: str):
        """매칭 세부사항 분석"""
        print(f"\n    📋 {timeframe} 매칭 세부 통계:")
        
        # 중복 상위 사이클 확인
        higher_cycle_counts = {}
        for detail in matching_details:
            higher_id = detail['higher_cycle_id']
            if higher_id in higher_cycle_counts:
                higher_cycle_counts[higher_id] += 1
            else:
                higher_cycle_counts[higher_id] = 1
        
        # 중복 매칭 분석
        overlapping_cycles = {k: v for k, v in higher_cycle_counts.items() if v > 1}
        unique_cycles = len(higher_cycle_counts)
        
        print(f"       고유한 {timeframe} 사이클: {unique_cycles}개")
        print(f"       중복 매칭 사이클: {len(overlapping_cycles)}개")
        
        if overlapping_cycles:
            print(f"       중복 매칭 상세:")
            for higher_id, count in sorted(overlapping_cycles.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"         • {higher_id}: {count}개 하위 사이클 포함")
        
        # 커버리지 효율성
        coverage_efficiency = len(matching_details) / unique_cycles if unique_cycles > 0 else 0
        print(f"       커버리지 효율성: {coverage_efficiency:.2f} (하위사이클/상위사이클 비율)")

    def _visualize_matched_higher_cycles(self, matched_df: pd.DataFrame, timeframe: str, matching_details: List[Dict]):
        """매칭된 상위 타임프레임 사이클 시각화"""
        print(f"\n🎨 {timeframe} 매칭 사이클 시각화 생성 중...")
        
        # 가격변화율 컬럼 찾기
        price_change_col = 'change.price_pct'
        if price_change_col not in matched_df.columns:
            alt_cols = [col for col in matched_df.columns if 'price' in col and ('change' in col or 'pct' in col)]
            price_change_col = alt_cols[0] if alt_cols else None
        
        if not price_change_col:
            print(f"     ⚠️ {timeframe} 가격변화율 데이터 없음")
            return
        
        price_data = matched_df[price_change_col].dropna()
        if len(price_data) == 0:
            print(f"     ⚠️ {timeframe} 유효한 가격 데이터 없음")
            return
        
        # 2x2 시각화 레이아웃
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'📊 {timeframe} 매칭 사이클 개별 분석', fontsize=16, fontweight='bold')
        
        # 1. 매칭된 사이클들의 수익률 분포
        ax1 = axes[0, 0]
        ax1.hist(price_data, bins=15, alpha=0.7, color='#4A90E2', edgecolor='darkblue')
        ax1.axvline(price_data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'평균: {price_data.mean():.2f}%')
        ax1.axvline(0, color='black', linestyle=':', alpha=0.7, label='손익분기점')
        ax1.set_title(f'{timeframe} 매칭 사이클 수익률 분포')
        ax1.set_xlabel('가격변화율 (%)')
        ax1.set_ylabel('빈도')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 시간별 매칭 관계 (시계열)
        ax2 = axes[0, 1]
        if len(matching_details) > 1:
            # 시간 순서로 정렬
            sorted_details = sorted(matching_details, key=lambda x: x['lower_start'])
            
            # 하위 사이클 시작 시점과 상위 사이클 수익률 관계
            x_times = [detail['lower_start'] for detail in sorted_details]
            y_returns = []
            
            for detail in sorted_details:
                higher_id = detail['higher_cycle_id']
                if higher_id in matched_df.index and price_change_col in matched_df.columns:
                    return_val = matched_df.loc[higher_id, price_change_col]
                    y_returns.append(return_val if pd.notna(return_val) else 0)
                else:
                    y_returns.append(0)
            
            ax2.plot(x_times, y_returns, 'o-', color='#E94B3C', linewidth=2, markersize=6, alpha=0.8)
            ax2.axhline(y=0, color='black', linestyle=':', alpha=0.7)
            ax2.axhline(y=np.mean(y_returns), color='red', linestyle='--', alpha=0.8, 
                       label=f'평균: {np.mean(y_returns):.2f}%')
            
            ax2.set_title(f'{self.current_timeframe} → {timeframe} 매칭 시계열')
            ax2.set_xlabel('하위 사이클 시작 시점')
            ax2.set_ylabel(f'{timeframe} 사이클 수익률 (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 날짜 레이블 회전
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax2.text(0.5, 0.5, '시계열 데이터 부족', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'{self.current_timeframe} → {timeframe} 매칭 시계열')
        
        # 3. 매칭 타입별 성과 분석
        ax3 = axes[1, 0]
        if 'cycle_type' in matched_df.columns:
            up_returns = matched_df[matched_df['cycle_type'] == 'up'][price_change_col].dropna()
            down_returns = matched_df[matched_df['cycle_type'] == 'down'][price_change_col].dropna()
            
            box_data = []
            box_labels = []
            
            if len(up_returns) > 0:
                box_data.append(up_returns)
                box_labels.append(f'상승\n(n={len(up_returns)})')
            
            if len(down_returns) > 0:
                box_data.append(down_returns)
                box_labels.append(f'하락\n(n={len(down_returns)})')
            
            if box_data:
                bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True)
                
                # 색상 설정
                if len(bp['boxes']) >= 1:
                    bp['boxes'][0].set_facecolor('#90EE90')  # 연한 초록
                if len(bp['boxes']) >= 2:
                    bp['boxes'][1].set_facecolor('#FFB6C1')  # 연한 빨강
                
                ax3.axhline(y=0, color='black', linestyle=':', alpha=0.7)
                ax3.set_title(f'{timeframe} 사이클 타입별 수익률')
                ax3.set_ylabel('수익률 (%)')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, '타입별 데이터 없음', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title(f'{timeframe} 사이클 타입별 수익률')
        else:
            ax3.text(0.5, 0.5, '사이클 타입 정보 없음', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(f'{timeframe} 사이클 타입별 수익률')
        
        # 4. 매칭 커버리지 분석
        ax4 = axes[1, 1]
        
        # 매칭 성공률 계산
        total_attempts = len(matching_details) + (len([d for d in matching_details if 'failed' in str(d)]))  # 실패 포함
        successful_matches = len(matching_details)
        success_rate = successful_matches / total_attempts * 100 if total_attempts > 0 else 0
        
        # 매칭 성공률 시각화
        if successful_matches > 0:
            # 중복 상위 사이클 vs 고유 상위 사이클
            higher_cycle_counts = {}
            for detail in matching_details:
                higher_id = detail['higher_cycle_id']
                higher_cycle_counts[higher_id] = higher_cycle_counts.get(higher_id, 0) + 1
            
            unique_higher = len(higher_cycle_counts)
            overlapping_count = sum(1 for count in higher_cycle_counts.values() if count > 1)
            
            sizes = [unique_higher - overlapping_count, overlapping_count]
            labels = [f'고유매칭\n({unique_higher - overlapping_count}개)', f'중복매칭\n({overlapping_count}개)']
            colors = ['#32CD32', '#FFD700']
            
            # 0이 아닌 값만 표시
            non_zero_data = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
            if len(non_zero_data) > 0:
                if len(non_zero_data) == 1:
                    # 모두 같은 타입인 경우
                    ax4.text(0.5, 0.5, f'{labels[0] if sizes[0] > 0 else labels[1]}\n100%', 
                            ha='center', va='center', transform=ax4.transAxes, 
                            fontsize=14, fontweight='bold', color='green')
                else:
                    sizes_nz, labels_nz, colors_nz = zip(*non_zero_data)
                    wedges, texts, autotexts = ax4.pie(sizes_nz, labels=labels_nz, colors=colors_nz, 
                                                      autopct='%1.1f%%', startangle=90)
                    
                    # 중앙에 총 매칭수 표시
                    ax4.text(0, 0, f'{len(matching_details)}개\n매칭', 
                            ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, '매칭 실패', ha='center', va='center', transform=ax4.transAxes, 
                    fontsize=14, fontweight='bold', color='red')
        
        ax4.set_title(f'{self.current_timeframe} → {timeframe} 매칭 분석')
        
        plt.tight_layout()
        plt.show()

    def _compare_timeframe_matching_results(self, filtered_dates: List[Dict]):
        """타임프레임 간 매칭 결과 비교 분석"""
        print(f"\n📊 타임프레임 간 매칭 결과 비교")
        print("=" * 80)
        
        comparison_results = {}
        
        # 각 타임프레임별 매칭 결과 수집
        for timeframe in sorted(self.higher_timeframe_paths.keys()):
            higher_df = self._load_higher_timeframe_data(timeframe)
            
            if len(higher_df) == 0:
                continue
            
            # 매칭 수행
            matched_cycles = []
            for cycle_info in filtered_dates:
                containing_cycle = self._find_cycle_containing_timestamp(higher_df, cycle_info['start_date'])
                if len(containing_cycle) > 0:
                    matched_cycles.append(containing_cycle.iloc[0])
            
            if matched_cycles:
                matched_df = pd.DataFrame(matched_cycles)
                
                # 성과 지표 계산
                price_change_col = 'change.price_pct'
                if price_change_col not in matched_df.columns:
                    alt_cols = [col for col in matched_df.columns if 'price' in col and ('change' in col or 'pct' in col)]
                    price_change_col = alt_cols[0] if alt_cols else None
                
                if price_change_col:
                    price_data = matched_df[price_change_col].dropna()
                    
                    comparison_results[timeframe] = {
                        'total_cycles': len(filtered_dates),
                        'matched_cycles': len(matched_cycles),
                        'matching_rate': len(matched_cycles) / len(filtered_dates) * 100,
                        'avg_return': price_data.mean() if len(price_data) > 0 else 0,
                        'win_rate': (price_data > 0).mean() * 100 if len(price_data) > 0 else 0,
                        'volatility': price_data.std() if len(price_data) > 0 else 0
                    }
        
        if not comparison_results:
            print("   비교할 매칭 결과가 없습니다.")
            return
        
        # 비교 테이블 출력
        print(f"\n📈 타임프레임별 매칭 성과 비교:")
        print("─" * 80)
        print(f"{'타임프레임':<8} {'매칭률':>8} {'평균수익':>8} {'승률':>8} {'변동성':>8} {'매칭수':>8}")
        print("─" * 80)
        
        for timeframe in sorted(comparison_results.keys()):
            data = comparison_results[timeframe]
            print(f"{timeframe:<8} {data['matching_rate']:>7.1f}% {data['avg_return']:>7.2f}% "
                  f"{data['win_rate']:>7.1f}% {data['volatility']:>7.2f}% "
                  f"{data['matched_cycles']:>7}개")
        
        # 최적 타임프레임 추천
        print(f"\n🎯 매칭 분석 결과:")
        
        # 매칭률이 높은 타임프레임
        best_matching = max(comparison_results.items(), key=lambda x: x[1]['matching_rate'])
        print(f"   🔍 최고 매칭률: {best_matching[0]} ({best_matching[1]['matching_rate']:.1f}%)")
        
        # 수익률이 높은 타임프레임
        best_return = max(comparison_results.items(), key=lambda x: x[1]['avg_return'])
        print(f"   💰 최고 평균수익: {best_return[0]} ({best_return[1]['avg_return']:.2f}%)")
        
        # 종합 점수 (매칭률 + 수익률 + 승률 - 변동성)
        for timeframe, data in comparison_results.items():
            score = (data['matching_rate'] * 0.3 + 
                    data['avg_return'] * 2 + 
                    data['win_rate'] * 0.5 - 
                    data['volatility'] * 0.3)
            comparison_results[timeframe]['composite_score'] = score
        
        best_overall = max(comparison_results.items(), key=lambda x: x[1]['composite_score'])
        print(f"   🏆 종합 최적: {best_overall[0]} (점수: {best_overall[1]['composite_score']:.2f})")

    def analyze_higher_timeframes(self, filtered_dates: List[Dict]):
        """상위 타임프레임 종합 분석 (개별 시점 매칭 방식)"""
        if not filtered_dates:
            print("\n❌ 분석할 날짜 데이터가 없습니다.")
            return
        
        if not self.higher_timeframe_paths:
            print(f"\n🔍 {self.current_timeframe}보다 상위 타임프레임이 없습니다.")
            return
        
        print(f"\n🔍 상위 타임프레임 개별 시점 매칭 분석")
        print("=" * 90)
        print(f"📅 분석 대상: {len(filtered_dates)}개 {self.current_timeframe} 사이클")
        print(f"🎯 분석 방식: 각 시작 시점이 포함된 상위 사이클 개별 매칭")
        
        # 샘플 시점 표시
        print(f"\n📋 분석할 시점들 (샘플 5개):")
        for i, cycle_info in enumerate(filtered_dates[:5], 1):
            start_time = cycle_info['start_date'].strftime('%Y-%m-%d %H:%M')
            print(f"   {i}. {cycle_info['cycle_id']}: {start_time}")
        
        if len(filtered_dates) > 5:
            print(f"   ... 외 {len(filtered_dates) - 5}개 더")
        
        # 개별 시점 기반 분석 실행
        self._analyze_higher_timeframe_cycles(filtered_dates)
        
        # 타임프레임 간 비교 분석 (선택 사항)
        if len(self.higher_timeframe_paths) > 1:
            if input(f"\n📊 타임프레임 간 비교 분석을 하시겠습니까? (y/n): ").lower() == 'y':
                self._compare_timeframe_matching_results(filtered_dates)

    def _visualize_price_statistics(self, filtered_df: pd.DataFrame):
        """개선된 가격변화율 및 최대상승/하락 통계 시각화"""
        print("\n" + "─" * 90)
        print("📊 가격변화율 및 극값 통계 시각화")
        print("─" * 90)
        
        # 필요한 컬럼들 찾기
        price_change_col = 'change.price_pct'
        max_high_col = 'volatility.max_high_pct'
        max_loss_col = 'volatility.max_loss_pct'
        
        # 대안 컬럼명 찾기
        if price_change_col not in filtered_df.columns:
            alt_cols = [col for col in filtered_df.columns if 'price' in col and ('change' in col or 'pct' in col)]
            price_change_col = alt_cols[0] if alt_cols else None
            
        if max_high_col not in filtered_df.columns:
            alt_cols = [col for col in filtered_df.columns if 'max' in col and 'high' in col]
            max_high_col = alt_cols[0] if alt_cols else None
            
        if max_loss_col not in filtered_df.columns:
            alt_cols = [col for col in filtered_df.columns if 'max' in col and ('loss' in col or 'low' in col)]
            max_loss_col = alt_cols[0] if alt_cols else None
        
        if not price_change_col:
            print("   ⚠️ 가격변화율 데이터를 찾을 수 없습니다.")
            return
        
        # 데이터 준비
        price_data = filtered_df[price_change_col].dropna()
        total_price_data = self.master_df[price_change_col].dropna()
        
        max_high_data = filtered_df[max_high_col].dropna() if max_high_col else pd.Series()
        max_loss_data = filtered_df[max_loss_col].dropna() if max_loss_col else pd.Series()
        
        if len(price_data) == 0:
            print("   ⚠️ 유효한 가격변화율 데이터가 없습니다.")
            return
        
        # 6개 차트 구성: 2x3 레이아웃으로 변경하여 더 많은 정보 표시
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('📊 가격 성과 및 리스크 종합 분석', fontsize=18, fontweight='bold', y=0.95)
        
        # 1. 향상된 히스토그램 (분포 비교)
        ax1 = axes[0, 0]
        
        # 더 명확한 색상과 스타일 사용
        ax1.hist(total_price_data, bins=40, alpha=0.4, color='lightgray', 
                label=f'전체 (n={len(total_price_data):,})', density=True, edgecolor='gray')
        
        ax1.hist(price_data, bins=25, alpha=0.8, color='#1f77b4', 
                label=f'필터링 (n={len(price_data):,})', density=True, edgecolor='darkblue')
        
        # 평균선을 더 굵고 명확하게
        ax1.axvline(total_price_data.mean(), color='gray', linestyle='--', linewidth=2, 
                   label=f'전체 평균: {total_price_data.mean():.2f}%')
        ax1.axvline(price_data.mean(), color='#d62728', linestyle='-', linewidth=3, 
                   label=f'그룹 평균: {price_data.mean():.2f}%')
        
        # 0% 기준선 추가
        ax1.axvline(0, color='black', linestyle=':', alpha=0.5, label='손익분기점')
        
        ax1.set_title('💹 가격변화율 분포 비교', fontsize=12, fontweight='bold')
        ax1.set_xlabel('가격변화율 (%)', fontweight='bold')
        ax1.set_ylabel('밀도', fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. 향상된 박스플롯 (분위수 + 통계정보)
        ax2 = axes[0, 1]
        
        box_data = [total_price_data.values, price_data.values]
        box_labels = ['전체', '필터링']
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, 
                        showmeans=True, meanline=True)
        bp['boxes'][0].set_facecolor('lightgray')
        bp['boxes'][1].set_facecolor('#1f77b4')
        
        # 평균 표시 개선
        bp['means'][0].set_color('gray')
        bp['means'][1].set_color('red')
        bp['means'][0].set_linewidth(2)
        bp['means'][1].set_linewidth(2)
        
        ax2.set_title('📦 분위수 및 이상치 비교', fontsize=12, fontweight='bold')
        ax2.set_ylabel('가격변화율 (%)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 상세 통계 정보
        ax2.text(0.02, 0.98, 
                f'📊 전체: 평균 {total_price_data.mean():.2f}%, 표준편차 {total_price_data.std():.2f}%\n'
                f'🎯 필터링: 평균 {price_data.mean():.2f}%, 표준편차 {price_data.std():.2f}%\n'
                f'📄 차이: {price_data.mean()-total_price_data.mean():+.2f}% ({((price_data.mean()-total_price_data.mean())/total_price_data.mean()*100):+.1f}%)', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        
        # 3. 새로운 기능: 최대 상승/하락 분석
        ax3 = axes[0, 2]
        
        if len(max_high_data) > 0 and len(max_loss_data) > 0:
            # 상승/하락 극값 비교
            extreme_data = {
                '최대상승': max_high_data.values,
                '최대하락': np.abs(max_loss_data.values)  # 절댓값으로 변환
            }
            
            positions = [1, 2]
            bp_extreme = ax3.boxplot([extreme_data['최대상승'], extreme_data['최대하락']], 
                                   labels=['최대상승%', '최대하락%'], patch_artist=True,
                                   showmeans=True)
            
            # 색상 구분: 상승은 초록, 하락은 빨강
            bp_extreme['boxes'][0].set_facecolor('#2E8B57')  # 초록
            bp_extreme['boxes'][1].set_facecolor('#DC143C')  # 빨강
            
            ax3.set_title('🎢 사이클내 최대 상승/하락 분석', fontsize=12, fontweight='bold')
            ax3.set_ylabel('변화율 (%)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # 극값 통계 정보
            max_up_avg = max_high_data.mean()
            max_down_avg = np.abs(max_loss_data).mean()
            risk_reward = max_up_avg / max_down_avg if max_down_avg > 0 else 0
            
            ax3.text(0.02, 0.98, 
                    f'📈 평균 최대상승: {max_up_avg:.2f}%\n'
                    f'📉 평균 최대하락: {max_down_avg:.2f}%\n'
                    f'⚖️ 상승/하락 비율: {risk_reward:.2f}', 
                    transform=ax3.transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        else:
            ax3.text(0.5, 0.5, '최대상승/하락\n데이터 없음', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('🎢 사이클내 최대 상승/하락 분석', fontsize=12, fontweight='bold')
        
        # 4. 성과 구간별 세분화 분석 (도넛 차트로 개선)
        ax4 = axes[1, 0]
        
        # 더 세분화된 성과 구간
        extreme_gain = (price_data > 20).sum()
        high_gain = ((price_data > 10) & (price_data <= 20)).sum()
        moderate_gain = ((price_data > 0) & (price_data <= 10)).sum()
        moderate_loss = ((price_data >= -10) & (price_data < 0)).sum()
        high_loss = ((price_data >= -20) & (price_data < -10)).sum()
        extreme_loss = (price_data < -20).sum()
        
        sizes = [extreme_gain, high_gain, moderate_gain, moderate_loss, high_loss, extreme_loss]
        labels = ['극고수익\n(>20%)', '고수익\n(10~20%)', '소수익\n(0~10%)', 
                 '소손실\n(0~-10%)', '고손실\n(-10~-20%)', '극손실\n(<-20%)']
        colors = ['#006400', '#32CD32', '#90EE90', '#FFB6C1', '#FF6347', '#8B0000']
        
        # 0이 아닌 값만 표시 (도넛 차트)
        non_zero_sizes = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
        if non_zero_sizes:
            sizes_nz, labels_nz, colors_nz = zip(*non_zero_sizes)
            wedges, texts, autotexts = ax4.pie(sizes_nz, labels=labels_nz, colors=colors_nz, 
                                              autopct='%1.1f%%', startangle=90, 
                                              wedgeprops=dict(width=0.5))  # 도넛 차트
            
            # 중앙에 핵심 통계 표시
            ax4.text(0, 0, f'총 {len(price_data)}개\n사이클', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=ax4.transAxes)
            
        ax4.set_title('🎯 세분화된 성과 구간 분포', fontsize=12, fontweight='bold')
        
        # 5. 수익률 vs 최대상승률 산점도 (리스크-리워드 분석)
        ax5 = axes[1, 1]
        
        if len(max_high_data) > 0:
            # 공통 인덱스 확인
            common_idx = price_data.index.intersection(max_high_data.index)
            if len(common_idx) > 10:
                x_data = price_data.loc[common_idx]
                y_data = max_high_data.loc[common_idx]
                
                # 산점도 생성 (성과별 색상 구분)
                colors = ['red' if x < 0 else 'green' if x > 10 else 'orange' for x in x_data]
                scatter = ax5.scatter(x_data, y_data, c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                
                # 추세선 추가
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax5.plot(x_data.sort_values(), p(x_data.sort_values()), "r--", alpha=0.8, linewidth=2)
                
                # 기준선들
                ax5.axhline(y=y_data.mean(), color='blue', linestyle=':', alpha=0.7, label=f'평균 최대상승: {y_data.mean():.1f}%')
                ax5.axvline(x=0, color='black', linestyle=':', alpha=0.5, label='손익분기점')
                
                ax5.set_title('🎯 수익률 vs 최대상승률 (리스크-리워드)', fontsize=12, fontweight='bold')
                ax5.set_xlabel('최종 가격변화율 (%)', fontweight='bold')
                ax5.set_ylabel('사이클내 최대상승률 (%)', fontweight='bold')
                ax5.legend(fontsize=9)
                ax5.grid(True, alpha=0.3)
                
                # 상관관계 정보
                correlation = np.corrcoef(x_data, y_data)[0, 1]
                ax5.text(0.02, 0.98, f'상관계수: {correlation:.3f}', 
                        transform=ax5.transAxes, verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                ax5.text(0.5, 0.5, '산점도 분석을 위한\n충분한 데이터 없음', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('🎯 수익률 vs 최대상승률', fontsize=12, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, '최대상승률\n데이터 없음', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('🎯 수익률 vs 최대상승률', fontsize=12, fontweight='bold')
        
        # 6. 리스크 히트맵 (수익률 vs 최대손실률)
        ax6 = axes[1, 2]
        
        if len(max_loss_data) > 0:
            common_idx = price_data.index.intersection(max_loss_data.index)
            if len(common_idx) > 10:
                x_data = price_data.loc[common_idx]
                y_data = np.abs(max_loss_data.loc[common_idx])  # 절댓값
                
                # 리스크 히트맵 (2D 히스토그램)
                hb = ax6.hexbin(x_data, y_data, gridsize=15, cmap='YlOrRd', alpha=0.8)
                
                # 컬러바 추가
                cb = plt.colorbar(hb, ax=ax6, shrink=0.8)
                cb.set_label('사이클 밀도', fontweight='bold')
                
                ax6.set_title('🔥 리스크 히트맵 (수익률 vs 최대손실률)', fontsize=12, fontweight='bold')
                ax6.set_xlabel('최종 가격변화율 (%)', fontweight='bold')
                ax6.set_ylabel('사이클내 최대손실률 (%)', fontweight='bold')
                
                # 안전 구역 표시
                ax6.axhline(y=y_data.quantile(0.25), color='green', linestyle='--', alpha=0.7, label='낮은 리스크')
                ax6.axhline(y=y_data.quantile(0.75), color='red', linestyle='--', alpha=0.7, label='높은 리스크')
                ax6.legend(fontsize=9)
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, '히트맵 생성을 위한\n충분한 데이터 없음', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('🔥 리스크 히트맵', fontsize=12, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, '최대손실률\n데이터 없음', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('🔥 리스크 히트맵', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 향상된 통계 요약
        print(f"\n📈 가격변화율 및 극값 핵심 통계:")
        print("─" * 60)
        print(f"📊 최종 수익률:")
        print(f"   평균: {price_data.mean():>8.2f}% (전체: {total_price_data.mean():>8.2f}%)")
        print(f"   중앙값: {price_data.median():>6.2f}% (전체: {total_price_data.median():>8.2f}%)")
        print(f"   승률: {(price_data > 0).mean()*100:>8.1f}% (전체: {(total_price_data > 0).mean()*100:>8.1f}%)")
        
        if len(max_high_data) > 0:
            print(f"\n🚀 사이클내 최대 상승:")
            print(f"   평균: {max_high_data.mean():>8.2f}%")
            print(f"   최대: {max_high_data.max():>8.2f}%")
            print(f"   20% 이상: {(max_high_data > 20).mean()*100:>5.1f}%")
        
        if len(max_loss_data) > 0:
            max_loss_abs = np.abs(max_loss_data)
            print(f"\n📉 사이클내 최대 하락:")
            print(f"   평균: {max_loss_abs.mean():>8.2f}%")
            print(f"   최대: {max_loss_abs.max():>8.2f}%")
            print(f"   20% 이상: {(max_loss_abs > 20).mean()*100:>5.1f}%")
            
            # 리스크-리워드 비율
            if len(max_high_data) > 0:
                risk_reward = max_high_data.mean() / max_loss_abs.mean()
                print(f"\n⚖️ 리스크-리워드 비율: {risk_reward:.2f} (높을수록 유리)")

    def _show_individual_cycles(self, filtered_df: pd.DataFrame):
        """새로운 기능: 개별 사이클 상세 조회"""
        print("\n" + "─" * 90)
        print("🔍 개별 사이클 상세 분석")
        print("─" * 90)
        
        if len(filtered_df) == 0:
            print("   분석할 사이클이 없습니다.")
            return
        
        # 상위 N개 사이클 표시 옵션
        print(f"\n📋 필터링된 {len(filtered_df)}개 사이클 중 조회 옵션:")
        print("   1. 상위 10개 (가격변화율 높은 순)")
        print("   2. 하위 10개 (가격변화율 낮은 순)")
        print("   3. 랜덤 10개")
        print("   4. 특정 사이클 ID로 검색")
        print("   5. 전체 요약만 보기")
        
        choice = input("\n선택하세요 (1-5): ").strip()
        
        price_change_col = 'change.price_pct'
        if price_change_col not in filtered_df.columns:
            alt_cols = [col for col in filtered_df.columns if 'price' in col and ('change' in col or 'pct' in col)]
            price_change_col = alt_cols[0] if alt_cols else None
        
        if choice == '1' and price_change_col:
            # 상위 10개
            top_cycles = filtered_df.nlargest(10, price_change_col)
            self._display_cycle_details(top_cycles, "상위 10개 고수익 사이클")
            
        elif choice == '2' and price_change_col:
            # 하위 10개
            bottom_cycles = filtered_df.nsmallest(10, price_change_col)
            self._display_cycle_details(bottom_cycles, "하위 10개 저수익 사이클")
            
        elif choice == '3':
            # 랜덤 10개
            random_cycles = filtered_df.sample(min(10, len(filtered_df)))
            self._display_cycle_details(random_cycles, "랜덤 10개 사이클")
            
        elif choice == '4':
            # 특정 ID 검색
            cycle_id = input("사이클 ID를 입력하세요: ").strip()
            if cycle_id in filtered_df.index:
                specific_cycle = filtered_df.loc[[cycle_id]]
                self._display_cycle_details(specific_cycle, f"사이클 {cycle_id} 상세 정보")
            else:
                print(f"   ❌ 사이클 ID '{cycle_id}'를 찾을 수 없습니다.")
                
        elif choice == '5':
            # 전체 요약
            self._display_cycles_summary(filtered_df)
        else:
            print("   ❌ 잘못된 선택입니다.")

    def _display_cycle_details(self, cycles_df: pd.DataFrame, title: str):
        """개별 사이클 상세 정보 표시 (확장된 특징값들 포함)"""
        print(f"\n📋 {title}")
        print("─" * 100)
        
        # 확장된 주요 컬럼 선택 (존재하는 것만)
        key_columns = []
        column_mapping = {
            # 기본 정보
            'start_date': '시작일',
            'end_date': '종료일', 
            'cycle_type': '타입',
            
            # 성과 관련
            'change.price_pct': '최종수익률(%)',
            'volatility.max_high_pct': '최대상승률(%)',
            'volatility.max_loss_pct': '최대하락률(%)',
            
            # 구조적 특징
            'shape.duration_candles': '지속기간(캔들)',
            'shape.core_count': '핵심캔들수',
            'shape.noise_count': '노이즈캔들수',
            'shape.direction_change': '방향전환횟수',
            
            # 강도 특징
            'strength.direction_pct': '일관성(%)',
            'strength.hist_positive_ratio': '히스토그램양수비율(%)',
            'strength.price_up_ratio': '상승캔들비율(%)',
            'strength.price_down_ratio': '하락캔들비율(%)',
            
            # 시작점 지표
            'start.price': '시작가격',
            'start.rsi': '시작RSI',
            'start.macd': '시작MACD',
            'start.hist': '시작히스토그램',
            'start.volume': '시작거래량',
            
            # 종료점 지표
            'end.price': '종료가격',
            'end.rsi': '종료RSI',
            'end.macd': '종료MACD',
            'end.hist': '종료히스토그램',
            'end.volume': '종료거래량',
            
            # 변화량
            'change.rsi': 'RSI변화',
            'change.macd': 'MACD변화',
            'change.hist': '히스토그램변화',
            'change.macd_signal': 'MACD시그널변화',
            
            # 변동성
            'volatility.avg_true_range': '평균변동폭(ATR)',
            'volatility.price_change_deviation': '가격변동편차',
            'volatility.max_intraday_high_pct': '최대일중상승(%)',
            'volatility.max_intraday_loss_pct': '최대일중하락(%)',
            
            # 거래량
            'aggregate.volume': '총거래량'
        }
        
        # 실제 존재하는 컬럼만 필터링
        for col, korean_name in column_mapping.items():
            if col in cycles_df.columns:
                key_columns.append((col, korean_name))
        
        # 그룹별로 컬럼 정리
        basic_info = [(col, name) for col, name in key_columns if any(x in col for x in ['date', 'type'])]
        performance = [(col, name) for col, name in key_columns if any(x in col for x in ['change.price', 'max_high', 'max_loss'])]
        structure = [(col, name) for col, name in key_columns if col.startswith('shape.')]
        strength = [(col, name) for col, name in key_columns if col.startswith('strength.')]
        indicators_start = [(col, name) for col, name in key_columns if col.startswith('start.')]
        indicators_end = [(col, name) for col, name in key_columns if col.startswith('end.')]
        changes = [(col, name) for col, name in key_columns if col.startswith('change.') and 'price' not in col]
        volatility = [(col, name) for col, name in key_columns if col.startswith('volatility.') and 'max_high' not in col and 'max_loss' not in col]
        volume = [(col, name) for col, name in key_columns if col.startswith('aggregate.')]
        
        for i, (cycle_id, row) in enumerate(cycles_df.iterrows(), 1):
            print(f"\n🔍 {i:2d}. 사이클 ID: {cycle_id}")
            print("    " + "=" * 80)
            
            # 기본 정보
            if basic_info:
                print("    📋 기본 정보:")
                for col, korean_name in basic_info:
                    value = row[col]
                    if pd.isna(value):
                        display_value = "N/A"
                    elif isinstance(value, pd.Timestamp):
                        display_value = value.strftime('%Y-%m-%d %H:%M')
                    else:
                        display_value = str(value)
                    print(f"        {korean_name:15}: {display_value}")
            
            # 성과 분석
            if performance:
                print("\n    💰 성과 분석:")
                for col, korean_name in performance:
                    value = row[col]
                    if pd.isna(value):
                        display_value = "N/A"
                    else:
                        display_value = f"{value:>8.2f}%"
                        # 성과에 따른 이모지 추가
                        if 'price' in col and value > 10:
                            display_value += " 🔥"
                        elif 'price' in col and value > 5:
                            display_value += " 📈"
                        elif 'price' in col and value < -10:
                            display_value += " 📉"
                        elif 'max_high' in col and value > 20:
                            display_value += " 🚀"
                        elif 'max_loss' in col and abs(value) > 20:
                            display_value += " ⚠️"
                    print(f"        {korean_name:15}: {display_value}")
            
            # 구조적 특징
            if structure:
                print("\n    🗗️ 구조적 특징:")
                for col, korean_name in structure:
                    value = row[col]
                    if pd.isna(value):
                        display_value = "N/A"
                    else:
                        if 'pct' in col or '%' in korean_name:
                            display_value = f"{value:>8.2f}%"
                        else:
                            display_value = f"{value:>8.0f}"
                        
                        # 구조적 특징 평가
                        if 'noise' in col and value < 3:
                            display_value += " ✨"  # 낮은 노이즈
                        elif 'duration' in col and value > 15:
                            display_value += " ⏳"  # 긴 지속
                        elif 'core' in col and value > 10:
                            display_value += " 💪"  # 강한 코어
                    print(f"        {korean_name:15}: {display_value}")
            
            # 강도 및 일관성
            if strength:
                print("\n    💪 강도 및 일관성:")
                for col, korean_name in strength:
                    value = row[col]
                    if pd.isna(value):
                        display_value = "N/A"
                    else:
                        display_value = f"{value:>8.2f}%"
                        # 강도 평가
                        if 'direction' in col and value > 85:
                            display_value += " 💎"  # 매우 일관됨
                        elif 'direction' in col and value > 70:
                            display_value += " ⭐"  # 일관됨
                        elif 'positive' in col and value > 70:
                            display_value += " 📊"  # 양수 우세
                    print(f"        {korean_name:15}: {display_value}")
            
            # 시작점 지표 (2열로 표시)
            if indicators_start:
                print("\n    🎯 시작점 지표:")
                for j in range(0, len(indicators_start), 2):
                    line_items = indicators_start[j:j+2]
                    line_text = "        "
                    for col, korean_name in line_items:
                        value = row[col]
                        if pd.isna(value):
                            display_value = "N/A"
                        elif 'price' in col:
                            display_value = f"{value:>10.2f}"
                        elif 'volume' in col:
                            display_value = f"{value:>10.0f}"
                        else:
                            display_value = f"{value:>8.2f}"
                        line_text += f"{korean_name:12}: {display_value}    "
                    print(line_text)
            
            # 종료점 지표 (2열로 표시)
            if indicators_end:
                print("\n    🏁 종료점 지표:")
                for j in range(0, len(indicators_end), 2):
                    line_items = indicators_end[j:j+2]
                    line_text = "        "
                    for col, korean_name in line_items:
                        value = row[col]
                        if pd.isna(value):
                            display_value = "N/A"
                        elif 'price' in col:
                            display_value = f"{value:>10.2f}"
                        elif 'volume' in col:
                            display_value = f"{value:>10.0f}"
                        else:
                            display_value = f"{value:>8.2f}"
                        line_text += f"{korean_name:12}: {display_value}    "
                    print(line_text)
            
            # 변화량 (2열로 표시)
            if changes:
                print("\n    📈 기술지표 변화량:")
                for j in range(0, len(changes), 2):
                    line_items = changes[j:j+2]
                    line_text = "        "
                    for col, korean_name in line_items:
                        value = row[col]
                        if pd.isna(value):
                            display_value = "N/A"
                        else:
                            display_value = f"{value:>+8.3f}"
                        line_text += f"{korean_name:15}: {display_value}  "
                    print(line_text)
            
            # 변동성 지표
            if volatility:
                print("\n    🎢 변동성 지표:")
                for col, korean_name in volatility:
                    value = row[col]
                    if pd.isna(value):
                        display_value = "N/A"
                    else:
                        if 'pct' in col:
                            display_value = f"{value:>8.2f}%"
                        else:
                            display_value = f"{value:>8.3f}"
                        
                        # 변동성 평가
                        if 'deviation' in col and value < 2:
                            display_value += " 🔒"  # 낮은 변동성
                        elif 'deviation' in col and value > 5:
                            display_value += " ⚡"  # 높은 변동성
                    print(f"        {korean_name:15}: {display_value}")
            
            # 거래량 정보
            if volume:
                print("\n    💹 거래량 정보:")
                for col, korean_name in volume:
                    value = row[col]
                    if pd.isna(value):
                        display_value = "N/A"
                    else:
                        # 거래량은 큰 숫자이므로 적절히 포맷
                        if value > 1e9:
                            display_value = f"{value/1e9:>8.2f}B"
                        elif value > 1e6:
                            display_value = f"{value/1e6:>8.2f}M"
                        elif value > 1e3:
                            display_value = f"{value/1e3:>8.2f}K"
                        else:
                            display_value = f"{value:>8.0f}"
                    print(f"        {korean_name:15}: {display_value}")
            
            # 종합 성과 평가 (더 상세하게)
            self._evaluate_cycle_performance(row)

    def _evaluate_cycle_performance(self, cycle_row: pd.Series):
        """개별 사이클의 종합 성과 평가"""
        print("\n    🎭 종합 성과 평가:")
        
        # 성과 지표들 수집
        price_change = cycle_row.get('change.price_pct', 0)
        max_high = cycle_row.get('volatility.max_high_pct', 0)
        max_loss = cycle_row.get('volatility.max_loss_pct', 0)
        direction_pct = cycle_row.get('strength.direction_pct', 0)
        noise_count = cycle_row.get('shape.noise_count', 0)
        duration = cycle_row.get('shape.duration_candles', 0)
        
        evaluations = []
        
        # 수익성 평가
        if price_change > 20:
            evaluations.append("💎 극고수익")
        elif price_change > 10:
            evaluations.append("🔥 고수익")
        elif price_change > 5:
            evaluations.append("📈 양호한수익")
        elif price_change > 0:
            evaluations.append("📊 소수익")
        elif price_change > -5:
            evaluations.append("📉 소손실")
        elif price_change > -10:
            evaluations.append("⚠️ 고손실")
        else:
            evaluations.append("❄️ 극손실")
        
        # 리스크 평가
        max_loss_abs = abs(max_loss) if not pd.isna(max_loss) else 0
        if max_loss_abs < 5:
            evaluations.append("🛡️ 저리스크")
        elif max_loss_abs < 15:
            evaluations.append("⚖️ 중간리스크")
        else:
            evaluations.append("⚠️ 고리스크")
        
        # 일관성 평가
        if direction_pct > 85:
            evaluations.append("🎯 매우일관됨")
        elif direction_pct > 70:
            evaluations.append("✅ 일관됨")
        elif direction_pct > 50:
            evaluations.append("📊 보통")
        else:
            evaluations.append("🌪️ 불안정")
        
        # 지속성 평가
        if duration > 20:
            evaluations.append("⏳ 장기추세")
        elif duration > 10:
            evaluations.append("📅 중기추세")
        else:
            evaluations.append("⚡ 단기추세")
        
        # 노이즈 평가
        if noise_count <= 1:
            evaluations.append("✨ 매우깔끔")
        elif noise_count <= 3:
            evaluations.append("🧹 깔끔함")
        else:
            evaluations.append("🌊 노이즈많음")
        
        # 리스크-리워드 비율
        if max_high > 0 and max_loss_abs > 0:
            risk_reward = max_high / max_loss_abs
            if risk_reward > 2:
                evaluations.append("🏆 우수한RR비율")
            elif risk_reward > 1.5:
                evaluations.append("👍 양호한RR비율")
            else:
                evaluations.append("👎 낮은RR비율")
        
        # 평가 결과 출력
        print("        " + " | ".join(evaluations))
        
        # 투자 관점 종합 평가
        score = 0
        if price_change > 10: score += 3
        elif price_change > 5: score += 2
        elif price_change > 0: score += 1
        
        if max_loss_abs < 10: score += 2
        elif max_loss_abs < 15: score += 1
        
        if direction_pct > 80: score += 2
        elif direction_pct > 70: score += 1
        
        if noise_count <= 2: score += 1
        
        # 종합 점수 해석
        if score >= 7:
            overall = "🏆 최우수 사이클"
        elif score >= 5:
            overall = "⭐ 우수 사이클"
        elif score >= 3:
            overall = "👍 양호한 사이클"
        elif score >= 1:
            overall = "📊 평균 사이클"
        else:
            overall = "⚠️ 주의 사이클"
        
        print(f"        🎯 종합 평가: {overall} (점수: {score}/8)")

    def _display_cycles_summary(self, filtered_df: pd.DataFrame):
        """필터링된 사이클들의 전체 요약"""
        print(f"\n📊 {len(filtered_df)}개 사이클 전체 요약")
        print("─" * 60)
        
        # 타입별 분포
        if 'cycle_type' in filtered_df.columns:
            type_dist = filtered_df['cycle_type'].value_counts()
            print("🎲 사이클 타입 분포:")
            for cycle_type, count in type_dist.items():
                pct = count / len(filtered_df) * 100
                symbol = "📈" if cycle_type == 'up' else "📉"
                print(f"   {symbol} {cycle_type}: {count}개 ({pct:.1f}%)")
        
        # 주요 수치 특징 요약
        key_numeric_features = [
            ('change.price_pct', '가격변화율'),
            ('shape.duration_candles', '지속기간'),
            ('strength.direction_pct', '일관성'),
            ('start.rsi', '시작RSI'),
            ('end.rsi', '종료RSI')
        ]
        
        print(f"\n📊 주요 특징 통계:")
        print(f"{'특징':<12} {'평균':>8} {'중앙값':>8} {'표준편차':>8} {'최소':>8} {'최대':>8}")
        print("─" * 65)
        
        for feature, korean_name in key_numeric_features:
            if feature in filtered_df.columns:
                data = filtered_df[feature].dropna()
                if len(data) > 0:
                    print(f"{korean_name:<12} {data.mean():>8.2f} {data.median():>8.2f} {data.std():>8.2f} {data.min():>8.2f} {data.max():>8.2f}")

    def _analysis_summary(self, filtered_df: pd.DataFrame, conditions: List[Dict], total_count: int, found_count: int):
        """1단계: 요약 정보"""
        print("\n" + "─" * 90)
        print("📊 1단계: 요약 정보 - 이 그룹은 무엇인가? (Summary)")
        print("─" * 90)
        
        # 적용된 필터 조건
        print("🎯 적용된 필터 조건:")
        if conditions:
            for i, cond in enumerate(conditions, 1):
                feature, op, val = cond['feature'], cond['operator'], cond['value']
                print(f"   {i}. {feature} {op} {val}")
        else:
            print("   (필터 조건 없음 - 전체 데이터)")
        
        # 표본 크기 및 희귀도 분석
        sample_pct = found_count / total_count * 100
        print(f"\n📈 표본 크기:")
        print(f"   선택된 사이클: {found_count:,}개")
        print(f"   전체 대비 비율: {sample_pct:.2f}%")
        
        # 희귀도 평가
        if sample_pct < 1:
            rarity = "매우 희귀한 조건"
            print(f"   💎 평가: {rarity} (< 1%)")
        elif sample_pct < 5:
            rarity = "드문 조건"
            print(f"   🔸 평가: {rarity} (1-5%)")
        elif sample_pct < 20:
            rarity = "일반적인 조건"
            print(f"   🔹 평가: {rarity} (5-20%)")
        else:
            rarity = "흔한 조건"
            print(f"   ⚪ 평가: {rarity} (> 20%)")
    
    def _performance_analysis_detailed(self, filtered_df: pd.DataFrame):
        """2단계: 성과 분석 (새로운 특징명 적용)"""
        print("\n" + "─" * 90)
        print("📈 2단계: 성과 분석 - 이 그룹은 좋은 성과를 내는가? (Performance Analysis)")
        print("─" * 90)
        
        # 2-1. 사이클 타입 분포 분석
        print("\n🎲 사이클 타입 분포 (Up/Down Composition):")
        if 'cycle_type' in filtered_df.columns:
            type_counts = filtered_df['cycle_type'].value_counts()
            type_pcts = filtered_df['cycle_type'].value_counts(normalize=True) * 100
            
            print("   타입별 구성:")
            for cycle_type in type_counts.index:
                count = type_counts[cycle_type]
                pct = type_pcts[cycle_type]
                bar_len = int(pct / 3)  # 30개 문자 기준
                bar = "█" * bar_len
                symbol = "📈" if cycle_type == 'up' else "📉"
                print(f"   {symbol} {cycle_type:>4}: {count:>4,}개 ({pct:>5.1f}%) {bar}")
            
            # 타입별 성향 분석
            up_ratio = type_pcts.get('up', 0)
            if up_ratio > 80:
                tendency = "강한 상승 편향"
            elif up_ratio > 60:
                tendency = "상승 편향"
            elif up_ratio >= 40:
                tendency = "균형"
            elif up_ratio >= 20:
                tendency = "하락 편향"
            else:
                tendency = "강한 하락 편향"
            print(f"   🎯 성향: {tendency} (상승 {up_ratio:.1f}%)")
        
        # 2-2. 가격 변화율 상세 분석 (새로운 특징명)
        price_change_col = 'change.price_pct'  # 새로운 경로
        if price_change_col not in filtered_df.columns:
            # 대안 컬럼명 찾기
            alt_cols = [col for col in filtered_df.columns if 'price' in col and ('change' in col or 'pct' in col)]
            price_change_col = alt_cols[0] if alt_cols else None
        
        print(f"\n💰 가격 변화율 ({price_change_col}) 상세 분석:")
        if price_change_col and price_change_col in filtered_df.columns:
            price_data = filtered_df[price_change_col].dropna()
            
            if len(price_data) > 0:
                # 기본 통계
                stats = price_data.describe()
                print(f"   📊 기본 통계:")
                print(f"      평균 수익률: {stats['mean']:>8.2f}%")
                print(f"      중앙값:     {stats['50%']:>8.2f}%") 
                print(f"      표준편차:   {stats['std']:>8.2f}%")
                print(f"      범위:       [{stats['min']:>6.2f}%, {stats['max']:>6.2f}%]")
                
                # 성과 구간 분석
                positive_ratio = (price_data > 0).mean() * 100
                high_gain_ratio = (price_data > 5).mean() * 100
                high_loss_ratio = (price_data < -5).mean() * 100
                
                print(f"   🎯 성과 구간 분석:")
                print(f"      수익 사이클: {positive_ratio:>5.1f}% (> 0%)")
                print(f"      고수익:     {high_gain_ratio:>5.1f}% (> 5%)")
                print(f"      고손실:     {high_loss_ratio:>5.1f}% (< -5%)")
                
                # 위험-수익 평가
                sharpe_like = stats['mean'] / stats['std'] if stats['std'] > 0 else 0
                print(f"   ⚖️ 위험-수익 비율: {sharpe_like:.3f} (높을수록 좋음)")
        else:
            print(f"   ⚠️ 가격 변화율 데이터를 찾을 수 없습니다.")
        
        # 전체 대비 성과 비교
        self._compare_performance_to_total(filtered_df, price_change_col)
    
    def _compare_performance_to_total(self, filtered_df: pd.DataFrame, price_change_col: str):
        """전체 데이터 대비 성과 비교"""
        print("\n📄 전체 데이터 대비 성과 비교:")
        
        if price_change_col and price_change_col in filtered_df.columns and price_change_col in self.master_df.columns:
            filtered_mean = filtered_df[price_change_col].mean()
            total_mean = self.master_df[price_change_col].mean()
            
            filtered_std = filtered_df[price_change_col].std()
            total_std = self.master_df[price_change_col].std()
            
            mean_diff = filtered_mean - total_mean
            std_diff = filtered_std - total_std
            
            print(f"   평균 수익률: 필터링 {filtered_mean:>6.2f}% vs 전체 {total_mean:>6.2f}% (차이: {mean_diff:+6.2f}%)")
            print(f"   변동성:     필터링 {filtered_std:>6.2f}% vs 전체 {total_std:>6.2f}% (차이: {std_diff:+6.2f}%)")
            
            # 성과 평가
            if mean_diff > 2 and std_diff < 0:
                evaluation = "🏆 우수한 저위험 고수익 그룹"
            elif mean_diff > 2:
                evaluation = "📈 고수익 그룹 (높은 변동성)"
            elif mean_diff > 0 and std_diff < 0:
                evaluation = "🛡️ 안정적인 수익 그룹"
            elif mean_diff > 0:
                evaluation = "📊 평균 이상 수익 그룹"
            elif std_diff < -2:
                evaluation = "🔒 안정적인 그룹"
            else:
                evaluation = "⚪ 평균적인 그룹"
            
            print(f"   🎯 종합 평가: {evaluation}")
        else:
            print("   ⚠️ 비교 가능한 성과 데이터가 없습니다.")
    
    def _group_profiling_analysis(self, filtered_df: pd.DataFrame):
        """3단계: 그룹 프로파일링 (새로운 특징명 적용)"""
        print("\n" + "─" * 90)
        print("🧬 3단계: 그룹 프로파일링 - 이 그룹은 어떤 DNA를 가졌는가? (Profiling)")
        print("─" * 90)
        
        # 3-1. 주요 특징 통계 비교
        self._feature_comparison_table(filtered_df)
        
        # 3-2. 레이더 차트 프로필
        self._create_radar_profile(filtered_df)
    
    def _feature_comparison_table(self, filtered_df: pd.DataFrame):
        """포괄적 특징 통계 비교 분석 (새로운 중첩 구조 반영)"""
        print("\n📋 포괄적 특징 통계 비교 (필터링 그룹 vs 전체):")
        
        # 특징을 카테고리별로 체계적으로 분류 (새로운 중첩 구조)
        feature_categories = {
            "🎯 시작점 특징": [f for f in self.numeric_features if f.startswith('start.')],
            "🏁 끝점 특징": [f for f in self.numeric_features if f.startswith('end.')],
            "📈 변화량 특징": [f for f in self.numeric_features if f.startswith('change.')],
            "⏱️ 지속성 특징": [f for f in self.numeric_features if f.startswith('shape.')],
            "💪 강도 특징": [f for f in self.numeric_features if f.startswith('strength.')],
            "🎢 변동성 특징": [f for f in self.numeric_features if f.startswith('volatility.')],
            "💰 거래량 특징": [f for f in self.numeric_features if f.startswith('aggregate.')]
        }
        
        # 카테고리별 분석
        all_significant_differences = []
        total_analyzed_features = 0
        
        for category, candidate_features in feature_categories.items():
            available_features = []
            for feature in candidate_features:
                if (feature in filtered_df.columns and 
                    feature in self.master_df.columns and
                    filtered_df[feature].notna().sum() > 0 and
                    self.master_df[feature].notna().sum() > 0):
                    available_features.append(feature)
            
            if not available_features:
                continue
                
            print(f"\n{category} ({len(available_features)}개 특징):")
            print("   " + "─" * 85)
            print(f"   {'특징명':<25} {'필터링':>10} {'전체':>10} {'차이':>10} {'변화율':>8} {'유의성':>6}")
            print("   " + "─" * 85)
            
            category_significant = []
            
            for feature in available_features:
                # 기본 통계 계산
                filtered_data = filtered_df[feature].dropna()
                total_data = self.master_df[feature].dropna()
                
                if len(filtered_data) == 0 or len(total_data) == 0:
                    continue
                
                filtered_mean = filtered_data.mean()
                total_mean = total_data.mean()
                difference = filtered_mean - total_mean
                change_pct = (difference / total_mean * 100) if total_mean != 0 else 0
                
                # 통계적 유의성 검증 (t-test)
                try:
                    from scipy import stats as scipy_stats
                    t_stat, p_value = scipy_stats.ttest_ind(filtered_data, total_data.sample(min(len(total_data), 1000)))
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                except:
                    significance = ""
                
                # 실용적 유의미성 판단
                practical_significance = ""
                if abs(change_pct) > 50:
                    practical_significance = "🔥"
                    category_significant.append((feature, change_pct, difference, "매우 높음"))
                elif abs(change_pct) > 25:
                    practical_significance = "⚡"
                    category_significant.append((feature, change_pct, difference, "높음"))
                elif abs(change_pct) > 10:
                    practical_significance = "⚠️"
                    category_significant.append((feature, change_pct, difference, "중간"))
                else:
                    practical_significance = ""
                
                # 특징명을 짧게 표시 (중첩 경로 축약)
                display_name = feature.split('.')[-1] if '.' in feature else feature
                print(f"{practical_significance:>2} {display_name:<23} {filtered_mean:>10.3f} {total_mean:>10.3f} {difference:>+10.3f} {change_pct:>+7.1f}% {significance:>6}")
                
                total_analyzed_features += 1
            
            # 카테고리별 요약
            if category_significant:
                print(f"\n   📊 {category} 주요 차이점:")
                for feature, change_pct, diff, intensity in sorted(category_significant, key=lambda x: abs(x[1]), reverse=True)[:3]:
                    direction = "↗️ 높음" if change_pct > 0 else "↘️ 낮음"
                    display_name = feature.split('.')[-1] if '.' in feature else feature
                    print(f"      • {display_name}: {direction} ({change_pct:+.1f}%, {intensity} 차이)")
                
                all_significant_differences.extend(category_significant)
        
        # 전체 종합 분석
        print(f"\n" + "="*85)
        print(f"📈 종합 분석 결과 ({total_analyzed_features}개 특징 분석)")
        print("="*85)
        
        if all_significant_differences:
            # 상위 차이점들
            print(f"\n🏆 가장 큰 차이를 보이는 특징들 (Top 10):")
            top_differences = sorted(all_significant_differences, key=lambda x: abs(x[1]), reverse=True)[:10]
            
            for i, (feature, change_pct, diff, intensity) in enumerate(top_differences, 1):
                direction = "↗️" if change_pct > 0 else "↘️"
                display_name = feature.split('.')[-1] if '.' in feature else feature
                print(f"   {i:2d}. {display_name:<25} {direction} {change_pct:>+7.1f}% ({intensity})")
            
            # 카테고리별 영향도 분석
            print(f"\n📊 카테고리별 특징 변화 요약:")
            category_impact = {}
            for feature, change_pct, diff, intensity in all_significant_differences:
                for category, features in feature_categories.items():
                    if feature in features:
                        if category not in category_impact:
                            category_impact[category] = []
                        category_impact[category].append(abs(change_pct))
                        break
            
            for category, impacts in category_impact.items():
                avg_impact = sum(impacts) / len(impacts)
                max_impact = max(impacts)
                print(f"   {category:<20} 평균 변화: {avg_impact:6.1f}% | 최대: {max_impact:6.1f}% | 영향 특징: {len(impacts)}개")
            
            # 그룹 성향 분석
            print(f"\n🎯 필터링된 그룹의 주요 성향:")
            
            # 성과 관련 특징들 분석
            performance_features = [f for f in all_significant_differences if any(x in f[0] for x in ['change.price', 'volatility.max', 'strength.'])]
            if performance_features:
                avg_performance_change = sum(abs(f[1]) for f in performance_features) / len(performance_features)
                if avg_performance_change > 25:
                    performance_tendency = "고성과 추구형"
                elif avg_performance_change > 15:
                    performance_tendency = "성과 지향형"
                elif avg_performance_change > 10:
                    performance_tendency = "균형형"
                else:
                    performance_tendency = "안정 추구형"
                print(f"   • 성과 성향: {performance_tendency} (평균 변화폭 {avg_performance_change:.1f}%)")
            
            # 위험 관련 특징들 분석
            risk_features = [f for f in all_significant_differences if any(x in f[0] for x in ['shape.noise', 'strength.direction', 'volatility.'])]
            if risk_features:
                # shape.noise_count는 낮을수록 좋고, strength.direction_pct는 높을수록 안정적
                risk_score = 0
                for feature, change_pct, _, _ in risk_features:
                    if 'noise' in feature:
                        risk_score -= change_pct  # 노이즈는 반대로
                    elif 'direction' in feature:
                        risk_score += change_pct  # 방향성은 그대로
                    else:
                        risk_score -= abs(change_pct)  # 변동성은 일반적으로 위험
                
                avg_risk_score = risk_score / len(risk_features)
                if avg_risk_score > 10:
                    risk_tendency = "저위험 선호형"
                elif avg_risk_score > 0:
                    risk_tendency = "안정 지향형"
                elif avg_risk_score > -10:
                    risk_tendency = "중립형"
                else:
                    risk_tendency = "고위험 감수형"
                print(f"   • 위험 성향: {risk_tendency} (종합 지수 {avg_risk_score:+.1f})")
            
        else:
            print("\n   📊 이 그룹은 전체 평균과 유사한 특성을 보입니다.")
            print("   💡 더 세밀한 조건으로 필터링하면 뚜렷한 차이를 발견할 수 있습니다.")

    def _create_radar_profile(self, filtered_df: pd.DataFrame):
        """레이더 차트로 그룹 프로필 시각화 (새로운 특징명 적용)"""
        print("\n🕸️ 종합 프로필 레이더 차트 생성 중...")
        
        # 레이더 차트용 특징 선택 (새로운 중첩 구조에서 선택)
        radar_features = []
        candidate_features = [
            'change.price_pct', 'shape.duration_candles', 'strength.direction_pct', 
            'shape.noise_count', 'start.rsi', 'change.rsi', 'volatility.max_high_pct'
        ]
        
        # 대안 특징명도 고려
        for feature in candidate_features:
            if feature in filtered_df.columns and feature in self.master_df.columns:
                radar_features.append(feature)
            else:
                # 유사한 특징 찾기
                alt_features = [f for f in self.numeric_features if any(part in f.lower() for part in feature.split('.')) and f in filtered_df.columns]
                if alt_features:
                    radar_features.append(alt_features[0])
            
            if len(radar_features) >= 7:  # 최대 7개
                break
        
        if len(radar_features) < 3:
            print("   레이더 차트를 위한 특징이 부족합니다.")
            return
        
        try:
            import numpy as np
            
            # 데이터 준비
            filtered_values = []
            total_values = []
            
            for feature in radar_features:
                # 정규화: 0-1 범위로 스케일링 (min-max scaling)
                total_data = self.master_df[feature].dropna()
                min_val, max_val = total_data.min(), total_data.max()
                
                if max_val > min_val:  # 분모가 0이 아닌 경우만
                    filtered_norm = (filtered_df[feature].mean() - min_val) / (max_val - min_val)
                    total_norm = (total_data.mean() - min_val) / (max_val - min_val)
                else:
                    filtered_norm = total_norm = 0.5  # 중앙값
                
                filtered_values.append(max(0, min(1, filtered_norm)))  # 0-1 범위 제한
                total_values.append(max(0, min(1, total_norm)))
            
            # 레이더 차트 생성
            angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
            angles += angles[:1]  # 원을 닫기 위해 첫 번째 각도 추가
            
            filtered_values += filtered_values[:1]  # 원을 닫기 위해 첫 번째 값 추가
            total_values += total_values[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # 전체 평균 (기준선)
            ax.plot(angles, total_values, 'o-', linewidth=2, label='전체 평균', color='gray', alpha=0.7)
            ax.fill(angles, total_values, alpha=0.1, color='gray')
            
            # 필터링된 그룹
            ax.plot(angles, filtered_values, 'o-', linewidth=3, label='필터링 그룹', color='#2E86AB')
            ax.fill(angles, filtered_values, alpha=0.3, color='#2E86AB')
            
            # 축 레이블 설정 (특징명 간소화)
            display_labels = []
            for f in radar_features:
                if '.' in f:
                    parts = f.split('.')
                    simplified = f"{parts[0][:4]}.{parts[1][:8]}"  # 축약된 표시
                else:
                    simplified = f[:12]  # 길이 제한
                display_labels.append(simplified)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(display_labels, fontsize=10)
            
            # 그리드 및 범위 설정
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
            ax.grid(True)
            
            # 제목 및 범례
            plt.title('그룹 특성 프로필 (레이더 차트)\n전체 평균 대비 필터링 그룹의 특징', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            plt.tight_layout()
            plt.show()
            
            # DNA 해석 제공
            self._interpret_radar_profile(radar_features, filtered_values[:-1], total_values[:-1])
            
        except Exception as e:
            print(f"   레이더 차트 생성 오류: {e}")
    
    def _interpret_radar_profile(self, features: List[str], filtered_vals: List[float], total_vals: List[float]):
        """레이더 차트 해석 및 DNA 분석"""
        print("\n🧬 그룹 DNA 해석:")
        
        # 각 축별 차이 분석
        strong_features = []  # 전체 평균보다 강한 특징
        weak_features = []    # 전체 평균보다 약한 특징
        
        for i, feature in enumerate(features):
            diff = filtered_vals[i] - total_vals[i]
            display_name = feature.split('.')[-1] if '.' in feature else feature
            
            if diff > 0.2:  # 20% 이상 차이
                strength = "매우강함" if diff > 0.4 else "강함"
                strong_features.append((display_name, strength, diff))
            elif diff < -0.2:
                weakness = "매우약함" if diff < -0.4 else "약함"
                weak_features.append((display_name, weakness, abs(diff)))
        
        if strong_features:
            print("   🔥 강점 특징:")
            for feature, strength, diff in strong_features:
                print(f"      • {feature}: {strength} (+{diff*100:.0f}%)")
        
        if weak_features:
            print("   🔹 약점 특징:")
            for feature, weakness, diff in weak_features:
                print(f"      • {feature}: {weakness} (-{diff*100:.0f}%)")
        
        # 전체적인 성격 분석
        avg_strength = np.mean(filtered_vals) - np.mean(total_vals)
        if avg_strength > 0.15:
            overall_character = "공격적/고성과 추구형"
        elif avg_strength > 0.05:
            overall_character = "적극적/성장 지향형"  
        elif avg_strength > -0.05:
            overall_character = "균형/안정형"
        elif avg_strength > -0.15:
            overall_character = "보수적/안정 추구형"
        else:
            overall_character = "매우 보수적/저성과형"
        
        print(f"   🎭 전체적 성격: {overall_character}")
        
        # 투자 시사점
        if len(strong_features) > len(weak_features) and any('price_change' in f[0] or 'price' in f[0] for f in strong_features):
            insight = "💡 고수익 잠재력이 높은 우량 조건으로 보입니다."
        elif any(f[0] in ['noise_count', 'noise'] for f in weak_features) and any('direction' in f[0] for f in strong_features):
            insight = "💡 깔끔하고 일관성 있는 추세를 나타내는 조건입니다."
        elif len(weak_features) > len(strong_features):
            insight = "💡 보수적이고 안정적인 특성을 가진 조건입니다."
        else:
            insight = "💡 평균적인 특성을 가진 조건으로, 추가 분석이 필요합니다."
        
        print(f"   {insight}")

    def _comprehensive_statistics(self, filtered_df: pd.DataFrame, applied_conditions: List[Dict] = None):
        """3단계 체계적 분석 리포트 (새로운 특징명 적용) + 새로운 기능 추가"""
        total_count = len(self.master_df)
        found_count = len(filtered_df)
        
        if found_count == 0:
            print("❌ 분석할 데이터가 없습니다.")
            return
        
        print("\n" + "="*90)
        print("📋 필터링된 사이클 그룹 분석 리포트")
        print("="*90)
        
        # 1단계: 요약 정보 (Summary)
        self._analysis_summary(filtered_df, applied_conditions, total_count, found_count)
        
        # 2단계: 성과 분석 (Performance Analysis)  
        self._performance_analysis_detailed(filtered_df)
        
        # 3단계: 그룹 프로파일링 (Profiling)
        self._group_profiling_analysis(filtered_df)
        
        # 시간대 정보 및 날짜 목록 출력
        filtered_dates = self._show_cycle_timeframe(filtered_df)
        
        # 4단계: 가격변화율 및 극값 시각화
        self._visualize_price_statistics(filtered_df)
        
        # 5단계: 상위 타임프레임 분석 (옵션)
        if filtered_dates and self.higher_timeframe_paths:
            if input(f"\n🔍 상위 타임프레임({', '.join(self.higher_timeframe_paths.keys())}) 분석을 하시겠습니까? (y/n): ").lower() == 'y':
                self.analyze_higher_timeframes(filtered_dates)
        
        # 6단계: 개별 사이클 조회 (옵션)
        if input("\n개별 사이클 상세 정보를 보시겠습니까? (y/n): ").lower() == 'y':
            self._show_individual_cycles(filtered_df)

    def run_analysis(self):
        """메인 분석 실행 - 체계적 분석 + 시각화 + 개별조회"""
        print("\n🚀 고급 사이클 분석기")
        print(f"📊 로딩된 데이터: {len(self.master_df):,}개 사이클")
        print("🎯 체계적 분석: 요약 → 성과 → 프로파일링 → 시간대 → 시각화 → 개별조회")
        
        try:
            while True:
                # 1. 조건 입력
                conditions = self._get_user_conditions()
                filtered_df = self._filter_cycles(conditions)
                
                if len(filtered_df) == 0:
                    print("❌ 조건을 만족하는 사이클이 없습니다.")
                    continue
                
                # 2. 체계적 분석 리포트 (기존 3단계 + 시간대 + 시각화 + 개별조회)
                self._comprehensive_statistics(filtered_df, conditions)
                
                # 계속 여부
                if input("\n다른 조건으로 분석하시겠습니까? (y/n): ").lower() != 'y':
                    break
                    
        except KeyboardInterrupt:
            print("\n분석 중단")
        finally:
            print("✅ 분석 완료")


if __name__ == '__main__':
    file_path = Path("./data/cycle_data/structured/cycles_4h.parquet")
    try:
        analyzer = AdvancedCycleAnalyzer(file_path)
        analyzer.run_analysis()
    except Exception as e:
        print(f"❌ 오류: {e}")
        input("엔터키로 종료...")