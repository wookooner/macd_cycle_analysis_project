"""
Multi-Timeframe Cycle Analysis (Improved Version)
상위 사이클의 특성을 하위 사이클 시퀀스 패턴으로 정의하는 종합 분석 도구

작성자: AI Assistant
날짜: 2025-09-06
목적: 상위 시간대(A) 사이클의 특성을 하위 시간대(B) 사이클들의 연속적인 패턴으로 분석
개선사항:
- 인터랙티브 인터페이스 추가
- avg_down_price_change_pct 음수 표현
- price_extremes 에러 해결
- dominant_direction 판단 기준 개선
- macd_hist 데이터 처리 개선
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class HierarchicalCycleAnalyzer:
    """
    계층적 사이클 분석기
    상위 시간대 사이클을 하위 시간대 사이클 시퀀스로 분석
    """
    
    def __init__(self, data_path: str = "data/cycle_data/structured"):
        """
        초기화
        
        Args:
            data_path: 사이클 데이터 파일들이 저장된 경로
        """
        self.data_path = Path(data_path)
        self.timeframe_hierarchy = {
            '1w': '1d',
            '1d': '4h', 
            '4h': '1h',
            '1h': '1m'
        }
        self.timeframe_seconds = {
            '1m': 60,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400,
            '1w': 604800
        }
        self.loaded_data = {}
        
    def load_cycle_data(self, timeframe: str) -> pd.DataFrame:
        """
        특정 시간대의 사이클 데이터 로딩 (캐싱 지원)
        
        Args:
            timeframe: 시간대 ('1h', '4h', '1d', '1w')
            
        Returns:
            사이클 데이터 DataFrame
        """
        if timeframe in self.loaded_data:
            return self.loaded_data[timeframe]
            
        file_path = self.data_path / f"cycles_{timeframe}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"사이클 데이터 파일이 없습니다: {file_path}")
            
        print(f"📊 {timeframe} 사이클 데이터 로딩 중...")
        df = pd.read_parquet(file_path)
        
        # 시간 컬럼을 datetime으로 변환 - 더 안전한 변환
        try:
            # start_date가 문자열인 경우
            if df['start_date'].dtype == 'object':
                df['start_datetime'] = pd.to_datetime(df['start_date'])
                df['end_datetime'] = pd.to_datetime(df['end_date'])
            else:
                # Unix timestamp인 경우
                df['start_datetime'] = pd.to_datetime(df['start_date'], unit='s')
                df['end_datetime'] = pd.to_datetime(df['end_date'], unit='s')
        except Exception as e:
            print(f"⚠️ 날짜 변환 오류: {e}")
            # 대체 방법 시도
            df['start_datetime'] = pd.to_datetime(df['start_date'].astype(float), unit='s')
            df['end_datetime'] = pd.to_datetime(df['end_date'].astype(float), unit='s')
        
        self.loaded_data[timeframe] = df
        print(f"✅ {timeframe} 데이터 로딩 완료: {len(df):,}개 사이클")
        
        return df
    
    def get_available_timeframes(self) -> List[str]:
        """사용 가능한 시간대 목록 반환"""
        available = []
        for timeframe in ['1m', '1h', '4h', '1d', '1w']:
            file_path = self.data_path / f"cycles_{timeframe}.parquet"
            if file_path.exists():
                available.append(timeframe)
        return available
    
    def interactive_timeframe_selection(self) -> Tuple[str, str]:
        """
        인터랙티브 시간대 선택
        
        Returns:
            (상위 시간대, 하위 시간대) 튜플
        """
        available = self.get_available_timeframes()
        
        print("\n" + "="*60)
        print("📊 계층적 사이클 분석 - 시간대 선택")
        print("="*60)
        print(f"사용 가능한 시간대: {', '.join(available)}")
        
        # 상위 시간대 선택
        upper_options = [tf for tf in self.timeframe_hierarchy.keys() if tf in available]
        
        print(f"\n상위 시간대 선택 가능 옵션:")
        for i, tf in enumerate(upper_options, 1):
            print(f"  {i}. {tf}")
        
        while True:
            try:
                choice = input(f"\n상위 시간대를 선택하세요 (1-{len(upper_options)}): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(upper_options):
                    upper_timeframe = upper_options[int(choice) - 1]
                    break
                print("올바른 번호를 입력해주세요.")
            except KeyboardInterrupt:
                print("\n취소되었습니다.")
                exit()
        
        # 하위 시간대 자동 결정
        lower_timeframe = self.timeframe_hierarchy[upper_timeframe]
        
        if lower_timeframe not in available:
            print(f"⚠️ 하위 시간대 {lower_timeframe} 데이터가 없습니다.")
            return None, None
        
        print(f"\n✅ 선택된 분석 경로: {upper_timeframe} → {lower_timeframe}")
        
        return upper_timeframe, lower_timeframe
    
    def explore_cycle_data_structure(self, timeframe: str) -> Dict[str, Any]:
        """
        사이클 데이터 구조 탐색
        
        Args:
            timeframe: 대상 시간대
            
        Returns:
            데이터 구조 정보 딕셔너리
        """
        df = self.load_cycle_data(timeframe)
        
        structure = {
            'basic_columns': [],
            'cycle_features_fields': {},
            'sample_cycle_features': None
        }
        
        # 기본 컬럼들
        structure['basic_columns'] = list(df.columns)
        
        # cycle_features 구조 탐색
        if 'cycle_features' in df.columns:
            sample_features = None
            for _, row in df.iterrows():
                try:
                    if pd.notna(row['cycle_features']) and isinstance(row['cycle_features'], dict):
                        sample_features = row['cycle_features']
                        break
                except:
                    continue
            
            if sample_features:
                structure['sample_cycle_features'] = sample_features
                structure['cycle_features_fields'] = self._flatten_dict(sample_features, 'cycle_features')
        
        return structure
    
    def _flatten_dict(self, d: Dict, prefix: str = '') -> Dict[str, Any]:
        """중첩된 딕셔너리를 평면화"""
        items = []
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, type(v).__name__))
        return dict(items)
    
    def interactive_filter_setup(self, timeframe: str) -> Dict[str, Any]:
        """
        인터랙티브 필터 설정 (개선된 버전)
        
        Args:
            timeframe: 대상 시간대
            
        Returns:
            필터 조건 딕셔너리
        """
        print("\n" + "="*60)
        print("🔍 사이클 필터링 조건 설정")
        print("="*60)
        
        # 데이터 구조 탐색
        print("📊 데이터 구조 탐색 중...")
        structure = self.explore_cycle_data_structure(timeframe)
        
        filters = {}
        
        # 사용 가능한 필드들 표시
        print(f"\n📋 사용 가능한 필터링 필드:")
        print(f"   기본 컬럼: {', '.join(structure['basic_columns'])}")
        
        if structure['cycle_features_fields']:
            print(f"   cycle_features 필드:")
            for field, field_type in structure['cycle_features_fields'].items():
                print(f"      - {field} ({field_type})")
        
        print(f"\n💡 필터링 방법:")
        print(f"   - 단일 값: 'field_name > 10' 또는 'field_name == up'")
        print(f"   - 범위 값: 'field_name 10-20' (10 < field_name < 20)")
        print(f"   - 절댓값: 'field_name abs> 5' (|field_name| > 5)")
        print(f"   - 종료: 'done' 입력")
        
        # 필터 입력 루프
        filter_count = 0
        while True:
            filter_count += 1
            print(f"\n--- 필터 {filter_count} ---")
            filter_input = input("필터 조건 입력 (종료: done): ").strip()
            
            if filter_input.lower() == 'done':
                break
            
            if not filter_input:
                continue
            
            # 필터 파싱
            parsed_filter = self._parse_filter_input(filter_input, structure)
            if parsed_filter:
                field, operator, value = parsed_filter
                filters[field] = (operator, value)
                print(f"✅ 추가됨: {field} {operator} {value}")
            else:
                print("❌ 잘못된 형식입니다. 다시 입력해주세요.")
        
        if not filters:
            print("⚠️ 필터가 설정되지 않았습니다. 기본 필터를 적용합니다.")
            filters = {
                'duration_candles': ('>', 5),
                'cycle_features.change.price_pct': ('abs_>', 1.0)
            }
        
        print(f"\n✅ 설정된 필터: {filters}")
        
        return filters
    
    def _parse_filter_input(self, input_str: str, structure: Dict[str, Any]) -> Optional[Tuple[str, str, Any]]:
        """
        필터 입력 문자열 파싱
        
        Args:
            input_str: 사용자 입력 문자열
            structure: 데이터 구조 정보
        
        Returns:
            (필드명, 연산자, 값) 튜플 또는 None
        """
        try:
            # 범위 필터 처리 (예: "price_pct 10-20")
            if ' ' in input_str and '-' in input_str:
                parts = input_str.split(' ')
                if len(parts) == 2:
                    field = parts[0]
                    range_str = parts[1]
                    if '-' in range_str:
                        min_val, max_val = map(float, range_str.split('-'))
                        return field, 'range', (min_val, max_val)
            
            # 단일 값 필터 처리
            operators = ['>=', '<=', '>', '<', '==', '!=', 'abs>', 'abs<', 'abs>=', 'abs<=']
            
            for op in operators:
                if f' {op} ' in input_str:
                    field, value_str = input_str.split(f' {op} ', 1)
                    field = field.strip()
                    value_str = value_str.strip()
                    
                    # 값 타입 변환
                    if value_str.lower() in ['true', 'false']:
                        value = value_str.lower() == 'true'
                    elif value_str.isdigit():
                        value = int(value_str)
                    elif value_str.replace('.', '').replace('-', '').isdigit():
                        value = float(value_str)
                    else:
                        value = value_str
                    
                    return field, op, value
            
            return None
            
        except Exception as e:
            return None
    
    
    def get_filtered_cycles(self, 
                          timeframe: str, 
                           filter_conditions: Dict[str, Any] = None) -> pd.DataFrame:
        """
        필터링된 모든 사이클 반환 (개선된 버전)
        
        Args:
            timeframe: 상위 시간대
            filter_conditions: 필터링 조건 딕셔너리
            
        Returns:
            필터링된 사이클들의 DataFrame
        """
        df = self.load_cycle_data(timeframe)
        
        # 기본 필터 조건 적용
        if filter_conditions is None:
            filter_conditions = {
                'duration_candles': ('>', 5),
                'cycle_features.change.price_pct': ('abs_>', 1.0)
            }
        
        filtered_df = df.copy()
        
        # 필터 조건 적용 (개선된 버전)
        for condition, (operator, value) in filter_conditions.items():
            if '.' in condition:  # nested field 처리
                if condition.startswith('cycle_features.'):
                    field_path = condition.split('.')[1:]
                    series_data = []
                    
                    for _, row in filtered_df.iterrows():
                        try:
                            data = row['cycle_features']
                            for field in field_path:
                                data = data[field]
                            series_data.append(data)
                        except (KeyError, TypeError):
                            series_data.append(None)
                    
                    series = pd.Series(series_data, index=filtered_df.index)
                    filtered_df = self._apply_filter_operator(filtered_df, series, operator, value)
            else:
                # 일반 컬럼 처리
                series = filtered_df[condition]
                filtered_df = self._apply_filter_operator(filtered_df, series, operator, value)
        
        if len(filtered_df) == 0:
            raise ValueError("필터 조건을 만족하는 사이클이 없습니다.")
        
        print(f"\n📊 필터링 결과: {len(filtered_df)}개 사이클 발견")
        
        # 시간순 정렬 (최신부터)
        filtered_df = filtered_df.sort_values('start_datetime', ascending=False)
        
        return filtered_df
    
    def _apply_filter_operator(self, df: pd.DataFrame, series: pd.Series, operator: str, value: Any) -> pd.DataFrame:
        """필터 연산자 적용"""
        try:
            if operator == '>':
                return df[series > value]
            elif operator == '<':
                return df[series < value]
            elif operator == '>=':
                return df[series >= value]
            elif operator == '<=':
                return df[series <= value]
            elif operator == '==':
                return df[series == value]
            elif operator == '!=':
                return df[series != value]
            elif operator == 'abs>':
                return df[abs(series) > value]
            elif operator == 'abs<':
                return df[abs(series) < value]
            elif operator == 'abs>=':
                return df[abs(series) >= value]
            elif operator == 'abs<=':
                return df[abs(series) <= value]
            elif operator == 'range':
                min_val, max_val = value
                return df[(series > min_val) & (series < max_val)]
            else:
                return df
        except Exception as e:
            print(f"⚠️ 필터 적용 오류 ({operator}): {e}")
            return df
    
    def extract_sub_cycles(self, 
                          target_cycle: pd.Series, 
                          sub_timeframe: str) -> pd.DataFrame:
        """
        상위 사이클 기간 내 포함되는 하위 사이클 추출
        
        Args:
            target_cycle: 상위 사이클 데이터
            sub_timeframe: 하위 시간대
            
        Returns:
            포함되는 하위 사이클들의 DataFrame
        """
        sub_df = self.load_cycle_data(sub_timeframe)
        
        # 시간 범위 필터링 (일부라도 겹치는 사이클 포함)
        start_time = target_cycle['start_datetime']
        end_time = target_cycle['end_datetime']
        
        # 조건: 하위 사이클의 끝이 상위 사이클 시작 이후이고,
        #       하위 사이클의 시작이 상위 사이클 끝 이전인 경우
        mask = (sub_df['end_datetime'] >= start_time) & (sub_df['start_datetime'] <= end_time)
        
        overlapping_cycles = sub_df[mask].copy()
        
        # 시간순 정렬
        overlapping_cycles = overlapping_cycles.sort_values('start_datetime')
        
        print(f"📋 {sub_timeframe} 하위 사이클 {len(overlapping_cycles)}개 추출됨")
        
        return overlapping_cycles
    
    def analyze_upper_cycle_profile(self, target_cycle: pd.Series) -> Dict[str, Any]:
        """
        상위 사이클(A) 프로파일 분석
        
        Args:
            target_cycle: 분석 대상 상위 사이클
            
        Returns:
            상위 사이클 프로파일 딕셔너리
        """
        features = target_cycle['cycle_features']
        
        # 실제 시간 계산
        duration_seconds = (target_cycle['end_datetime'] - target_cycle['start_datetime']).total_seconds()
        duration_days = duration_seconds / 86400
        duration_hours = duration_seconds / 3600
        
        profile = {
            'basic_info': {
                'cycle_id': target_cycle['cycle_id'],
                'timeframe': target_cycle['timeframe'],
                'cycle_type': target_cycle['cycle_type'],
                'start_date': target_cycle['start_datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': target_cycle['end_datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'duration_candles': target_cycle['duration_candles'],
                'duration_days': round(duration_days, 2),
                'duration_hours': round(duration_hours, 2)
            },
            'price_movement': {
                'price_change_pct': features['change']['price_pct'],
                'start_price': features['start']['price'],
                'end_price': features['end']['price'],
                'max_high_pct': features.get('volatility', {}).get('max_high_pct', 0),
                'max_loss_pct': features.get('volatility', {}).get('max_loss_pct', 0)
            },
            'technical_indicators': {
                'start_rsi': features['start']['rsi'],
                'end_rsi': features['end']['rsi'],
                'rsi_change': features['change']['rsi'],
                'start_macd': features['start']['macd'],
                'end_macd': features['end']['macd'],
                'macd_change': features['change']['macd'],
                'start_hist': features['start']['hist'],
                'end_hist': features['end']['hist'],
                'hist_change': features['change']['hist']
            },
            'cycle_structure': {
                'core_count': features['shape']['core_count'],
                'noise_count': features['shape']['noise_count'],
                'direction_pct': features['strength']['direction_pct'],
                'hist_positive_ratio': features['strength']['hist_positive_ratio']
            }
        }
        
        return profile
    
    def analyze_sub_cycle_sequence(self, 
                                 sub_cycles: pd.DataFrame, 
                                 target_cycle: pd.Series) -> Dict[str, Any]:
        """
        하위 사이클(B) 시퀀스 분석
        
        Args:
            sub_cycles: 하위 사이클들의 DataFrame
            target_cycle: 상위 사이클 데이터
            
        Returns:
            하위 사이클 시퀀스 분석 결과 딕셔너리
        """
        if len(sub_cycles) == 0:
            return {'error': '분석할 하위 사이클이 없습니다.'}
        
        # 상승/하락 사이클 분리
        up_cycles = sub_cycles[sub_cycles['cycle_type'] == 'up']
        down_cycles = sub_cycles[sub_cycles['cycle_type'] == 'down']
        
        analysis = {
            'composition': self._analyze_composition(sub_cycles, up_cycles, down_cycles),
            'duration': self._analyze_duration(sub_cycles, up_cycles, down_cycles),
            'magnitude': self._analyze_magnitude(sub_cycles, up_cycles, down_cycles),
            'structure': self._analyze_structure(sub_cycles, target_cycle)
        }
        
        return analysis
    
    def _analyze_composition(self, 
                           all_cycles: pd.DataFrame, 
                           up_cycles: pd.DataFrame, 
                           down_cycles: pd.DataFrame) -> Dict[str, Any]:
        """사이클 구성 분석"""
        total_count = len(all_cycles)
        up_count = len(up_cycles)
        down_count = len(down_cycles)
        
        return {
            'total_cycle_count': total_count,
            'up_cycle_count': up_count,
            'down_cycle_count': down_count,
            'up_cycle_ratio': up_count / total_count if total_count > 0 else 0,
            'down_cycle_ratio': down_count / total_count if total_count > 0 else 0
        }
    
    def _analyze_duration(self, 
                        all_cycles: pd.DataFrame, 
                        up_cycles: pd.DataFrame, 
                        down_cycles: pd.DataFrame) -> Dict[str, Any]:
        """기간 분석"""
        total_duration = all_cycles['duration_candles'].sum()
        up_duration = up_cycles['duration_candles'].sum()
        down_duration = down_cycles['duration_candles'].sum()
        
        return {
            'total_duration_candles': total_duration,
            'up_duration_candles': up_duration,
            'down_duration_candles': down_duration,
            'up_duration_ratio': up_duration / total_duration if total_duration > 0 else 0,
            'down_duration_ratio': down_duration / total_duration if total_duration > 0 else 0,
            'avg_cycle_duration': all_cycles['duration_candles'].mean(),
            'avg_up_cycle_duration': up_cycles['duration_candles'].mean() if len(up_cycles) > 0 else 0,
            'avg_down_cycle_duration': down_cycles['duration_candles'].mean() if len(down_cycles) > 0 else 0,
            'duration_std': all_cycles['duration_candles'].std()
        }
    
    def _analyze_magnitude(self, 
                         all_cycles: pd.DataFrame, 
                         up_cycles: pd.DataFrame, 
                         down_cycles: pd.DataFrame) -> Dict[str, Any]:
        """강도 분석 (개선된 버전)"""
        # 가격 변화율 추출
        all_price_changes = []
        up_price_changes = []
        down_price_changes = []
        
        for _, cycle in all_cycles.iterrows():
            try:
                price_change = cycle['cycle_features']['change']['price_pct']
                all_price_changes.append(price_change)
                
                if cycle['cycle_type'] == 'up':
                    up_price_changes.append(price_change)
                else:
                    # 하락 사이클의 경우 음수로 표현
                    down_price_changes.append(abs(price_change) * -1)
            except:
                continue
        
        # MACD 히스토그램 분포 분석 (개선)
        hist_analysis = self._analyze_macd_histogram_distribution(all_cycles)
        
        # 총 가격 변화 계산 (dominant_direction 판단용)
        total_up_movement = sum(up_price_changes) if up_price_changes else 0
        total_down_movement = sum(down_price_changes) if down_price_changes else 0
        net_movement = total_up_movement + total_down_movement  # down은 이미 음수
        
        result = {
            'avg_price_change_pct': np.mean(all_price_changes) if all_price_changes else 0,
            'avg_up_price_change_pct': np.mean(up_price_changes) if up_price_changes else 0,
            'avg_down_price_change_pct': np.mean(down_price_changes) if down_price_changes else 0,  # 음수로 표현
            'total_up_price_change_pct': total_up_movement,
            'total_down_price_change_pct': total_down_movement,  # 음수로 표현
            'net_price_movement': net_movement,
            'price_change_std': np.std(all_price_changes) if all_price_changes else 0,
            'strongest_up_change': max(up_price_changes) if up_price_changes else 0,
            'strongest_down_change': min(down_price_changes) if down_price_changes else 0  # 가장 큰 하락(음수)
        }
        
        result.update(hist_analysis)
        
        return result
    
    def _analyze_macd_histogram_distribution(self, cycles: pd.DataFrame) -> Dict[str, Any]:
        """MACD 히스토그램 분포 분석 (개선된 버전)"""
        all_hist_values = []
        positive_hist_count = 0
        negative_hist_count = 0
        
        for _, cycle in cycles.iterrows():
            try:
                candle_data = cycle['candle_data']
                
                # numpy array인 경우 처리
                if isinstance(candle_data, np.ndarray):
                    for candle in candle_data:
                        if isinstance(candle, dict):
                            hist_val = candle.get('macd_hist', 0)
                        else:
                            # structured array인 경우
                            try:
                                hist_val = candle['macd_hist']
                            except:
                                hist_val = 0
                        
                        if hist_val != 0:  # 0이 아닌 값만 수집
                            all_hist_values.append(hist_val)
                            
                            if hist_val > 0:
                                positive_hist_count += 1
                            elif hist_val < 0:
                                negative_hist_count += 1
                                
                # list인 경우 처리                
                elif isinstance(candle_data, list):
                    for candle in candle_data:
                        hist_val = candle.get('macd_hist', 0)
                        if hist_val != 0:
                            all_hist_values.append(hist_val)
                            
                            if hist_val > 0:
                                positive_hist_count += 1
                            elif hist_val < 0:
                                negative_hist_count += 1
            except Exception as e:
                continue
        
        total_hist_count = len(all_hist_values)
        
        return {
            'macd_hist_positive_ratio': positive_hist_count / total_hist_count if total_hist_count > 0 else 0,
            'macd_hist_negative_ratio': negative_hist_count / total_hist_count if total_hist_count > 0 else 0,
            'avg_macd_hist': np.mean(all_hist_values) if all_hist_values else 0,
            'macd_hist_std': np.std(all_hist_values) if all_hist_values else 0,
            'macd_hist_data_count': total_hist_count
        }
    
    def _analyze_structure(self, 
                         sub_cycles: pd.DataFrame, 
                         target_cycle: pd.Series) -> Dict[str, Any]:
        """구조 분석 (개선된 버전)"""
        # 총 노이즈 개수
        total_noise = 0
        for _, cycle in sub_cycles.iterrows():
            try:
                noise_count = cycle['cycle_features']['shape']['noise_count']
                total_noise += noise_count
            except:
                continue
        
        # 가격 극점 위치 분석 (개선)
        extremes = self._find_price_extremes(sub_cycles, target_cycle)
        
        return {
            'total_noise_count': total_noise,
            'avg_noise_per_cycle': total_noise / len(sub_cycles) if len(sub_cycles) > 0 else 0,
            'price_extremes': extremes
        }
    
    def _find_price_extremes(self, 
                           sub_cycles: pd.DataFrame, 
                           target_cycle: pd.Series) -> Dict[str, Any]:
        """가격 극점 위치 찾기 (디버깅 강화 버전)"""
        all_prices = []
        debug_info = []
        
        target_start = target_cycle['start_datetime']
        target_duration = (target_cycle['end_datetime'] - target_start).total_seconds()
        
        print(f"🔍 가격 극점 분석 시작 - 하위 사이클 {len(sub_cycles)}개")
        
        for cycle_idx, (_, cycle) in enumerate(sub_cycles.iterrows()):
            try:
                candle_data = cycle['candle_data']
                debug_info.append(f"사이클 {cycle_idx}: candle_data 타입 = {type(candle_data)}")
                
                # candle_data가 None이거나 비어있는 경우
                if candle_data is None or (hasattr(candle_data, '__len__') and len(candle_data) == 0):
                    debug_info.append(f"사이클 {cycle_idx}: candle_data가 비어있음")
                    continue
                
                # 다양한 데이터 타입 처리
                if isinstance(candle_data, (list, tuple, np.ndarray)):
                    debug_info.append(f"사이클 {cycle_idx}: list/tuple/numpy 처리, 길이 = {len(candle_data)}")
                    for candle_idx, candle in enumerate(candle_data):
                        try:
                            if isinstance(candle, dict):
                                timestamp_str = str(candle.get('timestamp', ''))
                                high = float(candle.get('high', 0))
                                low = float(candle.get('low', 0))
                                
                                # timestamp 변환 (문자열 timestamp 처리)
                                timestamp = self._safe_timestamp_convert(timestamp_str)
                                
                            else:
                                # 다른 타입의 경우 속성 접근 시도
                                timestamp_str = str(getattr(candle, 'timestamp', ''))
                                high = float(getattr(candle, 'high', 0))
                                low = float(getattr(candle, 'low', 0))
                                
                                # timestamp 변환 (문자열 timestamp 처리)
                                timestamp = self._safe_timestamp_convert(timestamp_str)
                            
                            if high > 0 and low > 0:
                                all_prices.append({
                                    'timestamp': timestamp,
                                    'high': high,
                                    'low': low
                                })
                        except Exception as e:
                            debug_info.append(f"사이클 {cycle_idx}, 캔들 {candle_idx}: 파싱 오류 - {e}")
                            continue
                            
                
                else:
                    debug_info.append(f"사이클 {cycle_idx}: 알 수 없는 candle_data 타입 - {type(candle_data)}")
                    
            except Exception as e:
                debug_info.append(f"사이클 {cycle_idx}: 전체 처리 오류 - {e}")
                continue
        
        print(f"📊 가격 데이터 수집 결과: {len(all_prices)}개 캔들")
        if len(all_prices) == 0:
            print("⚠️ 디버깅 정보:")
            for info in debug_info[:10]:  # 처음 10개만 출력
                print(f"   {info}")
            
            # 첫 번째 사이클의 candle_data 샘플 출력
            if len(sub_cycles) > 0:
                sample_cycle = sub_cycles.iloc[0]
                sample_candle_data = sample_cycle['candle_data']
                print(f"   첫 번째 사이클 candle_data 샘플:")
                print(f"   - 타입: {type(sample_candle_data)}")
                if hasattr(sample_candle_data, '__len__'):
                    print(f"   - 길이: {len(sample_candle_data)}")
                    if len(sample_candle_data) > 0:
                        print(f"   - 첫 번째 캔들: {sample_candle_data[0]}")
                        print(f"   - 첫 번째 캔들 타입: {type(sample_candle_data[0])}")
                        if hasattr(sample_candle_data[0], 'dtype'):
                            print(f"   - 첫 번째 캔들 dtype: {sample_candle_data[0].dtype}")
        
        # candle_data에서 가격을 찾지 못한 경우, cycle_features에서 시도
        if not all_prices:
            print("⚠️ candle_data에서 가격을 찾지 못함. cycle_features에서 시도...")
            all_prices = self._extract_prices_from_cycle_features(sub_cycles, target_cycle)
        
        if not all_prices:
            return {
                'error': '가격 데이터를 찾을 수 없습니다.',
                'highest_price': 0,
                'highest_price_timestamp': 'N/A',
                'highest_price_position': 0,
                'lowest_price': 0,
                'lowest_price_timestamp': 'N/A',
                'lowest_price_position': 0,
                'debug_info': debug_info[:5]  # 디버깅 정보 포함
            }
        
        # 최고점과 최저점 찾기
        max_price_data = max(all_prices, key=lambda x: x['high'])
        min_price_data = min(all_prices, key=lambda x: x['low'])
        
        # 상대적 위치 계산 (0.0 ~ 1.0)
        try:
            high_position = (max_price_data['timestamp'] - target_start).total_seconds() / target_duration
            low_position = (min_price_data['timestamp'] - target_start).total_seconds() / target_duration
            
            # 위치 값 범위 제한
            high_position = max(0, min(1, high_position))
            low_position = max(0, min(1, low_position))
        except Exception as e:
            print(f"⚠️ 위치 계산 오류: {e}")
            high_position = 0.5
            low_position = 0.5
        
        print(f"✅ 가격 극점 발견 - 최고: ${max_price_data['high']:,.2f}, 최저: ${min_price_data['low']:,.2f}")
        
        return {
            'highest_price': max_price_data['high'],
            'highest_price_timestamp': max_price_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'highest_price_position': high_position,
            'lowest_price': min_price_data['low'],
            'lowest_price_timestamp': min_price_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'lowest_price_position': low_position,
            'price_range': max_price_data['high'] - min_price_data['low'],
            'price_range_pct': ((max_price_data['high'] - min_price_data['low']) / min_price_data['low'] * 100) if min_price_data['low'] > 0 else 0
        }
    
    def _extract_prices_from_cycle_features(self, 
                                          sub_cycles: pd.DataFrame, 
                                          target_cycle: pd.Series) -> List[Dict[str, Any]]:
        """cycle_features에서 가격 정보 추출 (대안 방법)"""
        all_prices = []
        
        target_start = target_cycle['start_datetime']
        target_duration = (target_cycle['end_datetime'] - target_start).total_seconds()
        
        print("🔍 cycle_features에서 가격 정보 추출 시도...")
        
        for cycle_idx, (_, cycle) in enumerate(sub_cycles.iterrows()):
            try:
                features = cycle['cycle_features']
                start_time = cycle['start_datetime']
                end_time = cycle['end_datetime']
                
                # cycle_features에서 가격 정보 추출 (다양한 필드명 시도)
                start_price = 0
                end_price = 0
                max_high_pct = 0
                max_loss_pct = 0
                
                # 가능한 가격 필드명들 시도
                price_fields = ['start_price', 'price_start', 'open_price', 'price_open']
                for field in price_fields:
                    if field in features and features[field] is not None:
                        start_price = float(features[field])
                        break
                
                price_fields = ['end_price', 'price_end', 'close_price', 'price_close']
                for field in price_fields:
                    if field in features and features[field] is not None:
                        end_price = float(features[field])
                        break
                
                # 최고가/최저가 퍼센트 필드들 시도
                high_fields = ['max_high_pct', 'max_high_change', 'highest_pct', 'max_gain_pct']
                for field in high_fields:
                    if field in features and features[field] is not None:
                        max_high_pct = float(features[field])
                        break
                
                low_fields = ['max_loss_pct', 'max_loss_change', 'lowest_pct', 'max_loss_pct']
                for field in low_fields:
                    if field in features and features[field] is not None:
                        max_loss_pct = float(features[field])
                        break
                
                # 디버깅 정보 출력 (첫 번째 사이클만)
                if cycle_idx == 0:
                    print(f"   첫 번째 사이클 cycle_features 필드들:")
                    for key, value in features.items():
                        if 'price' in key.lower() or 'high' in key.lower() or 'low' in key.lower():
                            print(f"     {key}: {value}")
                
                # 가격이 유효한지 확인
                if start_price > 0 and end_price > 0 and not np.isnan(start_price) and not np.isnan(end_price):
                    # 최고가와 최저가 계산
                    if max_high_pct > 0:
                        highest_price = start_price * (1 + max_high_pct / 100)
                    else:
                        highest_price = max(start_price, end_price)
                    
                    if max_loss_pct < 0:
                        lowest_price = start_price * (1 + max_loss_pct / 100)
                    else:
                        lowest_price = min(start_price, end_price)
                    
                    # 시간대별로 가격 정보 추가
                    cycle_duration = (end_time - start_time).total_seconds()
                    
                    # 시작점
                    all_prices.append({
                        'timestamp': start_time,
                        'high': highest_price,
                        'low': lowest_price
                    })
                    
                    # 중간점 (있는 경우)
                    if cycle_duration > 0:
                        mid_time = start_time + timedelta(seconds=cycle_duration / 2)
                        all_prices.append({
                            'timestamp': mid_time,
                            'high': highest_price,
                            'low': lowest_price
                        })
                    
                    # 끝점
                    all_prices.append({
                        'timestamp': end_time,
                        'high': highest_price,
                        'low': lowest_price
                    })
                    
            except Exception as e:
                print(f"⚠️ cycle_features에서 가격 추출 오류: {e}")
                continue
        
        print(f"📊 cycle_features에서 {len(all_prices)}개 가격 데이터 추출")
        return all_prices
    
    def _safe_timestamp_convert(self, timestamp_str: str) -> pd.Timestamp:
        """안전한 timestamp 변환 함수"""
        try:
            # 빈 문자열이나 None 처리
            if not timestamp_str or timestamp_str.strip() == '':
                return pd.Timestamp.now()
            
            # 숫자로 변환 시도
            try:
                timestamp_num = int(timestamp_str)
                # Unix timestamp (초)로 시도
                return pd.to_datetime(timestamp_num, unit='s')
            except (ValueError, OverflowError):
                try:
                    # Unix timestamp (밀리초)로 시도
                    return pd.to_datetime(timestamp_num, unit='ms')
                except (ValueError, OverflowError):
                    # 문자열로 직접 파싱 시도
                    return pd.to_datetime(timestamp_str)
        except Exception:
            # 모든 변환이 실패하면 현재 시간 사용
            return pd.Timestamp.now()
    
    def _generate_summary(self, 
                        upper_profile: Dict[str, Any], 
                        sub_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과 요약 생성 (개선된 버전)"""
        if 'error' in sub_analysis:
            return {'error': sub_analysis['error']}
        
        composition = sub_analysis['composition']
        duration = sub_analysis['duration']
        magnitude = sub_analysis['magnitude']
        
        # dominant_direction을 net_price_movement 기준으로 판단
        if magnitude['net_price_movement'] > 0:
            dominant_direction = 'up'
            direction_strength = abs(magnitude['total_up_price_change_pct']) / (abs(magnitude['total_up_price_change_pct']) + abs(magnitude['total_down_price_change_pct'])) if (magnitude['total_up_price_change_pct'] + abs(magnitude['total_down_price_change_pct'])) > 0 else 0
        else:
            dominant_direction = 'down'
            direction_strength = abs(magnitude['total_down_price_change_pct']) / (abs(magnitude['total_up_price_change_pct']) + abs(magnitude['total_down_price_change_pct'])) if (magnitude['total_up_price_change_pct'] + abs(magnitude['total_down_price_change_pct'])) > 0 else 0
        
        return {
            'upper_cycle_strength': {
                'direction': upper_profile['basic_info']['cycle_type'],
                'price_change': upper_profile['price_movement']['price_change_pct'],
                'duration_days': upper_profile['basic_info']['duration_days'],
                'efficiency': upper_profile['cycle_structure']['direction_pct']
            },
            'sub_cycle_pattern': {
                'total_cycles': composition['total_cycle_count'],
                'dominant_direction': dominant_direction,  # 개선: net movement 기준
                'direction_strength': direction_strength,  # 개선: 강도 기준
                'direction_consistency': max(composition['up_cycle_ratio'], composition['down_cycle_ratio']),
                'time_efficiency': duration['up_duration_ratio'] if upper_profile['basic_info']['cycle_type'] == 'up' else duration['down_duration_ratio'],
                'avg_sub_magnitude': magnitude['avg_price_change_pct'],
                'net_price_movement': magnitude['net_price_movement'],
                'momentum_alignment': magnitude['macd_hist_positive_ratio']
            },
            'pattern_quality_score': self._calculate_pattern_quality_score(upper_profile, sub_analysis)
        }
    
    def _calculate_pattern_quality_score(self, 
                                       upper_profile: Dict[str, Any], 
                                       sub_analysis: Dict[str, Any]) -> float:
        """패턴 품질 점수 계산 (0-100) - 개선된 버전"""
        if 'error' in sub_analysis:
            return 0.0
        
        try:
            # 상위 사이클 방향성
            upper_direction = upper_profile['basic_info']['cycle_type']
            
            # 하위 사이클 일관성 (net movement 기준)
            magnitude = sub_analysis['magnitude']
            if upper_direction == 'up':
                direction_alignment = magnitude['net_price_movement'] > 0
                strength = abs(magnitude['total_up_price_change_pct']) / max(abs(magnitude['total_up_price_change_pct']) + abs(magnitude['total_down_price_change_pct']), 1)
            else:
                direction_alignment = magnitude['net_price_movement'] < 0
                strength = abs(magnitude['total_down_price_change_pct']) / max(abs(magnitude['total_up_price_change_pct']) + abs(magnitude['total_down_price_change_pct']), 1)
            
            # 시간 효율성
            duration = sub_analysis['duration']
            time_efficiency = duration['up_duration_ratio'] if upper_direction == 'up' else duration['down_duration_ratio']
            
            # 모멘텀 일관성
            momentum_alignment = magnitude['macd_hist_positive_ratio'] if upper_direction == 'up' else (1 - magnitude['macd_hist_positive_ratio'])
            
            # 가중 평균 점수
            score = (
                (1.0 if direction_alignment else 0.5) * strength * 40 +
                time_efficiency * 30 +
                momentum_alignment * 20 +
                upper_profile['cycle_structure']['direction_pct'] * 10
            )
            
            return round(score * 100, 2)
            
        except Exception as e:
            print(f"점수 계산 오류: {e}")
            return 0.0
    
    def run_hierarchical_analysis(self, 
                                 upper_timeframe: str = None,
                                 filter_conditions: Dict[str, Any] = None,
                                 interactive: bool = True,
                                 save_results: bool = True) -> Dict[str, Any]:
        """
        계층적 사이클 분석 실행 (다중 사이클 분석 버전)
        
        Args:
            upper_timeframe: 상위 시간대 (None이면 인터랙티브)
            filter_conditions: 상위 사이클 필터링 조건 (None이면 인터랙티브)
            interactive: 인터랙티브 모드 여부
            save_results: 결과 저장 여부
            
        Returns:
            전체 분석 결과 딕셔너리
        """
        print(f"\n🚀 계층적 사이클 분석 시작 (다중 사이클 분석)")
        
        # 인터랙티브 모드
        if interactive:
            # 시간대 선택
            upper_timeframe, lower_timeframe = self.interactive_timeframe_selection()
            if not upper_timeframe:
                return {'error': '유효한 시간대를 선택할 수 없습니다.'}
            
            # 필터 설정
            filter_conditions = self.interactive_filter_setup(upper_timeframe)
        else:
            if upper_timeframe not in self.timeframe_hierarchy:
                raise ValueError(f"지원하지 않는 상위 시간대입니다: {upper_timeframe}")
            lower_timeframe = self.timeframe_hierarchy[upper_timeframe]
        
        print(f"\n   상위 시간대: {upper_timeframe}")
        print(f"   하위 시간대: {lower_timeframe}")
        
        # 1단계: 필터링된 모든 상위 사이클 가져오기
        try:
            filtered_cycles = self.get_filtered_cycles(upper_timeframe, filter_conditions)
        except ValueError as e:
            return {'error': str(e)}
        
        # 2단계: 각 사이클에 대해 분석 실행
        print(f"\n📊 {len(filtered_cycles)}개 사이클 분석 시작...")
        
        all_analyses = []
        for idx, (_, target_cycle) in enumerate(filtered_cycles.iterrows(), 1):
            print(f"\n--- 사이클 {idx}/{len(filtered_cycles)} 분석 중 ---")
            print(f"   ID: {target_cycle['cycle_id']}")
            print(f"   기간: {target_cycle['start_datetime']} ~ {target_cycle['end_datetime']}")
            print(f"   타입: {target_cycle['cycle_type'].upper()}")
            
            try:
                # 하위 사이클 추출
                sub_cycles = self.extract_sub_cycles(target_cycle, lower_timeframe)
                
                # 개별 사이클 분석
                upper_profile = self.analyze_upper_cycle_profile(target_cycle)
                sub_analysis = self.analyze_sub_cycle_sequence(sub_cycles, target_cycle)
        
                analysis_result = {
                    'cycle_id': target_cycle['cycle_id'],
                    'upper_cycle_profile': upper_profile,
                    'sub_cycle_sequence_analysis': sub_analysis,
                    'summary': self._generate_summary(upper_profile, sub_analysis)
                }
                
                all_analyses.append(analysis_result)
                print(f"   ✅ 분석 완료")
                
            except Exception as e:
                print(f"   ❌ 분석 실패: {e}")
                continue
        
        if not all_analyses:
            return {'error': '분석할 수 있는 사이클이 없습니다.'}
        
        # 3단계: 전체 결과 종합
        print(f"\n📊 전체 {len(all_analyses)}개 사이클 분석 완료")
        print(f"📈 종합 분석 실행 중...")
        
        # 종합 분석 결과 생성
        results = {
            'analysis_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'upper_timeframe': upper_timeframe,
                'lower_timeframe': lower_timeframe,
                'filter_conditions': filter_conditions,
                'total_cycles_analyzed': len(all_analyses),
                'total_cycles_filtered': len(filtered_cycles)
            },
            'individual_analyses': all_analyses,
            'comprehensive_analysis': self._generate_comprehensive_analysis(all_analyses)
        }
        
        # 결과 저장
        if save_results:
            self._save_results(results, upper_timeframe, lower_timeframe)
        
        return results
    
    def _generate_comprehensive_analysis(self, all_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        모든 개별 분석 결과를 종합한 분석 생성
        
        Args:
            all_analyses: 모든 개별 분석 결과 리스트
            
        Returns:
            종합 분석 결과 딕셔너리
        """
        if not all_analyses:
            return {'error': '분석할 데이터가 없습니다.'}
        
        # 기본 통계
        total_cycles = len(all_analyses)
        up_cycles = sum(1 for analysis in all_analyses 
                       if analysis['upper_cycle_profile']['basic_info']['cycle_type'] == 'up')
        down_cycles = total_cycles - up_cycles
        
        # 패턴 품질 점수 통계
        quality_scores = []
        for analysis in all_analyses:
            if 'error' not in analysis['summary']:
                quality_scores.append(analysis['summary']['pattern_quality_score'])
        
        # 상위 사이클 특성 통계
        upper_characteristics = self._analyze_upper_cycle_characteristics(all_analyses)
        
        # 하위 사이클 패턴 통계
        sub_patterns = self._analyze_sub_cycle_patterns(all_analyses)
        
        # 시간대별 성능 분석
        timeframe_performance = self._analyze_timeframe_performance(all_analyses)
        
        return {
            'overview': {
                'total_cycles_analyzed': total_cycles,
                'up_cycles_count': up_cycles,
                'down_cycles_count': down_cycles,
                'up_cycles_ratio': up_cycles / total_cycles if total_cycles > 0 else 0,
                'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
                'quality_score_std': np.std(quality_scores) if quality_scores else 0,
                'best_quality_score': max(quality_scores) if quality_scores else 0,
                'worst_quality_score': min(quality_scores) if quality_scores else 0
            },
            'upper_cycle_characteristics': upper_characteristics,
            'sub_cycle_patterns': sub_patterns,
            'timeframe_performance': timeframe_performance,
            'recommendations': self._generate_recommendations(all_analyses, quality_scores)
        }
    
    def _analyze_upper_cycle_characteristics(self, all_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """상위 사이클 특성 분석"""
        durations = []
        price_changes = []
        rsi_changes = []
        macd_changes = []
        
        for analysis in all_analyses:
            profile = analysis['upper_cycle_profile']
            durations.append(profile['basic_info']['duration_days'])
            price_changes.append(profile['price_movement']['price_change_pct'])
            rsi_changes.append(profile['technical_indicators']['rsi_change'])
            macd_changes.append(profile['technical_indicators']['macd_change'])
        
        return {
            'avg_duration_days': np.mean(durations),
            'duration_std': np.std(durations),
            'avg_price_change_pct': np.mean(price_changes),
            'price_change_std': np.std(price_changes),
            'avg_rsi_change': np.mean(rsi_changes),
            'avg_macd_change': np.mean(macd_changes),
            'strongest_price_change': max(price_changes) if price_changes else 0,
            'weakest_price_change': min(price_changes) if price_changes else 0
        }
    
    def _analyze_sub_cycle_patterns(self, all_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """하위 사이클 패턴 분석"""
        total_sub_cycles = 0
        total_up_sub_cycles = 0
        total_down_sub_cycles = 0
        all_net_movements = []
        all_direction_consistencies = []
        
        for analysis in all_analyses:
            sub_analysis = analysis['sub_cycle_sequence_analysis']
            if 'error' not in sub_analysis:
                comp = sub_analysis['composition']
                mag = sub_analysis['magnitude']
                
                total_sub_cycles += comp['total_cycle_count']
                total_up_sub_cycles += comp['up_cycle_count']
                total_down_sub_cycles += comp['down_cycle_count']
                
                all_net_movements.append(mag['net_price_movement'])
                all_direction_consistencies.append(max(comp['up_cycle_ratio'], comp['down_cycle_ratio']))
        
        return {
            'total_sub_cycles': total_sub_cycles,
            'avg_sub_cycles_per_upper': total_sub_cycles / len(all_analyses) if all_analyses else 0,
            'up_sub_cycles_ratio': total_up_sub_cycles / total_sub_cycles if total_sub_cycles > 0 else 0,
            'down_sub_cycles_ratio': total_down_sub_cycles / total_sub_cycles if total_sub_cycles > 0 else 0,
            'avg_net_movement': np.mean(all_net_movements),
            'net_movement_std': np.std(all_net_movements),
            'avg_direction_consistency': np.mean(all_direction_consistencies)
        }
    
    def _analyze_timeframe_performance(self, all_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """시간대별 성능 분석"""
        # 상위 시간대별 성능 분석
        timeframe_stats = {}
        
        for analysis in all_analyses:
            profile = analysis['upper_cycle_profile']
            timeframe = profile['basic_info']['timeframe']
            
            if timeframe not in timeframe_stats:
                timeframe_stats[timeframe] = {
                    'count': 0,
                    'quality_scores': [],
                    'price_changes': [],
                    'durations': []
                }
            
            timeframe_stats[timeframe]['count'] += 1
            if 'error' not in analysis['summary']:
                timeframe_stats[timeframe]['quality_scores'].append(
                    analysis['summary']['pattern_quality_score']
                )
            timeframe_stats[timeframe]['price_changes'].append(
                profile['price_movement']['price_change_pct']
            )
            timeframe_stats[timeframe]['durations'].append(
                profile['basic_info']['duration_days']
            )
        
        # 통계 계산
        performance = {}
        for timeframe, stats in timeframe_stats.items():
            performance[timeframe] = {
                'cycle_count': stats['count'],
                'avg_quality_score': np.mean(stats['quality_scores']) if stats['quality_scores'] else 0,
                'avg_price_change': np.mean(stats['price_changes']),
                'avg_duration_days': np.mean(stats['durations'])
            }
        
        return performance
    
    def _generate_recommendations(self, all_analyses: List[Dict[str, Any]], quality_scores: List[float]) -> List[str]:
        """분석 결과 기반 추천사항 생성"""
        recommendations = []
        
        if not quality_scores:
            return ["분석 데이터가 부족하여 추천사항을 생성할 수 없습니다."]
        
        avg_quality = np.mean(quality_scores)
        
        if avg_quality > 80:
            recommendations.append("🎯 높은 품질의 사이클 패턴이 발견되었습니다. 현재 전략을 유지하세요.")
        elif avg_quality > 60:
            recommendations.append("⚠️ 중간 품질의 사이클 패턴입니다. 필터링 조건을 조정해보세요.")
        else:
            recommendations.append("❌ 낮은 품질의 사이클 패턴입니다. 분석 기준을 재검토하세요.")
        
        # 상위 사이클 특성 기반 추천
        up_cycles = [a for a in all_analyses if a['upper_cycle_profile']['basic_info']['cycle_type'] == 'up']
        down_cycles = [a for a in all_analyses if a['upper_cycle_profile']['basic_info']['cycle_type'] == 'down']
        
        if len(up_cycles) > len(down_cycles) * 1.5:
            recommendations.append("📈 상승 사이클이 우세합니다. 상승 추세에 집중하세요.")
        elif len(down_cycles) > len(up_cycles) * 1.5:
            recommendations.append("📉 하락 사이클이 우세합니다. 하락 추세에 주의하세요.")
        
        return recommendations
    
    def _save_results(self, 
                     results: Dict[str, Any], 
                     upper_timeframe: str, 
                     lower_timeframe: str) -> None:
        """분석 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"hierarchical_analysis_{upper_timeframe}_to_{lower_timeframe}_{timestamp}.json"
        
        # 결과 디렉토리 생성
        results_dir = Path("analysis_results")
        results_dir.mkdir(exist_ok=True)
        
        # JSON 저장
        with open(results_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📁 분석 결과 저장됨: {results_dir / filename}")
    
    def print_analysis_results(self, results: Dict[str, Any]) -> None:
        """다중 사이클 분석 결과 출력 (개선된 버전)"""
        if 'error' in results:
            print(f"❌ 분석 오류: {results['error']}")
            return
        
        print(f"\n{'='*80}")
        print(f"🎯 다중 사이클 계층적 분석 결과")
        print(f"{'='*80}")
        
        # 기본 정보
        info = results['analysis_info']
        print(f"\n📊 분석 개요")
        print(f"   분석 시간: {info['timestamp']}")
        print(f"   상위 시간대: {info['upper_timeframe']} → 하위 시간대: {info['lower_timeframe']}")
        print(f"   필터링된 사이클: {info['total_cycles_filtered']}개")
        print(f"   분석 완료된 사이클: {info['total_cycles_analyzed']}개")
        
        # 종합 분석 결과
        if 'comprehensive_analysis' in results:
            comp_analysis = results['comprehensive_analysis']
            
            # 개요
            overview = comp_analysis['overview']
            print(f"\n📈 종합 분석 개요")
            print(f"   분석된 사이클: {overview['total_cycles_analyzed']}개")
            print(f"   상승 사이클: {overview['up_cycles_count']}개 ({overview['up_cycles_ratio']:.1%})")
            print(f"   하락 사이클: {overview['down_cycles_count']}개 ({1-overview['up_cycles_ratio']:.1%})")
            print(f"   평균 품질 점수: {overview['avg_quality_score']:.1f}/100")
            print(f"   최고 품질 점수: {overview['best_quality_score']:.1f}/100")
            print(f"   최저 품질 점수: {overview['worst_quality_score']:.1f}/100")
            
            # 상위 사이클 특성
            upper_char = comp_analysis['upper_cycle_characteristics']
            print(f"\n🔍 상위 사이클 특성")
            print(f"   평균 기간: {upper_char['avg_duration_days']:.1f}일 (±{upper_char['duration_std']:.1f})")
            print(f"   평균 가격 변화: {upper_char['avg_price_change_pct']:.2f}% (±{upper_char['price_change_std']:.2f}%)")
            print(f"   최대 상승: {upper_char['strongest_price_change']:.2f}%")
            print(f"   최대 하락: {upper_char['weakest_price_change']:.2f}%")
            print(f"   평균 RSI 변화: {upper_char['avg_rsi_change']:.1f}")
            print(f"   평균 MACD 변화: {upper_char['avg_macd_change']:.4f}")
            
            # 하위 사이클 패턴
            sub_patterns = comp_analysis['sub_cycle_patterns']
            print(f"\n📈 하위 사이클 패턴")
            print(f"   총 하위 사이클: {sub_patterns['total_sub_cycles']}개")
            print(f"   상위 사이클당 평균: {sub_patterns['avg_sub_cycles_per_upper']:.1f}개")
            print(f"   상승 하위 사이클 비율: {sub_patterns['up_sub_cycles_ratio']:.1%}")
            print(f"   하락 하위 사이클 비율: {sub_patterns['down_sub_cycles_ratio']:.1%}")
            print(f"   평균 순 움직임: {sub_patterns['avg_net_movement']:.2f}%")
            print(f"   평균 방향 일관성: {sub_patterns['avg_direction_consistency']:.1%}")
            
            # 시간대별 성능
            timeframe_perf = comp_analysis['timeframe_performance']
            if timeframe_perf:
                print(f"\n⏰ 시간대별 성능")
                for timeframe, perf in timeframe_perf.items():
                    print(f"   {timeframe}: {perf['cycle_count']}개 사이클, "
                          f"평균 품질 {perf['avg_quality_score']:.1f}, "
                          f"평균 변화 {perf['avg_price_change']:.2f}%")
            
            # 추천사항
            recommendations = comp_analysis['recommendations']
            if recommendations:
                print(f"\n💡 추천사항")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
        
        # 개별 분석 결과 요약 (상위 5개만)
        if 'individual_analyses' in results:
            individual = results['individual_analyses']
            print(f"\n📋 개별 사이클 분석 결과 (상위 5개)")
            
            # 품질 점수 기준으로 정렬
            sorted_analyses = sorted(individual, 
                                   key=lambda x: x['summary'].get('pattern_quality_score', 0), 
                                   reverse=True)
            
            for i, analysis in enumerate(sorted_analyses[:5], 1):
                profile = analysis['upper_cycle_profile']
                summary = analysis['summary']
                
                print(f"\n   {i}. {profile['basic_info']['cycle_id']}")
                print(f"      타입: {profile['basic_info']['cycle_type'].upper()}")
                print(f"      기간: {profile['basic_info']['duration_days']:.1f}일")
                print(f"      가격 변화: {profile['price_movement']['price_change_pct']:.2f}%")
                if 'error' not in summary:
                    print(f"      품질 점수: {summary['pattern_quality_score']:.1f}/100")
                    pattern = summary['sub_cycle_pattern']
                    print(f"      지배적 방향: {pattern['dominant_direction'].upper()}")
                    print(f"      방향 강도: {pattern['direction_strength']:.1%}")
        
        print(f"\n{'='*80}")
        print(f"✅ 다중 사이클 분석 완료")
        print(f"{'='*80}")

def main():
    """메인 실행 함수"""
    # 분석기 초기화
    analyzer = HierarchicalCycleAnalyzer()
    
    print("="*60)
    print("🚀 Multi-Timeframe Cycle Analysis Tool")
    print("="*60)
    
    # 사용 가능한 시간대 확인
    available_timeframes = analyzer.get_available_timeframes()
    print(f"\n사용 가능한 시간대: {', '.join(available_timeframes)}")
    
    # 실행 모드 선택
    print("\n실행 모드를 선택하세요:")
    print("1. 인터랙티브 모드 (추천)")
    print("2. 빠른 실행 (기본값 사용)")
    
    mode = input("\n선택 (1-2, Enter: 인터랙티브): ").strip()
    
    try:
        if mode == '2':
            # 빠른 실행 모드
            results = analyzer.run_hierarchical_analysis(
                upper_timeframe='4h',
                filter_conditions={'duration_candles': ('>', 5)},
                interactive=False
            )
        else:
            # 인터랙티브 모드 (기본)
            results = analyzer.run_hierarchical_analysis(
                interactive=True
            )
        
        # 결과 출력
        analyzer.print_analysis_results(results)
        
        # 추가 분석 여부
        while True:
            another = input("\n다른 사이클을 분석하시겠습니까? (y/n): ").strip().lower()
            if another == 'y':
                results = analyzer.run_hierarchical_analysis(interactive=True)
                analyzer.print_analysis_results(results)
            else:
                print("\n분석을 종료합니다. 감사합니다!")
                break
                
    except Exception as e:
        print(f"❌ 분석 실행 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()