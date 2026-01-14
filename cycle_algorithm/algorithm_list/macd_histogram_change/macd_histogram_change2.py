"""
간단한 MACD Histogram Detection (무한루프 수정 버전)
직접적인 경로 지정으로 안전하게 알고리즘 로드
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys
import logging

# 로거 설정
def setup_simple_logger(name: str):
    """간단한 로거 설정"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# 설정 클래스
class CycleConfig:
    """사이클 감지 설정"""
    def __init__(self, max_negative_tolerance=2, max_positive_tolerance=2, min_cycle_duration=3):
        self.max_negative_tolerance = max_negative_tolerance
        self.max_positive_tolerance = max_positive_tolerance
        self.min_cycle_duration = min_cycle_duration

# 간단한 MACD 알고리즘 (텍스트 기준 정확한 4단계 구현)
class SimpleMACDAlgorithm:
    """텍스트 기준 정확한 MACD 알고리즘 구현 (노이즈 허용 모델)"""
    
    def __init__(self):
        self.name = "MACD Histogram Cycle Detection (4-Step)"
        self.version = "1.0"
    
    def detect_cycles(self, data: pd.DataFrame, config: CycleConfig) -> Tuple[List, pd.Series]:
        """텍스트 기준 4단계 사이클 감지"""
        # 컬럼 확인 및 가능한 대안 찾기
        macd_columns = [col for col in data.columns if 'macd' in col.lower() and 'hist' in col.lower()]
        
        if not macd_columns:
            # macd_hist가 없으면 다른 가능한 컬럼 찾기
            possible_columns = [col for col in data.columns if 'macd' in col.lower()]
            if possible_columns:
                macd_column = possible_columns[0]
                print(f"⚠️  'macd_hist' 컬럼 없음. '{macd_column}' 사용")
            else:
                raise ValueError(f"MACD 관련 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(data.columns)}")
        else:
            macd_column = macd_columns[0]
        
        macd_hist = data[macd_column].dropna()
        
        if len(macd_hist) < config.min_cycle_duration:
            print(f"⚠️  데이터가 너무 짧습니다 ({len(macd_hist)} < {config.min_cycle_duration})")
            return [], pd.Series(0, index=macd_hist.index)
        
        print(f"📊 MACD 알고리즘 시작: {len(macd_hist)} 포인트 처리")
        
        # 1단계: 원시 방향성(Raw Direction) 판별
        raw_directions = self._calculate_raw_directions(macd_hist)
        print(f"✅ 1단계 완료: 원시 방향성 판별 ({len(raw_directions)} 포인트)")
        
        # 2단계: 핵심 추세 구간(Core Streak) 식별
        core_streaks = self._identify_core_streaks(raw_directions)
        print(f"✅ 2단계 완료: 핵심 추세 구간 식별 ({len(core_streaks)} 구간)")
        
        # 3단계: 노이즈 허용 및 추세 연결
        merged_cycles = self._merge_cycles_with_noise_tolerance(core_streaks, raw_directions, config)
        print(f"✅ 3단계 완료: 노이즈 허용 병합 ({len(merged_cycles)} 사이클)")
        
        # 4단계: 최종 사이클 확정 및 필터링
        final_cycles = self._filter_cycles_by_duration(merged_cycles, config)
        print(f"✅ 4단계 완료: 최종 필터링 ({len(final_cycles)} 사이클)")
        
        # 분류 시리즈 생성
        classification = self._create_classification_series(len(macd_hist), final_cycles, macd_hist.index)
        
        return final_cycles, classification
    
    def _calculate_raw_directions(self, macd_hist: pd.Series) -> pd.Series:
        """1단계: 원시 방향성 판별"""
        # 변화량 계산
        changes = macd_hist.diff()
        
        # 부호 판별: +1(상승), -1(하락), 0(보합)
        directions = np.sign(changes)
        
        # 첫 번째 값은 NaN이므로 0으로 설정
        directions.iloc[0] = 0
        
        return directions.astype(int)
    
    def _identify_core_streaks(self, directions: pd.Series) -> List[Dict]:
        """2단계: 핵심 추세 구간 식별"""
        streaks = []
        current_direction = None
        start_idx = 0
        
        for i, direction in enumerate(directions):
            # 방향이 바뀌거나 마지막 인덱스인 경우
            if direction != current_direction or i == len(directions) - 1:
                # 이전 구간이 있고, 의미있는 방향(+1 or -1)인 경우 저장
                if current_direction in [1, -1] and i > start_idx:
                    end_idx = i - 1 if direction != current_direction else i
                    streak = {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'direction': current_direction,
                        'length': end_idx - start_idx + 1
                    }
                    streaks.append(streak)
                
                # 새로운 구간 시작
                current_direction = direction
                start_idx = i
        
        return streaks
    
    def _merge_cycles_with_noise_tolerance(self, core_streaks: List[Dict], directions: pd.Series, config: CycleConfig) -> List[Dict]:
        """3단계: 노이즈 허용 및 추세 연결 (핵심 로직)"""
        if not core_streaks:
            return []
        
        # 상승과 하락 구간을 분리
        rising_streaks = [s for s in core_streaks if s['direction'] == 1]
        falling_streaks = [s for s in core_streaks if s['direction'] == -1]
        
        # 각각에 대해 병합 수행
        rising_cycles = self._merge_same_direction_streaks(
            rising_streaks, directions, config.max_negative_tolerance, 'rising'
        )
        falling_cycles = self._merge_same_direction_streaks(
            falling_streaks, directions, config.max_positive_tolerance, 'falling'
        )
        
        # 시간 순으로 정렬
        all_cycles = rising_cycles + falling_cycles
        all_cycles.sort(key=lambda x: x['start_idx'])
        
        return all_cycles
    
    def _merge_same_direction_streaks(self, streaks: List[Dict], directions: pd.Series, tolerance: int, cycle_type: str) -> List[Dict]:
        """같은 방향의 구간들을 노이즈 허용하여 병합"""
        if not streaks:
            return []
        
        cycles = []
        i = 0
        
        while i < len(streaks):
            # 현재 구간부터 시작하여 병합 가능한 구간들을 찾음
            merged_streaks = [streaks[i]]
            current_end = streaks[i]['end_idx']
            j = i + 1
            
            while j < len(streaks):
                next_start = streaks[j]['start_idx']
                
                # 두 구간 사이의 간격 확인
                gap_length = next_start - current_end - 1
                
                # 간격이 허용 범위 내이고, 모두 반대 방향인지 확인
                if gap_length <= tolerance and gap_length >= 0:
                    gap_directions = directions.iloc[current_end + 1:next_start]
                    
                    # 간격이 모두 반대 방향으로만 구성되어 있는지 확인
                    target_direction = -1 if cycle_type == 'rising' else 1
                    if len(gap_directions) == 0 or all(d == target_direction or d == 0 for d in gap_directions):
                        merged_streaks.append(streaks[j])
                        current_end = streaks[j]['end_idx']
                        j += 1
                    else:
                        break
                else:
                    break
            
            # 병합된 사이클 생성
            cycle = self._create_cycle_from_streaks(merged_streaks, cycle_type)
            cycles.append(cycle)
            
            i = j
        
        return cycles
    
    def _create_cycle_from_streaks(self, streaks: List[Dict], cycle_type: str) -> Dict:
        """구간들로부터 사이클 객체 생성"""
        start_idx = streaks[0]['start_idx']
        end_idx = streaks[-1]['end_idx']
        total_length = end_idx - start_idx + 1
        
        # 핵심 추세 길이 계산
        core_length = sum(streak['length'] for streak in streaks)
        noise_periods = total_length - core_length
        
        # 건강도 계산 (핵심 추세 비율)
        health_score = core_length / total_length if total_length > 0 else 0
        
        # 강도 계산 (평균 연속성)
        avg_streak_length = core_length / len(streaks) if streaks else 0
        strength_score = avg_streak_length / total_length if total_length > 0 else 0
        
        return {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'cycle_type': cycle_type,
            'length': total_length,
            'core_streaks': streaks,
            'noise_periods': noise_periods,
            'health_score': health_score,
            'strength_score': strength_score
        }
    
    def _filter_cycles_by_duration(self, cycles: List[Dict], config: CycleConfig) -> List[Dict]:
        """4단계: 최소 기간 조건으로 사이클 필터링"""
        return [cycle for cycle in cycles if cycle['length'] >= config.min_cycle_duration]
    
    def _create_classification_series(self, data_length: int, cycles: List[Dict], original_index) -> pd.Series:
        """사이클 분류 시리즈 생성 (시간대 정확히 매칭)"""
        # 원본 인덱스와 동일한 분류 시리즈 생성
        classification = pd.Series(0, index=original_index, dtype=int)
        
        for cycle in cycles:
            value = 1 if cycle['cycle_type'] == 'rising' else -1
            start_idx = cycle['start_idx']
            end_idx = cycle['end_idx']
            
            # 인덱스 범위 안전 확인
            if start_idx < len(classification) and end_idx < len(classification):
                classification.iloc[start_idx:end_idx + 1] = value
        
        return classification

# 알고리즘 로더 (안전 버전)
def load_algorithm_safely():
    """안전하게 알고리즘 로드"""
    
    # 1. 현재 디렉토리 기준으로 가능한 경로들 정의
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    possible_paths = [
        # 가능한 알고리즘 경로들
        project_root / "cycle_algorithm" / "algorithm_list" / "macd_histogram_change",
        project_root / "cycle_detection" / "algorithm_list" / "macd_histogram_change", 
        current_dir / "algorithm_list" / "macd_histogram_change",
        current_dir / "macd_histogram_change",
        project_root / "cycle_algorithm" / "macd_histogram_change",
    ]
    
    print("🔍 알고리즘 파일 검색 중...")
    
    for i, path in enumerate(possible_paths):
        print(f"  {i+1}. 검색 경로: {path}")
        
        if path.exists() and path.is_dir():
            # 디렉토리에서 Python 파일 찾기
            py_files = list(path.glob("*.py"))
            if py_files:
                for py_file in py_files:
                    if 'macd' in py_file.name.lower():
                        print(f"     ✅ 파일 발견: {py_file}")
                        try:
                            # 동적 import 시도
                            sys.path.append(str(path))
                            module_name = py_file.stem
                            
                            import importlib.util
                            spec = importlib.util.spec_from_file_location(module_name, py_file)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # 알고리즘 클래스 찾기 (추상 클래스 제외)
                            algorithm_classes = []
                            for attr_name in dir(module):
                                attr = getattr(module, attr_name)
                                if (isinstance(attr, type) and 
                                    attr_name != 'SimpleMACDAlgorithm' and
                                    attr_name != 'BaseCycleAlgorithm' and  # 추상 클래스 제외
                                    'Algorithm' in attr_name):
                                    algorithm_classes.append((attr_name, attr))
                            
                            # 구체 클래스 우선 시도
                            for attr_name, attr in algorithm_classes:
                                try:
                                    # 인스턴스 생성 테스트
                                    test_instance = attr()
                                    if hasattr(test_instance, 'detect_cycles'):
                                        print(f"     ✅ 알고리즘 클래스 발견: {attr_name}")
                                        return test_instance
                                except Exception as e:
                                    print(f"     ❌ 클래스 {attr_name} 인스턴스화 실패: {e}")
                                    continue
                                    
                        except Exception as e:
                            print(f"     ❌ 로드 실패: {e}")
                            continue
        else:
            print(f"     ❌ 경로 없음")
    
    print("⚠️  외부 알고리즘을 찾을 수 없어 기본 알고리즘 사용")
    return SimpleMACDAlgorithm()


class MACDHistogramDetection:
    """MACD Histogram Detection (수정 버전)"""
    
    def __init__(self, config_params: Optional[Dict] = None):
        """초기화"""
        self.logger = setup_simple_logger("macd_detection")
        
        # 알고리즘 로드 (한 번만)
        self.logger.info("알고리즘 로딩 중...")
        self.algorithm = load_algorithm_safely()
        self.logger.info(f"알고리즘 로드 완료: {getattr(self.algorithm, 'name', 'Unknown')}")
        
        # 설정
        self.config = self._setup_config(config_params)
        
        # 경로 설정
        self.project_root = Path(__file__).parent.parent
        self.data_root = self.project_root / "data"
        self.base_data_path = self.data_root / "base_data"  # processed_data → base_data 변경
        self.cycle_data_path = self.data_root / "cycle_data"
        
        # 결과 저장 경로
        self.output_base_path = self.cycle_data_path / "macd_histogram_change_algo"
        self.cycle_path = self.output_base_path / "cycle"
        self.noise_path = self.output_base_path / "noise"
        
        # 디렉토리 생성
        self.cycle_path.mkdir(parents=True, exist_ok=True)
        self.noise_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("초기화 완료")
    
    def _setup_config(self, config_params: Optional[Dict] = None) -> CycleConfig:
        """설정 구성"""
        config = CycleConfig()
        
        if config_params:
            for key, value in config_params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    self.logger.info(f"설정 적용: {key} = {value}")
        
        return config
    
    def load_base_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """데이터 로드"""
        if file_path is None:
            # base_data 디렉토리에서 파일 찾기
            patterns = ["base_data_*.parquet", "base_data_*.csv", "btc_*.parquet", "btc_*.csv", "*.parquet", "*.csv"]
            
            for pattern in patterns:
                files = list(self.base_data_path.glob(pattern))
                if files:
                    file_path = max(files, key=os.path.getctime)
                    self.logger.info(f"자동 선택된 파일: {file_path}")
                    break
            
            if file_path is None:
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.base_data_path}")
        
        # 파일 로드
        file_path = Path(file_path)
        if file_path.suffix == '.parquet':
            data = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            data = pd.read_csv(file_path, index_col=0)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")
        
        self.logger.info(f"데이터 로드 완료: {len(data)} 행, {len(data.columns)} 열")
        self.logger.info(f"컬럼: {list(data.columns)}")
        
        return data
    
    def run_detection(self, input_file: Optional[str] = None) -> Dict:
        """전체 감지 프로세스 실행"""
        try:
            self.logger.info("=== MACD Detection 시작 ===")
            
            # 1. 데이터 로드
            data = self.load_base_data(input_file)
            
            # 2. 컬럼 확인
            self.logger.info(f"데이터 컬럼: {list(data.columns)}")
            
            # MACD 관련 컬럼 확인
            macd_columns = [col for col in data.columns if 'macd' in col.lower()]
            if not macd_columns:
                raise ValueError(f"MACD 관련 컬럼이 없습니다. 사용 가능한 컬럼: {list(data.columns)}")
            
            self.logger.info(f"MACD 관련 컬럼: {macd_columns}")
            
            # 3. 사이클 감지
            self.logger.info("사이클 감지 중...")
            cycles, classification = self.algorithm.detect_cycles(data, self.config)
            self.logger.info(f"사이클 감지 완료: {len(cycles)}개 발견")
            
            # 4. 데이터 분류 및 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result = self._save_results(data, cycles, classification, timestamp)
            
            self.logger.info("=== Detection 완료 ===")
            return result
            
        except Exception as e:
            self.logger.error(f"Detection 실패: {str(e)}")
            raise
    
    def _save_results(self, data: pd.DataFrame, cycles: List, classification: pd.Series, timestamp: str) -> Dict:
        """결과 저장 (시간대별 정확한 분류)"""
        
        print(f"📊 분류 데이터 처리 중...")
        print(f"   원본 데이터: {len(data)} 포인트")
        print(f"   분류 결과: {len(classification)} 포인트")
        
        # 원본 데이터 복사
        data_with_classification = data.copy()
        
        # classification의 NaN 값 처리
        classification_clean = classification.fillna(0).astype(int)
        
        # 시간대별 정확한 매칭을 위한 처리
        if len(classification_clean) != len(data):
            print(f"⚠️  길이 불일치: 데이터({len(data)}) vs 분류({len(classification_clean)})")
            
            # 원본 데이터와 동일한 인덱스로 새로운 분류 시리즈 생성
            full_classification = pd.Series(0, index=data.index, dtype=int)
            
            # 인덱스가 일치하는 부분만 매핑
            if hasattr(classification_clean, 'index'):
                # 분류 결과의 인덱스와 원본 데이터 인덱스의 교집합 찾기
                common_index = data.index.intersection(classification_clean.index)
                if len(common_index) > 0:
                    print(f"✅ 인덱스 기준 매칭: {len(common_index)} 포인트")
                    full_classification.loc[common_index] = classification_clean.loc[common_index]
                else:
                    # 인덱스가 다르면 위치 기준으로 매칭 (위험하므로 경고)
                    print(f"⚠️  인덱스 불일치로 위치 기준 매칭 시도")
                    copy_length = min(len(classification_clean), len(data))
                    if copy_length > 0:
                        full_classification.iloc[:copy_length] = classification_clean.iloc[:copy_length].values
            else:
                # 위치 기준 매칭
                copy_length = min(len(classification_clean), len(data))
                if copy_length > 0:
                    full_classification.iloc[:copy_length] = classification_clean[:copy_length].astype(int)
            
            classification_clean = full_classification
        else:
            # 길이가 같으면 인덱스만 확실히 맞추기
            classification_clean = pd.Series(classification_clean.values, index=data.index, dtype=int)
        
        # 시간대 정보 확인 (인덱스가 시간 정보를 포함하는 경우)
        if hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype):
            print(f"✅ 시간 인덱스 감지: {data.index[0]} ~ {data.index[-1]}")
        elif 'date' in data.columns:
            print(f"✅ 날짜 컬럼 감지: {data['date'].iloc[0]} ~ {data['date'].iloc[-1]}")
        
        # 분류 정보 추가 (인덱스 완전 일치 보장)
        data_with_classification['cycle_type'] = classification_clean
        data_with_classification['is_cycle'] = (classification_clean != 0)
        
        # 시간대별 분류 확인
        cycle_count = (classification_clean != 0).sum()
        noise_count = (classification_clean == 0).sum()
        
        print(f"📊 분류 결과 확인:")
        print(f"   사이클 포인트: {cycle_count} ({cycle_count/len(data)*100:.1f}%)")
        print(f"   노이즈 포인트: {noise_count} ({noise_count/len(data)*100:.1f}%)")
        
        # 데이터 분리 (안전한 불린 인덱싱)
        try:
            # 불린 마스크 생성 (인덱스 완전 일치)
            is_cycle_mask = data_with_classification['is_cycle'].astype(bool)
            
            # 데이터 분리
            cycle_data = data_with_classification[is_cycle_mask].copy()
            noise_data = data_with_classification[~is_cycle_mask].copy()
            
            print(f"✅ 데이터 분리 성공")
            print(f"   사이클 데이터: {len(cycle_data)} 행")
            print(f"   노이즈 데이터: {len(noise_data)} 행")
            
        except Exception as e:
            print(f"❌ 데이터 분리 실패: {e}")
            raise
        
        # 시간대별 분포 확인 (샘플링)
        if len(cycle_data) > 0:
            print(f"📅 사이클 데이터 시간 범위:")
            if hasattr(cycle_data.index, 'dtype') and 'datetime' in str(cycle_data.index.dtype):
                print(f"   시작: {cycle_data.index.min()}")
                print(f"   종료: {cycle_data.index.max()}")
            elif 'date' in cycle_data.columns:
                print(f"   시작: {cycle_data['date'].min()}")
                print(f"   종료: {cycle_data['date'].max()}")
        
        # 파일 저장 (Parquet 필수)
        cycle_file = self.cycle_path / f"cycle_data_{timestamp}.parquet"
        noise_file = self.noise_path / f"noise_data_{timestamp}.parquet"
        
        try:
            cycle_data.to_parquet(cycle_file, index=True)
            noise_data.to_parquet(noise_file, index=True)
            print(f"✅ Parquet 파일 저장 성공")
        except Exception as e:
            print(f"❌ Parquet 파일 저장 실패: {e}")
            print("📦 필요한 라이브러리: pip install pyarrow 또는 pip install fastparquet")
            raise RuntimeError(
                "Parquet 저장에 필요한 라이브러리가 없습니다. "
                "'pip install pyarrow' 또는 'pip install fastparquet'를 실행하세요."
            )
        
        # 통계 계산 (cycles가 객체인지 딕셔너리인지 확인)
        rising_cycles = 0
        falling_cycles = 0
        
        for c in cycles:
            try:
                # 객체인 경우
                if hasattr(c, 'cycle_type'):
                    cycle_type = c.cycle_type
                # 딕셔너리인 경우
                elif isinstance(c, dict):
                    cycle_type = c.get('cycle_type', 'unknown')
                else:
                    cycle_type = 'unknown'
                
                if cycle_type == 'rising':
                    rising_cycles += 1
                elif cycle_type == 'falling':
                    falling_cycles += 1
                    
            except Exception as e:
                print(f"⚠️  사이클 타입 확인 실패: {e}")
                continue
        
        # 사이클별 시간 정보 확인 (처음 몇 개만)
        print(f"🔍 사이클 시간 정보 (처음 5개):")
        for i, cycle in enumerate(cycles[:5]):
            try:
                start_idx = cycle.get('start_idx', 0) if isinstance(cycle, dict) else getattr(cycle, 'start_idx', 0)
                end_idx = cycle.get('end_idx', 0) if isinstance(cycle, dict) else getattr(cycle, 'end_idx', 0)
                cycle_type = cycle.get('cycle_type', 'unknown') if isinstance(cycle, dict) else getattr(cycle, 'cycle_type', 'unknown')
                
                if start_idx < len(data) and end_idx < len(data):
                    if hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype):
                        start_time = data.index[start_idx]
                        end_time = data.index[end_idx]
                        print(f"   사이클 {i+1} ({cycle_type}): {start_time} ~ {end_time}")
                    elif 'date' in data.columns:
                        start_time = data['date'].iloc[start_idx]
                        end_time = data['date'].iloc[end_idx]
                        print(f"   사이클 {i+1} ({cycle_type}): {start_time} ~ {end_time}")
                    else:
                        print(f"   사이클 {i+1} ({cycle_type}): 인덱스 {start_idx} ~ {end_idx}")
            except Exception as e:
                print(f"   사이클 {i+1}: 시간 정보 추출 실패 ({e})")
        
        result = {
            'timestamp': timestamp,
            'cycle_file': str(cycle_file),
            'noise_file': str(noise_file),
            'total_points': len(data),
            'cycle_points': len(cycle_data),
            'noise_points': len(noise_data),
            'cycle_ratio': len(cycle_data) / len(data) if len(data) > 0 else 0,
            'noise_ratio': len(noise_data) / len(data) if len(data) > 0 else 0,
            'total_cycles': len(cycles),
            'rising_cycles': rising_cycles,
            'falling_cycles': falling_cycles
        }
        
        # 메타데이터 저장 (cycles 처리 개선)
        cycles_for_metadata = []
        for i, c in enumerate(cycles[:100]):  # 처음 100개만
            try:
                if hasattr(c, '__dict__'):
                    # 객체인 경우 딕셔너리로 변환
                    cycle_dict = {
                        'cycle_id': i,
                        'cycle_type': getattr(c, 'cycle_type', 'unknown'),
                        'start_idx': getattr(c, 'start_idx', 0),
                        'end_idx': getattr(c, 'end_idx', 0),
                        'length': getattr(c, 'length', 0),
                        'health_score': getattr(c, 'health_score', 0),
                        'strength_score': getattr(c, 'strength_score', 0),
                        'noise_periods': getattr(c, 'noise_periods', 0)
                    }
                elif isinstance(c, dict):
                    # 딕셔너리인 경우 그대로 사용
                    cycle_dict = {
                        'cycle_id': i,
                        **c
                    }
                else:
                    # 기타인 경우 문자열로 변환
                    cycle_dict = {
                        'cycle_id': i,
                        'raw_data': str(c)
                    }
                cycles_for_metadata.append(cycle_dict)
            except Exception as e:
                print(f"⚠️  사이클 {i} 메타데이터 변환 실패: {e}")
                cycles_for_metadata.append({
                    'cycle_id': i,
                    'error': str(e)
                })
        
        metadata = {
            'detection_timestamp': datetime.now().isoformat(),
            'algorithm_info': {
                'name': getattr(self.algorithm, 'name', 'Unknown'),
                'version': getattr(self.algorithm, 'version', '1.0')
            },
            'config': {
                'max_negative_tolerance': self.config.max_negative_tolerance,
                'max_positive_tolerance': self.config.max_positive_tolerance,
                'min_cycle_duration': self.config.min_cycle_duration
            },
            'results': result,
            'data_info': {
                'original_columns': list(data.columns),
                'has_datetime_index': hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype),
                'has_date_column': 'date' in data.columns,
                'classification_stats': {
                    'total_points': len(classification_clean),
                    'cycle_points': (classification_clean != 0).sum(),
                    'noise_points': (classification_clean == 0).sum()
                }
            },
            'cycles_sample': cycles_for_metadata
        }
        
        metadata_file = self.output_base_path / f"metadata_{timestamp}.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            print(f"✅ 메타데이터 저장 성공")
        except Exception as e:
            print(f"❌ 메타데이터 저장 실패: {e}")
        
        # 상세한 결과 출력
        print("\n" + "=" * 60)
        print("🎯 MACD Detection 결과 요약")
        print("=" * 60)
        print(f"📊 전체 데이터: {len(data):,} 포인트")
        print(f"🔄 사이클 데이터: {len(cycle_data):,} 포인트 ({result['cycle_ratio']:.2%})")
        print(f"🔇 노이즈 데이터: {len(noise_data):,} 포인트 ({result['noise_ratio']:.2%})")
        print(f"📈 상승 사이클: {rising_cycles:,}개")
        print(f"📉 하락 사이클: {falling_cycles:,}개")
        print(f"📊 총 사이클 수: {len(cycles):,}개")
        print("=" * 60)
        print(f"💾 사이클 파일: {cycle_file}")
        print(f"💾 노이즈 파일: {noise_file}")
        print(f"💾 메타데이터: {metadata_file}")
        print("=" * 60)
        
        return result


# 간단한 실행 함수들
def run_detection_with_default_config(input_file: Optional[str] = None) -> Dict:
    """기본 설정으로 감지 실행"""
    detector = MACDHistogramDetection()
    return detector.run_detection(input_file)


def run_detection_with_presets(preset: str = "balanced", input_file: Optional[str] = None) -> Dict:
    """프리셋으로 감지 실행"""
    presets = {
        "conservative": {
            "max_negative_tolerance": 1,
            "max_positive_tolerance": 1,
            "min_cycle_duration": 5
        },
        "balanced": {
            "max_negative_tolerance": 2,
            "max_positive_tolerance": 2,
            "min_cycle_duration": 3
        },
        "aggressive": {
            "max_negative_tolerance": 4,
            "max_positive_tolerance": 4,
            "min_cycle_duration": 2
        }
    }
    
    if preset not in presets:
        raise ValueError(f"지원하지 않는 프리셋: {preset}")
    
    detector = MACDHistogramDetection(presets[preset])
    return detector.run_detection(input_file)


if __name__ == "__main__":
    # 테스트 실행
    print("🚀 Simple MACD Detection 테스트")
    
    try:
        result = run_detection_with_default_config()
        
        print("\n" + "=" * 60)
        print("🎉 Detection 성공!")
        print("=" * 60)
        print(f"📊 전체 데이터: {result['total_points']:,} 포인트")
        print(f"🔄 사이클 데이터: {result['cycle_points']:,} 포인트 ({result['cycle_ratio']:.2%})")
        print(f"🔇 노이즈 데이터: {result['noise_points']:,} 포인트 ({result['noise_ratio']:.2%})")
        print(f"📈 상승 사이클: {result['rising_cycles']:,}개")
        print(f"📉 하락 사이클: {result['falling_cycles']:,}개")
        print(f"📊 총 사이클 수: {result['total_cycles']:,}개")
        print("=" * 60)
        print(f"💾 결과 파일:")
        print(f"  - 사이클: {result['cycle_file']}")
        print(f"  - 노이즈: {result['noise_file']}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        
        # Parquet 관련 오류인 경우 설치 안내
        if "parquet" in str(e).lower() or "pyarrow" in str(e).lower() or "fastparquet" in str(e).lower():
            print("\n📦 해결 방법:")
            print("다음 중 하나를 실행하세요:")
            print("  pip install pyarrow")
            print("  또는")
            print("  pip install fastparquet")
            print("\n그 후 다시 실행하세요.")