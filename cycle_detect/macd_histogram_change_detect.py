"""
개선된 Multi-Timeframe MACD Detection (노이즈 허용 로직 수정)
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

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from feature_extract.macd_historgram_change_feature.feature_extract import CycleFeatureCalculator
    from feature_extract.macd_historgram_change_feature.config import DEFAULT_CONFIG
except ImportError as e:
    print(f"❌ 특징 계산기 import 실패: {e}")
    sys.exit(1)

def load_algorithm():
    """수정된 MACD 알고리즘 로드"""
    try:
        algorithm_path = project_root / "cycle_algorithm" / "algorithm_list" / "macd_histogram_change"
        sys.path.append(str(algorithm_path))
        
        # 수정된 알고리즘 파일을 사용
        from macd_histogram_change import SimpleMACDAlgorithm, SimpleConfig
        algorithm =  SimpleMACDAlgorithm()
        config = SimpleConfig(
            max_opposite_consecutive = 2 ,
            min_cycle_length = 3  
        )
        return algorithm, config
        
    except ImportError as e:
        print(f"❌ 알고리즘 로드 실패: {e}")
        raise


def find_timeframe_files():
    """타임프레임별 파일 찾기"""
    base_data_path = project_root / "data" / "base_data"
    
    if not base_data_path.exists():
        print(f"❌ 기본 데이터 폴더가 존재하지 않습니다: {base_data_path}")
        return {}
    
    timeframe_files = {}
    patterns = {
        '1h': ['*1h*', '*_1h.*', '*1hour*'],
        '4h': ['*4h*', '*_4h.*', '*4hour*'], 
        '1d': ['*1d*', '*_1d.*', '*1day*', '*daily*'],
        '1w': ['*1w*', '*_1w.*', '*1week*', '*weekly*'],
        '1m': ['*1m*', '*_1m.*', '*1month*', '*monthly*']
    }
    
    for timeframe, pattern_list in patterns.items():
        found = False
        for pattern in pattern_list:
            files = list(base_data_path.glob(pattern))
            if files:
                latest_file = max(files, key=os.path.getctime)
                timeframe_files[timeframe] = latest_file
                print(f"✅ {timeframe}: {latest_file.name}")
                found = True
                break
        
        if not found:
            print(f"⚠️  {timeframe}: 파일을 찾을 수 없습니다")
    
    return timeframe_files


def extract_cycles_from_classification(data: pd.DataFrame, classification: pd.Series, cycles_info: List):
    """분류 결과에서 실제 사이클들을 추출"""
    extracted_cycles = []
    current_type = None
    current_start = None
    
    for idx, cycle_type in enumerate(classification):
        if cycle_type != 0:  # 사이클인 경우
            if current_type != cycle_type:
                # 새로운 사이클 시작
                if current_start is not None:
                    # 이전 사이클 종료
                    cycle_data = {
                        'start_idx': current_start,
                        'end_idx': idx - 1,
                        'cycle_type': 'up' if current_type == 1 else 'down',
                        'length': idx - current_start
                    }
                    extracted_cycles.append(cycle_data)
                
                current_start = idx
                current_type = cycle_type
        else:
            # 노이즈인 경우, 현재 사이클 종료
            if current_start is not None:
                cycle_data = {
                    'start_idx': current_start,
                    'end_idx': idx - 1,
                    'cycle_type': 'up' if current_type == 1 else 'down',
                    'length': idx - current_start
                }
                extracted_cycles.append(cycle_data)
                current_start = None
                current_type = None
    
    # 마지막 사이클 처리
    if current_start is not None:
        cycle_data = {
            'start_idx': current_start,
            'end_idx': len(classification) - 1,
            'cycle_type': 'up' if current_type == 1 else 'down',
            'length': len(classification) - current_start
        }
        extracted_cycles.append(cycle_data)
    
    return extracted_cycles


def create_cycle_records_v3(data: pd.DataFrame, cycles: List[Dict], timeframe: str, algorithm_name: str):
    """수정된 알고리즘으로 사이클별 레코드 생성"""
    cycle_records = []
    
    # 특징 계산기 초기화
    feature_calculator = CycleFeatureCalculator()
    
    for i, cycle in enumerate(cycles):
        start_idx = cycle.start_idx if hasattr(cycle, 'start_idx') else cycle['start_idx']
        end_idx = cycle.end_idx if hasattr(cycle, 'end_idx') else cycle['end_idx']
        cycle_type = cycle.cycle_type if hasattr(cycle, 'cycle_type') else cycle['cycle_type']
        
        # cycle_type 변환 ('rising' -> 'up', 'falling' -> 'down')
        if cycle_type == 'rising':
            cycle_type = 'up'
        elif cycle_type == 'falling':
            cycle_type = 'down'
        
        # 사이클 데이터 추출
        cycle_data = data.iloc[start_idx:end_idx + 1].copy()
        
        # 캔들 데이터 생성
        candle_data = []
        for idx, (timestamp, row) in enumerate(cycle_data.iterrows()):
            # timestamp 처리
            if hasattr(timestamp, 'strftime'):
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(timestamp, str):
                timestamp_str = timestamp
            else:
                try:
                    if hasattr(data.index, 'to_series'):
                        original_timestamp = data.index.to_series().iloc[start_idx + idx]
                        if hasattr(original_timestamp, 'strftime'):
                            timestamp_str = original_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            timestamp_str = str(original_timestamp)
                    else:
                        timestamp_str = f"candle_{start_idx + idx}"
                except:
                    timestamp_str = f"candle_{start_idx + idx}"
            
            candle_record = {
                'timestamp': timestamp_str,
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'close': float(row.get('close', 0)),
                'volume': float(row.get('Volume USD', 0)),
                'macd': float(row.get('macd', 0)),
                'macd_signal': float(row.get('macd_signal', 0)),
                'macd_hist': float(row.get('macd_hist', 0)),
                'rsi': float(row.get('rsi', 50.0)) if 'rsi' in row and pd.notna(row.get('rsi')) else 50.0
            }
            candle_data.append(candle_record)
        
        # 날짜 정보 설정
        if candle_data:
            start_date_str = candle_data[0]['timestamp']
            end_date_str = candle_data[-1]['timestamp']
        else:
            start_date_str = f"idx_{start_idx}"
            end_date_str = f"idx_{end_idx}"
        
        # 사이클 ID 생성
        try:
            if candle_data and candle_data[0]['timestamp'] != f"candle_{start_idx}":
                date_part = candle_data[0]['timestamp'][:10].replace('-', '').replace(' ', '').replace(':', '')
                cycle_id = f"cycle_{timeframe}_{date_part}_{i+1:03d}"
            else:
                cycle_id = f"cycle_{timeframe}_idx{start_idx}_{i+1:03d}"
        except:
            cycle_id = f"cycle_{timeframe}_idx{start_idx}_{i+1:03d}"
        
        # 카테고리 결정
        start_macd = candle_data[0]['macd'] if candle_data else 0
        hist_direction = 'rising' if cycle_type == 'up' else 'falling'
        macd_zone = 'positive' if start_macd >= 0 else 'negative'
        category = f"{hist_direction}_{macd_zone}"
        
        # 특징 계산
        cycle_features = feature_calculator.extract_features_from_candle_data(candle_data)
        
        # 사이클 레코드 생성
        cycle_record = {
            'cycle_id': cycle_id,
            'timeframe': timeframe,
            'start_date': start_date_str,
            'end_date': end_date_str,
            'cycle_type': cycle_type,
            'duration_candles': len(cycle_data),
            'category': category,
            'algorithm_used': algorithm_name,
            'candle_data': candle_data,
            'cycle_features': cycle_features
        }
        
        cycle_records.append(cycle_record)
    
    return cycle_records


def detect_cycles_for_timeframe_v3(file_path, timeframe, algorithm, config):
    """수정된 알고리즘으로 특정 타임프레임에 대해 사이클 감지"""
    print(f"\n🔄 {timeframe} 처리 중 (수정된 알고리즘)...")
    
    # 데이터 로드
    if file_path.suffix == '.parquet':
        data = pd.read_parquet(file_path)
    elif file_path.suffix == '.csv':
        data = pd.read_csv(file_path, index_col=0)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")
    
    print(f"📊 데이터 로드: {len(data)} 행")
    
    # 원본 인덱스 백업
    original_index = data.index.copy()
    
    # MACD 지표 유효성 확인
    initial_rows = len(data)
    valid_mask = data['macd_hist'].notna()
    data = data[valid_mask].copy()
    original_index = original_index[valid_mask]
    data.reset_index(drop=True, inplace=True)
    
    print(f"📊 유효 데이터: {len(data)} 행 (제거: {initial_rows - len(data)} 행)")

    # 수정된 알고리즘으로 사이클 감지
    cycles, classification = algorithm.detect_cycles(data, config)
    print(f"✅ 사이클 감지 완료: {len(cycles)}개")
    
    # 분류 시리즈 정리
    classification_clean = classification.fillna(0).astype(int)
    
    if len(classification_clean) != len(data):
        print(f"⚠️  길이 불일치: 데이터({len(data)}) vs 분류({len(classification_clean)})")
        full_classification = pd.Series(0, index=range(len(data)), dtype=int)
        copy_length = min(len(classification_clean), len(data))
        if copy_length > 0:
            full_classification.iloc[:copy_length] = classification_clean.iloc[:copy_length]
        classification_clean = full_classification
    
    # 원본 인덱스를 data에 임시 추가
    data_with_original_index = data.copy()
    data_with_original_index.index = original_index
    
    # 사이클 레코드 생성
    cycle_records = create_cycle_records_v3(data_with_original_index, cycles, timeframe, algorithm.name)
    
    # 통계 계산
    cycle_points = sum([cycle['duration_candles'] for cycle in cycle_records])
    total_points = len(data)
    noise_points = total_points - cycle_points

    if total_points > 0:
        print(f"📈 사이클: {cycle_points} 포인트 ({cycle_points/total_points*100:.1f}%)")
        print(f"🔇 노이즈: {noise_points} 포인트 ({noise_points/total_points*100:.1f}%)")
    else:
        print(f"📈 사이클: {cycle_points} 포인트 (0.0%)")
        print(f"🔇 노이즈: {noise_points} 포인트 (0.0%)")
    
    return cycle_records, len(cycles)


def save_cycle_results_v3(cycle_records: List[Dict], timeframe: str):
    """수정된 구조로 사이클 결과 저장"""
    if not cycle_records:
        print(f"⚠️  {timeframe}: 저장할 사이클이 없습니다")
        return False
    
    # 저장 경로 설정
    output_path = project_root / "data" / "cycle_data" / "structured"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # DataFrame 생성
    df = pd.DataFrame(cycle_records)
    
    # 파일명 생성 (v3 표시)
    cycle_file = output_path / f"cycles_{timeframe}.parquet"
    
    try:
        df.to_parquet(cycle_file, index=False)
        print(f"💾 저장 완료: {cycle_file.name} ({len(cycle_records)}개 사이클)")
        
        return True
        
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        return False


def run_fixed_detection():
    """수정된 알고리즘으로 Detection 실행"""
    print("🚀 Fixed Multi-Timeframe MACD Detection 시작")
    print("="*60)
    
    # 1. 수정된 알고리즘 로드
    algorithm, config = load_algorithm()
    print(f"✅ 수정된 알고리즘 로드: {algorithm.name}")
    
    # 2. 타임프레임 파일 찾기
    timeframe_files = find_timeframe_files()
    
    if not timeframe_files:
        print("❌ 처리할 타임프레임 파일을 찾을 수 없습니다")
        return
    
    print(f"\n📊 처리 대상: {len(timeframe_files)}개 타임프레임")
    
    # 3. 각 타임프레임 처리
    results = {}
    total_cycles = 0
    
    for timeframe, file_path in timeframe_files.items():
        try:
            # 수정된 알고리즘으로 사이클 감지
            cycle_records, cycle_count = detect_cycles_for_timeframe_v3(
                file_path, timeframe, algorithm, config
            )
            
            # 저장
            if save_cycle_results_v3(cycle_records, timeframe):
                results[timeframe] = {
                    'cycle_count': cycle_count,
                    'total_candles': sum([c['duration_candles'] for c in cycle_records])
                }
                total_cycles += cycle_count
            
        except Exception as e:
            print(f"❌ {timeframe} 실패: {e}")
            import traceback
            print(f"   상세 오류: {traceback.format_exc()}")
            continue
    
    # 4. 결과 요약
    print("\n" + "="*60)
    print("🎉 Fixed MACD Detection 완료!")
    print("="*60)
    print(f"📊 성공적으로 처리된 타임프레임: {len(results)}개")
    print(f"📈 총 사이클 수: {total_cycles:,}개")
    
    if results:
        print("\n📁 생성된 파일 (수정된 알고리즘):")
        for timeframe, result in results.items():
            print(f"  🔄 {timeframe}: cycles_{timeframe}.parquet")
            print(f"     사이클 {result['cycle_count']:,}개, 캔들 {result['total_candles']:,}개")
    
    return results


def test_sample_data():
    """샘플 데이터로 수정된 알고리즘 테스트"""
    print("🧪 샘플 데이터 테스트")
    print("="*60)
    
    # 제공된 MACD 히스토그램 데이터
    hist_data = [-1133.12, -1265.86, -1352.62, -1403.18, -1038.2, -685.197, -434.487, -327.767, 
                 32.59714, 270.226, 336.4961, 403.8349, 184.2999, 303.8576, 302.3759, 351.2473, 
                 459.427, 469.2245, 300.8355, 234.2247, -301.629]
    
    # 기본 데이터 구조 생성
    df = pd.DataFrame({
        'macd_hist': hist_data,
        'open': [50000] * len(hist_data),
        'high': [51000] * len(hist_data),
        'low': [49000] * len(hist_data),
        'close': [50500] * len(hist_data),
        'Volume USD': [1000000] * len(hist_data),
        'macd': [0] * len(hist_data),
        'macd_signal': [0] * len(hist_data),
        'rsi': [50] * len(hist_data)
    })
    
    # 알고리즘 테스트
    algorithm, config = load_algorithm()
    
    cycles, classification = algorithm.detect_cycles(df, config)
    
    #print(f"\n📊 감지된 사이클: {len(cycles)}개")
    for i, cycle in enumerate(cycles):
        print(f"사이클 {i+1}: {cycle.cycle_type} | 인덱스 {cycle.start_idx}-{cycle.end_idx} | 길이 {cycle.length}")
    
    print(f"\n📋 분류 결과:")
    print(f"분류: {classification.tolist()}")
    
    # 예상 결과와 비교
    print(f"\n🎯 기대 결과:")
    print(f"인덱스 3-17이 하나의 상승 사이클로 분류되어야 함")
    
    # 실제로 하나의 긴 상승 사이클이 감지되었는지 확인
    rising_cycles = [c for c in cycles if c.cycle_type == 'rising']
    if rising_cycles:
        longest_cycle = max(rising_cycles, key=lambda x: x.length)
        print(f"✅ 가장 긴 상승 사이클: 인덱스 {longest_cycle.start_idx}-{longest_cycle.end_idx}, 길이 {longest_cycle.length}")
        
        if longest_cycle.start_idx <= 3 and longest_cycle.end_idx >= 17:
            print("✅ 수정 성공! 예상대로 긴 상승 사이클이 감지되었습니다.")
        else:
            print("⚠️  여전히 예상과 다른 결과입니다.")
    else:
        print("❌ 상승 사이클이 감지되지 않았습니다.")


if __name__ == "__main__":
    print("🚀 Fixed MACD Detection")
    
    print("\n선택하세요:")
    print("1: 샘플 데이터 테스트")
    print("2: 수정된 알고리즘으로 전체 Detection 실행")
    
    choice = input("선택 (1-2): ").strip()
    
    if choice == "1":
        test_sample_data()
    elif choice == "2":
        run_fixed_detection()
    else:
        print("잘못된 선택입니다.")