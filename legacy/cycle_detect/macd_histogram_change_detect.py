"""
개선된 Multi-Timeframe MACD Detection (노이즈 허용 로직 수정)

수정 이력:
  - find_timeframe_files(): OI 파일 등 지표가 없는 파일이 잘못 선택되던 문제 수정.
      기존 max(ctime) 방식은 가장 최근에 생성된 파일을 고르기 때문에
      새로 추가된 OI CSV가 BTCUSD 원본 CSV보다 먼저 선택되는 버그가 있었다.
      수정 방향: glob 결과에서 'oi', 'funding' 등 비지표 파일을 제외하고,
                 macd_hist 컬럼 존재 여부로 최종 검증.

  - save_cycle_results_v3(): 빈 struct로 인한 Parquet 저장 실패 수정.
      config에서 end 카테고리 특징이 모두 비활성화되면 cycle_features 안에
      'end': {} 같은 빈 dict가 생긴다.
      PyArrow는 필드가 하나도 없는 struct를 parquet에 쓸 수 없어 에러 발생.
      수정 방향: 저장 직전 빈 카테고리 dict를 제거 (_prune_empty_cycle_features).
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
    from feature_extract.macd_historgram_change_feature.feature_extract import (
        CycleFeatureCalculator,
        StructuredCycleProcessor,
    )
    from feature_extract.macd_historgram_change_feature.config import DEFAULT_CONFIG
except ImportError as e:
    print(f"❌ 특징 계산기 import 실패: {e}")
    sys.exit(1)


def load_algorithm():
    """수정된 MACD 알고리즘 로드"""
    try:
        algorithm_path = project_root / "cycle_algorithm" / "algorithm_list" / "macd_histogram_change"
        sys.path.append(str(algorithm_path))

        from macd_histogram_change import SimpleMACDAlgorithm, SimpleConfig
        algorithm = SimpleMACDAlgorithm()
        config = SimpleConfig(
            max_opposite_consecutive=2,
            min_cycle_length=3
        )
        return algorithm, config

    except ImportError as e:
        print(f"❌ 알고리즘 로드 실패: {e}")
        raise


# ──────────────────────────────────────────────────────────────────────
# 수정 1: 파일 선택 로직
# ──────────────────────────────────────────────────────────────────────

# 지표(MACD)가 없는 파일임을 나타내는 키워드 목록.
# glob 결과에서 이 문자열 중 하나라도 파일명에 포함되면 제외한다.
_NON_INDICATOR_KEYWORDS = ['oi', 'funding', 'interest', 'backfill']


def _has_macd_hist(file_path: Path) -> bool:
    """
    파일에 macd_hist 컬럼이 실제로 존재하는지 확인.
    CSV는 헤더만 읽어 확인하고(nrows=0), parquet은 스키마로 확인한다.
    """
    try:
        if file_path.suffix == '.csv':
            header = pd.read_csv(file_path, nrows=0)
            return 'macd_hist' in header.columns
        elif file_path.suffix == '.parquet':
            import pyarrow.parquet as pq
            schema = pq.read_schema(file_path)
            return 'macd_hist' in schema.names
    except Exception:
        pass
    return False


def find_timeframe_files():
    """
    타임프레임별 OHLCV+지표 파일 찾기.

    기존 max(ctime) 방식은 가장 최근에 생성된 파일을 선택하기 때문에,
    OI/펀딩비 파일처럼 지표가 없는 파일이 새로 추가되면 그 파일이 잘못 선택됐다.

    수정된 방식:
      1. 파일명에 비지표 키워드(_NON_INDICATOR_KEYWORDS)가 없는 파일만 후보로 남긴다.
      2. 후보 중 macd_hist 컬럼이 실제로 존재하는 파일만 남긴다(헤더 검증).
      3. 검증을 통과한 파일이 여럿이면 가장 최근에 수정된 파일을 선택한다.
    """
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
        '1m': ['*1m*', '*_1m.*', '*1month*', '*monthly*'],
    }

    for timeframe, pattern_list in patterns.items():
        found = False
        for pattern in pattern_list:
            raw_files = list(base_data_path.glob(pattern))

            # 1단계: 파일명 키워드 필터 (OI, 펀딩비 등 제외)
            keyword_filtered = [
                f for f in raw_files
                if not any(kw in f.name.lower() for kw in _NON_INDICATOR_KEYWORDS)
            ]

            # 2단계: macd_hist 컬럼 존재 여부 검증
            indicator_files = [f for f in keyword_filtered if _has_macd_hist(f)]

            if indicator_files:
                # 검증 통과 파일이 여럿이면 가장 최근 수정 파일 선택
                best_file = max(indicator_files, key=lambda f: f.stat().st_mtime)
                timeframe_files[timeframe] = best_file
                print(f"✅ {timeframe}: {best_file.name}")
                found = True
                break

            elif keyword_filtered:
                # macd_hist는 없지만 키워드 필터는 통과한 파일 → 경고 후 스킵
                names = [f.name for f in keyword_filtered]
                print(f"⚠️  {timeframe}: 후보 파일 {names}에 macd_hist 없음 → 스킵")

        if not found:
            print(f"⚠️  {timeframe}: 유효한 지표 파일을 찾을 수 없습니다")

    return timeframe_files


# ──────────────────────────────────────────────────────────────────────
# 수정 2: 빈 카테고리 dict 제거 헬퍼
# ──────────────────────────────────────────────────────────────────────

def _prune_empty_cycle_features(cycle_features: Dict) -> Dict:
    """
    cycle_features 딕셔너리에서 빈 카테고리 dict를 제거한다.

    config에서 특정 카테고리(예: end)의 모든 특징이 enabled=false이면
    해당 카테고리는 빈 dict {}로 계산된다.
    PyArrow는 필드 없는 struct를 Parquet에 쓸 수 없으므로,
    저장 전에 반드시 이 함수를 통해 빈 dict를 제거해야 한다.
    """
    if not isinstance(cycle_features, dict):
        return cycle_features
    return {k: v for k, v in cycle_features.items()
            if not (isinstance(v, dict) and len(v) == 0)}


def extract_cycles_from_classification(data: pd.DataFrame, classification: pd.Series, cycles_info: List):
    """분류 결과에서 실제 사이클들을 추출"""
    extracted_cycles = []
    current_type = None
    current_start = None

    for idx, cycle_type in enumerate(classification):
        if cycle_type != 0:
            if current_type != cycle_type:
                if current_start is not None:
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

    if current_start is not None:
        cycle_data = {
            'start_idx': current_start,
            'end_idx': len(classification) - 1,
            'cycle_type': 'up' if current_type == 1 else 'down',
            'length': len(classification) - current_start
        }
        extracted_cycles.append(cycle_data)

    return extracted_cycles


# 선택적 컬럼: CSV에 있으면 candle_record에 포함, 없으면 조용히 생략.
# 앞으로 새 지표가 추가될 때 이 목록만 업데이트하면 된다.
_OPTIONAL_CANDLE_COLS = ['volume_delta', 'cvd_rolling']


def create_cycle_records_v3(
    data: pd.DataFrame,
    cycles: List[Dict],
    timeframe: str,
    algorithm_name: str,
    funding_rate_df: Optional[pd.DataFrame] = None,
):
    """수정된 알고리즘으로 사이클별 레코드 생성"""
    cycle_records = []
    feature_calculator = CycleFeatureCalculator()

    for i, cycle in enumerate(cycles):
        start_idx = cycle.start_idx if hasattr(cycle, 'start_idx') else cycle['start_idx']
        end_idx = cycle.end_idx if hasattr(cycle, 'end_idx') else cycle['end_idx']
        cycle_type = cycle.cycle_type if hasattr(cycle, 'cycle_type') else cycle['cycle_type']

        if cycle_type == 'rising':
            cycle_type = 'up'
        elif cycle_type == 'falling':
            cycle_type = 'down'

        cycle_data = data.iloc[start_idx:end_idx + 1].copy()

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
                except Exception:
                    timestamp_str = f"candle_{start_idx + idx}"

            # 필수 컬럼은 명시적으로 매핑
            # volume은 CSV 출처에 따라 컬럼명이 다를 수 있으므로 우선순위 탐색
            candle_record = {
                'timestamp':   timestamp_str,
                'open':        float(row.get('open', 0)),
                'high':        float(row.get('high', 0)),
                'low':         float(row.get('low', 0)),
                'close':       float(row.get('close', 0)),
                'volume':      float(row.get('volume', row.get('Volume USD', 0))),
                'macd':        float(row.get('macd', 0)),
                'macd_signal': float(row.get('macd_signal', 0)),
                'macd_hist':   float(row.get('macd_hist', 0)),
                'rsi':         float(row.get('rsi', 50.0)) if pd.notna(row.get('rsi')) else 50.0,
            }

            # 선택적 컬럼: CSV에 값이 있으면 포함, 없거나 NaN이면 생략
            # feature_extract.py가 컬럼 없는 경우 NaN으로 폴백 처리함
            for col in _OPTIONAL_CANDLE_COLS:
                if col in row.index and pd.notna(row[col]):
                    candle_record[col] = float(row[col])

            candle_data.append(candle_record)

        if candle_data:
            start_date_str = candle_data[0]['timestamp']
            end_date_str = candle_data[-1]['timestamp']
        else:
            start_date_str = f"idx_{start_idx}"
            end_date_str = f"idx_{end_idx}"

        try:
            if candle_data and candle_data[0]['timestamp'] != f"candle_{start_idx}":
                date_part = candle_data[0]['timestamp'][:10].replace('-', '').replace(' ', '').replace(':', '')
                cycle_id = f"cycle_{timeframe}_{date_part}_{i+1:03d}"
            else:
                cycle_id = f"cycle_{timeframe}_idx{start_idx}_{i+1:03d}"
        except Exception:
            cycle_id = f"cycle_{timeframe}_idx{start_idx}_{i+1:03d}"

        start_macd = candle_data[0]['macd'] if candle_data else 0
        hist_direction = 'rising' if cycle_type == 'up' else 'falling'
        macd_zone = 'positive' if start_macd >= 0 else 'negative'
        category = f"{hist_direction}_{macd_zone}"

        # context_data 구성: funding_rate 조회
        context_data: Dict = {}
        if funding_rate_df is not None:
            try:
                fr = StructuredCycleProcessor.get_funding_rate_at(funding_rate_df, start_date_str)
                if fr is not None:
                    context_data['funding_rate'] = fr
            except Exception:
                pass  # funding_rate는 선택적 특징이므로 실패해도 무시

        cycle_features = feature_calculator.extract_features_from_candle_data(
            candle_data,
            context_data=context_data,
        )

        # 빈 카테고리 dict 제거 (PyArrow Parquet 저장 오류 방지)
        cycle_features = _prune_empty_cycle_features(cycle_features)

        cycle_record = {
            'cycle_id':        cycle_id,
            'timeframe':       timeframe,
            'start_date':      start_date_str,
            'end_date':        end_date_str,
            'cycle_type':      cycle_type,
            'duration_candles': len(cycle_data),
            'category':        category,
            'algorithm_used':  algorithm_name,
            'candle_data':     candle_data,
            'cycle_features':  cycle_features,
        }

        cycle_records.append(cycle_record)

    return cycle_records


def detect_cycles_for_timeframe_v3(
    file_path,
    timeframe,
    algorithm,
    config,
    funding_rate_df: Optional[pd.DataFrame] = None,
):
    """수정된 알고리즘으로 특정 타임프레임에 대해 사이클 감지"""
    print(f"\n🔄 {timeframe} 처리 중 (수정된 알고리즘)...")

    if file_path.suffix == '.parquet':
        data = pd.read_parquet(file_path)
    elif file_path.suffix == '.csv':
        data = pd.read_csv(file_path, index_col=0)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    print(f"📊 데이터 로드: {len(data)} 행")

    original_index = data.index.copy()

    initial_rows = len(data)
    valid_mask = data['macd_hist'].notna()
    data = data[valid_mask].copy()
    original_index = original_index[valid_mask]
    data.reset_index(drop=True, inplace=True)

    print(f"📊 유효 데이터: {len(data)} 행 (제거: {initial_rows - len(data)} 행)")

    cycles, classification = algorithm.detect_cycles(data, config)
    print(f"✅ 사이클 감지 완료: {len(cycles)}개")

    classification_clean = classification.fillna(0).astype(int)

    if len(classification_clean) != len(data):
        print(f"⚠️  길이 불일치: 데이터({len(data)}) vs 분류({len(classification_clean)})")
        full_classification = pd.Series(0, index=range(len(data)), dtype=int)
        copy_length = min(len(classification_clean), len(data))
        if copy_length > 0:
            full_classification.iloc[:copy_length] = classification_clean.iloc[:copy_length]
        classification_clean = full_classification

    data_with_original_index = data.copy()
    data_with_original_index.index = original_index

    cycle_records = create_cycle_records_v3(
        data_with_original_index, cycles, timeframe, algorithm.name, funding_rate_df
    )

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

    output_path = project_root / "data" / "cycle_data" / "structured"
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(cycle_records)

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
    print("=" * 60)

    algorithm, config = load_algorithm()
    print(f"✅ 수정된 알고리즘 로드: {algorithm.name}")

    # 펀딩비 데이터 로드 (선택적)
    funding_rate_df = None
    funding_rate_path = project_root / "data" / "base_data" / "BTCUSDT_funding_rate.csv"
    if funding_rate_path.exists():
        funding_rate_df = StructuredCycleProcessor.load_funding_rate(funding_rate_path)
        print(f"✅ 펀딩비 데이터 로드: {len(funding_rate_df) if funding_rate_df is not None else 0}행")
    else:
        print(f"⚠️  펀딩비 CSV 없음 ({funding_rate_path.name}) → funding_rate 특징은 None으로 저장됨")

    timeframe_files = find_timeframe_files()

    if not timeframe_files:
        print("❌ 처리할 타임프레임 파일을 찾을 수 없습니다")
        return

    print(f"\n📊 처리 대상: {len(timeframe_files)}개 타임프레임")

    results = {}
    total_cycles = 0

    for timeframe, file_path in timeframe_files.items():
        try:
            cycle_records, cycle_count = detect_cycles_for_timeframe_v3(
                file_path, timeframe, algorithm, config, funding_rate_df
            )

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

    print("\n" + "=" * 60)
    print("🎉 Fixed MACD Detection 완료!")
    print("=" * 60)
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
    print("=" * 60)

    hist_data = [-1133.12, -1265.86, -1352.62, -1403.18, -1038.2, -685.197, -434.487, -327.767,
                 32.59714, 270.226, 336.4961, 403.8349, 184.2999, 303.8576, 302.3759, 351.2473,
                 459.427, 469.2245, 300.8355, 234.2247, -301.629]

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

    algorithm, config = load_algorithm()
    cycles, classification = algorithm.detect_cycles(df, config)

    for i, cycle in enumerate(cycles):
        print(f"사이클 {i+1}: {cycle.cycle_type} | 인덱스 {cycle.start_idx}-{cycle.end_idx} | 길이 {cycle.length}")

    print(f"\n📋 분류 결과:")
    print(f"분류: {classification.tolist()}")

    print(f"\n🎯 기대 결과:")
    print(f"인덱스 3-17이 하나의 상승 사이클로 분류되어야 함")

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