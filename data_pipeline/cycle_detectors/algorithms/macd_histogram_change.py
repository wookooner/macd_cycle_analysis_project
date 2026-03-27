"""
무한루프 문제 해결된 단순 MACD 알고리즘
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SimpleConfig:
    max_opposite_consecutive: int = 2  # 반대 방향 최대 연속 개수
    min_cycle_length: int = 3         # 최소 사이클 길이


@dataclass
class CycleInfo:
    start_idx: int
    end_idx: int
    cycle_type: str  # 'up' or 'down'
    length: int


class SimpleMACDAlgorithm:
    """무한루프 문제 해결된 단순 MACD 알고리즘"""
    
    def __init__(self):
        self.name = "Simple Direction Preserving MACD Algorithm"
        self.version = "3.1"
        
    def detect_cycles(self, data: pd.DataFrame, config: SimpleConfig) -> Tuple[List[CycleInfo], pd.Series]:
        """단순한 사이클 감지"""
        
        # MACD 히스토그램 컬럼 찾기
        if 'macd_hist' not in data.columns:
            macd_columns = [col for col in data.columns if 'macd' in col.lower() and 'hist' in col.lower()]
            if macd_columns:
                macd_column = macd_columns[0]
            else:
                raise ValueError("MACD 히스토그램 컬럼을 찾을 수 없습니다.")
        else:
            macd_column = 'macd_hist'
        
        macd_hist = data[macd_column].dropna()
        original_index = macd_hist.index
        # If there's no valid MACD histogram data, return empty results
        if macd_hist.empty:
            return [], pd.Series([], index=original_index, dtype=int)
        
        #print(f"📊 단순 알고리즘 시작: {len(macd_hist)} 포인트")
        
        # 1단계: 방향성 계산 (절대 불변)
        directions = self._calculate_directions(macd_hist)
        
        # 2단계: 사이클 경계 결정 (노이즈 허용) - 수정된 로직
        cycles = self._find_cycles_fixed(directions, config)
        
        # 3단계: 원래 방향성 그대로 분류
        classification = pd.Series(directions, index=original_index)
        
        print(f"✅ 감지된 사이클: {len(cycles)}개")
        #print(f"✅ 방향성 보존 분류 완료")
        
        return cycles, classification
    
    def _calculate_directions(self, macd_hist: pd.Series) -> List[int]:
        """방향성 계산 (절대 불변)"""
        values = macd_hist.values
        directions = [0]  # 첫 번째는 항상 0
        
        for i in range(1, len(values)):
            current = float(values[i])
            previous = float(values[i-1])
            
            if current > previous:
                directions.append(1)
            elif current < previous:
                directions.append(-1)
            else:
                directions.append(0)
        
        return directions
    
    def _find_cycles_fixed(self, directions: List[int], config: SimpleConfig) -> List[CycleInfo]:
        """수정된 사이클 찾기 로직 (무한루프 방지)"""
        cycles = []
        i = 0
        
        while i < len(directions):
            # 의미있는 방향 찾기
            if directions[i] in [1, -1]:
                cycle_start = i
                main_direction = directions[i]
                cycle_type = 'rising' if main_direction == 1 else 'falling'
                
                #print(f"   사이클 시작: 인덱스 {i}, 방향 {main_direction}")
                
                # **수정된 확장 로직**
                cycle_end = self._find_cycle_end(directions, i, main_direction, config.max_opposite_consecutive)
                
                # 사이클 길이 확인
                cycle_length = cycle_end - cycle_start + 1
                
                if cycle_length >= config.min_cycle_length:
                    cycle = CycleInfo(
                        start_idx=cycle_start,
                        end_idx=cycle_end,
                        cycle_type=cycle_type,
                        length=cycle_length
                    )
                    cycles.append(cycle)
                    
                    #print(f"   사이클 {len(cycles)}: {cycle_type} | 인덱스 {cycle_start}-{cycle_end} | 길이 {cycle_length}")
                
                # **핵심 수정: 다음 위치로 안전하게 이동**
                i = cycle_end + 1
            else:
                i += 1
        
        return cycles
    
    def _find_cycle_end(self, directions: List[int], start_idx: int, main_direction: int, max_opposite: int) -> int:
        """사이클 종료 지점 찾기 (안전한 로직)"""
        current_pos = start_idx
        consecutive_opposite = 0
        last_valid_end = start_idx
        
        for j in range(start_idx + 1, len(directions)):
            current_dir = directions[j]
            
            if current_dir == main_direction:
                # 같은 방향이면 연속 카운트 리셋하고 유효한 끝 지점 업데이트
                consecutive_opposite = 0
                last_valid_end = j
                
            elif current_dir == -main_direction:
                # 반대 방향이면 연속 카운트 증가
                consecutive_opposite += 1
                
                if consecutive_opposite <= max_opposite:
                    # 허용 범위 내면 계속 진행
                    continue
                else:
                    # 허용 범위 초과하면 마지막 유효 지점에서 종료
                    break
                    
            # else: 방향이 0이면 그냥 계속
        
        return last_valid_end
    
    def get_cycle_statistics(self, cycles: List[CycleInfo]) -> Dict:
        """사이클 통계 정보 반환"""
        if not cycles:
            return {
                'total_cycles': 0,
                'up_cycles': 0,
                'down_cycles': 0,
                'avg_cycle_length': 0
            }

        # Accept both legacy labels ('up'/'down') and new labels ('rising'/'falling')
        up_cycles = [c for c in cycles if getattr(c, 'cycle_type', None) in ('up', 'rising')]
        down_cycles = [c for c in cycles if getattr(c, 'cycle_type', None) in ('down', 'falling')]
        
        return {
            'total_cycles': len(cycles),
            'up_cycles': len(up_cycles),
            'down_cycles': len(down_cycles),
            'avg_cycle_length': np.mean([c.length for c in cycles]),
            'cycle_lengths': [c.length for c in cycles]
        }


def test_fixed_algorithm():
    """수정된 알고리즘 테스트"""
    print("🧪 무한루프 수정된 알고리즘 테스트")
    print("=" * 50)
    
    # 테스트 데이터
    hist_data = [-1133.12, -1265.86, -1352.62, -1403.18, -1038.2, -685.197, 
                 -434.487, -327.767, 32.59714, 270.226, 336.4961, 403.8349, 
                 184.2999, 303.8576, 302.3759, 351.2473, 459.427, 469.2245, 
                 300.8355, 234.2247, -301.629]
    
    df = pd.DataFrame({'macd_hist': hist_data})
    
    algorithm = SimpleMACDAlgorithm()
    config = SimpleConfig(max_opposite_consecutive=2, min_cycle_length=3)
    
    print("알고리즘 실행 시작...")
    cycles, classification = algorithm.detect_cycles(df, config)
    print("알고리즘 실행 완료!")
    
    print(f"\n📋 방향성 (절대 불변):")
    print(f"   {classification.tolist()}")
    
    # 예상 결과 확인
    expected_directions = [0]
    values = hist_data
    for i in range(1, len(values)):
        if values[i] > values[i-1]:
            expected_directions.append(1)
        elif values[i] < values[i-1]:
            expected_directions.append(-1)
        else:
            expected_directions.append(0)
    
    print(f"\n🎯 기대 방향성:")
    print(f"   {expected_directions}")
    
    if classification.tolist() == expected_directions:
        print("✅ 성공! 방향성이 올바르게 보존되었습니다.")
    else:
        print("❌ 방향성 계산에 문제가 있습니다.")
    
    # 문제 구간 확인 (인덱스 12-14)
    if len(classification) > 14:
        problem_area = classification.iloc[12:15].tolist()
        expected_problem = expected_directions[12:15]
        print(f"\n🔍 문제 구간 (인덱스 12-14):")
        print(f"   실제: {problem_area}")
        print(f"   기대: {expected_problem}")
        
        if problem_area == expected_problem:
            print("   ✅ 문제 구간도 올바르게 계산됨! [-1, +1, -1]")
        else:
            print("   ❌ 문제 구간에 오류 있음")
    
    # 통계 출력
    stats = algorithm.get_cycle_statistics(cycles)
    print(f"\n📊 통계:")
    print(f"   총 사이클: {stats['total_cycles']}개")
    print(f"   상승 사이클: {stats['up_cycles']}개")
    print(f"   하락 사이클: {stats['down_cycles']}개")
    print(f"   평균 길이: {stats['avg_cycle_length']:.1f}")
    
    return cycles, classification


# 기존 인터페이스 호환성을 위한 함수들
def create_algorithm():
    """기존 인터페이스 호환"""
    return SimpleMACDAlgorithm()


def create_default_config():
    """기존 인터페이스 호환"""
    return SimpleConfig(max_opposite_consecutive=2, min_cycle_length=3)


if __name__ == "__main__":
    test_fixed_algorithm()