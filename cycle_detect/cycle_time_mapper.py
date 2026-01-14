"""
cycle_hierarchy_mapper.py
시간대별 사이클 간의 상위/하위 관계를 매핑하는 스크립트
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class CycleHierarchyMapper:
    """사이클 간의 계층적 관계를 매핑하는 클래스"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.timeframe_order = ['1m', '1w', '1d', '4h', '1h']  # 상위 -> 하위
        self.cycles_by_timeframe = {}
        self.hierarchy_map = {}
        
    def load_all_cycles(self) -> Dict[str, pd.DataFrame]:
        """모든 타임프레임의 사이클 데이터 로드"""
        print("📂 사이클 데이터 로딩 중...")
        
        for tf in self.timeframe_order:
            file_path = self.data_dir / f"cycles_{tf}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                
                # 날짜 데이터를 datetime으로 변환 (Unix timestamp 지원)
                df['start_datetime'] = self._safe_datetime_convert(df['start_date'])
                df['end_datetime'] = self._safe_datetime_convert(df['end_date'])
                
                self.cycles_by_timeframe[tf] = df
                print(f"  ✅ {tf}: {len(df)}개 사이클 로드")
            else:
                print(f"  ⚠️ {tf}: 파일 없음")
                
        return self.cycles_by_timeframe
    
    def _safe_datetime_convert(self, date_series):
        """날짜 데이터를 안전하게 datetime으로 변환 (Unix timestamp 지원)"""
        try:
            # 먼저 일반적인 문자열 날짜로 변환 시도
            return pd.to_datetime(date_series)
        except (ValueError, TypeError):
            try:
                # Unix timestamp (초)로 변환 시도
                return pd.to_datetime(date_series, unit='s')
            except (ValueError, TypeError):
                try:
                    # Unix timestamp (밀리초)로 변환 시도
                    return pd.to_datetime(date_series, unit='ms')
                except (ValueError, TypeError):
                    # 모든 변환이 실패하면 원본 반환
                    print(f"⚠️ 날짜 변환 실패: {date_series.iloc[0] if len(date_series) > 0 else 'N/A'}")
                    return date_series
    
    def check_overlap(self, cycle1_start: datetime, cycle1_end: datetime,
                     cycle2_start: datetime, cycle2_end: datetime) -> float:
        """
        두 사이클 간의 겹침 비율 계산
        Returns: 겹침 비율 (0.0 ~ 1.0)
        """
        # 겹치는 부분이 없으면 0 반환
        if cycle1_end < cycle2_start or cycle2_end < cycle1_start:
            return 0.0
        
        # 겹치는 구간 계산
        overlap_start = max(cycle1_start, cycle2_start)
        overlap_end = min(cycle1_end, cycle2_end)
        
        # 하위 사이클(cycle1) 기준으로 겹침 비율 계산
        cycle1_duration = (cycle1_end - cycle1_start).total_seconds()
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        
        if cycle1_duration == 0:
            return 0.0
            
        return overlap_duration / cycle1_duration
    
    def find_parent_cycles(self, child_cycle: pd.Series, parent_df: pd.DataFrame, 
                          min_overlap: float = 0.01) -> List[str]:
        """
        하위 사이클에 대한 상위 사이클 찾기
        min_overlap: 최소 겹침 비율 (기본값 1% 이상 겹치면 포함)
        """
        parent_cycles = []
        
        child_start = child_cycle['start_datetime']
        child_end = child_cycle['end_datetime']
        
        for _, parent_cycle in parent_df.iterrows():
            parent_start = parent_cycle['start_datetime']
            parent_end = parent_cycle['end_datetime']
            
            overlap_ratio = self.check_overlap(
                child_start, child_end,
                parent_start, parent_end
            )
            
            if overlap_ratio >= min_overlap:
                parent_cycles.append(parent_cycle['cycle_id'])
                
        return parent_cycles
    
    def find_child_cycles(self, parent_cycle: pd.Series, child_df: pd.DataFrame,
                         min_overlap: float = 0.01) -> List[str]:
        """
        상위 사이클에 포함되는 하위 사이클 찾기
        """
        child_cycles = []
        
        parent_start = parent_cycle['start_datetime']
        parent_end = parent_cycle['end_datetime']
        
        for _, child_cycle in child_df.iterrows():
            child_start = child_cycle['start_datetime']
            child_end = child_cycle['end_datetime']
            
            overlap_ratio = self.check_overlap(
                child_start, child_end,
                parent_start, parent_end
            )
            
            if overlap_ratio >= min_overlap:
                child_cycles.append(child_cycle['cycle_id'])
                
        return child_cycles
    
    def build_hierarchy_map(self, min_overlap: float = 0.01):
        """
        계층 관계 맵 생성 (교차 조인 방식으로 성능 극대화)
        """
        print("\n🔗 사이클 계층 관계 매핑 중 (고속화 버전)...")

        # 각 시간대별로 딕셔너리를 미리 초기화합니다.
        for timeframe in self.timeframe_order:
            if self.cycles_by_timeframe.get(timeframe) is not None:
                self.hierarchy_map[timeframe] = {}
        
        # 1. 모든 시간대 데이터 로드 및 맵 초기화
        for timeframe in self.timeframe_order:
            df = self.cycles_by_timeframe.get(timeframe)
            if df is not None:
                for _, cycle in df.iterrows():
                    self.hierarchy_map[timeframe][cycle['cycle_id']] = {
                        'cycle_type': cycle['cycle_type'],
                        'start_date': cycle['start_datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                        'end_date': cycle['end_datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                        'duration_candles': int(cycle['duration_candles']),
                        'parent_cycle_ids': {},
                        'child_cycle_ids': {}
                    }
        
        # 2. Top-Down 방식으로 관계 매핑 (1w -> 1d, 1d -> 4h ...)
        for i, parent_tf in enumerate(self.timeframe_order):
            if parent_tf not in self.cycles_by_timeframe:
                continue

            for child_tf in self.timeframe_order[i+1:]:
                if child_tf not in self.cycles_by_timeframe:
                    continue

                print(f"  🔄 {parent_tf}와 {child_tf} 관계 분석 중...")
                parent_df = self.cycles_by_timeframe[parent_tf]
                child_df = self.cycles_by_timeframe[child_tf]

                # 교차 조인을 위한 임시 키 생성
                parent_df['_key'] = 1
                child_df['_key'] = 1

                # 모든 조합 생성 (Cross Join)
                combined_df = pd.merge(parent_df, child_df, on='_key', suffixes=('_parent', '_child')).drop('_key', axis=1)

                if combined_df.empty:
                    continue

                # 겹침 조건 필터링
                # 자식 사이클이 부모 사이클에 완전히 포함되는 경우
                overlap_mask = (combined_df['start_datetime_child'] <= combined_df['end_datetime_parent']) & \
                        (combined_df['start_datetime_parent'] <= combined_df['end_datetime_child'])

                relationships = combined_df[overlap_mask]

                # 관계 설정
                for _, row in relationships.iterrows():
                    parent_id = row['cycle_id_parent']
                    child_id = row['cycle_id_child']

                    # 부모 -> 자식 관계 추가
                    if child_tf not in self.hierarchy_map[parent_tf][parent_id]['child_cycle_ids']:
                        self.hierarchy_map[parent_tf][parent_id]['child_cycle_ids'][child_tf] = []
                    self.hierarchy_map[parent_tf][parent_id]['child_cycle_ids'][child_tf].append(child_id)

                    # 자식 -> 부모 관계 추가
                    if parent_tf not in self.hierarchy_map[child_tf][child_id]['parent_cycle_ids']:
                        self.hierarchy_map[child_tf][child_id]['parent_cycle_ids'][parent_tf] = []
                    self.hierarchy_map[child_tf][child_id]['parent_cycle_ids'][parent_tf].append(parent_id)

        print("\n✅ 모든 관계 매핑 완료!")
        self._print_statistics()
        return self.hierarchy_map
    
    def _print_statistics(self):
        """계층 구조 통계 출력"""
        print("\n📊 계층 구조 통계:")
        
        for tf in self.timeframe_order:
            if tf not in self.hierarchy_map:
                continue
                
            total_cycles = len(self.hierarchy_map[tf])
            has_parent = sum(1 for c in self.hierarchy_map[tf].values() 
                           if c['parent_cycle_ids'])
            has_child = sum(1 for c in self.hierarchy_map[tf].values() 
                          if c['child_cycle_ids'])
            
            print(f"\n  {tf} 타임프레임:")
            print(f"    - 전체 사이클: {total_cycles}개")
            
            # 0으로 나누기 방지
            if total_cycles > 0:
                print(f"    - 상위 사이클 있음: {has_parent}개 ({has_parent/total_cycles*100:.1f}%)")
                print(f"    - 하위 사이클 있음: {has_child}개 ({has_child/total_cycles*100:.1f}%)")
            else:
                print(f"    - 상위 사이클 있음: {has_parent}개 (0.0%)")
                print(f"    - 하위 사이클 있음: {has_child}개 (0.0%)")
    
    def save_hierarchy_map(self, output_path: Path = None):
        """계층 구조를 JSON 파일로 저장"""
        if output_path is None:
            output_path = self.data_dir / "cycle_hierarchy_map.json"
        
        # datetime 객체 제거 (JSON 직렬화를 위해)
        clean_map = {}
        for tf, cycles in self.hierarchy_map.items():
            clean_map[tf] = {}
            for cycle_id, info in cycles.items():
                clean_map[tf][cycle_id] = {
                    k: v for k, v in info.items()
                    if not isinstance(v, (pd.Timestamp, datetime))
                }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_map, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 계층 구조 저장 완료: {output_path}")
        return output_path
    
    def get_cycle_family(self, cycle_id: str) -> Dict:
        """
        특정 사이클의 전체 가족 관계 조회
        (부모, 자식, 형제 등)
        """
        # 사이클이 속한 타임프레임 찾기
        cycle_tf = None
        for tf, cycles in self.hierarchy_map.items():
            if cycle_id in cycles:
                cycle_tf = tf
                break
        
        if not cycle_tf:
            return {}
        
        cycle_info = self.hierarchy_map[cycle_tf][cycle_id]
        
        # 형제 사이클 찾기 (같은 부모를 가진 사이클)
        siblings = defaultdict(list)
        for parent_tf, parent_ids in cycle_info['parent_cycle_ids'].items():
            for parent_id in parent_ids:
                # 부모의 자식들 중 자신을 제외한 나머지
                parent_info = self.hierarchy_map[parent_tf][parent_id]
                if cycle_tf in parent_info['child_cycle_ids']:
                    for sibling_id in parent_info['child_cycle_ids'][cycle_tf]:
                        if sibling_id != cycle_id:
                            siblings[parent_tf].append(sibling_id)
        
        return {
            'cycle_id': cycle_id,
            'timeframe': cycle_tf,
            'parents': cycle_info['parent_cycle_ids'],
            'children': cycle_info['child_cycle_ids'],
            'siblings': dict(siblings)
        }
    
    def analyze_cycle_patterns(self) -> Dict:
        """
        사이클 패턴 분석
        예: 상위 상승 사이클 내의 하위 하락 사이클 개수 등
        """
        patterns = {
            'alignment': {},  # 상위와 같은 방향
            'counter': {}     # 상위와 반대 방향
        }
        
        for tf_idx, child_tf in enumerate(self.timeframe_order[:-1]):
            if child_tf not in self.hierarchy_map:
                continue
                
            parent_tf = self.timeframe_order[tf_idx + 1]
            if parent_tf not in self.hierarchy_map:
                continue
            
            alignment_count = 0
            counter_count = 0
            
            for cycle_id, cycle_info in self.hierarchy_map[child_tf].items():
                child_type = cycle_info['cycle_type']
                
                if parent_tf in cycle_info['parent_cycle_ids']:
                    for parent_id in cycle_info['parent_cycle_ids'][parent_tf]:
                        parent_type = self.hierarchy_map[parent_tf][parent_id]['cycle_type']
                        
                        if child_type == parent_type:
                            alignment_count += 1
                        else:
                            counter_count += 1
            
            patterns['alignment'][f"{child_tf}_in_{parent_tf}"] = alignment_count
            patterns['counter'][f"{child_tf}_in_{parent_tf}"] = counter_count
        
        return patterns


def main():
    """메인 실행 함수"""
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "cycle_data" / "structured"
    
    print("="*60)
    print("🚀 사이클 계층 구조 매핑 시작")
    print("="*60)
    print("\n📌 타임프레임 계층 구조:")
    print("  1h (1시간) → 4h (4시간) → 1d (1일) → 1w (1주) → 1m (1개월)")
    print("  작은 단위                                      큰 단위")
    print("\n📝 관계 설명:")
    print("  - 1m (월): 최상위, parent 없음, 모든 하위 타임프레임 포함")
    print("  - 1w (주): 1m의 child, 1d/4h/1h의 parent")
    print("  - 1d (일): 1m/1w의 child, 4h/1h의 parent")
    print("  - 4h (4시간): 1m/1w/1d의 child, 1h의 parent")
    print("  - 1h (시간): 모든 상위 타임프레임의 child, parent 없음")
    
    # 매퍼 초기화
    mapper = CycleHierarchyMapper(data_dir)
    
    # 모든 사이클 로드
    mapper.load_all_cycles()
    
    # 계층 구조 생성
    hierarchy_map = mapper.build_hierarchy_map(min_overlap=0.01)  # 1% 이상 겹치면 포함
    
    # 결과 저장
    output_path = mapper.save_hierarchy_map()
    
    # 계층 구조 검증
    print("\n🔍 계층 구조 검증:")
    
    # 1m 사이클 검증
    if '1m' in mapper.hierarchy_map and mapper.hierarchy_map['1m']:
        sample_1m = list(mapper.hierarchy_map['1m'].keys())[0]
        info_1m = mapper.hierarchy_map['1m'][sample_1m]
        print(f"\n  ✅ 1m (월) 사이클 예시: {sample_1m}")
        print(f"     Parent: {len(info_1m['parent_cycle_ids'])}개 (0이어야 함)")
        if info_1m['child_cycle_ids']:
            for child_tf, child_ids in sorted(info_1m['child_cycle_ids'].items()):
                print(f"     Child {child_tf}: {len(child_ids)}개")
    
    # 1w 사이클 검증
    if '1w' in mapper.hierarchy_map and mapper.hierarchy_map['1w']:
        sample_1w = list(mapper.hierarchy_map['1w'].keys())[0]
        info_1w = mapper.hierarchy_map['1w'][sample_1w]
        print(f"\n  ✅ 1w (주) 사이클 예시: {sample_1w}")
        if info_1w['parent_cycle_ids']:
            for parent_tf, parent_ids in info_1w['parent_cycle_ids'].items():
                print(f"     Parent {parent_tf}: {len(parent_ids)}개")
        if info_1w['child_cycle_ids']:
            for child_tf, child_ids in sorted(info_1w['child_cycle_ids'].items()):
                print(f"     Child {child_tf}: {len(child_ids)}개")
    
    # 1h 사이클 검증
    if '1h' in mapper.hierarchy_map and mapper.hierarchy_map['1h']:
        sample_1h = list(mapper.hierarchy_map['1h'].keys())[0]
        info_1h = mapper.hierarchy_map['1h'][sample_1h]
        print(f"\n  ✅ 1h (시간) 사이클 예시: {sample_1h}")
        if info_1h['parent_cycle_ids']:
            for parent_tf, parent_ids in sorted(info_1h['parent_cycle_ids'].items()):
                print(f"     Parent {parent_tf}: {len(parent_ids)}개")
        print(f"     Child: {len(info_1h['child_cycle_ids'])}개 (0이어야 함)")
    
    # 패턴 분석
    print("\n📈 사이클 패턴 분석:")
    patterns = mapper.analyze_cycle_patterns()
    
    print("\n  정렬된 사이클 (상위와 같은 방향):")
    for key, count in patterns['alignment'].items():
        total = count + patterns['counter'].get(key, 0)
        if total > 0:
            child_tf, parent_tf = key.split('_in_')
            print(f"    {child_tf} → {parent_tf}: {count}개 ({count/total*100:.1f}%)")
    
    print("\n  역행 사이클 (상위와 반대 방향):")
    for key, count in patterns['counter'].items():
        total = count + patterns['alignment'].get(key, 0)
        if total > 0:
            child_tf, parent_tf = key.split('_in_')
            print(f"    {child_tf} → {parent_tf}: {count}개 ({count/total*100:.1f}%)")
    
    print("\n✅ 계층 구조 매핑 완료!")
    print(f"📁 결과 파일: {output_path}")
    
    return hierarchy_map


if __name__ == "__main__":
    main()