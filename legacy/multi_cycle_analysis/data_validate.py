"""
데이터 구조 검증 및 테스트 스크립트
====================================
계층적 분석기 실행 전 데이터 구조와 접근성을 확인하는 도구
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

def check_data_structure(base_path="C:/Users/Administrator/Desktop/macd_cycle_analysis_project"):
    """데이터 구조 및 접근성 검증"""
    
    base_path = Path(base_path)
    data_path = base_path / "data" / "cycle_data" / "structured"
    
    print("🔍 데이터 구조 검증 시작")
    print("=" * 60)
    
    # 1. 파일 존재 여부 확인
    timeframes = ['1m', '1w', '1d', '4h', '1h']
    available_files = []
    
    print("\n📁 파일 존재 여부 확인:")
    for tf in timeframes:
        file_path = data_path / f"cycles_{tf}.parquet"
        if file_path.exists():
            available_files.append(tf)
            print(f"  ✅ cycles_{tf}.parquet")
        else:
            print(f"  ❌ cycles_{tf}.parquet")
    
    # 2. 계층 관계 맵 확인
    hierarchy_map_path = data_path / "cycle_hierarchy_map.json"
    hierarchy_exists = hierarchy_map_path.exists()
    print(f"\n🗺️  계층 관계 맵: {'✅' if hierarchy_exists else '❌'} {hierarchy_map_path}")
    
    if not hierarchy_exists:
        print("❌ 계층 관계 맵이 없어 계층적 분석이 불가능합니다.")
        return False
    
    # 3. 샘플 데이터 로드 및 구조 확인
    if available_files:
        sample_tf = available_files[0]
        print(f"\n📊 샘플 데이터 구조 확인 ({sample_tf}):")
        
        try:
            df = pd.read_parquet(data_path / f"cycles_{sample_tf}.parquet")
            print(f"  - 총 사이클 수: {len(df)}")
            print(f"  - 컬럼: {list(df.columns)}")
            
            # 샘플 사이클의 구조 확인
            if len(df) > 0:
                sample_cycle = df.iloc[0]
                print(f"  - 샘플 사이클 ID: {sample_cycle['cycle_id']}")
                print(f"  - 사이클 타입: {sample_cycle['cycle_type']}")
                print(f"  - Duration: {sample_cycle['duration_candles']}")
                
                # cycle_features 구조 확인
                if 'cycle_features' in df.columns:
                    features = sample_cycle['cycle_features']
                    if isinstance(features, dict):
                        print(f"  - 특징 카테고리: {list(features.keys())}")
                    else:
                        print(f"  - 특징 데이터 타입: {type(features)}")
                
        except Exception as e:
            print(f"  ❌ 데이터 로드 실패: {e}")
            return False
    
    # 4. 계층 관계 맵 구조 확인
    if hierarchy_exists:
        print(f"\n🔗 계층 관계 구조 확인:")
        try:
            with open(hierarchy_map_path, 'r', encoding='utf-8') as f:
                hierarchy_map = json.load(f)
            
            print(f"  - 시간대 수: {len(hierarchy_map)}")
            
            for tf in available_files:
                if tf in hierarchy_map:
                    tf_cycles = hierarchy_map[tf]
                    print(f"  - {tf}: {len(tf_cycles)} 사이클")
                    
                    # 샘플 사이클의 관계 확인
                    if tf_cycles:
                        sample_cycle_id = list(tf_cycles.keys())[0]
                        sample_cycle_data = tf_cycles[sample_cycle_id]
                        
                        child_ids = sample_cycle_data.get('child_cycle_ids', {})
                        child_count = sum(len(ids) for ids in child_ids.values())
                        
                        print(f"    예시: {sample_cycle_id}")
                        print(f"    하위 사이클 시간대: {list(child_ids.keys())}")
                        print(f"    총 하위 사이클 수: {child_count}")
                else:
                    print(f"  - {tf}: 계층 맵에서 누락")
                    
        except Exception as e:
            print(f"  ❌ 계층 맵 로드 실패: {e}")
            return False
    
    print(f"\n✅ 데이터 구조 검증 완료")
    print(f"분석 가능한 시간대: {available_files}")
    
    return True

def test_basic_analysis(base_path="C:/Users/Administrator/Desktop/macd_cycle_analysis_project"):
    """기본 분석 기능 테스트"""
    
    print("\n🧪 기본 분석 기능 테스트")
    print("=" * 60)
    
    try:
        # HierarchicalCycleAnalyzer import 시도
        import sys
        sys.path.append(str(Path(base_path)))
        
        from hierarchical_cycle_analyzer import HierarchicalCycleAnalyzer
        
        # 분석기 초기화
        analyzer = HierarchicalCycleAnalyzer(base_path)
        
        # 사용 가능한 시간대 확인
        available_timeframes = analyzer.get_available_timeframes()
        print(f"✅ 사용 가능한 시간대: {available_timeframes}")
        
        if not available_timeframes:
            print("❌ 분석 가능한 데이터가 없습니다.")
            return False
        
        # 첫 번째 시간대로 테스트
        test_timeframe = available_timeframes[0]
        print(f"🔍 테스트 시간대: {test_timeframe}")
        
        # 데이터 로드 테스트
        df = analyzer.load_timeframe_data(test_timeframe)
        if df is None:
            print("❌ 데이터 로드 실패")
            return False
        
        print(f"✅ 데이터 로드 성공: {len(df)} 사이클")
        
        # 첫 번째 사이클로 하위 사이클 추출 테스트
        if len(df) > 0:
            test_cycle_id = df.iloc[0]['cycle_id']
            print(f"🔍 테스트 사이클: {test_cycle_id}")
            
            child_data = analyzer.extract_child_cycle_data(test_timeframe, test_cycle_id)
            
            if child_data:
                print(f"✅ 하위 사이클 추출 성공:")
                for child_tf, child_df in child_data.items():
                    print(f"  - {child_tf}: {len(child_df)} 사이클")
                
                # 기본 통계 계산 테스트
                stats = analyzer.calculate_basic_statistics(child_data)
                print(f"✅ 통계 계산 성공: {len(stats)} 시간대")
                
                return True
            else:
                print(f"⚠️  {test_cycle_id}에 대한 하위 사이클이 없습니다.")
                return True  # 데이터가 없는 것은 오류가 아님
        
    except ImportError as e:
        print(f"❌ 모듈 import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 테스트 실행 실패: {e}")
        return False

def find_cycles_with_children(base_path="C:/Users/Administrator/Desktop/macd_cycle_analysis_project"):
    """하위 사이클이 있는 상위 사이클들을 찾아서 보고"""
    
    print("\n🔎 하위 사이클이 있는 상위 사이클 찾기")
    print("=" * 60)
    
    base_path = Path(base_path)
    data_path = base_path / "data" / "cycle_data" / "structured"
    hierarchy_map_path = data_path / "cycle_hierarchy_map.json"
    
    try:
        with open(hierarchy_map_path, 'r', encoding='utf-8') as f:
            hierarchy_map = json.load(f)
        
        cycles_with_children = {}
        
        for timeframe, cycles in hierarchy_map.items():
            cycles_with_children[timeframe] = []
            
            for cycle_id, cycle_data in cycles.items():
                child_ids = cycle_data.get('child_cycle_ids', {})
                total_children = sum(len(ids) for ids in child_ids.values())
                
                if total_children > 0:
                    cycles_with_children[timeframe].append({
                        'cycle_id': cycle_id,
                        'cycle_type': cycle_data.get('cycle_type', 'unknown'),
                        'duration': cycle_data.get('duration_candles', 0),
                        'child_timeframes': list(child_ids.keys()),
                        'total_children': total_children
                    })
        
        # 결과 출력
        for timeframe, cycles in cycles_with_children.items():
            if cycles:
                print(f"\n📊 {timeframe} 시간대 ({len(cycles)}개 사이클이 하위 사이클 보유):")
                
                # 상위 5개만 출력
                for i, cycle in enumerate(cycles[:5], 1):
                    print(f"  {i}. {cycle['cycle_id']}")
                    print(f"     타입: {cycle['cycle_type']}, Duration: {cycle['duration']}")
                    print(f"     하위 시간대: {cycle['child_timeframes']}")
                    print(f"     총 하위 사이클: {cycle['total_children']}개")
                
                if len(cycles) > 5:
                    print(f"     ... 외 {len(cycles) - 5}개 더")
            else:
                print(f"\n📊 {timeframe} 시간대: 하위 사이클이 있는 사이클 없음")
        
        return cycles_with_children
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

def main():
    """메인 검증 함수"""
    print("🚀 계층적 사이클 분석기 데이터 검증 시작")
    
    # 1. 데이터 구조 검증
    if not check_data_structure():
        print("\n❌ 데이터 구조 검증 실패. 분석기 실행이 불가능합니다.")
        return
    
    # 2. 기본 분석 기능 테스트
    if not test_basic_analysis():
        print("\n❌ 기본 분석 기능 테스트 실패.")
        return
    
    # 3. 분석 가능한 사이클 찾기
    cycles_with_children = find_cycles_with_children()
    
    if cycles_with_children:
        print("\n✅ 모든 검증 완료! 계층적 분석기를 실행할 수 있습니다.")
        
        # 분석기 실행 여부 확인
        run_choice = input("\n계층적 분석기를 바로 실행하시겠습니까? (y/n): ")
        if run_choice.lower() == 'y':
            try:
                from hierarchical_cycle_analyzer import main as run_analyzer
                run_analyzer()
            except Exception as e:
                print(f"❌ 분석기 실행 실패: {e}")
    else:
        print("\n⚠️  검증은 완료되었으나 분석 가능한 데이터 확인이 필요합니다.")

if __name__ == "__main__":
    main()