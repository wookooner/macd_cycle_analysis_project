import pandas as pd
import json
import os
from datetime import datetime

def flatten_features(df):
    """
    DataFrame의 중첩된 'cycle_features' 딕셔너리를 평탄화하여 개별 컬럼으로 변환합니다.
    'cycle_features'가 딕셔너리가 아닌 경우 원본 DataFrame을 반환합니다.
    """
    if 'cycle_features' not in df.columns or df['cycle_features'].apply(isinstance, args=(dict,)).sum() == 0:
        return df

    # cycle_features가 딕셔너리인 행만 정규화
    features_normalized = pd.json_normalize(df['cycle_features'][df['cycle_features'].apply(isinstance, args=(dict,))])
    
    # 원래 DataFrame에서 cycle_features 열을 삭제하고 정규화된 데이터를 병합
    df_flat = df.drop(columns=['cycle_features']).join(features_normalized)
    
    return df_flat

def apply_filters(df, conditions):
    """
    주어진 필터 조건 리스트를 DataFrame에 적용합니다.
    """
    if not conditions:
        return df
    
    filtered_df = df.copy()
    for condition in conditions:
        feature = condition['feature']
        op = condition['op']
        value = condition['value']
        
        if feature not in filtered_df.columns:
            print(f"경고: '{feature}' 컬럼을 찾을 수 없습니다. 이 필터는 건너뜁니다.")
            continue
            
        if op == '>':
            filtered_df = filtered_df[filtered_df[feature] > value]
        elif op == '<':
            filtered_df = filtered_df[filtered_df[feature] < value]
        elif op == '>=':
            filtered_df = filtered_df[filtered_df[feature] >= value]
        elif op == '<=':
            filtered_df = filtered_df[filtered_df[feature] <= value]
        elif op == '==':
            filtered_df = filtered_df[filtered_df[feature] == value]
        elif op == '!=':
            filtered_df = filtered_df[filtered_df[feature] != value]
            
    return filtered_df

def filter_cycles_hierarchically(filters, base_path='.'):
    """
    계층적 관계와 특징을 기반으로 사이클 데이터를 필터링하고 결과를 Parquet과 JSON 파일로 저장합니다.

    Args:
        filters (dict): 시간대별 필터 조건을 담은 딕셔너리.
                        예: {'1w': [{'feature': 'change.price_pct', 'op': '>', 'value': 10}],
                             '1d': [{'feature': 'shape.duration_candles', 'op': '>=', 'value': 5}]}
        base_path (str): 프로젝트 루트 경로.

    Returns:
        pandas.DataFrame: 필터링된 모든 사이클 데이터가 통합된 DataFrame.
    """
    data_path = os.path.join(base_path, 'data', 'cycle_data', 'structured')
    hierarchy_map_path = os.path.join(data_path, 'cycle_hierarchy_map.json')

    # 1. 데이터 로드
    print("데이터 로드 중...")
    try:
        with open(hierarchy_map_path, 'r') as f:
            hierarchy_map = json.load(f)
    except FileNotFoundError:
        print(f"에러: 계층 구조 파일 '{hierarchy_map_path}'을 찾을 수 없습니다.")
        return pd.DataFrame()

    timeframes = ['1w', '1d', '4h', '1h', '1m']
    cycle_data = {}
    for tf in timeframes:
        file_path = os.path.join(data_path, f'cycles_{tf}.parquet')
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            cycle_data[tf] = flatten_features(df)
            print(f"- {tf} 사이클 데이터 로드 완료 ({len(df)}개)")
        else:
            print(f"- {tf} 사이클 파일이 없어 건너뜁니다.")

    # 2. 계층적 필터링 수행
    print("\n계층적 필터링 시작...")
    
    filter_order = [tf for tf in ['1w', '1d', '4h', '1h'] if tf in cycle_data]
    allowed_child_ids = None
    all_filtered_dfs = []

    for i, current_tf in enumerate(filter_order):
        print(f"--- {current_tf} 시간대 필터링 ---")
        current_df = cycle_data[current_tf]
        
        if allowed_child_ids is not None:
            current_df = current_df[current_df['cycle_id'].isin(allowed_child_ids)]
            print(f"상위 사이클에 의해 {len(current_df)}개 후보로 축소됨")

        tf_filters = filters.get(current_tf, [])
        if tf_filters:
            filtered_df = apply_filters(current_df, tf_filters)
            print(f"'{current_tf}' 필터 적용 후 {len(filtered_df)}개 사이클 통과")
        else:
            filtered_df = current_df
            print("사용자 정의 필터 없음. 모든 후보 사이클 통과")
        
        all_filtered_dfs.append(filtered_df)
        
        next_tf = filter_order[i+1] if i + 1 < len(filter_order) else None
        if not next_tf:
            continue

        parent_cycle_ids = set(filtered_df['cycle_id'])
        allowed_child_ids = set()
        hierarchy_tf_map = hierarchy_map.get(current_tf, {})
        
        for parent_id in parent_cycle_ids:
            child_ids_map = hierarchy_tf_map.get(parent_id, {}).get('child_cycle_ids', {})
            allowed_child_ids.update(child_ids_map.get(next_tf, []))
        
        print(f"다음 시간대({next_tf})를 위한 {len(allowed_child_ids)}개의 하위 사이클 ID 준비 완료")

    # 3. 결과 통합 및 저장
    if not all_filtered_dfs:
        print("\n필터링된 사이클이 없습니다.")
        return pd.DataFrame()

    final_df = pd.concat(all_filtered_dfs, ignore_index=True).reset_index(drop=True)
    
    output_dir = os.path.join(base_path, 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Parquet으로 저장
    output_path_parquet = os.path.join(output_dir, f'filtered_cycles_{timestamp}.parquet')
    final_df.to_parquet(output_path_parquet)
    
    # JSON으로 저장
    output_path_json = os.path.join(output_dir, f'filtered_cycles_{timestamp}.json')
    # orient='records'는 각 행을 JSON 객체로 변환하여 리스트에 담습니다.
    # force_ascii=False는 한글이 깨지지 않도록 합니다.
    final_df.to_json(output_path_json, orient='records', indent=4, force_ascii=False)

    print(f"\n🎉 필터링 완료! 총 {len(final_df)}개의 사이클을 다음 파일들에 저장했습니다:")
    print(f"  - Parquet: '{output_path_parquet}'")
    print(f"  - JSON:    '{output_path_json}'")
    
    return final_df

if __name__ == '__main__':
    # --- 필터 조건 설정 ---
    # 예시: 주봉(1w)에서 10% 이상 상승했고, 그에 속하는 일봉(1d) 사이클은 5일 이상 지속된 경우
    filter1= {
        '1d': [
            {'feature': 'cycle_type', 'op': '==', 'value': 'up'},
        ],
        '4h': [
            {'feature': 'cycle_type', 'op': '==', 'value': 'up'}
        ]
    }

    filter2= {
        '1d': [
            {'feature': 'cycle_type', 'op': '==', 'value': 'up'},
        ],
        '4h': [
            {'feature': 'cycle_type', 'op': '==', 'value': 'down'}
        ]
    }

    # --- 필터링 함수 실행 ---
    project_root_path = '.'
    
    print("지정된 조건으로 필터링을 실행합니다.")
    filtered_data = filter_cycles_hierarchically(filter1, base_path=project_root_path)
    
    if not filtered_data.empty:
        print("\n추출된 데이터 샘플 (첫 5개):")
        print(filtered_data.head())