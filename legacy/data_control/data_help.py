import pandas as pd
import pyarrow.parquet as pq
import sys
import pyarrow
import numpy as np
import json

def analyze_parquet_file(file_path):
    """
    Parquet 파일의 구조와 형식을 분석하여 상세히 출력합니다.
    (수정) 새로운 카테고리 기반 구조를 상세히 출력하도록 개선되었습니다.
    """
    try:
        print("="*60)
        print(f"🐍 분석 환경 정보")
        print(f"  - Pandas 버전: {pd.__version__}")
        print(f"  - PyArrow 버전: {pyarrow.__version__}")
        print(f"  - NumPy 버전: {np.__version__}")
        print("="*60)

        # 파일 메타데이터 읽기
        parquet_file = pq.ParquetFile(file_path)
        df = pd.read_parquet(file_path, engine='pyarrow')

        print(f"\n📁 파일 분석: {file_path}")
        print("=" * 60)
        
        # 1. Parquet 파일 스키마(구조) 출력
        print("\n📜 Parquet 파일 원본 데이터 구조 (Schema):")
        print(parquet_file.schema)
        
        # 2. Pandas DataFrame의 일반적인 데이터 타입
        print("\n📊 Pandas DataFrame 컬럼별 일반 데이터 타입:")
        print(df.dtypes)

        # 3. 상세 데이터 타입 분석 (내부 타입 확인)
        print("\n🔬 컬럼별 상세 데이터 타입 분석 (첫 행 기준):")
        print("-" * 50)
        for col in df.columns:
            dtype = df[col].dtype
            
            if dtype == 'object' and not df[col].empty:
                first_valid_element = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                
                if first_valid_element is not None:
                    element_type = type(first_valid_element)
                    print(f"  - '{col}': {dtype} (내부 데이터 타입: {element_type})")
                else:
                    print(f"  - '{col}': {dtype} (내부 데이터가 비어있음)")
            else:
                print(f"  - '{col}': {dtype}")

        # 4. DataFrame 기본 정보 요약
        print("\n📋 Pandas DataFrame 기본 정보 (요약):")
        df.info(verbose=False)
        
        print("\n📝 최상위 레벨 데이터 샘플 (첫 5개 행):")
        print(df.head())
        
        # 5. [수정] cycle_features 컬럼 상세 분석 (카테고리 구조 반영)
        if not df.empty and 'cycle_features' in df.columns:
            print("\n\n🔬 `cycle_features` 컬럼 상세 (첫 2개 사이클 예시)")
            print("-" * 50)
            for i in range(min(2, len(df))):
                features = df['cycle_features'].iloc[i]
                cycle_type = df['cycle_type'].iloc[i]
                print("="*10 + f" {i+1}번째 사이클 ({cycle_type}) " + "="*10)
                
                if isinstance(features, dict):
                    # json.dumps를 사용하여 보기 좋게 출력
                    print(json.dumps(features, indent=2, ensure_ascii=False))
                else:
                    print("  (cycle_features가 올바른 형식이 아닙니다)")
                print("\n")
        
        # 6. candle_data 컬럼 상세 분석 (기존과 동일)
        if not df.empty and 'candle_data' in df.columns:
            print("\n\n📈 `candle_data` 컬럼 상세 (첫 2개 사이클의 첫 5개 캔들 예시)")
            print("-" * 70)
            for i in range(min(2, len(df))):
                print("="*10 + f" {i+1}번째 cycle candle data " + "="*10)
                candle_data = df['candle_data'].iloc[i]
                if isinstance(candle_data, (np.ndarray, list)) and len(candle_data) > 0:
                    candle_df = pd.DataFrame(list(candle_data))
                    print(candle_df.head())
                else:
                    print("  (Candle data가 비어있거나 올바른 형식이 아닙니다)")
                print("\n")

    except FileNotFoundError:
        print(f"\n❌ 오류: 파일을 찾을 수 없습니다 -> {file_path}")
    except Exception as e:
        print(f"\n❌ 파일을 처리하는 중 오류가 발생했습니다: {e}")


def summarize_current_cycle_safe(file_path):
    """
    (수정) 새로운 카테고리 구조에 맞게 현재 사이클을 요약하는 함수.
    """
    try:
        df = pd.read_parquet(file_path)

        if df.empty:
            print("데이터 파일이 비어 있습니다.")
            return

        if 'cycle_features' not in df.columns:
            print(f"🚨 오류: '{file_path}' 파일에 'cycle_features' 컬럼이 없습니다.")
            return

        current_cycle = df.iloc[-1]
        features = current_cycle.get('cycle_features', {})
        
        # [수정] 각 카테고리별로 데이터를 안전하게 가져오기
        shape = features.get('shape', {})
        strength = features.get('strength', {})
        start = features.get('start', {})
        end = features.get('end', {})
        change = features.get('change', {})
        volatility = features.get('volatility', {})
        aggregate = features.get('aggregate', {})

        print("\n" + "="*50)
        print("🚀 현재 진행중인 사이클 요약 정보 (최신 구조)")
        print("="*50)

        # 1. 기본 사이클 정보
        print("\n[1. 사이클 기본 정보]")
        print(f"  - 사이클 ID: {current_cycle.get('cycle_id', 'N/A')}")
        print(f"  - 타입: {current_cycle.get('cycle_type', 'N/A')}")
        print(f"  - 타임프레임: {current_cycle.get('timeframe', 'N/A')}")
        print(f"  - 시작일: {current_cycle.get('start_date', 'N/A')}")
        print(f"  - 종료일: {current_cycle.get('end_date', 'N/A')}")
        print(f"  - 지속 기간 (캔들 수): {shape.get('duration_candles', 'N/A')}개")

        # 2. 현재 값 (사이클 종료 시점)
        print("\n[2. 현재 값 (End)]")
        print(f"  - 현재 가격: ${end.get('price', 0):,.2f}")
        print(f"  - 현재 MACD: {end.get('macd', 0):.2f}")
        print(f"  - 현재 MACD Hist: {end.get('hist', 0):.2f}")
        print(f"  - 현재 RSI: {end.get('rsi', 0):.2f}")
        print(f"  - 현재 거래량: {end.get('volume', 0):,.2f}")

        # 3. 사이클 요약 특징
        print("\n[3. 사이클 변화 및 강도]")
        print(f"  - 가격 변화율: {change.get('price_pct', 0):.2f}%")
        print(f"  - RSI 변화량: {change.get('rsi', 0):.2f}")
        print(f"  - MACD Hist 변화량: {change.get('hist', 0):.2f}")
        print(f"  - 핵심/노이즈 캔들: {shape.get('core_count', 'N/A')} / {shape.get('noise_count', 'N/A')} 개")
        print(f"  - 추세 견고성 (Direction Ratio): {strength.get('direction_ratio', 0):.2f}%")
        
        # 4. 변동성
        print("\n[4. 사이클 변동성]")
        print(f"  - 시작가 대비 최고 상승률: +{volatility.get('max_high_pct', 0):.2f}%")
        print(f"  - 시작가 대비 최고 하락률: {volatility.get('max_loss_pct', 0):.2f}%")
        print(f"  - 평균 변동폭 (ATR): {volatility.get('atr_avg', 0):.2f}")


    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
    except Exception as e:
        print(f"데이터 처리 중 다른 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    
    # 분석할 파일 경로 설정
    file_to_analyze = "data/cycle_data/structured/cycles_4h.parquet"

    # --- [수정] 실행할 기능 선택 메뉴 ---
    print("="*60)
    print(f"분석 대상 파일: {file_to_analyze}")
    print("="*60)
    print("1: 현재 사이클 요약 정보 보기 (기본)")
    print("2: 파일 구조 상세 분석 보기")
    
    choice = input("실행할 기능 번호를 입력하세요 (1-2): ").strip()

    if choice == '2':
        analyze_parquet_file(file_to_analyze)
    else:
        summarize_current_cycle_safe(file_to_analyze)