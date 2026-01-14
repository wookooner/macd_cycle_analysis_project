# 필요한 라이브러리를 설치합니다. (최초 1회만 실행)
# pip install pandas pyarrow

import pandas as pd

def summarize_current_cycle_safe(file_path):
    """
    'cycle_features' 컬럼 존재 여부를 먼저 확인하여 Key-Error를 방지하는
    안전한 버전의 사이클 요약 함수.
    """
    try:
        # Parquet 파일 읽기
        df = pd.read_parquet(file_path)

        if df.empty:
            print("데이터 파일이 비어 있습니다.")
            return

        # ★★★ 오류 방지 핵심 로직 ★★★
        # 'cycle_features' 컬럼이 있는지 먼저 확인합니다.
        if 'cycle_features' not in df.columns:
            print(f"🚨 오류: '{file_path}' 파일에 'cycle_features' 컬럼이 없습니다.")
            print("\n파일에 존재하는 실제 컬럼 목록입니다:")
            print(f"  - {list(df.columns)}")
            print("\n👉 해결 방법: data/cycle_data/structured/ 경로의 최종 Parquet 파일을 사용하고 있는지 확인해주세요.")
            return

        # 가장 마지막 행 (가장 최신 사이클) 선택
        current_cycle = df.iloc[-1]
        cycle_features = current_cycle['cycle_features']

        # --- 요약 정보 출력 ---
        print("="*50)
        print("🚀 현재 진행중인 사이클 요약 정보")
        print("="*50)

        # 1. 기본 사이클 정보
        print("\n[1. 사이클 기본 정보]")
        print(f"  - 사이클 ID: {current_cycle.get('cycle_id', 'N/A')}")
        print(f"  - 타입: {current_cycle.get('cycle_type', 'N/A')}")
        print(f"  - 타임프레임: {current_cycle.get('timeframe', 'N/A')}")
        print(f"  - 시작일: {current_cycle.get('start_date', 'N/A')}")
        print(f"  - 종료일: {current_cycle.get('end_date', 'N/A')}")
        print(f"  - 지속 기간 (캔들 수): {current_cycle.get('duration_candles', 'N/A')}개")

        # 2. 현재 값 (사이클 종료 시점)
        print("\n[2. 현재 값 (사이클 종료 시점)]")
        print(f"  - 현재 가격 (End Price): ${cycle_features.get('end_price', 0):,.2f}")
        print(f"  - 현재 MACD (End MACD): {cycle_features.get('end_macd', 0):.2f}")
        print(f"  - 현재 MACD Hist (End Hist): {cycle_features.get('end_hist', 0):.2f}")
        print(f"  - 현재 RSI (End RSI): {cycle_features.get('end_rsi', 0):.2f}")
        print(f"  - 현재 거래량 (End Volume): {cycle_features.get('end_volume', 0):,.2f}")

        # 3. 사이클 요약 특징
        print("\n[3. 사이클 요약 특징]")
        print(f"  - 가격 변화율: {cycle_features.get('price_change_pct', 0):.2f}%")
        print(f"  - RSI 변화량: {cycle_features.get('rsi_change', 0):.2f}")
        print(f"  - MACD 히스토그램 변화량: {cycle_features.get('macd_histogram_change', 0):.2f}")
        print(f"  - 노이즈 캔들 수: {cycle_features.get('noise_count', 'N/A')}개")
        print(f"  - 핵심 추세 구간 수: {cycle_features.get('core_count', 'N/A')}개")
        print(f"  - 추세 방향 변경 횟수: {cycle_features.get('direction_change', 'N/A')}회")


    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
    except Exception as e:
        print(f"데이터 처리 중 다른 오류가 발생했습니다: {e}")

# --- 실행 부분 ---
# ❗️ 분석할 파일 경로를 다시 한번 확인해주세요.
# 아래 경로는 특징 추출이 완료된 최종 파일이 위치한 곳입니다.
file_to_analyze = 'data/cycle_data/structured/cycles_4h.parquet'

# 개선된 함수 실행
summarize_current_cycle_safe(file_to_analyze)