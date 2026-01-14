# feature_analysis/config.py
"""
사이클 특징 분석 설정 파일
"""
import os
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 경로 설정
DATA_PATHS = {
    'structured_data': PROJECT_ROOT / 'data' / 'cycle_data' / 'structured',
    'output': PROJECT_ROOT / 'feature_analysis' / 'output'
}

# 사용 가능한 타임프레임
AVAILABLE_TIMEFRAMES = ['1m', '1h', '4h', '1d', '1w']

# 분석 카테고리 정의 (한국어/영어 버전)
ANALYSIS_CATEGORIES = {
    'successful_rising': {
        'name_ko': '성공적 상승',
        'name_en': 'Successful Rising',
        'description_ko': 'MACD 상승 예측 + 가격 상승 실현',
        'description_en': 'MACD Rising Prediction + Price Up',
        'condition': lambda cycle_type, price_change: cycle_type == 'up' and price_change > 0,
        'color': '#2E8B57',  # SeaGreen
        'marker': 'o'
    },
    'betrayal_rising': {
        'name_ko': '배신형 상승',
        'name_en': 'Betrayal Rising',
        'description_ko': 'MACD 상승 예측 + 가격 하락 실현',
        'description_en': 'MACD Rising Prediction + Price Down',
        'condition': lambda cycle_type, price_change: cycle_type == 'up' and price_change < 0,
        'color': '#FF6347',  # Tomato
        'marker': '^'
    },
    'reversal_falling': {
        'name_ko': '반전형 하락',
        'name_en': 'Reversal Falling',
        'description_ko': 'MACD 하락 예측 + 가격 상승 실현',
        'description_en': 'MACD Falling Prediction + Price Up',
        'condition': lambda cycle_type, price_change: cycle_type == 'down' and price_change > 0,
        'color': '#FFD700',  # Gold
        'marker': 's'
    },
    'successful_falling': {
        'name_ko': '성공적 하락',
        'name_en': 'Successful Falling',
        'description_ko': 'MACD 하락 예측 + 가격 하락 실현',
        'description_en': 'MACD Falling Prediction + Price Down',
        'condition': lambda cycle_type, price_change: cycle_type == 'down' and price_change < 0,
        'color': '#DC143C',  # Crimson
        'marker': 'v'
    }
}

# 시각화 설정
PLOT_CONFIG = {
    'figsize': (14, 8),  # 범례를 위해 가로 크기를 늘림
    'dpi': 300,
    'style': 'whitegrid',
    'alpha': 0.7,
    'markersize': 50
}

# 출력 파일 설정
OUTPUT_CONFIG = {
    'image_format': 'png',
    'save_dpi': 300,
    'bbox_inches': 'tight'
}