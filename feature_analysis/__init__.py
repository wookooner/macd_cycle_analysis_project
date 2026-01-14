# feature_analysis/__init__.py
"""
사이클 특징 분석 패키지
MACD 사이클 예측과 실제 가격 변화의 관계를 분석하고 시각화
"""

from .cycle_analyzer import CycleFeatureAnalyzer
from . import config

__version__ = "1.0.0"
__author__ = "MACD Cycle Analysis Project"

__all__ = [
    'CycleFeatureAnalyzer',
    'config'
]