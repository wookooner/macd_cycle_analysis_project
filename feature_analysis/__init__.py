"""
사이클 특징 분석 패키지 (Cycle Feature Analysis)
================================================
MACD 히스토그램 기반 사이클 데이터의 특징 분석, 통계 검정, 시각화를 위한 통합 패키지.

구조:
    feature_analysis/
    ├── core/
    │   ├── data_loader.py   → CycleDataLoader   (통합 데이터 로드/전처리)
    │   └── stat_engine.py   → StatEngine        (통계 검정 + 효과크기)
    ├── analyzers/
    │   ├── feature_profiler.py     → FeatureProfiler       (단변량 프로파일링)
    │   ├── conditional_analyzer.py → ConditionalAnalyzer    (핵심: 조건부 확률)
    │   └── sequence_analyzer.py    → SequenceAnalyzer       (전이 확률, 연속 패턴)
    ├── viz/
    │   └── cycle_plots.py   → CyclePlots        (전문 시각화)
    └── run_analysis.py      → AnalysisRunner     (통합 실행기)

빠른 시작:
    from feature_analysis.run_analysis import AnalysisRunner
    runner = AnalysisRunner(timeframe="4h")
    runner.run_full_report()
"""

__all__ = []
__version__ = "3.0.0"