"""
사이클 분석 통합 실행기 (Run Analysis)
======================================
모든 분석 모듈을 하나의 인터페이스로 연결.
CLI 메뉴 방식과 직접 호출 방식 모두 지원.

CLI 실행:
    python -m feature_analysis.run_analysis
    
직접 호출 (스크립트/노트북):
    from feature_analysis.run_analysis import AnalysisRunner
    
    runner = AnalysisRunner(timeframe="4h")
    
    # 피처 랭킹 (효과크기 순)
    runner.run_feature_ranking()
    
    # 조건부 확률 분석
    runner.run_conditional("start_hist", n_bins=6)
    
    # 전이 확률 행렬
    runner.run_transition_matrix()
"""

import sys
import traceback
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np

# When the module is executed as a script (e.g. `python feature_analysis/run_analysis.py`),
# Python sets __package__ to either None or an empty string. In that case the
# current directory (which is `feature_analysis/`) is added to sys.path, causing
# absolute imports like `feature_analysis.core...` to resolve incorrectly.
#
# To make the script runnable both as a module (`-m feature_analysis.run_analysis`)
# and directly, add the repository root to sys.path when there is no package
# context (None or empty string).
if __name__ == "__main__" and not __package__:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))


class AnalysisRunner:
    """
    분석 통합 실행기.
    
    모든 분석 모듈을 로딩하고 일관된 인터페이스로 실행.
    결과는 구조화된 딕셔너리 + 시각화 파일로 제공.
    """
    
    def __init__(self, timeframe: str = "4h",
                 data_dir: str = "data/cycle_data/structured",
                 config_path: str = "feature_extract/macd_historgram_change_feature/features_config_v2.json",
                 output_dir: str = "feature_analysis/output",
                 enriched: bool = True,
                 only_enabled: bool = False):
        """
        Args:
            timeframe: 분석 대상 타임프레임
            data_dir: 사이클 데이터 디렉토리
            config_path: features_config_v2.json 경로
            output_dir: 시각화 출력 디렉토리
            enriched: enriched parquet 사용 여부
            only_enabled: enabled 피처만 로드
        """
        # 지연 임포트 (순환 의존성 방지)
        from feature_analysis.core.data_loader import CycleDataLoader
        from feature_analysis.analyzers.feature_profiler import FeatureProfiler
        from feature_analysis.analyzers.conditional_analyzer import ConditionalAnalyzer
        from feature_analysis.analyzers.sequence_analyzer import SequenceAnalyzer
        from feature_analysis.viz.cycle_plots import CyclePlots
        
        self.timeframe = timeframe
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 로딩
        print(f"\n{'='*60}")
        print(f"  사이클 분석 시스템 초기화 (Timeframe: {timeframe})")
        print(f"{'='*60}")
        
        self.loader = CycleDataLoader(data_dir=data_dir, config_path=config_path)
        self.df = self.loader.load(timeframe, enriched=enriched, only_enabled=only_enabled)
        
        print(f"  데이터 로딩 완료: {len(self.df)} 사이클")
        
        if 'composite_category' in self.df.columns:
            cat_counts = self.df['composite_category'].value_counts()
            for cat, cnt in cat_counts.items():
                print(f"    {cat}: {cnt}개")
        
        # 분석 모듈 초기화
        self.profiler = FeatureProfiler(self.df, self.loader)
        self.conditional = ConditionalAnalyzer(self.df)
        self.sequence = SequenceAnalyzer(self.df)
        self.plots = CyclePlots(save_dir=str(self.output_dir))
        
        # 사용 가능한 피처 목록 캐시
        self.numeric_features = self.loader.get_numeric_feature_columns(self.df)
        self.enabled_features = self.loader.get_enabled_features()
        self.predictive_features = self.loader.get_predictive_features()
        
        print(f"  수치형 피처: {len(self.numeric_features)}개")
        print(f"  Enabled 피처: {len(self.enabled_features)}개")
        print(f"  예측 가능 피처 (start): {len(self.predictive_features)}개")
        print(f"  출력 디렉토리: {self.output_dir}")
        print(f"{'='*60}\n")
    
    # =========================================================
    # 분석 실행 메서드들 (직접 호출용)
    # =========================================================
    
    def run_feature_profile(self, feature: str) -> dict:
        """단일 피처 종합 프로파일 실행 + 시각화 저장."""
        print(f"\n[Feature Profile] {feature}")
        result = self.profiler.profile_feature(feature)
        
        if 'error' not in result:
            path = self.plots.plot_feature_profile(result)
            print(f"  시각화 저장: {path}")
            self._print_profile_summary(result)
        else:
            print(f"  Error: {result['error']}")
        
        return result
    
    def run_feature_ranking(self, features: Optional[List[str]] = None, top_n: int = 20) -> pd.DataFrame:
        """전체 피처 효과크기 랭킹 실행 + 시각화."""
        print(f"\n[Feature Ranking] 효과크기(Cliff's delta) 기준 피처 순위")
        ranking = self.profiler.rank_features_by_effect_size(features)
        
        if not ranking.empty:
            path = self.plots.plot_feature_ranking(ranking, top_n=top_n)
            print(f"  시각화 저장: {path}")
            self._print_ranking_summary(ranking, top_n)
        
        return ranking
    
    def run_conditional(self, feature: str, n_bins: int = 6,
                       bins: Optional[list] = None,
                       cycle_type: Optional[str] = None) -> dict:
        """단일 조건 분석 실행 + 시각화."""
        ct_str = f" (cycle: {cycle_type})" if cycle_type else ""
        print(f"\n[Conditional Analysis] {feature}{ct_str}")
        
        result = self.conditional.single_condition(
            feature, n_bins=n_bins, bins=bins, cycle_type=cycle_type
        )
        
        if 'error' not in result:
            suffix = f"_{cycle_type}" if cycle_type else ""
            path = self.plots.plot_conditional_table(
                result, save_name=f"conditional_{feature}{suffix}"
            )
            print(f"  시각화 저장: {path}")
            self._print_conditional_summary(result)
        else:
            print(f"  Error: {result['error']}")
        
        return result
    
    def run_combined_conditions(self, conditions: list,
                                cycle_type: Optional[str] = None) -> dict:
        """복합 조건 분석 실행."""
        print(f"\n[Combined Conditions] {len(conditions)}개 조건 조합")
        result = self.conditional.combined_conditions(conditions, cycle_type=cycle_type)
        
        if 'error' not in result:
            self._print_combined_summary(result)
        else:
            print(f"  Error: {result['error']}")
        
        return result
    
    def run_pattern_discovery(self, features: Optional[List[str]] = None,
                              max_conditions: int = 2,
                              cycle_type: Optional[str] = None,
                              top_k: int = 20) -> pd.DataFrame:
        """자동 패턴 탐색 실행 + 시각화."""
        print(f"\n[Pattern Discovery] 자동 패턴 탐색 (max_conditions={max_conditions})")
        
        patterns = self.conditional.discover_patterns(
            features=features, max_conditions=max_conditions,
            cycle_type=cycle_type, top_k=top_k
        )
        
        if not patterns.empty:
            suffix = f"_{cycle_type}" if cycle_type else ""
            path = self.plots.plot_discovered_patterns(
                patterns, save_name=f"patterns{suffix}"
            )
            print(f"  시각화 저장: {path}")
            print(f"\n  Top {min(10, len(patterns))} 패턴:")
            for i, row in patterns.head(10).iterrows():
                reliable = "✓" if row['n'] >= 30 else "⚠"
                print(f"    #{i} {reliable} 승률={row['win_rate']:.1f}% | "
                      f"수익={row['avg_return']:.4f}% | n={row['n']} | "
                      f"{row['conditions']}")
        
        return patterns
    
    def run_transition_matrix(self) -> dict:
        """전이 확률 행렬 분석 + 시각화."""
        print(f"\n[Transition Matrix] 사이클 타입 전이 분석")
        result = self.sequence.transition_matrix()
        
        path = self.plots.plot_transition_matrix(result)
        print(f"  시각화 저장: {path}")
        
        # 상세 출력
        details = result.get('details', {})
        for key, info in details.items():
            print(f"  {key}: 확률={info['probability']:.1f}% | "
                  f"승률={info['win_rate']:.1f}% | 평균수익={info['avg_return']:.4f}% | "
                  f"n={info['n']}")
        
        return result
    
    def run_streak_analysis(self, max_streak: int = 5) -> dict:
        """연속 패턴 분석 + 시각화."""
        print(f"\n[Streak Analysis] 연속 패턴 분석 (최대 {max_streak}연속)")
        result = self.sequence.streak_analysis(max_streak=max_streak)
        
        path = self.plots.plot_streak_analysis(result)
        print(f"  시각화 저장: {path}")
        
        streaks = result.get('streaks', pd.DataFrame())
        if not streaks.empty:
            for _, row in streaks.iterrows():
                print(f"  {row['streak_type']} {row['streak_length']}연속 후: "
                      f"승률={row['next_win_rate']:.1f}% | "
                      f"평균수익={row['next_avg_return']:.4f}% | n={row['n_occurrences']}")
        
        return result
    
    def run_threshold_search(self, feature: str,
                            cycle_type: Optional[str] = None) -> dict:
        """최적 임계값 탐색 + 시각화."""
        print(f"\n[Threshold Search] {feature}")
        result = self.conditional.find_optimal_threshold(feature, cycle_type=cycle_type)
        
        if 'error' not in result:
            path = self.plots.plot_threshold_search(result)
            print(f"  시각화 저장: {path}")
            
            if result.get('optimal_below'):
                opt = result['optimal_below']
                print(f"  최적 (below): {feature} < {opt['threshold']:.2f} → "
                      f"승률={opt['win_rate_below']:.1f}% (n={opt['n_below']})")
            if result.get('optimal_above'):
                opt = result['optimal_above']
                print(f"  최적 (above): {feature} >= {opt['threshold']:.2f} → "
                      f"승률={opt['win_rate_above']:.1f}% (n={opt['n_above']})")
        else:
            print(f"  Error: {result['error']}")
        
        return result
    
    def run_full_report(self):
        """
        전체 분석 리포트 일괄 실행.
        모든 핵심 분석을 한 번에 수행하고 시각화를 저장.
        """
        print("\n" + "="*70)
        print("  FULL ANALYSIS REPORT")
        print("="*70)
        
        # 1) 피처 랭킹
        self.run_feature_ranking()
        
        # 2) 예측 가능 피처별 조건부 확률 (start 카테고리)
        for feat in self.predictive_features:
            if feat in self.df.columns and pd.api.types.is_numeric_dtype(self.df[feat]):
                try:
                    self.run_conditional(feat)
                except Exception as e:
                    print(f"  {feat} 분석 실패: {e}")
        
        # 3) 전이 확률
        self.run_transition_matrix()
        
        # 4) 연속 패턴
        self.run_streak_analysis()
        
        # 5) 자동 패턴 탐색 (start 피처 중심)
        pred_in_df = [f for f in self.predictive_features 
                     if f in self.df.columns and pd.api.types.is_numeric_dtype(self.df[f])]
        if pred_in_df:
            self.run_pattern_discovery(features=pred_in_df)
        
        print(f"\n{'='*70}")
        print(f"  리포트 완료. 출력 디렉토리: {self.output_dir}")
        print(f"{'='*70}")
    
    # =========================================================
    # 출력 헬퍼
    # =========================================================
    
    def _print_profile_summary(self, result: dict):
        """프로파일 요약 출력."""
        overall = result.get('overall', {})
        desc = overall.get('descriptive', {})
        
        print(f"  분포: n={desc.get('n')}, mean={desc.get('mean', 0):.4f}, "
              f"median={desc.get('median', 0):.4f}, std={desc.get('std', 0):.4f}")
        
        up_down = result.get('up_vs_down', {})
        if 'effect_sizes' in up_down:
            cd = up_down['effect_sizes']['cliffs_delta']
            print(f"  UP vs DOWN: {cd['description']}")
            print(f"  해석: {up_down.get('interpretation', '')}")
    
    def _print_ranking_summary(self, ranking: pd.DataFrame, top_n: int):
        """랭킹 요약 출력."""
        significant = ranking[ranking['significant']]
        print(f"  통계적 유의: {len(significant)}/{len(ranking)}개")
        
        meaningful = ranking[ranking['magnitude'].isin(['medium', 'large'])]
        print(f"  실질적 의미 (medium+): {len(meaningful)}개")
        
        print(f"\n  Top {min(top_n, len(ranking))} 피처:")
        for _, row in ranking.head(top_n).iterrows():
            sig = "★" if row['significant'] else " "
            print(f"    {sig} {row['feature']:35s} | Cliff's δ={row['cliffs_delta']:+.4f} "
                  f"({row['magnitude']:10s}) | p={row['p_value']:.4e}")
    
    def _print_conditional_summary(self, result: dict):
        """조건부 분석 요약 출력."""
        overall = result.get('overall', {})
        print(f"  전체: 승률={overall.get('win_rate', 0):.1f}%, "
              f"평균수익={overall.get('avg_return', 0):.4f}% (n={overall.get('n', 0)})")
        
        best = result.get('best_range')
        if best:
            print(f"  최고구간: {best.get('range', '?')} → 승률={best.get('win_rate', 0):.1f}% "
                  f"(n={best.get('n', 0)}, {'✓ 신뢰' if best.get('reliable') else '⚠ 소표본'})")
        
        worst = result.get('worst_range')
        if worst:
            print(f"  최저구간: {worst.get('range', '?')} → 승률={worst.get('win_rate', 0):.1f}% "
                  f"(n={worst.get('n', 0)})")
    
    def _print_combined_summary(self, result: dict):
        """복합 조건 요약 출력."""
        print(f"  조건: {result.get('conditions', '?')}")
        
        matched = result.get('matched', {})
        not_matched = result.get('not_matched', {})
        
        print(f"  조건 충족: 승률={matched.get('win_rate', 0):.1f}%, "
              f"평균수익={matched.get('avg_return', 0):.4f}% (n={matched.get('n', 0)})")
        print(f"  조건 미충족: 승률={not_matched.get('win_rate', 0):.1f}%, "
              f"평균수익={not_matched.get('avg_return', 0):.4f}% (n={not_matched.get('n', 0)})")
        
        reliable = result.get('reliable', False)
        if not reliable:
            print(f"  ⚠ 샘플 수 부족으로 신뢰도 낮음")


# =========================================================
# CLI 메뉴 모드
# =========================================================

def interactive_menu():
    """인터랙티브 CLI 메뉴."""
    
    print("\n" + "="*60)
    print("  MACD Cycle Feature Analysis System")
    print("="*60)
    
    # 타임프레임 선택
    print("\n사용 가능한 타임프레임:")
    from feature_analysis.core.data_loader import CycleDataLoader
    loader = CycleDataLoader()
    tfs = loader.available_timeframes()
    
    for i, tf in enumerate(tfs, 1):
        print(f"  {i}. {tf}")
    
    try:
        choice = int(input(f"\n타임프레임 선택 (1-{len(tfs)}): ")) - 1
        timeframe = tfs[choice]
    except (ValueError, IndexError):
        timeframe = "4h"
        print(f"  기본값 사용: {timeframe}")
    
    runner = AnalysisRunner(timeframe=timeframe)
    
    while True:
        print(f"\n{'='*60}")
        print(f"  분석 메뉴 (Timeframe: {runner.timeframe})")
        print(f"{'='*60}")
        print("  1. 피처 효과크기 랭킹 (UP vs DOWN)")
        print("  2. 단일 피처 상세 프로파일")
        print("  3. 조건부 확률 분석 (구간별 승률)")
        print("  4. 복합 조건 분석")
        print("  5. 자동 패턴 탐색")
        print("  6. 전이 확률 행렬")
        print("  7. 연속 패턴 분석")
        print("  8. 임계값 탐색")
        print("  9. 전체 리포트 (일괄 실행)")
        print("  0. 종료")
        
        try:
            choice = input("\n선택: ").strip()
            
            if choice == '0':
                print("\n분석 시스템을 종료합니다.")
                break
            elif choice == '1':
                runner.run_feature_ranking()
            elif choice == '2':
                feat = _select_feature(runner)
                if feat:
                    runner.run_feature_profile(feat)
            elif choice == '3':
                feat = _select_feature(runner)
                if feat:
                    ct = _select_cycle_type()
                    runner.run_conditional(feat, cycle_type=ct)
            elif choice == '4':
                conditions = _build_conditions(runner)
                if conditions:
                    ct = _select_cycle_type()
                    runner.run_combined_conditions(conditions, cycle_type=ct)
            elif choice == '5':
                ct = _select_cycle_type()
                runner.run_pattern_discovery(cycle_type=ct)
            elif choice == '6':
                runner.run_transition_matrix()
            elif choice == '7':
                runner.run_streak_analysis()
            elif choice == '8':
                feat = _select_feature(runner)
                if feat:
                    ct = _select_cycle_type()
                    runner.run_threshold_search(feat, cycle_type=ct)
            elif choice == '9':
                runner.run_full_report()
            else:
                print("  잘못된 선택입니다.")
                
        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            break
        except Exception as e:
            print(f"\n  오류 발생: {e}")
            traceback.print_exc()


def _select_feature(runner: AnalysisRunner) -> Optional[str]:
    """피처 선택 헬퍼."""
    features = runner.numeric_features
    print(f"\n사용 가능한 피처 ({len(features)}개):")
    for i, f in enumerate(features, 1):
        enabled_mark = "●" if f in runner.enabled_features else "○"
        pred_mark = "→" if f in runner.predictive_features else " "
        print(f"  {i:3d}. {enabled_mark}{pred_mark} {f}")
    
    try:
        idx = int(input(f"\n피처 번호 (1-{len(features)}): ")) - 1
        return features[idx]
    except (ValueError, IndexError):
        print("  잘못된 선택")
        return None


def _select_cycle_type() -> Optional[str]:
    """사이클 타입 필터 선택."""
    print("\n사이클 타입 필터:")
    print("  1. 전체 (필터 없음)")
    print("  2. UP 사이클만")
    print("  3. DOWN 사이클만")
    
    try:
        choice = input("선택 (기본: 1): ").strip()
        if choice == '2':
            return 'up'
        elif choice == '3':
            return 'down'
        return None
    except Exception:
        return None


def _build_conditions(runner: AnalysisRunner) -> Optional[list]:
    """복합 조건 구성 헬퍼."""
    conditions = []
    print("\n조건 입력 (완료하려면 빈 줄 입력)")
    print("형식: 피처명 연산자 값  (예: start_hist < -50)")
    
    while len(conditions) < 3:
        try:
            line = input(f"  조건 {len(conditions)+1}: ").strip()
            if not line:
                break
            
            parts = line.split()
            if len(parts) != 3:
                print("    형식 오류. 예: start_hist < -50")
                continue
            
            feat, op, val = parts
            if feat not in runner.numeric_features:
                print(f"    '{feat}'은 유효한 피처가 아닙니다.")
                continue
            
            conditions.append((feat, op, float(val)))
            print(f"    ✓ 추가: {feat} {op} {val}")
            
        except (ValueError, KeyboardInterrupt):
            break
    
    return conditions if conditions else None


# =========================================================
# 엔트리 포인트
# =========================================================

if __name__ == "__main__":
    interactive_menu()