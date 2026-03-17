"""
조건부 확률 분석기 (Conditional Probability Analyzer)
=====================================================
"어떤 조건에서 → 어떤 확률로 → 어떤 방향(가격 상승/하락)으로 갈지"를 체계적으로 분석.

이 모듈이 분석하는 것:
1. 단일 조건 분석: start_hist 구간별 승률/수익률
2. 복합 조건 분석: start_hist < -50 AND start_rsi < 30일 때 승률
3. 자동 구간 탐색: 최적 분할점 자동 탐색
4. 신뢰도 관리: 샘플 수 기반 신뢰구간 계산

사용법:
    from feature_analysis.core.data_loader import CycleDataLoader
    from feature_analysis.analyzers.conditional_analyzer import ConditionalAnalyzer
    
    loader = CycleDataLoader()
    df = loader.load("4h")
    analyzer = ConditionalAnalyzer(df)
    
    # 단일 피처 구간별 분석
    result = analyzer.single_condition("start_hist", n_bins=6)
    
    # 사용자 정의 구간
    result = analyzer.single_condition("start_hist", 
                bins=[-np.inf, -100, -50, 0, 50, 100, np.inf])
    
    # 2-피처 복합 조건
    result = analyzer.combined_conditions(
        conditions=[("start_hist", "<", -50), ("start_rsi", "<", 30)]
    )
    
    # 자동 최적 조건 탐색
    top_patterns = analyzer.discover_patterns(max_conditions=2, min_samples=30)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats as sp_stats
import itertools


class ConditionalAnalyzer:
    """조건부 확률 분석기."""
    
    # 최소 샘플 수 기본값 (이 이하면 신뢰 불가)
    DEFAULT_MIN_SAMPLES = 30
    
    def __init__(self, df: pd.DataFrame, 
                 target_col: str = "change_price_pct",
                 type_col: str = "cycle_type"):
        """
        Args:
            df: 평탄화된 사이클 DataFrame
            target_col: 타깃 변수 (가격변화율)
            type_col: 사이클 타입 컬럼
        """
        self.df = df
        self.target_col = target_col
        self.type_col = type_col
    
    # =========================================================
    # 1. 단일 조건 분석 (구간별 승률/수익률 테이블)
    # =========================================================
    
    def single_condition(self, feature: str,
                         n_bins: int = 6,
                         bins: Optional[List[float]] = None,
                         cycle_type: Optional[str] = None,
                         min_samples: int = DEFAULT_MIN_SAMPLES) -> Dict[str, Any]:
        """
        하나의 피처를 구간으로 나눠 각 구간의 승률/수익률을 계산.
        
        Args:
            feature: 분석할 피처명 (예: "start_hist")
            n_bins: 자동 분할 시 구간 수 (quantile 기반)
            bins: 사용자 정의 구간 경계 (지정하면 n_bins 무시)
            cycle_type: None이면 전체, "up"/"down"이면 해당 타입만
            min_samples: 최소 샘플 수 (이 이하 구간은 경고 표시)
            
        Returns:
            {
                'feature': str,
                'cycle_type_filter': str or None,
                'table': DataFrame (구간별 통계),
                'overall': dict (전체 통계),
                'best_range': dict (최고 승률 구간),
                'worst_range': dict (최저 승률 구간)
            }
        """
        # 필터링
        data = self._prepare_data(feature, cycle_type)
        if data is None:
            return {'error': f"피처 '{feature}' 또는 타깃 '{self.target_col}' 없음"}
        
        if len(data) < min_samples:
            return {'error': f'데이터 부족 (n={len(data)}, 최소={min_samples})'}
        
        # 구간 생성
        if bins is not None:
            data['bin'] = pd.cut(data[feature], bins=bins, include_lowest=True)
        else:
            # quantile 기반 균등 분할
            data['bin'] = pd.qcut(data[feature], q=n_bins, duplicates='drop')
        
        # 구간별 통계 계산
        table = self._calculate_bin_stats(data, feature, min_samples)
        
        # 전체 통계
        overall = self._calculate_overall_stats(data)
        
        # 최고/최저 구간 (충분한 샘플이 있는 것 중에서)
        reliable = table[table['n'] >= min_samples]
        best_range = reliable.loc[reliable['win_rate'].idxmax()].to_dict() if len(reliable) > 0 else None
        worst_range = reliable.loc[reliable['win_rate'].idxmin()].to_dict() if len(reliable) > 0 else None
        
        return {
            'feature': feature,
            'cycle_type_filter': cycle_type,
            'table': table,
            'overall': overall,
            'best_range': best_range,
            'worst_range': worst_range,
            'total_samples': len(data)
        }
    
    # =========================================================
    # 2. 복합 조건 분석
    # =========================================================
    
    def combined_conditions(self, 
                           conditions: List[Tuple[str, str, float]],
                           cycle_type: Optional[str] = None,
                           min_samples: int = DEFAULT_MIN_SAMPLES) -> Dict[str, Any]:
        """
        복합 조건(AND)에서의 승률/수익률 계산.
        
        Args:
            conditions: [(feature, operator, value), ...] 리스트
                       operator: '<', '<=', '>', '>=', '==', 'between'
                       between인 경우 value는 (low, high) 튜플
            cycle_type: None이면 전체, "up"/"down"이면 해당 타입만
            
        사용 예:
            conditions=[("start_hist", "<", -50), ("start_rsi", "<", 30)]
            conditions=[("start_hist", "between", (-100, -50)), ("duration_candles", ">", 5)]
        """
        # 기본 데이터 준비
        data = self.df.copy()
        if cycle_type:
            data = data[data[self.type_col].str.upper() == cycle_type.upper()]
        
        # 필수 컬럼 확인
        required = [self.target_col] + [c[0] for c in conditions]
        missing = [c for c in required if c not in data.columns]
        if missing:
            return {'error': f"Missing columns: {missing}"}
        
        # 조건 적용
        mask = pd.Series(True, index=data.index)
        condition_descriptions = []
        
        for feature, op, value in conditions:
            col = data[feature]
            if op == '<':
                mask &= col < value
                condition_descriptions.append(f"{feature} < {value}")
            elif op == '<=':
                mask &= col <= value
                condition_descriptions.append(f"{feature} <= {value}")
            elif op == '>':
                mask &= col > value
                condition_descriptions.append(f"{feature} > {value}")
            elif op == '>=':
                mask &= col >= value
                condition_descriptions.append(f"{feature} >= {value}")
            elif op == '==':
                mask &= col == value
                condition_descriptions.append(f"{feature} == {value}")
            elif op == 'between':
                low, high = value
                mask &= (col >= low) & (col <= high)
                condition_descriptions.append(f"{low} <= {feature} <= {high}")
        
        filtered = data[mask]
        complement = data[~mask]
        
        # 통계 계산
        condition_str = " AND ".join(condition_descriptions)
        
        result = {
            'conditions': condition_str,
            'condition_list': conditions,
            'cycle_type_filter': cycle_type,
            'matched': self._calculate_group_stats(filtered, label="조건 충족"),
            'not_matched': self._calculate_group_stats(complement, label="조건 미충족"),
            'total_samples': len(data),
            'reliable': len(filtered) >= min_samples
        }
        
        # 조건 충족 vs 미충족 차이 검정
        if len(filtered) >= 5 and len(complement) >= 5:
            target_matched = filtered[self.target_col].dropna().values
            target_not = complement[self.target_col].dropna().values
            
            from feature_analysis.core.stat_engine import StatEngine
            result['comparison'] = StatEngine.compare_groups(
                target_matched, target_not,
                label_a="조건충족", label_b="조건미충족"
            )
        
        return result
    
    # =========================================================
    # 3. 자동 패턴 탐색 (Grid Search 방식)
    # =========================================================
    
    def discover_patterns(self, 
                          features: Optional[List[str]] = None,
                          max_conditions: int = 2,
                          n_splits: int = 3,
                          cycle_type: Optional[str] = None,
                          min_samples: int = DEFAULT_MIN_SAMPLES,
                          top_k: int = 20) -> pd.DataFrame:
        """
        피처 구간의 조합을 자동 탐색하여 높은 승률 패턴을 발견.
        
        Args:
            features: 탐색할 피처 리스트 (None이면 예측 가능한 피처)
            max_conditions: 최대 조건 조합 수 (1 또는 2)
            n_splits: 각 피처를 나눌 구간 수
            cycle_type: 사이클 타입 필터
            min_samples: 최소 샘플 수
            top_k: 상위 몇 개 패턴 반환
            
        Returns:
            DataFrame: conditions, win_rate, avg_return, n, confidence_lower, ...
        """
        data = self.df.copy()
        if cycle_type:
            data = data[data[self.type_col].str.upper() == cycle_type.upper()]
        
        if features is None:
            # 수치형 피처만
            features = [c for c in data.columns 
                       if pd.api.types.is_numeric_dtype(data[c])
                       and c != self.target_col
                       and c not in ('duration_candles',)]  # 메타 제외
        
        # 유효 피처만 (NaN 아닌 값이 충분한 것)
        valid_features = [f for f in features 
                         if data[f].notna().sum() >= min_samples * 2]
        
        # 피처별 분할점 생성
        split_points = {}
        for feat in valid_features:
            vals = data[feat].dropna()
            quantiles = np.linspace(0, 100, n_splits + 1)[1:-1]  # 내부 분위점
            points = np.percentile(vals, quantiles)
            split_points[feat] = sorted(set(points))  # 중복 제거
        
        patterns = []
        
        # === 1-조건 탐색 ===
        for feat in valid_features:
            for point in split_points.get(feat, []):
                # feature < point
                mask_below = data[feat] < point
                stats_below = self._quick_stats(data[mask_below], min_samples)
                if stats_below:
                    stats_below['conditions'] = f"{feat} < {point:.2f}"
                    stats_below['n_conditions'] = 1
                    patterns.append(stats_below)
                
                # feature >= point
                mask_above = data[feat] >= point
                stats_above = self._quick_stats(data[mask_above], min_samples)
                if stats_above:
                    stats_above['conditions'] = f"{feat} >= {point:.2f}"
                    stats_above['n_conditions'] = 1
                    patterns.append(stats_above)
        
        # === 2-조건 탐색 (max_conditions >= 2) ===
        if max_conditions >= 2 and len(valid_features) >= 2:
            # 피처 쌍 조합 (너무 많으면 상위 효과크기 피처로 제한)
            if len(valid_features) > 8:
                feat_subset = valid_features[:8]
            else:
                feat_subset = valid_features
            
            for feat1, feat2 in itertools.combinations(feat_subset, 2):
                for p1 in split_points.get(feat1, []):
                    for p2 in split_points.get(feat2, []):
                        # 4가지 조합: (<, <), (<, >=), (>=, <), (>=, >=)
                        for op1, mask_fn1 in [('<', data[feat1] < p1), ('>=', data[feat1] >= p1)]:
                            for op2, mask_fn2 in [('<', data[feat2] < p2), ('>=', data[feat2] >= p2)]:
                                mask = mask_fn1 & mask_fn2
                                s = self._quick_stats(data[mask], min_samples)
                                if s:
                                    s['conditions'] = f"{feat1} {op1} {p1:.2f} AND {feat2} {op2} {p2:.2f}"
                                    s['n_conditions'] = 2
                                    patterns.append(s)
        
        if not patterns:
            return pd.DataFrame()
        
        result = pd.DataFrame(patterns)
        
        # 승률 기준 정렬 (동률이면 샘플수 기준)
        result = result.sort_values(['win_rate', 'n'], ascending=[False, False])
        result = result.head(top_k).reset_index(drop=True)
        result.index = result.index + 1
        result.index.name = 'rank'
        
        return result
    
    # =========================================================
    # 4. 특정 피처의 최적 임계값 탐색
    # =========================================================
    
    def find_optimal_threshold(self, feature: str,
                               cycle_type: Optional[str] = None,
                               n_candidates: int = 20,
                               min_samples: int = DEFAULT_MIN_SAMPLES) -> Dict[str, Any]:
        """
        하나의 피처에 대해 승률을 최대화하는 임계값을 탐색.
        
        "start_hist가 얼마 이하일 때 UP 사이클의 가격 상승률이 가장 높은가?"
        
        Returns:
            {
                'feature': str,
                'threshold_table': DataFrame (각 임계값에서의 승률),
                'optimal_below': dict (feature < threshold 최적),
                'optimal_above': dict (feature >= threshold 최적)
            }
        """
        data = self._prepare_data(feature, cycle_type)
        if data is None or len(data) < min_samples * 2:
            return {'error': 'Insufficient data'}
        
        # 후보 임계값: 균등 분위
        percentiles = np.linspace(10, 90, n_candidates)
        thresholds = np.percentile(data[feature].dropna(), percentiles)
        thresholds = sorted(set(np.round(thresholds, 4)))
        
        records = []
        for thresh in thresholds:
            below = data[data[feature] < thresh]
            above = data[data[feature] >= thresh]
            
            s_below = self._quick_stats(below, min_samples=5)
            s_above = self._quick_stats(above, min_samples=5)
            
            records.append({
                'threshold': thresh,
                'n_below': len(below),
                'win_rate_below': s_below.get('win_rate', np.nan) if s_below else np.nan,
                'avg_return_below': s_below.get('avg_return', np.nan) if s_below else np.nan,
                'n_above': len(above),
                'win_rate_above': s_above.get('win_rate', np.nan) if s_above else np.nan,
                'avg_return_above': s_above.get('avg_return', np.nan) if s_above else np.nan,
            })
        
        table = pd.DataFrame(records)
        
        # 최적 임계값 찾기 (충분한 샘플이 있는 것 중에서)
        reliable_below = table[table['n_below'] >= min_samples]
        reliable_above = table[table['n_above'] >= min_samples]
        
        optimal_below = None
        if len(reliable_below) > 0 and reliable_below['win_rate_below'].notna().any():
            best_idx = reliable_below['win_rate_below'].idxmax()
            optimal_below = reliable_below.loc[best_idx].to_dict()
        
        optimal_above = None
        if len(reliable_above) > 0 and reliable_above['win_rate_above'].notna().any():
            best_idx = reliable_above['win_rate_above'].idxmax()
            optimal_above = reliable_above.loc[best_idx].to_dict()
        
        return {
            'feature': feature,
            'cycle_type_filter': cycle_type,
            'threshold_table': table,
            'optimal_below': optimal_below,
            'optimal_above': optimal_above
        }
    
    # =========================================================
    # 내부 헬퍼
    # =========================================================
    
    def _prepare_data(self, feature: str, cycle_type: Optional[str]) -> Optional[pd.DataFrame]:
        """필터링 및 NaN 제거된 데이터 준비."""
        required = [feature, self.target_col]
        if any(c not in self.df.columns for c in required):
            return None
        
        data = self.df[required + [self.type_col]].dropna(subset=required)
        
        if cycle_type:
            data = data[data[self.type_col].str.upper() == cycle_type.upper()]
        
        return data if len(data) > 0 else None
    
    def _calculate_bin_stats(self, data: pd.DataFrame, feature: str,
                             min_samples: int) -> pd.DataFrame:
        """구간별 통계 계산."""
        records = []
        
        for bin_label, group in data.groupby('bin', observed=True):
            target = group[self.target_col]
            n = len(target)
            
            if n == 0:
                continue
            
            win_rate = (target > 0).mean() * 100
            avg_return = target.mean()
            median_return = target.median()
            std_return = target.std() if n > 1 else 0
            
            # 승률의 95% 신뢰구간 (Wilson 구간)
            ci_lower, ci_upper = self._wilson_ci(win_rate / 100, n)
            
            records.append({
                'range': str(bin_label),
                'n': n,
                'win_rate': round(win_rate, 2),
                'ci_lower': round(ci_lower * 100, 2),
                'ci_upper': round(ci_upper * 100, 2),
                'avg_return': round(avg_return, 4),
                'median_return': round(median_return, 4),
                'std_return': round(std_return, 4),
                'feature_mean': round(group[feature].mean(), 4),
                'reliable': n >= min_samples
            })
        
        return pd.DataFrame(records)
    
    def _calculate_overall_stats(self, data: pd.DataFrame) -> Dict:
        """전체 데이터의 기본 통계."""
        target = data[self.target_col]
        n = len(target)
        win_rate = (target > 0).mean() * 100
        
        return {
            'n': n,
            'win_rate': round(win_rate, 2),
            'avg_return': round(target.mean(), 4),
            'median_return': round(target.median(), 4),
            'std_return': round(target.std(), 4)
        }
    
    def _calculate_group_stats(self, data: pd.DataFrame, label: str) -> Dict:
        """그룹의 통계 (복합 조건용)."""
        if self.target_col not in data.columns:
            return {'n': 0, 'label': label}
        
        target = data[self.target_col].dropna()
        n = len(target)
        
        if n == 0:
            return {'n': 0, 'label': label}
        
        win_rate = (target > 0).mean() * 100
        ci_lower, ci_upper = self._wilson_ci(win_rate / 100, n)
        
        return {
            'label': label,
            'n': n,
            'win_rate': round(win_rate, 2),
            'ci_lower': round(ci_lower * 100, 2),
            'ci_upper': round(ci_upper * 100, 2),
            'avg_return': round(target.mean(), 4),
            'median_return': round(target.median(), 4),
            'std_return': round(target.std(), 4),
            'min_return': round(target.min(), 4),
            'max_return': round(target.max(), 4)
        }
    
    def _quick_stats(self, data: pd.DataFrame, min_samples: int) -> Optional[Dict]:
        """빠른 통계 계산 (패턴 탐색용). 최소 샘플 미달 시 None 반환."""
        if self.target_col not in data.columns:
            return None
        
        target = data[self.target_col].dropna()
        n = len(target)
        
        if n < min_samples:
            return None
        
        win_rate = (target > 0).mean() * 100
        avg_return = target.mean()
        ci_lower, ci_upper = self._wilson_ci(win_rate / 100, n)
        
        return {
            'n': n,
            'win_rate': round(win_rate, 2),
            'ci_lower': round(ci_lower * 100, 2),
            'ci_upper': round(ci_upper * 100, 2),
            'avg_return': round(avg_return, 4),
            'median_return': round(target.median(), 4)
        }
    
    @staticmethod
    def _wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
        """
        Wilson 신뢰구간: 소표본에서도 안정적인 비율의 신뢰구간.
        일반적인 Wald 구간(p ± z*sqrt(p(1-p)/n))보다 소표본에서 훨씬 정확.
        """
        if n == 0:
            return (0.0, 1.0)
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        
        lower = max(0, center - margin)
        upper = min(1, center + margin)
        
        return (lower, upper)