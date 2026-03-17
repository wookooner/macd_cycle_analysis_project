"""
통계 엔진 (Statistical Engine)
==============================
모든 분석 모듈이 공유하는 통계 검정/효과크기/분포 분석 함수 모음.
모든 함수는 구조화된 딕셔너리를 반환 (print 없음).

사용법:
    from feature_analysis.core.stat_engine import StatEngine
    
    result = StatEngine.compare_groups(group_a, group_b)
    result = StatEngine.distribution_profile(series)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.stats import (
    mannwhitneyu, ks_2samp, levene, shapiro, 
    anderson, jarque_bera, normaltest,
    pearsonr, spearmanr, kruskal
)


class StatEngine:
    """통계 분석 엔진. 모든 메서드는 정적 메서드로 상태 없음."""
    
    # =========================================================
    # 1. 두 그룹 비교 (UP vs DOWN)
    # =========================================================
    
    @staticmethod
    def compare_groups(group_a: np.ndarray, group_b: np.ndarray,
                       label_a: str = "A", label_b: str = "B",
                       alpha: float = 0.05) -> Dict[str, Any]:
        """
        두 그룹의 종합 비교 분석.
        
        포함 검정:
        - Mann-Whitney U (비모수 중앙값 비교)
        - Kolmogorov-Smirnov (분포 전체 비교)
        - Levene (분산 동질성)
        
        포함 효과크기:
        - Cliff's delta (비모수 효과크기, 권장)
        - Rank-biserial correlation
        - Cohen's d (참고용, 정규성 가정)
        
        Returns:
            {
                'descriptive': {label_a: {...}, label_b: {...}},
                'tests': {'mann_whitney': {...}, 'ks': {...}, 'levene': {...}},
                'effect_sizes': {'cliffs_delta': ..., 'rank_biserial': ..., 'cohens_d': ...},
                'interpretation': str,
                'is_significant': bool
            }
        """
        a = np.array(group_a, dtype=float)
        b = np.array(group_b, dtype=float)
        
        # NaN 제거
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        
        if len(a) < 2 or len(b) < 2:
            return {'error': '각 그룹에 최소 2개 이상의 유효한 값이 필요합니다.',
                    'n_a': len(a), 'n_b': len(b)}
        
        # 기술통계
        descriptive = {
            label_a: StatEngine._descriptive_stats(a),
            label_b: StatEngine._descriptive_stats(b)
        }
        
        # 검정
        mw_stat, mw_p = mannwhitneyu(a, b, alternative='two-sided')
        ks_stat, ks_p = ks_2samp(a, b)
        lev_stat, lev_p = levene(a, b)
        
        tests = {
            'mann_whitney': {'statistic': mw_stat, 'p_value': mw_p,
                            'significant': mw_p < alpha},
            'ks': {'statistic': ks_stat, 'p_value': ks_p,
                   'significant': ks_p < alpha},
            'levene': {'statistic': lev_stat, 'p_value': lev_p,
                       'significant': lev_p < alpha}
        }
        
        # 효과크기
        cd = StatEngine.cliffs_delta(a, b)
        rb = StatEngine.rank_biserial(a, b)
        cohd = StatEngine.cohens_d(a, b)
        
        effect_sizes = {
            'cliffs_delta': cd,
            'rank_biserial': rb,
            'cohens_d': cohd
        }
        
        # 종합 해석
        interp = StatEngine._interpret_comparison(tests, effect_sizes, label_a, label_b)
        
        return {
            'descriptive': descriptive,
            'tests': tests,
            'effect_sizes': effect_sizes,
            'interpretation': interp,
            'is_significant': mw_p < alpha,
            'n_a': len(a),
            'n_b': len(b)
        }
    
    # =========================================================
    # 2. 효과크기 (Effect Size) 계산
    # =========================================================
    
    @staticmethod
    def cliffs_delta(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """
        Cliff's delta: 비모수 효과크기.
        -1 ~ +1 범위. |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
        의미: a에서 임의 추출한 값이 b에서 임의 추출한 값보다 클 확률 - 작을 확률.
        """
        a, b = np.asarray(a), np.asarray(b)
        # 모든 쌍의 비교 (벡터화)
        # a[i] > b[j]인 쌍의 수 - a[i] < b[j]인 쌍의 수
        n_a, n_b = len(a), len(b)
        
        # 메모리 효율적 계산: 큰 데이터는 청크로 처리
        if n_a * n_b > 1_000_000:
            delta = StatEngine._cliffs_delta_chunked(a, b)
        else:
            # 브로드캐스팅으로 모든 쌍 비교
            diff_matrix = a[:, None] - b[None, :]
            more = np.sum(diff_matrix > 0)
            less = np.sum(diff_matrix < 0)
            delta = (more - less) / (n_a * n_b)
        
        magnitude = StatEngine._effect_magnitude_cliffs(abs(delta))
        
        return {
            'value': delta,
            'magnitude': magnitude,
            'description': f"Cliff's delta = {delta:.4f} ({magnitude})"
        }
    
    @staticmethod
    def _cliffs_delta_chunked(a: np.ndarray, b: np.ndarray, chunk_size: int = 1000) -> float:
        """큰 배열에 대한 청크 기반 Cliff's delta 계산."""
        n_a, n_b = len(a), len(b)
        more, less = 0, 0
        for i in range(0, n_a, chunk_size):
            chunk_a = a[i:i + chunk_size]
            diff = chunk_a[:, None] - b[None, :]
            more += np.sum(diff > 0)
            less += np.sum(diff < 0)
        return (more - less) / (n_a * n_b)
    
    @staticmethod
    def rank_biserial(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """
        Rank-biserial correlation: Mann-Whitney U 기반 효과크기.
        r = 1 - 2U/(n1*n2). -1 ~ +1 범위.
        """
        a, b = np.asarray(a), np.asarray(b)
        n_a, n_b = len(a), len(b)
        
        try:
            u_stat, _ = mannwhitneyu(a, b, alternative='two-sided')
            r = 1 - (2 * u_stat) / (n_a * n_b)
        except Exception:
            r = 0.0
        
        magnitude = StatEngine._effect_magnitude_r(abs(r))
        
        return {
            'value': r,
            'magnitude': magnitude,
            'description': f"Rank-biserial r = {r:.4f} ({magnitude})"
        }
    
    @staticmethod
    def cohens_d(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """
        Cohen's d: 표준화된 평균 차이 (참고용, 정규분포 가정).
        |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, else large.
        """
        a, b = np.asarray(a), np.asarray(b)
        n_a, n_b = len(a), len(b)
        
        # pooled std
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        if pooled_std == 0:
            d = 0.0
        else:
            d = (np.mean(a) - np.mean(b)) / pooled_std
        
        magnitude = StatEngine._effect_magnitude_d(abs(d))
        
        return {
            'value': d,
            'magnitude': magnitude,
            'description': f"Cohen's d = {d:.4f} ({magnitude})"
        }
    
    # =========================================================
    # 3. 분포 프로파일링
    # =========================================================
    
    @staticmethod
    def distribution_profile(data: np.ndarray, name: str = "") -> Dict[str, Any]:
        """
        단일 변수의 종합 분포 프로파일.
        
        포함: 기술통계, 정규성 검정, 이상치 탐지, 분포 특성.
        """
        arr = np.array(data, dtype=float)
        arr = arr[~np.isnan(arr)]
        
        if len(arr) < 8:
            return {'error': f'유효 데이터 부족 (n={len(arr)})', 'name': name}
        
        result = {
            'name': name,
            'n': len(arr),
            'descriptive': StatEngine._descriptive_stats(arr),
            'normality': StatEngine._normality_tests(arr),
            'outliers': StatEngine._outlier_detection(arr),
            'distribution_shape': StatEngine._distribution_shape(arr)
        }
        
        return result
    
    # =========================================================
    # 4. 상관분석
    # =========================================================
    
    @staticmethod
    def correlation(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Pearson + Spearman 상관계수 동시 계산."""
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        
        if len(x_clean) < 3:
            return {'error': '유효 데이터 부족'}
        
        pr, pp = pearsonr(x_clean, y_clean)
        sr, sp = spearmanr(x_clean, y_clean)
        
        return {
            'pearson': {'r': pr, 'p_value': pp},
            'spearman': {'r': sr, 'p_value': sp},
            'n': len(x_clean)
        }
    
    # =========================================================
    # 5. 다중 그룹 비교 (4분류 등)
    # =========================================================
    
    @staticmethod
    def compare_multiple_groups(groups: Dict[str, np.ndarray],
                                alpha: float = 0.05) -> Dict[str, Any]:
        """
        3개 이상 그룹의 비교.
        Kruskal-Wallis 검정 + 사후 pairwise 비교.
        """
        # NaN 제거
        clean_groups = {}
        for name, arr in groups.items():
            arr = np.array(arr, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) >= 2:
                clean_groups[name] = arr
        
        if len(clean_groups) < 2:
            return {'error': '유효한 그룹이 2개 미만'}
        
        # Kruskal-Wallis
        group_arrays = list(clean_groups.values())
        h_stat, h_p = kruskal(*group_arrays)
        
        # 기술통계
        descriptive = {name: StatEngine._descriptive_stats(arr) 
                      for name, arr in clean_groups.items()}
        
        # Pairwise 비교
        pairwise = {}
        names = list(clean_groups.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pair_key = f"{names[i]}_vs_{names[j]}"
                a, b = clean_groups[names[i]], clean_groups[names[j]]
                mw_stat, mw_p = mannwhitneyu(a, b, alternative='two-sided')
                cd = StatEngine.cliffs_delta(a, b)
                pairwise[pair_key] = {
                    'mann_whitney_p': mw_p,
                    'significant': mw_p < alpha,
                    'cliffs_delta': cd['value'],
                    'effect_magnitude': cd['magnitude']
                }
        
        return {
            'kruskal_wallis': {'statistic': h_stat, 'p_value': h_p, 
                              'significant': h_p < alpha},
            'descriptive': descriptive,
            'pairwise': pairwise
        }
    
    # =========================================================
    # 내부 헬퍼
    # =========================================================
    
    @staticmethod
    def _descriptive_stats(arr: np.ndarray) -> Dict:
        """기술통계 딕셔너리."""
        return {
            'n': len(arr),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
            'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            'skew': float(pd.Series(arr).skew()),
            'kurtosis': float(pd.Series(arr).kurtosis())
        }
    
    @staticmethod
    def _normality_tests(arr: np.ndarray) -> Dict:
        """정규성 검정 모음."""
        results = {}
        
        # Shapiro-Wilk (n <= 5000)
        if len(arr) <= 5000:
            try:
                sw_stat, sw_p = shapiro(arr)
                results['shapiro_wilk'] = {'statistic': sw_stat, 'p_value': sw_p,
                                           'is_normal': sw_p > 0.05}
            except Exception:
                pass
        
        # Anderson-Darling
        try:
            ad_result = anderson(arr, dist='norm')
            # 5% 유의수준 기준
            idx = list(ad_result.significance_level).index(5.0) if 5.0 in ad_result.significance_level else 2
            results['anderson_darling'] = {
                'statistic': ad_result.statistic,
                'critical_value_5pct': ad_result.critical_values[idx],
                'is_normal': ad_result.statistic < ad_result.critical_values[idx]
            }
        except Exception:
            pass
        
        # Jarque-Bera
        try:
            jb_stat, jb_p = jarque_bera(arr)
            results['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_p,
                                      'is_normal': jb_p > 0.05}
        except Exception:
            pass
        
        # D'Agostino-Pearson
        if len(arr) >= 20:
            try:
                dag_stat, dag_p = normaltest(arr)
                results['dagostino_pearson'] = {'statistic': dag_stat, 'p_value': dag_p,
                                                'is_normal': dag_p > 0.05}
            except Exception:
                pass
        
        # 종합 판단
        normal_votes = sum(1 for t in results.values() if t.get('is_normal', False))
        results['consensus'] = {
            'normal_votes': normal_votes,
            'total_tests': len(results) - 1,  # consensus 자체 제외
            'likely_normal': normal_votes > (len(results) - 1) / 2
        }
        
        return results
    
    @staticmethod
    def _outlier_detection(arr: np.ndarray) -> Dict:
        """이상치 탐지 (IQR, Z-score, Modified Z-score)."""
        q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q75 - q25
        
        # IQR 기반
        iqr_lower = q25 - 1.5 * iqr
        iqr_upper = q75 + 1.5 * iqr
        iqr_outliers = np.sum((arr < iqr_lower) | (arr > iqr_upper))
        
        # Z-score 기반
        if np.std(arr) > 0:
            z_scores = np.abs((arr - np.mean(arr)) / np.std(arr))
            z_outliers = np.sum(z_scores > 3)
        else:
            z_outliers = 0
        
        # Modified Z-score (MAD 기반, 로버스트)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad > 0:
            modified_z = 0.6745 * (arr - med) / mad
            mad_outliers = np.sum(np.abs(modified_z) > 3.5)
        else:
            mad_outliers = 0
        
        return {
            'iqr': {'count': int(iqr_outliers), 'pct': iqr_outliers / len(arr) * 100,
                    'bounds': (iqr_lower, iqr_upper)},
            'z_score': {'count': int(z_outliers), 'pct': z_outliers / len(arr) * 100},
            'modified_z': {'count': int(mad_outliers), 'pct': mad_outliers / len(arr) * 100}
        }
    
    @staticmethod
    def _distribution_shape(arr: np.ndarray) -> Dict:
        """분포 형태 특성 분석."""
        skew = float(pd.Series(arr).skew())
        kurt = float(pd.Series(arr).kurtosis())
        
        # 왜도 해석
        if abs(skew) < 0.5:
            skew_desc = "대칭"
        elif skew > 0:
            skew_desc = "오른쪽 꼬리" if skew < 1 else "강한 오른쪽 꼬리"
        else:
            skew_desc = "왼쪽 꼬리" if skew > -1 else "강한 왼쪽 꼬리"
        
        # 첨도 해석 (excess kurtosis: 0이 정규분포)
        if abs(kurt) < 0.5:
            kurt_desc = "정규분포와 유사"
        elif kurt > 0:
            kurt_desc = "뾰족한 분포 (heavy tails)"
        else:
            kurt_desc = "납작한 분포 (light tails)"
        
        return {
            'skewness': skew,
            'skew_description': skew_desc,
            'kurtosis': kurt,
            'kurtosis_description': kurt_desc,
            'range': float(np.max(arr) - np.min(arr)),
            'cv': float(np.std(arr) / np.mean(arr)) if np.mean(arr) != 0 else np.inf
        }
    
    @staticmethod
    def _effect_magnitude_cliffs(abs_delta: float) -> str:
        if abs_delta < 0.147:
            return "negligible"
        elif abs_delta < 0.33:
            return "small"
        elif abs_delta < 0.474:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def _effect_magnitude_d(abs_d: float) -> str:
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def _effect_magnitude_r(abs_r: float) -> str:
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def _interpret_comparison(tests: Dict, effect_sizes: Dict,
                              label_a: str, label_b: str) -> str:
        """종합 해석문 생성."""
        mw_sig = tests['mann_whitney']['significant']
        mw_p = tests['mann_whitney']['p_value']
        cd = effect_sizes['cliffs_delta']
        
        if not mw_sig:
            return (f"{label_a}와 {label_b} 간 통계적으로 유의한 차이 없음 "
                    f"(p={mw_p:.4f}). 효과크기도 {cd['magnitude']} 수준.")
        
        direction = f"{label_a} > {label_b}" if cd['value'] > 0 else f"{label_b} > {label_a}"
        return (f"{label_a}와 {label_b} 간 유의한 차이 있음 (p={mw_p:.4f}). "
                f"방향: {direction}. "
                f"효과크기: {cd['description']}")