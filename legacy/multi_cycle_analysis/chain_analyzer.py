"""
4단계 체인 분석기 (Multi-Timeframe Chain Analyzer)
==================================================
1w → 1d → 4h → 1h 4개 시간대의 사이클 방향 조합(2⁴ = 16가지)별로
1h 사이클의 price_pct 통계를 분석.

핵심 가정: 큰 시간대일수록 더 강한 영향력과 방향성을 가진다.

사용법:
    from feature_analysis.analyzers.chain_analyzer import ChainAnalyzer
    
    analyzer = ChainAnalyzer(
        data_dir="data/cycle_data/structured",
    )
    results = analyzer.run_full_analysis()
    analyzer.plot_results(results, save_dir="feature_analysis/output")
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import product as iter_product

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings('ignore', category=FutureWarning)


# ==============================================================
# 통계 유틸
# ==============================================================

def compute_group_stats(series: pd.Series) -> dict:
    """단일 그룹의 가격변화율 종합 통계."""
    clean = series.dropna()
    n = len(clean)
    
    if n < 3:
        return {'n': n, 'insufficient_data': True}
    
    # 기술 통계
    desc = {
        'n': n,
        'mean': float(clean.mean()),
        'median': float(clean.median()),
        'std': float(clean.std()),
        'min': float(clean.min()),
        'max': float(clean.max()),
        'q25': float(clean.quantile(0.25)),
        'q75': float(clean.quantile(0.75)),
        'iqr': float(clean.quantile(0.75) - clean.quantile(0.25)),
        'skewness': float(sp_stats.skew(clean)),
        'kurtosis': float(sp_stats.kurtosis(clean)),
    }
    
    # 95% 신뢰구간 (평균)
    se = clean.std() / np.sqrt(n)
    ci_margin = sp_stats.t.ppf(0.975, df=n - 1) * se
    desc['mean_ci_lower'] = desc['mean'] - ci_margin
    desc['mean_ci_upper'] = desc['mean'] + ci_margin
    
    # 확률 통계
    n_pos = int((clean > 0).sum())
    n_neg = int((clean < 0).sum())
    
    prob = {
        'p_positive': n_pos / n,
        'p_negative': n_neg / n,
        'n_positive': n_pos,
        'n_negative': n_neg,
    }
    
    # Wilson 신뢰구간 (상승 확률)
    p_hat = n_pos / n
    z = 1.96
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
    prob['p_positive_ci_lower'] = max(0, center - spread)
    prob['p_positive_ci_upper'] = min(1, center + spread)
    
    # 조건부 통계
    pos_vals = clean[clean > 0]
    neg_vals = clean[clean < 0]
    
    cond = {
        'mean_when_positive': float(pos_vals.mean()) if len(pos_vals) > 0 else None,
        'mean_when_negative': float(neg_vals.mean()) if len(neg_vals) > 0 else None,
        'median_when_positive': float(pos_vals.median()) if len(pos_vals) > 0 else None,
        'median_when_negative': float(neg_vals.median()) if len(neg_vals) > 0 else None,
    }
    
    # 기대값
    ev = 0
    if len(pos_vals) > 0:
        ev += prob['p_positive'] * cond['mean_when_positive']
    if len(neg_vals) > 0:
        ev += prob['p_negative'] * cond['mean_when_negative']
    cond['expected_value'] = ev
    
    # Profit Factor
    total_gain = pos_vals.sum() if len(pos_vals) > 0 else 0
    total_loss = abs(neg_vals.sum()) if len(neg_vals) > 0 else 0
    cond['profit_factor'] = float(total_gain / total_loss) if total_loss > 0 else float('inf')
    
    # 분위수
    quantiles = {
        'p5': float(clean.quantile(0.05)),
        'p10': float(clean.quantile(0.10)),
        'p90': float(clean.quantile(0.90)),
        'p95': float(clean.quantile(0.95)),
    }
    
    return {
        'descriptive': desc,
        'probability': prob,
        'conditional': cond,
        'quantiles': quantiles,
    }


def compare_two_groups(a: pd.Series, b: pd.Series,
                       label_a: str = "A", label_b: str = "B") -> dict:
    """두 그룹 Mann-Whitney U + Cliff's delta 비교."""
    a_clean = a.dropna()
    b_clean = b.dropna()
    
    if len(a_clean) < 5 or len(b_clean) < 5:
        return {'insufficient_data': True, 'n_a': len(a_clean), 'n_b': len(b_clean)}
    
    u_stat, u_p = sp_stats.mannwhitneyu(a_clean, b_clean, alternative='two-sided')
    ks_stat, ks_p = sp_stats.ks_2samp(a_clean, b_clean)
    
    # Cliff's delta
    diff = np.subtract.outer(a_clean.values, b_clean.values)
    cliffs_d = float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(a_clean) * len(b_clean)))
    
    if abs(cliffs_d) < 0.147:
        mag = 'negligible'
    elif abs(cliffs_d) < 0.33:
        mag = 'small'
    elif abs(cliffs_d) < 0.474:
        mag = 'medium'
    else:
        mag = 'large'
    
    return {
        'mann_whitney_p': float(u_p),
        'ks_p': float(ks_p),
        'cliffs_delta': cliffs_d,
        'cliffs_magnitude': mag,
        'mean': {label_a: float(a_clean.mean()), label_b: float(b_clean.mean())},
        'median': {label_a: float(a_clean.median()), label_b: float(b_clean.median())},
        'n': {label_a: len(a_clean), label_b: len(b_clean)},
        'is_significant': u_p < 0.05,
    }


# ==============================================================
# ChainAnalyzer 메인 클래스
# ==============================================================

class ChainAnalyzer:
    """
    1w → 1d → 4h → 1h 4단계 체인 분석기.
    
    1h 사이클을 기준으로, 각 상위 시간대의 사이클 방향을 매핑하여
    2⁴ = 16가지 조합별 price_pct 통계를 분석.
    """
    
    CHAIN_TFS = ['1w', '1d', '4h', '1h']  # 상위 → 하위 순서
    TARGET_TF = '1h'
    
    def __init__(self,
                 data_dir: str = "data/cycle_data/structured",
                 hierarchy_filename: str = "cycle_hierarchy_map.json"):
        
        self.data_dir = Path(data_dir)
        self.hierarchy_path = self.data_dir / hierarchy_filename
        
        self.hierarchy = self._load_hierarchy()
        self.cycle_data = self._load_cycle_data()
    
    def _load_hierarchy(self) -> dict:
        with open(self.hierarchy_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_cycle_data(self) -> Dict[str, pd.DataFrame]:
        """parquet → flat DataFrame 로딩."""
        data = {}
        for tf in self.CHAIN_TFS:
            path = self.data_dir / f"cycles_{tf}.parquet"
            if not path.exists():
                print(f"  ⚠ {path} 없음")
                continue
            df = pd.read_parquet(path)
            df = self._flatten_features(df)
            data[tf] = df
            print(f"  ✓ {tf}: {len(df)}개 로딩")
        return data
    
    def _flatten_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'cycle_features' not in df.columns:
            return df
        
        records = []
        for feat_dict in df['cycle_features']:
            flat = {}
            if isinstance(feat_dict, dict):
                for cat, cat_dict in feat_dict.items():
                    if isinstance(cat_dict, dict):
                        for key, val in cat_dict.items():
                            flat[f"{cat}_{key}"] = val
            records.append(flat)
        
        feat_df = pd.DataFrame(records, index=df.index)
        base_cols = [c for c in df.columns if c not in ('cycle_features', 'candle_data')]
        return pd.concat([df[base_cols], feat_df], axis=1)
    
    # ==========================================================
    # 체인 구축
    # ==========================================================
    
    def build_chain_table(self) -> pd.DataFrame:
        """
        1h 사이클 기준으로 4단계 체인 테이블 구축.
        
        각 1h 사이클에 대해 parent_cycle_ids에서
        4h, 1d, 1w의 cycle_type을 직접 조회.
        
        Returns:
            DataFrame columns:
                cycle_id_1h, type_1w, type_1d, type_4h, type_1h,
                chain_key (예: "UP_UP_DOWN_UP"),
                price_pct_1h, duration_1h, ...
                alignment_score (가중 정렬 점수)
        """
        rows = []
        
        h1_hierarchy = self.hierarchy.get('1h', {})
        
        # 각 시간대 type 빠른 조회
        type_lookup = {}
        for tf in self.CHAIN_TFS:
            if tf in self.hierarchy:
                type_lookup[tf] = {
                    cid: cdata['cycle_type'].upper()
                    for cid, cdata in self.hierarchy[tf].items()
                }
        
        # 1h parquet에서 피처 조회
        h1_lookup = None
        if '1h' in self.cycle_data:
            h1_lookup = self.cycle_data['1h'].set_index('cycle_id')
        
        for h1_id, h1_data in h1_hierarchy.items():
            parents = h1_data.get('parent_cycle_ids', {})
            
            # 각 상위 시간대 type 조회
            types = {}
            types['1h'] = h1_data['cycle_type'].upper()
            
            chain_complete = True
            for parent_tf in ['4h', '1d', '1w']:
                parent_ids = parents.get(parent_tf, [])
                if parent_ids and parent_ids[0] in type_lookup.get(parent_tf, {}):
                    types[parent_tf] = type_lookup[parent_tf][parent_ids[0]]
                else:
                    chain_complete = False
                    break
            
            if not chain_complete:
                continue
            
            # chain_key: "UP_DOWN_UP_DOWN" (1w_1d_4h_1h 순서)
            chain_key = '_'.join(types[tf] for tf in self.CHAIN_TFS)
            
            # 1h 피처
            price_pct = None
            duration = h1_data['duration_candles']
            extra = {}
            
            if h1_lookup is not None and h1_id in h1_lookup.index:
                row_data = h1_lookup.loc[h1_id]
                price_pct = row_data.get('change_price_pct', None)
                extra = {
                    'direction_pct': row_data.get('strength_direction_pct', None),
                    'avg_true_range': row_data.get('volatility_avg_true_range', None),
                    'start_rsi': row_data.get('start_rsi', None),
                    'start_hist': row_data.get('start_hist', None),
                    'peak_position': row_data.get('shape_peak_price_position', None),
                    'trough_position': row_data.get('shape_trough_price_position', None),
                }
            
            # Alignment Score (가중치: 1w=8, 1d=4, 4h=2, 1h=1)
            weights = {'1w': 8, '1d': 4, '4h': 2, '1h': 1}
            score = sum(
                weights[tf] * (1 if types[tf] == 'UP' else -1)
                for tf in self.CHAIN_TFS
            )
            
            # 정렬 수준
            n_up = sum(1 for tf in self.CHAIN_TFS if types[tf] == 'UP')
            
            row = {
                'cycle_id_1h': h1_id,
                'type_1w': types['1w'],
                'type_1d': types['1d'],
                'type_4h': types['4h'],
                'type_1h': types['1h'],
                'chain_key': chain_key,
                'price_pct': price_pct,
                'duration': duration,
                'alignment_score': score,
                'n_up_levels': n_up,
                **extra,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 타입 변환
        numeric_cols = ['price_pct', 'direction_pct', 'avg_true_range',
                        'start_rsi', 'start_hist', 'peak_position', 'trough_position']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 수익 여부
        df['profitable'] = df['price_pct'] > 0
        
        # 전체 정렬 상태
        df['full_alignment'] = df['chain_key'].apply(
            lambda k: 'ALL_UP' if k == 'UP_UP_UP_UP'
            else ('ALL_DOWN' if k == 'DOWN_DOWN_DOWN_DOWN'
                  else 'MIXED')
        )
        
        return df
    
    # ==========================================================
    # 분석 파이프라인
    # ==========================================================
    
    def run_full_analysis(self) -> dict:
        """
        전체 분석 실행.
        
        Returns:
            {
                'chain_table': DataFrame,
                'combo_16_stats': {chain_key: stats, ...},
                'alignment_score_analysis': {...},
                'n_up_levels_analysis': {...},
                'top_vs_bottom': {...},
                'per_level_impact': {...},
            }
        """
        print("\n" + "=" * 70)
        print("  4-Level Chain Analysis: 1w → 1d → 4h → 1h")
        print("  16 Combinations (2⁴)")
        print("=" * 70)
        
        # 1) 체인 테이블
        print("\n[1/6] 체인 테이블 구축...")
        table = self.build_chain_table()
        print(f"  완성된 체인: {len(table)}개 (전체 1h의 {len(table)/len(self.hierarchy['1h'])*100:.1f}%)")
        
        # 2) 16가지 조합별 통계
        print("\n[2/6] 16가지 조합별 price_pct 통계...")
        combo_stats = {}
        for key in sorted(table['chain_key'].unique()):
            subset = table[table['chain_key'] == key]['price_pct']
            combo_stats[key] = compute_group_stats(subset)
            combo_stats[key]['chain_key'] = key
        
        # 3) Alignment Score 분석 (-15 ~ +15)
        print("\n[3/6] Alignment Score 분석...")
        alignment_score = self._analyze_alignment_score(table)
        
        # 4) n_up_levels 분석 (0~4개 시간대 UP)
        print("\n[4/6] UP 시간대 수별 분석...")
        n_up_analysis = self._analyze_n_up_levels(table)
        
        # 5) Top vs Bottom 조합 비교
        print("\n[5/6] 최고 vs 최저 조합 비교...")
        top_bottom = self._compare_top_bottom(combo_stats, table)
        
        # 6) 각 시간대별 개별 영향도 (다른 조건 통제)
        print("\n[6/6] 시간대별 개별 영향도 분석...")
        per_level = self._analyze_per_level_impact(table)
        
        results = {
            'chain_table': table,
            'combo_16_stats': combo_stats,
            'alignment_score_analysis': alignment_score,
            'n_up_levels_analysis': n_up_analysis,
            'top_vs_bottom': top_bottom,
            'per_level_impact': per_level,
        }
        
        self._print_report(results)
        
        return results
    
    def _analyze_alignment_score(self, table: pd.DataFrame) -> dict:
        """
        Alignment Score (-15 ~ +15) vs price_pct 관계.
        1w=±8, 1d=±4, 4h=±2, 1h=±1
        """
        score_groups = {}
        for score in sorted(table['alignment_score'].unique()):
            subset = table[table['alignment_score'] == score]['price_pct']
            score_groups[int(score)] = compute_group_stats(subset)
        
        # Spearman 상관
        valid = table[['alignment_score', 'price_pct']].dropna()
        if len(valid) > 10:
            rho, p_val = sp_stats.spearmanr(valid['alignment_score'], valid['price_pct'])
        else:
            rho, p_val = None, None
        
        return {
            'by_score': score_groups,
            'spearman_rho': float(rho) if rho is not None else None,
            'spearman_p': float(p_val) if p_val is not None else None,
        }
    
    def _analyze_n_up_levels(self, table: pd.DataFrame) -> dict:
        """0~4개 시간대가 UP일 때의 price_pct 분석."""
        result = {}
        
        for n_up in range(5):
            subset = table[table['n_up_levels'] == n_up]['price_pct']
            result[n_up] = compute_group_stats(subset)
        
        # Kruskal-Wallis (전체 그룹 비교)
        groups = [
            table[table['n_up_levels'] == n]['price_pct'].dropna().values
            for n in range(5) if len(table[table['n_up_levels'] == n]) >= 5
        ]
        
        if len(groups) >= 2:
            kw_stat, kw_p = sp_stats.kruskal(*groups)
        else:
            kw_stat, kw_p = None, None
        
        return {
            'by_n_up': result,
            'kruskal_wallis_stat': float(kw_stat) if kw_stat else None,
            'kruskal_wallis_p': float(kw_p) if kw_p else None,
        }
    
    def _compare_top_bottom(self, combo_stats: dict, table: pd.DataFrame) -> dict:
        """기대값 기준 상위/하위 조합 비교."""
        # 충분한 샘플의 조합만
        valid = {
            k: v for k, v in combo_stats.items()
            if not v.get('insufficient_data') and v['descriptive']['n'] >= 20
        }
        
        if len(valid) < 4:
            return {'insufficient_data': True}
        
        # 기대값 정렬
        ranked = sorted(
            valid.items(),
            key=lambda x: x[1]['conditional']['expected_value'],
            reverse=True
        )
        
        top_keys = [k for k, _ in ranked[:3]]
        bottom_keys = [k for k, _ in ranked[-3:]]
        
        top_data = table[table['chain_key'].isin(top_keys)]['price_pct']
        bottom_data = table[table['chain_key'].isin(bottom_keys)]['price_pct']
        
        return {
            'top_3': {k: combo_stats[k] for k in top_keys},
            'bottom_3': {k: combo_stats[k] for k in bottom_keys},
            'comparison': compare_two_groups(top_data, bottom_data, "top_3", "bottom_3"),
        }
    
    def _analyze_per_level_impact(self, table: pd.DataFrame) -> dict:
        """
        각 시간대의 개별 영향도.
        다른 조건 고정 없이 단순 분리 → Cliff's delta로 영향 크기 측정.
        """
        result = {}
        
        for tf in self.CHAIN_TFS:
            col = f'type_{tf}'
            up_data = table[table[col] == 'UP']['price_pct']
            down_data = table[table[col] == 'DOWN']['price_pct']
            
            comparison = compare_two_groups(up_data, down_data, f"{tf}_UP", f"{tf}_DOWN")
            result[tf] = comparison
        
        return result
    
    # ==========================================================
    # 리포트 출력
    # ==========================================================
    
    def _print_report(self, results: dict):
        table = results['chain_table']
        combo = results['combo_16_stats']
        score_a = results['alignment_score_analysis']
        n_up_a = results['n_up_levels_analysis']
        per_level = results['per_level_impact']
        
        print("\n" + "=" * 70)
        print("  📊 4-LEVEL CHAIN ANALYSIS REPORT")
        print("=" * 70)
        
        # [1] 16가지 조합 테이블
        print("\n" + "-" * 70)
        print("  [1] 16가지 조합별 price_pct (1h 사이클 기준)")
        print("-" * 70)
        header = f"  {'1w':>4s} {'1d':>4s} {'4h':>4s} {'1h':>4s}  {'n':>5s} {'평균':>8s} {'중앙값':>8s} {'std':>7s} {'상승확률':>7s} {'95%CI':>13s} {'기대값':>8s} {'PF':>6s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        
        # 정렬: alignment_score 내림차순
        sorted_keys = sorted(
            combo.keys(),
            key=lambda k: combo[k].get('conditional', {}).get('expected_value', 0)
            if not combo[k].get('insufficient_data') else -999,
            reverse=True
        )
        
        for key in sorted_keys:
            s = combo[key]
            parts = key.split('_')  # UP_DOWN_UP_DOWN
            
            if s.get('insufficient_data'):
                print(f"  {parts[0]:>4s} {parts[1]:>4s} {parts[2]:>4s} {parts[3]:>4s}  {s['n']:5d}   (데이터 부족)")
                continue
            
            d = s['descriptive']
            p = s['probability']
            c = s['conditional']
            pf = c['profit_factor']
            pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
            ci = f"[{p['p_positive_ci_lower']:.1%},{p['p_positive_ci_upper']:.1%}]"
            
            # 하이라이트: 기대값 상위 3개
            marker = ""
            
            print(f"  {parts[0]:>4s} {parts[1]:>4s} {parts[2]:>4s} {parts[3]:>4s}  "
                  f"{d['n']:5d} {d['mean']:+8.3f} {d['median']:+8.3f} {d['std']:7.3f} "
                  f"{p['p_positive']:7.1%} {ci:>13s} {c['expected_value']:+8.3f} {pf_str:>6s}{marker}")
        
        # [2] Alignment Score 요약
        print("\n" + "-" * 70)
        print("  [2] Alignment Score (가중 정렬 점수) vs price_pct")
        print("      가중치: 1w=±8, 1d=±4, 4h=±2, 1h=±1")
        print("-" * 70)
        
        if score_a['spearman_rho'] is not None:
            sig = "✓" if score_a['spearman_p'] < 0.05 else "✗"
            print(f"  Spearman ρ = {score_a['spearman_rho']:+.4f}, p = {score_a['spearman_p']:.6f} {sig}")
        
        print(f"\n  {'Score':>6s} {'n':>5s} {'평균':>8s} {'중앙값':>8s} {'상승확률':>8s}")
        print("  " + "-" * 40)
        for score in sorted(score_a['by_score'].keys()):
            s = score_a['by_score'][score]
            if s.get('insufficient_data'):
                print(f"  {score:>6d} {s['n']:5d}   (부족)")
                continue
            d = s['descriptive']
            p = s['probability']
            print(f"  {score:>+6d} {d['n']:5d} {d['mean']:+8.3f} {d['median']:+8.3f} {p['p_positive']:8.1%}")
        
        # [3] UP 시간대 수별
        print("\n" + "-" * 70)
        print("  [3] UP 시간대 수 (0~4) vs price_pct")
        print("-" * 70)
        
        kw_p = n_up_a['kruskal_wallis_p']
        if kw_p is not None:
            sig = "✓ 그룹 간 유의한 차이" if kw_p < 0.05 else "✗ 그룹 간 차이 없음"
            print(f"  Kruskal-Wallis H p = {kw_p:.6f} → {sig}")
        
        print(f"\n  {'#UP':>4s} {'n':>5s} {'평균':>8s} {'중앙값':>8s} {'상승확률':>8s} {'기대값':>8s}")
        print("  " + "-" * 45)
        for n_up in range(5):
            s = n_up_a['by_n_up'].get(n_up, {})
            if s.get('insufficient_data') or 'descriptive' not in s:
                continue
            d = s['descriptive']
            p = s['probability']
            c = s['conditional']
            print(f"  {n_up:4d} {d['n']:5d} {d['mean']:+8.3f} {d['median']:+8.3f} "
                  f"{p['p_positive']:8.1%} {c['expected_value']:+8.3f}")
        
        # [4] 시간대별 개별 영향도
        print("\n" + "-" * 70)
        print("  [4] 시간대별 개별 영향도 (UP vs DOWN)")
        print("      큰 시간대가 더 강한 영향 → Cliff's δ가 더 큰가?")
        print("-" * 70)
        
        print(f"\n  {'시간대':>6s} {'Cliff δ':>10s} {'크기':>12s} {'p-value':>10s} {'유의':>4s} "
              f"{'UP평균':>8s} {'DN평균':>8s}")
        print("  " + "-" * 65)
        
        for tf in self.CHAIN_TFS:
            p = per_level.get(tf, {})
            if p.get('insufficient_data'):
                print(f"  {tf:>6s}   (데이터 부족)")
                continue
            
            sig = "✓" if p['is_significant'] else "✗"
            up_key = f"{tf}_UP"
            dn_key = f"{tf}_DOWN"
            
            print(f"  {tf:>6s} {p['cliffs_delta']:+10.4f} {p['cliffs_magnitude']:>12s} "
                  f"{p['mann_whitney_p']:10.6f} {sig:>4s} "
                  f"{p['mean'][up_key]:+8.3f} {p['mean'][dn_key]:+8.3f}")
        
        # [5] Top vs Bottom 조합
        print("\n" + "-" * 70)
        print("  [5] 최고 기대값 3개 vs 최저 기대값 3개 조합")
        print("-" * 70)
        
        tb = results['top_vs_bottom']
        if not tb.get('insufficient_data'):
            print("\n  ▸ Top 3:")
            for key, s in tb['top_3'].items():
                c = s['conditional']
                d = s['descriptive']
                print(f"    {key:25s} EV={c['expected_value']:+.3f}, WR={s['probability']['p_positive']:.1%}, n={d['n']}")
            
            print("\n  ▸ Bottom 3:")
            for key, s in tb['bottom_3'].items():
                c = s['conditional']
                d = s['descriptive']
                print(f"    {key:25s} EV={c['expected_value']:+.3f}, WR={s['probability']['p_positive']:.1%}, n={d['n']}")
            
            comp = tb['comparison']
            if not comp.get('insufficient_data'):
                print(f"\n  Top vs Bottom: Cliff's δ = {comp['cliffs_delta']:+.4f} ({comp['cliffs_magnitude']}), "
                      f"p = {comp['mann_whitney_p']:.6f}")
        
        print("\n" + "=" * 70)
        print("  분석 완료")
        print("=" * 70)
    
    # ==========================================================
    # 시각화
    # ==========================================================
    
    def plot_results(self, results: dict, save_dir: str = "feature_analysis/output"):
        """전체 시각화 (6개 차트)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        
        for font in ['Noto Sans CJK KR', 'NanumGothic', 'Malgun Gothic', 'DejaVu Sans']:
            try:
                matplotlib.font_manager.findfont(font, fallback_to_default=False)
                plt.rcParams['font.family'] = font
                break
            except Exception:
                continue
        plt.rcParams['axes.unicode_minus'] = False
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        table = results['chain_table']
        combo = results['combo_16_stats']
        paths = []
        
        # ============================
        # Chart 1: 16 조합 종합 바 차트
        # ============================
        valid_combos = {
            k: v for k, v in combo.items()
            if not v.get('insufficient_data') and v['descriptive']['n'] >= 10
        }
        
        sorted_keys = sorted(
            valid_combos.keys(),
            key=lambda k: valid_combos[k]['conditional']['expected_value'],
            reverse=True
        )
        
        fig, axes = plt.subplots(3, 1, figsize=(18, 14), 
                                  gridspec_kw={'height_ratios': [2, 1, 1]})
        fig.suptitle('16 Combinations: 1w → 1d → 4h → 1h Chain Analysis',
                     fontsize=16, fontweight='bold')
        
        labels = [k.replace('_', '→', 3) for k in sorted_keys]
        x = np.arange(len(sorted_keys))
        
        # 바 색상: 정렬 정도에 따라
        bar_colors = []
        for k in sorted_keys:
            parts = k.split('_')
            n_up = sum(1 for p in parts if p == 'UP')
            if n_up == 4:
                bar_colors.append('#27ae60')
            elif n_up == 3:
                bar_colors.append('#2ecc71')
            elif n_up == 2:
                bar_colors.append('#95a5a6')
            elif n_up == 1:
                bar_colors.append('#e74c3c')
            else:
                bar_colors.append('#c0392b')
        
        # Top: Expected Value
        ax = axes[0]
        evs = [valid_combos[k]['conditional']['expected_value'] for k in sorted_keys]
        bars = ax.bar(x, evs, color=bar_colors, alpha=0.85, edgecolor='white')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Expected Value (%)', fontsize=11)
        ax.set_title('Expected Value by Combination', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, evs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=8, fontweight='bold')
        
        # Middle: Win Rate
        ax = axes[1]
        wrs = [valid_combos[k]['probability']['p_positive'] * 100 for k in sorted_keys]
        ax.bar(x, wrs, color=bar_colors, alpha=0.85, edgecolor='white')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
        ax.set_ylabel('Win Rate (%)', fontsize=11)
        ax.set_title('Win Rate (P positive)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        # Bottom: Sample Size
        ax = axes[2]
        ns = [valid_combos[k]['descriptive']['n'] for k in sorted_keys]
        ax.bar(x, ns, color=bar_colors, alpha=0.6, edgecolor='white')
        ax.set_ylabel('Sample Count', fontsize=11)
        ax.set_title('Sample Size (n)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        p = save_path / "chain_16_combinations.png"
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(str(p))
        print(f"  ✓ {p}")
        
        # ============================
        # Chart 2: Alignment Score vs price_pct
        # ============================
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Alignment Score (-15 to +15) vs 1h price_pct',
                     fontsize=14, fontweight='bold')
        
        score_data = results['alignment_score_analysis']['by_score']
        
        # Left: 평균/중앙값 + CI
        ax = axes[0]
        scores = sorted(score_data.keys())
        means = []
        medians = []
        ci_lower = []
        ci_upper = []
        ns = []
        
        for s in scores:
            d = score_data[s]
            if d.get('insufficient_data'):
                means.append(np.nan)
                medians.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
                ns.append(d['n'])
            else:
                means.append(d['descriptive']['mean'])
                medians.append(d['descriptive']['median'])
                ci_lower.append(d['descriptive']['mean_ci_lower'])
                ci_upper.append(d['descriptive']['mean_ci_upper'])
                ns.append(d['descriptive']['n'])
        
        ax.plot(scores, means, 'o-', color='#3498db', label='Mean', linewidth=2, markersize=6)
        ax.plot(scores, medians, 's--', color='#e67e22', label='Median', linewidth=1.5, markersize=5)
        ax.fill_between(scores, ci_lower, ci_upper, alpha=0.2, color='#3498db', label='95% CI')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Alignment Score')
        ax.set_ylabel('price_pct (%)')
        ax.set_title('Mean & Median by Score')
        ax.legend()
        ax.grid(alpha=0.3)
        
        rho = results['alignment_score_analysis']['spearman_rho']
        rho_p = results['alignment_score_analysis']['spearman_p']
        if rho is not None:
            ax.text(0.02, 0.98, f'Spearman ρ={rho:+.4f}\np={rho_p:.4e}',
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Right: 상승확률 by score
        ax = axes[1]
        win_rates = []
        for s in scores:
            d = score_data[s]
            if d.get('insufficient_data'):
                win_rates.append(np.nan)
            else:
                win_rates.append(d['probability']['p_positive'] * 100)
        
        colors_bar = ['#2ecc71' if s > 0 else '#e74c3c' if s < 0 else '#95a5a6' for s in scores]
        ax.bar(scores, win_rates, color=colors_bar, alpha=0.8, width=1.5)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Alignment Score')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate by Score')
        ax.grid(axis='y', alpha=0.3)
        
        # n 표시
        for s, wr, n in zip(scores, win_rates, ns):
            if not np.isnan(wr):
                ax.text(s, wr + 1, f'n={n}', ha='center', fontsize=7, rotation=45)
        
        plt.tight_layout()
        p = save_path / "chain_alignment_score.png"
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(str(p))
        print(f"  ✓ {p}")
        
        # ============================
        # Chart 3: n_up_levels (0~4)
        # ============================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Number of UP Timeframes (0-4) vs 1h price_pct',
                     fontsize=14, fontweight='bold')
        
        n_up_data = results['n_up_levels_analysis']['by_n_up']
        
        # Left: Boxplot
        ax = axes[0]
        bp_data = []
        bp_labels = []
        for n_up in range(5):
            subset = table[table['n_up_levels'] == n_up]['price_pct'].dropna()
            if len(subset) > 0:
                bp_data.append(subset.values)
                bp_labels.append(f"{n_up}UP\n(n={len(subset)})")
        
        bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops=dict(color='white', linewidth=2))
        
        green_red = ['#c0392b', '#e74c3c', '#95a5a6', '#2ecc71', '#27ae60']
        for patch, color in zip(bp['boxes'], green_red[:len(bp_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('price_pct (%)')
        ax.set_title('Distribution by # UP Levels')
        ax.grid(axis='y', alpha=0.3)
        
        # Right: Win Rate + EV
        ax = axes[1]
        n_ups = []
        wrs = []
        evs_val = []
        for n_up in range(5):
            s = n_up_data.get(n_up, {})
            if s.get('insufficient_data') or 'probability' not in s:
                continue
            n_ups.append(n_up)
            wrs.append(s['probability']['p_positive'] * 100)
            evs_val.append(s['conditional']['expected_value'])
        
        ax2 = ax.twinx()
        ax.bar([n - 0.2 for n in n_ups], wrs, width=0.35, color='#3498db', alpha=0.7, label='Win Rate')
        ax2.bar([n + 0.2 for n in n_ups], evs_val, width=0.35, color='#e67e22', alpha=0.7, label='Expected Value')
        
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel('# UP Timeframes')
        ax.set_ylabel('Win Rate (%)', color='#3498db')
        ax2.set_ylabel('Expected Value (%)', color='#e67e22')
        ax.set_title('Win Rate & Expected Value')
        ax.set_xticks(range(5))
        ax.grid(axis='y', alpha=0.3)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        p = save_path / "chain_n_up_levels.png"
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(str(p))
        print(f"  ✓ {p}")
        
        # ============================
        # Chart 4: 시간대별 영향도 (Cliff's delta)
        # ============================
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Per-Level Impact: Cliff's Delta (UP vs DOWN)",
                     fontsize=14, fontweight='bold')
        
        per_level = results['per_level_impact']
        tfs = []
        deltas = []
        ps = []
        
        for tf in self.CHAIN_TFS:
            p_data = per_level.get(tf, {})
            if p_data.get('insufficient_data'):
                continue
            tfs.append(tf)
            deltas.append(p_data['cliffs_delta'])
            ps.append(p_data['mann_whitney_p'])
        
        colors_tf = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]
        bars = ax.barh(tfs, deltas, color=colors_tf, alpha=0.8, height=0.5)
        
        # 유의성 마커
        for i, (tf, d, p_val) in enumerate(zip(tfs, deltas, ps)):
            sig = "★" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
            ax.text(d + 0.005 * np.sign(d), i, f'{d:+.4f} {sig}\n(p={p_val:.4f})',
                    va='center', fontsize=10, fontweight='bold')
        
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Cliff's Delta")
        ax.set_title("가설 검증: 큰 시간대 → 더 큰 영향?")
        ax.grid(axis='x', alpha=0.3)
        
        # 효과크기 기준선
        for val, label in [(0.147, 'small'), (0.33, 'medium'), (0.474, 'large')]:
            ax.axvline(x=val, color='gray', linestyle=':', alpha=0.3)
            ax.axvline(x=-val, color='gray', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        p_path = save_path / "chain_per_level_impact.png"
        fig.savefig(p_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(str(p_path))
        print(f"  ✓ {p_path}")
        
        # ============================
        # Chart 5: 16조합 히트맵 (4×4 격자)
        # ============================
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('16 Combinations Heatmap (1h price_pct statistics)',
                     fontsize=14, fontweight='bold')
        
        # 4×4 격자: rows = 1w×1d (4 조합), cols = 4h×1h (4 조합)
        row_combos = list(iter_product(['UP', 'DOWN'], repeat=2))  # (1w, 1d)
        col_combos = list(iter_product(['UP', 'DOWN'], repeat=2))  # (4h, 1h)
        
        metrics = {
            'Mean price_pct': lambda s: s['descriptive']['mean'],
            'Win Rate (%)': lambda s: s['probability']['p_positive'] * 100,
            'Expected Value': lambda s: s['conditional']['expected_value'],
            'Profit Factor': lambda s: min(s['conditional']['profit_factor'], 5),
        }
        
        for idx, (title, extractor) in enumerate(metrics.items()):
            ax = axes[idx // 2][idx % 2]
            
            matrix = np.full((4, 4), np.nan)
            for i, (r1w, r1d) in enumerate(row_combos):
                for j, (c4h, c1h) in enumerate(col_combos):
                    key = f"{r1w}_{r1d}_{c4h}_{c1h}"
                    if key in combo and not combo[key].get('insufficient_data'):
                        matrix[i, j] = extractor(combo[key])
            
            cmap = 'RdYlGn'
            if title == 'Mean price_pct' or title == 'Expected Value':
                abs_max = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.01)
                im = ax.imshow(matrix, cmap=cmap, vmin=-abs_max, vmax=abs_max, aspect='auto')
            elif title == 'Win Rate (%)':
                im = ax.imshow(matrix, cmap=cmap, vmin=30, vmax=70, aspect='auto')
            else:
                im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(np.nanmax(matrix), 1), aspect='auto')
            
            ax.set_xticks(range(4))
            ax.set_xticklabels([f'4h{a}\n1h{b}' for a, b in col_combos], fontsize=9)
            ax.set_yticks(range(4))
            ax.set_yticklabels([f'1w{a} 1d{b}' for a, b in row_combos], fontsize=9)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('4h × 1h')
            ax.set_ylabel('1w × 1d')
            
            for i in range(4):
                for j in range(4):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        r1w, r1d = row_combos[i]
                        c4h, c1h = col_combos[j]
                        key = f"{r1w}_{r1d}_{c4h}_{c1h}"
                        n = combo[key]['descriptive']['n'] if key in combo and not combo[key].get('insufficient_data') else 0
                        fmt = f'{val:.1f}' if 'Rate' in title else f'{val:.3f}'
                        ax.text(j, i, f'{fmt}\nn={n}', ha='center', va='center',
                                fontsize=9, fontweight='bold')
            
            fig.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        p = save_path / "chain_16_heatmap.png"
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(str(p))
        print(f"  ✓ {p}")
        
        # ============================
        # Chart 6: 전체 vs 완전정렬 vs 완전역행 분포 비교
        # ============================
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle('Full Alignment Comparison: ALL_UP vs ALL_DOWN vs MIXED',
                     fontsize=14, fontweight='bold')
        
        for state, color, ls in [('ALL_UP', '#27ae60', '-'), ('ALL_DOWN', '#c0392b', '-'), ('MIXED', '#7f8c8d', '--')]:
            subset = table[table['full_alignment'] == state]['price_pct'].dropna()
            if len(subset) > 10:
                subset_clipped = subset.clip(subset.quantile(0.02), subset.quantile(0.98))
                ax.hist(subset_clipped, bins=50, alpha=0.4, color=color, density=True,
                        label=f'{state} (n={len(subset)}, μ={subset.mean():+.3f})',
                        edgecolor='none')
                # KDE
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(subset_clipped)
                    x_kde = np.linspace(subset_clipped.min(), subset_clipped.max(), 200)
                    ax.plot(x_kde, kde(x_kde), color=color, linewidth=2, linestyle=ls)
                except Exception:
                    pass
        
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('price_pct (%)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        p = save_path / "chain_full_alignment_dist.png"
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(str(p))
        print(f"  ✓ {p}")
        
        print(f"\n  총 {len(paths)}개 차트 생성")
        return paths