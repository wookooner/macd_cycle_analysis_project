"""
시퀀스/연속 사이클 분석기 (Sequence Analyzer)
=============================================
사이클 데이터의 시계열 특성을 활용하여
이전 사이클(N-1)이 현재 사이클(N)에 미치는 영향을 분석.

분석 내용:
1. 전이 확률 행렬: UP→UP, UP→DOWN, DOWN→UP, DOWN→DOWN 확률 및 수익률
2. 연속 패턴: N연속 DOWN 후 UP의 수익률 등
3. 이전 사이클 컨텍스트 피처 생성: prev_cycle_type, prev_price_pct 등

사용법:
    from feature_analysis.analyzers.sequence_analyzer import SequenceAnalyzer
    
    analyzer = SequenceAnalyzer(df)
    
    # 전이 확률 행렬
    matrix = analyzer.transition_matrix()
    
    # 연속 패턴 분석
    streaks = analyzer.streak_analysis(max_streak=5)
    
    # 이전 사이클 컨텍스트 피처 추가
    enriched_df = analyzer.add_context_features(n_prev=3)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


class SequenceAnalyzer:
    """사이클 시퀀스 및 전이 패턴 분석기."""
    
    def __init__(self, df: pd.DataFrame,
                 target_col: str = "change_price_pct",
                 type_col: str = "cycle_type"):
        """
        Args:
            df: 평탄화된 사이클 DataFrame (시간순 정렬 필요)
        """
        self.df = self._ensure_sorted(df)
        self.target_col = target_col
        self.type_col = type_col
    
    # =========================================================
    # 1. 전이 확률 행렬
    # =========================================================
    
    def transition_matrix(self) -> Dict[str, Any]:
        """
        사이클 타입 간 전이 확률 행렬 생성.
        
        Returns:
            {
                'probability_matrix': DataFrame (UP/DOWN → UP/DOWN 확률),
                'return_matrix': DataFrame (전이별 평균 수익률),
                'count_matrix': DataFrame (전이별 샘플 수),
                'details': dict (각 전이의 상세 통계)
            }
        """
        types = self.df[self.type_col].str.upper()
        target = self.df[self.target_col]
        
        # 현재 → 다음 매핑
        current_type = types.iloc[:-1].values
        next_type = types.iloc[1:].values
        next_return = target.iloc[1:].values
        
        # 전이별 집계
        transitions = ['UP', 'DOWN']
        
        prob_data = {}
        return_data = {}
        count_data = {}
        details = {}
        
        for from_type in transitions:
            prob_row = {}
            return_row = {}
            count_row = {}
            
            for to_type in transitions:
                mask = (current_type == from_type) & (next_type == to_type)
                n = mask.sum()
                total_from = (current_type == from_type).sum()
                
                prob = n / total_from if total_from > 0 else 0
                returns = next_return[mask]
                avg_ret = np.nanmean(returns) if n > 0 else 0
                
                prob_row[to_type] = round(prob * 100, 2)
                return_row[to_type] = round(avg_ret, 4) if n > 0 else 0
                count_row[to_type] = int(n)
                
                key = f"{from_type}→{to_type}"
                details[key] = {
                    'n': int(n),
                    'probability': round(prob * 100, 2),
                    'avg_return': round(avg_ret, 4) if n > 0 else 0,
                    'median_return': round(float(np.nanmedian(returns)), 4) if n > 0 else 0,
                    'win_rate': round((returns > 0).mean() * 100, 2) if n > 0 else 0,
                    'std_return': round(float(np.nanstd(returns)), 4) if n > 0 else 0
                }
            
            prob_data[from_type] = prob_row
            return_data[from_type] = return_row
            count_data[from_type] = count_row
        
        return {
            'probability_matrix': pd.DataFrame(prob_data).T,
            'return_matrix': pd.DataFrame(return_data).T,
            'count_matrix': pd.DataFrame(count_data).T,
            'details': details
        }
    
    # =========================================================
    # 2. 연속 패턴 분석 (Streak Analysis)
    # =========================================================
    
    def streak_analysis(self, max_streak: int = 5) -> Dict[str, Any]:
        """
        N연속 같은 방향 후 다음 사이클 분석.
        
        예: "3연속 DOWN 후 다음 UP 사이클의 평균 수익률은?"
        
        Args:
            max_streak: 분석할 최대 연속 횟수
            
        Returns:
            {
                'streaks': DataFrame (연속 횟수별 다음 사이클 통계),
                'up_streaks': dict,
                'down_streaks': dict
            }
        """
        types = self.df[self.type_col].str.upper().values
        target = self.df[self.target_col].values
        n = len(types)
        
        # 각 위치에서의 연속 횟수 계산
        # streak_count[i] = i번째 사이클까지 같은 방향이 몇 번 연속인지
        streak_counts = np.ones(n, dtype=int)
        for i in range(1, n):
            if types[i] == types[i-1]:
                streak_counts[i] = streak_counts[i-1] + 1
            else:
                streak_counts[i] = 1
        
        records = []
        
        for streak_type in ['UP', 'DOWN']:
            for streak_len in range(1, max_streak + 1):
                # streak_len 연속인 마지막 위치를 찾고, 그 다음 사이클 분석
                next_returns = []
                next_types = []
                
                for i in range(n - 1):
                    if (types[i] == streak_type and 
                        streak_counts[i] == streak_len and
                        (i + 1 >= n or types[i + 1] != streak_type or i == n - 2)):
                        # streak이 정확히 streak_len에서 끝나는 위치
                        pass
                    
                    # 단순화: streak_count == streak_len인 위치 다음
                    if (types[i] == streak_type and 
                        streak_counts[i] >= streak_len and
                        (i == 0 or streak_counts[i-1] < streak_len or types[i-1] != streak_type)):
                        # streak이 streak_len 이상이고 해당 지점에서 시작점 이후
                        pass
                
                # 더 정확한 접근: streak이 정확히 streak_len인 종료 지점 이후
                for i in range(streak_len - 1, n - 1):
                    # i에서 streak_count == streak_len 이고
                    # i+1에서 타입이 바뀌거나 배열 끝
                    if (streak_counts[i] == streak_len and 
                        types[i] == streak_type and
                        i + 1 < n):
                        # 다음 위치가 다른 타입이거나 streak이 끝난 지점
                        if i + 1 < n and (types[i+1] != streak_type or streak_counts[i+1] == 1):
                            # 정확히 streak_len에서 끝남
                            if not np.isnan(target[i+1]):
                                next_returns.append(target[i+1])
                                next_types.append(types[i+1])
                
                # streak_len 이상인 경우도 포함하는 간단한 버전
                # i번째가 streak의 마지막이고 streak_len 이상
                next_returns_ge = []
                for i in range(n - 1):
                    if (types[i] == streak_type and
                        streak_counts[i] >= streak_len and
                        (i + 1 >= n or types[i+1] != streak_type)):
                        if i + 1 < n and not np.isnan(target[i+1]):
                            next_returns_ge.append(target[i+1])
                
                if len(next_returns_ge) >= 3:
                    rets = np.array(next_returns_ge)
                    records.append({
                        'streak_type': streak_type,
                        'streak_length': f"{streak_len}+",
                        'n_occurrences': len(rets),
                        'next_win_rate': round((rets > 0).mean() * 100, 2),
                        'next_avg_return': round(np.mean(rets), 4),
                        'next_median_return': round(np.median(rets), 4),
                        'next_std_return': round(np.std(rets), 4)
                    })
        
        df_streaks = pd.DataFrame(records) if records else pd.DataFrame()
        
        return {'streaks': df_streaks}
    
    # =========================================================
    # 3. 컨텍스트 피처 생성
    # =========================================================
    
    def add_context_features(self, n_prev: int = 1,
                             features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        이전 N개 사이클의 특성을 현재 행에 컨텍스트 피처로 추가.
        
        Args:
            n_prev: 참조할 이전 사이클 수 (1이면 직전만, 2이면 직전+전전)
            features: 컨텍스트로 가져올 피처 리스트. 
                      None이면 기본 세트 (cycle_type, change_price_pct, duration_candles, start_hist)
            
        Returns:
            컨텍스트 피처가 추가된 DataFrame
            추가되는 컬럼: prev1_cycle_type, prev1_change_price_pct, ...
        """
        df = self.df.copy()
        
        if features is None:
            features = [self.type_col, self.target_col]
            # 선택적으로 추가
            optional = ['duration_candles', 'start_hist', 'start_rsi', 'start_macd']
            features.extend([f for f in optional if f in df.columns])
        
        for lag in range(1, n_prev + 1):
            prefix = f"prev{lag}"
            for feat in features:
                if feat in df.columns:
                    new_col = f"{prefix}_{feat}"
                    df[new_col] = df[feat].shift(lag)
        
        return df
    
    # =========================================================
    # 4. 이전 사이클 조건부 분석
    # =========================================================
    
    def prev_cycle_conditional(self, 
                                prev_conditions: List[Tuple[str, str, float]],
                                current_cycle_type: Optional[str] = None,
                                min_samples: int = 20) -> Dict[str, Any]:
        """
        이전 사이클의 조건을 지정하고 현재 사이클의 성과를 분석.
        
        예: "직전 DOWN 사이클이 -5% 이상 하락한 후 오는 UP 사이클의 성과"
        
        Args:
            prev_conditions: 이전 사이클에 적용할 조건 (prev1_ 접두사 자동 추가)
            current_cycle_type: 현재 사이클 타입 필터
        """
        # 컨텍스트 피처 추가
        df = self.add_context_features(n_prev=1)
        
        # 조건 적용 (prev1_ 접두사)
        mask = pd.Series(True, index=df.index)
        descriptions = []
        
        for feature, op, value in prev_conditions:
            col_name = f"prev1_{feature}" if not feature.startswith("prev") else feature
            
            if col_name not in df.columns:
                return {'error': f"Column '{col_name}' not found"}
            
            col = df[col_name]
            if op == '<':
                mask &= col < value
            elif op == '<=':
                mask &= col <= value
            elif op == '>':
                mask &= col > value
            elif op == '>=':
                mask &= col >= value
            elif op == '==':
                if isinstance(value, str):
                    mask &= col.astype(str).str.upper() == value.upper()
                else:
                    mask &= col == value
            
            descriptions.append(f"이전: {feature} {op} {value}")
        
        # 현재 사이클 타입 필터
        if current_cycle_type:
            mask &= df[self.type_col].str.upper() == current_cycle_type.upper()
            descriptions.append(f"현재: {current_cycle_type}")
        
        filtered = df[mask]
        complement = df[~mask]
        
        if len(filtered) < 3:
            return {'error': f'조건 충족 데이터 부족 (n={len(filtered)})',
                    'conditions': " AND ".join(descriptions)}
        
        target_filtered = filtered[self.target_col].dropna()
        target_comp = complement[self.target_col].dropna()
        
        result = {
            'conditions': " AND ".join(descriptions),
            'matched': {
                'n': len(target_filtered),
                'win_rate': round((target_filtered > 0).mean() * 100, 2),
                'avg_return': round(target_filtered.mean(), 4),
                'median_return': round(target_filtered.median(), 4),
                'std_return': round(target_filtered.std(), 4),
                'reliable': len(target_filtered) >= min_samples
            },
            'baseline': {
                'n': len(target_comp),
                'win_rate': round((target_comp > 0).mean() * 100, 2) if len(target_comp) > 0 else 0,
                'avg_return': round(target_comp.mean(), 4) if len(target_comp) > 0 else 0,
            }
        }
        
        return result
    
    # =========================================================
    # 내부 헬퍼
    # =========================================================
    
    def _ensure_sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간순 정렬 보장."""
        if 'start_datetime' in df.columns:
            return df.sort_values('start_datetime').reset_index(drop=True)
        elif 'start_date' in df.columns:
            try:
                temp = df.copy()
                temp['_sort_key'] = pd.to_datetime(temp['start_date'], errors='coerce')
                temp = temp.sort_values('_sort_key').drop(columns='_sort_key').reset_index(drop=True)
                return temp
            except Exception:
                pass
        return df