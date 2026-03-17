"""
피처 프로파일러 (Feature Profiler)
==================================
개별 피처의 종합 프로파일 + UP vs DOWN 비교 분석.
효과크기(Cliff's delta)를 포함하여 "진짜 유용한 예측 변수"를 식별.

사용법:
    from feature_analysis.core.data_loader import CycleDataLoader
    from feature_analysis.analyzers.feature_profiler import FeatureProfiler
    
    loader = CycleDataLoader()
    df = loader.load("4h")
    profiler = FeatureProfiler(df, loader)
    
    # 단일 피처 프로파일
    result = profiler.profile_feature("start_hist")
    
    # 전체 피처 일괄 비교 (UP vs DOWN) — 효과크기 순 정렬
    ranking = profiler.rank_features_by_effect_size()
    
    # 4분류 비교
    result = profiler.compare_4way("start_hist")
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from feature_analysis.core.stat_engine import StatEngine
from feature_analysis.core.data_loader import CycleDataLoader


class FeatureProfiler:
    """개별 피처의 분포 분석 및 그룹 간 비교."""
    
    def __init__(self, df: pd.DataFrame, loader: CycleDataLoader):
        self.df = df
        self.loader = loader
    
    # =========================================================
    # 1. 단일 피처 종합 프로파일
    # =========================================================
    
    def profile_feature(self, feature: str) -> Dict[str, Any]:
        """
        하나의 피처에 대한 종합 프로파일.
        
        포함:
        - 전체 분포 프로파일 (기술통계, 정규성, 이상치, 형태)
        - UP vs DOWN 비교 (검정 + 효과크기)
        - 4분류 비교 (UP_PROFIT, UP_LOSS, DOWN_PROFIT, DOWN_LOSS)
        - 피처 메타 정보 (config에서 가져옴)
        """
        if feature not in self.df.columns:
            return {'error': f"피처 '{feature}'를 DataFrame에서 찾을 수 없습니다.",
                    'available': self.loader.get_feature_columns(self.df)[:20]}
        
        data = self.df[feature].dropna().values
        
        if len(data) < 10:
            return {'error': f"유효 데이터 부족 (n={len(data)})", 'feature': feature}
        
        # 메타 정보
        meta = self.loader.get_feature_info(feature)
        
        # 전체 분포
        overall = StatEngine.distribution_profile(data, name=feature)
        
        # UP vs DOWN 비교
        up_down = self._compare_up_down(feature)
        
        # 4분류 비교
        four_way = self._compare_4way(feature)
        
        return {
            'feature': feature,
            'meta': meta,
            'overall': overall,
            'up_vs_down': up_down,
            'four_way': four_way
        }
    
    # =========================================================
    # 2. UP vs DOWN 비교
    # =========================================================
    
    def compare_up_down(self, feature: str) -> Dict[str, Any]:
        """단일 피처의 UP vs DOWN 비교."""
        return self._compare_up_down(feature)
    
    def _compare_up_down(self, feature: str) -> Dict[str, Any]:
        """UP vs DOWN 내부 구현."""
        if feature not in self.df.columns or 'cycle_type' not in self.df.columns:
            return {'error': 'Required columns missing'}
        
        valid = self.df[[feature, 'cycle_type']].dropna()
        
        up_data = valid[valid['cycle_type'].str.upper() == 'UP'][feature].values
        down_data = valid[valid['cycle_type'].str.upper() == 'DOWN'][feature].values
        
        if len(up_data) < 2 or len(down_data) < 2:
            return {'error': 'Insufficient data in groups'}
        
        return StatEngine.compare_groups(up_data, down_data, label_a="UP", label_b="DOWN")
    
    # =========================================================
    # 3. 4분류 비교
    # =========================================================
    
    def compare_4way(self, feature: str) -> Dict[str, Any]:
        """4분류(UP_PROFIT, UP_LOSS, DOWN_PROFIT, DOWN_LOSS) 비교."""
        return self._compare_4way(feature)
    
    def _compare_4way(self, feature: str) -> Dict[str, Any]:
        """4분류 비교 내부 구현."""
        if feature not in self.df.columns or 'composite_category' not in self.df.columns:
            return {'error': 'Required columns missing'}
        
        valid = self.df[[feature, 'composite_category']].dropna()
        
        groups = {}
        for cat in ['UP_PROFIT', 'UP_LOSS', 'DOWN_PROFIT', 'DOWN_LOSS']:
            data = valid[valid['composite_category'] == cat][feature].values
            if len(data) >= 2:
                groups[cat] = data
        
        if len(groups) < 2:
            return {'error': 'Insufficient groups with data'}
        
        return StatEngine.compare_multiple_groups(groups)
    
    # =========================================================
    # 4. 전체 피처 랭킹 (효과크기 순)
    # =========================================================
    
    def rank_features_by_effect_size(self, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        모든 (또는 지정된) 피처를 UP vs DOWN 효과크기 순으로 정렬.
        
        이것이 "어떤 피처가 진짜로 UP/DOWN 구분에 유용한가?"에 대한 답.
        
        Returns:
            DataFrame: feature, cliffs_delta, magnitude, p_value, 
                       mean_up, mean_down, interpretation 컬럼
        """
        if features is None:
            features = self.loader.get_numeric_feature_columns(self.df)
        
        records = []
        for feat in features:
            result = self._compare_up_down(feat)
            if 'error' in result:
                continue
            
            cd = result['effect_sizes']['cliffs_delta']
            desc_up = result['descriptive'].get('UP', {})
            desc_down = result['descriptive'].get('DOWN', {})
            
            records.append({
                'feature': feat,
                'cliffs_delta': cd['value'],
                'abs_cliffs_delta': abs(cd['value']),
                'magnitude': cd['magnitude'],
                'p_value': result['tests']['mann_whitney']['p_value'],
                'significant': result['tests']['mann_whitney']['significant'],
                'cohens_d': result['effect_sizes']['cohens_d']['value'],
                'mean_up': desc_up.get('mean', np.nan),
                'mean_down': desc_down.get('mean', np.nan),
                'median_up': desc_up.get('median', np.nan),
                'median_down': desc_down.get('median', np.nan),
                'interpretation': result['interpretation']
            })
        
        if not records:
            return pd.DataFrame()
        
        ranking = pd.DataFrame(records)
        ranking = ranking.sort_values('abs_cliffs_delta', ascending=False)
        ranking = ranking.reset_index(drop=True)
        ranking.index = ranking.index + 1  # 1부터 시작하는 순위
        ranking.index.name = 'rank'
        
        return ranking
    
    # =========================================================
    # 5. 피처 카테고리별 요약
    # =========================================================
    
    def summarize_by_category(self) -> Dict[str, Any]:
        """
        features_config의 카테고리별로 피처 유용성을 요약.
        
        Returns:
            카테고리별 {category: {features: [...], avg_effect_size: ..., best_feature: ...}}
        """
        config = self.loader.get_config()
        categories = config.get("feature_categories", {})
        
        result = {}
        for cat_name, cat_info in categories.items():
            cat_features = self.loader.get_features_by_category(cat_name)
            # DataFrame에 실제 존재하는 피처만
            existing = [f for f in cat_features if f in self.df.columns 
                       and pd.api.types.is_numeric_dtype(self.df[f])]
            
            if not existing:
                continue
            
            ranking = self.rank_features_by_effect_size(existing)
            
            if ranking.empty:
                continue
            
            result[cat_name] = {
                'description': cat_info.get('description', ''),
                'feature_count': len(existing),
                'features': ranking.to_dict('records'),
                'avg_abs_effect': ranking['abs_cliffs_delta'].mean(),
                'best_feature': ranking.iloc[0]['feature'] if len(ranking) > 0 else None,
                'significant_count': ranking['significant'].sum()
            }
        
        return result