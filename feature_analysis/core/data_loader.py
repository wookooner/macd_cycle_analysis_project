"""
통합 데이터 로더 (Unified Data Loader)
======================================
모든 분석 모듈의 데이터 접근 관문.
parquet 로딩 → cycle_features 평탄화 → features_config 기반 필터링 →
복합 카테고리 생성을 한 곳에서 처리.

사용법:
    loader = CycleDataLoader(data_dir="data/cycle_data/structured",
                             config_path="feature_extract/macd_historgram_change_feature/features_config_v2.json")
    df = loader.load("4h")                 # enriched 우선 → fallback to 기본
    df = loader.load("4h", enriched=False)  # 기본 parquet 강제
    
    # enabled 피처만 가져오기
    enabled_features = loader.get_enabled_features()
    
    # 특정 카테고리의 피처만
    start_features = loader.get_features_by_category("start")
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class CycleDataLoader:
    """
    사이클 데이터 로딩 및 전처리 통합 클래스.
    
    핵심 원칙:
    - 모든 분석 모듈은 이 클래스를 통해서만 데이터에 접근
    - iterrows() 대신 벡터화 연산으로 평탄화
    - features_config_v2.json의 enabled/disabled 설정을 자동 반영
    - enriched parquet이 있으면 우선 사용, 없으면 기본 parquet fallback
    """
    
    # 피처 컬럼 명명 규칙: {category}_{feature_name} (예: start_hist, shape_duration_candles)
    META_COLUMNS = [
        'cycle_id', 'timeframe', 'start_date', 'end_date',
        'cycle_type', 'duration_candles', 'category', 'algorithm_used'
    ]
    
    DERIVED_COLUMNS = ['start_datetime', 'end_datetime', 'composite_category']
    
    def __init__(self, data_dir: str = "data/cycle_data/structured",
                 config_path: str = "feature_extract/macd_historgram_change_feature/features_config_v2.json"):
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self._config_cache: Optional[Dict] = None
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
    # =========================================================
    # 공개 API
    # =========================================================
    
    def load(self, timeframe: str, enriched: bool = True,
             only_enabled: bool = False, force_reload: bool = False) -> pd.DataFrame:
        """
        사이클 데이터를 로드하고 평탄화된 DataFrame을 반환.
        
        Args:
            timeframe: '1h', '4h', '1d', '1w', '1m'
            enriched: True이면 enriched 파일 우선 사용
            only_enabled: True이면 features_config에서 enabled=true인 피처만 포함
            force_reload: True이면 캐시 무시하고 재로딩
            
        Returns:
            평탄화된 DataFrame (메타 컬럼 + 피처 컬럼 + 파생 컬럼)
        """
        cache_key = f"{timeframe}_{'enr' if enriched else 'raw'}_{'en' if only_enabled else 'all'}"
        
        if not force_reload and cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        # 1) parquet 파일 경로 결정
        raw_df = self._load_parquet(timeframe, enriched)
        
        # 2) 평탄화 (cycle_features dict → 개별 컬럼)
        df = self._flatten_features(raw_df)
        
        # 3) 날짜 변환
        df = self._convert_datetimes(df)
        
        # 4) 복합 카테고리 생성 (UP_PROFIT, UP_LOSS, DOWN_PROFIT, DOWN_LOSS)
        df = self._create_composite_category(df)
        
        # 5) enabled 필터링 (옵션)
        if only_enabled:
            df = self._filter_enabled_features(df)
        
        self._data_cache[cache_key] = df
        return df
    
    def get_config(self) -> Dict:
        """features_config_v2.json 전체를 딕셔너리로 반환."""
        if self._config_cache is None:
            self._config_cache = self._load_config()
        return self._config_cache
    
    def get_enabled_features(self) -> List[str]:
        """enabled=true인 피처의 평탄화된 컬럼명 리스트를 반환."""
        config = self.get_config()
        enabled = []
        for cat_name, cat_info in config.get("feature_categories", {}).items():
            for feat_name, feat_info in cat_info.get("features", {}).items():
                if feat_info.get("enabled", False):
                    enabled.append(f"{cat_name}_{feat_name}")
        return enabled
    
    def get_disabled_features(self) -> List[str]:
        """enabled=false인 피처의 평탄화된 컬럼명 리스트를 반환."""
        config = self.get_config()
        disabled = []
        for cat_name, cat_info in config.get("feature_categories", {}).items():
            for feat_name, feat_info in cat_info.get("features", {}).items():
                if not feat_info.get("enabled", False):
                    disabled.append(f"{cat_name}_{feat_name}")
        return disabled
    
    def get_features_by_category(self, category: str, only_enabled: bool = False) -> List[str]:
        """특정 카테고리의 피처명 리스트를 반환."""
        config = self.get_config()
        cat_info = config.get("feature_categories", {}).get(category, {})
        features = []
        for feat_name, feat_info in cat_info.get("features", {}).items():
            if only_enabled and not feat_info.get("enabled", False):
                continue
            features.append(f"{category}_{feat_name}")
        return features
    
    def get_feature_info(self, flat_name: str) -> Optional[Dict]:
        """
        평탄화된 피처명(예: 'start_hist')으로 config 정보를 조회.
        반환값: {description, enabled, data_type, category, original_name, ...}
        """
        config = self.get_config()
        # flat_name = "start_hist" → category="start", feature="hist"
        # 단, "shape_duration_candles" → category="shape", feature="duration_candles"
        for cat_name, cat_info in config.get("feature_categories", {}).items():
            for feat_name, feat_info in cat_info.get("features", {}).items():
                if f"{cat_name}_{feat_name}" == flat_name:
                    return {
                        **feat_info,
                        "category": cat_name,
                        "original_name": feat_name,
                        "flat_name": flat_name
                    }
        return None
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """DataFrame에서 메타/파생 컬럼을 제외한 순수 피처 컬럼만 반환."""
        exclude = set(self.META_COLUMNS + self.DERIVED_COLUMNS + ['candle_data', 'cycle_features'])
        return [c for c in df.columns if c not in exclude]
    
    def get_numeric_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """DataFrame에서 수치형 피처 컬럼만 반환."""
        feature_cols = self.get_feature_columns(df)
        return [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    def get_predictive_features(self) -> List[str]:
        """
        사이클 시작 시점에 알 수 있는 피처만 반환.
        (start 카테고리 + shape의 일부)
        진입 결정에 사용 가능한 피처.
        """
        config = self.get_config()
        predictive = []
        
        # start 카테고리: 모두 진입 시점에 사용 가능
        for feat_name, feat_info in config.get("feature_categories", {}).get("start", {}).get("features", {}).items():
            if feat_info.get("enabled", False):
                predictive.append(f"start_{feat_name}")
        
        return predictive
    
    def get_target_variable(self) -> str:
        """분석 타깃 변수명 반환 (change_price_pct)."""
        return "change_price_pct"
    
    def available_timeframes(self) -> List[str]:
        """사용 가능한 타임프레임 목록 반환."""
        available = []
        for pattern in ["cycles_*.parquet", "cycles_*_enriched.parquet"]:
            for f in self.data_dir.glob(pattern):
                # cycles_4h.parquet → '4h', cycles_4h_enriched.parquet → '4h'
                name = f.stem.replace("cycles_", "").replace("_enriched", "")
                if name not in available:
                    available.append(name)
        # 계층 순서로 정렬
        order = {'1w': 0, '1d': 1, '4h': 2, '1h': 3, '1m': 4}
        return sorted(available, key=lambda x: order.get(x, 99))
    
    # =========================================================
    # 내부 구현
    # =========================================================
    
    def _load_parquet(self, timeframe: str, enriched: bool) -> pd.DataFrame:
        """parquet 파일 로딩. enriched 우선, 없으면 기본 파일 fallback."""
        enriched_path = self.data_dir / f"cycles_{timeframe}_enriched.parquet"
        basic_path = self.data_dir / f"cycles_{timeframe}.parquet"
        
        if enriched and enriched_path.exists():
            path = enriched_path
        elif basic_path.exists():
            path = basic_path
        else:
            raise FileNotFoundError(
                f"사이클 데이터 파일을 찾을 수 없습니다: {basic_path} 또는 {enriched_path}"
            )
        
        df = pd.read_parquet(path)
        return df
    
    def _flatten_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        cycle_features 딕셔너리를 개별 컬럼으로 평탄화.
        pd.json_normalize 활용으로 iterrows() 대비 수십 배 빠름.
        """
        if 'cycle_features' not in df.columns:
            return df
        
        # cycle_features가 dict인 행만 처리
        features_series = df['cycle_features']
        
        # None/NaN인 행은 빈 딕셔너리로 대체
        features_list = []
        for feat in features_series:
            if isinstance(feat, dict):
                flat = {}
                for cat_name, cat_dict in feat.items():
                    if isinstance(cat_dict, dict):
                        for feat_name, value in cat_dict.items():
                            flat[f"{cat_name}_{feat_name}"] = value
                features_list.append(flat)
            else:
                features_list.append({})
        
        features_df = pd.DataFrame(features_list, index=df.index)
        
        # 메타 컬럼과 합치기 (candle_data, cycle_features 제외)
        meta_cols = [c for c in df.columns if c not in ('candle_data', 'cycle_features')]
        result = pd.concat([df[meta_cols].reset_index(drop=True),
                           features_df.reset_index(drop=True)], axis=1)
        
        return result
    
    def _convert_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """start_date, end_date를 datetime으로 변환."""
        for col, new_col in [('start_date', 'start_datetime'), ('end_date', 'end_datetime')]:
            if col not in df.columns:
                continue
            try:
                if df[col].dtype in ['int64', 'float64']:
                    df[new_col] = pd.to_datetime(df[col], unit='s')
                else:
                    df[new_col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                df[new_col] = pd.NaT
        return df
    
    def _create_composite_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        4분류 복합 카테고리 생성:
        UP_PROFIT, UP_LOSS, DOWN_PROFIT, DOWN_LOSS
        
        UP 사이클에서 가격 상승 = UP_PROFIT (정상)
        UP 사이클에서 가격 하락 = UP_LOSS (배신)
        DOWN 사이클에서 가격 하락 = DOWN_PROFIT (정상 — short 관점 수익)
        DOWN 사이클에서 가격 상승 = DOWN_LOSS (반전)
        """
        if 'change_price_pct' not in df.columns or 'cycle_type' not in df.columns:
            return df
        
        cycle_upper = df['cycle_type'].str.upper()
        price_positive = df['change_price_pct'] > 0
        
        conditions = [
            (cycle_upper == 'UP') & price_positive,
            (cycle_upper == 'UP') & ~price_positive,
            (cycle_upper == 'DOWN') & ~price_positive,
            (cycle_upper == 'DOWN') & price_positive,
        ]
        choices = ['UP_PROFIT', 'UP_LOSS', 'DOWN_PROFIT', 'DOWN_LOSS']
        
        df['composite_category'] = np.select(conditions, choices, default='UNKNOWN')
        return df
    
    def _filter_enabled_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """enabled=true인 피처 컬럼만 남기고 나머지 피처 컬럼 제거."""
        enabled = set(self.get_enabled_features())
        feature_cols = self.get_feature_columns(df)
        
        # 메타+파생 컬럼은 항상 유지, 피처 컬럼 중 enabled인 것만 유지
        keep_cols = [c for c in df.columns if c not in feature_cols or c in enabled]
        return df[keep_cols]
    
    def _load_config(self) -> Dict:
        """features_config_v2.json 로딩."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # config 경로를 못 찾으면 여러 후보 경로 시도
        candidates = [
            Path("features_config_v2.json"),
            Path("feature_config.json"),
        ]
        for candidate in candidates:
            if candidate.exists():
                with open(candidate, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # 최종 fallback: 빈 config
        print("⚠️ features_config_v2.json을 찾을 수 없습니다. 빈 설정으로 진행합니다.")
        return {"feature_categories": {}}


# ==============================================================================
# 편의 함수: 모듈 레벨에서 바로 사용 가능
# ==============================================================================

def quick_load(timeframe: str = "4h", **kwargs) -> Tuple[pd.DataFrame, CycleDataLoader]:
    """
    빠른 데이터 로딩 편의 함수.
    
    Returns:
        (DataFrame, loader) 튜플
        
    사용법:
        df, loader = quick_load("4h")
        df, loader = quick_load("1d", only_enabled=True)
    """
    loader = CycleDataLoader()
    df = loader.load(timeframe, **kwargs)
    return df, loader