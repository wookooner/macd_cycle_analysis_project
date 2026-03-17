"""
feature_extract.py
카테고리 기반 사이클 특징 추출기

변경 이력 (v3.0):
  - 신규 특징 추가:
      start.cvd_rolling    - 사이클 시작 시점의 롤링 CVD (단기 매수/매도 압력)
      start.funding_rate   - 사이클 시작 시점의 펀딩비 (롱/숏 쏠림)
      aggregate.cvd        - 사이클 전체 volume_delta 합산 (누적 순매수 압력)
  - context_data 주입 구조 추가:
      funding_rate처럼 candle_data 외부 데이터가 필요한 특징을 위해
      extract_features_from_candle_data(candle_data, context_data=None) 인터페이스 추가.
      context_data는 {'funding_rate': float, ...} 형태의 dict.
      계산 함수가 context_data 파라미터를 가지면 자동으로 주입됨 (inspect 기반).
  - 성능 개선:
      calc_max_intraday_high_pct / calc_max_intraday_loss_pct 의 O(n²) 중첩 루프를
      numpy 누적 최대/최소(accumulate) 연산으로 O(n)으로 교체.
  - StructuredCycleProcessor 개선:
      process_and_enrich_cycles(funding_rate_path=None) 파라미터 추가.
      펀딩비 CSV를 로드하여 각 사이클 시작 시간의 가장 가까운 이전 펀딩비를 조회.
"""

import inspect
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
from pathlib import Path
import sys
import json

# 프로젝트 경로 설정: 상대/절대 실행을 모두 지원하도록 안전한 import 처리
try:
    # 패키지로 임포트할 때 (권장)
    from .config import DEFAULT_CONFIG
except Exception:
    # 스크립트로 직접 실행할 때 같은 디렉토리의 config.py를 사용하도록 경로 추가
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from config import DEFAULT_CONFIG

warnings.filterwarnings('ignore')


def _find_project_root() -> Path:
    """
    프로젝트 루트를 안정적으로 찾는 함수.

    Windows venv 환경에서 __file__이 'python.exe script.py' 형태의
    전체 명령줄 문자열이 되는 버그가 있다.
    이 경우 Path(__file__).resolve()는 잘못된 경로를 반환한다.

    해결 전략 (순서대로 시도):
      1차: __file__ 기반 (일반적인 환경)
      2차: sys.argv[0] 기반 — Windows venv에서 __file__이 오염됐을 때 대안.
           sys.argv[0]에는 스크립트 경로만 올바르게 담긴다.
      3차: 현재 작업 디렉토리(cwd)에서 상위로 탐색

    마커 조건: 프로젝트 루트에는 반드시 'data/base_data/'와 'feature_extract/'가 있다.
    """

    def _is_project_root(path: Path) -> bool:
        return (
            (path / "data" / "base_data").exists()
            and (path / "feature_extract").exists()
        )

    def _try_from_script_file(script_path_str: str) -> Optional[Path]:
        """스크립트 경로 문자열에서 3단계 상위가 프로젝트 루트인지 확인"""
        try:
            candidate = Path(script_path_str).resolve()
            # 이 파일 위치: project_root/feature_extract/macd_historgram_change_feature/feature_extract.py
            # 따라서 .parent 3번 = 프로젝트 루트
            for _ in range(3):
                candidate = candidate.parent
            if _is_project_root(candidate):
                return candidate
        except Exception:
            pass
        return None

    # 1차: __file__ 기반
    result = _try_from_script_file(__file__)
    if result:
        return result

    # 2차: sys.argv[0] 기반
    # Windows venv에서 __file__이 'python.exe script.py' 전체 문자열로 오염될 때,
    # sys.argv[0]에는 스크립트 경로만 정확하게 담겨 있다.
    if sys.argv:
        result = _try_from_script_file(sys.argv[0])
        if result:
            return result

    # 3차: cwd에서 상위로 탐색
    # IDE나 터미널에서 프로젝트 내 어느 위치에서든 실행하는 경우 커버
    for path in [Path.cwd()] + list(Path.cwd().parents):
        if _is_project_root(path):
            return path

    # 최후 수단: cwd 반환 후 경고
    print("\u26a0\ufe0f  프로젝트 루트를 자동으로 찾지 못했습니다. cwd를 사용합니다.")
    return Path.cwd()


# 모듈 레벨에서 한 번만 계산 — convert_all_timeframes() 등에서 참조
project_root = _find_project_root()


class CycleFeatureCalculator:
    """카테고리 기반 사이클 특징 계산기"""

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.name = "Categorized Cycle Feature Calculator"
        self.version = "3.0"

    def extract_features_from_candle_data(
        self,
        candle_data: List[Dict],
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict]:
        """
        캔들 데이터로부터 카테고리별 특징 추출.

        context_data는 candle_data 밖에서 조회해야 하는 값들을 담는 dict다.
        예: {'funding_rate': 0.0001}

        계산 함수(calc_*)의 시그니처에 'context_data' 파라미터가 있으면
        자동으로 context_data를 전달하고, 없으면 df만 전달한다.
        이 방식은 기존 함수를 수정하지 않고도 새 외부 데이터 특징을 추가할 수 있게 해준다.
        """
        if candle_data is None or len(candle_data) == 0:
            return self.config.get_default_cycle_features_structure()

        if context_data is None:
            context_data = {}

        # 캔들 데이터를 DataFrame으로 변환
        df = pd.DataFrame(candle_data)

        # 필수 컬럼 기본값 보정
        required_columns = ['open', 'high', 'low', 'close', 'volume',
                             'macd', 'macd_signal', 'macd_hist']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        if 'rsi' not in df.columns:
            df['rsi'] = 50.0

        # 신규 지표 컬럼 — 없으면 NaN으로 두어 calc 함수가 None 반환하도록 함
        # (backfill + indicator.py 실행 전 기존 parquet은 이 컬럼이 없을 수 있음)
        for col in ['volume_delta', 'cvd_rolling']:
            if col not in df.columns:
                df[col] = np.nan

        # 카테고리별 특징 계산
        features: Dict[str, Dict] = {}

        for category_name, category_data in self.config.FEATURE_CATEGORIES.items():
            features[category_name] = {}

            for feature_name, feature_config in category_data['features'].items():
                if not feature_config['enabled']:
                    continue

                try:
                    calculator_method = getattr(self, feature_config['calculator'])

                    # context_data가 필요한 함수인지 시그니처로 판단
                    sig = inspect.signature(calculator_method)
                    if 'context_data' in sig.parameters:
                        value = calculator_method(df, context_data=context_data)
                    else:
                        value = calculator_method(df)

                    # None은 NaN 대신 그대로 유지 (새 지표가 데이터 없을 때)
                    if value is None:
                        features[category_name][feature_name] = None
                        continue

                    # 데이터 타입 변환
                    if feature_config['data_type'] == 'int':
                        value = int(value) if pd.notna(value) else feature_config['default_value']
                    elif feature_config['data_type'] == 'float':
                        value = float(value) if pd.notna(value) else feature_config['default_value']

                    features[category_name][feature_name] = value

                except Exception as e:
                    print(f"특징 계산 오류 ({category_name}.{feature_name}): {e}")
                    features[category_name][feature_name] = feature_config['default_value']

        return features

    # =========================================================
    # SHAPE 카테고리
    # =========================================================

    def calc_duration_candles(self, df: pd.DataFrame) -> int:
        """사이클의 전체 길이 (캔들 수)"""
        return len(df)

    def calc_core_count(self, df: pd.DataFrame) -> int:
        """사이클 방향과 일치하는 핵심 캔들 수 (히스토그램 변화량 기준)"""
        if len(df) < 2:
            return 0
        hist_changes = df['macd_hist'].diff().dropna()
        if len(hist_changes) == 0:
            return 0
        overall_change = df['macd_hist'].iloc[-1] - df['macd_hist'].iloc[0]
        trend_direction = 1 if overall_change > 0 else -1
        return int((np.sign(hist_changes) == trend_direction).sum())

    def calc_noise_count(self, df: pd.DataFrame) -> int:
        """허용된 노이즈(반대 방향) 캔들 수"""
        duration = self.calc_duration_candles(df)
        core_count = self.calc_core_count(df)
        # -1: 첫 번째 diff가 NaN이므로 총 비교 가능한 캔들은 duration-1
        return max(0, duration - core_count - 1)

    def calc_direction_change(self, df: pd.DataFrame) -> int:
        """사이클 내에서 모멘텀 방향이 전환된 횟수"""
        if len(df) < 2:
            return 0
        hist_changes = df['macd_hist'].diff().dropna()
        if len(hist_changes) == 0:
            return 0
        directions = np.sign(hist_changes)
        return int((directions.diff() != 0).sum())

    def calc_peak_price_position(self, df: pd.DataFrame) -> float:
        """사이클 내 최고가(high) 캔들의 위치 비율 (0~1)"""
        if len(df) == 0:
            return 0.5
        peak_index = int(np.argmax(df['high'].values))
        if len(df) == 1:
            return 0.5
        return round(peak_index / (len(df) - 1), 4)

    def calc_trough_price_position(self, df: pd.DataFrame) -> float:
        """사이클 내 최저가(low) 캔들의 위치 비율 (0~1)"""
        if len(df) == 0:
            return 0.5
        trough_index = int(np.argmin(df['low'].values))
        if len(df) == 1:
            return 0.5
        return round(trough_index / (len(df) - 1), 4)

    # =========================================================
    # STRENGTH 카테고리
    # =========================================================

    def calc_direction_pct(self, df: pd.DataFrame) -> float:
        """핵심 캔들의 비율 (core_count / duration_candles * 100)"""
        duration = self.calc_duration_candles(df)
        if duration == 0:
            return 0.0
        return self.calc_core_count(df) / duration * 100

    def calc_hist_positive_ratio(self, df: pd.DataFrame) -> float:
        """MACD 히스토그램이 양수였던 캔들의 비율"""
        if len(df) == 0:
            return 0.0
        return (df['macd_hist'] > 0).sum() / len(df) * 100

    def calc_price_up_ratio(self, df: pd.DataFrame) -> float:
        """양봉 캔들의 비율"""
        if len(df) == 0:
            return 0.0
        return (df['close'] > df['open']).sum() / len(df) * 100

    def calc_price_down_ratio(self, df: pd.DataFrame) -> float:
        """음봉 캔들의 비율"""
        if len(df) == 0:
            return 0.0
        return (df['close'] < df['open']).sum() / len(df) * 100

    # =========================================================
    # START 카테고리
    # =========================================================

    def calc_start_price(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 종가"""
        return float(df['close'].iloc[0]) if len(df) > 0 else 0.0

    def calc_start_volume(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 거래량"""
        return float(df['volume'].iloc[0]) if len(df) > 0 else 0.0

    def calc_start_rsi(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 RSI 값"""
        return float(df['rsi'].iloc[0]) if len(df) > 0 else 50.0

    def calc_start_macd(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 MACD 값"""
        return float(df['macd'].iloc[0]) if len(df) > 0 else 0.0

    def calc_start_macd_signal(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 MACD Signal 값"""
        return float(df['macd_signal'].iloc[0]) if len(df) > 0 else 0.0

    def calc_start_hist(self, df: pd.DataFrame) -> float:
        """사이클 첫 캔들의 MACD Histogram 값 (핵심 진입 지표)"""
        return float(df['macd_hist'].iloc[0]) if len(df) > 0 else 0.0

    def calc_start_cvd_rolling(self, df: pd.DataFrame) -> Optional[float]:
        """
        사이클 첫 캔들의 롤링 CVD 값.

        cvd_rolling은 직전 N캔들의 volume_delta 합산으로, 현재 시장의
        단기 매수/매도 압력 균형을 보여준다. 값이 양수면 최근 매수 압력이
        우세하고, 음수면 매도 압력이 우세한 상태에서 사이클이 시작된 것이다.

        candle_data에 cvd_rolling 컬럼이 없으면 None을 반환한다.
        (backfill + indicator.py 실행 전 기존 parquet은 이 값이 없음)
        """
        if len(df) == 0:
            return None
        val = df['cvd_rolling'].iloc[0]
        return float(val) if pd.notna(val) else None

    def calc_start_funding_rate(
        self,
        df: pd.DataFrame,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """
        사이클 시작 시점의 펀딩비.

        펀딩비가 높으면(+) 롱 포지션이 과도하게 쏠려 있는 상태를 의미한다.
        이 상태에서 롱을 잡으면 추가 매수 여력이 제한될 수 있다.
        반대로 펀딩비가 낮거나 음수면 롱이 아직 쏠리지 않은 상태다.

        context_data는 StructuredCycleProcessor가 외부 CSV에서 조회한 값을
        {'funding_rate': float} 형태로 전달한다.
        """
        if context_data is None:
            return None
        val = context_data.get('funding_rate')
        return float(val) if val is not None and pd.notna(val) else None

    # =========================================================
    # END 카테고리
    # =========================================================

    def calc_end_price(self, df: pd.DataFrame) -> float:
        return float(df['close'].iloc[-1]) if len(df) > 0 else 0.0

    def calc_end_volume(self, df: pd.DataFrame) -> float:
        return float(df['volume'].iloc[-1]) if len(df) > 0 else 0.0

    def calc_end_rsi(self, df: pd.DataFrame) -> float:
        return float(df['rsi'].iloc[-1]) if len(df) > 0 else 50.0

    def calc_end_macd(self, df: pd.DataFrame) -> float:
        return float(df['macd'].iloc[-1]) if len(df) > 0 else 0.0

    def calc_end_macd_signal(self, df: pd.DataFrame) -> float:
        return float(df['macd_signal'].iloc[-1]) if len(df) > 0 else 0.0

    def calc_end_hist(self, df: pd.DataFrame) -> float:
        return float(df['macd_hist'].iloc[-1]) if len(df) > 0 else 0.0

    # =========================================================
    # CHANGE 카테고리
    # =========================================================

    def calc_price_change_pct(self, df: pd.DataFrame) -> float:
        """시작가 대비 종료가의 등락률 (분석 타깃 변수 Y)"""
        if len(df) == 0:
            return 0.0
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        if start_price == 0:
            return 0.0
        return ((end_price - start_price) / start_price) * 100

    def calc_rsi_change(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        return float(df['rsi'].iloc[-1] - df['rsi'].iloc[0])

    def calc_macd_change(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        return float(df['macd'].iloc[-1] - df['macd'].iloc[0])

    def calc_macd_signal_change(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        return float(df['macd_signal'].iloc[-1] - df['macd_signal'].iloc[0])

    def calc_macd_histogram_change(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        return float(df['macd_hist'].iloc[-1] - df['macd_hist'].iloc[0])

    # =========================================================
    # VOLATILITY 카테고리
    # =========================================================

    def calc_max_high_pct(self, df: pd.DataFrame) -> float:
        """시작가 대비 사이클 내 최고가 상승률"""
        if len(df) == 0:
            return 0.0
        start_price = df['close'].iloc[0]
        if start_price == 0:
            return 0.0
        return max(0.0, (df['high'].max() - start_price) / start_price * 100)

    def calc_max_loss_pct(self, df: pd.DataFrame) -> float:
        """시작가 대비 사이클 내 최저가 하락률"""
        if len(df) == 0:
            return 0.0
        start_price = df['close'].iloc[0]
        if start_price == 0:
            return 0.0
        return min(0.0, (df['low'].min() - start_price) / start_price * 100)

    def calc_max_intraday_high_pct(self, df: pd.DataFrame) -> float:
        """
        사이클 내 최대 잠재 수익률.

        원래 O(n²) 중첩 루프로 구현되어 있던 것을 numpy 누적 최대값(accumulate)으로
        O(n)으로 교체. 아이디어는 이렇다:
          각 캔들 i에서 그 이후 최고가(high)는 high[i+1:]의 최대값이다.
          이를 numpy로 한 번에 구하려면:
          1. high 배열을 뒤집어서 누적 최대값을 계산한다 (cummax from right).
          2. 다시 뒤집으면 future_max_high[i] = max(high[i:]) 가 된다.
          3. i 이후를 보려면 1칸 앞으로 이동 (shift)한다.
        """
        if len(df) < 2:
            return 0.0

        closes = df['close'].values
        highs = df['high'].values

        # future_max_high[i] = max(highs[i:])
        future_max_high = np.maximum.accumulate(highs[::-1])[::-1]
        # i 이후의 최대값이므로 1칸 shift (마지막은 NaN)
        future_max_high_after = np.empty(len(highs))
        future_max_high_after[:-1] = future_max_high[1:]
        future_max_high_after[-1] = np.nan

        with np.errstate(invalid='ignore', divide='ignore'):
            gains = np.where(
                (closes > 0) & ~np.isnan(future_max_high_after),
                (future_max_high_after - closes) / closes * 100,
                np.nan,
            )

        valid = gains[~np.isnan(gains)]
        return float(np.max(valid)) if len(valid) > 0 else 0.0

    def calc_max_intraday_loss_pct(self, df: pd.DataFrame) -> float:
        """
        사이클 내 최대 잠재 손실률.

        calc_max_intraday_high_pct와 동일한 O(n) 벡터화 방식.
        high 대신 low, max 대신 min을 사용한다.
        """
        if len(df) < 2:
            return 0.0

        closes = df['close'].values
        lows = df['low'].values

        # future_min_low[i] = min(lows[i:])
        future_min_low = np.minimum.accumulate(lows[::-1])[::-1]
        future_min_low_after = np.empty(len(lows))
        future_min_low_after[:-1] = future_min_low[1:]
        future_min_low_after[-1] = np.nan

        with np.errstate(invalid='ignore', divide='ignore'):
            losses = np.where(
                (closes > 0) & ~np.isnan(future_min_low_after),
                (future_min_low_after - closes) / closes * 100,
                np.nan,
            )

        valid = losses[~np.isnan(losses)]
        return float(np.min(valid)) if len(valid) > 0 else 0.0

    def calc_avg_true_range(self, df: pd.DataFrame) -> float:
        """사이클 내 캔들의 평균 ATR (통계 유의성 p=0.037)"""
        if len(df) == 0:
            return 0.0

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # TR = max(high-low, |high-prev_close|, |low-prev_close|)
        hl = highs - lows
        hpc = np.abs(np.diff(closes, prepend=closes[0]) - (closes - highs))  # 근사
        # 정확한 계산
        prev_closes = np.concatenate([[closes[0]], closes[:-1]])
        tr = np.maximum(hl, np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)))
        return float(np.mean(tr))

    def calc_price_change_deviation(self, df: pd.DataFrame) -> float:
        """사이클 내 캔들별 종가 변동률의 표준편차"""
        if len(df) < 2:
            return 0.0
        price_changes = df['close'].pct_change().dropna()
        if len(price_changes) == 0:
            return 0.0
        return float(price_changes.std() * 100)

    # =========================================================
    # AGGREGATE 카테고리
    # =========================================================

    def calc_all_volume(self, df: pd.DataFrame) -> float:
        """사이클 내 모든 캔들의 거래량 총합"""
        return float(df['volume'].sum()) if len(df) > 0 else 0.0

    def calc_aggregate_cvd(self, df: pd.DataFrame) -> Optional[float]:
        """
        사이클 전체의 volume_delta 합산 (누적 순매수 압력).

        volume_delta = 2 * taker_buy_base - volume.
        양수면 사이클 기간 동안 공격적 매수가 공격적 매도보다 많았다는 뜻이고,
        음수면 반대다.

        start.cvd_rolling이 "사이클 시작 시점의 단기 압력 상태"라면,
        aggregate.cvd는 "사이클이 진행되는 동안 얼마나 많은 매수/매도 압력이
        실제로 발생했는가"를 측정한다.

        candle_data에 volume_delta 컬럼이 없으면 None을 반환한다.
        """
        if len(df) == 0:
            return None
        if 'volume_delta' not in df.columns or df['volume_delta'].isna().all():
            return None
        total = df['volume_delta'].sum()
        return float(total) if pd.notna(total) else None

    # =========================================================
    # 레거시 변환 (하위 호환)
    # =========================================================

    def convert_legacy_features_to_categorized(
        self, legacy_features: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """기존 flat 구조의 특징들을 새로운 카테고리 구조로 변환"""
        categorized_features = self.config.get_default_cycle_features_structure()

        legacy_mapping = {
            'duration_candles':      ('shape',      'duration_candles'),
            'core_count':            ('shape',      'core_count'),
            'noise_count':           ('shape',      'noise_count'),
            'direction_change':      ('shape',      'direction_change'),
            'peak_price_position':   ('shape',      'peak_price_position'),
            'trough_price_position': ('shape',      'trough_price_position'),
            'direction_pct':         ('strength',   'direction_pct'),
            'hist_positive_ratio':   ('strength',   'hist_positive_ratio'),
            'price_up_ratio':        ('strength',   'price_up_ratio'),
            'price_down_ratio':      ('strength',   'price_down_ratio'),
            'start_price':           ('start',      'price'),
            'start_volume':          ('start',      'volume'),
            'start_rsi':             ('start',      'rsi'),
            'start_macd':            ('start',      'macd'),
            'start_macd_signal':     ('start',      'macd_signal'),
            'start_hist':            ('start',      'hist'),
            'end_price':             ('end',        'price'),
            'end_volume':            ('end',        'volume'),
            'end_rsi':               ('end',        'rsi'),
            'end_macd':              ('end',        'macd'),
            'end_macd_signal':       ('end',        'macd_signal'),
            'end_hist':              ('end',        'hist'),
            'price_change_pct':      ('change',     'price_pct'),
            'rsi_change':            ('change',     'rsi'),
            'macd_change':           ('change',     'macd'),
            'macd_signal_change':    ('change',     'macd_signal'),
            'macd_histogram_change': ('change',     'hist'),
            'max_high_pct':          ('volatility', 'max_high_pct'),
            'max_loss_pct':          ('volatility', 'max_loss_pct'),
            'max_high_change':       ('volatility', 'max_intraday_high_pct'),
            'max_loss_change':       ('volatility', 'max_intraday_loss_pct'),
            'avg_true_range':        ('volatility', 'avg_true_range'),
            'price_change_deviation':('volatility', 'price_change_deviation'),
            'all_volume':            ('aggregate',  'volume'),
        }

        for legacy_name, value in legacy_features.items():
            if legacy_name in legacy_mapping:
                category, feature = legacy_mapping[legacy_name]
                if category in categorized_features:
                    categorized_features[category][feature] = value

        return categorized_features


class StructuredCycleProcessor:
    """
    구조화된 사이클 처리기.

    parquet 파일을 읽어 각 사이클의 특징을 계산하고 저장한다.
    외부 데이터(펀딩비 CSV)를 선택적으로 받아 사이클 시작 시점의 컨텍스트 특징을 추가한다.
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.calculator = CycleFeatureCalculator()

    # ──────────────────────────────────────────────────────────────
    # 펀딩비 조회 유틸리티
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def load_funding_rate(funding_rate_path: Path) -> Optional[pd.DataFrame]:
        """
        펀딩비 CSV를 로드하고 타임스탬프 기준으로 정렬.

        CSV 컬럼: unix(Unix timestamp), date(datetime string), symbol, funding_rate
        또는: timestamp, funding_rate 형식도 지원.
        """
        try:
            df = pd.read_csv(funding_rate_path)
            if 'funding_rate' not in df.columns:
                print(f"⚠️  펀딩비 CSV에 'funding_rate' 컬럼이 없음: {funding_rate_path.name}")
                return None

            # timestamp 컬럼 확보: 'unix' 또는 'timestamp' 중 우선 순위 시도
            if 'unix' in df.columns:
                # unix (Unix timestamp in seconds) -> datetime 변환
                df['timestamp'] = pd.to_datetime(df['unix'], unit='s', utc=True)
            elif 'date' in df.columns:
                # date (datetime string) -> datetime 변환
                df['timestamp'] = pd.to_datetime(df['date'])
            elif 'timestamp' in df.columns:
                # timestamp (datetime string or numeric) -> datetime 변환
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                print(f"⚠️  펀딩비 CSV에 시간 정보 컬럼이 없음 (unix/date/timestamp): {funding_rate_path.name}")
                return None

            df = df.sort_values('timestamp').reset_index(drop=True)
            print(f"✅ 펀딩비 로드 완료: {len(df)}행 ({df['timestamp'].min()} ~ {df['timestamp'].max()})")
            return df
        except Exception as e:
            print(f"⚠️  펀딩비 CSV 로드 실패: {e}")
            return None

    @staticmethod
    def get_funding_rate_at(
        funding_rate_df: pd.DataFrame,
        start_date_str: str,
    ) -> Optional[float]:
        """
        사이클 시작 시점(start_date_str)보다 같거나 이전인 가장 최근 펀딩비를 반환.

        펀딩비는 8시간 간격(00:00, 08:00, 16:00 UTC)으로 발행된다.
        가장 최근 발행된 펀딩비가 현재 사이클 진입 시점에서 볼 수 있는 최신 값이다.
        """
        try:
            start_dt = pd.to_datetime(start_date_str)
            mask = funding_rate_df['timestamp'] <= start_dt
            if mask.any():
                return float(funding_rate_df.loc[mask, 'funding_rate'].iloc[-1])
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────────────────────────
    # 메인 처리 함수
    # ──────────────────────────────────────────────────────────────

    def process_and_enrich_cycles(
        self,
        output_path: Optional[Path] = None,
        funding_rate_path: Optional[Path] = None,
    ):
        """
        사이클 감지만 완료된 'raw' 파일을 받아, 각 사이클의 특징을 계산하고
        'enriched' 파일을 저장하는 후처리 함수.

        funding_rate_path: BTCUSDT_funding_rate.csv 경로.
          None이면 start.funding_rate 특징은 모두 None으로 저장된다.
        """
        print(f"🚀 특징 계산 및 보강 시작: {self.data_path.name}")

        try:
            df = pd.read_parquet(self.data_path)
            print(f"로드된 사이클: {len(df)}개")

            # 펀딩비 데이터 로드 (선택적)
            funding_rate_df = None
            if funding_rate_path is not None:
                funding_rate_df = self.load_funding_rate(Path(funding_rate_path))

            new_features_list = []

            for idx, row in df.iterrows():
                print(
                    f"특징 계산 진행: {idx+1}/{len(df)} ({(idx+1)/len(df)*100:.1f}%)",
                    end='\r',
                )

                candle_data = row['candle_data']

                # context_data 구성 — 현재는 funding_rate만 포함
                # 나중에 OI 등을 추가할 때 이 dict에 key를 추가하면 됨
                context_data: Dict[str, Any] = {}

                if funding_rate_df is not None:
                    fr = self.get_funding_rate_at(funding_rate_df, row['start_date'])
                    if fr is not None:
                        context_data['funding_rate'] = fr

                if isinstance(candle_data, (list, np.ndarray)) and len(candle_data) > 0:
                    enriched_features = self.calculator.extract_features_from_candle_data(
                        list(candle_data),
                        context_data=context_data,
                    )
                else:
                    enriched_features = self.calculator.config.get_default_cycle_features_structure()

                new_features_list.append(enriched_features)

            # PyArrow cannot write empty struct fields. Remove empty category dicts
            # (e.g., 'end': {}) to avoid "Cannot write struct type ... with no child field".
            def _prune_empty_categories(feat: Dict[str, Any]) -> Dict[str, Any]:
                return {k: v for k, v in feat.items() if not (isinstance(v, dict) and len(v) == 0)}

            pruned_features = [_prune_empty_categories(f) for f in new_features_list]
            df['cycle_features'] = pruned_features
            print(f"\n✅ 특징 계산 완료: {len(df)}개 사이클 보강")

            if output_path is None:
                output_name = self.data_path.name.replace('.parquet', '_enriched.parquet')
                output_path = self.data_path.with_name(output_name)

            df.to_parquet(output_path, index=False)
            print(f"💾 보강된 파일 저장 완료: {output_path.name}")

            self.validate_new_structure(output_path)
            return output_path, len(df)

        except Exception as e:
            print(f"\n❌ 특징 계산 실패: {e}")
            import traceback
            print(traceback.format_exc())
            return None, 0

    def convert_existing_cycles_to_new_structure(
        self, output_path: Optional[Path] = None
    ):
        """기존 사이클 데이터를 새로운 구조로 변환"""
        print(f"기존 사이클 데이터 구조 변환 시작: {self.data_path}")

        try:
            df = pd.read_parquet(self.data_path)
            print(f"로드된 사이클: {len(df)}개")

            converted_cycles = []

            for idx, row in df.iterrows():
                print(f"변환 진행: {idx+1}/{len(df)} ({(idx+1)/len(df)*100:.1f}%)", end='\r')

                candle_data = row['candle_data']
                if isinstance(candle_data, (list, np.ndarray)) and len(candle_data) > 0:
                    if isinstance(candle_data[0], dict):
                        candle_dict_list = list(candle_data)
                    else:
                        candle_dict_list = [
                            c if isinstance(c, dict) else {'close': float(c)}
                            for c in candle_data
                        ]
                    new_features = self.calculator.extract_features_from_candle_data(candle_dict_list)
                else:
                    legacy_features = row['cycle_features'] if isinstance(row['cycle_features'], dict) else {}
                    new_features = self.calculator.convert_legacy_features_to_categorized(legacy_features)

                converted_cycles.append({
                    'cycle_id':       row['cycle_id'],
                    'timeframe':      row['timeframe'],
                    'start_date':     row['start_date'],
                    'end_date':       row['end_date'],
                    'cycle_type':     row['cycle_type'],
                    'duration_candles': row['duration_candles'],
                    'category':       row['category'],
                    'algorithm_used': row['algorithm_used'],
                    'candle_data':    row['candle_data'],
                    'cycle_features': new_features,
                })

            print(f"\n변환 완료: {len(converted_cycles)}개 사이클")

            # Prune empty category dicts to avoid Parquet struct-without-fields error
            def _prune_cycle_dict(c: Dict[str, Any]) -> Dict[str, Any]:
                cf = c.get('cycle_features')
                if isinstance(cf, dict):
                    pruned = {k: v for k, v in cf.items() if not (isinstance(v, dict) and len(v) == 0)}
                    c['cycle_features'] = pruned
                return c

            converted_cycles = [_prune_cycle_dict(c) for c in converted_cycles]
            new_df = pd.DataFrame(converted_cycles)

            if output_path is None:
                output_path = self.data_path.with_name(f"converted_{self.data_path.name}")

            new_df.to_parquet(output_path, index=False)
            print(f"새로운 구조로 저장: {output_path}")
            return output_path, len(converted_cycles)

        except Exception as e:
            print(f"변환 실패: {e}")
            import traceback
            print(traceback.format_exc())
            return None, 0

    def validate_new_structure(self, converted_file: Path):
        """새로운 구조가 올바르게 변환되었는지 검증"""
        try:
            df = pd.read_parquet(converted_file)
            print(f"\n검증 시작: {len(df)}개 사이클")

            for i in range(min(3, len(df))):
                cycle = df.iloc[i]
                features = cycle['cycle_features']
                print(f"\n사이클 {i+1} 구조 검증:")
                print(f"  ID: {cycle['cycle_id']}")
                if isinstance(features, dict):
                    for category, feature_dict in features.items():
                        if isinstance(feature_dict, dict):
                            print(f"  {category}: {len(feature_dict)}개 특징")
                        else:
                            print(f"  {category}: 구조 오류")
                else:
                    print(f"  특징 구조 오류: {type(features)}")

            print("✅ 구조 검증 완료")
            return True

        except Exception as e:
            print(f"검증 실패: {e}")
            return False


# ──────────────────────────────────────────────────────────────────────
# 실행 함수들
# ──────────────────────────────────────────────────────────────────────

def convert_all_timeframes():
    """모든 타임프레임의 기존 데이터를 새로운 구조로 변환"""
    print("🔄 전체 타임프레임 데이터 구조 변환 시작")
    print("=" * 60)

    structured_path = project_root.parent / "data" / "cycle_data" / "structured"
    if not structured_path.exists():
        print(f"❌ 디렉토리 없음: {structured_path}")
        return

    parquet_files = [
        p for p in structured_path.glob("cycles_*.parquet")
        if "_v2" not in p.name and "converted_" not in p.name
    ]

    if not parquet_files:
        print("❌ 변환할 parquet 파일 없음")
        return

    print(f"📁 발견된 파일: {len(parquet_files)}개")
    conversion_results = {}

    for file_path in parquet_files:
        print(f"\n🔄 변환 중: {file_path.name}")
        try:
            processor = StructuredCycleProcessor(file_path)
            converted_path, cycle_count = processor.convert_existing_cycles_to_new_structure()

            if converted_path and cycle_count > 0:
                success = processor.validate_new_structure(converted_path)
                conversion_results[file_path.name] = {
                    'converted_path': converted_path,
                    'cycle_count': cycle_count,
                    'status': 'success' if success else 'validation_failed',
                }
            else:
                conversion_results[file_path.name] = {'status': 'conversion_failed'}

        except Exception as e:
            print(f"❌ {file_path.name} 변환 실패: {e}")
            conversion_results[file_path.name] = {'status': 'error', 'error': str(e)}

    _print_conversion_summary(conversion_results)
    return conversion_results


def process_all_timeframes_for_enrichment(funding_rate_path: Optional[str] = None):
    """
    모든 타임프레임의 감지된 데이터를 특징 계산으로 보강.

    funding_rate_path: BTCUSDT_funding_rate.csv 경로 (str 또는 None).
      예) process_all_timeframes_for_enrichment(
              funding_rate_path='data/base_data/BTCUSDT_funding_rate.csv'
          )
    """
    print("🔄 전체 타임프레임 특징 계산(보강) 시작")
    print("=" * 60)

    _project_root = _find_project_root()
    structured_path = _project_root / "data" / "cycle_data" / "structured"
    print(f"\U0001f4c2 프로젝트 루트: {_project_root}")

    # 펀딩비 경로 자동 탐색: 직접 지정이 없으면 base_data 폴더에서 찾음
    if funding_rate_path is None:
        default_fr_path = _project_root / "data" / "base_data" / "BTCUSDT_funding_rate.csv"
        if default_fr_path.exists():
            funding_rate_path = str(default_fr_path)
            print(f"\U0001f4b0 펀딩비 CSV 자동 탐색 성공: {default_fr_path.name}")
        else:
            print(f"\u26a0\ufe0f  펀딩비 CSV 없음 ({default_fr_path.name}) \u2192 start.funding_rate는 None으로 저장됨")
    else:
        print(f"\U0001f4b0 펀딩비 CSV 지정됨: {funding_rate_path}")

    raw_files = [
        p for p in structured_path.glob("cycles_*.parquet")
        if "_enriched" not in p.name and "converted_" not in p.name
    ]

    if not raw_files:
        print("❌ 특징을 계산할 raw 사이클 파일 없음. macd_histogram_change_detect.py 먼저 실행 필요")
        return

    print(f"📁 처리 대상 파일: {len(raw_files)}개")
    if funding_rate_path:
        print(f"💰 펀딩비 CSV: {funding_rate_path}")
    else:
        print("⚠️  펀딩비 CSV 미지정 → start.funding_rate 특징은 None으로 저장됨")

    for file_path in raw_files:
        processor = StructuredCycleProcessor(file_path)
        processor.process_and_enrich_cycles(funding_rate_path=funding_rate_path)


def _print_conversion_summary(conversion_results: Dict):
    print("\n" + "=" * 60)
    print("🎉 변환 결과 요약")
    print("=" * 60)
    successful = [k for k, v in conversion_results.items() if v.get('status') == 'success']
    failed = [k for k, v in conversion_results.items() if v.get('status') != 'success']

    print(f"✅ 성공: {len(successful)}개")
    for name in successful:
        r = conversion_results[name]
        print(f"   {name} → {r['converted_path'].name} ({r['cycle_count']}개 사이클)")

    if failed:
        print(f"\n❌ 실패: {len(failed)}개")
        for name in failed:
            r = conversion_results[name]
            print(f"   {name}: {r.get('status', 'unknown')}")
    print("=" * 60)


def test_new_structure():
    """새로운 구조 테스트 (CVD, 펀딩비 포함)"""
    print("🧪 새로운 특징 구조 테스트")

    # 샘플 캔들 데이터 (volume_delta, cvd_rolling 포함)
    sample_candles = [
        {
            'timestamp': '2024-01-01 00:00:00',
            'open': 42000, 'high': 42500, 'low': 41800, 'close': 42150,
            'volume': 120.5, 'macd': 235.67, 'macd_signal': 280.90,
            'macd_hist': -45.23, 'rsi': 55.2,
            'volume_delta': -150.3, 'cvd_rolling': -320.5,
        },
        {
            'timestamp': '2024-01-01 04:00:00',
            'open': 42150, 'high': 43200, 'low': 42000, 'close': 43000,
            'volume': 150.2, 'macd': 400.89, 'macd_signal': 350.15,
            'macd_hist': 50.74, 'rsi': 62.8,
            'volume_delta': 280.7, 'cvd_rolling': 120.4,
        },
        {
            'timestamp': '2024-01-01 08:00:00',
            'open': 43000, 'high': 44000, 'low': 42800, 'close': 43890,
            'volume': 180.8, 'macd': 445.89, 'macd_signal': 390.15,
            'macd_hist': 55.74, 'rsi': 65.8,
            'volume_delta': 350.2, 'cvd_rolling': 450.6,
        },
    ]

    # 펀딩비 컨텍스트
    context_data = {'funding_rate': 0.0001}

    calculator = CycleFeatureCalculator()
    features = calculator.extract_features_from_candle_data(sample_candles, context_data=context_data)

    print("\n계산된 특징 구조:")
    print(json.dumps(features, indent=2, ensure_ascii=False, default=str))

    # 신규 특징 확인
    print("\n=== 신규 특징 검증 ===")
    print(f"start.cvd_rolling   : {features.get('start', {}).get('cvd_rolling')}")
    print(f"start.funding_rate  : {features.get('start', {}).get('funding_rate')}")
    print(f"aggregate.cvd       : {features.get('aggregate', {}).get('cvd')}")

    return features


if __name__ == "__main__":
    print("🚀 카테고리 기반 특징 추출기 시작")

    print("\n선택하세요:")
    print("1: 새로운 구조 테스트 (CVD, 펀딩비 포함)")
    print("2: 기존 데이터 구조 변환")
    print("3: 감지된 사이클 파일에 특징 계산 및 저장")
    print("4: 설정 관리")

    choice = input("선택 (1-4): ").strip()

    if choice == "1":
        test_new_structure()
    elif choice == "2":
        convert_all_timeframes()
    elif choice == "3":
        fr_path = input("펀딩비 CSV 경로 (없으면 Enter): ").strip() or None
        process_all_timeframes_for_enrichment(funding_rate_path=fr_path)
    elif choice == "4":
        try:
            from .config import main as config_main
        except ImportError:
            from config import main as config_main
        config_main()
    else:
        print("잘못된 선택입니다.")