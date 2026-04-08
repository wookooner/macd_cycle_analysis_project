"""
gold_collector.py
=================
API Ninjas /v1/commoditypricehistorical 를 사용한 금 선물 OHLCV 수집기.

API Ninjas 제약 및 처리 방식:
  - API 직접 지원 타임프레임: 1h / 4h / 1d
  - 1w / 1m(월봉): API 미지원 → 1d 데이터를 resample 하여 자동 생성
      resample 규칙 (표준 OHLCV):
        open   = 주/월의 첫 번째 1d 캔들의 open
        high   = 주/월 내 최고가
        low    = 주/월 내 최저가
        close  = 주/월의 마지막 1d 캔들의 close
        volume = 주/월 내 volume 합산
      1w 기준: ISO 월요일 시작 (freq='W-MON')
      1m 기준: 월 시작일 (freq='MS')
  - taker_buy_base 없음 → volume_delta / CVD 계산 불가
  - 응답이 내림차순(최신→과거) → 저장 전 오름차순 반전
  - 요청 1회당 최대 1000 rows (start/end 슬라이싱으로 분할 수집)
  - API 키: X-Api-Key 헤더

저장 위치:
  data/base_data/GOLD_1h.csv   ← API 직접 수집
  data/base_data/GOLD_4h.csv   ← API 직접 수집
  data/base_data/GOLD_1d.csv   ← API 직접 수집
  data/base_data/GOLD_1w.csv   ← GOLD_1d.csv 에서 resample
  data/base_data/GOLD_1m.csv   ← GOLD_1d.csv 에서 resample

컬럼 구조 (BTC CSV와 동일하게 맞춤, 없는 항목은 NA):
  unix, date, symbol, open, high, low, close, Volume USD,
  volume, taker_buy_base, volume_delta,
  macd, macd_signal, macd_hist, rsi
"""

import os
import time
import shutil
import logging
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.common.paths import PROJECT_PATHS

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR        = PROJECT_PATHS.project_root
DATA_DIR        = PROJECT_PATHS.legacy_data_root
RAW_DATA_DIR    = PROJECT_PATHS.base_data_dir
BACKUP_DATA_DIR = PROJECT_PATHS.backup_data_dir

# ── API 설정 ───────────────────────────────────────────────────────────────────
API_NINJAS_BASE_URL = "https://api.api-ninjas.com/v1"
COMMODITY_NAME      = "gold"        # API Ninjas name 파라미터값
SYMBOL_LABEL        = "GOLD"        # CSV에 저장할 심볼 이름

# API에서 직접 수집하는 타임프레임
GOLD_API_INTERVALS = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# 1d resample로 생성하는 타임프레임
# freq: pandas resample 주기 문자열
#   W-MON : 월요일 시작 주봉 (ISO 주간 기준)
#   MS    : 월 시작일 기준 월봉
GOLD_RESAMPLE_INTERVALS = {
    "1w": {"freq": "W-MON", "label": "주봉 (1d resample)"},
    "1m": {"freq": "MS",    "label": "월봉 (1d resample)"},
}

# 전체 지원 타임프레임 (API + resample 포함)
GOLD_ALL_TIMEFRAMES = list(GOLD_API_INTERVALS.keys()) + list(GOLD_RESAMPLE_INTERVALS.keys())

GOLD_DATA_FILES = {
    "1h": "GOLD_1h.csv",
    "4h": "GOLD_4h.csv",
    "1d": "GOLD_1d.csv",
    "1w": "GOLD_1w.csv",
    "1m": "GOLD_1m.csv",
}

# 히스토리 시작일 (API 직접 수집 타임프레임만 해당)
GOLD_HISTORY_START = {
    "1h": datetime(2020, 1, 1, tzinfo=timezone.utc),
    "4h": datetime(2018, 1, 1, tzinfo=timezone.utc),
    "1d": datetime(2010, 1, 1, tzinfo=timezone.utc),
}

# 요청당 최대 rows
MAX_ROWS_PER_REQUEST = 1000

# 타임프레임별 1 row의 초 단위 길이 (슬라이싱 윈도우 계산용)
INTERVAL_SECONDS = {
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# API 요청 간 딜레이 (rate limit 대응)
REQUEST_DELAY = 0.3  # seconds

# 반올림 설정
DECIMAL_PLACES = {
    "price":  2,
    "volume": 2,
}


def _ensure_directories():
    for d in [DATA_DIR, RAW_DATA_DIR, BACKUP_DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("gold_collector")


class GoldDataCollector:
    """
    API Ninjas 기반 금 선물 OHLCV 수집기.
    BTC 수집기(AdvancedBTCDataCollectorV2)와 퍼블릭 인터페이스를 맞춰
    파이프라인에서 동일하게 호출할 수 있도록 설계.
    """

    def __init__(self, api_key: str = None):
        self.logger = _setup_logging()
        _ensure_directories()

        # API 키: 인자 > 환경변수 순으로 읽기
        self.api_key = api_key or os.getenv("API_NINJAS_KEY", "")
        if not self.api_key:
            self.logger.warning(
                "⚠️  API_NINJAS_KEY 가 설정되지 않았습니다. "
                "환경변수 API_NINJAS_KEY 를 설정하거나 생성자에 api_key 를 전달하세요."
            )

        self._headers = {"X-Api-Key": self.api_key}

    # ── 내부 유틸 ────────────────────────────────────────────────────────────

    def _load_existing(self, file_path: Path) -> pd.DataFrame:
        """기존 CSV 로드. 없으면 빈 DataFrame 반환."""
        if file_path.exists():
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                self.logger.warning(f"기존 파일 로드 실패 ({file_path.name}): {e}")
        return pd.DataFrame()

    def _backup_and_save(self, df: pd.DataFrame, file_path: Path):
        """백업 생성 후 CSV 저장."""
        if file_path.exists():
            ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = BACKUP_DATA_DIR / f"{file_path.name}.backup_{ts}"
            try:
                shutil.copy2(file_path, backup_path)
                self.logger.info(f"백업: {backup_path.name}")
            except Exception as e:
                self.logger.error(f"백업 실패: {e}")
        try:
            df.to_csv(file_path, index=False)
            self.logger.info(f"✅ 저장: {file_path.name} ({len(df)}행)")
        except Exception as e:
            self.logger.error(f"저장 실패: {e}")

    # ── API 호출 ──────────────────────────────────────────────────────────────

    def _fetch_chunk(
        self, interval: str, start_unix: int, end_unix: int
    ) -> list[dict] | None:
        """
        단일 API 호출로 금 OHLCV 데이터 취득.
        반환값은 API 응답 그대로의 list (내림차순, 최대 1000 rows).
        실패 시 None 반환.
        """
        url    = f"{API_NINJAS_BASE_URL}/commoditypricehistorical"
        params = {
            "name":     COMMODITY_NAME,
            "interval": GOLD_API_INTERVALS[interval],
            "start":    start_unix,
            "end":      end_unix,
        }
        try:
            resp = requests.get(url, headers=self._headers, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                # API Ninjas 응답 구조: list 또는 {"data": list}
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                self.logger.warning(f"예상치 못한 응답 구조: {type(data)}")
                return []
            elif resp.status_code == 429:
                self.logger.warning("Rate limit 도달. 5초 대기 후 재시도.")
                time.sleep(5)
                return self._fetch_chunk(interval, start_unix, end_unix)
            else:
                self.logger.error(
                    f"API 오류 {resp.status_code}: {resp.text[:200]}"
                )
                return None
        except requests.RequestException as e:
            self.logger.error(f"요청 실패: {e}")
            return None

    def _fetch_range(
        self, interval: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        """
        start_dt ~ end_dt 전체 기간을 MAX_ROWS_PER_REQUEST 단위로 분할 수집.
        API 응답이 내림차순이므로 슬라이싱을 end → start 역방향으로 진행.
        """
        interval_sec = INTERVAL_SECONDS[interval]
        window_sec   = MAX_ROWS_PER_REQUEST * interval_sec

        start_unix = int(start_dt.timestamp())
        end_unix   = int(end_dt.timestamp())

        all_rows: list[dict] = []
        chunk_end   = end_unix
        chunk_start = max(start_unix, chunk_end - window_sec)

        self.logger.info(
            f"[GOLD {interval}] 수집 시작: "
            f"{datetime.fromtimestamp(start_unix, tz=timezone.utc).strftime('%Y-%m-%d')} ~ "
            f"{datetime.fromtimestamp(end_unix,   tz=timezone.utc).strftime('%Y-%m-%d')}"
        )

        while chunk_start >= start_unix:
            rows = self._fetch_chunk(interval, chunk_start, chunk_end)
            if rows is None:
                self.logger.error("청크 수집 실패, 수집 중단")
                break
            if rows:
                all_rows.extend(rows)
                self.logger.info(
                    f"  청크 {len(rows)}개 수집, 누적 {len(all_rows)}개"
                )

            # 다음 청크: 한 칸 더 과거로
            chunk_end   = chunk_start - interval_sec
            chunk_start = max(start_unix, chunk_end - window_sec)

            if chunk_end < start_unix:
                break

            time.sleep(REQUEST_DELAY)

        if not all_rows:
            return pd.DataFrame()

        return self._format_rows(all_rows)

    def _format_rows(self, rows: list[dict]) -> pd.DataFrame:
        """
        API 응답을 BTC CSV와 동일한 컬럼 구조로 변환.
        API Ninjas 응답 필드: timestamp, open, high, low, close, volume
        """
        df = pd.DataFrame(rows)

        # 필드명 정규화 (API 버전에 따라 다를 수 있음)
        rename_map = {}
        for src, dst in [("t", "timestamp"), ("o", "open"), ("h", "high"),
                          ("l", "low"), ("c", "close"), ("v", "volume")]:
            if src in df.columns and dst not in df.columns:
                rename_map[src] = dst
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # timestamp 컬럼 확보
        if "timestamp" not in df.columns:
            self.logger.error(f"timestamp 컬럼 없음. 실제 컬럼: {list(df.columns)}")
            return pd.DataFrame()

        df["unix"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = pd.NA

        # 가격 반올림
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].round(DECIMAL_PLACES["price"])
        if "volume" in df.columns:
            df["volume"] = df["volume"].round(DECIMAL_PLACES["volume"])

        df["date"]   = pd.to_datetime(df["unix"], unit="s", utc=True).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        df["symbol"] = SYMBOL_LABEL

        # BTC CSV와 동일한 컬럼 순서로 맞춤
        # CVD 관련 컬럼은 NA (금 선물은 taker 데이터 없음)
        df["Volume USD"]      = df["volume"]          # quote volume 대용
        df["taker_buy_base"]  = pd.NA
        df["volume_delta"]    = pd.NA

        # 기술적 지표: indicator.py에서 채워짐
        for col in ["macd", "macd_signal", "macd_hist", "rsi"]:
            df[col] = pd.NA

        final_cols = [
            "unix", "date", "symbol",
            "open", "high", "low", "close",
            "Volume USD", "volume",
            "taker_buy_base", "volume_delta",
            "macd", "macd_signal", "macd_hist", "rsi",
        ]
        df = df[[c for c in final_cols if c in df.columns]].copy()

        # 오름차순 정렬 (API 응답은 내림차순)
        df = df.dropna(subset=["unix"])
        df = df.sort_values("unix").reset_index(drop=True)
        df = df.drop_duplicates(subset=["unix"], keep="last")

        return df

    # ── 퍼블릭 인터페이스 ─────────────────────────────────────────────────────

    def update_ohlcv(self, timeframe: str):
        """
        특정 타임프레임 OHLCV 업데이트.
        - API 직접 지원(1h/4h/1d): Binance와 동일한 증분 수집
        - resample 타임프레임(1w/1m): GOLD_1d.csv 에서 자동 생성
        """
        if timeframe in GOLD_API_INTERVALS:
            self._update_from_api(timeframe)
        elif timeframe in GOLD_RESAMPLE_INTERVALS:
            self._update_from_resample(timeframe)
        else:
            self.logger.error(
                f"지원하지 않는 타임프레임: {timeframe}. "
                f"사용 가능: {GOLD_ALL_TIMEFRAMES}"
            )

    def _update_from_api(self, timeframe: str):
        """API에서 직접 수집 (1h / 4h / 1d)."""
        self.logger.info(f"===== GOLD {timeframe} 업데이트 (API) =====")
        file_path   = RAW_DATA_DIR / GOLD_DATA_FILES[timeframe]
        existing_df = self._load_existing(file_path)

        if not existing_df.empty and "unix" in existing_df.columns:
            last_unix   = int(existing_df["unix"].iloc[-1])
            start_fetch = datetime.fromtimestamp(last_unix, tz=timezone.utc)
            self.logger.info(f"  기존 마지막: {start_fetch.strftime('%Y-%m-%d %H:%M')}")
        else:
            start_fetch = GOLD_HISTORY_START[timeframe]
            self.logger.info(f"  최초 수집 시작: {start_fetch.strftime('%Y-%m-%d')}")

        new_df = self._fetch_range(timeframe, start_fetch, datetime.now(timezone.utc))

        if new_df.empty:
            self.logger.info("  추가할 신규 데이터 없음")
            return

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        before   = len(combined)
        combined.drop_duplicates(subset=["unix"], keep="last", inplace=True)
        combined.sort_values("unix", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        self.logger.info(f"  중복 {before - len(combined)}개 제거 → {len(combined)}행")

        self._backup_and_save(combined, file_path)

    def _update_from_resample(self, timeframe: str):
        """
        1d 데이터를 resample하여 1w / 1m 생성.

        resample 규칙:
          open   = 기간 내 첫 캔들 open
          high   = 기간 내 최고가
          low    = 기간 내 최저가
          close  = 기간 내 마지막 캔들 close
          volume = 기간 내 volume 합산

        MACD/RSI 컬럼은 resample 후 NA로 초기화.
        indicator.py가 resample된 close 가격으로 다시 계산함.
        """
        cfg = GOLD_RESAMPLE_INTERVALS[timeframe]
        self.logger.info(f"===== GOLD {timeframe} 업데이트 ({cfg['label']}) =====")

        # 1d 원본 로드
        src_path = RAW_DATA_DIR / GOLD_DATA_FILES["1d"]
        if not src_path.exists():
            self.logger.error(
                f"GOLD_1d.csv 없음. 먼저 update_ohlcv('1d') 를 실행하세요."
            )
            return

        df_1d = self._load_existing(src_path)
        if df_1d.empty or "unix" not in df_1d.columns:
            self.logger.error("GOLD_1d.csv 데이터가 비어있거나 unix 컬럼 없음")
            return

        # datetime 인덱스 설정
        df_1d = df_1d.copy()
        df_1d["unix"] = pd.to_numeric(df_1d["unix"], errors="coerce")
        df_1d.dropna(subset=["unix"], inplace=True)
        df_1d.sort_values("unix", inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df_1d.columns:
                df_1d[col] = pd.to_numeric(df_1d[col], errors="coerce")

        df_1d.index = pd.to_datetime(df_1d["unix"], unit="s", utc=True)

        # OHLCV resample
        freq = cfg["freq"]
        agg_rules = {
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }
        # 존재하는 컬럼만 집계
        agg_rules = {k: v for k, v in agg_rules.items() if k in df_1d.columns}

        resampled = df_1d.resample(freq).agg(agg_rules)
        resampled.dropna(subset=["close"], inplace=True)  # 비어있는 기간 제거

        # unix / date 재생성 (resample 후 인덱스가 기간 시작 datetime)
        resampled["unix"]   = (resampled.index.astype("int64") // 10**9).astype(int)
        resampled["date"]   = resampled.index.strftime("%Y-%m-%d %H:%M:%S")
        resampled["symbol"] = SYMBOL_LABEL

        # 가격 반올림
        for col in ["open", "high", "low", "close"]:
            if col in resampled.columns:
                resampled[col] = resampled[col].round(DECIMAL_PLACES["price"])
        if "volume" in resampled.columns:
            resampled["volume"] = resampled["volume"].round(DECIMAL_PLACES["volume"])

        resampled["Volume USD"]     = resampled.get("volume", pd.NA)
        resampled["taker_buy_base"] = pd.NA
        resampled["volume_delta"]   = pd.NA

        # 지표 컬럼 초기화 (indicator.py가 재계산)
        for col in ["macd", "macd_signal", "macd_hist", "rsi"]:
            resampled[col] = pd.NA

        final_cols = [
            "unix", "date", "symbol",
            "open", "high", "low", "close",
            "Volume USD", "volume",
            "taker_buy_base", "volume_delta",
            "macd", "macd_signal", "macd_hist", "rsi",
        ]
        result = resampled.reset_index(drop=True)
        result = result[[c for c in final_cols if c in result.columns]]

        self.logger.info(f"  resample 결과: {len(result)}개 캔들")

        dst_path = RAW_DATA_DIR / GOLD_DATA_FILES[timeframe]
        self._backup_and_save(result, dst_path)

    def update_all_ohlcv(self):
        """
        모든 타임프레임 OHLCV 일괄 업데이트.
        순서: 1d 먼저 수집 → 1w/1m resample (1d 의존성 보장)
        """
        self.logger.info("🚀 GOLD 전체 OHLCV 업데이트 시작")

        # API 직접 수집 순서: 1d를 가장 먼저 (resample 소스)
        api_order = ["1d", "4h", "1h"]
        for tf in api_order:
            self._update_from_api(tf)

        # resample 생성 (1d 완료 후)
        for tf in GOLD_RESAMPLE_INTERVALS.keys():
            self._update_from_resample(tf)

        self.logger.info("🎉 GOLD 전체 OHLCV 업데이트 완료")


# ── 직접 실행 진입점 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    api_key = os.getenv("API_NINJAS_KEY", "")
    if not api_key:
        api_key = input("API Ninjas API 키를 입력하세요: ").strip()

    collector = GoldDataCollector(api_key=api_key)

    print("\n🥇 Gold 데이터 수집기")
    print("─" * 40)
    print("1. 전체 타임프레임 업데이트 (1d/4h/1h API + 1w/1m resample)")
    print("2. 특정 타임프레임만 업데이트")
    print(f"   API 직접: {list(GOLD_API_INTERVALS.keys())}")
    print(f"   Resample: {list(GOLD_RESAMPLE_INTERVALS.keys())} (GOLD_1d.csv 필요)")

    choice = input("\n선택 (1~2): ").strip()

    if choice == "1":
        collector.update_all_ohlcv()
    elif choice == "2":
        print(f"사용 가능: {GOLD_ALL_TIMEFRAMES}")
        tf = input("타임프레임 입력: ").strip()
        collector.update_ohlcv(tf)
    else:
        print("❌ 잘못된 선택")
