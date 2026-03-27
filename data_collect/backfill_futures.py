# backfill_futures_columns.py
#
# 목적:
#   기존 OHLCV CSV(BTCUSD_1d.csv 등)에 taker_buy_base, volume_delta 컬럼이
#   선물 상장일(2019-09-13) 이후부터 NaN으로 비어있는 상태를 해결한다.
#
# 동작 방식:
#   1. 기존 CSV 로드 → taker_buy_base가 NaN인 행 중 선물 상장일 이후 구간 추출
#   2. UMFutures API로 해당 기간의 BTCUSDT 선물 klines 수집 (taker_buy_base 포함)
#   3. unix timestamp 기준으로 매핑 → 빈 컬럼만 채움 (기존 OHLCV 값은 보존)
#   4. 백업 후 저장
#
# 주의:
#   - 현물(BTCUSD)과 선물(BTCUSDT)의 OHLCV 수치는 약간 다를 수 있음
#   - 여기서는 기존 현물 OHLCV는 그대로 두고, taker_buy_base/volume_delta만 선물에서 가져옴
#   - 선물 volume(base asset)도 함께 채워 CVD 계산의 분모로 활용

import pandas as pd
import time
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path

from binance.um_futures import UMFutures as FuturesClient
from binance.error import ClientError

# config.py에서 API 키와 경로 가져옴
from config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY,
    RAW_DATA_DIR, BACKUP_DATA_DIR,
    DATA_FILES, BINANCE_INTERVALS,
    ENABLE_BACKUP, REQUEST_DELAY, MAX_LIMIT
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# BTCUSDT 무기한 선물 상장일 unix timestamp (2019-09-13 00:00:00 UTC)
FUTURES_LAUNCH_UNIX = 1568332800
FUTURES_SYMBOL      = "BTCUSDT"


def setup_futures_client() -> FuturesClient | None:
    try:
        client = FuturesClient(key=BINANCE_API_KEY, secret=BINANCE_SECRET_KEY)
        client.ping()
        logger.info("✅ Futures 클라이언트 연결 완료")
        return client
    except Exception as e:
        logger.error(f"❌ Futures 클라이언트 초기화 실패: {e}")
        return None


def fetch_futures_klines(
    client: FuturesClient,
    interval: str,
    start_unix_s: int,
    end_unix_s: int
) -> pd.DataFrame:
    """
    BTCUSDT 선물 klines를 페이지네이션으로 전부 수집.
    반환 컬럼: unix, taker_buy_base_futures, volume_futures, volume_delta
    """
    start_ms = start_unix_s * 1000
    end_ms   = end_unix_s   * 1000

    logger.info(f"선물 klines 수집: {interval}  "
                f"{datetime.fromtimestamp(start_unix_s, tz=timezone.utc).date()} ~ "
                f"{datetime.fromtimestamp(end_unix_s,   tz=timezone.utc).date()}")

    all_klines   = []
    fetch_start  = start_ms

    while True:
        try:
            klines = client.klines(
                symbol   = FUTURES_SYMBOL,
                interval = interval,
                startTime = fetch_start,
                endTime   = end_ms,
                limit     = MAX_LIMIT
            )
            if not klines:
                break

            all_klines.extend(klines)
            fetch_start = klines[-1][0] + 1  # 마지막 open_time + 1ms

            logger.info(f"  {len(klines)}개 수집, 누적 {len(all_klines)}개")

            # 마지막 페이지면 종료
            if len(klines) < MAX_LIMIT:
                break

            time.sleep(REQUEST_DELAY)

        except ClientError as e:
            logger.error(f"API 오류: {e.status_code} - {e.error_message}")
            break
        except Exception as e:
            logger.error(f"수집 중 예외: {e}")
            break

    if not all_klines:
        logger.warning("수집된 klines 없음")
        return pd.DataFrame()

    # Binance klines 컬럼 순서
    # [0]open_time [1]open [2]high [3]low [4]close [5]volume
    # [6]close_time [7]quote_vol [8]trades [9]taker_buy_base [10]taker_buy_quote [11]ignore
    cols = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades_count',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ]
    df = pd.DataFrame(all_klines, columns=cols)

    numeric = ['volume', 'taker_buy_base']
    df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce')

    # unix 초 단위로 변환 (기존 CSV와 키 맞춤)
    df['unix'] = (df['open_time'] // 1000).astype(int)

    # volume_delta: 양수 = 매수 압력, 음수 = 매도 압력
    # taker_sell_base = volume - taker_buy_base
    df['volume_delta'] = 2 * df['taker_buy_base'] - df['volume']

    # 기존 CSV에 채울 컬럼만 반환 (unix를 merge 키로 사용)
    result = df[['unix', 'taker_buy_base', 'volume', 'volume_delta']].copy()
    result = result.rename(columns={
        'taker_buy_base': 'taker_buy_base',   # 그대로
        'volume':         'volume_futures',    # 기존 volume(현물)과 구분
    })
    result = result.drop_duplicates(subset=['unix']).sort_values('unix')

    logger.info(f"선물 klines 수집 완료: {len(result)}행")
    return result


def backfill_timeframe(timeframe: str):
    """
    단일 타임프레임 CSV에 대해 taker_buy_base, volume_delta를 소급 적용.
    """
    file_path = RAW_DATA_DIR / DATA_FILES[timeframe]
    if not file_path.exists():
        logger.error(f"파일 없음: {file_path}")
        return

    interval = BINANCE_INTERVALS[timeframe]

    # ── 1. 기존 데이터 로드 ──────────────────────────────────
    logger.info(f"\n{'='*55}")
    logger.info(f"[{timeframe}] 백필 시작: {file_path.name}")
    existing = pd.read_csv(file_path)

    total_rows   = len(existing)
    missing_mask = existing['taker_buy_base'].isna() & (existing['unix'] >= FUTURES_LAUNCH_UNIX)
    missing_rows = missing_mask.sum()

    if missing_rows == 0:
        logger.info(f"✅ [{timeframe}] 채울 데이터 없음. 이미 완료된 상태.")
        return

    # 채워야 할 구간의 시작/끝 unix
    target = existing[missing_mask]
    start_unix = int(target['unix'].iloc[0])
    end_unix   = int(target['unix'].iloc[-1])

    logger.info(f"  전체 {total_rows}행 중 {missing_rows}행 백필 대상")
    logger.info(f"  구간: {datetime.fromtimestamp(start_unix, tz=timezone.utc).date()} "
                f"~ {datetime.fromtimestamp(end_unix, tz=timezone.utc).date()}")

    # ── 2. 선물 klines 수집 ──────────────────────────────────
    client = setup_futures_client()
    if client is None:
        return

    futures_df = fetch_futures_klines(client, interval, start_unix, end_unix)
    if futures_df.empty:
        logger.error("선물 klines 수집 실패. 백필 중단.")
        return

    # ── 3. 매핑 및 컬럼 채우기 ───────────────────────────────
    # unix 기준으로 left join → 기존 행 유지, 선물 데이터만 병합
    existing = existing.set_index('unix')
    futures_df = futures_df.set_index('unix')

    # 매칭 인덱스 계산 (keyerror 방지)
    intersect_idx = existing.index.intersection(futures_df.index)

    # taker_buy_base: 선물 값으로 채움 (존재하는 인덱스만)
    existing.loc[intersect_idx, 'taker_buy_base'] = futures_df.loc[intersect_idx, 'taker_buy_base']

    # volume: 기존에 NaN인 행만 선물 volume으로 채움
    #         (현물 데이터가 있던 행은 그대로 두고, 없던 행만 채움)
    nan_volume_mask = existing['volume'].isna()
    fill_idx = nan_volume_mask & existing.index.isin(futures_df.index)
    existing.loc[fill_idx, 'volume'] = futures_df.loc[
        futures_df.index.isin(existing[nan_volume_mask].index), 'volume_futures'
    ]

    # volume_delta: 선물 값으로 채움 (존재하는 인덱스만)
    existing.loc[intersect_idx, 'volume_delta'] = futures_df.loc[intersect_idx, 'volume_delta']

    existing = existing.reset_index().sort_values('unix')

    # 최종 확인
    filled_after = existing['taker_buy_base'].notna().sum()
    logger.info(f"  채우기 완료: taker_buy_base 유효행 {filled_after}/{len(existing)}")

    # 매핑되지 않은 행 체크 (타임스탬프 불일치 등)
    still_missing = existing[
        existing['taker_buy_base'].isna() & (existing['unix'] >= FUTURES_LAUNCH_UNIX)
    ]
    if not still_missing.empty:
        logger.warning(f"  ⚠ 선물 상장 이후 여전히 NaN인 행: {len(still_missing)}개")
        logger.warning(f"    (타임스탬프 불일치 가능성 — 확인 필요)")
        logger.warning(still_missing[['unix','date']].head(5).to_string())

    # ── 4. 백업 후 저장 ──────────────────────────────────────
    if ENABLE_BACKUP:
        ts          = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = BACKUP_DATA_DIR / f"{file_path.name}.backfill_backup_{ts}"
        shutil.copy2(file_path, backup_path)
        logger.info(f"  백업 저장: {backup_path.name}")

    existing.to_csv(file_path, index=False)
    logger.info(f"✅ [{timeframe}] 저장 완료: {file_path}")


def backfill_all():
    """config.py의 DATA_FILES에 정의된 모든 타임프레임 백필."""
    logger.info("🚀 전체 타임프레임 백필 시작")
    for tf in DATA_FILES.keys():
        backfill_timeframe(tf)
    logger.info("🎉 전체 백필 완료")


if __name__ == "__main__":
    print("\n📥 선물 klines 소급 적용 (taker_buy_base / volume_delta 백필)")
    print("─" * 50)
    print("1. 특정 타임프레임만 백필")
    print("2. 전체 타임프레임 백필")

    choice = input("\n선택 (1/2): ").strip()

    if choice == "1":
        tf_list = list(DATA_FILES.keys())
        print(f"사용 가능: {tf_list}")
        tf = input("타임프레임 입력: ").strip()
        if tf in tf_list:
            backfill_timeframe(tf)
        else:
            print("❌ 잘못된 타임프레임")

    elif choice == "2":
        backfill_all()

    else:
        print("❌ 잘못된 선택")