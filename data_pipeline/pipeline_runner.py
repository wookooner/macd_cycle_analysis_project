"""
update_pipeline.py
==================
BTC / 금(Gold) 데이터 업데이트 통합 파이프라인

실행 순서:
  1. collect   - 데이터 수집  (BTC: Binance / Gold: API Ninjas)
  2. indicator - MACD / RSI / CVD 지표 계산
  3. detect    - 사이클 감지 → parquet 저장
  4. map       - 사이클 계층 관계 매핑

자산별 차이점:
  BTC  : 1h/4h/1d/1w/1m  |  CVD/funding_rate 지원  (Binance)
  Gold : 1h/4h/1d/1w/1m  |  CVD 없음               (API Ninjas + 1d resample)

저장 경로:
  BTC  data/base_data/BTCUSD_*.csv
       data/cycle_data/structured/btc/cycles_*.parquet          (자산별)
       data/cycle_data/structured/btc/cycle_hierarchy_map.json
       data/cycle_data/structured/cycles_*.parquet              (레거시 호환)
       data/cycle_data/structured/cycle_hierarchy_map.json

  Gold data/base_data/GOLD_*.csv
       data/cycle_data/structured/gold/cycles_*.parquet
       data/cycle_data/structured/gold/cycle_hierarchy_map.json

사용 예시:
  python update_pipeline.py                        # BTC 전체 (기본값)
  python update_pipeline.py --asset gold           # Gold 전체
  python update_pipeline.py --asset all            # BTC + Gold 순서대로
  python update_pipeline.py --asset btc --steps 2 3 4   # BTC 지표부터
  python update_pipeline.py --asset gold --steps 3 4    # Gold 사이클 재감지
  python update_pipeline.py --asset btc --force         # BTC 지표 전체 재계산
  python update_pipeline.py --asset btc --no-futures    # BTC 선물 수집 제외
"""

import shutil
import sys
import time
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
from data_pipeline.utils.io import atomic_write_csv, prune_backup_files

# ── 프로젝트 루트 ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# 자산별 설정 레지스트리
# ══════════════════════════════════════════════════════════════════════════════

ASSET_CONFIG = {
    "btc": {
        "label":        "BTC (Binance)",
        "data_files":   {
            "1h": "BTCUSD_1h.csv",
            "4h": "BTCUSD_4h.csv",
            "1d": "BTCUSD_1d.csv",
            "1w": "BTCUSD_1w.csv",
            "1m": "BTCUSD_1m.csv",
        },
        "cycle_dir":    PROJECT_ROOT / "data" / "cycle_data" / "structured" / "btc",
        # 기존 호환: structured/ 루트에도 동일 파일 저장 (기존 분석 도구/업로드 호환)
        "legacy_cycle_dir": PROJECT_ROOT / "data" / "cycle_data" / "structured",
        "has_cvd":      True,
        # 타임프레임별 CVD 롤링 윈도우 (20거래일 기준)
        "cvd_rolling":  {"1h": 480, "4h": 120, "1d": 20, "1w": 20, "1m": 20},
    },
    "gold": {
        "label":        "Gold (API Ninjas)",
        "data_files":   {
            "1h": "GOLD_1h.csv",
            "4h": "GOLD_4h.csv",
            "1d": "GOLD_1d.csv",
            "1w": "GOLD_1w.csv",   # 1d resample 생성
            "1m": "GOLD_1m.csv",   # 1d resample 생성
        },
        "cycle_dir":    PROJECT_ROOT / "data" / "cycle_data" / "structured" / "gold",
        "has_cvd":      False,   # taker 데이터 없으므로 CVD 계산 불가
        "cvd_rolling":  {"1h": 480, "4h": 120, "1d": 20, "1w": 20, "1m": 20},
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════

def _section(title: str):
    bar = "=" * 60
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 데이터 수집
# ══════════════════════════════════════════════════════════════════════════════

def step_collect(asset: str, collect_futures: bool = True) -> bool:
    _section(f"Step 1 / 4 : 데이터 수집 — {ASSET_CONFIG[asset]['label']}")
    t0 = time.time()

    if asset == "btc":
        try:
            from data_pipeline.collectors.new_collcetor import AdvancedBTCDataCollectorV2
        except ImportError as e:
            logger.error(f"new_collcetor.py import 실패: {e}")
            return False
        try:
            collector = AdvancedBTCDataCollectorV2()
            logger.info("▶ BTC OHLCV 전체 타임프레임 업데이트")
            collector.update_all_ohlcv()
            if collect_futures:
                logger.info("▶ 선물 데이터(펀딩비 + OI) 업데이트")
                collector.update_all_futures_data()
            else:
                logger.info("⏭  선물 데이터 수집 생략 (--no-futures)")
        except Exception:
            logger.error(f"❌ BTC 수집 실패:\n{traceback.format_exc()}")
            return False

    elif asset == "gold":
        try:
            from data_pipeline.collectors.gold_collector import GoldDataCollector
        except ImportError as e:
            logger.error(f"gold_collector.py import 실패: {e}")
            return False
        try:
            collector = GoldDataCollector()
            if not collector.api_key:
                logger.error("API_NINJAS_KEY 환경변수가 설정되지 않았습니다.")
                return False
            logger.info("▶ Gold OHLCV 전체 타임프레임 업데이트 (1h / 4h / 1d)")
            collector.update_all_ohlcv()
        except Exception:
            logger.error(f"❌ Gold 수집 실패:\n{traceback.format_exc()}")
            return False

    logger.info(f"✅ Step 1 완료 ({_elapsed(t0)})")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 지표 계산
# ══════════════════════════════════════════════════════════════════════════════

def step_indicator(asset: str, force: bool = False) -> bool:
    _section(f"Step 2 / 4 : 지표 계산 — {ASSET_CONFIG[asset]['label']}")
    t0 = time.time()

    try:
        from data_pipeline.indicators.indicator import IndicatorCalculator
    except ImportError as e:
        logger.error(f"indicator.py import 실패: {e}")
        return False

    cfg      = ASSET_CONFIG[asset]
    data_dir = PROJECT_ROOT / "data" / "base_data"
    calculator = IndicatorCalculator()
    success_count = fail_count = 0

    for tf, filename in cfg["data_files"].items():
        file_path = data_dir / filename
        if not file_path.exists():
            logger.warning(f"  파일 없음, 건너뜀: {filename}")
            continue

        rolling = cfg["cvd_rolling"].get(tf, 20)
        logger.info(f"  처리: {filename}  (cvd_rolling={rolling}, has_cvd={cfg['has_cvd']})")

        # Gold처럼 CVD 계산 불가 자산은 force_recalculate=True 여도
        # indicator.py가 volume_delta 없으면 자동으로 NA 처리함
        ok = calculator.process_file(
            file_path,
            force_recalculate=force,
            recalc_last_n=100,
            cvd_rolling_period=rolling,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"  결과: 성공 {success_count}개 / 실패 {fail_count}개")

    # BTC 전용: 지표 계산 후 OI / 펀딩비를 OHLCV CSV에 병합
    if asset == "btc":
        for tf in ["1h", "4h", "1d"]:
            filename = cfg["data_files"].get(tf)
            if filename and (data_dir / filename).exists():
                merged = _merge_futures_into_ohlcv(data_dir, tf, filename)
                if merged:
                    logger.info(f"  ✅ OI/펀딩비 병합 완료: {filename}")
                else:
                    logger.info(f"  ⏭  OI/펀딩비 병합 생략 (파일 없음): {filename}")

    logger.info(f"✅ Step 2 완료 ({_elapsed(t0)})")
    return fail_count == 0


# ══════════════════════════════════════════════════════════════════════════════
# OI / 펀딩비 → OHLCV 병합
# ══════════════════════════════════════════════════════════════════════════════

def _merge_futures_into_ohlcv(data_dir: Path, tf: str, ohlcv_filename: str) -> bool:
    """OI(oi, oi_usd)와 펀딩비(funding_rate)를 OHLCV CSV에 시간 기준으로 병합.

    - OI    : 같은 타임프레임 파일이 있을 때 date exact-match (left join)
    - 펀딩비 : 8h 간격이므로 merge_asof(backward) → 각 캔들에 직전 펀딩비 채움
    - 데이터가 없는 구간(OI 시작 전 등)은 NaN 유지
    """
    ohlcv_path = data_dir / ohlcv_filename
    if not ohlcv_path.exists():
        return False

    try:
        df = pd.read_csv(ohlcv_path)
        df["date"] = pd.to_datetime(df["date"])
        changed = False

        # ── OI 병합 ──────────────────────────────────────────────────────────
        oi_path = data_dir / f"BTCUSDT_oi_{tf}.csv"
        if oi_path.exists():
            oi_df = pd.read_csv(oi_path, usecols=["date", "oi", "oi_usd"])
            oi_df["date"] = pd.to_datetime(oi_df["date"])
            oi_df = oi_df.drop_duplicates("date").sort_values("date")
            # 기존 컬럼 제거 후 재병합 (최신 데이터 반영)
            df = df.drop(columns=[c for c in ["oi", "oi_usd"] if c in df.columns])
            df = df.merge(oi_df[["date", "oi", "oi_usd"]], on="date", how="left")
            changed = True

        # ── 펀딩비 병합 (backward fill) ──────────────────────────────────────
        fr_path = data_dir / "BTCUSDT_funding_rate.csv"
        if fr_path.exists():
            fr_df = pd.read_csv(fr_path, usecols=["date", "funding_rate"])
            fr_df["date"] = pd.to_datetime(fr_df["date"])
            fr_df = fr_df.drop_duplicates("date").sort_values("date")
            df = df.drop(columns=[c for c in ["funding_rate"] if c in df.columns])
            original_order = df.index.copy()
            df_sorted = df.sort_values("date").reset_index(drop=True)
            df_sorted = pd.merge_asof(
                df_sorted,
                fr_df,
                on="date",
                direction="backward",
            )
            # unix 기준으로 원래 정렬 복원
            sort_col = "unix" if "unix" in df_sorted.columns else "date"
            df = df_sorted.sort_values(sort_col).reset_index(drop=True)
            changed = True

        if not changed:
            return False

        # 백업 후 저장
        backup_dir = ohlcv_path.parent.parent / "backup_data"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{ohlcv_path.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(ohlcv_path, backup_path)
        prune_backup_files(backup_dir)

        # date 컬럼은 문자열로 되돌려 저장 (원본 형식 유지)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_csv(df, ohlcv_path)
        return True

    except Exception as e:
        logger.error(f"  OI/펀딩비 병합 실패 ({ohlcv_filename}): {e}\n{traceback.format_exc()}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: 사이클 감지
# ══════════════════════════════════════════════════════════════════════════════

def step_detect(asset: str) -> bool:
    _section(f"Step 3 / 4 : 사이클 감지 — {ASSET_CONFIG[asset]['label']}")
    t0 = time.time()

    cfg       = ASSET_CONFIG[asset]
    cycle_dir = cfg["cycle_dir"]
    legacy_dir = cfg.get("legacy_cycle_dir")
    cycle_dir.mkdir(parents=True, exist_ok=True)

    try:
        from data_pipeline.cycle_detectors.macd_histogram_change_detect import (
            load_algorithm,
            find_timeframe_files,
            detect_cycles_for_timeframe_v3,
            save_cycle_results_v3,
        )
    except ImportError as e:
        logger.error(f"macd_histogram_change_detect.py import 실패: {e}")
        return False

    try:
        algorithm, config = load_algorithm()

        # 자산별 파일 목록만 전달 (find_timeframe_files 를 자산 필터로 래핑)
        data_dir = PROJECT_ROOT / "data" / "base_data"
        timeframe_files = {}
        for tf, filename in cfg["data_files"].items():
            fp = data_dir / filename
            if fp.exists():
                timeframe_files[tf] = fp
            else:
                logger.warning(f"  {filename} 없음, 건너뜀")

        if not timeframe_files:
            logger.error("처리할 파일이 없습니다.")
            return False

        # 펀딩비 로드 (BTC만)
        funding_rate_df = None
        if asset == "btc":
            from data_pipeline.feature_extractors.macd_historgram_change_feature.feature_extract import (
                StructuredCycleProcessor,
            )
            fr_path = PROJECT_ROOT / "data" / "base_data" / "BTCUSDT_funding_rate.csv"
            if fr_path.exists():
                funding_rate_df = StructuredCycleProcessor.load_funding_rate(fr_path)
                logger.info(f"✅ 펀딩비 데이터 로드: {len(funding_rate_df)}행")
            else:
                logger.warning("⚠️  펀딩비 CSV 없음 → funding_rate 특징은 None으로 저장")

        results = {}
        for tf, fp in timeframe_files.items():
            try:
                cycle_records, cycle_count = detect_cycles_for_timeframe_v3(
                    fp, tf, algorithm, config, funding_rate_df
                )
                # 자산별 cycle_dir에 저장 (+ legacy_dir)
                if _save_cycles(cycle_records, tf, cycle_dir, legacy_dir):
                    results[tf] = cycle_count
            except Exception:
                logger.error(f"❌ {tf} 실패:\n{traceback.format_exc()}")

        logger.info(f"✅ Step 3 완료 ({_elapsed(t0)}) — {len(results)}개 타임프레임")
        return len(results) > 0

    except Exception:
        logger.error(f"❌ Step 3 실패:\n{traceback.format_exc()}")
        return False


def _save_cycles(cycle_records: list, timeframe: str, output_dir: Path,
                 legacy_dir: Path | None = None) -> bool:
    """사이클 결과를 자산별 디렉토리에 저장. legacy_dir이 있으면 거기에도 복사."""
    import pandas as pd
    import shutil

    if not cycle_records:
        logger.warning(f"⚠️  {timeframe}: 저장할 사이클 없음")
        return False
    try:
        df = pd.DataFrame(cycle_records)
        # 빈 카테고리 dict 제거 (PyArrow 저장 오류 방지)
        df["cycle_features"] = df["cycle_features"].apply(_prune_empty_features)
        out_path = output_dir / f"cycles_{timeframe}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"💾 저장: {out_path.name} ({len(cycle_records)}개)")

        # 기존 위치(structured/)에도 동일 파일 저장
        if legacy_dir and legacy_dir != output_dir:
            legacy_dir.mkdir(parents=True, exist_ok=True)
            legacy_path = legacy_dir / f"cycles_{timeframe}.parquet"
            shutil.copy2(out_path, legacy_path)
            logger.info(f"💾 레거시 복사: {legacy_path}")

        return True
    except Exception as e:
        logger.error(f"저장 실패 ({timeframe}): {e}")
        return False


def _prune_empty_features(features: dict) -> dict:
    """빈 카테고리 dict 제거."""
    if not isinstance(features, dict):
        return features
    return {k: v for k, v in features.items() if isinstance(v, dict) and v}


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: 계층 구조 매핑
# ══════════════════════════════════════════════════════════════════════════════

def step_map(asset: str) -> bool:
    _section(f"Step 4 / 4 : 계층 구조 매핑 — {ASSET_CONFIG[asset]['label']}")
    t0 = time.time()

    cfg       = ASSET_CONFIG[asset]
    cycle_dir = cfg["cycle_dir"]
    legacy_dir = cfg.get("legacy_cycle_dir")

    if not cycle_dir.exists():
        logger.error(f"사이클 디렉토리 없음: {cycle_dir}")
        return False

    try:
        from data_pipeline.cycle_detectors.cycle_time_mapper import CycleHierarchyMapper
    except ImportError as e:
        logger.error(f"cycle_time_mapper.py import 실패: {e}")
        return False

    try:
        mapper = CycleHierarchyMapper(cycle_dir)
        mapper.load_all_cycles()
        mapper.build_hierarchy_map(min_overlap=0.01)
        output_path = mapper.save_hierarchy_map()
        logger.info(f"  저장: {output_path}")

        # 기존 위치(structured/)에도 hierarchy_map 복사
        if legacy_dir and legacy_dir != cycle_dir:
            import shutil
            legacy_dir.mkdir(parents=True, exist_ok=True)
            legacy_map = legacy_dir / "cycle_hierarchy_map.json"
            shutil.copy2(output_path, legacy_map)
            logger.info(f"  레거시 복사: {legacy_map}")

        logger.info(f"✅ Step 4 완료 ({_elapsed(t0)})")
        return True
    except Exception:
        logger.error(f"❌ Step 4 실패:\n{traceback.format_exc()}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 파이프라인 실행기
# ══════════════════════════════════════════════════════════════════════════════

ALL_STEPS = [1, 2, 3, 4]

STEP_NAMES = {
    1: "collect",
    2: "indicator",
    3: "detect",
    4: "map",
}


def run_pipeline(
    asset: str,
    steps: list[int],
    force: bool = False,
    collect_futures: bool = True,
) -> bool:
    total_start = time.time()
    results: dict[int, bool] = {}

    logger.info(f"🚀 파이프라인 시작 — 자산: {ASSET_CONFIG[asset]['label'].upper()}")
    logger.info(f"   실행 스텝: {steps}")
    logger.info(f"   지표 강제 재계산: {force}")

    STEP_FNS = {
        1: lambda: step_collect(asset, collect_futures=collect_futures),
        2: lambda: step_indicator(asset, force=force),
        3: lambda: step_detect(asset),
        4: lambda: step_map(asset),
    }

    for step_num in sorted(steps):
        if step_num not in STEP_FNS:
            logger.warning(f"알 수 없는 스텝: {step_num}")
            continue
        try:
            ok = STEP_FNS[step_num]()
        except Exception:
            logger.error(f"예외 (step {step_num}):\n{traceback.format_exc()}")
            ok = False

        results[step_num] = ok
        if not ok:
            logger.warning(f"⚠️  Step {step_num} ({STEP_NAMES[step_num]}) 실패 — 계속 진행")

    # 요약
    logger.info("=" * 60)
    logger.info(f"📊 결과 요약 [{ASSET_CONFIG[asset]['label']}]")
    logger.info("=" * 60)
    all_ok = True
    for step_num, ok in results.items():
        icon = "✅" if ok else "❌"
        logger.info(f"  {icon} Step {step_num}: {STEP_NAMES[step_num]}")
        if not ok:
            all_ok = False
    logger.info(f"\n  총 소요 시간: {_elapsed(total_start)}")
    logger.info("🎉 완료!" if all_ok else "⚠️  일부 스텝 실패. 로그 확인.")
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(
        description="BTC / Gold 데이터 업데이트 통합 파이프라인",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
자산 (--asset):
  btc   BTC (Binance)     — 1h/4h/1d/1w/1m, CVD/펀딩비 지원
  gold  Gold (API Ninjas) — 1h/4h/1d API + 1w/1m resample, CVD 없음
  all   btc → gold 순서대로 실행

스텝 (--steps):
  1  collect   데이터 수집
  2  indicator MACD/RSI/CVD 지표 계산
  3  detect    사이클 감지
  4  map       계층 구조 매핑

예시:
  python update_pipeline.py                          # BTC 전체
  python update_pipeline.py --asset gold             # Gold 전체
  python update_pipeline.py --asset all              # BTC + Gold 전체
  python update_pipeline.py --asset gold --steps 3 4 # Gold 사이클 재감지
  python update_pipeline.py --asset btc --no-futures # BTC 선물 수집 제외
  python update_pipeline.py --asset btc --force      # BTC 지표 전체 재계산
        """,
    )
    parser.add_argument(
        "--asset",
        choices=["btc", "gold", "all"],
        default="btc",
        help="대상 자산 (기본값: btc)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=ALL_STEPS,
        choices=ALL_STEPS,
        metavar="N",
        help="실행할 스텝 번호 (기본값: 1 2 3 4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="지표를 처음부터 전체 재계산",
    )
    parser.add_argument(
        "--no-futures",
        dest="no_futures",
        action="store_true",
        help="[BTC 전용] 선물 데이터(펀딩비/OI) 수집 생략",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    assets_to_run = ["btc", "gold"] if args.asset == "all" else [args.asset]
    overall_ok    = True

    for asset in assets_to_run:
        ok = run_pipeline(
            asset=asset,
            steps=args.steps,
            force=args.force,
            collect_futures=not args.no_futures,
        )
        if not ok:
            overall_ok = False

    sys.exit(0 if overall_ok else 1)
