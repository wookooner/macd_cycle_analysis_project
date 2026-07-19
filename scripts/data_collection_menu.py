from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.microstructure.binance_collector import run_collector
from data_pipeline.microstructure.binance_public_data import DATASETS, DEFAULT_DATASETS, _parse_date, backfill_public_data
from data_pipeline.microstructure.features import build_microstructure_features, feature_dir, normalize_timeframe
from data_pipeline.pipeline_runner import run_pipeline, validate_runtime_paths
from scripts.sync_intraday_market_data import main as sync_intraday_market_data
from src.common.paths import PROJECT_PATHS


LOGGER = logging.getLogger(__name__)

RESEARCH_DATASETS = (
    "aggTrades",
    "metrics",
    "bookDepth",
    "fundingRate",
    "markPriceKlines",
    "premiumIndexKlines",
    "klines",
)

ALL_PUBLIC_DATASETS = tuple(DATASETS)


def _setup_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in {None, ""} else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else (default or "")


def _prompt_bool(label: str, default: bool = False) -> bool:
    default_text = "y" if default else "n"
    value = _prompt(f"{label} (y/n)", default_text).lower()
    return value in {"y", "yes", "1", "true", "t", "ㅛ", "예", "네"}


def _prompt_choice(label: str, choices: list[str], default: str) -> str:
    choices_text = "/".join(choices)
    while True:
        value = _prompt(f"{label} ({choices_text})", default)
        if value in choices:
            return value
        print(f"잘못된 선택입니다: {value}")


def _prompt_date(label: str, default: date) -> date:
    while True:
        value = _prompt(label, default.isoformat())
        try:
            return _parse_date(value)
        except ValueError:
            print("YYYY-MM-DD 형식으로 입력하세요.")


def _prompt_datasets(default: tuple[str, ...] = RESEARCH_DATASETS) -> tuple[str, ...]:
    print("사용 가능한 데이터셋:", ", ".join(DATASETS))
    value = _prompt("데이터셋 (공백 구분, 전체는 all)", " ".join(default))
    datasets = DEFAULT_DATASETS if value == "all" else tuple(part.strip() for part in value.split() if part.strip())
    unknown = sorted(set(datasets).difference(DATASETS))
    if unknown:
        print(f"알 수 없는 데이터셋은 제외합니다: {', '.join(unknown)}")
        datasets = tuple(dataset for dataset in datasets if dataset in DATASETS)
    return datasets


def _write_features(symbol: str, timeframe: str, aux_tolerance: str, funding_tolerance: str, close_tolerance: str | None) -> Path:
    timeframe = normalize_timeframe(timeframe)
    df = build_microstructure_features(
        symbol=symbol,
        timeframe=timeframe,
        aux_tolerance=aux_tolerance,
        funding_tolerance=funding_tolerance,
        close_tolerance=close_tolerance,
    )
    if df.empty:
        raise RuntimeError(f"{symbol} {timeframe} 미시구조 피처 행이 없습니다.")
    out_dir = feature_dir(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"microstructure_features_{timeframe}.parquet"
    df.to_parquet(out_path, index=False)
    LOGGER.info("미시구조 피처 저장 완료: rows=%s path=%s", len(df), out_path)
    return out_path


def _default_dates() -> tuple[date, date]:
    today = date.today()
    default_start = today.replace(day=1)
    default_end = today - timedelta(days=1)
    if default_end < default_start:
        default_end = default_start
    return default_start, default_end


def _run_default_mode() -> None:
    print("\n기본모드: 경로 검증 -> 기존 BTC/GOLD 데이터 수집/지표 -> Binance 신규 데이터 전체 백필 -> 피처 생성")
    default_start, default_end = _default_dates()
    symbol = _prompt("심볼", "BTCUSDT").upper()
    start = _prompt_date("Binance 백필 시작일", default_start)
    end = _prompt_date("Binance 백필 종료일", default_end)
    interval = _prompt("캔들/프리미엄 interval", "1m")
    timeframe = _prompt("피처 timeframe", "1min")
    period = _prompt_choice("백필 단위", ["auto", "daily", "monthly"], "auto")
    overwrite = _prompt_bool("기존 ZIP/parquet 결과 덮어쓰기", False)
    force = _prompt_bool("기존 파이프라인 강제 재계산", False)
    run_gold = _prompt_bool("GOLD 기존 데이터도 함께 수집", True)

    _validate_paths()

    assets = ["btc", "gold"] if run_gold else ["btc"]
    for asset in assets:
        LOGGER.info("기존 데이터 파이프라인 실행: asset=%s steps=1,2", asset)
        ok = run_pipeline(
            asset=asset,
            steps=[1, 2],
            force=force,
            collect_futures=True,
            dry_run=False,
            microstructure_keep_files=0,
        )
        if not ok:
            raise RuntimeError(f"{asset} 기존 데이터 파이프라인 실패")

    results = backfill_public_data(
        symbol=symbol,
        datasets=ALL_PUBLIC_DATASETS,
        start=start,
        end=end,
        interval=interval,
        period=period,
        overwrite=overwrite,
        dry_run=False,
    )
    LOGGER.info("Binance 신규 데이터 백필 완료: 변환 객체 %s개", len(results))
    _write_features(symbol, timeframe, "10min", "9h", None)


def _collect_public_backfill(build_features: bool) -> None:
    default_start, default_end = _default_dates()

    symbol = _prompt("심볼", "BTCUSDT").upper()
    datasets = _prompt_datasets()
    start = _prompt_date("시작일", default_start)
    end = _prompt_date("종료일", default_end)
    interval = _prompt("캔들/프리미엄 interval", "1m")
    period = _prompt_choice("백필 단위", ["auto", "daily", "monthly"], "auto")
    overwrite = _prompt_bool("기존 ZIP/parquet 결과 덮어쓰기", False)

    results = backfill_public_data(
        symbol=symbol,
        datasets=datasets,
        start=start,
        end=end,
        interval=interval,
        period=period,
        overwrite=overwrite,
        dry_run=False,
    )
    LOGGER.info("백필 변환 완료: archive 객체 %s개", len(results))

    if build_features:
        timeframe = _prompt("피처 timeframe", "1min")
        aux_tolerance = _prompt("보조 데이터 tolerance", "10min")
        funding_tolerance = _prompt("확정 funding tolerance", "9h")
        close_tolerance_input = _prompt("라벨 close tolerance (빈 값 = 기본값)", "")
        close_tolerance = close_tolerance_input or None
        _write_features(symbol, timeframe, aux_tolerance, funding_tolerance, close_tolerance)


def _build_features_only() -> None:
    symbol = _prompt("심볼", "BTCUSDT").upper()
    timeframe = _prompt("피처 timeframe", "1min")
    aux_tolerance = _prompt("보조 데이터 tolerance", "10min")
    funding_tolerance = _prompt("확정 funding tolerance", "9h")
    close_tolerance_input = _prompt("라벨 close tolerance (빈 값 = 기본값)", "")
    _write_features(symbol, timeframe, aux_tolerance, funding_tolerance, close_tolerance_input or None)


def _run_live_microstructure() -> None:
    symbol = _prompt("심볼", "BTCUSDT").upper()
    depth_stream = _prompt("Depth 스트림", "depth20@100ms")
    flush_seconds = int(_prompt("저장 주기(초)", "30"))
    keep_files = int(_prompt("최근 raw 파일 보관 개수, 0 = 전체 보관", "0"))
    rest_poll_seconds = int(_prompt("REST 폴링 주기(초)", "60"))
    batch_rows = int(_prompt("배치 행 수", "200000"))
    depth_levels = int(_prompt("Depth 레벨 수", "20"))
    print("라이브 수집기는 Ctrl+C를 누를 때까지 계속 실행됩니다.")
    args = SimpleNamespace(
        symbol=symbol,
        streams=None,
        depth_stream=depth_stream,
        depth_levels=depth_levels,
        batch_rows=batch_rows,
        flush_seconds=flush_seconds,
        keep_files=keep_files,
        rest_poll_seconds=rest_poll_seconds,
        rest_periods=["5m", "15m", "1h"],
    )
    asyncio.run(run_collector(args))


def _run_pipeline_menu() -> None:
    asset = _prompt_choice("자산", ["btc", "gold", "all"], "btc")
    steps_text = _prompt("파이프라인 단계 (1 수집, 2 지표, 3 탐지, 4 매핑, 5 컨텍스트, 6 라이브 미시구조)", "1 2")
    steps = [int(part) for part in steps_text.split() if part.strip()]
    force = _prompt_bool("강제 재계산", False)
    dry_run = _prompt_bool("실행하지 않고 점검만", False)
    timeframes_text = _prompt("timeframe 제한 (빈 값 = 전체)", "")
    timeframes = [part.strip() for part in timeframes_text.split() if part.strip()] or None

    assets = ["btc", "gold"] if asset == "all" else [asset]
    overall_ok = True
    for selected_asset in assets:
        ok = run_pipeline(
            asset=selected_asset,
            steps=steps,
            force=force,
            dry_run=dry_run,
            timeframes=timeframes,
            microstructure_keep_files=0,
        )
        overall_ok = overall_ok and ok
    if not overall_ok:
        raise RuntimeError("하나 이상의 파이프라인 실행이 실패했습니다.")


def _validate_paths() -> None:
    PROJECT_PATHS.ensure_runtime_dirs()
    issues = validate_runtime_paths()
    print("경로 요약:")
    for key, value in PROJECT_PATHS.summary().items():
        print(f"  {key}: {value}")
    if issues:
        print("문제:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("경로 문제 없음.")


def _sync_intraday() -> None:
    code = sync_intraday_market_data()
    if code != 0:
        raise RuntimeError(f"Intraday 동기화 실패: code={code}")


def _menu_items() -> dict[str, tuple[str, Callable[[], None]]]:
    return {
        "1": ("기본모드: 기존 데이터 + Binance 신규 데이터 전체 수집 + 피처 생성", _run_default_mode),
        "2": ("Binance 공식 과거 데이터 백필 + 피처 생성", lambda: _collect_public_backfill(build_features=True)),
        "3": ("Binance 공식 과거 데이터 백필만 실행", lambda: _collect_public_backfill(build_features=False)),
        "4": ("기존 raw parquet에서 미시구조 피처만 생성", _build_features_only),
        "5": ("Binance 라이브 미시구조 수집기 실행", _run_live_microstructure),
        "6": ("기존 표준 데이터 파이프라인 단계 실행", _run_pipeline_menu),
        "7": ("BTC intraday 시장 데이터 + 지표 동기화", _sync_intraday),
        "8": ("데이터 경로 검증", _validate_paths),
        "0": ("종료", lambda: None),
    }


def run_menu() -> int:
    while True:
        print("\n=== 데이터 수집 통합 메뉴 ===")
        for key, (label, _) in _menu_items().items():
            print(f"{key}. {label}")
        choice = _prompt("선택", "1")
        items = _menu_items()
        if choice not in items:
            print(f"잘못된 선택입니다: {choice}")
            continue
        if choice == "0":
            return 0
        try:
            items[choice][1]()
        except KeyboardInterrupt:
            print("\n중단되었습니다.")
        except Exception as exc:
            LOGGER.exception("메뉴 실행 실패: %s", exc)
        if not _prompt_bool("다른 작업을 이어서 실행", False):
            return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="프로젝트 데이터 수집 통합 메뉴.")
    parser.add_argument("--validate-only", action="store_true", help="데이터 경로만 검증하고 종료합니다.")
    parser.add_argument("--default-mode", action="store_true", help="기본모드를 바로 실행합니다.")
    return parser.parse_args()


def main() -> int:
    _setup_logging()
    args = parse_args()
    if args.validate_only:
        _validate_paths()
        return 0
    if args.default_mode:
        _run_default_mode()
        return 0
    return run_menu()


if __name__ == "__main__":
    raise SystemExit(main())
