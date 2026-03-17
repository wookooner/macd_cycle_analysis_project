"""
trading_bot/main.py
====================
BTC MACD Cycle Chain 반자동 매매 봇 진입점.

사용법:
    python main.py              # 드라이런 (기본)
    python main.py --live       # 실거래

경로 구조:
    macd_cycle_analysis_project/
    └── trading_bot/            ← 이 폴더
        ├── main.py             ← 이 파일
        ├── .env                ← 환경변수 (API 키 등)
        ├── config/
        │   ├── settings.py
        │   ├── ai_prompt.py
        │   └── trading_rules.md  ← AI가 읽는 규칙 문서
        ├── ...
"""

import sys
import os
import asyncio
import argparse
import logging
from pathlib import Path

# ── 1. 경로 설정 (최우선) ────────────────────────────────────────
# 이 파일(main.py)이 있는 디렉토리 = trading_bot/ = BOT_ROOT
BOT_ROOT = Path(__file__).parent.resolve()
# BOT_ROOT를 sys.path 최상단에 추가 → from config.xxx, from cycle.xxx 등 동작
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))


# ── 2. .env 로드 (Settings 생성 전에 반드시 먼저) ──────────────
def _load_env():
    env_path = BOT_ROOT / ".env"
    try:
        from dotenv import load_dotenv
        if env_path.exists():
            load_dotenv(env_path, override=True)
            return True
        else:
            print(f"⚠️  .env 파일 없음: {env_path}")
            print(f"   .env.template을 복사하여 .env를 생성하세요.")
            return False
    except ImportError:
        print("⚠️  python-dotenv 미설치 — 시스템 환경변수만 사용")
        return False


# ── 3. 로깅 설정 ────────────────────────────────────────────────
def _setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        from loguru import logger as loguru_logger
        loguru_logger.add(
            str(log_dir / "bot_{time:YYYY-MM-DD}.log"),
            rotation="1 day", retention="30 days", level="INFO",
            format="{time:HH:mm:ss} | {level:<7} | {name}:{function} | {message}",
            encoding="utf-8",
        )

        class _InterceptHandler(logging.Handler):
            def emit(self, record):
                loguru_logger.opt(depth=6, exception=record.exc_info).log(
                    record.levelname, record.getMessage()
                )

        logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO)
    except ImportError:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / "bot.log", encoding="utf-8"),
            ],
        )


# ── 4. 봇 실행 ──────────────────────────────────────────────────
async def _run_bot(live: bool = False):
    if live:
        os.environ["DRY_RUN"] = "false"
        os.environ["BINANCE_TESTNET"] = "false"

    # Settings는 .env 로드 후에 생성 (load_settings 사용)
    from config.settings import load_settings
    settings = load_settings()

    # 설정 검증
    missing = settings.validate()
    if missing:
        print(f"\n❌ 필수 환경변수 누락:")
        for m in missing:
            print(f"   - {m}")
        print(f"\n   .env 파일을 확인하세요: {BOT_ROOT / '.env'}")
        sys.exit(1)

    # 로깅
    _setup_logging(settings.paths.log_dir)
    logger = logging.getLogger("bot")

    mode = "🔴 LIVE" if live else "🟢 DRY RUN"
    logger.info(f"Starting bot [{mode}]")
    logger.info(f"BOT_ROOT: {BOT_ROOT}")
    logger.info(f"PROJECT_ROOT: {settings.paths.analysis_project_root}")
    logger.info(f"Schedule: 매 시간 {settings.scheduler.cron_minute}분")
    logger.info(f"Rules doc: {settings.paths.rules_doc_path}")

    from scheduler.main_loop import TradingScheduler
    scheduler = TradingScheduler(settings)

    try:
        await scheduler.start()
        # 무한 대기 (Ctrl+C 종료)
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C)...")
    finally:
        await scheduler.stop()


# ── 5. CLI ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="BTC MACD Cycle Chain Trading Bot"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="실거래 모드 (기본: 드라이런)"
    )
    args = parser.parse_args()

    # .env 로드 (최우선)
    _load_env()

    # 라이브 모드 경고
    if args.live:
        print("=" * 50)
        print("⚠️  실거래 모드!")
        print("   실제 자금으로 거래가 실행됩니다.")
        print("=" * 50)
        confirm = input("계속? (yes/no): ")
        if confirm.lower() != "yes":
            print("취소.")
            sys.exit(0)

    asyncio.run(_run_bot(live=args.live))


if __name__ == "__main__":
    main()
