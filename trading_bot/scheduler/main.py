"""
trading_bot/main.py
====================
BTC MACD Cycle Chain 반자동 매매 봇 진입점.

사용법:
    python main.py              # 기본 실행 (DRY_RUN=True)
    python main.py --live       # 실거래 모드

환경변수 (.env 파일 또는 시스템 환경변수):
    BINANCE_API_KEY       Binance API 키
    BINANCE_API_SECRET    Binance API 시크릿
    TELEGRAM_BOT_TOKEN    Telegram 봇 토큰
    TELEGRAM_CHAT_ID      Telegram 채팅 ID
    ANTHROPIC_API_KEY     Claude API 키
    DRY_RUN               드라이런 모드 (기본: True)
    BINANCE_TESTNET       테스트넷 사용 (기본: True)
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_BOT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_BOT_ROOT))


def setup_logging(log_dir: Path):
    """로깅 설정"""
    log_dir.mkdir(parents=True, exist_ok=True)

    # loguru가 있으면 사용, 없으면 기본 logging
    try:
        from loguru import logger as loguru_logger
        loguru_logger.add(
            str(log_dir / "bot_{time:YYYY-MM-DD}.log"),
            rotation="1 day",
            retention="30 days",
            level="INFO",
            format="{time:HH:mm:ss} | {level:<7} | {name}:{function} | {message}",
        )
        # 표준 logging → loguru 연결
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                loguru_logger.opt(
                    depth=6, exception=record.exc_info
                ).log(record.levelname, record.getMessage())

        logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)

    except ImportError:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    log_dir / "bot.log", encoding="utf-8"
                ),
            ],
        )

    return logging.getLogger("bot")


def load_env():
    """dotenv 로드 (있으면)"""
    try:
        from dotenv import load_dotenv
        env_path = _BOT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return True
    except ImportError:
        pass
    return False


async def run_bot(live: bool = False):
    """봇 실행"""
    import os

    # 라이브 모드 오버라이드
    if live:
        os.environ["DRY_RUN"] = "false"
        os.environ["BINANCE_TESTNET"] = "false"

    from config.settings import Settings
    from scheduler.main_loop import TradingScheduler

    settings = Settings()

    # 설정 검증
    missing = settings.validate()
    if missing:
        print(f"❌ 필수 환경변수 누락: {', '.join(missing)}")
        print("   .env 파일을 생성하거나 환경변수를 설정하세요.")
        sys.exit(1)

    logger = logging.getLogger("bot")

    mode = "🔴 LIVE" if live else "🟢 DRY RUN"
    logger.info(f"Starting bot in {mode} mode")

    scheduler = TradingScheduler(settings)

    try:
        await scheduler.start()

        # 무한 대기 (Ctrl+C로 종료)
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C)...")
    finally:
        await scheduler.stop()


def main():
    parser = argparse.ArgumentParser(
        description="BTC MACD Cycle Chain Trading Bot"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="실거래 모드 (기본: 드라이런)"
    )
    args = parser.parse_args()

    # .env 로드
    load_env()

    # 로깅 설정
    from config.settings import settings
    setup_logging(settings.paths.log_dir)

    # 라이브 모드 경고
    if args.live:
        print("=" * 50)
        print("⚠️  실거래 모드입니다!")
        print("   실제 자금으로 거래가 실행됩니다.")
        print("=" * 50)
        confirm = input("계속하시겠습니까? (yes/no): ")
        if confirm.lower() != "yes":
            print("취소됨.")
            sys.exit(0)

    asyncio.run(run_bot(live=args.live))


if __name__ == "__main__":
    main()