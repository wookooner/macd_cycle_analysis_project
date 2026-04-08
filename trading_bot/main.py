"""
Trading bot entrypoint.

Typical usage:
    python main.py
    python main.py --mode approve
    python main.py --mode auto --live
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path


BOT_ROOT = Path(__file__).parent.resolve()
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))


def _load_env():
    env_path = BOT_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        if env_path.exists():
            load_dotenv(env_path, override=True)
            return True
        print(f".env file not found: {env_path}")
        print("Copy .env.template to .env and fill in the required values.")
        return False
    except ImportError:
        print("python-dotenv is not installed. Falling back to system environment variables.")
        return False


def _setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "bot.log", encoding="utf-8"),
        ],
    )


async def _run_bot(live: bool = False, mode: str | None = None):
    if live:
        os.environ["DRY_RUN"] = "false"
        os.environ["BINANCE_TESTNET"] = "false"
    if mode:
        os.environ["BOT_MODE"] = mode

    from config.settings import load_settings

    settings = load_settings()
    missing = settings.validate()
    if missing:
        print("\nMissing required configuration:")
        for item in missing:
            print(f"  - {item}")
        print(f"\nCheck: {BOT_ROOT / '.env'}")
        sys.exit(1)

    _setup_logging(settings.paths.log_dir)
    logger = logging.getLogger("bot")
    logger.info("Starting bot")
    logger.info("Project root: %s", settings.paths.analysis_project_root)
    logger.info("Mode: %s", settings.trading.mode.value.upper())
    logger.info("Dry run: %s", settings.trading.dry_run)
    logger.info("Data source mode: %s", settings.trading.data_source_mode.value)

    from scheduler.main_loop import TradingScheduler

    scheduler = TradingScheduler(settings)
    try:
        await scheduler.start()
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await scheduler.stop()


def main():
    parser = argparse.ArgumentParser(description="BTC cycle trading bot")
    parser.add_argument("--live", action="store_true", help="Run with real orders instead of dry-run.")
    parser.add_argument(
        "--mode",
        choices=["analyze", "approve", "auto"],
        help="Override BOT_MODE from .env for this run.",
    )
    args = parser.parse_args()

    _load_env()

    if args.live:
        print("=" * 50)
        print("LIVE mode requested.")
        print("Real orders may be sent if your config allows it.")
        print("=" * 50)
        confirm = input("Continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            sys.exit(0)

    asyncio.run(_run_bot(live=args.live, mode=args.mode))


if __name__ == "__main__":
    main()
