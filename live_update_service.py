"""Compatibility entrypoint for the live update service."""

from src.services.live_update_service import LiveUpdateService, main, setup_logging


if __name__ == "__main__":
    raise SystemExit(main())
