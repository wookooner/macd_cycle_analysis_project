"""Compatibility entrypoint for the integrated API server."""

from src.dashboard_api.api_server import app, create_app, main


if __name__ == "__main__":
    raise SystemExit(main())
