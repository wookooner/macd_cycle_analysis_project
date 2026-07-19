import sys

from src.btc_macd_cycle_ai_analyst.agent import build_agent_summary, run_query


def _configure_stdio() -> None:
    """Prefer UTF-8 console output so model responses do not crash on Windows."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_stdio()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        print(build_agent_summary())
        print()
        query = input("Query> ").strip()

    if not query:
        print("No query provided.")
        return

    try:
        answer = run_query(query)
    except ModuleNotFoundError as exc:
        print(f"Missing runtime dependency: {exc.name}")
        print("Run this entrypoint with the ai_analyst virtual environment.")
        return

    print()
    print("=== Analysis Result ===")
    print(answer)


if __name__ == "__main__":
    main()
