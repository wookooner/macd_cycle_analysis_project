import sys

from src.btc_macd_cycle_ai_analyst.agent import run_query
from src.btc_macd_cycle_ai_analyst.settings import get_settings


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    _configure_stdio()

    query = (
        "Analyze the relationship between 1h cycle strength and n_up_4 "
        "when the higher 4h context is up and combo_4 is UUDU."
    )

    print(f"{get_settings().llm_model} analysis start...\n")
    answer = run_query(query)

    print("=== Analysis Result ===")
    print(answer)
