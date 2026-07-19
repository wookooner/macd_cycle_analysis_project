from __future__ import annotations

from ..prompts import ANALYST_SYSTEM_PROMPT
from ..services.paths import AnalystPaths
from ..settings import get_settings


def build_agent_summary() -> str:
    settings = get_settings()
    paths = AnalystPaths()
    return "\n".join(
        [
            "BTC MACD Cycle AI Analyst",
            f"App root: {settings.app_root}",
            f"Config path: {settings.config_path}",
            f"Data root env: {settings.data_root_env_name}",
            f"Resolved data root: {paths.data_root}",
            f"Default asset: {settings.default_asset}",
            f"LLM base URL: {settings.llm_base_url}",
            f"LLM model: {settings.llm_model}",
        ]
    )


def create_llm():
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def build_agent():
    from langgraph.prebuilt import create_react_agent
    from ..tools import get_default_tools

    llm = create_llm()
    tools = get_default_tools()
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=ANALYST_SYSTEM_PROMPT,
    )


def run_query(query: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    for msg in result["messages"]:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"[TOOL CALL] {tc['name']}({tc['args']})")
        elif msg.__class__.__name__ == "ToolMessage":
            content = str(msg.content)[:400]
            print(f"[TOOL RESULT] {msg.name}: {content}")
    return result["messages"][-1].content