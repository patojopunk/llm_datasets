# chatbot/cli.py
import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


SYSTEM_PROMPT = (
    "You are a helpful chatbot.\n"
    "Use MCP tools for arithmetic:\n"
    "- If user asks to add/sum/plus, use tool `add`.\n"
    "- If user asks to subtract/difference/minus, use tool `subtract`.\n"
    "For everything else, respond normally."
)


def _build_llm() -> ChatOpenAI:
    model = os.getenv("LLM_MODEL", "gpt-oss:20b")
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


def _get_mcp_connections() -> dict:
    # HTTP transport: server must already be running
    return {
        "calculator": {
            "transport": "http",
            "url": os.getenv("MCP_URL", "http://127.0.0.1:8080/mcp"),
        }
    }


def _extract_last_assistant_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") in ("assistant", "ai"):
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)

        msg_type = getattr(msg, "type", None)
        if msg_type in ("ai", "assistant"):
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)

    return ""


async def _amain():
    load_dotenv()

    llm = _build_llm()
    connections = _get_mcp_connections()

    print("Starting chatbot with:")
    print(f"  LLM model:    {os.getenv('LLM_MODEL', 'gpt-oss:20b')}")
    print(f"  LLM base URL: {os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1')}")
    print(f"  MCP URL:      {os.getenv('MCP_URL', 'http://127.0.0.1:8080/mcp')}")
    print()

    client = MultiServerMCPClient(connections)

    # Keep one MCP session open for the lifetime of this CLI
    async with client.session("calculator") as session:
        tools = await load_mcp_tools(session)

        agent = create_agent(
            llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

        print("Chatbot ready. Type 'exit' or 'quit' to stop.\n")

        messages: list[Any] = []

        while True:
            user_text = input("> ").strip()
            if user_text.lower() in {"exit", "quit"}:
                break
            if not user_text:
                continue

            state_in = {"messages": [*messages, {"role": "user", "content": user_text}]}
            state_out = await agent.ainvoke(state_in)

            messages = state_out.get("messages", messages)

            print(_extract_last_assistant_text(messages))
            print()


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
