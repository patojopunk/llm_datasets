# chatbot/cli.py
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


SYSTEM_PROMPT = (
    "You are a helpful chatbot.\n"
    "When the user asks you to add two numbers (sum/plus/addition), "
    "use the MCP tool `add` rather than doing the math yourself.\n"
    "For everything else, respond normally."
)


def _build_stdio_command(server_path: Path) -> tuple[str, list[str]]:
    """
    Starts the MCP server in stdio mode.

    Conda notes:
      - If you run this after `conda activate <env>`, sys.executable is that env's Python
        and is the simplest/recommended way to spawn the server.
      - If you want to force conda, set:
            MCP_SPAWN=conda_run
            CONDA_ENV_NAME=<env>
    """
    spawn = os.getenv("MCP_SPAWN", "python").lower()

    if spawn == "conda_run":
        conda_exe = os.getenv("CONDA_EXE", "conda")
        env_name = os.getenv("CONDA_ENV_NAME") or os.getenv("CONDA_DEFAULT_ENV") or ""
        if not env_name:
            raise RuntimeError(
                "MCP_SPAWN=conda_run but no CONDA_ENV_NAME/CONDA_DEFAULT_ENV found. "
                "Set CONDA_ENV_NAME in your .env."
            )
        return conda_exe, ["run", "-n", env_name, "python", str(server_path)]

    return sys.executable, [str(server_path)]


def _get_mcp_connections() -> dict:
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()

    if transport == "http":
        return {
            "calculator": {
                "transport": "http",
                "url": os.getenv("MCP_URL", "http://localhost:8080/mcp"),
            }
        }

    repo_root = Path(__file__).resolve().parents[1]
    server_path = repo_root / "mcp_server" / "calculator_server.py"
    command, args = _build_stdio_command(server_path)

    return {
        "calculator": {
            "transport": "stdio",
            "command": command,
            "args": args,
        }
    }


def _build_llm() -> ChatOpenAI:
    """
    Uses an OpenAI-compatible endpoint (Ollama/vLLM/LM Studio/etc.)

    For Ollama in Docker:
      LLM_MODEL=gpt-oss:20b
      LLM_BASE_URL=http://localhost:11434/v1
      LLM_API_KEY=ollama  (dummy; often ignored by local servers)
    """
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


async def main():
    load_dotenv()

    llm = _build_llm()
    connections = _get_mcp_connections()

    print("Starting chatbot with:")
    print(f"  LLM model:     {os.getenv('LLM_MODEL', 'gpt-oss:20b')}")
    print(f"  LLM base URL:  {os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1')}")
    print(f"  MCP transport: {os.getenv('MCP_TRANSPORT', 'stdio')}")
    if os.getenv("MCP_TRANSPORT", "stdio").lower() == "http":
        print(f"  MCP URL:       {os.getenv('MCP_URL', 'http://localhost:8080/mcp')}")
    else:
        print(f"  MCP spawn:     {os.getenv('MCP_SPAWN', 'python')}")
    print()

    # IMPORTANT:
    # MultiServerMCPClient is NOT an async context manager in >=0.1.0.
    # So we instantiate it normally, and use client.session(...) when we want lifecycle mgmt.
    client = MultiServerMCPClient(connections)

    # Keep one session open for the lifetime of this CLI (recommended esp. for stdio).
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


if __name__ == "__main__":
    asyncio.run(main())
