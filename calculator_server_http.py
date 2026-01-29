# mcp_server/calculator_server.py
import logging
import os
import sys
from fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("Calculator")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


if __name__ == "__main__":
    # HTTP (streamable) settings
    host = os.getenv("MCP_HOST", "127.0.0.1")   # use 0.0.0.0 if you want LAN access
    port = int(os.getenv("MCP_PORT", "8080"))
    path = os.getenv("MCP_PATH", "/mcp")

    # Run as an HTTP MCP server
    mcp.run(transport="http", host=host, port=port, path=path)
