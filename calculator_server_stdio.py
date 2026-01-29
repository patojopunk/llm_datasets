# import os
# from fastmcp import FastMCP

# mcp = FastMCP("Calculator")

# @mcp.tool()
# def add(a: float, b: float) -> float:
#     """Add two numbers. Use this whenever you need to compute a + b."""
#     return a + b


# if __name__ == "__main__":
#     transport = os.getenv("MCP_TRANSPORT", "stdio").lower()

#     if transport == "http":
#         host = os.getenv("MCP_HOST", "0.0.0.0")
#         port = int(os.getenv("MCP_PORT", "8080"))
#         path = os.getenv("MCP_PATH", "/mcp")
#         mcp.run(transport="http", host=host, port=port, path=path)
#     else:
#         mcp.run(transport="stdio")


# mcp_server/calculator_server.py
import logging
import sys
from fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("Calculator")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    print ("used add tool")
    return a + b

if __name__ == "__main__":
    # IMPORTANT: do not print() to stdout in stdio mode
    mcp.run(transport="stdio")
