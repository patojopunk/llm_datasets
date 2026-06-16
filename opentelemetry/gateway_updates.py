from mcp_server.calculator_server import mcp as calculator_mcp
from mcp_server.quote_counter_server import mcp as quote_counter_mcp

gateway.mount(calculator_mcp, namespace="math")
gateway.mount(quote_counter_mcp, namespace="quote")
