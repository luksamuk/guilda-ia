"""Servidor MCP HTTP — mesma lógica, mas acessível via rede.

Rodar com: uv run server_http.py
Acessível em: http://localhost:11111/mcp
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math", host="0.0.0.0", port=11111)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="streamable-http")