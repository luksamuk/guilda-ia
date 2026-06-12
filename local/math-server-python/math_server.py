"""Servidor MCP com ferramentas matemáticas — Guilda de IA S07.

Rodar com: uv run math_server.py (stdio)
   ou:  uv run math_server_http.py (HTTP, porta 11111)
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()