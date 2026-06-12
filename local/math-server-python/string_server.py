"""Servidor MCP de strings — Guilda de IA S07.

Rodar com: uv run string_server.py (stdio)
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Strings")


@mcp.tool()
def reverse_string(text: str) -> str:
    """Reverse a string. Use when the user wants to flip or reverse text."""
    return text[::-1]


@mcp.tool()
def count_words(text: str) -> int:
    """Count the number of words in a text.
    Use when the user asks how many words are in a sentence.
    """
    return len(text.split())


if __name__ == "__main__":
    mcp.run(transport="stdio")