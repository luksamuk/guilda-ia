"""Cliente MCP HTTP — consome o servidor na porta 11111.

Rodar com: uv run client_http.py
Certifique-se de que o servidor HTTP está rodando: uv run server_http.py
"""

import asyncio
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


async def main():
    # Conecta ao servidor MCP HTTP
    async with streamablehttp_client("http://localhost:11111/mcp") as (read, write, _):
        from mcp import ClientSession
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Carrega ferramentas do servidor MCP
            mcp_tools = await load_mcp_tools(session)
            print(f"✅ {len(mcp_tools)} ferramentas carregadas do servidor MCP:")
            for t in mcp_tools:
                print(f"   - {t.name}: {t.description}")

            # Cria agente com o LLM local + ferramentas MCP
            llm = ChatOpenAI(
                model="gemma4-e2b",
                base_url="http://localhost:12434/v1",
                api_key="nao_precisa",
                temperature=0,
            )

            agente = create_agent(llm, mcp_tools)

            # Testa com uma pergunta que requer encadeamento
            resultado = await agente.ainvoke(
                {"messages": "Quanto é 3 + 5 multiplicado por 2?"}
            )
            print(f"\n🤖 Resposta: {resultado['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())