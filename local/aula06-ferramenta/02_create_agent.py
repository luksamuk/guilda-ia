from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

BASE = "http://localhost:12434/v1"
MODEL = "lfm2.5-24b-a2b"
LINHA = "-" * 72
INSTRUCOES = (
    "Você é um assistente que responde em português. "
    "Use a ferramenta calculate para qualquer cálculo numérico."
)


@tool
def calculate(a: float, b: float, operation: str) -> float:
    """Performs a mathematical operation between two numbers and returns the exact result.
    Use when you need to perform numeric calculations.

    Args:
        a: First number
        b: Second number
        operation: One of 'add', 'subtract', 'multiply', 'divide'
    """
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            return "Erro: divisão por zero"
        return a / b
    return f"Erro: operação '{operation}' não suportada"


FERRAMENTAS = {calculate.name: calculate}


def montar_agente():
    llm = ChatOpenAI(
        model=MODEL,
        base_url=BASE,
        api_key="nao_precisa",
        temperature=0,
        max_tokens=800,
    )
    return create_agent(
        llm,
        list(FERRAMENTAS.values()),
        system_prompt=INSTRUCOES,
    )


def main():
    agente = montar_agente()
    historico = []

    print(f"Agente com ferramenta ({MODEL}) — feche com Ctrl+D")
    try:
        while True:
            msg = input("Você: ").strip()
            if not msg:
                continue
            quantas_antes = len(historico)
            try:
                historico.append(HumanMessage(content=msg))
                resultado = agente.invoke({"messages": historico})
                historico = resultado["messages"]
            except Exception as e:
                del historico[quantas_antes:]
                bloco("Erro", f"{e}\n(tenta de novo — o servidor sobe sozinho)")
                continue

            for mensagem in historico[quantas_antes:]:
                if mensagem.type == "ai" and mensagem.tool_calls:
                    for tc in mensagem.tool_calls:
                        argumentos = ", ".join(
                            f"{k}={v!r}" for k, v in tc["args"].items()
                        )
                        bloco("Tool call", f"{tc['name']}({argumentos})")
                elif mensagem.type == "tool":
                    bloco("Resultado da ferramenta", mensagem.content)
                elif mensagem.type == "ai":
                    bloco("Assistente", mensagem.content)

            bloco(f"Histórico — {len(historico)} mensagens")
    except (EOFError, KeyboardInterrupt):
        print("\nTchau!")


def bloco(titulo, texto=""):
    print(f"\n[{titulo}]")
    if texto:
        print(texto.rstrip())
    print(LINHA)


if __name__ == "__main__":
    main()
