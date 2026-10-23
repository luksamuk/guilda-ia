from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

BASE = "http://localhost:12434/v1"
MODEL = "lfm2.5-24b-a2b"
#MODEL = "bonsai2-9b"
MAX_PASSOS = 5
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
    llm_com_ferramentas = llm.bind_tools(list(FERRAMENTAS.values()))
    historico = [SystemMessage(content=INSTRUCOES)]
    return llm_com_ferramentas, historico


def main():
    llm_com_ferramentas, historico = montar_agente()

    print(f"Agente com ferramenta ({MODEL}) — feche com Ctrl+D")
    try:
        while True:
            msg = input("Você: ").strip()
            if not msg:
                continue
            marca = len(historico)
            try:
                historico.append(HumanMessage(content=msg))
                resposta = llm_com_ferramentas.invoke(historico)

                for _ in range(MAX_PASSOS):
                    mostrar_thinking(resposta)
                    historico.append(resposta)

                    if not resposta.tool_calls:
                        bloco("Assistente", resposta.content)
                        break

                    for tc in resposta.tool_calls:
                        argumentos = ", ".join(
                            f"{k}={v!r}" for k, v in tc["args"].items()
                        )
                        bloco("Tool call", f"{tc['name']}({argumentos})")
                        resultado = FERRAMENTAS[tc["name"]].invoke(tc["args"])
                        bloco("Resultado da ferramenta", str(resultado))
                        historico.append(
                            ToolMessage(content=str(resultado), tool_call_id=tc["id"])
                        )

                    resposta = llm_com_ferramentas.invoke(historico)
                else:
                    bloco("Parada", "limite de passos atingido sem resposta final")
            except Exception as e:
                del historico[marca:]
                bloco("Erro", f"{e}\n(tenta de novo — o servidor sobe sozinho)")
                continue
            bloco(f"Histórico — {len(historico)} mensagens")
    except (EOFError, KeyboardInterrupt):
        print("\nTchau!")


def bloco(titulo, texto=""):
    print(f"\n[{titulo}]")
    if texto:
        print(texto.rstrip())
    print(LINHA)


def mostrar_thinking(mensagem):
    if not mensagem.tool_calls:
        return
    tokens = (mensagem.usage_metadata or {}).get("output_tokens", 0)
    if tokens:
        print(f"[Thinking] {tokens} tokens")


if __name__ == "__main__":
    main()
