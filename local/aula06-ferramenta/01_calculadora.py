from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

BASE = "http://localhost:12434/v1"
MODEL = "lfm2.5-24b-a2b"

llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE,
    api_key="nao_precisa",
    temperature=0,
    max_tokens=800,
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

llm_com_ferramentas = llm.bind_tools([calculate])

print(f"Agente com ferramenta ({MODEL}) — feche com Ctrl+D")
try:
    while True:
        msg = input("Você: ").strip()
        if not msg:
            continue
        try:
            mensagens = [HumanMessage(content=msg)]
            resposta = llm_com_ferramentas.invoke(mensagens)
            mensagens.append(resposta)

            if resposta.tool_calls:
                for tc in resposta.tool_calls:
                    print(f"  [tool_call] {tc['name']}({tc['args']})")
                    resultado = FERRAMENTAS[tc["name"]].invoke(tc["args"])
                    print(f"  [resultado] {resultado}")
                    mensagens.append(
                        ToolMessage(content=str(resultado), tool_call_id=tc["id"])
                    )
                final = llm_com_ferramentas.invoke(mensagens)
                print("Agente:", final.content)
            else:
                print("Agente:", resposta.content)
        except Exception as e:
            print(f"  [o servidor caiu: {e}]")
            print("  (tenta de novo — ele sobe sozinho)")
            continue
        print()
except (EOFError, KeyboardInterrupt):
    print("\nTchau!")
