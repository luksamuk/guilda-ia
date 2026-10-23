from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

BASE = "http://localhost:12434/v1"
MODEL = "lfm2.5-24b-a2b"
MAX_PASSOS = 5


class ChatComReasoning(ChatOpenAI):
    def _create_chat_result(self, response, generation_info=None):
        resultado = super()._create_chat_result(response, generation_info)
        bruto = response if isinstance(response, dict) else response.model_dump()
        for geracao, choice in zip(resultado.generations, bruto.get("choices", [])):
            pensamento = (choice.get("message") or {}).get("reasoning_content")
            if pensamento:
                geracao.message.additional_kwargs["reasoning_content"] = pensamento
        return resultado


llm = ChatComReasoning(
    model=MODEL,
    base_url=BASE,
    api_key="nao_precisa",
    temperature=0,
    max_tokens=800,
)


def mostrar_reasoning(mensagem):
    pensamento = mensagem.additional_kwargs.get("reasoning_content")
    if pensamento:
        print("  [thinking]", pensamento.strip().replace("\n", "\n              "))


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

historico = [
    SystemMessage(
        content="Você é um assistente que responde em português. "
        "Use a ferramenta calculate para qualquer cálculo numérico."
    )
]

print(f"Agente com ferramenta ({MODEL}) — feche com Ctrl+D")
try:
    while True:
        msg = input("Você: ").strip()
        if not msg:
            continue
        print()
        marca = len(historico)
        try:
            historico.append(HumanMessage(content=msg))
            resposta = llm_com_ferramentas.invoke(historico)

            for _ in range(MAX_PASSOS):
                mostrar_reasoning(resposta)
                historico.append(resposta)

                if not resposta.tool_calls:
                    print("Agente:", resposta.content)
                    break

                for tc in resposta.tool_calls:
                    print(f"  [tool_call] {tc['name']}({tc['args']})")
                    resultado = FERRAMENTAS[tc["name"]].invoke(tc["args"])
                    print(f"  [resultado] {resultado}")
                    historico.append(
                        ToolMessage(content=str(resultado), tool_call_id=tc["id"])
                    )

                resposta = llm_com_ferramentas.invoke(historico)
            else:
                print("  [parei: passos demais sem resposta final]")
        except Exception as e:
            del historico[marca:]
            print(f"  [o servidor caiu: {e}]")
            print("  (tenta de novo — ele sobe sozinho)")
            continue
        print(f"  [historico: {len(historico)} mensagens]")
        print()
except (EOFError, KeyboardInterrupt):
    print("\nTchau!")
