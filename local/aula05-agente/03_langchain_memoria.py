from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

BASE = "http://localhost:12434/v1"
MODEL = "bonsai2-4b"

llm = ChatOpenAI(model=MODEL, base_url=BASE, api_key="nao_precisa", temperature=0.7)

session = InMemoryChatMessageHistory()

prompt_mem = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente amigável que responde em português."),
    ("placeholder", "{history}"),
    ("human", "{pergunta}"),
])

cadeia_mem = prompt_mem | llm

print("Agente LangChain COM memoria — feche com Ctrl+D")
try:
    while True:
        msg = input("Você: ").strip()
        if not msg:
            continue
        try:
            resposta = cadeia_mem.invoke({"pergunta": msg, "history": session.messages})
            print("Agente:", resposta.content)
        except Exception as e:
            print(f"  [o servidor caiu: {e}]")
            print("  (tenta de novo — ele sobe sozinho)")
            continue
        print()
        session.add_message(HumanMessage(content=msg))
        session.add_message(AIMessage(content=resposta.content))
except (EOFError, KeyboardInterrupt):
    print("\nTchau!")
