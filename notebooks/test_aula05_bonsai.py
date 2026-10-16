#!/usr/bin/env python3
"""Teste E2E do fluxo da aula 05 contra o llama-swap real, modelo bonsai2-4b."""
import json, time, requests

BASE = "http://localhost:12434/v1"
MODEL = "bonsai2-4b"
T0 = time.time()

def medir(tag):
    print(f"[{tag}] {time.time()-T0:.1f}s desde o inicio")

# ==== 0. Agente na mao (requests puro) — igual ao notebook ====
def criar_agente(instrucoes, max_historico=20):
    return {"instrucoes": instrucoes, "historico": [], "max_historico": max_historico}

def conversar(agente, mensagem):
    messages = [{"role": "system", "content": agente["instrucoes"]}]
    limite = agente["max_historico"]
    for msg in agente["historico"][-limite:]:
        messages.append(msg)
    messages.append({"role": "user", "content": mensagem})
    response = requests.post(
        f"{BASE}/chat/completions",
        json={"model": MODEL, "messages": messages, "stream": False},
        timeout=300,
    )
    resposta = response.json()["choices"][0]["message"]["content"]
    agente["historico"].append({"role": "user", "content": mensagem})
    agente["historico"].append({"role": "assistant", "content": resposta})
    return resposta

agente = criar_agente("Você é um assistente amigável que responde em português.")
r1 = conversar(agente, "Oi! Meu nome é Ana.")
medir("agente requests - r1")
print(f"r1: {r1[:120]}")
r2 = conversar(agente, "Qual é o meu nome?")
medir("agente requests - r2 (memoria)")
print(f"r2: {r2[:120]}")
assert r1 and r2, "resposta vazia"
assert "ana" in r2.lower(), f"memoria falhou no fluxo requests: {r2!r}"
print("OK: requests puro lembra do nome")

# ==== 1. LangChain ChatOpenAI ====
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model=MODEL, base_url=BASE, api_key="nao_precisa", temperature=0.7)
resposta = llm.invoke("O que é LangChain em uma frase?")
medir("langchain invoke")
print(f"langchain: {resposta.content[:120]}")
assert resposta.content.strip(), "content vazio no ChatOpenAI"
print("OK: ChatOpenAI -> content chegou")

# ==== 2. Chat com memoria (InMemoryChatMessageHistory) ====
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

session = InMemoryChatMessageHistory()
prompt_mem = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente amigável que responde em português."),
    ("placeholder", "{history}"),
    ("human", "{pergunta}")
])
cadeia_mem = prompt_mem | llm

r1lc = cadeia_mem.invoke({"pergunta": "Meu nome é Ana e eu estudo engenharia.", "history": session.messages})
medir("cadeia memoria - r1")
session.add_message(HumanMessage(content="Meu nome é Ana e eu estudo engenharia."))
session.add_message(AIMessage(content=r1lc.content))

r2lc = cadeia_mem.invoke({"pergunta": "Qual é o meu nome e o que eu estudo?", "history": session.messages})
medir("cadeia memoria - r2")
session.add_message(HumanMessage(content="Qual é o meu nome e o que eu estudo?"))
session.add_message(AIMessage(content=r2lc.content))

print(f"r1lc: {r1lc.content[:100]}")
print(f"r2lc: {r2lc.content[:100]}")
assert "ana" in r2lc.content.lower(), f"memoria falhou no fluxo langchain: {r2lc.content!r}"

session_nova = InMemoryChatMessageHistory()
r3lc = cadeia_mem.invoke({"pergunta": "Qual é o meu nome?", "history": session_nova.messages})
medir("cadeia sessao nova")
print(f"sessao nova: {r3lc.content[:100]}")
print()
print("=== E2E PASS: fluxo completo da aula roda contra bonsai2-4b no swap ===")