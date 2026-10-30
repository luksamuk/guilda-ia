"""
Guilda de IA — Aula 08 (Ferramentas Múltiplas e MCP)
LangChain: create_agent contra o llama-swap local.

Modelo: lfm3-1.7b:think — o handle :think liga o reasoning do LFM3 1.7B
(dual-mode; o plain responde direto). O content chega limpo — o raciocínio
fica embutido na saída total.

Roda em qualquer python com langchain-openai instalado:
    python3 aula08-basico.py
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# --- 1. Modelo local via llama-swap ---------------------------------
llm = ChatOpenAI(
    model="lfm3-1.7b:think",
    base_url="http://localhost:12434/v1",
    api_key="nao_precisa",   # servidor local — sem chave real
    temperature=0,
)

# --- 2. Ferramentas — a descrição decide TUDO -------------------------
@tool
def calculate(a: float, b: float, operation: str) -> float:
    """Performs a math operation between two numbers.
    Use when you need to calculate.
    Args:
        a: First number
        b: Second number
        operation: One of 'add', 'subtract', 'multiply', 'divide'
    """
    ops = {'add': a+b, 'subtract': a-b,
           'multiply': a*b, 'divide': a/b}
    return ops[operation]

@tool
def get_word_length(word: str) -> int:
    """Returns the number of characters in a word.
    Use when the user asks about the length or letters of a word."""
    return len(word)

# --- 3. O agente completo numa linha ----------------------------------
# create_agent repete Thought→Action→Observation até responder.
SISTEMA = ("You answer in Portuguese. For any calculation or word-length "
           "question, ALWAYS call the available tool instead of "
           "answering from memory.")

agente = create_agent(
    llm,
    [calculate, get_word_length],
    system_prompt=SISTEMA,
)

# --- 4. Chat em ciclo ---------------------------------------------------
mensagens = []
try:
    while (pergunta := input("\n> ")):
        mensagens.append(pergunta)
        resposta = agente.invoke({"messages": mensagens})
        print(resposta["messages"][-1].content)
        mensagens = resposta["messages"]
except (KeyboardInterrupt, EOFError):
    pass