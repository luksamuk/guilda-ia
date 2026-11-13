# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "requests",
#     "faiss-cpu",
#     "langchain",
#     "langchain-openai",
#     "langgraph",
# ]
# ///
"""
Guilda de IA — Aula 09 (Embeddings, Similaridade e RAG)
Exemplo FAISS local: banco de fatos -> embeddings -> FAISS -> busca semântica.
Roda contra o llama-swap local (sem Ollama):

    uv run aula09-rag.py

Modelos: embeddinggemma-2 (embeddings, CPU-only) + lfm3-1.7b:think (agente).
"""

import numpy as np
import requests


BASE_URL = "http://localhost:12434/v1"
EMBED_MODEL = "embeddinggemma-2"
CHAT_MODEL = "lfm3-1.7b"
EMBED_DIM = 256
LINHA = "=" * 60


fatos = [
    "Alan Turing published 'On Computable Numbers, with an Application to the Entscheidungsproblem' in 1936, introducing the Turing Machine as a formal model of algorithm and demonstrating the undecidability of the decision problem.",
    "The Church-Turing Thesis states that any effectively calculable cognitive process realizable by a finite sequence of steps can be simulated by a Turing Machine; it is a foundation of computational cognitive science.",
    "The Halting Problem is undecidable: no Turing Machine can decide, for every input, whether an arbitrary TM halts or loops forever. The proof uses diagonalization, a technique related to Gödel's theorem.",
    "Rice's Theorem (1953) establishes that every non-trivial property of recursively enumerable languages is undecidable — for example, deciding whether a TM accepts an empty, finite, or regular language.",
    "Kurt Gödel demonstrated in 1931 his incompleteness theorems: in any consistent formal system sufficient for arithmetic, there exist true propositions that are undecidable within the system.",
    "David Hilbert presented in 1900 a list of 23 mathematical problems; the 10th asked for an algorithm to decide the solvability of Diophantine equations, which proved undecidable via Turing Machines.",
    "The Chomsky Hierarchy classifies formal languages into four types (regular, context-free, context-sensitive, recursively enumerable), each recognized by an automaton of increasing power up to the Turing Machine.",
    "John von Neumann proposed in 1948 models of cellular automata and self-replicating machines, providing the theoretical basis for cellular computation and artificial life.",
    "Warren McCulloch and Walter Pitts published in 1943 'A Logical Calculus of the Ideas Immanent in Nervous Activity', establishing mathematical foundations for artificial neural networks.",
    "Claude Shannon founded Information Theory in 1948 ('A Mathematical Theory of Communication'), quantifying information via entropy and establishing limits of data compression and transmission.",
    "John Searle proposed in 1980 the 'Chinese Room' thought experiment, arguing that a machine can manipulate symbols syntactically without semantic understanding, criticizing strong AI.",
    "Andy Clark and David Chalmers published 'The Extended Mind' in 1998, defending the Parity Principle: if an external process would be cognitive if done in the head, then it is cognitive even outside it (Otto's notebook example).",
    "Francisco Varela, Evan Thompson, and Eleanor Rosch published 'The Embodied Mind' (1991), founding enactivism: cognition as embodied situated action, not internal processing of representations.",
    "Humberto Maturana and Francisco Varela proposed the concept of autopoiesis: living systems self-produce and maintain organizational closure, forming the basis of the biology of knowing ('The Tree of Knowledge', 1987).",
    "Douglas Hofstadter introduced the concept of 'strange loop' in 'Gödel, Escher, Bach' (1979) and deepened it in 'I Am a Strange Loop' (2007), describing the 'I' as a self-referential pattern distributed across multiple substrates.",
    "Ludwig Wittgenstein, in 'Philosophical Investigations' (1953), proposed that the meaning of a word is its use in language (language games) and argued against private language with the 'beetle in a box' argument.",
    "Daniel Dennett argues that the self is an 'intentional system' rather than a thinking substance, a critique of Cartesianism present in works such as 'Consciousness Explained' (1991).",
    "Shaun Gallagher, in 'Enactivist Interventions' (2017), argues that memory is not reliving the past but a description constructed in the present — an enactive anti-representational perspective.",
    "Aristotle, in the Metaphysics (Book III), enumerates the four causes — material, formal, efficient, and final — as modes of explaining being and change, forming the basis of classical ontology.",
    "Baruch Spinoza, in the Ethics (1677), argues that there exists a single infinite substance — Deus sive Natura — and that mind and body are parallel attributes of the same substance, denying Cartesian dualism.",
    "René Descartes, in the Meditations on First Philosophy (1641), formulates the mind-body substantial dualism and the 'cogito ergo sum' as the foundation of certain knowledge.",
    "Plato, in the Republic, presents the Socratic maieutics — a method of discovering truth through successive questioning — and the allegory of the cave regarding knowledge of the Forms.",
    "Gilles Deleuze, in 'Difference and Repetition' (1968), critiques representation and proposes an ontology of pure difference, influencing the philosophy of immanence and non-representational thought.",
]


def gerar_embedding(texto):
    resp = requests.post(
        f"{BASE_URL}/embeddings",
        json={"model": EMBED_MODEL, "input": texto},
    )
    resp.raise_for_status()
    emb = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
    emb = emb[:EMBED_DIM]
    return emb / np.linalg.norm(emb)


def montar_busca():
    import faiss

    vetores = np.array([gerar_embedding(f) for f in fatos])
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(vetores)
    print(f"Banco de fatos no ar: {index.ntotal} vetores de {EMBED_DIM} dims (FAISS)")
    return index


def buscar(index, pergunta, k=3):
    query = gerar_embedding(pergunta).reshape(1, -1)
    distancias, indices = index.search(query, k)
    resultados = []
    for dist, idx in zip(distancias[0], indices[0]):
        resultados.append((fatos[idx], 1 - dist / 2, int(idx)))
    return resultados


SYSTEM_PROMPT = (
    "You are a RAG agent with access to a knowledge base of facts about "
    "philosophy and computer science. You MUST follow these rules:\n"
    "1. ALWAYS use the search_facts tool before answering. NEVER answer "
    "from your own knowledge — your parametric knowledge is NOT trusted.\n"
    "2. If the question involves multiple topics, call search_facts multiple "
    "times with different queries to gather information from each angle.\n"
    "3. Only answer AFTER you have retrieved relevant facts from the tool.\n"
    "4. Base your answer exclusively on the facts returned by the tool.\n"
    "5. If the tool returns no relevant facts, say you couldn't find information.\n"
    "Remember: searching is not optional. It is the entire point of this system.\n"
    "6. ALWAYS answer the user in Portuguese (Brazilian). The facts are in English "
    "but your response must be in Portuguese. Translate and reformulate — do not "
    "copy-paste the English text. Present the information naturally in Portuguese."
)


def montar_agente(index):
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool

    @tool
    def search_facts(query: str, k: int = 3) -> str:
        """Search a knowledge base of facts about philosophy and computer science.
        All facts are stored in ENGLISH — always write your search queries in English
        for best results, even if the user's question is in another language.
        Returns the k most semantically similar facts to the query.
        Use this tool when you need to find information about philosophers,
        computer scientists, logic, consciousness, or the history of computing."""

        resultados = buscar(index, query, k=min(k, 5))
        if not resultados:
            return "No facts found."
        saida = []
        for i, (fato, sim, idx) in enumerate(resultados, 1):
            saida.append(f"Fact {i} (similarity {sim:.2f}): {fato}")
        return "\n\n".join(saida)

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        base_url=BASE_URL,
        api_key="nao_precisa",
        temperature=0,
    )
    agente = create_agent(llm, [search_facts], system_prompt=SYSTEM_PROMPT)
    print(f"Agente RAG pronto: {CHAT_MODEL} + ferramenta search_facts (FAISS)")
    return agente


def agente_stream(agente, pergunta):
    import textwrap

    print(f"\nPergunta: {pergunta}")
    print("-" * 60)
    from langchain_core.messages import HumanMessage

    resultado = agente.invoke({"messages": [HumanMessage(content=pergunta)]})
    for msg in resultado["messages"]:
        for tc in getattr(msg, "tool_calls", []) or []:
            args_str = str(tc["args"])
            if len(args_str) > 50:
                args_str = args_str[:47] + "..."
            print(f"\n[busca] {tc['name']}({args_str})")
            print("-" * 60)
    conteudo = resultado["messages"][-1].content
    print("\nAgente:")
    for line in conteudo.split("\n"):
        print(textwrap.fill(line, 60))
    print("-" * 60)


def main():
    import textwrap

    print("Guilda de IA — RAG com FAISS (local, via llama-swap)")
    print(LINHA)
    index = montar_busca()

    demos = [
        "Quem trabalhou com lógica e pensamento?",
        "Como linguagem e significado se relacionam?",
    ]
    for pergunta in demos:
        print(f"\nPergunta: {pergunta}")
        print("-" * 60)
        for fato, sim, idx in buscar(index, pergunta):
            print(f"  [{sim:.3f}] #{idx:02d}: {' '.join(textwrap.fill(fato, 58).split())}")

    agente = montar_agente(index)
    print(f"\n{LINHA}")
    print("Modo agente: digite perguntas (Ctrl+C para sair)")
    try:
        while (pergunta := input("\n> ")):
            agente_stream(agente, pergunta)
    except (KeyboardInterrupt, EOFError):
        pass


# Teste: "Qual o título exato do artigo de 1943 que fundou as bases matemáticas das redes neurais artificiais, e quem o escreveu?"
    
if __name__ == "__main__":
    main()
