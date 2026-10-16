import requests
import json

BASE = "http://localhost:12434/v1/chat/completions"
MODEL = "bonsai2-4b"


def criar_agente(instrucoes, max_historico=20):
    return {
        "instrucoes": instrucoes,
        "historico": [],
        "max_historico": max_historico,
    }


def rotacionar(agente):
    """Esquece o mais antigo quando o historico passa do max."""
    limite = agente["max_historico"]
    if len(agente["historico"]) > limite:
        agente["historico"][:] = agente["historico"][-limite:]


def conversar(agente, mensagem):
    messages = [{"role": "system", "content": agente["instrucoes"]}]
    for msg in agente["historico"]:
        messages.append(msg)
    messages.append({"role": "user", "content": mensagem})

    response = requests.post(
        BASE,
        json={"model": MODEL, "messages": messages},
        timeout=300,
    )
    resposta = response.json()["choices"][0]["message"]["content"]

    agente["historico"].append({"role": "user", "content": mensagem})
    agente["historico"].append({"role": "assistant", "content": resposta})
    rotacionar(agente)
    return resposta


print("Agente COM memoria (max_historico=6: lembra das 3 ultimas trocas) — feche com Ctrl+D")
agente = criar_agente(
    "Você é um assistente amigável que responde em português.",
    max_historico=6,
)
try:
    while True:
        msg = input("Você: ").strip()
        if not msg:
            continue
        try:
            print("Agente:", conversar(agente, msg))
        except Exception as e:
            print(f"  [o servidor caiu: {e}]")
            print("  (tenta de novo — ele sobe sozinho)")
            continue
        print()
        print("historico:", json.dumps(agente["historico"], ensure_ascii=False))
        print()
except (EOFError, KeyboardInterrupt):
    print("\nTchau!")