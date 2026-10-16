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


def conversar(agente, mensagem):
    messages = [{"role": "system", "content": agente["instrucoes"]}]
    limite = agente["max_historico"]
    for msg in agente["historico"][-limite:]:
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
    return resposta


print("Agente COM memoria — feche com Ctrl+D")
agente = criar_agente("Você é um assistente amigável que responde em português.")
try:
    while True:
        msg = input("Você: ").strip()
        if not msg:
            continue
        print("Agente:", conversar(agente, msg))
        print()
        print("historico:", json.dumps(agente["historico"], ensure_ascii=False))
        print()
except (EOFError, KeyboardInterrupt):
    print("\nTchau!")