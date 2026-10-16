import requests

BASE = "http://localhost:12434/v1/chat/completions"
MODEL = "bonsai2-4b"


def perguntar(mensagem):
    response = requests.post(
        BASE,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": mensagem}],
        },
        timeout=300,
    )
    return response.json()["choices"][0]["message"]["content"]


print("Agente SEM memoria — feche com Ctrl+D")
try:
    while True:
        msg = input("Você: ").strip()
        if not msg:
            continue
        print("Agente:", perguntar(msg))
        print()
except (EOFError, KeyboardInterrupt):
    print("\nTchau!")
