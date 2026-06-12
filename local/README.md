# Aula 07 — Local

Versão local do Colab da Aula 07 (Múltiplas Ferramentas e MCP).

## Estrutura

```
local/
├── aula07-local.org          ← Arquivo principal (abra no Emacs/org-mode)
├── math-server-python/       ← Servidor MCP em Python (FastMCP)
│   ├── pyproject.toml
│   ├── math_server.py        ← Servidor stdio (add, multiply)
│   ├── string_server.py      ← Servidor stdio (reverse_string, count_words)
│   ├── server_http.py        ← Mesmas tools, mas via HTTP (porta 11111)
│   └── client_http.py        ← Cliente MCP HTTP (LangChain + Ollama)
└── math-server-rust/         ← Servidor MCP em Rust (mesmas tools, binário compilado)
    ├── Cargo.toml
    ├── src/main.rs
    └── target/release/math-server-mcp  ← 2.1MB, sem dependências
```

## Como rodar

### Servidor Python (stdio)

```bash
cd math-server-python
uv run math_server.py       # servidor de matemática
uv run string_server.py     # servidor de strings
```

### Servidor Python (HTTP)

```bash
cd math-server-python
uv run server_http.py       # http://localhost:11111/mcp
# Em outro terminal:
uv run client_http.py       # cliente que conecta via HTTP
```

### Servidor Rust (stdio)

```bash
# Já compilado em release:
./math-server-rust/target/release/math-server-mcp

# Para recompilar:
cd math-server-rust && cargo build --release
```

### Arquivo org

Abra `aula07-local.org` no Emacs. Cada bloco de código pode ser executado com `C-c C-c` (org-babel).

## Requisitos

- **Ollama** com modelo `gemma4:e2b-it-qat`
- **uv** (gerenciador de pacotes Python)
- **Rust/Cargo** (apenas para recompilar o servidor Rust)
- **Emacs** com org-mode (para executar o arquivo .org)