+++
title = "Tool Calling no MiniCPM5-1B: lições aprendidas migrando do llama.cpp pro vLLM"
date = "2026-05-28"
tags = ["llm", "tool-calling", "vllm", "minicpm5", "llama.cpp", "deploy"]
+++

# Tool Calling no MiniCPM5-1B: lições aprendidas migrando do llama.cpp pro vLLM

## O problema

O MiniCPM5-1B (OpenBMB, 1.08B params) é um modelinho compacto com suporte nativo a **tool calling via XML** e **modo think/no-think**. Na teoria, perfeito pra fleet local numa RTX 3050 6GB.

Na prática, rodar no llama.cpp revelou dois bugs:

1. **Bug #1**: O filtro Jinja2 `|min` não está registrado no minja (engine de templates do llama.cpp). Template quebrado na hora de renderizar.
2. **Bug #2**: O autoparser do llama.cpp infere errado os boundaries do formato `<param name="X">value</param>` — `arg_name_suffix` vira `''` em vez de `'">'`, e `arg_value_prefix` vira `'">'` em vez de `''`. Resultado: error 500 na qualquer tool call.

Conclusão: **structured tool calling no MiniCPM5-1B é inutilizável no llama.cpp** (issue #23781, sem fix até o momento).

## A solução: vLLM

O PR [#43175](https://github.com/vllm-project/vllm/pull/43175) adicionou o parser `minicpm5xml` ao vLLM, com suporte completo ao formato XML de tool calling do MiniCPM5:

- `<function name="X"><param name="Y">value</param></function>` → parseado corretamente em `tool_calls` estruturados
- Suporte a CDATA, tipos não-string, streaming, validação de schema
- Mergeado em mai/2026

### Configuração

```bash
# venv dedicado (Python 3.14 + vLLM 0.21.0 + parser do git main)
python3 -m venv ~/.vllm-venv
~/.vllm-venv/bin/pip install vllm

# Parser minicpm5xml ainda não está na release 0.21.0 — baixar do git main
curl -sL "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/tool_parsers/minicpm5xml_tool_parser.py" \
  -o ~/.vllm-venv/lib/python3.14/site-packages/vllm/tool_parsers/minicpm5xml_tool_parser.py

# Registrar no __init__.py dos tool_parsers
# Adicionar: "minicpm5": ("minicpm5xml_tool_parser", "MiniCPM5XMLToolParser")
```

### Wrapper script (`~/.local/bin/vllm-minicpm5-1b`)

```bash
#!/usr/bin/env bash
# Workaround: FlashInfer JIT quebra em Python 3.14 (path mismatch)
export VLLM_USE_FLASHINFER_SAMPLER=0
exec ~/.vllm-venv/bin/python -m vllm.entrypoints.openai.api_server "$@"
```

### Flag essenciais

```bash
~/.local/bin/vllm-minicpm5-1b \
  --model openbmb/MiniCPM5-1B \
  --served-model-name minicpm5-1b \
  --port $PORT \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tool-call-parser minicpm5 \       # ← parser de XML tool calls
  --enable-auto-tool-choice \          # ← modelo decide quando chamar
  --max-model-len 8192 \              # ← 9.3x concurrency na RTX 3050 6GB
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 2 \
  --dtype half \                       # ← FP16 (modelo é BF16, cast automático)
  --enforce-eager                     # ← evita crash de TorchInductor
```

### Métricas na RTX 3050 6GB

| Métrica | Valor |
|---------|-------|
| Pesos (BF16 safetensors) | ~2.09 GB |
| KV cache | ~1.74 GB |
| VRAM total | ~3.83 GB |
| Concorrência (8K ctx) | 9.3x |
| Throughput | ~76 tok/s |

## ⚠️ A armadilha do prompt JSON

Tool calling **não é a mesma coisa que JSON mode**. Isso parece óbvio, mas é fácil cair nessa armadilha:

### ❌ Prompt problemático

```
You are a useful assistant that only answers concisely in JSON format,
within a Markdown-styled code block. You WILL call available tools...
```

Esse prompt instrui o modelo a **gerar JSON manualmente** em vez de usar o formato XML nativo (`<function name="..."><param name="...">`). O parser `minicpm5` só intercepta tags XML — se o modelo emite JSON no texto, tudo cai no `content` como texto cru, sem `tool_calls` estruturado.

Resultado: o modelo cospe um monólogo de 3500+ tokens "raciocinando" sobre quais tools chamar, seguido de um bloco JSON dentro de code block — e nenhum `tool_call` na resposta da API.

### ✅ Prompt correto

```
You are a helpful assistant. When the user asks something that can be
answered using the available tools, call the appropriate tool(s).
When you have enough information to answer directly, do so concisely.
Always respond in the user's language.
```

**Não instrua formato.** O parser cuida disso. O `--tool-call-parser minicpm5` converte XML em `tool_calls` antes do texto chegar no `content`. O `--enable-auto-tool-choice` + `tool_choice: auto` diz pro modelo *quando* chamar tools. *Como* chamar é responsabilidade do parser.

### Regra prática

> **Nunca instrua o modelo a emitir JSON quando ele tem tool calling nativo.**

Isso vale pra qualquer modelo com parser dedicado (Hermes, Qwen3 XML, MiniCPM5, etc.). O formato de saída é determinado pelo parser, não pelo prompt.

## Caveats

### FlashInfer JIT quebra em Python 3.14

O FlashInfer tenta compilar CUDA kernels JIT e hardcoded o path pra `python3.12`. Solução: `VLLM_USE_FLASHINFER_SAMPLER=0`. Impacto em 1B params com max 2 seqs: **zero mensurável**. O gargalo é memoria/bandwidth, não sampling.

### `--reasoning-parser minicpm5` crasha no vLLM 0.21.0

O parser de reasoning ainda não está na release. Resultado: `reasoning_content` não é populado na resposta — as tags `<think>` aparecem dentro do `content`. Funcionalmente o raciocínio está lá, mas não separado em campo dedicado.

Solução pra quando o parser chegar: atualizar vLLM ou baixar o arquivo do git main (mesma abordagem do `minicpm5xml_tool_parser.py`).

### Contexto limitado a 8K

Na RTX 3050 6GB com `gpu-memory-utilization 0.70`, 8K tokens dá 9.3x concurrency. 32K seria ~2.3x — marginal. Preferimos o headroom.

## Referências

- [vLLM PR #43175 — MiniCPM5 XML tool call parser](https://github.com/vllm-project/vllm/pull/43175)
- [llama.cpp issue #23781 — MiniCPM5 tool calling bug](https://github.com/ggerganov/llama.cpp/issues/23781)
- [MiniCPM5-1B no HuggingFace](https://huggingface.co/openbmb/MiniCPM5-1B)