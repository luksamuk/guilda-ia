# Notebooks da Guilda de IA

Jupyter notebooks para uso no Google Colab.

## Como usar

Clique no badge "Open in Colab" no notebook ou na página da guilda para abrir uma cópia editável no Google Colab.

## Notebooks disponíveis

| Arquivo | Aula | Descrição |
|---------|-------|-----------|
| `aula03_python_minimo.ipynb` | Semana 03 | Python mínimo: tipos, listas, dicionários, funções com estado, strings, JSON, loops/condicionais |
| `poc_llm_local_colab.ipynb` | Experimental (legado) | POC: inferência local de Qwen3-4B via llama-cpp-python |
| `poc_ollama_qwen35_colab.ipynb` | Experimental | POC: inferência local de Qwen3.5-4B via Ollama (API REST HTTP) |
| `poc_ollama_gemma4_colab.ipynb` | Experimental | POC: inferência local de Gemma 4 E2B via Ollama (API REST HTTP) |
| `poc_ollama_qwen35_4b_colab.ipynb` | Experimental | POC: Qwen 3.5 4B via Ollama no Colab (warm up ~141s) |
| `poc_ollama_qwen35_9b_colab.ipynb` | Experimental | POC: Qwen 3.5 9B via Ollama no Colab (minimalista) |
| `poc_ollama_gemma4_e4b_colab.ipynb` | Experimental | POC: Gemma 4 E4B via Ollama no Colab (minimalista) |

## Estrutura dos notebooks

- Cada notebook é **auto-contido**: não depende de arquivos externos
- Seguem a filosofia **funções + dicionários** (sem classes) para iniciantes
- Incluem exercícios marcados com 🧪
- O badge "Open in Colab" no topo aponta para `github/luksamuk/guilda-ia`

## Badges Colab

Para adicionar badge a um novo notebook, inclua no primeiro cell markdown:

```markdown
<a href="https://colab.research.google.com/github/luksamuk/guilda-ia/blob/main/notebooks/NOME.ipynb" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
```

Os badges na página da guilda (`index.org`) usam o mesmo formato, com `style="height:1.2em"` para ficarem proporcionais na tabela.