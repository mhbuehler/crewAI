# XPU Validation Scripts

Manual hardware checks for CrewAI on Intel GPU.

## Tests

| Script | Checks |
|---|---|
| [xpu_sentence_transformer_smoke.py](xpu_sentence_transformer_smoke.py) | CrewAI embeddings run on PyTorch XPU with valid vectors |
| [xpu_ollama_crew_e2e.py](xpu_ollama_crew_e2e.py) | `Crew.kickoff()` succeeds and Ollama reports model loaded in VRAM |
| [xpu_rag_ollama_e2e.py](xpu_rag_ollama_e2e.py) | RAG ingestion/query embeddings run on XPU and final answer is correct |

All scripts fail if they cannot prove GPU execution.

## Requirements

- Linux with Intel GPU driver
- Python 3.10-3.13
- PyTorch XPU wheel
- For E2E tests: running Ollama service with model installed

The normal repo dev environment installs CPU PyTorch. Use a separate XPU env.

## Setup

From repo root:

```bash
python3 -m venv .venv-xpu
source .venv-xpu/bin/activate
python -m pip install --upgrade pip

python -m pip install --no-cache-dir 'torch==2.12.0+xpu' \
  --index-url https://download.pytorch.org/whl/xpu

python -m pip install --no-cache-dir sentence-transformers chromadb

python -m pip install --no-cache-dir --no-deps \
  -e ./lib/crewai-core \
  -e ./lib/cli \
  -e ./lib/crewai
```

Verify XPU:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("XPU runtime:", torch.version.xpu)
print("XPU available:", torch.xpu.is_available())
print("XPU count:", torch.xpu.device_count())
if torch.xpu.is_available():
    print("Device:", torch.xpu.get_device_name(0))
PY
```

## Ollama Service (E2E only)

```bash
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="qwen2.5:0.5b"
export SENTENCE_TRANSFORMER_MODEL="all-MiniLM-L6-v2"
```

API Checks:

```bash
curl -fsS "${OLLAMA_HOST}/api/tags"
curl -fsS "${OLLAMA_HOST}/api/ps"
```

## Run Tests

```bash
python scripts/xpu_sentence_transformer_smoke.py
python scripts/xpu_ollama_crew_e2e.py
python scripts/xpu_rag_ollama_e2e.py
```

Saved sample outputs:

- [xpu_sentence_transformer_smoke.json](xpu_sentence_transformer_smoke.json)
- [xpu_ollama_crew_e2e.json](xpu_ollama_crew_e2e.json)
- [xpu_rag_ollama_e2e.json](xpu_rag_ollama_e2e.json)

## Pass Criteria

Smoke test:

- Model parameters and forward tensors are on `xpu:*`
- Embeddings are valid and non-degenerate
- XPU memory allocation is nonzero

E2E-1:

- `Crew.kickoff()` returns required facts
- Ollama `/api/ps` reports `size_vram > 0`

E2E-2:

- Retrieved context contains source-only facts
- Final answer contains required facts
- Ingestion and query embedding calls show XPU tensors
- Ollama `/api/ps` reports `size_vram > 0`
- Note: With small models (e.g. `qwen2.5:0.5b`), `knowledge_search_query` may 
include hallucinations; this is okay if retrieved context and final output
are correct

## Common Issues

- `torch.xpu.is_available() returned False`: wrong PyTorch wheel or driver issue
- Import mismatch across local editable packages: reinstall all three `-e` packages shown above
- `No space left on device`: move caches/temp dirs to a larger filesystem
- Model missing from `/api/tags`: pull/install model in Ollama
- `size_vram == 0`: run rejected as CPU-only
