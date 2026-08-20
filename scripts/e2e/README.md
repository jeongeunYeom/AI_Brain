# E2E smoke tests

Run after Ollama is installed and the required models are pulled:

```bash
ollama pull qwen3:8b
ollama pull qwen2.5vl:7b
```

## Docker Compose E2E

```bash
python scripts/e2e/run_e2e.py --use-compose
```

The Docker-based script creates a temporary PDF and graph image, uploads them through the API, restarts the backend when `--use-compose` is set, verifies `/data/vector_db` files are still present, asks a question without reuploading the PDF, checks citation document/page metadata, and validates the vision endpoint.

Use `--skip-vision` when the vision model is not installed and you only want to validate Phase 1 RAG persistence.

## Docker-free local E2E

```bash
python scripts/e2e/run_local_e2e.py
```

This script starts the backend with `uvicorn app.main:app --reload`, verifies `/api/health`, uploads a generated PDF, asks a question, confirms `/data/vector_db` files exist, restarts the backend process, asks again without reuploading, and asserts the same source document/page is returned.

## Ollama-down / missing-model error check

To verify the friendly 503 error path, run a second backend instance with `OLLAMA_BASE_URL` pointing to an unused port and set `E2E_BAD_OLLAMA_API` to that instance before running the Docker-based script. The script will assert the error payload contains a user-facing Korean message under `detail.message`.

## MRST/CO₂ Agent E2E

```bash
python scripts/e2e/run_agent_mrst_e2e.py
```

This test does not require Ollama or Docker. It starts FastAPI with temporary
data/workspace directories and validates the complete approval-based Agent flow:

1. create a representative Johansen `srco2` CSV through the Agent API;
2. plan and approve the MRST/CO₂ analysis;
3. validate the summary CSV, PNG chart, Markdown report, previews, and downloads;
4. confirm that the input CSV was read but not modified;
5. restart FastAPI and restore the conversation, tasks, validation results, and outputs.
