# Petroleum Engineering AI Agent

A local-first petroleum engineering RAG web application. The system ingests engineering PDFs and images once, persists extracted text, figure notes, metadata, and ChromaDB vectors to disk, and answers only from retrieved evidence with document/page citations.

## Stack

- Frontend: Next.js, TypeScript, TailwindCSS
- Backend: FastAPI, Python 3.11
- Vector DB: ChromaDB persistent client
- Document processing: PyMuPDF, pdfplumber
- Embeddings: BGE-M3 (`BAAI/bge-m3`)
- Text model: Qwen3 8B via Ollama
- Vision model: Qwen2.5-VL via Ollama
- Graph rendering: Plotly

## Repository layout

```text
/frontend
/backend
/data
/data/raw
/data/extracted
/data/figures
/data/figure_notes
/data/vector_db
/data/metadata
```

## Quick start

First install and start Ollama, then pull the local models:

```bash
ollama pull qwen3:8b
ollama pull qwen2.5vl:7b
```

Copy the example environment file if you want to customize paths or model names:

```bash
cp .env.example .env
```

The default backend data path is the repository-root `./data` directory, so ChromaDB, metadata, raw files, extracted text, figures, and figure notes persist across restarts in the same location for Docker and non-Docker development.

### A. Docker Compose 실행

```bash
docker compose up --build
```

Open the web app at <http://localhost:3000>. The backend API is at <http://localhost:8000/api>.

Run the Docker-based E2E smoke test after the stack is up, or let it start the stack for you:

```bash
python scripts/e2e/run_e2e.py --use-compose
```

### B. Docker 없이 로컬 실행

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DATA_DIR=../data
export OLLAMA_BASE_URL=http://127.0.0.1:11434
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
corepack enable pnpm
pnpm install --no-frozen-lockfile --config.confirmModulesPurge=false
cp .env.example .env.local
pnpm dev
```

If your network blocks `registry.npmjs.org`, keep the pnpm workflow and set an approved registry mirror, for example:

```bash
pnpm install --no-frozen-lockfile --config.confirmModulesPurge=false --registry=https://registry.npmmirror.com
```

The committed pnpm workspace and minimal lockfile are intentionally valid for a first unrestricted `pnpm install`; pnpm will hydrate the full lock metadata on that first successful install.

Run the Docker-free E2E smoke test, which starts `uvicorn app.main:app --reload`, uploads a generated PDF, restarts the backend, and asks again without reuploading:

```bash
python scripts/e2e/run_local_e2e.py
```

Use the first-run checklist endpoint to confirm readiness:

```bash
curl http://127.0.0.1:8000/api/system/checklist
```

## Phase coverage

- Phase 1: PDF upload, extraction, ChromaDB persistence, and citation-grounded Q&A.
- Phase 2: Image and graph analysis through the Ollama vision model, plus PDF figure extraction and persistent figure notes.
- Phase 3: Plotly graph payload generation for browser rendering.
- Phase 4: Authentication and authorization extension points are documented but intentionally not implemented yet.

## Connect-AI reuse

This project reuses the local-first Ollama integration pattern and autonomous agent discipline from `wonseokjung/connect-ai` while excluding VS Code-specific extension code. The reusable ideas are implemented as web/backend modules: local model configuration, Ollama `/api/chat` calls, model listing, file ingestion, and structured agent prompts.
