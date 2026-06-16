# Architecture

## Agent workflow

1. Uploads are saved under `/data/raw` with a SHA-256 prefix.
2. Metadata in `/data/metadata/<sha>.json` prevents duplicate processing on future uploads or server restarts.
3. Extracted page text and chunk JSON are written to `/data/extracted`.
4. PDF images are written to `/data/figures` and Qwen2.5-VL notes are written to `/data/figure_notes`.
5. BGE-M3 embeddings are upserted into persistent ChromaDB under `/data/vector_db`.
6. Questions run vector search first, then Qwen3 receives only retrieved context and must refuse unsupported answers.

## Connect-AI reuse boundary

The project intentionally avoids VS Code extension APIs, command tags, and webviews from Connect-AI. Reused concepts are:

- local Ollama `/api/chat` text/vision calls;
- local model discovery through `/api/tags`;
- disk-first knowledge persistence;
- agent prompts that require action grounded in local context.

## Phase 4 extension point

Add auth middleware around the FastAPI router and a user-scoped Chroma collection name such as `petroleum_knowledge_<tenant_id>`. The current code keeps a single-user local-first deployment model for Phases 1-3.
