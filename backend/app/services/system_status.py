import tempfile
from pathlib import Path
from typing import Any

import chromadb

from app.core.config import Settings
from app.core.error_mapping import ExternalServiceError
from app.services.ollama import OllamaClient


def count_documents(metadata_dir: Path) -> int:
    return sum(1 for path in metadata_dir.glob("*.json") if path.is_file())


def data_dir_writable(data_dir: Path) -> bool:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=data_dir, prefix="write-check-", delete=True) as handle:
            handle.write(b"ok")
            handle.flush()
        return True
    except OSError:
        return False


def chroma_status(settings: Settings) -> tuple[bool, int, str | None]:
    try:
        client = chromadb.PersistentClient(path=str(settings.vector_db_dir))
        collection = client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return True, collection.count(), None
    except Exception as exc:  # noqa: BLE001 - returned as a diagnostic checklist item
        return False, 0, str(exc)


async def build_checklist(settings: Settings, ollama: OllamaClient) -> dict[str, Any]:
    ollama_running = False
    models: list[str] = []
    ollama_error: str | None = None
    try:
        models = await ollama.list_models()
        ollama_running = True
    except ExternalServiceError as exc:
        ollama_error = exc.user_message

    chroma_ok, chunk_count, chroma_error = chroma_status(settings)
    text_model_installed = settings.text_model in models
    vision_model_installed = settings.vision_model in models

    checks = {
        "ollama": {
            "ok": ollama_running,
            "base_url": settings.ollama_base_url,
            "message": "Ollama is reachable." if ollama_running else ollama_error,
        },
        "text_model": {
            "ok": text_model_installed,
            "model": settings.text_model,
            "message": "Installed." if text_model_installed else f"Run `ollama pull {settings.text_model}`.",
        },
        "vision_model": {
            "ok": vision_model_installed,
            "model": settings.vision_model,
            "message": "Installed." if vision_model_installed else f"Run `ollama pull {settings.vision_model}`.",
        },
        "data_dir": {
            "ok": (writable := data_dir_writable(settings.data_dir)),
            "path": str(settings.data_dir),
            "message": "Writable." if writable else "Data directory is not writable.",
        },
        "chroma": {
            "ok": chroma_ok,
            "path": str(settings.vector_db_dir),
            "message": "Accessible." if chroma_ok else chroma_error,
        },
    }
    return {
        "ok": all(check["ok"] for check in checks.values()),
        "checks": checks,
        "models": models,
        "knowledge_base": {
            "documents": count_documents(settings.metadata_dir),
            "chunks": chunk_count,
        },
    }
