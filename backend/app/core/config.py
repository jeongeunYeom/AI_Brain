from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_data_dir(value: str | None = None) -> Path:
    raw_value = value or os.getenv("DATA_DIR") or "data"
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    app_name: str = "Petroleum Engineering AI Agent"
    data_dir: Path = field(default_factory=resolve_data_dir)
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    text_model: str = field(default_factory=lambda: os.getenv("TEXT_MODEL", "qwen3:8b"))
    vision_model: str = field(default_factory=lambda: os.getenv("VISION_MODEL", "qwen2.5vl:7b"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "petroleum_knowledge"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1200")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "180")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "10")))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.35")))
    ollama_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600")))
    ollama_temperature: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TEMPERATURE", "0")))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def figures_dir(self) -> Path:
        return self.data_dir / "figures"

    @property
    def figure_notes_dir(self) -> Path:
        return self.data_dir / "figure_notes"

    @property
    def vector_db_dir(self) -> Path:
        return self.data_dir / "vector_db"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    for directory in [
        settings.raw_dir,
        settings.extracted_dir,
        settings.figures_dir,
        settings.figure_notes_dir,
        settings.vector_db_dir,
        settings.metadata_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return settings
