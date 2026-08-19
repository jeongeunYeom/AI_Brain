from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"

# Load local environment variables from D:\AI_Brain\.env when present.
load_dotenv(ENV_FILE)


def _resolve_project_path(value: str | None, default: str) -> Path:
    path = Path(value or default).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_data_dir(value: str | None = None) -> Path:
    return _resolve_project_path(value or os.getenv("DATA_DIR"), "data")


def resolve_agent_workspace_dir(value: str | None = None) -> Path:
    return _resolve_project_path(
        value or os.getenv("AGENT_WORKSPACE_DIR"),
        "workspace",
    )


@dataclass(frozen=True)
class Settings:
    app_name: str = "Petroleum Engineering AI Agent"

    data_dir: Path = field(default_factory=resolve_data_dir)
    agent_workspace_dir: Path = field(default_factory=resolve_agent_workspace_dir)

    ollama_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434",
        )
    )

    text_model: str = field(
        default_factory=lambda: os.getenv(
            "TEXT_MODEL",
            "qwen3:8b",
        )
    )

    comparison_models: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            model.strip()
            for model in os.getenv(
                "COMPARISON_MODELS",
                "qwen3:8b,gemma4:latest",
            ).split(",")
            if model.strip()
        )
    )

    vision_model: str = field(
        default_factory=lambda: os.getenv(
            "VISION_MODEL",
            "qwen2.5vl:7b",
        )
    )

    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL",
            "BAAI/bge-m3",
        )
    )

    embedding_model_path: str | None = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_PATH") or None
    )

    collection_name: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_COLLECTION",
            "petroleum_knowledge",
        )
    )

    anonymized_telemetry: bool = field(
        default_factory=lambda: os.getenv("ANONYMIZED_TELEMETRY", "False").lower() == "true"
    )

    chunk_size: int = field(
        default_factory=lambda: int(
            os.getenv("CHUNK_SIZE", "1200")
        )
    )

    chunk_overlap: int = field(
        default_factory=lambda: int(
            os.getenv("CHUNK_OVERLAP", "180")
        )
    )

    top_k: int = field(
        default_factory=lambda: int(
            os.getenv("TOP_K", "10")
        )
    )

    similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("SIMILARITY_THRESHOLD", "0.10")
        )
    )

    ollama_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.getenv("OLLAMA_TIMEOUT_SECONDS", "600")
        )
    )

    ollama_temperature: float = field(
        default_factory=lambda: float(
            os.getenv("OLLAMA_TEMPERATURE", "0")
        )
    )

    aggregate_batch_size: int = field(
        default_factory=lambda: int(
            os.getenv("AGGREGATE_BATCH_SIZE", "500")
        )
    )

    aggregate_max_results: int = field(
        default_factory=lambda: int(
            os.getenv("AGGREGATE_MAX_RESULTS", "100")
        )
    )

    aggregate_max_per_document: int = field(
        default_factory=lambda: int(
            os.getenv("AGGREGATE_MAX_PER_DOCUMENT", "10")
        )
    )

    aggregate_context_max_chunks: int = field(
        default_factory=lambda: int(
            os.getenv("AGGREGATE_CONTEXT_MAX_CHUNKS", "24")
        )
    )

    aggregate_context_max_characters: int = field(
        default_factory=lambda: int(
            os.getenv("AGGREGATE_CONTEXT_MAX_CHARACTERS", "24000")
        )
    )

    figure_note_min_confidence: float = field(
        default_factory=lambda: float(
            os.getenv("FIGURE_NOTE_MIN_CONFIDENCE", "0.5")
        )
    )

    allow_legacy_figure_notes: bool = field(
        default_factory=lambda: os.getenv("ALLOW_LEGACY_FIGURE_NOTES", "0") == "1"
    )

    figure_analysis_max_vision_calls: int = field(
        default_factory=lambda: int(
            os.getenv("FIGURE_ANALYSIS_MAX_VISION_CALLS", "80")
        )
    )

    figure_page_render_fallback: bool = field(
        default_factory=lambda: os.getenv(
            "FIGURE_PAGE_RENDER_FALLBACK", "1"
        ) == "1"
    )

    figure_page_render_scale: float = field(
        default_factory=lambda: float(
            os.getenv("FIGURE_PAGE_RENDER_SCALE", "3.0")
        )
    )

    agent_max_file_bytes: int = field(
        default_factory=lambda: int(
            os.getenv("AGENT_MAX_FILE_BYTES", str(5 * 1024 * 1024))
        )
    )

    agent_max_read_characters: int = field(
        default_factory=lambda: int(
            os.getenv("AGENT_MAX_READ_CHARACTERS", "50000")
        )
    )

    agent_python_timeout_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("AGENT_PYTHON_TIMEOUT_SECONDS", "30")
        )
    )

    agent_python_max_code_characters: int = field(
        default_factory=lambda: int(
            os.getenv("AGENT_PYTHON_MAX_CODE_CHARACTERS", "30000")
        )
    )

    agent_python_output_characters: int = field(
        default_factory=lambda: int(
            os.getenv("AGENT_PYTHON_OUTPUT_CHARACTERS", "20000")
        )
    )

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
    def figure_candidates_dir(self) -> Path:
        return self.data_dir / "figure_candidates"

    @property
    def figure_analysis_inputs_dir(self) -> Path:
        return self.data_dir / "figure_analysis_inputs"

    @property
    def vector_db_dir(self) -> Path:
        return self.data_dir / "vector_db"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def evaluation_dir(self) -> Path:
        return self.data_dir / "evaluation"

    @property
    def agent_runs_dir(self) -> Path:
        return self.data_dir / "agent_runs"

    @property
    def agent_conversations_dir(self) -> Path:
        return self.data_dir / "agent_conversations"

    @property
    def agent_backups_dir(self) -> Path:
        return self.data_dir / "agent_backups"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()

    for directory in [
        settings.raw_dir,
        settings.extracted_dir,
        settings.figures_dir,
        settings.figure_notes_dir,
        settings.figure_candidates_dir,
        settings.figure_analysis_inputs_dir,
        settings.vector_db_dir,
        settings.metadata_dir,
        settings.evaluation_dir,
        settings.agent_runs_dir,
        settings.agent_conversations_dir,
        settings.agent_backups_dir,
        settings.agent_workspace_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return settings
