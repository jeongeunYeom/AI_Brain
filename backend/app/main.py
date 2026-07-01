from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

ALLOWED_FIGURE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def _safe_image_file(root: Path, filename: str, *, label: str) -> tuple[Path, str]:
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
    ):
        raise HTTPException(status_code=400, detail=f"Invalid {label} filename")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_FIGURE_SUFFIXES:
        raise HTTPException(status_code=404, detail=f"{label.title()} not found")

    resolved_root = root.resolve()
    candidate = (resolved_root / filename).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {label} path") from exc

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"{label.title()} not found")
    return candidate, suffix


@app.get("/api/figures/{filename}", include_in_schema=False)
def get_figure_image(filename: str) -> FileResponse:
    """Serve one extracted figure without exposing arbitrary filesystem paths."""
    candidate, suffix = _safe_image_file(
        settings.figures_dir,
        filename,
        label="figure",
    )

    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    return FileResponse(
        path=candidate,
        media_type=media_types[suffix],
        filename=candidate.name,
    )


@app.get("/api/figure-previews/{filename}", include_in_schema=False)
def get_figure_preview(filename: str) -> FileResponse:
    """Serve one generated display preview from its dedicated cache root."""
    candidate, suffix = _safe_image_file(
        settings.data_dir / "figure_display_previews",
        filename,
        label="figure preview",
    )
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    return FileResponse(
        path=candidate,
        media_type=media_types[suffix],
        filename=candidate.name,
    )
