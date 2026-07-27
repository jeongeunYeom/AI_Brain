from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.core.config import Settings, get_settings

router = APIRouter()

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
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
    if suffix not in _MEDIA_TYPES:
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


def _serve_image(root: Path, filename: str, *, label: str) -> FileResponse:
    candidate, suffix = _safe_image_file(root, filename, label=label)
    return FileResponse(
        path=candidate,
        media_type=_MEDIA_TYPES[suffix],
        filename=candidate.name,
    )


@router.get("/figures/{filename}", include_in_schema=False)
def get_figure_image(
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve one extracted figure without exposing arbitrary filesystem paths."""
    return _serve_image(settings.figures_dir, filename, label="figure")


@router.get("/figure-previews/{filename}", include_in_schema=False)
def get_figure_preview(
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve one generated display preview from its dedicated cache root."""
    return _serve_image(
        settings.data_dir / "figure_display_previews",
        filename,
        label="figure preview",
    )
