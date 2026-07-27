from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.figure_files import _serve_image, router as figure_files_router
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
app.include_router(figure_files_router, prefix="/api")


# Backward-compatible direct-call helpers retained for tests and local scripts.
def get_figure_image(filename: str) -> FileResponse:
    return _serve_image(settings.figures_dir, filename, label="figure")


def get_figure_preview(filename: str) -> FileResponse:
    return _serve_image(
        settings.data_dir / "figure_display_previews",
        filename,
        label="figure preview",
    )
