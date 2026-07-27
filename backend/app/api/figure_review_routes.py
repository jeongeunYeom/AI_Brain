from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.config import Settings, get_settings
from app.models.schemas import (
    FigureReviewUpdateRequest,
    FigureRotationUpdateRequest,
)
from app.services.figure_review import FigureReviewError, FigureReviewService

router = APIRouter(prefix="/review", tags=["figure-review"])


def _service(settings: Settings) -> FigureReviewService:
    return FigureReviewService(settings)


def _http_error(exc: FigureReviewError) -> HTTPException:
    message = str(exc)
    status_code = 404 if "not found" in message.lower() else 400
    return HTTPException(status_code=status_code, detail=message)


@router.get("/summary")
async def figure_review_summary(
    settings: Settings = Depends(get_settings),
) -> dict:
    return _service(settings).summary()


@router.get("/candidates")
async def figure_review_candidates(
    document_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    page: int | None = Query(default=None, ge=1),
    q: str | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        return _service(settings).list_candidates(
            document_id=document_id,
            status=status,
            page=page,
            query=q,
            offset=offset,
            limit=limit,
        )
    except FigureReviewError as exc:
        raise _http_error(exc) from exc


@router.get("/candidates/{candidate_id}")
async def figure_review_candidate(
    candidate_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        return _service(settings).get_candidate(candidate_id)
    except FigureReviewError as exc:
        raise _http_error(exc) from exc


@router.patch("/candidates/{candidate_id}")
async def update_figure_review_candidate(
    candidate_id: str,
    request: FigureReviewUpdateRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        return _service(settings).update_candidate(
            candidate_id,
            request.model_dump(exclude_unset=True),
        )
    except FigureReviewError as exc:
        raise _http_error(exc) from exc


@router.post("/candidates/{candidate_id}/rotation")
async def update_figure_review_rotation(
    candidate_id: str,
    request: FigureRotationUpdateRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        return _service(settings).set_rotation(
            candidate_id,
            rotation=request.rotation,
            pdf_crop_rotation=request.pdf_crop_rotation,
            enhance=request.enhance,
            regenerate=request.regenerate,
        )
    except FigureReviewError as exc:
        raise _http_error(exc) from exc


@router.post("/candidates/{candidate_id}/preview")
async def regenerate_figure_review_preview(
    candidate_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        return _service(settings).regenerate_preview(candidate_id)
    except FigureReviewError as exc:
        raise _http_error(exc) from exc


@router.get(
    "/candidates/{candidate_id}/preview-image",
    include_in_schema=False,
)
async def figure_review_preview_image(
    candidate_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    try:
        result = _service(settings).preview_file(candidate_id)
    except FigureReviewError as exc:
        raise _http_error(exc) from exc

    return FileResponse(
        path=result.path,
        media_type="image/png",
        filename=result.path.name,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/audit")
async def figure_review_audit(
    limit: int = Query(default=100, ge=1, le=500),
    settings: Settings = Depends(get_settings),
) -> list[dict]:
    return _service(settings).recent_audit(limit)
