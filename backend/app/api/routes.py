from pathlib import Path
from functools import lru_cache
import json

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.config import Settings, get_settings
from app.core.error_mapping import ExternalServiceError
from app.core.errors import to_http_exception
from app.models.schemas import (
    ChatCompareRequest,
    ChatCompareResponse,
    ChatRequest,
    ChatResponse,
    FigureReviewUpdateRequest,
    FigureRotationUpdateRequest,
    PlotRequest,
    PlotResponse,
    UploadResponse,
    VisionResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.jobs import create_job, get_job, update_job
from app.services.figure_review import FigureReviewError, FigureReviewService
from app.services.ollama import OllamaClient
from app.services.plots import build_plot
from app.services.qa import QAService
from app.services.system_status import build_checklist
from app.services.vector_store import VectorStore

router = APIRouter()


@lru_cache(maxsize=1)
def cached_ollama() -> OllamaClient:
    return OllamaClient(get_settings())


@lru_cache(maxsize=1)
def cached_vector_store() -> VectorStore:
    return VectorStore(get_settings())


def get_ollama() -> OllamaClient:
    return cached_ollama()


def get_vector_store() -> VectorStore:
    return cached_vector_store()


@router.get("/health")
async def health(settings: Settings = Depends(get_settings), ollama: OllamaClient = Depends(get_ollama)) -> dict:
    models: list[str] = []
    try:
        models = await ollama.list_models()
    except ExternalServiceError:
        models = []
    return {
        "status": "ok",
        "app": settings.app_name,
        "data_dir": str(settings.data_dir),
        "ollama_base_url": settings.ollama_base_url,
        "models": models,
    }


@router.get("/system/checklist")
async def system_checklist(settings: Settings = Depends(get_settings), ollama: OllamaClient = Depends(get_ollama)) -> dict:
    return await build_checklist(settings, ollama)


@router.post("/jobs", status_code=201)
async def start_job() -> dict:
    return {"job_id": create_job()}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    analyze_figures: bool = True,
    job_id: str | None = Query(default=None),
    settings: Settings = Depends(get_settings),
    ollama: OllamaClient = Depends(get_ollama),
    vector_store: VectorStore = Depends(get_vector_store),
) -> UploadResponse:
    processor = DocumentProcessor(settings, ollama)
    try:
        update_job(job_id, 1, "[1/6] PDF 저장 중")
        saved_path, digest = await processor.save_upload(file)
        existing = processor.load_existing(digest)
        if existing:
            update_job(job_id, 6, f"완료: 기존 문서 재사용 ({existing.chunks} chunks)", status="completed")
            return UploadResponse(document=existing, skipped=True)
        update_job(job_id, 2, "[2/6] 텍스트 추출 중")
        record, chunks = await processor.extract(saved_path, digest, analyze_figures=analyze_figures)
        update_job(job_id, 3, "[3/6] 이미지 추출 중")
        update_job(job_id, 4, "[4/6] BGE-M3 embedding model 로드 중")
        update_job(job_id, 5, "[5/6] Embedding 생성 중")
        update_job(job_id, 6, "[6/6] ChromaDB 저장 중")
        vector_store.add_chunks(chunks)
        update_job(job_id, 6, f"Processed {record.chunks} chunks", status="completed")
        return UploadResponse(document=record, skipped=False)
    except ValueError as exc:
        update_job(job_id, 0, "업로드 실패", status="failed", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ExternalServiceError as exc:
        update_job(job_id, 0, "Ollama/외부 서비스 오류", status="failed", error=exc.user_message)
        raise to_http_exception(exc) from exc


@router.get("/documents")
async def list_documents(settings: Settings = Depends(get_settings)) -> list[dict]:
    records = []
    for path in sorted(settings.metadata_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _allowed_answer_models(settings: Settings) -> list[str]:
    return list(
        dict.fromkeys(
            [
                settings.text_model,
                *settings.comparison_models,
            ]
        )
    )


def _validate_answer_models(
    settings: Settings,
    models: list[str],
) -> list[str]:
    allowed = _allowed_answer_models(settings)
    normalized = list(
        dict.fromkeys(
            str(model or "").strip()
            for model in models
            if str(model or "").strip()
        )
    )
    unsupported = [
        model
        for model in normalized
        if model not in allowed
    ]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported answer model.",
                "unsupported": unsupported,
                "allowed": allowed,
            },
        )
    return normalized


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    ollama: OllamaClient = Depends(get_ollama),
    vector_store: VectorStore = Depends(get_vector_store),
) -> ChatResponse:
    selected_model = request.model or settings.text_model
    _validate_answer_models(settings, [selected_model])
    service = QAService(settings, vector_store, ollama)
    try:
        return await service.answer(
            request.question,
            request.top_k,
            model=selected_model,
            benchmark_id=request.benchmark_id,
        )
    except ExternalServiceError as exc:
        raise to_http_exception(exc) from exc


@router.post(
    "/chat/compare",
    response_model=ChatCompareResponse,
)
async def compare_chat_models(
    request: ChatCompareRequest,
    settings: Settings = Depends(get_settings),
    ollama: OllamaClient = Depends(get_ollama),
    vector_store: VectorStore = Depends(get_vector_store),
) -> ChatCompareResponse:
    requested_models = (
        request.models
        if request.models is not None
        else list(settings.comparison_models)
    )
    selected_models = _validate_answer_models(
        settings,
        requested_models,
    )
    if len(selected_models) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Comparison requires at least two "
                "configured answer models."
            ),
        )

    installed = set(await ollama.list_models())
    missing = [
        model
        for model in selected_models
        if model not in installed
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Comparison model is not installed in Ollama.",
                "missing": missing,
            },
        )

    service = QAService(settings, vector_store, ollama)
    try:
        return await service.compare(
            request.question,
            selected_models,
            request.top_k,
        )
    except ExternalServiceError as exc:
        raise to_http_exception(exc) from exc


@router.post("/vision/analyze", response_model=VisionResponse)
async def analyze_image(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    ollama: OllamaClient = Depends(get_ollama),
) -> VisionResponse:
    suffix = Path(file.filename or "image.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail="Only PNG/JPG images are supported for vision analysis.")
    content = await file.read()
    target = settings.figures_dir / Path(file.filename or "image.png").name
    target.write_bytes(content)
    try:
        analysis = await ollama.describe_image(target)
    except ExternalServiceError as exc:
        raise to_http_exception(exc) from exc
    (settings.figure_notes_dir / f"{target.stem}.md").write_text(analysis, encoding="utf-8")
    return VisionResponse(filename=target.name, analysis=analysis)


@router.post("/plots", response_model=PlotResponse)
async def create_plot(request: PlotRequest) -> PlotResponse:
    return PlotResponse(figure=build_plot(request))

def _review_service(settings: Settings) -> FigureReviewService:
    return FigureReviewService(settings)


def _review_http_error(exc: FigureReviewError) -> HTTPException:
    message = str(exc)
    status_code = 404 if "not found" in message.lower() else 400
    return HTTPException(status_code=status_code, detail=message)


@router.get("/review/summary")
async def figure_review_summary(
    settings: Settings = Depends(get_settings),
) -> dict:
    return _review_service(settings).summary()


@router.get("/review/candidates")
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
        return _review_service(settings).list_candidates(
            document_id=document_id,
            status=status,
            page=page,
            query=q,
            offset=offset,
            limit=limit,
        )
    except FigureReviewError as exc:
        raise _review_http_error(exc) from exc


@router.get("/review/candidates/{candidate_id}")
async def figure_review_candidate(
    candidate_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        return _review_service(settings).get_candidate(candidate_id)
    except FigureReviewError as exc:
        raise _review_http_error(exc) from exc


@router.patch("/review/candidates/{candidate_id}")
async def update_figure_review_candidate(
    candidate_id: str,
    request: FigureReviewUpdateRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        return _review_service(settings).update_candidate(
            candidate_id,
            request.model_dump(exclude_unset=True),
        )
    except FigureReviewError as exc:
        raise _review_http_error(exc) from exc


@router.post("/review/candidates/{candidate_id}/rotation")
async def update_figure_review_rotation(
    candidate_id: str,
    request: FigureRotationUpdateRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        return _review_service(settings).set_rotation(
            candidate_id,
            rotation=request.rotation,
            pdf_crop_rotation=request.pdf_crop_rotation,
            enhance=request.enhance,
            regenerate=request.regenerate,
        )
    except FigureReviewError as exc:
        raise _review_http_error(exc) from exc


@router.post("/review/candidates/{candidate_id}/preview")
async def regenerate_figure_review_preview(
    candidate_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        return _review_service(settings).regenerate_preview(candidate_id)
    except FigureReviewError as exc:
        raise _review_http_error(exc) from exc


@router.get(
    "/review/candidates/{candidate_id}/preview-image",
    include_in_schema=False,
)
async def figure_review_preview_image(
    candidate_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    try:
        result = _review_service(settings).preview_file(candidate_id)
    except FigureReviewError as exc:
        raise _review_http_error(exc) from exc

    return FileResponse(
        path=result.path,
        media_type="image/png",
        filename=result.path.name,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/review/audit")
async def figure_review_audit(
    limit: int = Query(default=100, ge=1, le=500),
    settings: Settings = Depends(get_settings),
) -> list[dict]:
    return _review_service(settings).recent_audit(limit)

