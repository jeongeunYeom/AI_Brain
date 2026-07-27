from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import Settings, get_settings
from app.services.benchmark_dashboard import (
    BenchmarkDashboardService,
    BenchmarkRunNotFound,
)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _service(settings: Settings) -> BenchmarkDashboardService:
    return BenchmarkDashboardService(settings)


def _not_found(exc: BenchmarkRunNotFound) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


@router.get("/runs")
async def list_evaluation_runs(
    limit: int = Query(default=50, ge=1, le=200),
    settings: Settings = Depends(get_settings),
) -> list[dict]:
    return _service(settings).list_runs(limit=limit)


@router.get("/latest")
async def latest_evaluation_run(
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        return _service(settings).get_latest()
    except BenchmarkRunNotFound as exc:
        raise _not_found(exc) from exc


@router.get("/runs/{run_id}")
async def evaluation_run(
    run_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        return _service(settings).get_run(run_id)
    except BenchmarkRunNotFound as exc:
        raise _not_found(exc) from exc
