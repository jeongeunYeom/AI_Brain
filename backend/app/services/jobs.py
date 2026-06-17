from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

_LOCK = Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def create_job() -> str:
    job_id = uuid4().hex
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "step": 0,
            "total_steps": 6,
            "message": "대기 중",
            "status": "running",
            "error": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    return job_id


def update_job(job_id: str | None, step: int, message: str, status: str = "running", error: str | None = None) -> None:
    if not job_id:
        return
    with _LOCK:
        job = _JOBS.setdefault(job_id, {"job_id": job_id, "total_steps": 6})
        job.update({
            "step": step,
            "message": message,
            "status": status,
            "error": error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None
