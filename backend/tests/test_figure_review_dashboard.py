from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.core.config import Settings
from app.services.figure_review import (
    FigureReviewError,
    FigureReviewService,
)


# Keep fixture paths intentionally short for Windows MAX_PATH.
DOCUMENT_ID = "doc_a"


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=tmp_path)
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
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return settings


def write_candidate(
    settings: Settings,
    *,
    status: str = "review_required",
    page: int = 10,
) -> Path:
    image_path = (
        settings.figures_dir
        / f"figure_p{page}_1.png"
    )
    Image.new("RGB", (300, 200), "white").save(image_path)

    candidate_dir = (
        settings.figure_candidates_dir / DOCUMENT_ID
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)

    suffix = (
        ".json"
        if status == "valid"
        else f".{status}.json"
    )
    path = candidate_dir / f"candidate_p{page}{suffix}"

    payload = {
        "candidate_status": status,
        "asset_path": str(image_path),
        "document_id": DOCUMENT_ID,
        "document_name": "sample.pdf",
        "page_number": page,
        "image_index": 1,
        "effective_classification": "graph",
        "classification_confidence": 0.9,
        "validation_errors": ["sample error"],
        "manual_review_reasons": ["needs review"],
        "final_note_data": {
            "document_id": DOCUMENT_ID,
            "document_name": "sample.pdf",
            "page_number": page,
            "image_index": 1,
            "image_path": str(image_path),
            "image_type": "graph",
            "confidence": 0.9,
            "title": "Old title",
            "analysis": "Old analysis",
            "x_axis": "Time",
            "x_axis_unit": "h",
            "y_axis": "Pressure",
            "y_axis_unit": "psi",
            "series_descriptions": [],
            "reference_lines": [],
            "trend_summary": "Old trend",
            "engineering_meaning": None,
        },
    }

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return path


def test_summary_and_filters(tmp_path):
    settings = make_settings(tmp_path)
    write_candidate(settings, status="valid", page=10)
    write_candidate(
        settings,
        status="review_required",
        page=11,
    )

    service = FigureReviewService(settings)
    summary = service.summary()

    assert summary["totals"]["valid"] == 1
    assert summary["totals"]["review_required"] == 1
    assert summary["automatic_reindex"] is False

    result = service.list_candidates(
        status="review_required",
        page=11,
    )
    assert result["total"] == 1
    assert result["items"][0]["page_number"] == 11


def test_safe_update_creates_backup_and_audit(tmp_path):
    settings = make_settings(tmp_path)
    candidate_path = write_candidate(settings)
    service = FigureReviewService(settings)
    candidate = service.list_candidates()["items"][0]

    updated = service.update_candidate(
        candidate["candidate_id"],
        {
            "status": "valid",
            "title": "Reviewed title",
            "trend_summary": "Reviewed trend",
        },
    )

    assert updated["status"] == "valid"
    assert updated["title"] == "Reviewed title"
    assert updated["needs_reindex"] is True

    assert list(
        (tmp_path / "figure_review_backups").rglob(
            "*.json"
        )
    )

    audit = service.recent_audit()
    assert audit[0]["action"] == "candidate_update"

    persisted = json.loads(
        candidate_path.read_text(encoding="utf-8")
    )
    assert persisted["dashboard_needs_reindex"] is True


def test_rotation_override_and_cache_clear(
    tmp_path,
    monkeypatch,
):
    settings = make_settings(tmp_path)
    write_candidate(settings)
    service = FigureReviewService(settings)
    candidate = service.list_candidates()["items"][0]

    asset_stem = Path(
        candidate["asset_filename"]
    ).stem

    stale = (
        tmp_path
        / "figure_display_previews"
        / f"{asset_stem}_display_old.png"
    )
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old")

    class Result:
        name = "new.png"
        source_type = "extracted_image"
        rotation_applied = 90
        enhanced = True

    monkeypatch.setattr(
        service,
        "_create_preview",
        lambda *args, **kwargs: Result(),
    )

    updated = service.set_rotation(
        candidate["candidate_id"],
        rotation=90,
        pdf_crop_rotation=0,
        enhance=True,
        regenerate=True,
    )

    assert not stale.exists()
    assert updated["rotation"] == 90

    payload = json.loads(
        service.overrides_path.read_text(
            encoding="utf-8"
        )
    )
    assert payload[
        candidate["asset_filename"]
    ]["pdf_crop_rotation"] == 0


def test_invalid_candidate_id_is_rejected(tmp_path):
    settings = make_settings(tmp_path)
    write_candidate(settings)
    service = FigureReviewService(settings)

    for value in [
        "../secret",
        "..\\secret",
        "x" * 24,
    ]:
        try:
            service.get_candidate(value)
        except FigureReviewError:
            pass
        else:
            raise AssertionError(
                f"Expected rejection for {value}"
            )
