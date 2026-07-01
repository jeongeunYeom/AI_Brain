from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from app.core.config import Settings
from app.services.figure_analysis import data_sha256
from app.services.figure_preview import FigurePreviewService, PreviewResult


ALLOWED_STATUSES = {
    "valid",
    "review_required",
    "failed",
    "ignored",
}

EDITABLE_NOTE_FIELDS = {
    "title",
    "analysis",
    "x_axis",
    "x_axis_unit",
    "y_axis",
    "y_axis_unit",
    "trend_summary",
    "engineering_meaning",
    "series_descriptions",
    "reference_lines",
}

ROTATION_VALUES = {0, 90, 180, 270}


class FigureReviewError(ValueError):
    pass


class FigureReviewService:
    """Safe review-only access to Figure candidate JSON and display previews.

    This service deliberately does not update Figure Notes, extracted chunks,
    ChromaDB, embeddings, or uploaded PDFs. Candidate edits are marked as
    pending reindex so a later explicit workflow can apply them.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.candidate_root = settings.figure_candidates_dir.resolve()
        self.figures_root = settings.figures_dir.resolve()
        self.preview_root = (
            settings.data_dir / "figure_display_previews"
        ).resolve()
        self.overrides_path = (
            settings.data_dir / "figure_display_overrides.json"
        )
        self.backup_root = (
            settings.data_dir / "figure_review_backups"
        )
        self.audit_root = (
            settings.data_dir / "figure_review_audit"
        )
        self.audit_path = self.audit_root / "audit.jsonl"
        self.preview_service = FigurePreviewService(
            self.preview_root,
            self.overrides_path,
            raw_dir=settings.raw_dir,
            metadata_dir=settings.metadata_dir,
        )

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        documents = self._document_metadata()
        rows: dict[str, dict[str, Any]] = {}

        for document_id, metadata in documents.items():
            rows[document_id] = {
                "document_id": document_id,
                "filename": str(
                    metadata.get("filename")
                    or metadata.get("document_name")
                    or document_id
                ),
                "counts": {
                    "valid": 0,
                    "review_required": 0,
                    "failed": 0,
                    "ignored": 0,
                },
                "total": 0,
                "figure_note_count": self._document_file_count(
                    self.settings.figure_notes_dir,
                    document_id,
                    "*.md",
                ),
                "preview_count": self._document_file_count(
                    self.preview_root,
                    document_id,
                    "*",
                ),
            }

        for _, path, payload in self._iter_candidates():
            document_id = str(
                payload.get("document_id")
                or path.parent.name
            )
            if document_id not in rows:
                rows[document_id] = {
                    "document_id": document_id,
                    "filename": str(
                        payload.get("document_name")
                        or document_id
                    ),
                    "counts": {
                        "valid": 0,
                        "review_required": 0,
                        "failed": 0,
                        "ignored": 0,
                    },
                    "total": 0,
                    "figure_note_count": self._document_file_count(
                        self.settings.figure_notes_dir,
                        document_id,
                        "*.md",
                    ),
                    "preview_count": self._document_file_count(
                        self.preview_root,
                        document_id,
                        "*",
                    ),
                }

            status = self._status(payload)
            if status not in ALLOWED_STATUSES:
                continue
            rows[document_id]["counts"][status] += 1
            rows[document_id]["total"] += 1

        ordered = sorted(
            rows.values(),
            key=lambda item: str(item["filename"]).lower(),
        )

        totals = {
            "valid": 0,
            "review_required": 0,
            "failed": 0,
            "ignored": 0,
            "total": 0,
            "figure_note_count": 0,
            "preview_count": 0,
            "documents": len(ordered),
        }

        for row in ordered:
            for status in ALLOWED_STATUSES:
                totals[status] += int(row["counts"][status])
            totals["total"] += int(row["total"])
            totals["figure_note_count"] += int(
                row["figure_note_count"]
            )
            totals["preview_count"] += int(
                row["preview_count"]
            )

        return {
            "documents": ordered,
            "totals": totals,
            "chroma_changed": False,
            "automatic_reindex": False,
        }

    def list_candidates(
        self,
        *,
        document_id: str | None = None,
        status: str | None = None,
        page: int | None = None,
        query: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        normalized_status = (
            str(status or "").strip().lower()
        )
        if normalized_status in {"", "all"}:
            normalized_status = ""
        elif normalized_status not in ALLOWED_STATUSES:
            raise FigureReviewError(
                f"Unsupported status: {status}"
            )

        normalized_query = str(query or "").strip().lower()
        matches: list[dict[str, Any]] = []

        for candidate_id, path, payload in self._iter_candidates():
            if (
                document_id
                and str(payload.get("document_id")) != document_id
            ):
                continue

            candidate_status = self._status(payload)
            if (
                normalized_status
                and candidate_status != normalized_status
            ):
                continue

            candidate_page = self._as_int(
                payload.get("page_number")
            )
            if page is not None and candidate_page != page:
                continue

            item = self._candidate_view(
                candidate_id,
                path,
                payload,
            )

            if normalized_query:
                searchable = " ".join(
                    [
                        str(item.get("document_name") or ""),
                        str(item.get("title") or ""),
                        str(item.get("classification") or ""),
                        str(item.get("analysis") or ""),
                        str(item.get("trend_summary") or ""),
                        str(item.get("engineering_meaning") or ""),
                        str(item.get("page_number") or ""),
                    ]
                ).lower()
                if normalized_query not in searchable:
                    continue

            matches.append(item)

        matches.sort(
            key=lambda item: (
                str(item.get("document_name") or "").lower(),
                int(item.get("page_number") or 0),
                int(item.get("image_index") or 0),
                str(item.get("candidate_id") or ""),
            )
        )

        total = len(matches)
        safe_offset = max(0, int(offset))
        safe_limit = min(100, max(1, int(limit)))
        items = matches[
            safe_offset:safe_offset + safe_limit
        ]

        return {
            "items": items,
            "total": total,
            "offset": safe_offset,
            "limit": safe_limit,
            "has_more": safe_offset + len(items) < total,
        }

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        path, payload = self._resolve_candidate(candidate_id)
        return self._candidate_view(
            candidate_id,
            path,
            payload,
        )

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------
    def update_candidate(
        self,
        candidate_id: str,
        changes: dict[str, Any],
    ) -> dict[str, Any]:
        path, payload = self._resolve_candidate(candidate_id)
        before = json.loads(json.dumps(payload))
        fields_changed: list[str] = []

        status_supplied = "status" in changes
        new_status = str(
            changes.get("status") or ""
        ).strip().lower()

        if status_supplied:
            if new_status not in ALLOWED_STATUSES:
                raise FigureReviewError(
                    f"Unsupported status: {new_status}"
                )
            if (
                new_status == "valid"
                and not isinstance(
                    payload.get("final_note_data"),
                    dict,
                )
            ):
                raise FigureReviewError(
                    "A candidate without final_note_data "
                    "cannot be marked valid."
                )
            if new_status != self._status(payload):
                payload["candidate_status"] = new_status
                payload["manual_review_required"] = (
                    new_status == "review_required"
                )
                payload["apply_ready"] = new_status == "valid"
                fields_changed.append("status")

        note_changes = {
            key: value
            for key, value in changes.items()
            if key in EDITABLE_NOTE_FIELDS
        }

        if note_changes:
            note_data = payload.get("final_note_data")
            if not isinstance(note_data, dict):
                raise FigureReviewError(
                    "This candidate has no editable "
                    "final_note_data."
                )

            for key, value in note_changes.items():
                if key in {
                    "series_descriptions",
                    "reference_lines",
                }:
                    if value is None:
                        normalized_value: Any = []
                    elif isinstance(value, list):
                        normalized_value = [
                            str(item).strip()
                            for item in value
                            if str(item).strip()
                        ]
                    else:
                        normalized_value = [
                            part.strip()
                            for part in str(value).splitlines()
                            if part.strip()
                        ]
                else:
                    normalized_value = (
                        None
                        if value is None
                        else str(value).strip() or None
                    )

                if note_data.get(key) != normalized_value:
                    note_data[key] = normalized_value
                    fields_changed.append(key)

            payload["final_note_data"] = note_data
            payload["final_note_data_sha256"] = (
                data_sha256(note_data)
            )

        if not fields_changed:
            return self._candidate_view(
                candidate_id,
                path,
                payload,
            )

        now = datetime.now(timezone.utc).isoformat()
        payload["dashboard_reviewed_at"] = now
        payload["dashboard_needs_reindex"] = True
        payload["dashboard_review_fields"] = sorted(
            set(fields_changed)
        )

        backup_path = self._backup_file(
            path,
            candidate_id,
            category="candidate",
        )
        self._atomic_write_json(path, payload)

        self._append_audit(
            {
                "timestamp": now,
                "action": "candidate_update",
                "candidate_id": candidate_id,
                "candidate_path": str(path),
                "backup_path": str(backup_path),
                "fields_changed": sorted(
                    set(fields_changed)
                ),
                "before_status": self._status(before),
                "after_status": self._status(payload),
                "chroma_changed": False,
                "figure_note_changed": False,
            }
        )

        return self._candidate_view(
            candidate_id,
            path,
            payload,
        )

    def set_rotation(
        self,
        candidate_id: str,
        *,
        rotation: int | None,
        pdf_crop_rotation: int | None,
        enhance: bool,
        regenerate: bool = True,
    ) -> dict[str, Any]:
        path, payload = self._resolve_candidate(candidate_id)
        asset_path = self._asset_path(payload)

        normalized_rotation = self._rotation(rotation)
        normalized_pdf_rotation = self._rotation(
            pdf_crop_rotation
        )

        override_payload = self._read_json_dict(
            self.overrides_path
        )
        before_entry = override_payload.get(
            asset_path.name,
            {},
        )

        entry = dict(
            before_entry
            if isinstance(before_entry, dict)
            else {}
        )

        if normalized_rotation is None:
            entry.pop("rotation", None)
        else:
            entry["rotation"] = normalized_rotation

        if normalized_pdf_rotation is None:
            entry.pop("pdf_crop_rotation", None)
        else:
            entry[
                "pdf_crop_rotation"
            ] = normalized_pdf_rotation

        entry["enhance"] = bool(enhance)
        override_payload[asset_path.name] = entry

        backup_path = self._backup_optional_file(
            self.overrides_path,
            "figure_display_overrides",
            category="overrides",
        )
        self._atomic_write_json(
            self.overrides_path,
            override_payload,
        )
        removed = self._clear_preview_cache(
            asset_path.stem
        )

        preview_result: PreviewResult | None = None
        if regenerate:
            preview_result = self._create_preview(
                payload,
                asset_path,
                overwrite=True,
            )

        now = datetime.now(timezone.utc).isoformat()
        self._append_audit(
            {
                "timestamp": now,
                "action": "rotation_update",
                "candidate_id": candidate_id,
                "candidate_path": str(path),
                "asset_filename": asset_path.name,
                "backup_path": (
                    str(backup_path)
                    if backup_path
                    else None
                ),
                "before": before_entry,
                "after": entry,
                "removed_preview_count": removed,
                "preview_regenerated": bool(
                    preview_result
                ),
                "chroma_changed": False,
                "figure_note_changed": False,
            }
        )

        item = self._candidate_view(
            candidate_id,
            path,
            payload,
        )
        item["preview_regenerated"] = bool(
            preview_result
        )
        if preview_result:
            item["preview_source"] = (
                preview_result.source_type
            )
            item["rotation_applied"] = (
                preview_result.rotation_applied
            )
            item["preview_url"] = (
                "/api/figure-previews/"
                f"{quote(preview_result.name, safe='')}"
            )
        return item

    def regenerate_preview(
        self,
        candidate_id: str,
    ) -> dict[str, Any]:
        path, payload = self._resolve_candidate(candidate_id)
        asset_path = self._asset_path(payload)
        removed = self._clear_preview_cache(
            asset_path.stem
        )
        result = self._create_preview(
            payload,
            asset_path,
            overwrite=True,
        )
        if result is None:
            raise FigureReviewError(
                "Preview generation failed."
            )

        now = datetime.now(timezone.utc).isoformat()
        self._append_audit(
            {
                "timestamp": now,
                "action": "preview_regenerate",
                "candidate_id": candidate_id,
                "candidate_path": str(path),
                "asset_filename": asset_path.name,
                "removed_preview_count": removed,
                "preview_source": result.source_type,
                "rotation_applied": result.rotation_applied,
                "chroma_changed": False,
                "figure_note_changed": False,
            }
        )

        return {
            "candidate_id": candidate_id,
            "preview_url": (
                "/api/figure-previews/"
                f"{quote(result.name, safe='')}"
            ),
            "preview_source": result.source_type,
            "rotation_applied": (
                result.rotation_applied
            ),
            "enhanced": result.enhanced,
        }

    def preview_file(
        self,
        candidate_id: str,
    ) -> PreviewResult:
        _, payload = self._resolve_candidate(candidate_id)
        asset_path = self._asset_path(payload)
        result = self._create_preview(
            payload,
            asset_path,
            overwrite=False,
        )
        if result is None:
            raise FigureReviewError(
                "Preview generation failed."
            )
        return result

    def recent_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.audit_path.is_file():
            return []

        rows: list[dict[str, Any]] = []
        with self.audit_path.open(
            "r",
            encoding="utf-8-sig",
        ) as source:
            for line in source:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows[-max(1, min(500, limit)):][::-1]

    # ------------------------------------------------------------------
    # Candidate helpers
    # ------------------------------------------------------------------
    def _iter_candidates(self):
        if not self.candidate_root.is_dir():
            return

        for path in sorted(
            self.candidate_root.rglob("*.json")
        ):
            try:
                resolved = path.resolve()
                resolved.relative_to(self.candidate_root)
                payload = json.loads(
                    path.read_text(encoding="utf-8-sig")
                )
            except (
                OSError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue

            if not isinstance(payload, dict):
                continue

            relative = resolved.relative_to(
                self.candidate_root
            ).as_posix()
            candidate_id = hashlib.sha256(
                relative.encode("utf-8")
            ).hexdigest()[:24]
            yield candidate_id, resolved, payload

    def _resolve_candidate(
        self,
        candidate_id: str,
    ) -> tuple[Path, dict[str, Any]]:
        normalized = str(candidate_id or "").strip().lower()
        if (
            len(normalized) != 24
            or any(
                character not in "0123456789abcdef"
                for character in normalized
            )
        ):
            raise FigureReviewError(
                "Invalid candidate ID."
            )

        for current_id, path, payload in self._iter_candidates():
            if current_id == normalized:
                return path, payload

        raise FigureReviewError(
            "Candidate not found."
        )

    def _candidate_view(
        self,
        candidate_id: str,
        path: Path,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        note_data = payload.get("final_note_data")
        if not isinstance(note_data, dict):
            note_data = {}

        asset_path: Path | None
        try:
            asset_path = self._asset_path(payload)
        except FigureReviewError:
            asset_path = None

        filename = (
            asset_path.name
            if asset_path
            else Path(
                str(payload.get("asset_path") or "")
            ).name
        )

        overrides = self._read_json_dict(
            self.overrides_path
        )
        override = overrides.get(filename, {})
        if not isinstance(override, dict):
            override = {}

        return {
            "candidate_id": candidate_id,
            "candidate_filename": path.name,
            "document_id": str(
                payload.get("document_id") or ""
            ),
            "document_name": str(
                payload.get("document_name") or ""
            ),
            "page_number": self._as_int(
                payload.get("page_number")
            ),
            "image_index": self._as_int(
                payload.get("image_index")
            ),
            "status": self._status(payload),
            "classification": str(
                payload.get(
                    "effective_classification"
                )
                or payload.get(
                    "automatic_classification"
                )
                or ""
            ),
            "confidence": payload.get(
                "classification_confidence"
            ),
            "title": note_data.get("title"),
            "image_type": note_data.get(
                "image_type"
            )
            or payload.get(
                "effective_classification"
            ),
            "analysis": note_data.get("analysis"),
            "x_axis": note_data.get("x_axis"),
            "x_axis_unit": note_data.get(
                "x_axis_unit"
            ),
            "y_axis": note_data.get("y_axis"),
            "y_axis_unit": note_data.get(
                "y_axis_unit"
            ),
            "trend_summary": note_data.get(
                "trend_summary"
            ),
            "engineering_meaning": note_data.get(
                "engineering_meaning"
            ),
            "series_descriptions": list(
                note_data.get(
                    "series_descriptions"
                )
                or []
            ),
            "reference_lines": list(
                note_data.get(
                    "reference_lines"
                )
                or []
            ),
            "validation_errors": list(
                payload.get("validation_errors")
                or []
            ),
            "manual_review_reasons": list(
                payload.get(
                    "manual_review_reasons"
                )
                or []
            ),
            "schema_valid": bool(
                payload.get("schema_valid")
            ),
            "information_quality_passed": bool(
                payload.get(
                    "information_quality_passed"
                )
            ),
            "apply_ready": bool(
                payload.get("apply_ready")
            ),
            "needs_reindex": bool(
                payload.get(
                    "dashboard_needs_reindex"
                )
            ),
            "rotation": self._rotation(
                override.get("rotation")
            ),
            "pdf_crop_rotation": self._rotation(
                override.get(
                    "pdf_crop_rotation"
                )
            ),
            "enhance": bool(
                override.get("enhance", True)
            ),
            "asset_filename": filename,
            "original_url": (
                f"/api/figures/{quote(filename, safe='')}"
                if asset_path
                else None
            ),
            "preview_url": (
                "/api/review/candidates/"
                f"{candidate_id}/preview-image"
                if asset_path
                else None
            ),
            "preview_source": None,
            "updated_at": payload.get(
                "dashboard_reviewed_at"
            )
            or payload.get("created_at"),
            "editable": isinstance(
                payload.get("final_note_data"),
                dict,
            ),
        }

    def _asset_path(
        self,
        payload: dict[str, Any],
    ) -> Path:
        filename = Path(
            str(
                payload.get("asset_path")
                or (
                    payload.get("final_note_data")
                    or {}
                ).get("image_path")
                or ""
            )
        ).name

        if not filename:
            raise FigureReviewError(
                "Candidate has no asset path."
            )

        candidate = (
            self.figures_root / filename
        ).resolve()
        try:
            candidate.relative_to(self.figures_root)
        except ValueError as exc:
            raise FigureReviewError(
                "Invalid asset path."
            ) from exc

        if not candidate.is_file():
            raise FigureReviewError(
                "Figure asset not found."
            )

        return candidate

    def _create_preview(
        self,
        payload: dict[str, Any],
        asset_path: Path,
        *,
        overwrite: bool,
    ) -> PreviewResult | None:
        note_data = payload.get("final_note_data")
        if not isinstance(note_data, dict):
            note_data = {}

        return self.preview_service.get_or_create_preview(
            asset_path,
            document_id=str(
                payload.get("document_id") or ""
            )
            or None,
            document_name=str(
                payload.get("document_name") or ""
            )
            or None,
            page=self._as_int(
                payload.get("page_number")
            ),
            image_index=self._as_int(
                payload.get("image_index")
            ),
            image_type=str(
                note_data.get("image_type")
                or payload.get(
                    "effective_classification"
                )
                or ""
            )
            or None,
            overwrite=overwrite,
        )

    # ------------------------------------------------------------------
    # Safe filesystem helpers
    # ------------------------------------------------------------------
    def _backup_file(
        self,
        source: Path,
        identifier: str,
        *,
        category: str,
    ) -> Path:
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S_%f")
        destination = (
            self.backup_root
            / timestamp
            / category
            / f"{identifier}{source.suffix}"
        )
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination.write_bytes(
            source.read_bytes()
        )
        return destination

    def _backup_optional_file(
        self,
        source: Path,
        identifier: str,
        *,
        category: str,
    ) -> Path | None:
        if not source.is_file():
            return None
        return self._backup_file(
            source,
            identifier,
            category=category,
        )

    def _atomic_write_json(
        self,
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp.json",
                delete=False,
            ) as temporary:
                temporary_path = Path(
                    temporary.name
                )

            temporary_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(
                    missing_ok=True
                )

    def _append_audit(
        self,
        event: dict[str, Any],
    ) -> None:
        self.audit_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        with self.audit_path.open(
            "a",
            encoding="utf-8",
        ) as target:
            target.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    def _clear_preview_cache(
        self,
        stem: str,
    ) -> int:
        if not self.preview_root.is_dir():
            return 0

        removed = 0
        for path in self.preview_root.glob(
            f"{stem}_display_*"
        ):
            if path.is_file():
                path.unlink()
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Small parsing helpers
    # ------------------------------------------------------------------
    def _document_metadata(
        self,
    ) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for path in self.settings.metadata_dir.glob(
            "*.json"
        ):
            try:
                payload = json.loads(
                    path.read_text(
                        encoding="utf-8-sig"
                    )
                )
            except (
                OSError,
                json.JSONDecodeError,
            ):
                continue
            if isinstance(payload, dict):
                document_id = str(
                    payload.get("document_id")
                    or path.stem
                )
                records[document_id] = payload
        return records

    def _document_file_count(
        self,
        root: Path,
        document_id: str,
        pattern: str,
    ) -> int:
        if not root.is_dir():
            return 0
        return sum(
            1
            for path in root.glob(
                f"{document_id}_*"
            )
            if path.is_file()
            and (
                pattern == "*"
                or path.match(pattern)
            )
        )

    def _status(
        self,
        payload: dict[str, Any],
    ) -> str:
        status = str(
            payload.get("candidate_status")
            or payload.get("status")
            or "failed"
        ).strip().lower()
        return (
            status
            if status in ALLOWED_STATUSES
            else "failed"
        )

    def _rotation(
        self,
        value: Any,
    ) -> int | None:
        if value in {None, ""}:
            return None
        try:
            rotation = int(value)
        except (TypeError, ValueError):
            raise FigureReviewError(
                f"Invalid rotation: {value}"
            )
        if rotation not in ROTATION_VALUES:
            raise FigureReviewError(
                f"Invalid rotation: {rotation}"
            )
        return rotation

    def _read_json_dict(
        self,
        path: Path,
    ) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8-sig"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return {}
        return (
            payload
            if isinstance(payload, dict)
            else {}
        )

    def _as_int(
        self,
        value: Any,
    ) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
