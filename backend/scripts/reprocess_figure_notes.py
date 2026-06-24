from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings
from app.services.document_processor import DocumentProcessor
from app.services.ollama import OllamaClient


def normalize_document_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def find_image(settings: Settings, note_path: Path) -> Path | None:
    for suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = settings.figures_dir / f"{note_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def matches_document(note_path: Path, content: str, document: str | None) -> bool:
    if not document:
        return True
    needle = normalize_document_key(document)
    return needle in normalize_document_key(note_path.name) or needle in normalize_document_key(content)


def reprocess_notes(
    *,
    settings: Settings,
    document: str | None,
    apply: bool,
    update_chroma: bool,
) -> dict[str, int]:
    ollama = OllamaClient(settings)
    processor = DocumentProcessor(settings, ollama)
    counts = {
        "scanned": 0,
        "document_matches": 0,
        "processed": 0,
        "skipped_document_mismatch": 0,
        "skipped_already_structured": 0,
        "skipped_missing_image": 0,
        "skipped_missing_metadata": 0,
        "skipped_invalid_note": 0,
        "skipped_low_quality": 0,
        "failed": 0,
    }

    if document:
        print("document filter fields: note filename, note content")

    for note_path in sorted(settings.figure_notes_dir.glob("*.md")):
        counts["scanned"] += 1
        try:
            content = note_path.read_text(encoding="utf-8")
            if not matches_document(note_path, content, document):
                counts["skipped_document_mismatch"] += 1
                continue
            counts["document_matches"] += 1
            if "[Figure Note Metadata]" in content:
                counts["skipped_already_structured"] += 1
                continue
            if not content.strip():
                counts["skipped_invalid_note"] += 1
                continue

            image_path = find_image(settings, note_path)
            if image_path is None:
                counts["skipped_missing_image"] += 1
                continue

            candidate = processor.classify_image_candidate(image_path)
            if not candidate.get("should_analyze"):
                counts["skipped_low_quality"] += 1
                continue
            structured_note = processor.build_figure_note(
                document_name=document or note_path.stem,
                page_number=0,
                image_index=1,
                image_path=image_path,
                note=content,
                page_text="",
                candidate=candidate,
            )

            if apply:
                backup_path = note_path.with_suffix(
                    f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.md"
                )
                shutil.copy2(note_path, backup_path)
                note_path.write_text(structured_note, encoding="utf-8")
                print(f"UPDATED {note_path} backup={backup_path.name}")
            else:
                print(f"DRY-RUN would update {note_path}")
            counts["processed"] += 1
        except Exception as exc:  # noqa: BLE001 - continue processing other notes
            counts["failed"] += 1
            print(f"FAIL {note_path}: {type(exc).__name__}: {exc}")

    if update_chroma:
        print("ChromaDB update was requested, but this safe script only rewrites note files. Re-ingest explicitly after review.")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", help="Only process notes whose filename or content mentions this document.")
    parser.add_argument("--apply", action="store_true", help="Write structured notes. Default is dry-run.")
    parser.add_argument("--update-chroma", action="store_true", help="Do not update by default; prints an explicit notice.")
    args = parser.parse_args()

    counts = reprocess_notes(
        settings=Settings(),
        document=args.document,
        apply=args.apply,
        update_chroma=args.update_chroma,
    )
    print(
        "\n".join(f"{key}={value}" for key, value in counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
