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


def find_image(settings: Settings, note_path: Path) -> Path | None:
    for suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = settings.figures_dir / f"{note_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def matches_document(note_path: Path, content: str, document: str | None) -> bool:
    if not document:
        return True
    needle = document.lower()
    return needle in note_path.name.lower() or needle in content.lower()


def reprocess_notes(
    *,
    settings: Settings,
    document: str | None,
    apply: bool,
    update_chroma: bool,
) -> dict[str, int]:
    processor = DocumentProcessor(settings)
    counts = {"processed": 0, "skipped": 0, "failed": 0}

    for note_path in sorted(settings.figure_notes_dir.glob("*.md")):
        try:
            content = note_path.read_text(encoding="utf-8")
            if not matches_document(note_path, content, document):
                counts["skipped"] += 1
                continue
            if "[Figure Note Metadata]" in content:
                counts["skipped"] += 1
                continue

            image_path = find_image(settings, note_path)
            if image_path is None:
                counts["failed"] += 1
                print(f"FAIL no image: {note_path}")
                continue

            candidate = processor.classify_image_candidate(image_path)
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
        "processed={processed} skipped={skipped} failed={failed}".format(
            **counts
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
