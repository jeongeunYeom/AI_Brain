from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings
from app.services.document_processor import DocumentProcessor
from app.services.ollama import OllamaClient


UNCERTAIN_MARKERS = (
    "일 수 있습니다",
    "있을 것입니다",
    "보입니다",
    "추정됩니다",
    "가능성이 있습니다",
    "일반적으로",
    "아마",
    "확인하기 어렵습니다",
)

VISION_PROMPT = """Analyze only what is directly visible in this image.
Return concise Korean key-value lines with these keys:
image_type, analysis, x_axis, x_axis_unit, y_axis, y_axis_unit, trend, engineering_meaning.
If an axis, unit, or trend is not directly readable, write 확인할 수 없음.
Do not use speculative phrases such as 일 수 있습니다, 있을 것입니다, 보입니다, 추정됩니다, 가능성이 있습니다, 일반적으로, 아마.
Do not infer petroleum engineering meaning unless it is visible in the image.
"""


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


def matches_note(note_path: Path, note_stem: str | None) -> bool:
    return note_stem is None or note_path.stem == note_stem


def parse_stem(stem: str) -> tuple[str, int, int]:
    match = re.match(r"(?P<doc>.+)_p(?P<page>\d+)_fig(?P<fig>\d+)$", stem)
    if not match:
        return stem, 0, 1
    return match.group("doc"), int(match.group("page")), int(match.group("fig"))


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z_ ]+)\s*[:：]\s*(.+?)\s*$", line)
        if match:
            key = match.group(1).strip().lower().replace(" ", "_")
            values[key] = match.group(2).strip()
    return values


def is_uncertain(value: str) -> bool:
    return not value.strip() or any(marker in value for marker in UNCERTAIN_MARKERS)


def verified_value(value: str) -> tuple[str, bool]:
    if is_uncertain(value) or "확인할 수 없음" in value:
        return "확인할 수 없음", False
    return value.strip(), True


def prepare_vision_image(image_path: Path) -> Path:
    """Create a temporary RGB/autocontrast PNG for Vision without touching the original."""
    with Image.open(image_path) as image:
        image.load()
        prepared = ImageOps.autocontrast(image.convert("RGB"))

    tmp = NamedTemporaryFile(prefix=f"{image_path.stem}_vision_", suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    prepared.save(tmp_path, format="PNG")
    return tmp_path


def classify_prepared_vision_image(original_path: Path, prepared_path: Path) -> dict[str, object]:
    """Filter only clearly unusable images; dark plots can pass via edge density."""
    try:
        file_size = original_path.stat().st_size
        with Image.open(prepared_path) as image:
            width, height = image.size
            gray = image.convert("L")
            small = gray.copy()
            small.thumbnail((256, 256))
            contrast = ImageStat.Stat(small).stddev[0]
            edges = small.filter(ImageFilter.FIND_EDGES)
            histogram = edges.histogram()
            edge_pixels = sum(histogram[26:])
            edge_density = edge_pixels / max(small.width * small.height, 1)

        area = width * height
        aspect_ratio = width / max(height, 1)
        result: dict[str, object] = {
            "image_type": "unknown",
            "confidence": 0.6,
            "should_analyze": True,
            "width": width,
            "height": height,
            "file_size": file_size,
            "contrast": round(float(contrast), 2),
            "edge_density": round(float(edge_density), 4),
            "reason": "",
        }

        if file_size < 5_000 or width < 150 or height < 120 or area < 30_000:
            result.update(
                {
                    "image_type": "decorative",
                    "confidence": 0.1,
                    "should_analyze": False,
                    "reason": "too small or low resolution",
                }
            )
            return result

        if aspect_ratio > 12 or aspect_ratio < 0.08:
            result.update(
                {
                    "image_type": "decorative",
                    "confidence": 0.2,
                    "should_analyze": False,
                    "reason": "extreme aspect ratio",
                }
            )
            return result

        if contrast < 4 and edge_density < 0.01:
            result.update(
                {
                    "image_type": "decorative",
                    "confidence": 0.1,
                    "should_analyze": False,
                    "reason": "low contrast and low edge density",
                }
            )
            return result

        return result
    except (OSError, ValueError, UnidentifiedImageError):
        return {
            "image_type": "unknown",
            "confidence": 0.0,
            "should_analyze": False,
            "reason": "unreadable image",
        }


def build_reprocessed_note(
    *,
    settings: Settings,
    note_path: Path,
    image_path: Path,
    vision_text: str,
    candidate: dict[str, object],
    backup_path: Path | None,
) -> tuple[str, float]:
    document_id, page_number, image_index = parse_stem(note_path.stem)
    fields = parse_key_values(vision_text)
    classified = DocumentProcessor(settings, OllamaClient(settings))._classify_figure_note(vision_text, candidate)

    raw_image_type = fields.get("image_type") or str(classified["image_type"])
    image_type = raw_image_type
    if any(term in raw_image_type.lower() for term in ["plot", "graph"]):
        image_type = "graph"
    elif "chart" in raw_image_type.lower():
        image_type = "chart"
    confidence = float(classified["confidence"])
    x_axis, x_verified = verified_value(fields.get("x_axis", ""))
    y_axis, y_verified = verified_value(fields.get("y_axis", ""))
    trend, trend_verified = verified_value(fields.get("trend", ""))
    x_unit, _ = verified_value(fields.get("x_axis_unit", ""))
    y_unit, _ = verified_value(fields.get("y_axis_unit", ""))
    analysis = re.sub(r"\s+", " ", fields.get("analysis") or vision_text).strip()
    engineering_meaning = fields.get("engineering_meaning", "").strip() or "확인할 수 없음"
    if is_uncertain(engineering_meaning):
        engineering_meaning = "확인할 수 없음"

    lines = [
        "[Figure Note Metadata]",
        f"document_name: {note_path.name}",
        f"document_id: {document_id}",
        f"page_number: {page_number}",
        f"image_index: {image_index}",
        f"image_path: {image_path}",
        f"image_type: {image_type}",
        f"confidence: {confidence}",
        f"analysis: {analysis}",
        f"x_axis: {x_axis}",
        f"x_axis_unit: {x_unit}",
        f"x_axis_verified: {str(x_verified).lower()}",
        f"y_axis: {y_axis}",
        f"y_axis_unit: {y_unit}",
        f"y_axis_verified: {str(y_verified).lower()}",
        f"trend: {trend}",
        f"trend_verified: {str(trend_verified).lower()}",
        f"engineering_meaning: {engineering_meaning}",
        f"vision_model: {settings.vision_model}",
        f"created_at: {datetime.now(timezone.utc).isoformat()}",
        f"legacy_note_backup: {backup_path or ''}",
    ]
    return "\n".join(lines), confidence


async def reprocess_notes(
    *,
    settings: Settings,
    document: str | None,
    note_stem: str | None,
    limit: int | None,
    apply: bool,
    analyze_dry_run: bool,
    force_vision: bool,
    update_chroma: bool,
) -> dict[str, int]:
    ollama = OllamaClient(settings)
    processor = DocumentProcessor(settings, ollama)
    counts = {
        "scanned": 0,
        "document_matches": 0,
        "matched_notes": 0,
        "matched_images": 0,
        "would_call_vision": 0,
        "would_reanalyze": 0,
        "would_reuse_legacy_text": 0,
        "processed": 0,
        "missing_images": 0,
        "skipped_document_mismatch": 0,
        "skipped_already_structured": 0,
        "skipped_missing_image": 0,
        "skipped_missing_metadata": 0,
        "skipped_invalid_note": 0,
        "skipped_low_quality": 0,
        "rejected_low_confidence": 0,
        "failed": 0,
    }

    if document:
        print("document filter fields: note filename, note content")

    handled = 0
    for note_path in sorted(settings.figure_notes_dir.glob("*.md")):
        counts["scanned"] += 1
        if limit is not None and handled >= limit:
            break
        try:
            content = note_path.read_text(encoding="utf-8")
            if not matches_note(note_path, note_stem):
                continue
            if not matches_document(note_path, content, document):
                counts["skipped_document_mismatch"] += 1
                continue
            counts["document_matches"] += 1
            counts["matched_notes"] += 1
            handled += 1

            if "[Figure Note Metadata]" in content:
                counts["skipped_already_structured"] += 1
                continue
            if not content.strip():
                counts["skipped_invalid_note"] += 1
                continue

            image_path = find_image(settings, note_path)
            if image_path is None:
                counts["missing_images"] += 1
                counts["skipped_missing_image"] += 1
                continue
            counts["matched_images"] += 1

            vision_image_path: Path | None = None
            try:
                vision_image_path = prepare_vision_image(image_path)
                candidate = classify_prepared_vision_image(image_path, vision_image_path)
                if not force_vision and not candidate.get("should_analyze"):
                    counts["skipped_low_quality"] += 1
                    continue

                counts["would_call_vision"] += 1
                counts["would_reanalyze"] += 1

                if not apply and not analyze_dry_run:
                    print(f"DRY-RUN would reanalyze {note_path.name} from {image_path.name}")
                    continue

                try:
                    vision_text = (await ollama.describe_image(vision_image_path, prompt=VISION_PROMPT)).strip()
                except Exception as exc:  # noqa: BLE001 - one image failure must not stop the run
                    counts["failed"] += 1
                    print(f"FAIL vision {note_path}: {type(exc).__name__}: {exc}")
                    continue

                if not vision_text:
                    counts["skipped_invalid_note"] += 1
                    continue

                backup_path = note_path.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.md")
                structured_note, confidence = build_reprocessed_note(
                    settings=settings,
                    note_path=note_path,
                    image_path=image_path,
                    vision_text=vision_text,
                    candidate=candidate,
                    backup_path=backup_path if apply else None,
                )

                if confidence < settings.figure_note_min_confidence:
                    counts["rejected_low_confidence"] += 1
                    if analyze_dry_run and not apply:
                        print(structured_note)
                    print(f"REJECT low confidence {confidence}: {note_path.name}")
                    continue

                if analyze_dry_run and not apply:
                    print(structured_note)
                    counts["processed"] += 1
                    continue

                if apply:
                    shutil.copy2(note_path, backup_path)
                    note_path.write_text(structured_note, encoding="utf-8")
                    print(f"UPDATED {note_path} backup={backup_path}")
                    counts["processed"] += 1
            finally:
                if vision_image_path and vision_image_path.exists():
                    vision_image_path.unlink()
        except Exception as exc:  # noqa: BLE001 - continue processing other notes
            counts["failed"] += 1
            print(f"FAIL {note_path}: {type(exc).__name__}: {exc}")

    if update_chroma:
        print("ChromaDB update was requested, but this script does not mutate ChromaDB yet. Re-ingest explicitly after review.")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", help="Only process notes whose filename or content mentions this document.")
    parser.add_argument("--note", dest="note_stem", help="Only process one note stem, without .md.")
    parser.add_argument("--limit", type=int, help="Maximum matched notes to process.")
    parser.add_argument("--apply", action="store_true", help="Write structured notes. Default is dry-run.")
    parser.add_argument("--analyze-dry-run", action="store_true", help="Call Vision and print the new note without writing files.")
    parser.add_argument("--force-vision", action="store_true", help="Bypass only the pre-Vision quality filter.")
    parser.add_argument("--update-chroma", action="store_true", help="Do not update by default; prints an explicit notice.")
    args = parser.parse_args()

    counts = asyncio.run(
        reprocess_notes(
            settings=Settings(),
            document=args.document,
            note_stem=args.note_stem,
            limit=args.limit,
            apply=args.apply,
            analyze_dry_run=args.analyze_dry_run,
            force_vision=args.force_vision,
            update_chroma=args.update_chroma,
        )
    )
    print("\n".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
