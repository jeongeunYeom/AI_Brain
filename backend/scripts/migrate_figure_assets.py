from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import logging
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings
from app.services.ollama import OllamaClient
from scripts.reprocess_figure_notes import prepare_vision_image, parse_key_values, verified_value


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
LOGO_TYPES = {"logo", "icon", "decoration", "decorative", "cover_art"}
KEEP_TYPES = {"graph", "chart", "table", "diagram", "schematic", "equation"}
GRAPH_TYPES = {"graph", "chart"}
PROTECTED_TYPES = {"graph", "chart", "table", "diagram", "schematic", "equation"}
GRAPH_PROMPT = """Analyze only what is directly visible in this graph or chart.
Return concise key-value lines in Korean with these keys:
image_type, confidence, title, x_axis, x_axis_unit, x_axis_scale, y_axis,
y_axis_unit, y_axis_scale, series_count, series_descriptions, legend,
reference_lines, trend, engineering_meaning.
For multiple curves, describe each curve separately. Do not merge all curves into one trend.
If labels, legend, units, scale, or engineering meaning are not directly readable, write 확인할 수 없음.
Do not infer petroleum engineering meaning without page context.
"""
GRAPH_PROMPT += """
Also include plateau, peak, decline, and slope_changes.
Distinguish continuously rising curves, rise-then-plateau curves, and diagonal reference lines when visible.
Report log/linear/unknown scale for each axis.
Do not guess series names when the legend is missing or unreadable.
"""

CATEGORIES = ("logo", "graph", "table", "diagram", "equation", "photo", "page_decoration", "unknown")

DIAGRAM_PROMPT = """Analyze only what is directly visible in this engineering diagram or schematic.
Return concise key-value lines in Korean with these keys:
image_type, confidence, title, components, connections, labels, engineering_purpose, uncertain_items.
Do not invent axes, trends, or series for diagrams.
If a component or label is unreadable, write 확인할 수 없음.
"""


def normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def parse_asset_stem(stem: str) -> tuple[str, int, int] | None:
    match = re.match(r"(?P<doc>.+)_p(?P<page>\d+)_fig(?P<fig>\d+)$", stem)
    if not match:
        return None
    return match.group("doc"), int(match.group("page")), int(match.group("fig"))


def safe_document_stem(document_name: str, document_id: str, max_len: int = 120) -> str:
    stem = document_name[:-4] if document_name.lower().endswith(".pdf") else document_name
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip(" ._") or "unresolved_document"
    if len(stem) > max_len:
        keep = max_len - 9
        stem = f"{stem[:keep].rstrip('._')}_{document_id[:8]}"
    return stem


def target_stem(document_name: str, document_id: str, page: int, fig: int) -> str:
    return f"{safe_document_stem(document_name, document_id)}_p{page:04d}_fig{fig:02d}"


def read_note_fields(note_path: Path | None) -> dict[str, str]:
    if not note_path or not note_path.exists():
        return {}
    return parse_key_values(note_path.read_text(encoding="utf-8", errors="replace"))


def resolve_documents(settings: Settings) -> dict[str, str]:
    mapping: dict[str, str] = {}
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
        client = chromadb.PersistentClient(
            path=str(settings.vector_db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_collection(settings.collection_name)
        data = collection.get(include=["metadatas"])
        for meta in data.get("metadatas", []):
            doc_id = str((meta or {}).get("document_id") or "")
            document = str((meta or {}).get("document") or "")
            if doc_id and document:
                mapping[doc_id] = document
    except Exception:
        pass

    for path in settings.metadata_dir.glob("*.json"):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        doc_id = str(data.get("document_id") or data.get("sha256") or path.stem)
        filename = str(data.get("filename") or "")
        if doc_id and filename and doc_id not in mapping:
            mapping[doc_id] = filename

    for path in settings.raw_dir.glob("*"):
        if not path.is_file():
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        if digest not in mapping:
            name = path.name
            if name.startswith(f"{digest}_"):
                name = name[len(digest) + 1 :]
            mapping[digest] = name

    return mapping


def image_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 128), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash(path: Path) -> int:
    try:
        with Image.open(path) as image:
            normalized = ImageOps.autocontrast(image.convert("RGB"))
            gray = ImageOps.grayscale(normalized).resize((8, 8))
            pixels = list(gray.getdata())
    except (OSError, UnidentifiedImageError):
        return 0
    avg = sum(pixels) / max(len(pixels), 1)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | int(pixel >= avg)
    return bits


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def exact_group_id(digest: str) -> str:
    return f"exact_{digest[:8]}"


def near_group_id(hash_value: int) -> str:
    return f"near_{hash_value:016x}"[:13]


def prompt_for_classification(classification: str) -> str:
    if classification in {"diagram", "schematic"}:
        return DIAGRAM_PROMPT
    return GRAPH_PROMPT


def load_decisions(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("groups", {})


def image_metrics(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        with Image.open(path) as image:
            width, height = image.size
            gray = ImageOps.grayscale(image.convert("RGB"))
            small = gray.copy()
            small.thumbnail((256, 256))
            stat = ImageStat.Stat(small)
            contrast = stat.stddev[0]
            brightness = stat.mean[0]
            edges = small.filter(ImageFilter.FIND_EDGES)
            edge_density = sum(edges.histogram()[26:]) / max(small.width * small.height, 1)
        return {
            "width": width,
            "height": height,
            "file_size": size,
            "aspect_ratio": width / max(height, 1),
            "contrast": round(float(contrast), 2),
            "brightness": round(float(brightness), 2),
            "edge_density": round(float(edge_density), 4),
        }
    except (OSError, UnidentifiedImageError):
        return {"width": 0, "height": 0, "file_size": 0, "aspect_ratio": 0, "contrast": 0, "brightness": 0, "edge_density": 0}


def classify_asset(
    note_fields: dict[str, str],
    metrics: dict[str, Any],
    duplicate_count: int,
    page_count: int,
    note_text: str,
) -> tuple[str, float, list[str], list[str], str]:
    image_type = str(note_fields.get("image_type") or note_fields.get("image type") or "").lower()
    try:
        confidence = float(note_fields.get("confidence") or 0)
    except ValueError:
        confidence = 0.0

    text = f"{image_type} {note_text}".lower()
    reasons: list[str] = []
    protections: list[str] = []
    aspect = float(metrics.get("aspect_ratio") or 0)
    edge_density = float(metrics.get("edge_density") or 0)
    brightness = float(metrics.get("brightness") or 0)
    wide_rectangle = 1.2 <= aspect <= 5.5 and int(metrics.get("width") or 0) >= 350
    dark_graph = brightness < 90 and edge_density >= 0.015 and wide_rectangle
    graph_terms = ["graph", "chart", "plot", "log-log", "pressure", "time", "psi", "hours", "delta"]

    if wide_rectangle:
        protections.append("wide_rectangle")
    if dark_graph:
        protections.append("dark_graph")
    if any(term in text for term in graph_terms):
        protections.append("graph_text")

    if "equation" in image_type:
        return "equation", max(confidence, 0.6), ["note_image_type"], protections, "equation"
    if "table" in image_type:
        return "table", max(confidence, 0.6), ["note_image_type"], protections, "table"
    if "diagram" in image_type or "schematic" in image_type:
        return "diagram", max(confidence, 0.6), ["note_image_type"], protections, "diagram"
    if "photo" in image_type or "photograph" in image_type:
        return "photo", max(confidence, 0.5), ["note_image_type"], protections, "photo"
    if "graph" in image_type or "chart" in image_type or "plot" in image_type or protections:
        return "graph", max(confidence, 0.6), ["graph_protection"], protections, "graph"

    small = int(metrics.get("width") or 0) < 350 or int(metrics.get("height") or 0) < 250 or int(metrics.get("file_size") or 0) < 20_000
    odd_shape = aspect > 8 or aspect < 0.125
    low_detail = float(metrics.get("contrast") or 0) < 5 and edge_density < 0.01
    if small:
        reasons.append("small")
    if odd_shape:
        reasons.append("odd_aspect_ratio")
    if low_detail:
        reasons.append("low_detail")

    logo_type = any(kind in image_type for kind in LOGO_TYPES)
    if logo_type and confidence >= 0.9 and duplicate_count >= 3 and page_count >= 3:
        reasons.extend(["logo_type", "high_confidence", "repeated_3_pages"])
        return "logo", confidence, reasons, protections, "logo"
    if logo_type or (duplicate_count >= 3 and page_count >= 3 and (small or odd_shape or low_detail)):
        reasons.append("insufficient_logo_evidence")
        return "unknown", confidence, reasons, protections, "uncertain_logo"
    if small or odd_shape or low_detail:
        return "page_decoration", confidence, reasons, protections, "page_decoration"
    return "unknown", confidence, reasons, protections, "unknown"


def is_backup_note(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".bak.md")
        or name.endswith(".backup.md")
        or re.search(r"\.\d{8}_\d{6}\.bak\.md$", name) is not None
    )


def is_temporary_file(path: Path) -> bool:
    name = path.name.lower()
    return ".tmp." in name or "_vision_" in name


def collect_assets(settings: Settings) -> tuple[dict[str, dict[str, Path | None]], dict[str, int]]:
    assets: dict[str, dict[str, Path | None]] = {}
    excluded = {
        "active_notes": 0,
        "backup_notes_excluded": 0,
        "temporary_files_excluded": 0,
        "quarantined_files_excluded": 0,
    }
    for image in settings.figures_dir.glob("*"):
        if is_temporary_file(image):
            excluded["temporary_files_excluded"] += 1
            continue
        if image.suffix.lower() in IMAGE_SUFFIXES:
            assets.setdefault(image.stem, {})["image"] = image
    for note in settings.figure_notes_dir.glob("*.md"):
        if note.name == ".gitkeep":
            continue
        if is_temporary_file(note):
            excluded["temporary_files_excluded"] += 1
            continue
        if is_backup_note(note):
            excluded["backup_notes_excluded"] += 1
            continue
        excluded["active_notes"] += 1
        assets.setdefault(note.stem, {})["note"] = note
    quarantine_root = settings.data_dir / "figure_quarantine"
    if quarantine_root.exists():
        excluded["quarantined_files_excluded"] = sum(1 for path in quarantine_root.rglob("*") if path.is_file())
    return assets, excluded


def update_note_text(text: str, replacements: dict[str, str]) -> str:
    lines = text.splitlines()
    seen: set[str] = set()
    updated: list[str] = []
    for line in lines:
        match = re.match(r"^([A-Za-z_ ]+):", line)
        key = match.group(1).strip().lower().replace(" ", "_") if match else ""
        if key in replacements:
            updated.append(f"{key}: {replacements[key]}")
            seen.add(key)
        else:
            updated.append(line)
    for key, value in replacements.items():
        if key not in seen:
            updated.append(f"{key}: {value}")
    return "\n".join(updated) + "\n"


def apply_manual_decisions(
    settings: Settings,
    operations: list[dict[str, Any]],
    decisions: dict[str, Any],
    quarantine_logos: bool,
) -> None:
    if not quarantine_logos or not decisions:
        return
    for op in operations:
        decision_key = op.get("canonical_group_id") if op.get("canonical_group_id") in decisions else op.get("group_id")
        decision = decisions.get(decision_key or "")
        if not decision:
            continue
        action = str(decision.get("action") or "").lower()
        classification = str(decision.get("classification") or "").lower()
        if action != "quarantine" or classification not in {"logo", "page_decoration"}:
            continue
        if decision_key == op.get("canonical_group_id"):
            group_ops = [item for item in operations if item.get("canonical_group_id") == decision_key]
        else:
            group_ops = [item for item in operations if item.get("group_id") == decision_key]
        if any(item.get("classification") in PROTECTED_TYPES for item in group_ops):
            continue
        for item in group_ops:
            item["classification"] = classification
            item["review_bucket"] = classification
            item["classification_reasons"] = [*item.get("classification_reasons", []), f"human_decision:{decision_key}"]
            item["decision_reason"] = str(decision.get("reason") or "")
            if item.get("new_image"):
                target = Path(str(item["new_image"]))
                item["new_image"] = str(settings.data_dir / "figure_quarantine" / "logos" / target.name)
            if item.get("new_note"):
                target = Path(str(item["new_note"]))
                item["new_note"] = str(settings.data_dir / "figure_quarantine" / "figure_notes" / target.name)
            item["quarantine"] = True


def detect_collisions(operations: list[dict[str, Any]]) -> list[str]:
    seen: set[Path] = set()
    collisions: list[str] = []
    for op in operations:
        for old_key, new_key in [("old_image", "new_image"), ("old_note", "new_note")]:
            target_value = op.get(new_key)
            if not target_value:
                continue
            target = Path(str(target_value))
            old = Path(str(op.get(old_key))) if op.get(old_key) else None
            if target in seen or (target.exists() and target != old):
                collisions.append(str(target))
            seen.add(target)
    return collisions


def graph_note_text(
    *,
    document_name: str,
    document_id: str,
    page: int,
    fig: int,
    image_path: Path,
    vision_text: str,
    settings: Settings,
    backup_path: Path | None,
) -> str:
    fields = parse_key_values(vision_text)
    x_axis, x_ok = verified_value(fields.get("x_axis", ""))
    y_axis, y_ok = verified_value(fields.get("y_axis", ""))
    trend, trend_ok = verified_value(fields.get("trend", ""))
    legend, legend_ok = verified_value(fields.get("legend", ""))
    title, title_ok = verified_value(fields.get("title", ""))
    engineering, engineering_ok = verified_value(fields.get("engineering_meaning", ""))
    return "\n".join(
        [
            "[Figure Note Metadata]",
            f"document_name: {document_name}",
            f"document_id: {document_id}",
            f"page_number: {page}",
            f"image_index: {fig}",
            f"image_path: {image_path}",
            f"image_type: {fields.get('image_type', 'graph')}",
            f"confidence: {fields.get('confidence', '0.6')}",
            f"title: {title}",
            f"title_verified: {str(title_ok).lower()}",
            f"x_axis: {x_axis}",
            f"x_axis_unit: {fields.get('x_axis_unit', '확인할 수 없음')}",
            f"x_axis_scale: {fields.get('x_axis_scale', 'unknown')}",
            f"x_axis_verified: {str(x_ok).lower()}",
            f"y_axis: {y_axis}",
            f"y_axis_unit: {fields.get('y_axis_unit', '확인할 수 없음')}",
            f"y_axis_scale: {fields.get('y_axis_scale', 'unknown')}",
            f"y_axis_verified: {str(y_ok).lower()}",
            f"series_count: {fields.get('series_count', '확인할 수 없음')}",
            "series_count_verified: false",
            f"series_descriptions: {fields.get('series_descriptions', '확인할 수 없음')}",
            f"legend: {legend}",
            f"legend_verified: {str(legend_ok).lower()}",
            f"reference_lines: {fields.get('reference_lines', '확인할 수 없음')}",
            f"plateau: {fields.get('plateau', '확인할 수 없음')}",
            f"peak: {fields.get('peak', '확인할 수 없음')}",
            f"decline: {fields.get('decline', '확인할 수 없음')}",
            f"slope_changes: {fields.get('slope_changes', '확인할 수 없음')}",
            f"trend: {trend}",
            f"trend_verified: {str(trend_ok).lower()}",
            f"engineering_meaning: {engineering}",
            f"engineering_meaning_verified: {str(engineering_ok).lower()}",
            f"vision_model: {settings.vision_model}",
            f"created_at: {datetime.now(timezone.utc).isoformat()}",
            f"legacy_note_backup: {backup_path or ''}",
            "",
        ]
    )


def build_plan(
    settings: Settings,
    *,
    document: str | None,
    limit: int | None,
    quarantine_logos: bool,
    reanalyze_graphs: bool,
    update_chroma: bool,
    decisions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decisions = decisions or {}
    documents = resolve_documents(settings)
    assets, excluded = collect_assets(settings)
    image_hashes = {
        stem: image_hash(paths["image"])
        for stem, paths in assets.items()
        if paths.get("image")
    }
    exact_groups: dict[str, list[str]] = {}
    for stem, digest in image_hashes.items():
        exact_groups.setdefault(digest, []).append(stem)
    duplicate_pages = {
        stem: {
            parse_asset_stem(other)[1]
            for other in exact_groups.get(digest, [])
            if parse_asset_stem(other)
        }
        for stem, digest in image_hashes.items()
    }

    near_hashes = {stem: average_hash(paths["image"]) for stem, paths in assets.items() if paths.get("image")}
    near_groups: dict[str, list[str]] = {}
    stem_to_near: dict[str, str] = {}
    for stem, left_hash in sorted(near_hashes.items()):
        if stem in stem_to_near:
            continue
        group = [other for other, right_hash in near_hashes.items() if hamming(left_hash, right_hash) <= 8]
        canonical_hash = min(near_hashes[item] for item in group)
        canonical_id = near_group_id(canonical_hash)
        near_groups[canonical_id] = sorted(group)
        for item in group:
            stem_to_near[item] = canonical_id

    operations: list[dict[str, Any]] = []
    collisions: list[str] = []
    rename_examples: list[dict[str, str]] = []
    quarantine_examples: list[str] = []
    stats = {
        **excluded,
        "scanned_images": sum(1 for item in assets.values() if item.get("image")),
        "scanned_notes": sum(1 for item in assets.values() if item.get("note")),
        "resolved_documents": 0,
        "resolved_document_count": 0,
        "resolved_asset_count": 0,
        "filtered_document_images": 0,
        "filtered_document_active_notes": 0,
        "unresolved_documents": 0,
        "rename_candidates": 0,
        "name_collisions": 0,
        "matched_image_note_pairs": 0,
        "orphan_images": 0,
        "orphan_notes": 0,
        "exact_duplicate_groups": sum(1 for group in exact_groups.values() if len(group) > 1),
        "near_duplicate_groups": 0,
        "classified_total": 0,
        "classification_count_matches": False,
        "logo_candidates_before_graph_protection": 0,
        "protected_graph_candidates": 0,
        "dark_graph_candidates": 0,
        "logo_candidates_after_graph_protection": 0,
        "final_logo_candidates": 0,
        "logo_candidates": 0,
        "uncertain_logo_candidates": 0,
        "graph_candidates": 0,
        "table_candidates": 0,
        "diagram_candidates": 0,
        "equation_candidates": 0,
        "photo_candidates": 0,
        "page_decoration_candidates": 0,
        "unknown_candidates": 0,
        "would_quarantine": 0,
        "would_delete": 0,
        "would_reanalyze": 0,
        "would_update_chroma": 0,
        "rename_image_candidates": 0,
        "rename_active_note_candidates": 0,
        "rename_quarantine_candidates": 0,
        "rename_skipped_backups": excluded["backup_notes_excluded"],
        "rename_skipped_temporary": excluded["temporary_files_excluded"],
        "rename_unresolved": 0,
    }
    category_counts = {category: 0 for category in CATEGORIES}

    near_grouped: set[str] = set()
    for stem, left_hash in near_hashes.items():
        if stem in near_grouped:
            continue
        group = [other for other, right_hash in near_hashes.items() if hamming(left_hash, right_hash) <= 8]
        if len(group) > 1:
            stats["near_duplicate_groups"] += 1
            near_grouped.update(group)

    used_targets: set[Path] = set()
    resolved_docs: set[str] = set()
    handled = 0
    for stem, paths in sorted(assets.items()):
        parsed = parse_asset_stem(stem)
        if not parsed:
            continue
        doc_id, page, fig = parsed
        doc_name = documents.get(doc_id)
        if not doc_name:
            stats["unresolved_documents"] += 1
            stats["rename_unresolved"] += 1
            continue
        if document and normalize_key(document) not in normalize_key(doc_name) and normalize_key(document) not in normalize_key(doc_id):
            continue
        if limit is not None and handled >= limit:
            break
        handled += 1
        stats["resolved_documents"] += 1
        stats["resolved_asset_count"] += 1
        resolved_docs.add(doc_id)

        image_path = paths.get("image")
        note_path = paths.get("note")
        if image_path:
            stats["filtered_document_images"] += 1
        if note_path:
            stats["filtered_document_active_notes"] += 1
        if image_path and note_path:
            stats["matched_image_note_pairs"] += 1
        elif image_path:
            stats["orphan_images"] += 1
        elif note_path:
            stats["orphan_notes"] += 1

        note_text = note_path.read_text(encoding="utf-8", errors="replace") if note_path else ""
        note_fields = parse_key_values(note_text) if note_text else {}
        metrics = image_metrics(image_path) if image_path else {}
        duplicate_count = len(exact_groups.get(image_hashes.get(stem, ""), []))
        page_count = len(duplicate_pages.get(stem, set()))
        raw_type = str(note_fields.get("image_type") or note_fields.get("image type") or "").lower()
        if any(kind in raw_type for kind in LOGO_TYPES):
            stats["logo_candidates_before_graph_protection"] += 1
        classification, confidence, reasons, protections, review_bucket = classify_asset(
            note_fields,
            metrics,
            duplicate_count,
            page_count,
            note_text,
        )
        if image_path:
            category_counts[classification] += 1
            stats["classified_total"] += 1
        if protections:
            stats["protected_graph_candidates"] += 1
        if "dark_graph" in protections:
            stats["dark_graph_candidates"] += 1
        if classification == "logo":
            stats["logo_candidates"] += 1
            stats["logo_candidates_after_graph_protection"] += 1
        elif classification == "graph":
            stats["graph_candidates"] += 1
        elif classification == "table":
            stats["table_candidates"] += 1
        elif classification == "diagram":
            stats["diagram_candidates"] += 1
        elif classification == "equation":
            stats["equation_candidates"] += 1
        elif classification == "photo":
            stats["photo_candidates"] += 1
        elif classification == "page_decoration":
            stats["page_decoration_candidates"] += 1
        elif classification == "unknown":
            stats["unknown_candidates"] += 1
        if review_bucket == "uncertain_logo":
            stats["uncertain_logo_candidates"] += 1

        new_stem = target_stem(doc_name, doc_id, page, fig)
        image_target = settings.figures_dir / f"{new_stem}{image_path.suffix.lower()}" if image_path else None
        note_target = settings.figure_notes_dir / f"{new_stem}.md" if note_path else None
        quarantine = False
        if (image_path and image_path != image_target) or (note_path and note_path != note_target):
            stats["rename_candidates"] += 1
            if len(rename_examples) < 20:
                rename_examples.append({"old": stem, "new": new_stem})

        for target in [image_target, note_target]:
            if not target:
                continue
            current = image_path if target.suffix.lower() in IMAGE_SUFFIXES else note_path
            if target in used_targets or (target.exists() and target != current):
                collisions.append(str(target))
            used_targets.add(target)

        if classification in GRAPH_TYPES and reanalyze_graphs:
            stats["would_reanalyze"] += 1
        if update_chroma:
            stats["would_update_chroma"] += 1

        operations.append(
            {
                "stem": stem,
                "group_id": exact_group_id(image_hashes.get(stem, "")) if image_path else "",
                "canonical_group_id": stem_to_near.get(stem, ""),
                "source_groups": sorted(
                    {exact_group_id(image_hashes.get(item, "")) for item in near_groups.get(stem_to_near.get(stem, ""), [])}
                ),
                "representative_path": str(paths.get("image") or ""),
                "exact_duplicate_count": duplicate_count,
                "near_duplicate_count": len(near_groups.get(stem_to_near.get(stem, ""), [])),
                "document_id": doc_id,
                "document_name": doc_name,
                "page": page,
                "figure": fig,
                "classification": classification,
                "confidence": confidence,
                "duplicate_count": duplicate_count,
                "page_count": page_count,
                "metrics": metrics,
                "detected_text": " ".join(str(note_fields.get(key, "")) for key in ["image_type", "title", "analysis", "trend"]).strip(),
                "classification_reasons": reasons,
                "protection_reasons": protections,
                "review_bucket": review_bucket,
                "old_image": str(image_path) if image_path else None,
                "new_image": str(image_target) if image_target else None,
                "old_note": str(note_path) if note_path else None,
                "new_note": str(note_target) if note_target else None,
                "old_note_text": note_path.read_text(encoding="utf-8", errors="replace") if note_path else None,
                "quarantine": quarantine,
                "reanalyze": classification in GRAPH_TYPES and reanalyze_graphs,
                "update_chroma": update_chroma,
                "status": "planned",
            }
        )

    stats["resolved_document_count"] = len(resolved_docs)
    stats["classification_count_matches"] = stats["classified_total"] == stats["filtered_document_images"]
    for category, count in category_counts.items():
        stats[f"{category}_classified"] = count
    apply_manual_decisions(settings, operations, decisions, quarantine_logos)
    collisions = detect_collisions(operations)
    stats["name_collisions"] = len(collisions)
    stats["would_quarantine"] = sum(1 for op in operations if op["quarantine"])
    stats["final_logo_candidates"] = sum(1 for op in operations if op["quarantine"] and op.get("classification") == "logo")
    stats["rename_quarantine_candidates"] = stats["would_quarantine"]
    stats["rename_image_candidates"] = sum(1 for op in operations if op.get("old_image") and op.get("old_image") != op.get("new_image"))
    stats["rename_active_note_candidates"] = sum(1 for op in operations if op.get("old_note") and op.get("old_note") != op.get("new_note"))
    quarantine_examples[:] = [op["stem"] for op in operations if op["quarantine"]][:20]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "rename_examples": rename_examples,
        "quarantine_examples": quarantine_examples,
        "collisions": collisions,
        "operations": operations,
    }


async def apply_plan(settings: Settings, manifest: dict[str, Any]) -> None:
    ollama = OllamaClient(settings)
    for op in manifest["operations"]:
        if op["quarantine"] or op["old_image"] != op["new_image"]:
            move_path(op["old_image"], op["new_image"])
        if op["old_note"]:
            old_note = Path(op["old_note"])
            new_note = Path(op["new_note"])
            new_note.parent.mkdir(parents=True, exist_ok=True)
            note_text = op["old_note_text"] or ""
            replacements = {
                "document_name": op["document_name"],
                "document_id": op["document_id"],
                "image_path": op["new_image"] or "",
                "page_number": str(op["page"]),
                "image_index": str(op["figure"]),
            }
            note_text = update_note_text(note_text, replacements)
            if op["reanalyze"] and op["new_image"]:
                tmp_image = prepare_vision_image(Path(op["new_image"]))
                try:
                    vision_text = await ollama.describe_image(
                        tmp_image,
                        prompt=prompt_for_classification(str(op.get("classification") or "")),
                    )
                finally:
                    if tmp_image.exists():
                        tmp_image.unlink()
                if vision_text.strip():
                    backup = new_note.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.md")
                    note_text = graph_note_text(
                        document_name=op["document_name"],
                        document_id=op["document_id"],
                        page=int(op["page"]),
                        fig=int(op["figure"]),
                        image_path=Path(op["new_image"]),
                        vision_text=vision_text,
                        settings=settings,
                        backup_path=backup,
                    )
            if old_note != new_note and old_note.exists():
                if new_note.exists():
                    raise FileExistsError(new_note)
                old_note.rename(new_note)
            new_note.write_text(note_text, encoding="utf-8")
    if any(op.get("update_chroma") for op in manifest["operations"]):
        update_chroma_references(settings, manifest["operations"])


def update_chroma_references(settings: Settings, operations: list[dict[str, Any]]) -> int:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
    client = chromadb.PersistentClient(
        path=str(settings.vector_db_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(settings.collection_name)
    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    updated = 0

    for item_id, document, metadata in zip(ids, documents, metadatas):
        new_document = str(document or "")
        new_metadata = dict(metadata or {})
        changed = False
        for op in operations:
            replacements = {
                op.get("old_image"): op.get("new_image"),
                op.get("old_note"): op.get("new_note"),
                op.get("stem"): Path(str(op.get("new_note") or op.get("new_image") or op.get("stem"))).stem,
            }
            for old, new in replacements.items():
                if not old or not new:
                    continue
                if old in new_document:
                    new_document = new_document.replace(str(old), str(new))
                    changed = True
                for key, value in list(new_metadata.items()):
                    if isinstance(value, str) and old in value:
                        new_metadata[key] = value.replace(str(old), str(new))
                        changed = True
        if changed:
            collection.update(ids=[item_id], documents=[new_document], metadatas=[new_metadata])
            updated += 1
    print(f"updated_chroma_records={updated}")
    return updated


def move_path(old_value: str | None, new_value: str | None) -> None:
    if not old_value or not new_value:
        return
    old_path = Path(old_value)
    new_path = Path(new_value)
    if old_path == new_path or not old_path.exists():
        return
    if new_path.exists():
        raise FileExistsError(new_path)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.rename(new_path)


def restore_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for op in reversed(manifest.get("operations", [])):
        move_path(op.get("new_image"), op.get("old_image"))
        new_note = Path(op["new_note"]) if op.get("new_note") else None
        old_note = Path(op["old_note"]) if op.get("old_note") else None
        if new_note and old_note and new_note.exists():
            old_note.parent.mkdir(parents=True, exist_ok=True)
            if old_note.exists():
                raise FileExistsError(old_note)
            new_note.rename(old_note)
            old_note.write_text(op.get("old_note_text") or "", encoding="utf-8")


def write_manifest(settings: Settings, manifest: dict[str, Any], path: Path | None) -> Path:
    target = path or settings.data_dir / "migrations" / f"figure_assets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def print_report(manifest: dict[str, Any]) -> None:
    stats = manifest["stats"]
    for key, value in stats.items():
        if key == "resolved_documents":
            continue
        print(f"{key}={value}")
    print("\nrename_examples:")
    for item in manifest["rename_examples"][:20]:
        print(f"OLD: {item['old']}")
        print(f"NEW: {item['new']}")
    print("\nquarantine_examples:")
    for item in manifest["quarantine_examples"][:20]:
        print(item)
    for collision in manifest["collisions"][:20]:
        print(f"COLLISION: {collision}")


def thumbnail_data_uri(path_value: str | None) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists():
        return ""
    try:
        with Image.open(path) as image:
            image.thumbnail((220, 160))
            from io import BytesIO

            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=75)
    except (OSError, UnidentifiedImageError):
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def group_summaries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for op in manifest["operations"]:
        key = str(op.get("canonical_group_id") or op.get("group_id") or op.get("stem"))
        groups.setdefault(key, []).append(op)
    summaries: list[dict[str, Any]] = []
    for group_id, ops in groups.items():
        representative = next((op for op in ops if op.get("old_image")), ops[0])
        pages = sorted({int(op["page"]) for op in ops if op.get("page") is not None})
        classifications = sorted({str(op.get("classification")) for op in ops})
        summary = {
            "group_id": group_id,
            "representative_path": representative.get("old_image") or representative.get("old_note") or "",
            "planned_path": representative.get("new_image") or representative.get("new_note") or "",
            "exact_duplicate_count": max(int(op.get("exact_duplicate_count") or 0) for op in ops),
            "near_duplicate_count": len(ops),
            "page_count": len(pages),
            "pages": f"{pages[0]}-{pages[-1]}" if pages else "",
            "classification": classifications[0] if len(classifications) == 1 else "mixed",
            "confidence": max(float(op.get("confidence") or 0) for op in ops),
            "suggested_action": "quarantine" if any(op.get("quarantine") for op in ops) else "rename",
            "source_groups": sorted({group for op in ops for group in op.get("source_groups", [])}),
            "classification_reasons": sorted({reason for op in ops for reason in op.get("classification_reasons", [])}),
            "protection_reasons": sorted({reason for op in ops for reason in op.get("protection_reasons", [])}),
            "metrics": representative.get("metrics") or {},
            "detected_text": representative.get("detected_text") or "",
            "items": len(ops),
        }
        summaries.append(summary)
    return sorted(summaries, key=lambda item: (-int(item["items"]), str(item["group_id"])))


def export_review_report(manifest: dict[str, Any], path: Path) -> None:
    summaries = group_summaries(manifest)
    sections = [
        ("Final logo quarantine candidates", lambda item: item["suggested_action"] == "quarantine"),
        ("Uncertain logo candidates", lambda item: "insufficient_logo_evidence" in item["classification_reasons"]),
        ("Protected graph candidates", lambda item: bool(item["protection_reasons"])),
        ("Unknown candidates", lambda item: item["classification"] == "unknown"),
        ("Orphan image", lambda item: any(op.get("old_image") and not op.get("old_note") for op in manifest["operations"] if (op.get("canonical_group_id") or op.get("group_id")) == item["group_id"])),
        ("Rename before/after", lambda item: item["representative_path"] != item["planned_path"]),
    ]
    rows: list[str] = [
        "<!doctype html><meta charset='utf-8'><title>Figure Asset Review</title>",
        "<style>body{font-family:Arial,sans-serif} table{border-collapse:collapse;width:100%;margin-bottom:32px}"
        "td,th{border:1px solid #ddd;padding:6px;vertical-align:top;font-size:12px} img{max-width:220px;max-height:160px}</style>",
        "<h1>Figure Asset Review</h1>",
    ]
    for title, predicate in sections:
        matching = [item for item in summaries if predicate(item)]
        displayed = matching[:200]
        rows.append(f"<h2>{html.escape(title)}</h2>")
        rows.append(f"<p>Showing {len(displayed)} of {len(matching)} groups</p>")
        rows.append(
            "<table><tr><th>thumbnail</th><th>group_id</th><th>representative_path</th><th>planned_path</th>"
            "<th>classification</th><th>confidence</th><th>exact</th><th>near</th><th>pages</th><th>metrics</th>"
            "<th>detected_text</th><th>classification_reasons</th><th>protection_reasons</th></tr>"
        )
        for item in displayed:
            metrics = item.get("metrics") or {}
            image_uri = thumbnail_data_uri(item.get("representative_path"))
            thumbnail = f'<img src="{image_uri}">' if image_uri else ""
            rows.append(
                "<tr>"
                f"<td>{thumbnail}</td>"
                f"<td>{html.escape(str(item.get('group_id') or ''))}</td>"
                f"<td>{html.escape(str(item.get('representative_path') or ''))}</td>"
                f"<td>{html.escape(str(item.get('planned_path') or ''))}</td>"
                f"<td>{html.escape(str(item.get('classification') or ''))}</td>"
                f"<td>{html.escape(str(item.get('confidence') or ''))}</td>"
                f"<td>{html.escape(str(item.get('exact_duplicate_count') or 0))}</td>"
                f"<td>{html.escape(str(item.get('near_duplicate_count') or 0))}</td>"
                f"<td>{html.escape(str(item.get('pages') or ''))}</td>"
                f"<td>aspect_ratio={metrics.get('aspect_ratio')}<br>contrast={metrics.get('contrast')}"
                f"<br>edge_density={metrics.get('edge_density')}</td>"
                f"<td>{html.escape(str(item.get('detected_text') or ''))}</td>"
                f"<td>{html.escape(', '.join(item.get('classification_reasons') or []))}</td>"
                f"<td>{html.escape(', '.join(item.get('protection_reasons') or []))}</td>"
                "</tr>"
            )
        rows.append("</table>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")
    json_path = path.with_suffix(".json")
    csv_path = path.with_suffix(".csv")
    json_path.write_text(
        json.dumps({"stats": manifest["stats"], "groups": summaries, "operations": manifest["operations"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "group_id",
            "representative_path",
            "planned_path",
            "exact_duplicate_count",
            "near_duplicate_count",
            "page_count",
            "pages",
            "classification",
            "confidence",
            "suggested_action",
            "source_groups",
            "classification_reasons",
            "protection_reasons",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: json.dumps(item.get(key), ensure_ascii=False) if isinstance(item.get(key), list) else item.get(key) for key in fieldnames})


def delete_quarantined(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for op in manifest.get("operations", []):
        if not op.get("quarantine"):
            continue
        for key in ["new_image", "new_note"]:
            path = Path(op[key]) if op.get(key) else None
            if path and path.exists():
                path.unlink()


async def main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Default. Build a plan without changing files.")
    parser.add_argument("--apply", action="store_true", help="Apply the planned rename/quarantine/reanalysis changes.")
    parser.add_argument("--document")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--quarantine-logos", action="store_true")
    parser.add_argument("--delete-quarantined", action="store_true")
    parser.add_argument("--reanalyze-graphs", action="store_true")
    parser.add_argument("--update-chroma", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--export-review-report", type=Path)
    args = parser.parse_args()

    settings = Settings()
    if args.restore:
        if not args.apply:
            print("restore requires --apply")
            return 2
        restore_manifest(args.restore)
        print(f"restored={args.restore}")
        return 0

    if args.delete_quarantined:
        if not args.apply or not args.manifest or not args.manifest.exists():
            print("delete requires --apply --manifest existing_manifest.json")
            return 2
        delete_quarantined(args.manifest)
        print(f"deleted_quarantined_from={args.manifest}")
        return 0

    manifest = build_plan(
        settings,
        document=args.document,
        limit=args.limit,
        quarantine_logos=args.quarantine_logos,
        reanalyze_graphs=args.reanalyze_graphs,
        update_chroma=args.update_chroma,
        decisions=load_decisions(args.decisions),
    )
    print_report(manifest)
    if args.export_review_report:
        export_review_report(manifest, args.export_review_report)
        print(f"review_report={args.export_review_report}")
        print(f"review_json={args.export_review_report.with_suffix('.json')}")
        print(f"review_csv={args.export_review_report.with_suffix('.csv')}")

    if args.apply:
        if manifest["collisions"]:
            print("refusing to apply with name collisions")
            return 2
        manifest_path = write_manifest(settings, manifest, args.manifest)
        await apply_plan(settings, manifest)
        print(f"manifest={manifest_path}")
    elif args.manifest:
        print(f"dry_run_manifest_not_written={args.manifest}")
    return 0


def main() -> int:
    import asyncio

    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
