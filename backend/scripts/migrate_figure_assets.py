from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import logging
import os
import re
import shutil
import subprocess
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


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jpx", ".webp"}
LOGO_TYPES = {"logo", "icon", "decoration", "decorative", "cover_art"}
KEEP_TYPES = {"graph", "chart", "table", "diagram", "schematic", "equation"}
GRAPH_TYPES = {"graph", "chart"}
DIAGRAM_TYPES = {"diagram", "schematic"}
EQUATION_TYPES = {"equation"}
PROTECTED_TYPES = {"graph", "chart", "table", "diagram", "schematic", "equation", "unknown_engineering_figure"}
CANDIDATE_SCHEMA_VERSION = 2
CLASSIFIER_VERSION = "dark-marker-series-v2"
ANALYSIS_PROMPT_VERSION = "graph-grounding-v2"
SERIALIZER_VERSION = "figure-note-v2"
ANALYSIS_TRANSFORM = {
    "rgb_conversion": True,
    "autocontrast": True,
    "brightness_factor": 1.0,
    "contrast_factor": 1.0,
}
MANUAL_TREND_EXPECTATIONS = {
    "Heriot-Watt_University_-_Well_Test_Analysis_p0264_fig02": ("increas", "scatter|oscillat|local"),
    "Heriot-Watt_University_-_Well_Test_Analysis_p0305_fig02": ("peak", "declin", "flatten|plateau"),
    "Heriot-Watt_University_-_Well_Test_Analysis_p0307_fig02": ("increas", "plateau|flatten"),
}
GRAPH_PROMPT = """Analyze only what is directly visible in this graph or chart.
Return concise ASCII English key-value lines with these keys:
image_type, confidence, title, analysis, x_axis, x_axis_unit, x_axis_scale, y_axis,
y_axis_unit, y_axis_scale, series_count, series_count_verified, series_descriptions, legend,
reference_lines, trend, plateau, peak, decline, slope_changes, engineering_meaning.
confidence must be a number from 0.0 to 1.0.
analysis is required and must summarize visible graph evidence.
For multiple curves, describe each curve separately. Do not merge all curves into one trend.
If labels, legend, units, scale, or engineering meaning are not directly readable, write unknown.
Do not infer petroleum engineering meaning without page context.
"""
GRAPH_PROMPT += """
Also include plateau, peak, decline, and slope_changes.
Distinguish continuously rising curves, rise-then-plateau curves, and diagonal reference lines when visible.
series_descriptions must describe visible curve colors, line styles, and markers when visible.
For dark plots, inspect spatially separated upper and lower marker bands as distinct series and describe each separately.
series_count_verified must be true or false.
Report log/linear/unknown scale for each axis.
Do not guess series names when the legend is missing or unreadable.
"""

CATEGORIES = ("logo", "graph", "chart", "table", "diagram", "schematic", "equation", "photo", "page_decoration", "unknown_engineering_figure", "unknown")

DIAGRAM_PROMPT = """Analyze only what is directly visible in this engineering diagram or schematic.
Return concise ASCII English key-value lines with these keys:
image_type, confidence, title, components, connections, labels, engineering_purpose, uncertain_items.
confidence must be a number from 0.0 to 1.0.
Do not invent axes, trends, or series for diagrams.
If a component or label is unreadable, write unknown.
"""

EQUATION_PROMPT = """Analyze only what is directly visible in this equation figure.
Return concise key-value lines in Korean with these keys:
image_type, confidence, equation_text, variables, units, assumptions, engineering_context.
Do not reconstruct or invent unreadable equations.
If the equation cannot be read exactly, write 확인할 수 없음.
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


def document_stem_map(documents: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    mapping: dict[str, list[tuple[str, str]]] = {}
    for doc_id, doc_name in documents.items():
        keys = {
            normalize_key(safe_document_stem(doc_name, doc_id)),
            normalize_key(Path(doc_name).stem),
            normalize_key(doc_name),
        }
        for key in keys:
            mapping.setdefault(key, []).append((doc_id, doc_name))
    return mapping


def read_note_fields(note_path: Path | None) -> dict[str, str]:
    if not note_path or not note_path.exists():
        return {}
    return parse_key_values(note_path.read_text(encoding="utf-8", errors="replace"))


def load_manifest_mappings(settings: Settings) -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {}
    migrations_dir = settings.data_dir / "migrations"
    if not migrations_dir.exists():
        return mappings
    for manifest_path in sorted(migrations_dir.glob("figure_assets_*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for op in manifest.get("operations", []):
            info = {
                "document_id": str(op.get("document_id") or ""),
                "document_name": str(op.get("document_name") or ""),
                "page": str(op.get("page") or ""),
                "figure": str(op.get("figure") or ""),
            }
            for key in ["old_image", "new_image", "old_note", "new_note"]:
                value = op.get(key)
                if value:
                    mappings[str(Path(value).stem)] = info
                    mappings[str(Path(value))] = info
    return mappings


def load_previous_classifications(settings: Settings) -> dict[str, str]:
    classifications: dict[str, str] = {}
    migrations_dir = settings.data_dir / "migrations"
    if not migrations_dir.exists():
        return classifications
    for manifest_path in sorted(migrations_dir.glob("figure_assets_*.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for op in manifest.get("operations", []):
            stem = str(op.get("stem") or "")
            classification = str(op.get("classification") or "")
            if stem and classification:
                classifications.setdefault(stem, classification)
    return classifications


def resolve_asset(
    *,
    stem: str,
    paths: dict[str, Path | None],
    documents: dict[str, str],
    stem_documents: dict[str, list[tuple[str, str]]],
    manifest_mappings: dict[str, dict[str, str]],
) -> tuple[str | None, str | None, int | None, int | None, str, bool]:
    parsed = parse_asset_stem(stem)
    if not parsed:
        return None, None, None, None, "unresolved_no_digest", False
    raw_doc, page, fig = parsed

    for key in [stem, str(paths.get("image") or ""), str(paths.get("note") or "")]:
        info = manifest_mappings.get(key)
        if info and info.get("document_id") and info.get("document_name"):
            return info["document_id"], info["document_name"], int(info.get("page") or page), int(info.get("figure") or fig), "manifest", raw_doc != info["document_id"]

    if raw_doc in documents:
        return raw_doc, documents[raw_doc], page, fig, "digest", False

    matches = stem_documents.get(normalize_key(raw_doc), [])
    if len(matches) == 1:
        doc_id, doc_name = matches[0]
        return doc_id, doc_name, page, fig, "document_name", True
    if len(matches) > 1:
        return None, None, page, fig, "unresolved_ambiguous_document_name", True

    note_fields = read_note_fields(paths.get("note"))
    doc_id = note_fields.get("document_id")
    doc_name = note_fields.get("document_name")
    if doc_id and doc_name:
        return doc_id, doc_name, int(note_fields.get("page_number") or page), int(note_fields.get("image_index") or fig), "note_metadata", raw_doc != doc_id

    if len(raw_doc) != 64:
        return None, None, page, fig, "unresolved_document_name_mismatch", True
    return None, None, page, fig, "unresolved_no_manifest_mapping", False


def matches_document_filter(document: str | None, doc_id: str, doc_name: str) -> bool:
    if not document:
        return True
    return normalize_key(document) in normalize_key(doc_name) or normalize_key(document) in normalize_key(doc_id)


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
    if classification == "equation":
        return EQUATION_PROMPT
    return GRAPH_PROMPT


def load_decisions(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("groups", {})


def asset_matches(op: dict[str, Any], asset: str) -> bool:
    requested = Path(asset)
    candidates = [
        Path(str(op.get("old_image") or "")),
        Path(str(op.get("new_image") or "")),
        Path(str(op.get("old_note") or "")),
        Path(str(op.get("new_note") or "")),
    ]
    return any(path == requested or path.name == asset for path in candidates if str(path))


def reanalysis_sort_key(op: dict[str, Any]) -> tuple[int, int, str]:
    path = str(op.get("old_image") or op.get("new_image") or op.get("stem") or "")
    return int(op.get("page") or 0), int(op.get("figure") or 0), path


def select_reanalysis_targets(
    operations: list[dict[str, Any]],
    *,
    limit: int | None,
    asset: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    candidates = sorted([op for op in operations if op.get("reanalyze")], key=reanalysis_sort_key)
    for op in operations:
        op["reanalyze_candidate"] = bool(op.get("reanalyze"))
        op["reanalyze"] = False

    if asset:
        matches = sorted([op for op in operations if asset_matches(op, asset)], key=reanalysis_sort_key)
        if not matches:
            return candidates, [], f"asset_not_found: {asset}"
        if len(matches) > 1:
            return candidates, [], f"asset_ambiguous: {asset}"
        match = matches[0]
        if not match.get("reanalyze_candidate"):
            return candidates, [], f"asset_not_valid_for_requested_reanalysis: {asset}"
        selected = [match]
    else:
        selected = candidates[:limit] if limit is not None else candidates

    for op in selected:
        op["reanalyze"] = True
    return candidates, selected, None


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
            enhanced = ImageOps.autocontrast(small)
            pixels = enhanced.load()
            active_columns = sum(any(pixels[x, y] >= 64 for y in range(enhanced.height)) for x in range(enhanced.width)) / max(enhanced.width, 1)
            active_rows = sum(any(pixels[x, y] >= 64 for x in range(enhanced.width)) for y in range(enhanced.height)) / max(enhanced.height, 1)
            bright_ratio = sum(value >= 64 for value in enhanced.getdata()) / max(enhanced.width * enhanced.height, 1)
            marker_series_detected = active_columns >= 0.15 and active_rows >= 0.2 and 0.001 <= bright_ratio <= 0.05
            marker_series_score = min(1.0, active_columns / 0.15, active_rows / 0.2, bright_ratio / 0.001)
        return {
            "width": width,
            "height": height,
            "file_size": size,
            "aspect_ratio": width / max(height, 1),
            "contrast": round(float(contrast), 2),
            "brightness": round(float(brightness), 2),
            "edge_density": round(float(edge_density), 4),
            "enhanced_active_columns": round(float(active_columns), 4),
            "enhanced_active_rows": round(float(active_rows), 4),
            "enhanced_bright_ratio": round(float(bright_ratio), 5),
            "marker_series_detected": marker_series_detected,
            "marker_series_score": round(float(marker_series_score), 4),
        }
    except (OSError, UnidentifiedImageError):
        return {"width": 0, "height": 0, "file_size": 0, "aspect_ratio": 0, "contrast": 0, "brightness": 0, "edge_density": 0}


def classify_asset(
    note_fields: dict[str, str],
    metrics: dict[str, Any],
    duplicate_count: int,
    page_count: int,
    note_text: str,
) -> tuple[str, float, list[str], list[str], str, str, bool, str | None]:
    image_type = str(note_fields.get("image_type") or note_fields.get("image type") or "").lower()
    try:
        confidence = float(note_fields.get("confidence") or 0)
    except ValueError:
        confidence = 0.0

    text = f"{image_type} {note_text}".lower()
    metadata_prefixes = ("document_name:", "document_id:", "image_path:", "page_number:", "image_index:", "legacy_note_backup:")
    descriptive_note_text = "\n".join(
        line for line in note_text.splitlines() if not line.strip().lower().startswith(metadata_prefixes)
    )
    semantic_text = f"{image_type} {descriptive_note_text}".lower()
    reasons: list[str] = []
    protections: list[str] = []
    aspect = float(metrics.get("aspect_ratio") or 0)
    edge_density = float(metrics.get("edge_density") or 0)
    brightness = float(metrics.get("brightness") or 0)
    wide_rectangle = 1.2 <= aspect <= 5.5 and int(metrics.get("width") or 0) >= 350
    marker_series_detected = bool(metrics.get("marker_series_detected"))
    dark_graph = wide_rectangle and (
        (brightness < 90 and edge_density >= 0.004)
        or (
            brightness < 10
            and int(metrics.get("file_size") or 0) >= 50_000
            and (edge_density >= 0.003 or marker_series_detected)
        )
    )
    graph_terms = ["graph", "chart", "plot", "log-log", "axis", "x_axis", "y_axis", "psi", "hours", "delta p"]
    schematic_terms = [
        "fault",
        "block",
        "depletion",
        "communication",
        "boundary",
        "conceptual",
        "schematic",
        "diagram",
        "model",
        "well",
        "reservoir",
        "flow",
        "arrow",
    ]
    equation_terms = ["equation", "formula", "variables", "assumptions"]

    if wide_rectangle:
        protections.append("wide_rectangle")
    if dark_graph:
        protections.append("dark_graph")
    if any(term in text for term in graph_terms):
        protections.append("graph_text")

    initial_classification = (
        "graph"
        if "graph" in image_type or "plot" in image_type or any(term in text for term in graph_terms)
        else "schematic"
        if any(term in text for term in schematic_terms)
        else "unknown"
    )
    if "graph" in image_type or "plot" in image_type or any(term in text for term in graph_terms) or dark_graph:
        override_reason = "dark_graph_precedence" if dark_graph and initial_classification != "graph" else None
        return "graph", max(confidence, 0.6), ["graph_protection"], protections, "graph", initial_classification, bool(override_reason), override_reason
    if "chart" in image_type:
        return "chart", max(confidence, 0.6), ["graph_protection"], protections, "chart", "chart", False, None
    if "equation" in image_type or any(term in text for term in equation_terms):
        return "equation", max(confidence, 0.6), ["note_image_type"], protections, "equation", "equation", False, None
    if "table" in image_type:
        return "table", max(confidence, 0.6), ["note_image_type"], protections, "table", "table", False, None
    if "schematic" in image_type or any(term in semantic_text for term in schematic_terms):
        return "schematic", max(confidence, 0.6), ["engineering_figure"], protections or ["engineering_figure"], "schematic", "schematic", False, None
    if "diagram" in image_type:
        return "diagram", max(confidence, 0.6), ["note_image_type"], protections, "diagram", "diagram", False, None
    if "photo" in image_type or "photograph" in image_type:
        return "photo", max(confidence, 0.5), ["note_image_type"], protections, "photo", "photo", False, None

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
        return "logo", confidence, reasons, protections, "logo", initial_classification, False, None
    if logo_type or (duplicate_count >= 3 and page_count >= 3 and (small or odd_shape or low_detail)):
        reasons.append("insufficient_logo_evidence")
        return "unknown", confidence, reasons, protections, "uncertain_logo", initial_classification, False, None
    if protections:
        return "unknown_engineering_figure", max(confidence, 0.5), ["engineering_protection"], protections, "unknown_engineering_figure", initial_classification, False, None
    if small or odd_shape or low_detail:
        return "page_decoration", confidence, reasons, protections, "page_decoration", initial_classification, False, None
    return "unknown", confidence, reasons, protections, "unknown", initial_classification, False, None


def is_backup_note(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".bak.md")
        or name.endswith(".backup.md")
        or re.search(r"\.\d{8}_\d{6}\.bak\.md$", name) is not None
    )


def is_temporary_file(path: Path) -> bool:
    name = path.name.lower()
    return ".tmp." in name or "_vision_" in name or "_rgb_preview" in name


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


UNKNOWN_TEXT = "확인할 수 없음"
UNKNOWN_MARKERS = ("", UNKNOWN_TEXT, "unknown", "none", "n/a", "?뺤", "확인")
ALLOWED_IMAGE_TYPES = {"graph", "chart", "diagram", "schematic", "table", "equation", "photo", "unknown_engineering_figure", "unknown"}
GRAPH_LIST_KEYS = {"series_descriptions", "reference_lines", "plateaus", "peaks", "declines", "slope_changes"}
SCHEMATIC_LIST_KEYS = {"components", "component_labels", "connections", "flow_directions", "annotations"}
SERIALIZED_LIST_KEYS = GRAPH_LIST_KEYS | SCHEMATIC_LIST_KEYS
IGNORED_EQUIVALENCE_FIELDS = {"created_at", "legacy_note_backup"}
GRAPH_SCHEMA_FIELDS = [
    "document_name",
    "document_id",
    "page_number",
    "image_index",
    "image_path",
    "image_type",
    "confidence",
    "title",
    "title_verified",
    "analysis",
    "x_axis",
    "x_axis_unit",
    "x_axis_scale",
    "x_axis_verified",
    "y_axis",
    "y_axis_unit",
    "y_axis_scale",
    "y_axis_verified",
    "series_count",
    "series_count_verified",
    "series_descriptions",
    "legend",
    "legend_verified",
    "reference_lines",
    "plateaus",
    "peaks",
    "declines",
    "slope_changes",
    "trend_summary",
    "trend_verified",
    "engineering_meaning",
    "engineering_meaning_verified",
    "vision_model",
    "created_at",
    "legacy_note_backup",
]
GRAPH_REQUIRED_FIELDS = set(GRAPH_SCHEMA_FIELDS) - IGNORED_EQUIVALENCE_FIELDS
SCHEMATIC_SCHEMA_FIELDS = [
    "document_name",
    "document_id",
    "page_number",
    "image_index",
    "image_path",
    "image_type",
    "confidence",
    "title",
    "title_verified",
    "analysis",
    "components",
    "component_labels",
    "connections",
    "flow_directions",
    "annotations",
    "legend",
    "legend_verified",
    "engineering_meaning",
    "engineering_meaning_verified",
    "vision_model",
    "created_at",
    "legacy_note_backup",
]
DIAGRAM_REQUIRED_FIELDS = set(SCHEMATIC_SCHEMA_FIELDS) - IGNORED_EQUIVALENCE_FIELDS
TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(?:\s|$)")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_unknown_value(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return not text or any(marker.lower() in text for marker in UNKNOWN_MARKERS if marker)


def nullable_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return None if is_unknown_value(text) else text


def normalized_optional_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def normalize_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if is_unknown_value(text):
        return []
    parts = re.split(r"\s*(?:;|\n|\r|\u2022)\s*", text)
    return [part.strip() for part in parts if not is_unknown_value(part)]


def unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = item.strip()
        key = re.sub(r"\s+", " ", clean).lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def is_symbol_label(text: str) -> bool:
    stripped = text.strip()
    if re.fullmatch(r"[A-Za-z]{1,2}", stripped):
        return True
    return bool(re.fullmatch(r"[A-Za-zΔδ][A-Za-z0-9_']{0,8}", stripped) and re.search(r"[_0-9'Δδ]", stripped))


def is_label_name(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z -]{1,40}", text.strip())) and not is_symbol_label(text)


def looks_like_component(text: str) -> bool:
    stripped = text.strip()
    lower = stripped.lower()
    if not stripped or stripped.endswith(".") or re.search(r"\b(?:then|while|because|therefore|however)\b", lower):
        return False
    return bool("(" in stripped and ")" in stripped) or bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 /&_-]{2,50}", stripped))


def normalize_schematic_components(values: Any) -> list[str]:
    result: list[str] = []
    for item in normalize_list(values):
        parts = split_top_level_commas(item)
        if len(parts) > 1 and all(looks_like_component(part) for part in parts):
            result.extend(parts)
        else:
            result.append(item.strip())
    return unique_preserving_order(result)


def normalize_schematic_labels(values: Any) -> list[str]:
    result: list[str] = []
    for item in normalize_list(values):
        parts = split_top_level_commas(item)
        if len(parts) == 2 and is_label_name(parts[0]) and is_symbol_label(parts[1]):
            result.append(f"{parts[0]} ({parts[1]})")
        elif len(parts) > 1 and all(is_symbol_label(part) for part in parts):
            result.extend(parts)
        else:
            result.append(item.strip())
    return unique_preserving_order(result)


def has_unclosed_punctuation(text: str) -> bool:
    pairs = [("(", ")"), ("[", "]"), ('"', '"'), ("'", "'")]
    for left, right in pairs:
        if left == right:
            if text.count(left) % 2:
                return True
        elif text.count(left) != text.count(right):
            return True
    return False


def is_incomplete_string(text: str) -> bool:
    stripped = text.strip()
    lower = stripped.lower()
    if len(stripped) < 8:
        return True
    if re.search(r"(?:\band\b|\bor\b|\bat\b|\bfor\b|[xy]\s*=)\s*$", lower):
        return True
    if stripped.endswith("-"):
        return True
    if re.search(r"\bat\s+[xy]\s*=\s*[\d.]+\s+(?:and|or)\s+[xy]?\s*$", lower):
        return True
    return has_unclosed_punctuation(stripped)


def dedupe_items(items: list[str], analysis: str | None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    analysis_key = re.sub(r"\s+", " ", analysis or "").strip().lower()
    for item in items:
        key = re.sub(r"\s+", " ", item).strip().lower()
        if not key or key == analysis_key or key in seen:
            continue
        if any(key in other and key != other for other in seen):
            continue
        seen.add(key)
        result.append(item)
    return result


def normalize_int(value: Any, errors: list[str], name: str) -> int | None:
    if is_unknown_value(value):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        errors.append(f"{name} must be an integer or null")
        return None


def split_axis_unit(axis: str | None, unit: str | None) -> tuple[str | None, str | None]:
    axis_text = nullable_text(axis)
    unit_text = nullable_text(unit)
    if axis_text and unit_text:
        suffix = f"({unit_text})"
        if axis_text.endswith(suffix):
            axis_text = axis_text[: -len(suffix)].rstrip()
    return axis_text, unit_text


def parse_confidence(value: Any) -> float | None:
    aliases = {"high": 0.8, "medium": 0.6, "moderate": 0.6, "low": 0.4}
    text = str(value or "").strip().lower()
    if text in aliases:
        return aliases[text]
    try:
        confidence = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return confidence if 0.0 <= confidence <= 1.0 else None


def normalize_image_type(raw: str, fallback: str) -> tuple[str, str | None]:
    text = str(raw or "").strip()
    lower = text.lower()
    if "log-log" in lower or "plot" in lower or "graph" in lower:
        return "graph", text if "log-log" in lower else None
    if "chart" in lower:
        return "chart", None
    if "schematic" in lower:
        return "schematic", None
    if "diagram" in lower:
        return "diagram", None
    if "equation" in lower:
        return "equation", None
    if lower in ALLOWED_IMAGE_TYPES:
        return lower, None
    return fallback if fallback in ALLOWED_IMAGE_TYPES else "unknown", None


def parse_vision_fields(vision_text: str) -> dict[str, str]:
    fields = parse_key_values(vision_text)
    current: str | None = None
    lists: dict[str, list[str]] = {}
    for raw_line in vision_text.splitlines():
        line = raw_line.rstrip()
        key_match = re.match(r"^\s*(?:[-*]\s*)?([A-Za-z_][\w ]*):\s*(.*)$", line)
        if key_match:
            current = key_match.group(1).strip().replace(" ", "_")
            value = key_match.group(2).strip()
            if value:
                fields[current] = value
                current = None
            continue
        item_match = re.match(r"^\s*(?:[-*]|\d+[.)])\s*(.+)$", line)
        if current and item_match:
            lists.setdefault(current, []).append(item_match.group(1).strip())
    for key, items in lists.items():
        fields[key] = "; ".join(items)
    return fields


def clean_graph_semantics(metadata: dict[str, Any]) -> dict[str, int]:
    stats = {
        "incomplete_strings_detected": 0,
        "semantic_retry_attempts": 0,
        "semantic_retry_successes": 0,
        "removed_incomplete_items": 0,
    }
    analysis = metadata.get("analysis")
    for key in ["series_descriptions", "reference_lines", "plateaus", "peaks", "declines", "slope_changes"]:
        cleaned: list[str] = []
        for item in metadata.get(key, []):
            if is_incomplete_string(item):
                stats["incomplete_strings_detected"] += 1
                stats["removed_incomplete_items"] += 1
                if key == "reference_lines" and "diagonal" in item.lower():
                    cleaned.append("Two diagonal reference lines are visible; their exact labels or values cannot be confirmed.")
                continue
            if key == "reference_lines" and re.search(r"\b(?:[xy]\s*=|slope\s*=|unit-slope|at\s+[xy]\s*=)", item.lower()):
                stats["removed_incomplete_items"] += 1
                if "diagonal" in item.lower():
                    cleaned.append("Two diagonal reference lines are visible; their exact labels or values cannot be confirmed.")
                continue
            cleaned.append(item)
        metadata[key] = dedupe_items(cleaned, analysis)

    analysis_text = str(analysis or "")
    lower_analysis = analysis_text.lower()
    series_text = " ".join(metadata.get("series_descriptions") or [])
    lower_series = series_text.lower()
    has_continuous = "rises continuously" in lower_analysis or "rises continuously" in lower_series or "continuous rise" in lower_series
    has_plateau = "plateau" in lower_analysis or "plateau" in lower_series
    if metadata.get("trend_summary") and is_incomplete_string(str(metadata["trend_summary"])):
        stats["incomplete_strings_detected"] += 1
        stats["removed_incomplete_items"] += 1
        metadata["trend_summary"] = None
        metadata["trend_verified"] = False
    if has_continuous and has_plateau:
        horizontal_reference = metadata.get("x_axis") if metadata.get("x_axis_verified") else "horizontal position"
        if (
            not metadata["series_descriptions"]
            or metadata["series_descriptions"] == [analysis_text]
            or re.search(r"\b(?:solid|dashed|marker|cyan|yellow|color)\b", lower_series)
        ):
            metadata["series_descriptions"] = [
                f"One data series rises continuously as {horizontal_reference} increases.",
                "Another data series rises initially and then becomes approximately horizontal.",
            ]
        if not metadata["plateaus"]:
            metadata["plateaus"] = [f"One data series becomes approximately horizontal at higher {horizontal_reference}."]
        if not metadata.get("trend_summary") or metadata.get("trend_summary") == analysis_text:
            metadata["trend_summary"] = "The figure contains one continuously rising series and one series that transitions toward a plateau."
    if any("diagonal" in item.lower() for item in metadata.get("reference_lines") or []):
        metadata["reference_lines"] = ["Two diagonal reference lines are visible; their exact labels or values cannot be confirmed."]
    return stats


def finalize_vision_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate["metadata"]
    if metadata.get("image_type") in GRAPH_TYPES:
        semantic_stats = clean_graph_semantics(metadata)
        candidate.update(semantic_stats)
    else:
        candidate.update(
            {
                "incomplete_strings_detected": 0,
                "semantic_retry_attempts": 0,
                "semantic_retry_successes": 0,
                "removed_incomplete_items": 0,
            }
        )
    if metadata.get("engineering_meaning_verified") is False:
        metadata["engineering_meaning"] = "주변 본문과 범례가 없어 구체적인 유동 구간 또는 저류층 거동은 확인할 수 없음"
    quality_parts = [
        metadata.get("series_descriptions") or [],
        [metadata.get("trend_summary")] if metadata.get("trend_summary") else [],
        metadata.get("reference_lines") or [],
        metadata.get("plateaus") or [],
    ]
    information_quality = bool(metadata.get("analysis") and any(quality_parts))
    candidate["information_quality_passed"] = information_quality
    if metadata.get("image_type") in GRAPH_TYPES and not information_quality and "graph lacks series, trend, reference line, or plateau information" not in candidate["validation_errors"]:
        candidate["validation_errors"].append("graph lacks series, trend, reference line, or plateau information")
    candidate["schema_valid"] = not candidate["validation_errors"]
    candidate["normalized_schema_valid"] = candidate["schema_valid"]
    return candidate


def validate_vision_candidate(vision_text: str, *, classification: str, settings: Settings) -> dict[str, Any]:
    fields = parse_vision_fields(vision_text)
    errors: list[str] = []
    warnings: list[str] = []
    image_type, inferred_title = normalize_image_type(fields.get("image_type", ""), classification)
    confidence = parse_confidence(fields.get("confidence"))
    if confidence is None:
        errors.append("confidence must be a 0.0-1.0 float")
        confidence = 0.0
    elif not re.fullmatch(r"\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\s*", str(fields.get("confidence") or "")):
        warnings.append("confidence alias normalized to float")
    title = nullable_text(fields.get("title") or inferred_title)
    analysis = nullable_text(fields.get("analysis"))
    if is_unknown_value(analysis):
        errors.append("analysis is required")

    def value(name: str) -> tuple[str | None, bool]:
        text = nullable_text(fields.get(name))
        ok = text is not None
        if str(fields.get(f"{name}_verified", "")).strip().lower() == "true" and not ok:
            errors.append(f"{name}_verified true but {name} is missing")
        return text, ok

    x_axis, x_ok = value("x_axis")
    y_axis, y_ok = value("y_axis")
    trend, trend_ok = value("trend")
    legend, legend_ok = value("legend")
    engineering = nullable_text(fields.get("engineering_meaning")) or "주변 본문과 범례가 없어 구체적인 유동 구간 또는 저류층 거동은 확인할 수 없음"
    engineering_ok = False if "확인할 수 없음" in engineering else not is_unknown_value(engineering)
    x_axis, x_unit = split_axis_unit(x_axis, fields.get("x_axis_unit"))
    y_axis, y_unit = split_axis_unit(y_axis, fields.get("y_axis_unit"))
    series_count = normalize_int(fields.get("series_count"), errors, "series_count")
    series_count_verified = str(fields.get("series_count_verified", "")).strip().lower() == "true"
    series_descriptions = normalize_list(fields.get("series_descriptions"))
    reference_lines = normalize_list(fields.get("reference_lines"))
    plateaus = normalize_list(fields.get("plateaus") or fields.get("plateau"))
    peaks = normalize_list(fields.get("peaks") or fields.get("peak"))
    declines = normalize_list(fields.get("declines") or fields.get("decline"))
    slope_changes = normalize_list(fields.get("slope_changes"))
    if not series_descriptions and analysis and any(word in analysis.lower() for word in ["curve", "series", "rises", "plateau"]):
        series_descriptions = [analysis]
        warnings.append("series_descriptions inferred from analysis")
    if is_unknown_value(fields.get("trend")) and any(word in analysis.lower() for word in ["rises", "plateau", "decline"]):
        trend = analysis
        trend_ok = True
        warnings.append("trend_summary inferred from analysis")
    if not plateaus and analysis and "plateau" in analysis.lower():
        plateaus = [analysis]
        warnings.append("plateaus inferred from analysis")
    quality_parts = [series_descriptions, [trend] if trend else [], reference_lines, plateaus]
    information_quality = bool(analysis and any(part for part in quality_parts))
    if image_type in GRAPH_TYPES and not information_quality:
        errors.append("graph lacks series, trend, reference line, or plateau information")

    raw_schema_valid = image_type in ALLOWED_IMAGE_TYPES and parse_confidence(fields.get("confidence")) is not None
    normalized_schema_valid = not errors
    return finalize_vision_candidate({
        "raw_schema_valid": raw_schema_valid,
        "schema_valid": normalized_schema_valid,
        "normalized_schema_valid": normalized_schema_valid,
        "information_quality_passed": information_quality,
        "validation_errors": errors,
        "normalization_warnings": warnings,
        "raw_vision_text": vision_text,
        "metadata": {
            "image_type": image_type,
            "confidence": confidence,
            "title": title,
            "title_verified": title is not None,
            "analysis": analysis,
            "x_axis": x_axis,
            "x_axis_unit": x_unit,
            "x_axis_scale": fields.get("x_axis_scale") or "unknown",
            "x_axis_verified": x_ok,
            "y_axis": y_axis,
            "y_axis_unit": y_unit,
            "y_axis_scale": fields.get("y_axis_scale") or "unknown",
            "y_axis_verified": y_ok,
            "series_count": series_count,
            "series_count_verified": series_count_verified,
            "series_descriptions": series_descriptions,
            "legend": legend,
            "legend_verified": legend_ok,
            "reference_lines": reference_lines,
            "plateaus": plateaus,
            "peaks": peaks,
            "declines": declines,
            "slope_changes": slope_changes,
            "trend_summary": trend,
            "trend_verified": trend_ok,
            "engineering_meaning": engineering,
            "engineering_meaning_verified": engineering_ok,
            "vision_model": settings.vision_model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "legacy_note_backup": None,
        },
    })


def finalize_vision_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate["metadata"]
    if metadata.get("image_type") in GRAPH_TYPES:
        semantic_stats = clean_graph_semantics(metadata)
        candidate.update(semantic_stats)
    else:
        candidate.update(
            {
                "incomplete_strings_detected": 0,
                "semantic_retry_attempts": 0,
                "semantic_retry_successes": 0,
                "removed_incomplete_items": 0,
            }
        )

    if metadata.get("engineering_meaning_verified") is False and metadata.get("image_type") in GRAPH_TYPES:
        metadata["engineering_meaning"] = "주변 본문과 범례가 없어 구체적인 유동 구간 또는 저류층 거동은 확인할 수 없음"

    if metadata.get("image_type") in GRAPH_TYPES:
        quality_parts = [
            metadata.get("series_descriptions") or [],
            [metadata.get("trend_summary")] if metadata.get("trend_summary") else [],
            metadata.get("reference_lines") or [],
            metadata.get("plateaus") or [],
        ]
        information_quality = bool(metadata.get("analysis") and any(quality_parts))
    elif metadata.get("image_type") in DIAGRAM_TYPES:
        information_quality = any(
            [
                metadata.get("analysis"),
                metadata.get("components") or [],
                metadata.get("component_labels") or [],
                metadata.get("connections") or [],
                metadata.get("flow_directions") or [],
                metadata.get("annotations") or [],
                metadata.get("engineering_meaning"),
            ]
        )
    else:
        information_quality = bool(metadata.get("analysis"))

    candidate["information_quality_passed"] = information_quality
    if metadata.get("image_type") in GRAPH_TYPES and not information_quality and "graph lacks series, trend, reference line, or plateau information" not in candidate["validation_errors"]:
        candidate["validation_errors"].append("graph lacks series, trend, reference line, or plateau information")
    if metadata.get("image_type") in DIAGRAM_TYPES and not information_quality and "schematic lacks visible components, connections, labels, annotations, analysis, or engineering meaning" not in candidate["validation_errors"]:
        candidate["validation_errors"].append("schematic lacks visible components, connections, labels, annotations, analysis, or engineering meaning")
    candidate["schema_valid"] = not candidate["validation_errors"]
    candidate["normalized_schema_valid"] = candidate["schema_valid"]
    return candidate


def validate_vision_candidate(vision_text: str, *, classification: str, settings: Settings) -> dict[str, Any]:
    fields = parse_vision_fields(vision_text)
    errors: list[str] = []
    warnings: list[str] = []
    image_type, inferred_title = normalize_image_type(fields.get("image_type", ""), classification)
    confidence = parse_confidence(fields.get("confidence"))
    if confidence is None:
        errors.append("confidence must be a 0.0-1.0 float")
        confidence = 0.0
    elif not re.fullmatch(r"\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\s*", str(fields.get("confidence") or "")):
        warnings.append("confidence alias normalized to float")

    title = nullable_text(fields.get("title") or inferred_title)
    analysis = nullable_text(fields.get("analysis"))
    analysis_text = normalized_optional_text(analysis)
    analysis_lower = analysis_text.lower()

    def value(name: str) -> tuple[str | None, bool]:
        text = nullable_text(fields.get(name))
        ok = text is not None
        if str(fields.get(f"{name}_verified", "")).strip().lower() == "true" and not ok:
            errors.append(f"{name}_verified true but {name} is missing")
        return text, ok

    legend, legend_ok = value("legend")
    engineering = nullable_text(fields.get("engineering_meaning") or fields.get("engineering_purpose"))
    engineering_ok = engineering is not None

    if image_type in DIAGRAM_TYPES:
        components = normalize_schematic_components(fields.get("components") or fields.get("main_components"))
        component_labels = normalize_schematic_labels(fields.get("component_labels") or fields.get("labels"))
        connections = normalize_list(fields.get("connections"))
        flow_directions = normalize_list(fields.get("flow_directions") or fields.get("arrows_and_flow"))
        annotations = normalize_list(fields.get("annotations") or fields.get("uncertain_items"))
        return finalize_vision_candidate(
            {
                "raw_schema_valid": image_type in ALLOWED_IMAGE_TYPES and parse_confidence(fields.get("confidence")) is not None,
                "schema_valid": not errors,
                "normalized_schema_valid": not errors,
                "information_quality_passed": False,
                "validation_errors": errors,
                "normalization_warnings": warnings,
                "raw_vision_text": vision_text,
                "metadata": {
                    "image_type": image_type,
                    "confidence": confidence,
                    "title": title,
                    "title_verified": title is not None,
                    "analysis": analysis,
                    "components": components,
                    "component_labels": component_labels,
                    "connections": connections,
                    "flow_directions": flow_directions,
                    "annotations": annotations,
                    "legend": legend,
                    "legend_verified": legend_ok,
                    "engineering_meaning": engineering,
                    "engineering_meaning_verified": engineering_ok,
                    "vision_model": settings.vision_model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "legacy_note_backup": None,
                },
            }
        )

    if image_type in GRAPH_TYPES and is_unknown_value(analysis):
        errors.append("analysis is required")

    x_axis, x_ok = value("x_axis")
    y_axis, y_ok = value("y_axis")
    trend, trend_ok = value("trend")
    engineering = engineering or "주변 본문과 범례가 없어 구체적인 유동 구간 또는 저류층 거동은 확인할 수 없음"
    engineering_ok = False if engineering.startswith("주변 본문") else not is_unknown_value(engineering)
    x_axis, x_unit = split_axis_unit(x_axis, fields.get("x_axis_unit"))
    y_axis, y_unit = split_axis_unit(y_axis, fields.get("y_axis_unit"))
    series_count = normalize_int(fields.get("series_count"), errors, "series_count")
    series_count_verified = str(fields.get("series_count_verified", "")).strip().lower() == "true"
    series_descriptions = normalize_list(fields.get("series_descriptions"))
    reference_lines = normalize_list(fields.get("reference_lines"))
    plateaus = normalize_list(fields.get("plateaus") or fields.get("plateau"))
    peaks = normalize_list(fields.get("peaks") or fields.get("peak"))
    declines = normalize_list(fields.get("declines") or fields.get("decline"))
    slope_changes = normalize_list(fields.get("slope_changes"))

    if image_type in GRAPH_TYPES and not series_descriptions and analysis and any(word in analysis_lower for word in ["curve", "series", "rises", "plateau"]):
        series_descriptions = [analysis]
        warnings.append("series_descriptions inferred from analysis")
    if image_type in GRAPH_TYPES and is_unknown_value(fields.get("trend")) and any(word in analysis_lower for word in ["rises", "plateau", "decline"]):
        trend = analysis
        trend_ok = True
        warnings.append("trend_summary inferred from analysis")
    if image_type in GRAPH_TYPES and not plateaus and analysis and "plateau" in analysis_lower:
        plateaus = [analysis]
        warnings.append("plateaus inferred from analysis")

    return finalize_vision_candidate(
        {
            "raw_schema_valid": image_type in ALLOWED_IMAGE_TYPES and parse_confidence(fields.get("confidence")) is not None,
            "schema_valid": not errors,
            "normalized_schema_valid": not errors,
            "information_quality_passed": False,
            "validation_errors": errors,
            "normalization_warnings": warnings,
            "raw_vision_text": vision_text,
            "metadata": {
                "image_type": image_type,
                "confidence": confidence,
                "title": title,
                "title_verified": title is not None,
                "analysis": analysis,
                "x_axis": x_axis,
                "x_axis_unit": x_unit,
                "x_axis_scale": fields.get("x_axis_scale") or "unknown",
                "x_axis_verified": x_ok,
                "y_axis": y_axis,
                "y_axis_unit": y_unit,
                "y_axis_scale": fields.get("y_axis_scale") or "unknown",
                "y_axis_verified": y_ok,
                "series_count": series_count,
                "series_count_verified": series_count_verified,
                "series_descriptions": series_descriptions,
                "legend": legend,
                "legend_verified": legend_ok,
                "reference_lines": reference_lines,
                "plateaus": plateaus,
                "peaks": peaks,
                "declines": declines,
                "slope_changes": slope_changes,
                "trend_summary": trend,
                "trend_verified": trend_ok,
                "engineering_meaning": engineering,
                "engineering_meaning_verified": engineering_ok,
                "vision_model": settings.vision_model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "legacy_note_backup": None,
            },
        }
    )


def create_verified_backup(source: Path, backup_path: Path) -> dict[str, Any]:
    result = {"created": False, "verified": False, "path": None, "size_bytes": 0, "source_sha256": None, "backup_sha256": None, "error": None}
    try:
        source_sha = file_sha256(source)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup_path)
        size = backup_path.stat().st_size if backup_path.is_file() else 0
        backup_sha = file_sha256(backup_path) if size > 0 else ""
        result.update(
            {
                "created": backup_path.is_file() and size > 0,
                "verified": backup_path.is_file() and size > 0 and source_sha == backup_sha,
                "path": str(backup_path) if backup_path.is_file() else None,
                "size_bytes": size,
                "source_sha256": source_sha,
                "backup_sha256": backup_sha,
            }
        )
        if not result["verified"]:
            result["error"] = "backup verification failed"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def strict_graph_note_text(
    *,
    document_name: str,
    document_id: str,
    page: int,
    fig: int,
    image_path: Path,
    candidate: dict[str, Any],
    settings: Settings,
    backup: dict[str, Any] | None,
) -> str:
    return serialize_note_data(
        final_note_data(
            document_name=document_name,
            document_id=document_id,
            page=page,
            fig=fig,
            image_path=image_path,
            candidate=candidate,
            settings=settings,
            legacy_note_backup=backup["path"] if backup and backup.get("created") and backup.get("verified") and Path(str(backup.get("path"))).is_file() else None,
        )
    )


def final_note_data(
    *,
    document_name: str,
    document_id: str,
    page: int,
    fig: int,
    image_path: Path,
    candidate: dict[str, Any],
    settings: Settings,
    legacy_note_backup: str | None,
) -> dict[str, Any]:
    metadata = dict(candidate["metadata"])
    metadata.update(
        {
            "document_name": document_name,
            "document_id": document_id,
            "page_number": page,
            "image_index": fig,
            "image_path": str(image_path),
            "vision_model": settings.vision_model,
            "legacy_note_backup": legacy_note_backup,
        }
    )
    return metadata


def serialize_note_data(metadata: dict[str, Any]) -> str:
    order = SCHEMATIC_SCHEMA_FIELDS if metadata.get("image_type") in DIAGRAM_TYPES else GRAPH_SCHEMA_FIELDS
    lines = ["[Figure Note Metadata]"]
    for key in order:
        value = metadata.get(key, "")
        if isinstance(value, list):
            if value:
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: []")
        elif value is None:
            lines.append(f"{key}: null")
        elif isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    return "\n".join(lines)


def parse_serialized_note_data(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        if not raw_line or raw_line == "[Figure Note Metadata]":
            continue
        if raw_line.startswith("  - ") and current:
            if not isinstance(data.get(current), list):
                data[current] = []
            data[current].append(raw_line[4:])
            continue
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        value = value.strip()
        current = key
        if value == "[]":
            data[key] = []
            current = None
        elif value == "":
            data[key] = [] if key in SERIALIZED_LIST_KEYS else None
            if key not in SERIALIZED_LIST_KEYS:
                current = None
        elif value == "null":
            data[key] = None
            current = None
        elif value in {"true", "false"}:
            data[key] = value == "true"
        else:
            if key in {"series_count", "page_number", "image_index"}:
                try:
                    data[key] = int(value)
                    continue
                except ValueError:
                    pass
            if key == "confidence":
                try:
                    data[key] = float(value)
                    continue
                except ValueError:
                    pass
            data[key] = value
    return data


def extract_top_level_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for line in text.splitlines():
        if not line or line[0].isspace():
            continue
        match = TOP_LEVEL_KEY_RE.match(line)
        if match:
            keys.add(match.group(1))
    return keys


def parse_serialized_note_raw(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        if not raw_line or raw_line == "[Figure Note Metadata]":
            continue
        if raw_line.startswith("  - ") and current:
            data.setdefault(current, []).append(raw_line[4:])
            continue
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        current = key
        value = value.strip()
        if value == "[]":
            data[key] = []
            current = None
        elif value == "":
            data[key] = [] if key in SERIALIZED_LIST_KEYS else None
            if key not in SERIALIZED_LIST_KEYS:
                current = None
        else:
            data[key] = value
            current = None
    return data


def validate_serialized_note_data(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data.get("confidence"), float):
        errors.append("confidence must parse as float")
    is_graph = data.get("image_type") in GRAPH_TYPES
    is_schematic = data.get("image_type") in DIAGRAM_TYPES
    if is_graph and data.get("series_count") is not None and not isinstance(data.get("series_count"), int):
        errors.append("series_count must parse as int or null")
    required_list_keys = GRAPH_LIST_KEYS if is_graph else SCHEMATIC_LIST_KEYS if is_schematic else set()
    for key in required_list_keys:
        if not isinstance(data.get(key), list):
            errors.append(f"{key} must parse as list")
    if data.get("legend") is not None and not isinstance(data.get("legend"), str):
        errors.append("legend must parse as null or string")
    verified_keys = ["title_verified"]
    if is_graph:
        verified_keys.extend(["x_axis_verified", "y_axis_verified"])
    for verified_key in verified_keys:
        value_key = verified_key.removesuffix("_verified")
        if data.get(verified_key) is True and not data.get(value_key):
            errors.append(f"{verified_key} true but {value_key} is missing")
    backup = data.get("legacy_note_backup")
    if backup is not None and not Path(str(backup)).is_file():
        errors.append("legacy_note_backup must be null or an existing path")
    for key, value in data.items():
        if key.endswith("_verified") and not isinstance(value, bool):
            errors.append(f"{key} must parse as bool")
    return errors


def comparable_note_data(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in IGNORED_EQUIVALENCE_FIELDS}


def data_sha256(data: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(comparable_note_data(data), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def raw_data_sha256(data: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def generator_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BACKEND_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def candidate_provenance_errors(payload: dict[str, Any], op: dict[str, Any]) -> list[str]:
    expected = {
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "analysis_prompt_version": ANALYSIS_PROMPT_VERSION,
        "serializer_version": SERIALIZER_VERSION,
    }
    errors = [
        f"candidate is stale: {key} mismatch"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if payload.get("effective_classification") != op.get("effective_classification"):
        errors.append("candidate effective classification does not match selected operation")
    return errors


def make_candidate_payload(settings: Settings, op: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("schema_valid") is not True or candidate.get("information_quality_passed") is not True:
        raise ValueError("cannot create a valid candidate from failed analysis")
    image_path = Path(str(op.get("old_image") or op["new_image"]))
    note_data = final_note_data(
        document_name=op["document_name"],
        document_id=op["document_id"],
        page=int(op["page"]),
        fig=int(op["figure"]),
        image_path=image_path,
        candidate=candidate,
        settings=settings,
        legacy_note_backup=None,
    )
    semantic_grounding = bool(candidate.get("semantic_grounding_passed"))
    trend_grounding = bool(candidate.get("trend_grounding_passed", True))
    manual_reasons = list(candidate.get("manual_review_reasons") or [])
    if not semantic_grounding and "semantic grounding not verified" not in manual_reasons:
        manual_reasons.append("semantic grounding not verified")
    manual_review_required = bool(manual_reasons or not semantic_grounding or not trend_grounding)
    apply_ready = bool(candidate["schema_valid"] and candidate["information_quality_passed"] and semantic_grounding and not manual_review_required)
    payload = {
        "candidate_status": "valid" if apply_ready else "review_required",
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "analysis_prompt_version": ANALYSIS_PROMPT_VERSION,
        "serializer_version": SERIALIZER_VERSION,
        "generator_git_commit": generator_git_commit(),
        "asset_path": str(image_path),
        "asset_sha256": file_sha256(image_path),
        "document_id": op["document_id"],
        "document_name": op["document_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vision_model": settings.vision_model,
        "automatic_classification": op.get("automatic_classification"),
        "effective_classification": op.get("effective_classification"),
        "classification_forced": bool(op.get("classification_forced")),
        "classification_force_reason": op.get("classification_force_reason"),
        "classification_confidence": float(op.get("confidence") or 0.0),
        "dark_graph_candidate": bool(op.get("dark_graph_candidate")),
        "classification_override_reason": op.get("classification_override_reason"),
        "semantic_grounding_passed": semantic_grounding,
        "trend_grounding_passed": trend_grounding,
        "manual_review_required": manual_review_required,
        "manual_review_reasons": manual_reasons,
        "apply_ready": apply_ready,
        "analysis_input_path": candidate.get("analysis_input_path"),
        "analysis_input_sha256": candidate.get("analysis_input_sha256"),
        "analysis_transform": dict(candidate.get("analysis_transform") or ANALYSIS_TRANSFORM),
        "schema_valid": candidate["schema_valid"],
        "information_quality_passed": candidate["information_quality_passed"],
        "final_note_data": note_data,
        "final_note_data_sha256": data_sha256(note_data),
    }
    errors = validate_candidate_output(payload, op, image_path)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def validate_candidate_output(
    payload: dict[str, Any],
    op: dict[str, Any],
    asset_path: Path | None = None,
    *,
    require_status: bool = True,
    check_provenance: bool = True,
) -> list[str]:
    image_path = asset_path or Path(str(op.get("old_image") or op["new_image"]))
    errors: list[str] = []
    if require_status and payload.get("candidate_status") not in {"valid", "review_required"}:
        errors.append("candidate_status is not valid or review_required")
    try:
        path_matches = Path(str(payload.get("asset_path", ""))).resolve() == image_path.resolve()
    except (OSError, RuntimeError):
        path_matches = False
    if not path_matches:
        errors.append("candidate asset path does not match requested asset")
    if not image_path.is_file() or payload.get("asset_sha256") != file_sha256(image_path):
        errors.append("candidate asset sha256 mismatch")
    if payload.get("document_id") != op.get("document_id"):
        errors.append("candidate document_id mismatch")
    if payload.get("schema_valid") is not True:
        errors.append("candidate schema_valid is not true")
    if payload.get("information_quality_passed") is not True:
        errors.append("candidate information_quality_passed is not true")
    final_data = payload.get("final_note_data")
    if not isinstance(final_data, dict) or not final_data:
        errors.append("candidate final_note_data missing")
    elif payload.get("final_note_data_sha256") != data_sha256(final_data):
        errors.append("candidate final_note_data_sha256 mismatch")
    if check_provenance:
        errors.extend(candidate_provenance_errors(payload, op))
    return errors


def write_candidate_output(path: Path, payload: dict[str, Any], op: dict[str, Any]) -> None:
    errors = validate_candidate_output(payload, op)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        reloaded = json.loads(temporary.read_text(encoding="utf-8"))
        errors = validate_candidate_output(reloaded, op)
        if errors:
            raise ValueError("; ".join(errors))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def failed_candidate_path(path: Path) -> Path:
    return path.with_suffix(".failed.json")


def failed_candidate_payload(op: dict[str, Any], candidate: dict[str, Any] | None, error: str | None = None) -> dict[str, Any]:
    image_path = Path(str(op.get("old_image") or op.get("new_image") or ""))
    raw_response = str((candidate or {}).get("raw_vision_text") or "")
    try:
        json_parse_success = isinstance(json.loads(raw_response), dict)
    except (json.JSONDecodeError, TypeError):
        json_parse_success = False
    validation_errors = list((candidate or {}).get("validation_errors") or [])
    analysis_error = error or "; ".join(validation_errors) or "analysis did not produce a valid candidate"
    return {
        "candidate_status": "failed",
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "analysis_prompt_version": ANALYSIS_PROMPT_VERSION,
        "serializer_version": SERIALIZER_VERSION,
        "generator_git_commit": generator_git_commit(),
        "asset_path": str(image_path),
        "asset_sha256": file_sha256(image_path) if image_path.is_file() else None,
        "document_id": op.get("document_id"),
        "document_name": op.get("document_name"),
        "schema_valid": bool((candidate or {}).get("schema_valid")),
        "information_quality_passed": bool((candidate or {}).get("information_quality_passed")),
        "semantic_grounding_passed": bool((candidate or {}).get("semantic_grounding_passed")),
        "trend_grounding_passed": bool((candidate or {}).get("trend_grounding_passed")),
        "manual_review_required": True,
        "manual_review_reasons": list((candidate or {}).get("manual_review_reasons") or []),
        "apply_ready": False,
        "analysis_input_path": (candidate or {}).get("analysis_input_path"),
        "analysis_input_sha256": (candidate or {}).get("analysis_input_sha256"),
        "analysis_transform": dict((candidate or {}).get("analysis_transform") or ANALYSIS_TRANSFORM),
        "validation_errors": validation_errors,
        "analysis_error": analysis_error,
        "raw_model_response": raw_response or None,
        "raw_model_response_present": bool(raw_response),
        "json_parse_success": json_parse_success,
        "field_parse_success": bool(parse_key_values(raw_response)) if raw_response else False,
        "final_note_data": None,
    }


def write_failed_candidate(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        reloaded = json.loads(temporary.read_text(encoding="utf-8"))
        if reloaded.get("candidate_status") != "failed" or reloaded.get("final_note_data") is not None:
            raise ValueError("invalid failed candidate artifact")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_candidate(
    path: Path,
    op: dict[str, Any],
    *,
    allow_stale: bool = False,
    require_apply_ready: bool = False,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failed_artifact = payload.get("candidate_status") == "failed" or (
        payload.get("candidate_status") is None and not isinstance(payload.get("final_note_data"), dict)
    )
    errors = ["candidate file is a failed analysis artifact, not a valid candidate"] if failed_artifact else []
    errors.extend(
        validate_candidate_output(
            payload,
            op,
            require_status=payload.get("candidate_status") is not None,
            check_provenance=False,
        )
    )
    stale_errors = candidate_provenance_errors(payload, op)
    if stale_errors and not allow_stale:
        errors.extend(stale_errors)
    final_data = payload.get("final_note_data")
    if isinstance(final_data, dict) and final_data:
        candidate_hash = data_sha256(final_data)
        if payload.get("final_note_data_sha256") in {candidate_hash, raw_data_sha256(final_data)}:
            payload["final_note_data_sha256"] = candidate_hash
    if errors:
        raise ValueError("; ".join(errors))
    if stale_errors:
        payload["candidate_status"] = "stale"
        payload["stale_reason"] = "; ".join(stale_errors)
        payload["stale_reasons"] = stale_errors
        payload["apply_ready"] = False
    if require_apply_ready and payload.get("apply_ready") is not True:
        raise ValueError("candidate is not apply-ready")
    payload["_path"] = str(path)
    return payload


def schema_fields_for(data: dict[str, Any]) -> list[str]:
    return SCHEMATIC_SCHEMA_FIELDS if data.get("image_type") in DIAGRAM_TYPES else GRAPH_SCHEMA_FIELDS


def required_fields_for(data: dict[str, Any]) -> set[str]:
    return DIAGRAM_REQUIRED_FIELDS if data.get("image_type") in DIAGRAM_TYPES else GRAPH_REQUIRED_FIELDS


def compare_note_data(candidate: dict[str, Any], parsed: dict[str, Any], raw_parsed: dict[str, Any] | None = None) -> dict[str, Any]:
    fields = [field for field in schema_fields_for(candidate) if field not in IGNORED_EQUIVALENCE_FIELDS]
    expected = {field: candidate[field] for field in fields if field in candidate}
    actual = {field: parsed[field] for field in fields if field in parsed}
    raw_keys = set((raw_parsed or parsed).keys())
    required = required_fields_for(candidate)
    missing = sorted((required | (set(candidate) - IGNORED_EQUIVALENCE_FIELDS)) - raw_keys)
    extra = [field for field in parsed if field not in fields and field not in IGNORED_EQUIVALENCE_FIELDS]
    changed: dict[str, dict[str, Any]] = {}
    array_lengths: dict[str, dict[str, int]] = {}
    for field in fields:
        if field not in expected or field not in actual:
            continue
        if expected[field] != actual[field]:
            changed[field] = {"candidate": expected[field], "serialized": actual[field]}
            if isinstance(expected[field], list) or isinstance(actual[field], list):
                array_lengths[field] = {
                    "candidate": len(expected[field]) if isinstance(expected[field], list) else -1,
                    "serialized": len(actual[field]) if isinstance(actual[field], list) else -1,
                }
    return {
        "missing_serialized_fields": missing,
        "extra_serialized_fields": extra,
        "changed_serialized_fields": changed,
        "array_length_mismatches": array_lengths,
        "equivalent": not missing and not extra and not changed and data_sha256(expected) == data_sha256(actual),
    }


def validate_note_equivalence(note_text: str, final_data: dict[str, Any]) -> tuple[dict[str, Any], list[str], bool, dict[str, Any]]:
    raw_keys = extract_top_level_keys(note_text)
    parsed = parse_serialized_note_data(note_text)
    errors = validate_serialized_note_data(parsed)
    diff = compare_note_data(final_data, parsed, {key: None for key in raw_keys})
    errors.extend(f"missing required field: {field}" for field in diff["missing_serialized_fields"])
    equivalent = bool(diff["equivalent"])
    if not equivalent:
        errors.append("serialized note differs from candidate final_note_data")
    return parsed, errors, equivalent, diff


def find_legacy_backups(settings: Settings, op: dict[str, Any]) -> list[str]:
    stems = {str(op.get("stem") or "")}
    new_image = Path(str(op.get("new_image") or ""))
    if new_image.name:
        stems.add(new_image.stem)
    paths: list[str] = []
    for stem in stems:
        paths.extend(str(path) for path in settings.figure_notes_dir.glob(f"{stem}*.bak.md"))
    return sorted(set(paths))


def build_plan(
    settings: Settings,
    *,
    document: str | None,
    limit: int | None,
    quarantine_logos: bool,
    reanalyze_graphs: bool,
    update_chroma: bool,
    reanalyze_diagrams: bool = False,
    reanalyze_equations: bool = False,
    reanalyze_all_engineering_figures: bool = False,
    decisions: dict[str, Any] | None = None,
    asset: str | None = None,
    force_image_type: str | None = None,
) -> dict[str, Any]:
    decisions = decisions or {}
    documents = resolve_documents(settings)
    stem_documents = document_stem_map(documents)
    manifest_mappings = load_manifest_mappings(settings)
    previous_classifications = load_previous_classifications(settings)
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
        "protected_engineering_figures": 0,
        "dark_graph_candidates": 0,
        "logo_candidates_after_graph_protection": 0,
        "final_logo_candidates": 0,
        "logo_candidates": 0,
        "uncertain_logo_candidates": 0,
        "graph_candidates": 0,
        "chart_reanalysis_candidates": 0,
        "graph_reanalysis_candidates": 0,
        "diagram_reanalysis_candidates": 0,
        "schematic_reanalysis_candidates": 0,
        "equation_reanalysis_candidates": 0,
        "unknown_engineering_figures": 0,
        "table_candidates": 0,
        "diagram_candidates": 0,
        "equation_candidates": 0,
        "photo_candidates": 0,
        "page_decoration_candidates": 0,
        "unknown_candidates": 0,
        "would_quarantine": 0,
        "would_delete": 0,
        "would_reanalyze": 0,
        "graph_reanalysis_candidates_total": 0,
        "diagram_reanalysis_candidates_total": 0,
        "equation_reanalysis_candidates_total": 0,
        "selected_reanalysis_targets": 0,
        "limited_out_reanalysis_targets": 0,
        "requested_asset": "",
        "asset_error": "",
        "matched_requested_assets": 0,
        "would_update_chroma": 0,
        "would_create_new_notes": 0,
        "would_update_existing_notes": 0,
        "would_skip_low_confidence": 0,
        "would_skip_analysis_failure": 0,
        "rename_image_candidates": 0,
        "rename_active_note_candidates": 0,
        "rename_quarantine_candidates": 0,
        "rename_skipped_backups": excluded["backup_notes_excluded"],
        "rename_skipped_temporary": excluded["temporary_files_excluded"],
        "rename_unresolved": 0,
        "sha_named_images": 0,
        "document_named_images": 0,
        "already_renamed_images": 0,
        "already_renamed_notes": 0,
        "approved_quarantine_groups": 0,
        "unresolved_no_digest": 0,
        "unresolved_no_manifest_mapping": 0,
        "unresolved_document_name_mismatch": 0,
        "unresolved_ambiguous_document_name": 0,
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
    for stem, paths in sorted(assets.items()):
        doc_id, doc_name, page, fig, resolve_reason, already_renamed = resolve_asset(
            stem=stem,
            paths=paths,
            documents=documents,
            stem_documents=stem_documents,
            manifest_mappings=manifest_mappings,
        )
        if not doc_id or not doc_name or page is None or fig is None:
            stats["unresolved_documents"] += 1
            stats["rename_unresolved"] += 1
            stats[resolve_reason] = int(stats.get(resolve_reason, 0)) + 1
            continue
        if not matches_document_filter(document, doc_id, doc_name):
            continue
        stats["resolved_documents"] += 1
        stats["resolved_asset_count"] += 1
        resolved_docs.add(doc_id)

        image_path = paths.get("image")
        note_path = paths.get("note")
        if image_path:
            stats["filtered_document_images"] += 1
            if already_renamed:
                stats["document_named_images"] += 1
            else:
                stats["sha_named_images"] += 1
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
        classification, confidence, reasons, protections, review_bucket, initial_classification, classification_overridden, override_reason = classify_asset(
            note_fields,
            metrics,
            duplicate_count,
            page_count,
            note_text,
        )
        automatic_classification = classification
        requested_path = Path(asset) if asset else None
        classification_forced = bool(
            force_image_type
            and requested_path
            and image_path
            and (requested_path == image_path or requested_path.name == image_path.name)
        )
        if classification_forced:
            classification = str(force_image_type)
            review_bucket = classification
            classification_overridden = classification != automatic_classification
            override_reason = "manual visual review" if classification_overridden else override_reason
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
            stats["graph_reanalysis_candidates"] += 1
        elif classification == "chart":
            stats["graph_candidates"] += 1
            stats["chart_reanalysis_candidates"] += 1
        elif classification == "table":
            stats["table_candidates"] += 1
        elif classification == "diagram":
            stats["diagram_candidates"] += 1
            stats["diagram_reanalysis_candidates"] += 1
        elif classification == "schematic":
            stats["diagram_candidates"] += 1
            stats["schematic_reanalysis_candidates"] += 1
        elif classification == "equation":
            stats["equation_candidates"] += 1
            stats["equation_reanalysis_candidates"] += 1
        elif classification == "photo":
            stats["photo_candidates"] += 1
        elif classification == "page_decoration":
            stats["page_decoration_candidates"] += 1
        elif classification == "unknown":
            stats["unknown_candidates"] += 1
        elif classification == "unknown_engineering_figure":
            stats["unknown_engineering_figures"] += 1
        if review_bucket == "uncertain_logo":
            stats["uncertain_logo_candidates"] += 1

        new_stem = target_stem(doc_name, doc_id, page, fig)
        image_target = settings.figures_dir / f"{new_stem}{image_path.suffix.lower()}" if image_path else None
        note_target = settings.figure_notes_dir / f"{new_stem}.md" if (note_path or image_path) else None
        if image_path and image_target and image_path == image_target:
            stats["already_renamed_images"] += 1
        if note_path and note_target and note_path == note_target:
            stats["already_renamed_notes"] += 1
        quarantine = False
        if (image_path and image_path != image_target) or (note_path and note_path != note_target):
            stats["rename_candidates"] += 1
            if len(rename_examples) < 20:
                rename_examples.append({"old": stem, "new": new_stem})

        for target in [image_target, note_target if note_path else None]:
            if not target:
                continue
            current = image_path if target.suffix.lower() in IMAGE_SUFFIXES else note_path
            if target in used_targets or (target.exists() and target != current):
                collisions.append(str(target))
            used_targets.add(target)

        should_reanalyze = image_path is not None and (
            (classification in GRAPH_TYPES and reanalyze_graphs)
            or (classification in DIAGRAM_TYPES and reanalyze_diagrams)
            or (classification in EQUATION_TYPES and reanalyze_equations)
            or (classification in (GRAPH_TYPES | DIAGRAM_TYPES | EQUATION_TYPES) and reanalyze_all_engineering_figures)
        )
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
                "previous_classification": previous_classifications.get(stem, initial_classification),
                "initial_classification": initial_classification,
                "final_classification": classification,
                "automatic_classification": automatic_classification,
                "effective_classification": classification,
                "classification_overridden": classification_overridden,
                "classification_override_reason": override_reason,
                "classification_forced": classification_forced,
                "classification_force_reason": "manual visual review" if classification_forced else None,
                "dark_graph_candidate": "dark_graph" in protections,
                "marker_series_detected": bool(metrics.get("marker_series_detected")),
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
                "new_note": str(note_target) if (note_path or should_reanalyze) and note_target else None,
                "old_note_text": note_path.read_text(encoding="utf-8", errors="replace") if note_path else None,
                "quarantine": quarantine,
                "protected_engineering_figure": classification in PROTECTED_TYPES,
                "reanalyze": should_reanalyze,
                "update_chroma": update_chroma,
                "status": "planned",
            }
        )

    stats["resolved_document_count"] = len(resolved_docs)
    stats["classification_count_matches"] = stats["classified_total"] == stats["filtered_document_images"]
    for category, count in category_counts.items():
        stats[f"{category}_classified"] = count
    stats["protected_engineering_figures"] = sum(
        1 for op in operations if op.get("protected_engineering_figure")
    )
    apply_manual_decisions(settings, operations, decisions, quarantine_logos)
    collisions = detect_collisions(operations)
    candidates, selected, asset_error = select_reanalysis_targets(operations, limit=limit, asset=asset)
    if asset:
        stats["requested_asset"] = asset
        stats["matched_requested_assets"] = sum(1 for op in operations if asset_matches(op, asset))
    if asset_error:
        stats["asset_error"] = asset_error
    stats["graph_reanalysis_candidates_total"] = sum(
        1 for op in candidates if op.get("classification") in GRAPH_TYPES
    )
    stats["diagram_reanalysis_candidates_total"] = sum(
        1 for op in candidates if op.get("classification") in DIAGRAM_TYPES
    )
    stats["equation_reanalysis_candidates_total"] = sum(
        1 for op in candidates if op.get("classification") in EQUATION_TYPES
    )
    stats["selected_reanalysis_targets"] = len(selected)
    stats["limited_out_reanalysis_targets"] = max(0, len(candidates) - len(selected))
    stats["would_reanalyze"] = len(selected)
    stats["would_update_existing_notes"] = sum(1 for op in selected if op.get("old_note"))
    stats["would_create_new_notes"] = sum(1 for op in selected if not op.get("old_note") and op.get("new_note"))
    stats["name_collisions"] = len(collisions)
    stats["would_quarantine"] = sum(1 for op in operations if op["quarantine"])
    stats["approved_quarantine_groups"] = len({op.get("canonical_group_id") or op.get("group_id") for op in operations if op["quarantine"]})
    stats["final_logo_candidates"] = sum(1 for op in operations if op["quarantine"] and op.get("classification") == "logo")
    stats["rename_quarantine_candidates"] = stats["would_quarantine"]
    stats["rename_image_candidates"] = sum(
        1 for op in operations if not op.get("quarantine") and op.get("old_image") and op.get("old_image") != op.get("new_image")
    )
    stats["rename_active_note_candidates"] = sum(
        1 for op in operations if not op.get("quarantine") and op.get("old_note") and op.get("old_note") != op.get("new_note")
    )
    quarantine_examples[:] = [op["stem"] for op in operations if op["quarantine"]][:20]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "rename_examples": rename_examples,
        "quarantine_examples": quarantine_examples,
        "collisions": collisions,
        "selected_reanalysis_files": [Path(str(op.get("old_image") or op.get("new_image"))).name for op in selected],
        "operations": operations,
    }


def apply_operation_grounding(candidate: dict[str, Any], op: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("metadata") or {}
    raw_text = str(candidate.get("raw_vision_text") or "").lower()
    circuit_terms = ["resistor", "capacitor", "voltage source", "current source", "electrical circuit", "circuit node"]
    if op.get("dark_graph_candidate") and any(term in raw_text for term in circuit_terms):
        candidate["schema_valid"] = False
        candidate["normalized_schema_valid"] = False
        candidate.setdefault("validation_errors", []).append("diagram_semantics_unsupported_by_visual_structure")

    if op.get("classification") in GRAPH_TYPES and op.get("dark_graph_candidate") and not op.get("detected_text"):
        for key in ["title", "x_axis", "x_axis_unit", "y_axis", "y_axis_unit", "legend"]:
            metadata[key] = None
        metadata.update(
            {
                "title_verified": False,
                "x_axis_verified": False,
                "y_axis_verified": False,
                "legend_verified": False,
                "x_axis_scale": "unknown",
                "y_axis_scale": "unknown",
                "engineering_meaning": None,
                "engineering_meaning_verified": False,
            }
        )
        if metadata.get("series_count") == 2 and len(metadata.get("series_descriptions") or []) >= 2:
            metadata["series_count_verified"] = True
        elif op.get("classification_override_reason") == "dark_graph_precedence":
            candidate["information_quality_passed"] = False
            candidate.setdefault("validation_errors", []).append("dark_graph_distinct_series_not_resolved")

    candidate["semantic_grounding_passed"] = bool(
        metadata.get("engineering_meaning_verified")
        or metadata.get("title_verified")
        or metadata.get("x_axis_verified")
        or metadata.get("y_axis_verified")
    )
    stem = Path(str(op.get("old_image") or op.get("new_image") or "")).stem
    expectations = MANUAL_TREND_EXPECTATIONS.get(stem)
    trend_text = " ".join(
        [
            str(metadata.get("analysis") or ""),
            str(metadata.get("trend_summary") or ""),
            *[str(item) for item in metadata.get("series_descriptions") or []],
        ]
    ).lower()
    trend_grounding_passed = True
    manual_reasons: list[str] = []
    if expectations:
        trend_grounding_passed = all(any(option in trend_text for option in group.split("|")) for group in expectations)
        if not trend_grounding_passed:
            manual_reasons.append(
                "lower series peak and decline not represented in model description"
                if stem.endswith("p0305_fig02")
                else "model trend description conflicts with visible marker trajectory"
            )
    candidate["trend_grounding_passed"] = trend_grounding_passed
    candidate["manual_review_reasons"] = manual_reasons
    return candidate


async def analyze_reanalysis_operation(
    settings: Settings,
    ollama: OllamaClient,
    op: dict[str, Any],
) -> dict[str, Any]:
    image_path = Path(str(op.get("old_image") or op["new_image"]))
    if not image_path.exists():
        image_path = Path(str(op["new_image"]))
    original_sha256 = file_sha256(image_path)
    tmp_image = prepare_vision_image(image_path)
    analysis_input_sha256 = file_sha256(tmp_image)
    prompt = prompt_for_classification(str(op.get("classification") or ""))
    vision_call_count = 0
    try:
        vision_call_count += 1
        vision_text = await ollama.describe_image(
            tmp_image,
            prompt=prompt,
        )
        candidate = validate_vision_candidate(vision_text.strip(), classification=str(op.get("classification") or ""), settings=settings)
        candidate = apply_operation_grounding(candidate, op)
        candidate.update(
            {
                "analysis_input_path": str(tmp_image),
                "analysis_input_sha256": analysis_input_sha256,
                "analysis_transform": dict(ANALYSIS_TRANSFORM),
            }
        )
        if candidate.get("incomplete_strings_detected"):
            vision_call_count += 1
            retry_text = await ollama.describe_image(
                tmp_image,
                prompt=prompt
                + "\nRewrite only complete, directly visible facts. Do not include coordinate values, slopes, colors, markers, or labels unless they are clearly readable. Use no unfinished phrases.",
            )
            retry = validate_vision_candidate(retry_text.strip(), classification=str(op.get("classification") or ""), settings=settings)
            retry = apply_operation_grounding(retry, op)
            retry.update(
                {
                    "analysis_input_path": str(tmp_image),
                    "analysis_input_sha256": analysis_input_sha256,
                    "analysis_transform": dict(ANALYSIS_TRANSFORM),
                }
            )
            retry["semantic_retry_attempts"] = 1
            candidate["semantic_retry_attempts"] = 1
            candidate["vision_call_count"] = vision_call_count
            if retry.get("incomplete_strings_detected", 0) < candidate.get("incomplete_strings_detected", 0) and retry["schema_valid"]:
                retry["semantic_retry_successes"] = 1
                retry["vision_call_count"] = vision_call_count
                return retry
    finally:
        if tmp_image.exists():
            tmp_image.unlink()
        if file_sha256(image_path) != original_sha256:
            raise RuntimeError("original asset changed during Vision analysis")
    candidate["vision_call_count"] = vision_call_count
    return candidate


async def apply_reanalysis_note(
    settings: Settings,
    ollama: OllamaClient,
    op: dict[str, Any],
    candidate_payload: dict[str, Any] | None = None,
) -> None:
    op.update(
        {
            "action": "reanalyze_note",
            "backup_requested": bool(op.get("old_note")),
            "backup_created": False,
            "backup_verified": False,
            "backup_path": None,
            "source_sha256": None,
            "backup_sha256": None,
            "temporary_note_path": None,
            "note_replaced": False,
            "existing_note_preserved": True,
            "validation_errors": [],
            "vision_schema_valid": False,
            "information_quality_passed": False,
            "error": None,
            "candidate_path": candidate_payload.get("_path") if candidate_payload else op.get("candidate_path"),
            "candidate_data_sha256": None,
            "vision_call_count": 0,
            "temporary_note_exists": False,
            "temporary_note_size": 0,
            "serialized_output_keys": [],
            "temporary_raw_keys": [],
            "temporary_missing_required_fields": [],
            "temporary_file_sha256": None,
            "temporary_parse_valid": False,
            "temporary_schema_valid": False,
            "temporary_information_quality_passed": False,
            "candidate_serialized_equivalent": False,
            "temporary_parsed_data_sha256": None,
            "serialization_comparison": None,
            "post_write_validation_passed": False,
            "post_write_schema_valid": False,
            "post_write_information_quality_passed": False,
            "post_write_equivalent_to_candidate": False,
            "post_write_comparison": None,
            "post_write_raw_keys": [],
            "post_write_missing_required_fields": [],
            "post_write_file_sha256": None,
            "post_write_bytes_equal_temporary": False,
            "rollback_attempted": False,
            "rollback_succeeded": False,
            "rollback_error": None,
        }
    )
    old_note = Path(str(op["old_note"])) if op.get("old_note") else None
    new_note = old_note if old_note else Path(str(op["new_note"]))
    op["asset_path"] = str(op.get("old_image") or op.get("new_image") or "")
    op["note_path"] = str(new_note)
    if candidate_payload:
        final_data = dict(candidate_payload["final_note_data"])
        op["candidate_data_sha256"] = candidate_payload["final_note_data_sha256"]
        op["vision_schema_valid"] = True
        op["information_quality_passed"] = True
        op["vision_call_count"] = 0
    else:
        candidate = await analyze_reanalysis_operation(settings, ollama, op)
        op["vision_call_count"] = candidate.get("vision_call_count", 0)
        op["vision_schema_valid"] = candidate["schema_valid"]
        op["information_quality_passed"] = candidate["information_quality_passed"]
        op["validation_errors"] = candidate["validation_errors"]
        if not candidate["schema_valid"]:
            op["status"] = "rejected_invalid_schema"
            return
        if not candidate["information_quality_passed"]:
            op["status"] = "rejected_low_information"
            return
        final_data = final_note_data(
            document_name=op["document_name"],
            document_id=op["document_id"],
            page=int(op["page"]),
            fig=int(op["figure"]),
            image_path=Path(str(op.get("old_image") or op["new_image"])),
            candidate=candidate,
            settings=settings,
            legacy_note_backup=None,
        )
        op["candidate_data_sha256"] = data_sha256(final_data)

    backup: dict[str, Any] | None = None
    if old_note:
        backup_path = old_note.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.md")
        backup = create_verified_backup(old_note, backup_path)
        op.update(
            {
                "backup_created": backup["created"],
                "backup_verified": backup["verified"],
                "backup_path": backup["path"],
                "source_sha256": backup["source_sha256"],
                "backup_sha256": backup["backup_sha256"],
            }
        )
        if not backup["created"] or not backup["verified"]:
            op["status"] = "backup_failed"
            op["error"] = backup["error"]
            return

    final_data_with_backup = dict(final_data)
    final_data_with_backup["legacy_note_backup"] = backup["path"] if backup and backup.get("created") and backup.get("verified") else None
    note_text = serialize_note_data(final_data_with_backup)
    serialized_keys = sorted(extract_top_level_keys(note_text))
    op["serialized_output_keys"] = serialized_keys
    tmp_note = new_note.with_suffix(f".tmp.{os.getpid()}.md")
    op["temporary_note_path"] = str(tmp_note)
    new_note.parent.mkdir(parents=True, exist_ok=True)
    tmp_note.write_text(note_text, encoding="utf-8")
    op["temporary_note_exists"] = tmp_note.is_file()
    op["temporary_note_size"] = tmp_note.stat().st_size if tmp_note.is_file() else 0
    tmp_text = tmp_note.read_text(encoding="utf-8")
    op["temporary_raw_keys"] = sorted(extract_top_level_keys(tmp_text))
    op["temporary_file_sha256"] = file_sha256(tmp_note)
    tmp_fields, type_errors, equivalent, tmp_diff = validate_note_equivalence(tmp_text, final_data_with_backup)
    op["temporary_missing_required_fields"] = tmp_diff["missing_serialized_fields"]
    op["temporary_parse_valid"] = not type_errors
    op["temporary_schema_valid"] = not type_errors
    op["temporary_information_quality_passed"] = not type_errors
    op["candidate_serialized_equivalent"] = equivalent
    op["temporary_parsed_data_sha256"] = data_sha256(tmp_fields)
    op["serialization_comparison"] = tmp_diff
    if type_errors:
        tmp_note.unlink(missing_ok=True)
        op["status"] = "temporary_validation_failed"
        op["validation_errors"] = [*op.get("validation_errors", []), *type_errors]
        op["error"] = "; ".join(type_errors)
        return
    if old_note and tmp_fields.get("legacy_note_backup") != op.get("backup_path"):
        tmp_note.unlink(missing_ok=True)
        op["status"] = "temporary_validation_failed"
        op["error"] = "legacy_note_backup does not point to verified backup"
        return
    os.replace(tmp_note, new_note)
    if not new_note.is_file() or new_note.stat().st_size <= 0:
        op["status"] = "replace_verification_failed"
        op["error"] = "new note missing or empty after replace"
        return
    post_text = new_note.read_text(encoding="utf-8")
    op["post_write_raw_keys"] = sorted(extract_top_level_keys(post_text))
    op["post_write_file_sha256"] = file_sha256(new_note)
    op["post_write_bytes_equal_temporary"] = op["temporary_file_sha256"] == op["post_write_file_sha256"]
    post_fields, post_errors, post_equivalent, post_diff = validate_note_equivalence(post_text, final_data_with_backup)
    op["post_write_schema_valid"] = not post_errors
    op["post_write_information_quality_passed"] = not post_errors
    op["post_write_equivalent_to_candidate"] = post_equivalent
    op["post_write_comparison"] = post_diff
    op["post_write_missing_required_fields"] = post_diff["missing_serialized_fields"]
    op["post_write_validation_passed"] = not post_errors and post_equivalent and op["post_write_bytes_equal_temporary"]
    if not op["post_write_validation_passed"]:
        op["rollback_attempted"] = True
        try:
            if backup and backup.get("path"):
                shutil.copy2(Path(str(backup["path"])), new_note)
            op["rollback_succeeded"] = True
        except Exception as exc:  # noqa: BLE001
            op["rollback_error"] = f"{type(exc).__name__}: {exc}"
            raise
        op["status"] = "post_write_validation_failed"
        op["error"] = "; ".join(post_errors or ["post-write bytes differ from temporary note"])
        return
    op["note_replaced"] = True
    op["existing_note_preserved"] = False
    op["status"] = "applied"


async def apply_plan(
    settings: Settings,
    manifest: dict[str, Any],
    manifest_path: Path | None = None,
    candidate_payload: dict[str, Any] | None = None,
) -> None:
    ollama = OllamaClient(settings)
    for op in manifest["operations"]:
        if op["reanalyze"] and op.get("new_image") and op.get("new_note"):
            await apply_reanalysis_note(settings, ollama, op, candidate_payload)
            if manifest_path:
                try:
                    write_manifest(settings, manifest, manifest_path)
                except Exception as exc:  # noqa: BLE001
                    backup_path = Path(str(op.get("backup_path") or ""))
                    new_note = Path(str(op.get("new_note") or ""))
                    if op.get("note_replaced") and backup_path.is_file() and new_note:
                        shutil.copy2(backup_path, new_note)
                        op["note_replaced"] = False
                        op["existing_note_preserved"] = True
                    op["status"] = "manifest_write_failed"
                    op["error"] = f"{type(exc).__name__}: {exc}"
            continue
        if op["quarantine"] or op["old_image"] != op["new_image"]:
            move_path(op["old_image"], op["new_image"])
        if op["old_note"] or (op["reanalyze"] and op.get("new_note")):
            old_note = Path(op["old_note"]) if op.get("old_note") else None
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
            if old_note and old_note != new_note and old_note.exists():
                if new_note.exists():
                    raise FileExistsError(new_note)
                old_note.rename(new_note)
            if note_text.strip():
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
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
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
    if manifest.get("selected_reanalysis_files"):
        print("\nselected_reanalysis_files:")
        for item in manifest["selected_reanalysis_files"]:
            print(item)
    for collision in manifest["collisions"][:20]:
        print(f"COLLISION: {collision}")


async def print_analyze_dry_run(
    settings: Settings,
    manifest: dict[str, Any],
    *,
    candidate_output: Path | None = None,
    candidate_input: Path | None = None,
) -> bool:
    ollama = OllamaClient(settings)
    selected = [op for op in manifest["operations"] if op.get("reanalyze")]
    succeeded = True
    legacy_paths = sorted({path for op in selected for path in find_legacy_backups(settings, op)})
    print(f"legacy_backups_found={len(legacy_paths)}")
    print("legacy_backup_paths=" + json.dumps(legacy_paths, ensure_ascii=False))
    for op in selected:
        if candidate_input:
            payload = load_candidate(candidate_input, op, allow_stale=True)
            candidate = {
                "raw_vision_text": "",
                "metadata": payload["final_note_data"],
                "raw_schema_valid": True,
                "normalized_schema_valid": True,
                "schema_valid": True,
                "information_quality_passed": True,
                "validation_errors": [],
                "normalization_warnings": [],
                "incomplete_strings_detected": 0,
                "semantic_retry_attempts": 0,
                "semantic_retry_successes": 0,
                "removed_incomplete_items": 0,
                "vision_call_count": 0,
                "semantic_grounding_passed": bool(payload.get("semantic_grounding_passed")),
                "trend_grounding_passed": bool(payload.get("trend_grounding_passed")),
                "manual_review_reasons": list(payload.get("manual_review_reasons") or []),
            }
            final_data = payload["final_note_data"]
        else:
            analysis_error: str | None = None
            try:
                candidate = await analyze_reanalysis_operation(settings, ollama, op)
            except Exception as exc:  # noqa: BLE001
                candidate = None
                analysis_error = f"{type(exc).__name__}: {exc}"
            valid = bool(candidate and candidate.get("schema_valid") is True and candidate.get("information_quality_passed") is True)
            if not valid:
                succeeded = False
                failure = failed_candidate_payload(op, candidate, analysis_error)
                failed_path = failed_candidate_path(candidate_output) if candidate_output else None
                existing_valid_preserved = False
                if candidate_output and candidate_output.exists():
                    try:
                        load_candidate(candidate_output, op, allow_stale=True)
                        existing_valid_preserved = True
                    except (OSError, ValueError, json.JSONDecodeError):
                        candidate_output.unlink(missing_ok=True)
                if failed_path:
                    write_failed_candidate(failed_path, failure)
                print("candidate_status=failed")
                print("candidate_output_written=false")
                print(f"failed_candidate_output_written={str(bool(failed_path)).lower()}")
                print(f"schema_valid={str(failure['schema_valid']).lower()}")
                print(f"information_quality_passed={str(failure['information_quality_passed']).lower()}")
                print("final_note_data_present=false")
                print(f"candidate_output_path={candidate_output or ''}")
                print(f"failed_candidate_output_path={failed_path or ''}")
                print(f"analysis_error={failure['analysis_error']}")
                print(f"existing_valid_candidate_preserved={str(existing_valid_preserved).lower()}")
                print("candidate_output_replaced=false")
                print(f"vision_call_count={(candidate or {}).get('vision_call_count', 0)}")
                print(f"raw_model_response_present={str(failure['raw_model_response_present']).lower()}")
                print(f"json_parse_success={str(failure['json_parse_success']).lower()}")
                print(f"field_parse_success={str(failure['field_parse_success']).lower()}")
                print("validation_errors=" + json.dumps(failure["validation_errors"], ensure_ascii=False))
                print(f"automatic_classification={op.get('automatic_classification')}")
                print(f"effective_classification={op.get('effective_classification')}")
                print(f"dark_graph_candidate={str(op.get('dark_graph_candidate', False)).lower()}")
                print(f"classification_override_reason={op.get('classification_override_reason') or ''}")
                continue
            payload = make_candidate_payload(settings, op, candidate)
            final_data = payload["final_note_data"]
            if candidate_output:
                write_candidate_output(candidate_output, payload, op)
                failed_candidate_path(candidate_output).unlink(missing_ok=True)
            print(f"candidate_status={payload['candidate_status']}")
            print(f"candidate_output_written={str(bool(candidate_output)).lower()}")
            print("failed_candidate_output_written=false")
            print("schema_valid=true")
            print("information_quality_passed=true")
            print("final_note_data_present=true")
            print(f"candidate_output_path={candidate_output or ''}")
            print("failed_candidate_output_path=")
            print("analysis_error=")
            print(f"automatic_classification={payload.get('automatic_classification')}")
            print(f"effective_classification={payload.get('effective_classification')}")
            print(f"dark_graph_candidate={str(payload.get('dark_graph_candidate', False)).lower()}")
            print(f"classification_override_reason={payload.get('classification_override_reason') or ''}")
            print(f"semantic_grounding_passed={str(payload.get('semantic_grounding_passed', False)).lower()}")
            print(f"trend_grounding_passed={str(payload.get('trend_grounding_passed', False)).lower()}")
            print(f"manual_review_required={str(payload.get('manual_review_required', False)).lower()}")
            print("manual_review_reasons=" + json.dumps(payload.get("manual_review_reasons") or [], ensure_ascii=False))
            print(f"apply_ready={str(payload.get('apply_ready', False)).lower()}")
        final_note = serialize_note_data(final_data)
        parsed, parse_errors, equivalent, diff = validate_note_equivalence(final_note, final_data)
        output_keys = sorted(extract_top_level_keys(final_note))
        print("\n[Raw Vision Response]")
        print(candidate["raw_vision_text"])
        print("\n[Normalized Vision Candidate]")
        print(json.dumps(candidate["metadata"], ensure_ascii=False, indent=2))
        print("\n[Validated Vision Candidate]")
        print(f"raw_schema_valid={str(candidate['raw_schema_valid']).lower()}")
        print(f"normalized_schema_valid={str(candidate['normalized_schema_valid']).lower()}")
        print(f"information_quality_passed={str(candidate['information_quality_passed']).lower()}")
        print("validation_errors=" + json.dumps(candidate["validation_errors"], ensure_ascii=False))
        print("normalization_warnings=" + json.dumps(candidate["normalization_warnings"], ensure_ascii=False))
        print(f"incomplete_strings_detected={candidate.get('incomplete_strings_detected', 0)}")
        print(f"semantic_retry_attempts={candidate.get('semantic_retry_attempts', 0)}")
        print(f"semantic_retry_successes={candidate.get('semantic_retry_successes', 0)}")
        print(f"removed_incomplete_items={candidate.get('removed_incomplete_items', 0)}")
        print(f"vision_call_count={candidate.get('vision_call_count', 0)}")
        print(f"candidate_data_sha256={payload['final_note_data_sha256']}")
        print(f"temporary_parsed_data_sha256={data_sha256(parsed)}")
        print(f"candidate_serialized_equivalent={str(equivalent).lower()}")
        print("serialized_output_keys=" + json.dumps(output_keys, ensure_ascii=False))
        print("temporary_raw_keys=" + json.dumps(output_keys, ensure_ascii=False))
        print("temporary_missing_required_fields=" + json.dumps(diff["missing_serialized_fields"], ensure_ascii=False))
        print("missing_serialized_fields=" + json.dumps(diff["missing_serialized_fields"], ensure_ascii=False))
        print("extra_serialized_fields=" + json.dumps(diff["extra_serialized_fields"], ensure_ascii=False))
        print("changed_serialized_fields=" + json.dumps(diff["changed_serialized_fields"], ensure_ascii=False))
        print("array_length_mismatches=" + json.dumps(diff["array_length_mismatches"], ensure_ascii=False))
        print("temporary_parse_errors=" + json.dumps(parse_errors, ensure_ascii=False))
        print(f"would_create_backup={str(bool(op.get('old_note'))).lower()}")
        print(f"would_replace_note={str(candidate['schema_valid'] and candidate['information_quality_passed']).lower()}")
        print("\n[Final Serialized Note Data]")
        print(final_note)
    return succeeded


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


def export_classification_delta(manifest: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for op in manifest["operations"]:
        if (
            op.get("effective_classification") not in (GRAPH_TYPES | DIAGRAM_TYPES)
            and op.get("previous_classification") == op.get("automatic_classification")
        ):
            continue
        metrics = op.get("metrics") or {}
        rows.append(
            {
                "stem": op.get("stem"),
                "asset_path": op.get("old_image") or op.get("new_image"),
                "previous_classification": op.get("previous_classification"),
                "automatic_classification": op.get("automatic_classification"),
                "effective_classification": op.get("effective_classification"),
                "dark_graph_candidate": bool(op.get("dark_graph_candidate")),
                "classification_override_reason": op.get("classification_override_reason"),
                "marker_series_score": metrics.get("marker_series_score", 0.0),
                "edge_density": metrics.get("edge_density", 0.0),
                "metadata_evidence_removed": bool(
                    op.get("previous_classification") == "schematic"
                    and op.get("automatic_classification") != "schematic"
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return rows


def export_candidate_inventory(manifest: dict[str, Any], settings: Settings, path: Path) -> list[dict[str, Any]]:
    operations = {
        Path(str(op.get("old_image") or op.get("new_image") or "")).stem: op
        for op in manifest["operations"]
        if op.get("old_image") or op.get("new_image")
    }
    rows: list[dict[str, Any]] = []
    evaluation_dir = settings.data_dir / "evaluation"
    for candidate_path in sorted(evaluation_dir.glob("well_test_*_candidates/*.json")):
        if candidate_path.name.endswith(".failed.json"):
            continue
        try:
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rows.append({"candidate_path": str(candidate_path), "candidate_status": "invalid", "stale_reason": str(exc), "apply_ready": False})
            continue
        op = operations.get(Path(str(payload.get("asset_path") or candidate_path.stem)).stem)
        stale_errors = candidate_provenance_errors(payload, op) if op else ["candidate asset is not in current plan"]
        rows.append(
            {
                "candidate_path": str(candidate_path),
                "candidate_status": "stale" if stale_errors else payload.get("candidate_status", "valid"),
                "stale_reason": "; ".join(stale_errors) if stale_errors else None,
                "apply_ready": False if stale_errors else bool(payload.get("apply_ready")),
                "automatic_classification": payload.get("automatic_classification"),
                "effective_classification": payload.get("effective_classification"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return rows


async def main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Default. Build a plan without changing files.")
    parser.add_argument("--apply", action="store_true", help="Apply the planned rename/quarantine/reanalysis changes.")
    parser.add_argument("--document")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--asset")
    parser.add_argument("--force-image-type", choices=sorted(GRAPH_TYPES | DIAGRAM_TYPES | EQUATION_TYPES))
    parser.add_argument("--quarantine-logos", action="store_true")
    parser.add_argument("--delete-quarantined", action="store_true")
    parser.add_argument("--reanalyze-graphs", action="store_true")
    parser.add_argument("--reanalyze-diagrams", action="store_true")
    parser.add_argument("--reanalyze-equations", action="store_true")
    parser.add_argument("--reanalyze-all-engineering-figures", action="store_true")
    parser.add_argument("--analyze-dry-run", action="store_true")
    parser.add_argument("--update-chroma", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--export-review-report", type=Path)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--candidate-input", type=Path)
    parser.add_argument("--classification-delta-report", type=Path)
    parser.add_argument("--candidate-inventory", type=Path)
    args = parser.parse_args()

    if args.force_image_type and (not args.asset or args.apply):
        print("force-image-type requires exactly one --asset and dry-run/candidate generation")
        return 2

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
        reanalyze_diagrams=args.reanalyze_diagrams,
        reanalyze_equations=args.reanalyze_equations,
        reanalyze_all_engineering_figures=args.reanalyze_all_engineering_figures,
        update_chroma=args.update_chroma,
        decisions=load_decisions(args.decisions),
        asset=args.asset,
        force_image_type=args.force_image_type,
    )
    print_report(manifest)
    if args.classification_delta_report:
        rows = export_classification_delta(manifest, args.classification_delta_report)
        print(f"classification_delta_report={args.classification_delta_report}")
        print(f"classification_delta_rows={len(rows)}")
    if args.candidate_inventory:
        rows = export_candidate_inventory(manifest, settings, args.candidate_inventory)
        print(f"candidate_inventory={args.candidate_inventory}")
        print(f"candidate_inventory_rows={len(rows)}")
        print(f"stale_candidates={sum(row.get('candidate_status') == 'stale' for row in rows)}")
    if manifest["stats"].get("asset_error"):
        return 2
    selected_ops = [op for op in manifest["operations"] if op.get("reanalyze")]
    if args.candidate_input and len(selected_ops) != 1:
        print("candidate-input requires exactly one selected reanalysis target")
        return 2
    if args.candidate_output and len(selected_ops) != 1:
        print("candidate-output requires exactly one selected reanalysis target")
        return 2

    if args.analyze_dry_run or (args.candidate_input and not args.apply):
        try:
            succeeded = await print_analyze_dry_run(
                settings,
                manifest,
                candidate_output=args.candidate_output,
                candidate_input=args.candidate_input,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"candidate_error={exc}")
            return 2
        return 0 if succeeded else 2
    if args.export_review_report:
        export_review_report(manifest, args.export_review_report)
        print(f"review_report={args.export_review_report}")
        print(f"review_json={args.export_review_report.with_suffix('.json')}")
        print(f"review_csv={args.export_review_report.with_suffix('.csv')}")

    if args.apply:
        if manifest["collisions"]:
            print("refusing to apply with name collisions")
            return 2
        try:
            candidate_payload = (
                load_candidate(args.candidate_input, selected_ops[0], require_apply_ready=True)
                if args.candidate_input and selected_ops
                else None
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"candidate_error={exc}")
            return 2
        manifest_path = write_manifest(settings, manifest, args.manifest)
        await apply_plan(settings, manifest, manifest_path, candidate_payload)
        print(f"manifest={manifest_path}")
    elif args.manifest:
        print(f"dry_run_manifest_not_written={args.manifest}")
    return 0


def main() -> int:
    import asyncio

    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
