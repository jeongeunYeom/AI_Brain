from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from app.core.config import Settings

if TYPE_CHECKING:
    from app.services.ollama import OllamaClient

CANDIDATE_SCHEMA_VERSION = 2
CLASSIFIER_VERSION = "new-upload-vision-classifier-v1"
ANALYSIS_PROMPT_VERSION = "new-upload-grounding-v1"
SERIALIZER_VERSION = "figure-note-v2"

UNKNOWN_TEXT = "확인할 수 없음"
UNKNOWN_MARKERS = {
    "",
    "unknown",
    "none",
    "null",
    "n/a",
    "not readable",
    "not visible",
    "unreadable",
    "확인할 수 없음",
}

GRAPH_TYPES = {"graph", "chart"}
DIAGRAM_TYPES = {"diagram", "schematic"}
IGNORED_TYPES = {"logo", "page_decoration", "decorative", "photo"}
SUPPORTED_TYPES = GRAPH_TYPES | DIAGRAM_TYPES | {
    "table",
    "equation",
    "photo",
    "logo",
    "page_decoration",
    "unknown_engineering_figure",
    "unknown",
}

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

DIAGRAM_SCHEMA_FIELDS = [
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

CLASSIFICATION_PROMPT = """Analyze only what is directly visible in this image.
Return exactly four concise ASCII English key-value lines:
image_type: one of graph, chart, table, diagram, schematic, equation, photo, logo, page_decoration, unknown_engineering_figure, unknown
confidence: a number from 0.0 to 1.0
readable_labels: true or false
reason: one short directly visible reason
Do not infer the subject area. Do not invent labels, axes, or engineering meaning.
"""

GRAPH_PROMPT = """Analyze only what is directly visible in this graph or chart.
Return concise ASCII English key-value lines with exactly these keys:
image_type, confidence, title, analysis, x_axis, x_axis_unit, x_axis_scale, y_axis, y_axis_unit, y_axis_scale, series_count, series_count_verified, series_descriptions, legend, reference_lines, plateau, peak, decline, slope_changes, trend, engineering_meaning.
Rules:
- confidence must be a number from 0.0 to 1.0.
- Use integer series_count or unknown.
- Use true or false for series_count_verified.
- Separate multiple list items with semicolons on the same line.
- Describe each visible series separately, including line style and marker shape only when clearly visible.
- Distinguish continuous rise, decline, peak, minimum, plateau, and slope changes when visible.
- Report each axis scale as Linear, Logarithmic, or unknown.
- If title, axis, unit, legend, numeric value, or engineering meaning is unreadable, write unknown.
- Do not guess series names or engineering meaning.
- Do not use unfinished phrases.
"""

DIAGRAM_PROMPT = """Analyze only what is directly visible in this engineering diagram or schematic.
Return concise ASCII English key-value lines with exactly these keys:
image_type, confidence, title, analysis, components, component_labels, connections, flow_directions, annotations, legend, engineering_meaning.
Rules:
- confidence must be a number from 0.0 to 1.0.
- Separate multiple list items with semicolons on the same line.
- Do not invent axes, trends, series, components, labels, or engineering purpose.
- If a value is unreadable, write unknown.
"""

EQUATION_PROMPT = """Analyze only what is directly visible in this equation image.
Return concise ASCII English key-value lines with exactly these keys:
image_type, confidence, title, analysis, equation_text, variables, units, assumptions, engineering_meaning.
Do not reconstruct unreadable symbols. If uncertain, write unknown.
"""


@dataclass(slots=True)
class FigureAnalysisResult:
    status: str
    classification: str
    confidence: float
    note_text: str | None
    note_path: Path | None
    candidate_path: Path | None
    candidate: dict[str, Any]
    vision_calls: int
    should_index: bool


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparable_note_data(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if key not in {"created_at", "legacy_note_backup"}
    }


def data_sha256(data: dict[str, Any]) -> str:
    encoded = json.dumps(
        comparable_note_data(data),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generator_git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


KEY_VALUE_KEYS = {
    "image_type",
    "confidence",
    "readable_labels",
    "reason",
    "title",
    "analysis",
    "x_axis",
    "x_axis_unit",
    "x_axis_scale",
    "y_axis",
    "y_axis_unit",
    "y_axis_scale",
    "series_count",
    "series_count_verified",
    "series_descriptions",
    "legend",
    "reference_lines",
    "plateau",
    "plateaus",
    "peak",
    "peaks",
    "decline",
    "declines",
    "slope_changes",
    "trend",
    "trend_summary",
    "engineering_meaning",
    "components",
    "component_labels",
    "connections",
    "flow_directions",
    "annotations",
    "equation_text",
    "variables",
    "units",
    "assumptions",
}

_KEY_PATTERN = "|".join(
    sorted(
        (re.escape(key).replace(r"\_", r"[_ ]+") for key in KEY_VALUE_KEYS),
        key=len,
        reverse=True,
    )
)
KEY_VALUE_BOUNDARY_RE = re.compile(
    rf"(?:^|[;\n\r])\s*(?:[-*]\s*)?(?P<key>{_KEY_PATTERN})\s*[:：]\s*",
    re.IGNORECASE,
)


def canonical_key(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().lower())


def parse_key_values(text: str) -> dict[str, str]:
    """Parse both one-key-per-line and semicolon-collapsed Vision replies.

    Some local Vision models ignore requested line breaks and return the full response
    as one semicolon-delimited line.  Split only when a semicolon/newline is followed
    by a known schema key so semicolons inside list values remain intact.
    """
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json|text)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    matches = list(KEY_VALUE_BOUNDARY_RE.finditer(cleaned))
    values: dict[str, str] = {}

    if matches:
        for index, match in enumerate(matches):
            key = canonical_key(match.group("key"))
            end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
            value = cleaned[match.end():end].strip(" \t\r\n;")
            if not value:
                continue
            if key in values and values[key]:
                values[key] = f"{values[key]}; {value}"
            else:
                values[key] = value
        return values

    # Conservative fallback for unexpected but still line-oriented replies.
    current_key: str | None = None
    list_items: dict[str, list[str]] = {}
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        match = re.match(
            r"^\s*(?:[-*]\s*)?([A-Za-z_][A-Za-z0-9_ ]*)\s*[:：]\s*(.*)$",
            line,
        )
        if match:
            key = canonical_key(match.group(1))
            value = match.group(2).strip()
            if value:
                values[key] = value
                current_key = None
            else:
                current_key = key
            continue

        item_match = re.match(r"^\s*(?:[-*]|\d+[.)])\s*(.+)$", line)
        if current_key and item_match:
            list_items.setdefault(current_key, []).append(item_match.group(1).strip())

    for key, items in list_items.items():
        values[key] = "; ".join(items)
    return values


def is_unknown(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in UNKNOWN_MARKERS or "확인할 수 없음" in text


def nullable_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return None if is_unknown(text) else text


def normalize_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if is_unknown(text):
        return []
    parts = re.split(r"\s*(?:;|\n|\r|\u2022)\s*", text)
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        clean = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part).strip()
        key = re.sub(r"\s+", " ", clean).lower()
        if clean and not is_unknown(clean) and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def normalize_series_descriptions(value: Any, series_count: int | None) -> list[str]:
    descriptions = normalize_list(value)
    if series_count is None or series_count < 2 or len(descriptions) != 1:
        return descriptions

    # qwen2.5vl occasionally separates series with commas despite the prompt asking
    # for semicolons.  Split only for a verified multi-series graph and only when
    # the resulting count is plausible, avoiding generic comma-heavy prose.
    comma_parts = [part.strip() for part in descriptions[0].split(",") if part.strip()]
    if 2 <= len(comma_parts) <= min(max(series_count, 2), 6):
        return comma_parts
    return descriptions


def parse_confidence(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if 0.0 <= parsed <= 1.0 else default


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() == "true"


def parse_int(value: Any) -> int | None:
    if is_unknown(value):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def has_unclosed_punctuation(text: str) -> bool:
    for left, right in [("(", ")"), ("[", "]")]:
        if text.count(left) != text.count(right):
            return True
    for quote in ['"', "'"]:
        if text.count(quote) % 2:
            return True
    return False


def is_incomplete_string(text: str) -> bool:
    stripped = text.strip()
    lower = stripped.lower()
    if len(stripped) < 8:
        return True
    if stripped.endswith("-") or has_unclosed_punctuation(stripped):
        return True
    return bool(re.search(r"(?:\band\b|\bor\b|\bat\b|\bfor\b|[xy]\s*=)\s*$", lower))


def prepare_vision_image(image_path: Path) -> tuple[Path, dict[str, Any]]:
    with Image.open(image_path) as image:
        image.load()
        prepared = ImageOps.autocontrast(image.convert("RGB"))

    tmp = NamedTemporaryFile(
        prefix=f"{image_path.stem}_vision_",
        suffix=".png",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    prepared.save(tmp_path, format="PNG")
    return tmp_path, {
        "rgb_conversion": True,
        "autocontrast": True,
        "brightness_factor": 1.0,
        "contrast_factor": 1.0,
    }


def image_metrics(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        with Image.open(path) as image:
            width, height = image.size
            gray = ImageOps.grayscale(image.convert("RGB"))
            small = gray.copy()
            small.thumbnail((256, 256))
            stat = ImageStat.Stat(small)
            edges = small.filter(ImageFilter.FIND_EDGES)
            edge_pixels = list(edges.getdata())
            edge_density = sum(value >= 26 for value in edge_pixels) / max(
                small.width * small.height,
                1,
            )
            pixels = list(small.getdata())
            dark_ratio = sum(value <= 35 for value in pixels) / max(len(pixels), 1)
            bright_ratio = sum(value >= 220 for value in pixels) / max(len(pixels), 1)

            enhanced = ImageOps.autocontrast(small)
            enhanced_pixels = enhanced.load()
            active_columns = sum(
                any(enhanced_pixels[x, y] >= 64 for y in range(enhanced.height))
                for x in range(enhanced.width)
            ) / max(enhanced.width, 1)
            active_rows = sum(
                any(enhanced_pixels[x, y] >= 64 for x in range(enhanced.width))
                for y in range(enhanced.height)
            ) / max(enhanced.height, 1)
            enhanced_bright_ratio = sum(
                value >= 64 for value in enhanced.getdata()
            ) / max(enhanced.width * enhanced.height, 1)
            marker_series_detected = (
                active_columns >= 0.15
                and active_rows >= 0.20
                and 0.001 <= enhanced_bright_ratio <= 0.05
            )
            marker_series_score = min(
                1.0,
                active_columns / 0.15,
                active_rows / 0.20,
                enhanced_bright_ratio / 0.001,
            )

        return {
            "width": width,
            "height": height,
            "file_size": size,
            "aspect_ratio": width / max(height, 1),
            "contrast": round(float(stat.stddev[0]), 2),
            "brightness": round(float(stat.mean[0]), 2),
            "edge_density": round(float(edge_density), 5),
            "dark_ratio": round(float(dark_ratio), 5),
            "bright_ratio": round(float(bright_ratio), 5),
            "enhanced_active_columns": round(float(active_columns), 5),
            "enhanced_active_rows": round(float(active_rows), 5),
            "enhanced_bright_ratio": round(float(enhanced_bright_ratio), 6),
            "marker_series_detected": marker_series_detected,
            "marker_series_score": round(float(marker_series_score), 4),
        }
    except (OSError, UnidentifiedImageError, ValueError):
        return {
            "width": 0,
            "height": 0,
            "file_size": 0,
            "aspect_ratio": 0.0,
            "contrast": 0.0,
            "brightness": 0.0,
            "edge_density": 0.0,
            "dark_ratio": 0.0,
            "bright_ratio": 0.0,
            "enhanced_active_columns": 0.0,
            "enhanced_active_rows": 0.0,
            "enhanced_bright_ratio": 0.0,
            "marker_series_detected": False,
            "marker_series_score": 0.0,
        }

def should_analyze(metrics: dict[str, Any]) -> tuple[bool, str]:
    width = int(metrics.get("width") or 0)
    height = int(metrics.get("height") or 0)
    area = width * height
    size = int(metrics.get("file_size") or 0)
    aspect = float(metrics.get("aspect_ratio") or 0.0)
    contrast = float(metrics.get("contrast") or 0.0)
    edge_density = float(metrics.get("edge_density") or 0.0)

    if size < 100 or width < 150 or height < 120 or area < 30_000:
        return False, "too_small_or_low_resolution"
    if width < 300 and height < 300 and size < 20_000:
        return False, "small_square_thumbnail"
    if aspect > 12 or aspect < 0.08:
        return False, "extreme_aspect_ratio"
    if contrast < 4 and edge_density < 0.01:
        return False, "low_contrast_and_low_edge_density"
    return True, "technical_figure_candidate"


def source_is_dark(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("brightness") or 0.0) < 45
        or float(metrics.get("dark_ratio") or 0.0) > 0.75
    )


def figure_priority(
    metrics: dict[str, Any],
    page_text: str = "",
) -> dict[str, Any]:
    analyze, reason = should_analyze(metrics)
    if not analyze:
        return {
            "score": -1000.0,
            "forced_classification": None,
            "reason": reason,
            "should_analyze": False,
        }

    width = int(metrics.get("width") or 0)
    height = int(metrics.get("height") or 0)
    area = width * height
    aspect = float(metrics.get("aspect_ratio") or 0.0)
    brightness = float(metrics.get("brightness") or 0.0)
    edge_density = float(metrics.get("edge_density") or 0.0)
    marker_series = bool(metrics.get("marker_series_detected"))
    marker_score = float(metrics.get("marker_series_score") or 0.0)
    wide_rectangle = 1.2 <= aspect <= 5.5 and width >= 350
    dark_graph = wide_rectangle and (
        (brightness < 90 and edge_density >= 0.004)
        or (
            brightness < 10
            and int(metrics.get("file_size") or 0) >= 50_000
            and (edge_density >= 0.003 or marker_series)
        )
    )

    lower = re.sub(r"\s+", " ", str(page_text or "").lower())
    graph_terms = (
        "graph",
        "plot",
        "log-log",
        "semi-log",
        "pressure derivative",
        "pressure transient",
        "drawdown",
        "buildup",
        "rft",
        "pressure vs",
        "delta p",
        "time function",
    )
    diagram_terms = (
        "schematic",
        "diagram",
        "flow path",
        "wellbore",
        "reservoir model",
        "fault block",
        "boundary model",
    )
    equation_terms = (
        "equation",
        "formula",
        "where:",
        "defined as",
    )

    graph_hits = sum(term in lower for term in graph_terms)
    diagram_hits = sum(term in lower for term in diagram_terms)
    equation_hits = sum(term in lower for term in equation_terms)

    score = 0.0
    forced: str | None = None
    reasons: list[str] = []

    if area >= 300_000:
        score += 10.0
    if wide_rectangle:
        score += 18.0
        reasons.append("wide_rectangle")
    if edge_density >= 0.02:
        score += 8.0
    if marker_series:
        score += 45.0 * max(marker_score, 0.5)
        reasons.append("marker_series")
    if graph_hits:
        score += min(45.0, graph_hits * 18.0)
        reasons.append(f"graph_text_hits={graph_hits}")
    if diagram_hits:
        score += min(30.0, diagram_hits * 15.0)
        reasons.append(f"diagram_text_hits={diagram_hits}")
    if equation_hits:
        score += min(20.0, equation_hits * 10.0)
        reasons.append(f"equation_text_hits={equation_hits}")
    if dark_graph:
        score += 120.0
        forced = "graph"
        reasons.append("dark_graph")
    elif wide_rectangle and marker_series and graph_hits >= 1:
        score += 90.0
        forced = "graph"
        reasons.append("graph_marker_text_consensus")
    elif wide_rectangle and graph_hits >= 2:
        score += 75.0
        forced = "graph"
        reasons.append("graph_layout_text_consensus")
    elif diagram_hits >= 2 and graph_hits == 0:
        score += 55.0
        forced = "schematic"
        reasons.append("diagram_text_consensus")
    elif equation_hits >= 2 and graph_hits == 0:
        score += 40.0
        forced = "equation"
        reasons.append("equation_text_consensus")

    return {
        "score": round(score, 3),
        "forced_classification": forced,
        "reason": ",".join(reasons) or "generic_technical_candidate",
        "should_analyze": True,
        "dark_graph": dark_graph,
        "graph_hits": graph_hits,
        "diagram_hits": diagram_hits,
        "equation_hits": equation_hits,
    }


def normalize_classification(raw: str) -> str:
    lower = str(raw or "").strip().lower()
    if "log-log" in lower or "plot" in lower or "graph" in lower:
        return "graph"
    if "chart" in lower:
        return "chart"
    if "schematic" in lower:
        return "schematic"
    if "diagram" in lower:
        return "diagram"
    if "equation" in lower or "formula" in lower:
        return "equation"
    if "table" in lower:
        return "table"
    if "logo" in lower:
        return "logo"
    if "decoration" in lower or "ornament" in lower:
        return "page_decoration"
    if "photo" in lower:
        return "photo"
    if "unknown_engineering" in lower:
        return "unknown_engineering_figure"
    if lower in SUPPORTED_TYPES:
        return lower
    return "unknown"


def graph_metadata(
    text: str,
    *,
    settings: Settings,
    readable_labels: bool,
) -> tuple[dict[str, Any], list[str], list[str]]:
    fields = parse_key_values(text)
    errors: list[str] = []
    reasons: list[str] = []

    confidence = parse_confidence(fields.get("confidence"))
    if confidence == 0.0 and str(fields.get("confidence") or "").strip() not in {"0", "0.0"}:
        errors.append("confidence must be a 0.0-1.0 float")

    analysis = nullable_text(fields.get("analysis"))
    if not analysis:
        errors.append("analysis is required")

    title = nullable_text(fields.get("title"))
    x_axis = nullable_text(fields.get("x_axis"))
    x_unit = nullable_text(fields.get("x_axis_unit"))
    y_axis = nullable_text(fields.get("y_axis"))
    y_unit = nullable_text(fields.get("y_axis_unit"))
    legend = nullable_text(fields.get("legend"))
    engineering = nullable_text(fields.get("engineering_meaning"))
    series_count = parse_int(fields.get("series_count"))
    series_count_verified = parse_bool(fields.get("series_count_verified")) and series_count is not None

    series_descriptions = normalize_series_descriptions(
        fields.get("series_descriptions"),
        series_count,
    )
    reference_lines = normalize_list(fields.get("reference_lines"))
    plateaus = normalize_list(fields.get("plateaus") or fields.get("plateau"))
    peaks = normalize_list(fields.get("peaks") or fields.get("peak"))
    declines = normalize_list(fields.get("declines") or fields.get("decline"))
    slope_changes = normalize_list(fields.get("slope_changes"))
    trend = nullable_text(fields.get("trend") or fields.get("trend_summary"))

    list_fields = {
        "series_descriptions": series_descriptions,
        "reference_lines": reference_lines,
        "plateaus": plateaus,
        "peaks": peaks,
        "declines": declines,
        "slope_changes": slope_changes,
    }
    for name, values in list_fields.items():
        cleaned = [value for value in values if not is_incomplete_string(value)]
        if len(cleaned) != len(values):
            reasons.append(f"incomplete {name} items removed")
        list_fields[name] = cleaned

    series_descriptions = list_fields["series_descriptions"]
    reference_lines = list_fields["reference_lines"]
    plateaus = list_fields["plateaus"]
    peaks = list_fields["peaks"]
    declines = list_fields["declines"]
    slope_changes = list_fields["slope_changes"]

    if trend and is_incomplete_string(trend):
        reasons.append("incomplete trend removed")
        trend = None

    label_grounding = bool(
        readable_labels
        and (title or x_axis or y_axis or legend or engineering)
    )
    required_descriptions = 1 if series_count == 1 else 2
    visual_trend_grounding = bool(
        series_count_verified
        and series_count is not None
        and series_count >= 1
        and len(series_descriptions) >= min(series_count, required_descriptions)
        and (trend or reference_lines or plateaus or peaks or declines or slope_changes)
    )
    information_quality = bool(
        analysis
        and (
            series_descriptions
            or trend
            or reference_lines
            or plateaus
            or peaks
            or declines
            or slope_changes
        )
    )

    if not information_quality:
        reasons.append("graph lacks grounded series, trend, or reference information")
    if series_count is not None and series_count >= 2 and len(series_descriptions) < 2:
        reasons.append("multiple visible series were not described separately")
    if series_count is not None and not series_count_verified:
        reasons.append("series_count is not verified")

    semantic_grounding = label_grounding or visual_trend_grounding
    trend_grounding = bool(
        trend
        and (
            series_count is None
            or series_count <= 1
            or len(series_descriptions) >= 2
        )
    )
    if not semantic_grounding:
        reasons.append("semantic or visual grounding not verified")
    if series_count is not None and series_count >= 2 and not trend_grounding:
        reasons.append("multi-series trend grounding not verified")

    metadata = {
        "image_type": normalize_classification(fields.get("image_type") or "graph"),
        "confidence": confidence,
        "title": title,
        "title_verified": bool(title and readable_labels),
        "analysis": analysis,
        "x_axis": x_axis,
        "x_axis_unit": x_unit,
        "x_axis_scale": str(fields.get("x_axis_scale") or "unknown").strip(),
        "x_axis_verified": bool(x_axis and readable_labels),
        "y_axis": y_axis,
        "y_axis_unit": y_unit,
        "y_axis_scale": str(fields.get("y_axis_scale") or "unknown").strip(),
        "y_axis_verified": bool(y_axis and readable_labels),
        "series_count": series_count,
        "series_count_verified": series_count_verified,
        "series_descriptions": series_descriptions,
        "legend": legend,
        "legend_verified": bool(legend and readable_labels),
        "reference_lines": reference_lines,
        "plateaus": plateaus,
        "peaks": peaks,
        "declines": declines,
        "slope_changes": slope_changes,
        "trend_summary": trend,
        "trend_verified": bool(trend),
        "engineering_meaning": engineering or UNKNOWN_TEXT,
        "engineering_meaning_verified": bool(engineering and readable_labels),
        "vision_model": settings.vision_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "legacy_note_backup": None,
    }
    metadata["_schema_valid"] = not errors
    metadata["_information_quality"] = information_quality
    metadata["_semantic_grounding"] = semantic_grounding
    metadata["_trend_grounding"] = trend_grounding
    return metadata, errors, reasons


def diagram_metadata(
    text: str,
    *,
    settings: Settings,
    readable_labels: bool,
) -> tuple[dict[str, Any], list[str], list[str]]:
    fields = parse_key_values(text)
    errors: list[str] = []
    reasons: list[str] = []
    confidence = parse_confidence(fields.get("confidence"))
    analysis = nullable_text(fields.get("analysis"))
    components = normalize_list(fields.get("components"))
    labels = normalize_list(fields.get("component_labels"))
    connections = normalize_list(fields.get("connections"))
    flow = normalize_list(fields.get("flow_directions"))
    annotations = normalize_list(fields.get("annotations"))
    title = nullable_text(fields.get("title"))
    legend = nullable_text(fields.get("legend"))
    engineering = nullable_text(fields.get("engineering_meaning"))

    if not analysis and not any([components, labels, connections, flow, annotations]):
        errors.append("diagram analysis is empty")
    information_quality = bool(analysis or components or labels or connections or flow or annotations)
    semantic_grounding = bool(
        components
        or connections
        or annotations
        or (readable_labels and (title or labels or legend or engineering))
    )
    if not semantic_grounding:
        reasons.append("diagram grounding not verified")

    metadata = {
        "image_type": normalize_classification(fields.get("image_type") or "diagram"),
        "confidence": confidence,
        "title": title,
        "title_verified": bool(title and readable_labels),
        "analysis": analysis,
        "components": components,
        "component_labels": labels,
        "connections": connections,
        "flow_directions": flow,
        "annotations": annotations,
        "legend": legend,
        "legend_verified": bool(legend and readable_labels),
        "engineering_meaning": engineering or UNKNOWN_TEXT,
        "engineering_meaning_verified": bool(engineering and readable_labels),
        "vision_model": settings.vision_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "legacy_note_backup": None,
    }
    metadata["_schema_valid"] = not errors
    metadata["_information_quality"] = information_quality
    metadata["_semantic_grounding"] = semantic_grounding
    metadata["_trend_grounding"] = True
    return metadata, errors, reasons


def serialize_note_data(metadata: dict[str, Any]) -> str:
    order = DIAGRAM_SCHEMA_FIELDS if metadata.get("image_type") in DIAGRAM_TYPES else GRAPH_SCHEMA_FIELDS
    lines = ["[Figure Note Metadata]"]
    for key in order:
        value = metadata.get(key)
        if isinstance(value, list):
            if value:
                lines.append(f"{key}:")
                lines.extend(f"  - {item}" for item in value)
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


class FigureAnalysisService:
    def __init__(self, settings: Settings, ollama: OllamaClient):
        self.settings = settings
        self.ollama = ollama

    @property
    def candidate_root(self) -> Path:
        return self.settings.figure_candidates_dir

    def source_is_dark(self, image_path: Path) -> bool:
        return source_is_dark(image_metrics(image_path))

    def priority_for_image(
        self,
        image_path: Path,
        page_text: str = "",
    ) -> dict[str, Any]:
        return figure_priority(image_metrics(image_path), page_text)

    async def analyze_figure(
        self,
        *,
        document_name: str,
        document_id: str,
        page_number: int,
        image_index: int,
        image_path: Path,
        fallback_image_path: Path | None = None,
        remaining_vision_calls: int | None = None,
        forced_classification: str | None = None,
    ) -> FigureAnalysisResult:
        metrics = image_metrics(image_path)
        analyze, reason = should_analyze(metrics)
        if not analyze:
            return FigureAnalysisResult(
                status="ignored",
                classification="page_decoration",
                confidence=0.0,
                note_text=None,
                note_path=None,
                candidate_path=None,
                candidate={"status": "ignored", "reason": reason, "metrics": metrics},
                vision_calls=0,
                should_index=False,
            )

        normalized_forced = normalize_classification(forced_classification or "")
        required_calls = (
            1
            if normalized_forced in GRAPH_TYPES | DIAGRAM_TYPES | {"equation"}
            else 2
        )
        if remaining_vision_calls is not None and remaining_vision_calls < required_calls:
            return self._write_review_without_vision(
                document_name=document_name,
                document_id=document_id,
                page_number=page_number,
                image_index=image_index,
                image_path=image_path,
                reason="vision_call_limit_reached",
                metrics=metrics,
            )

        try:
            first = await self._analyze_input(
                image_path=image_path,
                metrics=metrics,
                forced_classification=normalized_forced or None,
                remaining_vision_calls=remaining_vision_calls,
            )
        except Exception as exc:  # one figure failure must not stop PDF ingestion
            first = {
                "status": "failed",
                "classification": "unknown_engineering_figure",
                "confidence": 0.0,
                "vision_calls": 1,
                "validation_errors": [f"{type(exc).__name__}: {exc}"],
                "schema_valid": False,
                "information_quality_passed": False,
                "semantic_grounding_passed": False,
                "trend_grounding_passed": False,
            }
        used_calls = int(first.get("vision_calls") or 0)

        if (
            first.get("status") != "valid"
            and fallback_image_path
            and fallback_image_path.is_file()
        ):
            remaining_after_first = None
            if remaining_vision_calls is not None:
                remaining_after_first = max(0, remaining_vision_calls - used_calls)
            if remaining_after_first is None or remaining_after_first >= required_calls:
                fallback_metrics = image_metrics(fallback_image_path)
                try:
                    second = await self._analyze_input(
                        image_path=fallback_image_path,
                        metrics=fallback_metrics,
                        forced_classification=normalized_forced or None,
                        remaining_vision_calls=remaining_after_first,
                    )
                except Exception as exc:  # keep the original result when fallback fails
                    second = {
                        "status": "failed",
                        "classification": first.get("classification", "unknown"),
                        "confidence": 0.0,
                        "vision_calls": 1,
                        "validation_errors": [f"fallback {type(exc).__name__}: {exc}"],
                    }
                used_calls += int(second.get("vision_calls") or 0)
                if self._result_rank(second) > self._result_rank(first):
                    first = second
                    first["fallback_used"] = True
                    first["fallback_path"] = str(fallback_image_path)

        return self._persist_result(
            document_name=document_name,
            document_id=document_id,
            page_number=page_number,
            image_index=image_index,
            image_path=image_path,
            analysis=first,
            vision_calls=used_calls,
            metrics=metrics,
        )

    async def _analyze_input(
        self,
        *,
        image_path: Path,
        metrics: dict[str, Any],
        forced_classification: str | None = None,
        remaining_vision_calls: int | None = None,
    ) -> dict[str, Any]:
        prepared_path, transform = prepare_vision_image(image_path)
        vision_calls = 0
        try:
            classification = normalize_classification(forced_classification or "")
            readable_labels = False
            classification_confidence = 0.0
            classification_text = ""

            if not forced_classification or classification in {"unknown", "unknown_engineering_figure"}:
                if remaining_vision_calls is not None and remaining_vision_calls < 1:
                    return {
                        "status": "review_required",
                        "classification": "unknown_engineering_figure",
                        "confidence": 0.0,
                        "vision_calls": 0,
                        "manual_review_reasons": ["vision_call_limit_reached"],
                        "schema_valid": False,
                        "information_quality_passed": False,
                        "semantic_grounding_passed": False,
                        "trend_grounding_passed": False,
                    }
                classification_text = (
                    await self.ollama.describe_image(
                        prepared_path,
                        prompt=CLASSIFICATION_PROMPT,
                        num_predict=180,
                    )
                ).strip()
                vision_calls += 1
                class_fields = parse_key_values(classification_text)
                classification = normalize_classification(class_fields.get("image_type"))
                readable_labels = parse_bool(class_fields.get("readable_labels"))
                classification_confidence = parse_confidence(class_fields.get("confidence"))
            else:
                # A local preclassifier can safely prioritize a figure type, but it
                # cannot verify that labels are readable. The detailed Vision call
                # must still ground titles, axes, units, and legends conservatively.
                readable_labels = False

            if classification in IGNORED_TYPES:
                return {
                    "status": "ignored",
                    "classification": classification,
                    "confidence": classification_confidence,
                    "vision_calls": vision_calls,
                    "classification_text": classification_text,
                }

            if classification not in GRAPH_TYPES | DIAGRAM_TYPES | {"equation"}:
                return {
                    "status": "review_required",
                    "classification": classification,
                    "confidence": classification_confidence,
                    "vision_calls": vision_calls,
                    "classification_text": classification_text,
                    "manual_review_reasons": ["figure type is not safely auto-serializable"],
                    "schema_valid": True,
                    "information_quality_passed": False,
                    "semantic_grounding_passed": False,
                    "trend_grounding_passed": False,
                }

            if remaining_vision_calls is not None and vision_calls >= remaining_vision_calls:
                return {
                    "status": "review_required",
                    "classification": classification,
                    "confidence": classification_confidence,
                    "vision_calls": vision_calls,
                    "classification_text": classification_text,
                    "manual_review_reasons": ["vision_call_limit_reached_before_detailed_analysis"],
                    "schema_valid": True,
                    "information_quality_passed": False,
                    "semantic_grounding_passed": False,
                    "trend_grounding_passed": False,
                }

            prompt = (
                GRAPH_PROMPT
                if classification in GRAPH_TYPES
                else DIAGRAM_PROMPT
                if classification in DIAGRAM_TYPES
                else EQUATION_PROMPT
            )
            raw_text = (
                await self.ollama.describe_image(
                    prepared_path,
                    prompt=prompt,
                    num_predict=900,
                )
            ).strip()
            vision_calls += 1

            if not raw_text:
                return {
                    "status": "failed",
                    "classification": classification,
                    "confidence": classification_confidence,
                    "vision_calls": vision_calls,
                    "classification_text": classification_text,
                    "raw_vision_text": raw_text,
                    "validation_errors": ["empty detailed vision response"],
                    "schema_valid": False,
                    "information_quality_passed": False,
                    "semantic_grounding_passed": False,
                    "trend_grounding_passed": False,
                }

            if classification in GRAPH_TYPES:
                metadata, errors, reasons = graph_metadata(
                    raw_text,
                    settings=self.settings,
                    readable_labels=readable_labels,
                )
            elif classification in DIAGRAM_TYPES:
                metadata, errors, reasons = diagram_metadata(
                    raw_text,
                    settings=self.settings,
                    readable_labels=readable_labels,
                )
            else:
                return {
                    "status": "review_required",
                    "classification": classification,
                    "confidence": classification_confidence,
                    "vision_calls": vision_calls,
                    "classification_text": classification_text,
                    "raw_vision_text": raw_text,
                    "manual_review_reasons": ["equation figures require exact symbol verification"],
                    "schema_valid": True,
                    "information_quality_passed": bool(raw_text),
                    "semantic_grounding_passed": False,
                    "trend_grounding_passed": True,
                }

            schema_valid = bool(metadata.pop("_schema_valid"))
            information_quality = bool(metadata.pop("_information_quality"))
            semantic_grounding = bool(metadata.pop("_semantic_grounding"))
            trend_grounding = bool(metadata.pop("_trend_grounding"))
            manual_reasons = list(dict.fromkeys(reasons))

            if errors:
                status = "failed"
            elif (
                schema_valid
                and information_quality
                and semantic_grounding
                and trend_grounding
                and not manual_reasons
            ):
                status = "valid"
            else:
                status = "review_required"

            return {
                "status": status,
                "classification": classification,
                "confidence": metadata.get("confidence", classification_confidence),
                "vision_calls": vision_calls,
                "classification_text": classification_text,
                "raw_vision_text": raw_text,
                "metadata": metadata,
                "validation_errors": errors,
                "manual_review_reasons": manual_reasons,
                "schema_valid": schema_valid,
                "information_quality_passed": information_quality,
                "semantic_grounding_passed": semantic_grounding,
                "trend_grounding_passed": trend_grounding,
                "analysis_input_sha256": file_sha256(prepared_path),
                "analysis_transform": transform,
                "input_metrics": metrics,
            }
        finally:
            prepared_path.unlink(missing_ok=True)

    def _persist_result(
        self,
        *,
        document_name: str,
        document_id: str,
        page_number: int,
        image_index: int,
        image_path: Path,
        analysis: dict[str, Any],
        vision_calls: int,
        metrics: dict[str, Any],
    ) -> FigureAnalysisResult:
        status = str(analysis.get("status") or "failed")
        classification = str(analysis.get("classification") or "unknown")
        confidence = float(analysis.get("confidence") or 0.0)
        metadata = analysis.get("metadata")

        if status == "ignored":
            return FigureAnalysisResult(
                status=status,
                classification=classification,
                confidence=confidence,
                note_text=None,
                note_path=None,
                candidate_path=None,
                candidate=analysis,
                vision_calls=vision_calls,
                should_index=False,
            )

        candidate_dir = self.candidate_root / document_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        note_path = self.settings.figure_notes_dir / f"{image_path.stem}.md"

        if isinstance(metadata, dict):
            metadata = dict(metadata)
            metadata.update(
                {
                    "document_name": document_name,
                    "document_id": document_id,
                    "page_number": page_number,
                    "image_index": image_index,
                    "image_path": str(image_path),
                    "vision_model": self.settings.vision_model,
                    "legacy_note_backup": None,
                }
            )

        payload = {
            "candidate_status": status,
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "classifier_version": CLASSIFIER_VERSION,
            "analysis_prompt_version": ANALYSIS_PROMPT_VERSION,
            "serializer_version": SERIALIZER_VERSION,
            "generator_git_commit": generator_git_commit(),
            "asset_path": str(image_path),
            "asset_sha256": file_sha256(image_path),
            "document_id": document_id,
            "document_name": document_name,
            "page_number": page_number,
            "image_index": image_index,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vision_model": self.settings.vision_model,
            "automatic_classification": classification,
            "effective_classification": classification,
            "classification_confidence": confidence,
            "source_metrics": metrics,
            "source_is_dark": source_is_dark(metrics),
            "fallback_used": bool(analysis.get("fallback_used")),
            "fallback_path": analysis.get("fallback_path"),
            "semantic_grounding_passed": bool(analysis.get("semantic_grounding_passed")),
            "trend_grounding_passed": bool(analysis.get("trend_grounding_passed")),
            "manual_review_required": status == "review_required",
            "manual_review_reasons": list(analysis.get("manual_review_reasons") or []),
            "apply_ready": status == "valid",
            "schema_valid": bool(analysis.get("schema_valid")),
            "information_quality_passed": bool(analysis.get("information_quality_passed")),
            "vision_call_count": vision_calls,
            "classification_response": analysis.get("classification_text"),
            "raw_vision_text": analysis.get("raw_vision_text"),
            "analysis_input_sha256": analysis.get("analysis_input_sha256"),
            "analysis_transform": analysis.get("analysis_transform"),
            "validation_errors": list(analysis.get("validation_errors") or []),
            "final_note_data": metadata,
            "final_note_data_sha256": data_sha256(metadata) if isinstance(metadata, dict) else None,
        }

        suffix = (
            ".json"
            if status == "valid"
            else ".review_required.json"
            if status == "review_required"
            else ".failed.json"
        )
        candidate_path = candidate_dir / f"{image_path.stem}{suffix}"
        candidate_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        note_text: str | None = None
        should_index = False
        persisted_note_path: Path | None = None
        if status == "valid" and isinstance(metadata, dict):
            note_text = serialize_note_data(metadata)
            note_path.write_text(note_text, encoding="utf-8")
            should_index = confidence >= self.settings.figure_note_min_confidence
            persisted_note_path = note_path

        return FigureAnalysisResult(
            status=status,
            classification=classification,
            confidence=confidence,
            note_text=note_text,
            note_path=persisted_note_path,
            candidate_path=candidate_path,
            candidate=payload,
            vision_calls=vision_calls,
            should_index=should_index,
        )

    def _write_review_without_vision(
        self,
        *,
        document_name: str,
        document_id: str,
        page_number: int,
        image_index: int,
        image_path: Path,
        reason: str,
        metrics: dict[str, Any],
    ) -> FigureAnalysisResult:
        return self._persist_result(
            document_name=document_name,
            document_id=document_id,
            page_number=page_number,
            image_index=image_index,
            image_path=image_path,
            analysis={
                "status": "review_required",
                "classification": "unknown_engineering_figure",
                "confidence": 0.0,
                "manual_review_reasons": [reason],
                "schema_valid": True,
                "information_quality_passed": False,
                "semantic_grounding_passed": False,
                "trend_grounding_passed": False,
            },
            vision_calls=0,
            metrics=metrics,
        )

    @staticmethod
    def _result_rank(result: dict[str, Any]) -> tuple[int, int, int, float]:
        status_rank = {"failed": 0, "review_required": 1, "valid": 2, "ignored": -1}
        return (
            status_rank.get(str(result.get("status")), 0),
            int(bool(result.get("semantic_grounding_passed"))),
            int(bool(result.get("trend_grounding_passed"))),
            float(result.get("confidence") or 0.0),
        )
