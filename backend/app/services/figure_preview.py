from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat


PREVIEW_ALGORITHM_VERSION = "v5_pdf_crop_rotation_final"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_RENDER_DPI = 250
DEFAULT_CROP_PADDING = 0.02

_preview_lock = threading.Lock()
_ASSET_POSITION_RE = re.compile(r"_p0*(\d+)_fig0*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class PreviewResult:
    path: Path
    source_type: str
    rotation_applied: int
    enhanced: bool

    @property
    def name(self) -> str:
        return self.path.name

    def __fspath__(self) -> str:
        return str(self.path)


@dataclass(frozen=True)
class _PdfFigure:
    pdf_path: Path
    pdf_sha256: str
    page_index: int
    page_number: int
    xref: int
    bbox: tuple[float, float, float, float]
    transform: tuple[float, float, float, float, float, float]
    crop_rotation: int


class FigurePreviewService:
    """Create display-only previews, preferring rendered PDF figure crops."""

    def __init__(
        self,
        preview_dir: Path,
        overrides_path: Path,
        *,
        raw_dir: Path | None = None,
        metadata_dir: Path | None = None,
        algorithm_version: str = PREVIEW_ALGORITHM_VERSION,
        render_dpi: int = DEFAULT_RENDER_DPI,
        crop_padding: float = DEFAULT_CROP_PADDING,
    ) -> None:
        self.preview_dir = Path(preview_dir)
        self.overrides_path = Path(overrides_path)
        data_dir = self.preview_dir.parent
        self.raw_dir = Path(raw_dir) if raw_dir else data_dir / "raw"
        self.metadata_dir = (
            Path(metadata_dir) if metadata_dir else data_dir / "metadata"
        )
        self.algorithm_version = algorithm_version
        self.render_dpi = max(72, int(render_dpi))
        self.crop_padding = max(0.0, min(0.10, float(crop_padding)))
        self.logger = logging.getLogger(__name__)
        self._pdf_resolution_cache: dict[
            tuple[str | None, str | None],
            tuple[Path, str] | None,
        ] = {}

    def get_or_create_preview(
        self,
        original_path: Path,
        document_id: str | None = None,
        document_name: str | None = None,
        page: int | None = None,
        image_index: int | None = None,
        image_type: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        *,
        overwrite: bool = False,
    ) -> PreviewResult | None:
        original_path = Path(original_path)
        if (
            not original_path.is_file()
            or original_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
        ):
            return None

        parsed_page, parsed_image_index = self._asset_position(original_path.name)
        page = page or parsed_page
        image_index = image_index or parsed_image_index

        try:
            original_sha256 = self._file_sha256(original_path)
            override = self._override_for(original_path.name)
            pdf_figure = self._resolve_pdf_figure(
                original_path,
                original_sha256=original_sha256,
                document_id=document_id,
                document_name=document_name,
                page=page,
                image_index=image_index,
                bbox=bbox,
                override=override,
            )
            if pdf_figure is not None:
                return self._pdf_crop_preview(
                    original_path,
                    original_sha256,
                    pdf_figure,
                    override,
                    image_type=image_type,
                    overwrite=overwrite,
                )
            return self._extracted_image_preview(
                original_path,
                original_sha256,
                override,
                document_id=document_id,
                page=page,
                image_type=image_type,
                overwrite=overwrite,
            )
        except Exception:
            self.logger.exception(
                "Failed to create display preview for %s",
                original_path.name,
            )
            return None

    def preferred_source_type(
        self,
        original_path: Path,
        *,
        document_id: str | None = None,
        document_name: str | None = None,
        page: int | None = None,
        image_index: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> str | None:
        original_path = Path(original_path)
        if (
            not original_path.is_file()
            or original_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
        ):
            return None
        parsed_page, parsed_image_index = self._asset_position(original_path.name)
        page = page or parsed_page
        image_index = image_index or parsed_image_index
        try:
            original_sha256 = self._file_sha256(original_path)
            override = self._override_for(original_path.name)
            pdf_figure = self._resolve_pdf_figure(
                original_path,
                original_sha256=original_sha256,
                document_id=document_id,
                document_name=document_name,
                page=page,
                image_index=image_index,
                bbox=bbox,
                override=override,
            )
            return "pdf_page_crop" if pdf_figure else "extracted_image"
        except Exception:
            return "extracted_image"

    def _pdf_crop_preview(
        self,
        original_path: Path,
        original_sha256: str,
        pdf_figure: _PdfFigure,
        override: dict,
        *,
        image_type: str | None,
        overwrite: bool,
    ) -> PreviewResult | None:
        cache_payload = self._base_cache_payload(
            original_path,
            original_sha256,
            source_type="pdf_page_crop",
            image_type=image_type,
        )
        cache_payload.update(
            {
                "pdf_sha256": pdf_figure.pdf_sha256,
                "pdf_size": pdf_figure.pdf_path.stat().st_size,
                "pdf_mtime_ns": pdf_figure.pdf_path.stat().st_mtime_ns,
                "page": pdf_figure.page_number,
                "xref": pdf_figure.xref,
                "bbox": pdf_figure.bbox,
                "render_dpi": self.render_dpi,
                "crop_padding": self.crop_padding,
                "crop_rotation": pdf_figure.crop_rotation,
                "pdf_crop_rotation_override": override.get("pdf_crop_rotation"),
                "fallback_rotation_override": override.get("rotation"),
                "fallback_enhance": override.get("enhance", True),
            }
        )
        preview_path = self._preview_path(original_path, cache_payload)
        if preview_path.is_file() and not overwrite:
            return PreviewResult(
                preview_path,
                "pdf_page_crop",
                pdf_figure.crop_rotation,
                False,
            )

        try:
            with _preview_lock:
                if preview_path.is_file() and not overwrite:
                    return PreviewResult(
                        preview_path,
                        "pdf_page_crop",
                        pdf_figure.crop_rotation,
                        False,
                    )
                self.preview_dir.mkdir(parents=True, exist_ok=True)
                self._write_pdf_crop(pdf_figure, preview_path)
            return PreviewResult(
                preview_path,
                "pdf_page_crop",
                pdf_figure.crop_rotation,
                False,
            )
        except Exception:
            self.logger.exception(
                "PDF crop preview failed for %s; using extracted image",
                original_path.name,
            )
            return self._extracted_image_preview(
                original_path,
                original_sha256,
                override,
                document_id=None,
                page=pdf_figure.page_number,
                image_type=image_type,
                overwrite=overwrite,
            )

    def _extracted_image_preview(
        self,
        original_path: Path,
        original_sha256: str,
        override: dict,
        *,
        document_id: str | None,
        page: int | None,
        image_type: str | None,
        overwrite: bool,
    ) -> PreviewResult | None:
        rotation = self._manual_rotation(override)
        enhance = bool(override.get("enhance", True))
        if rotation is None:
            rotation = self._suggest_rotation(original_path, image_type)
        cache_payload = self._base_cache_payload(
            original_path,
            original_sha256,
            source_type="extracted_image",
            image_type=image_type,
        )
        cache_payload.update(
            {
                "document_id": document_id,
                "page": page,
                "rotation_clockwise": rotation,
                "enhance": enhance,
            }
        )
        preview_path = self._preview_path(original_path, cache_payload)
        if preview_path.is_file() and not overwrite:
            return PreviewResult(preview_path, "extracted_image", rotation, enhance)

        with _preview_lock:
            if preview_path.is_file() and not overwrite:
                return PreviewResult(
                    preview_path,
                    "extracted_image",
                    rotation,
                    enhance,
                )
            self.preview_dir.mkdir(parents=True, exist_ok=True)
            self._write_extracted_preview(
                original_path,
                preview_path,
                rotation_clockwise=rotation,
                enhance=enhance,
            )
        return PreviewResult(preview_path, "extracted_image", rotation, enhance)

    def _resolve_pdf_figure(
        self,
        original_path: Path,
        *,
        original_sha256: str,
        document_id: str | None,
        document_name: str | None,
        page: int | None,
        image_index: int | None,
        bbox: tuple[float, float, float, float] | None,
        override: dict,
    ) -> _PdfFigure | None:
        if not page or page < 1 or not image_index or image_index < 1:
            return None
        resolved_pdf = self._resolve_pdf(document_id, document_name)
        if resolved_pdf is None:
            return None
        pdf_path, pdf_sha256 = resolved_pdf

        try:
            import fitz

            with fitz.open(pdf_path) as document:
                page_index = page - 1
                if page_index >= len(document):
                    return None
                pdf_page = document[page_index]
                images = list(pdf_page.get_images(full=True))
                if image_index > len(images):
                    return None
                xref = int(images[image_index - 1][0])
                extracted = document.extract_image(xref).get("image")
                if (
                    not extracted
                    or hashlib.sha256(extracted).hexdigest() != original_sha256
                ):
                    return None

                rect_transforms = list(
                    pdf_page.get_image_rects(xref, transform=True)
                )
                if bbox is not None:
                    rect = fitz.Rect(*bbox)
                    transform = rect_transforms[0][1] if rect_transforms else fitz.Matrix()
                elif rect_transforms:
                    rect, transform = max(
                        rect_transforms,
                        key=lambda item: float(item[0].width * item[0].height),
                    )
                else:
                    return None

                visible = rect & pdf_page.rect
                if visible.is_empty or visible.width <= 2 or visible.height <= 2:
                    return None
                rotation = self._crop_rotation(
                    original_path,
                    visible.width,
                    visible.height,
                    tuple(transform),
                    override,
                )
                return _PdfFigure(
                    pdf_path=pdf_path,
                    pdf_sha256=pdf_sha256,
                    page_index=page_index,
                    page_number=page,
                    xref=xref,
                    bbox=tuple(float(value) for value in visible),
                    transform=tuple(float(value) for value in transform),
                    crop_rotation=rotation,
                )
        except Exception as exc:
            self.logger.warning(
                "Could not resolve PDF crop for %s: %s",
                original_path.name,
                exc,
            )
            return None

    def _resolve_pdf(
        self,
        document_id: str | None,
        document_name: str | None,
    ) -> tuple[Path, str] | None:
        cache_key = (document_id, document_name)
        if cache_key in self._pdf_resolution_cache:
            return self._pdf_resolution_cache[cache_key]
        if not self.raw_dir.is_dir():
            self._pdf_resolution_cache[cache_key] = None
            return None
        raw_root = self.raw_dir.resolve()
        candidates: list[Path] = []

        if document_id:
            metadata_path = self.metadata_dir / f"{document_id}.json"
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    filename = Path(str(metadata.get("filename") or "")).name
                    if filename:
                        candidates.append(raw_root / f"{document_id}_{filename}")
                        candidates.append(raw_root / filename)
                except (OSError, json.JSONDecodeError):
                    pass
            candidates.extend(raw_root.glob(f"{document_id}_*.pdf"))
        elif document_name:
            safe_name = Path(document_name).name
            candidates.extend(raw_root.glob(f"*_{safe_name}"))
            candidates.append(raw_root / safe_name)

        unique: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(raw_root)
            except (OSError, ValueError):
                continue
            if (
                resolved not in seen
                and resolved.is_file()
                and resolved.suffix.lower() == ".pdf"
            ):
                seen.add(resolved)
                unique.append(resolved)
        if len(unique) != 1:
            self._pdf_resolution_cache[cache_key] = None
            return None

        pdf_path = unique[0]
        pdf_sha256 = self._file_sha256(pdf_path)
        if document_id and len(document_id) == 64 and pdf_sha256.lower() != document_id.lower():
            self._pdf_resolution_cache[cache_key] = None
            return None
        result = (pdf_path, pdf_sha256)
        self._pdf_resolution_cache[cache_key] = result
        return result

    def _write_pdf_crop(
        self,
        pdf_figure: _PdfFigure,
        preview_path: Path,
    ) -> None:
        import fitz

        with fitz.open(pdf_figure.pdf_path) as document:
            page = document[pdf_figure.page_index]
            rect = fitz.Rect(*pdf_figure.bbox)
            pad_x = rect.width * self.crop_padding
            pad_y = rect.height * self.crop_padding
            clip = fitz.Rect(
                max(page.rect.x0, rect.x0 - pad_x),
                max(page.rect.y0, rect.y0 - pad_y),
                min(page.rect.x1, rect.x1 + pad_x),
                min(page.rect.y1, rect.y1 + pad_y),
            )
            scale = self.render_dpi / 72.0
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                clip=clip,
                alpha=False,
            )
            if pixmap.width < 64 or pixmap.height < 64:
                raise ValueError("Rendered PDF crop is too small")
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")

        if pdf_figure.crop_rotation:
            image = image.rotate(-pdf_figure.crop_rotation, expand=True)
        if ImageStat.Stat(image.convert("L")).extrema[0][1] <= 3:
            raise ValueError("Rendered PDF crop is blank")
        self._atomic_save_png(image, preview_path)

    def _write_extracted_preview(
        self,
        original_path: Path,
        preview_path: Path,
        *,
        rotation_clockwise: int,
        enhance: bool,
    ) -> None:
        with Image.open(original_path) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")

        if rotation_clockwise:
            image = image.rotate(-rotation_clockwise, expand=True)
        if enhance:
            image = self._enhance_if_dark(image)
        self._atomic_save_png(image, preview_path)

    def _atomic_save_png(self, image: Image.Image, preview_path: Path) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=preview_path.parent,
                prefix=f".{preview_path.stem}.",
                suffix=".tmp.png",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            image.save(temporary_path, format="PNG", optimize=True)
            os.replace(temporary_path, preview_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    self.logger.warning(
                        "Could not remove temporary preview %s",
                        temporary_path,
                    )

    def _crop_rotation(
        self,
        original_path: Path,
        crop_width: float,
        crop_height: float,
        transform: tuple[float, float, float, float, float, float],
        override: dict,
    ) -> int:
        explicit_pdf_rotation = self._rotation_value(
            override.get("pdf_crop_rotation")
        )
        if explicit_pdf_rotation is not None:
            return explicit_pdf_rotation

        with Image.open(original_path) as source:
            oriented = ImageOps.exif_transpose(source)
            expected_width, expected_height = oriented.size
        fallback_rotation = self._manual_rotation(override) or 0
        if fallback_rotation in {90, 270}:
            expected_width, expected_height = expected_height, expected_width

        expected_landscape = expected_width >= expected_height
        crop_landscape = crop_width >= crop_height
        if expected_landscape == crop_landscape:
            return 0

        _, b, c, _, _, _ = transform
        if b < 0 < c:
            return 90
        if c < 0 < b:
            return 270
        return 90

    def _base_cache_payload(
        self,
        original_path: Path,
        original_sha256: str,
        *,
        source_type: str,
        image_type: str | None,
    ) -> dict:
        stat = original_path.stat()
        return {
            "algorithm": self.algorithm_version,
            "source_type": source_type,
            "filename": original_path.name,
            "extracted_size": stat.st_size,
            "extracted_mtime_ns": stat.st_mtime_ns,
            "extracted_sha256": original_sha256,
            "image_type": image_type,
        }

    def _preview_path(self, original_path: Path, cache_payload: dict) -> Path:
        cache_key = hashlib.sha256(
            json.dumps(
                cache_payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        source_type = str(cache_payload.get("source_type") or "preview")
        return (
            self.preview_dir
            / f"{original_path.stem}_display_{source_type}_{cache_key}.png"
        )

    def _asset_position(self, filename: str) -> tuple[int | None, int | None]:
        match = _ASSET_POSITION_RE.search(Path(filename).stem)
        if not match:
            return None, None
        return int(match.group(1)), int(match.group(2))

    def _enhance_if_dark(self, image: Image.Image) -> Image.Image:
        grayscale = image.convert("L")
        histogram = grayscale.histogram()
        pixel_count = max(1, sum(histogram))
        dark_ratio = sum(histogram[:33]) / pixel_count
        gray_mean = ImageStat.Stat(grayscale).mean[0]

        if gray_mean >= 70 and dark_ratio < 0.65:
            return image

        channels = []
        for channel in image.split():
            high = self._nonblack_percentile(channel.histogram(), 0.995)
            if high <= 8:
                channels.append(channel)
                continue
            scale = 235.0 / high
            channels.append(
                channel.point(
                    [max(0, min(255, round(value * scale))) for value in range(256)]
                )
            )

        enhanced = Image.merge("RGB", channels)
        enhanced = ImageOps.autocontrast(enhanced, cutoff=0)
        if gray_mean < 15 and dark_ratio > 0.85:
            gamma = 0.60
            gamma_lut = [
                round(255 * ((value / 255) ** gamma))
                for value in range(256)
            ]
            enhanced = enhanced.point(gamma_lut * 3)
        return ImageEnhance.Contrast(enhanced).enhance(1.05)

    def _nonblack_percentile(self, histogram: list[int], percentile: float) -> int:
        nonblack_count = sum(histogram[3:])
        if nonblack_count <= 0:
            return 0
        target = nonblack_count * percentile
        seen = 0
        for value in range(3, 256):
            seen += histogram[value]
            if seen >= target:
                return value
        return 255

    def _suggest_rotation(self, path: Path, image_type: str | None) -> int:
        normalized_type = str(image_type or "").strip().lower()
        if normalized_type not in {"graph", "chart", "plot", "diagram", "schematic"}:
            return 0

        with Image.open(path) as source:
            exif_orientation = int(source.getexif().get(274, 1) or 1)
            if exif_orientation != 1:
                return 0
            image = source.convert("L")

        scores = {
            rotation: self._orientation_score(
                image.rotate(-rotation, expand=True) if rotation else image
            )
            for rotation in (0, 90, 180, 270)
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_rotation, best_score = ranked[0]
        second_score = ranked[1][1]
        if best_rotation and best_score - second_score >= 0.18:
            return best_rotation
        return 0

    def _orientation_score(self, image: Image.Image) -> float:
        thumbnail = image.copy()
        thumbnail.thumbnail((640, 640))
        edges = thumbnail.filter(ImageFilter.FIND_EDGES)
        thresholded = edges.point(lambda value: 255 if value >= 48 else 0)
        width, height = thresholded.size
        pixels = thresholded.load()
        horizontal = sum(
            1
            for y in range(height)
            if sum(1 for x in range(width) if pixels[x, y]) >= width * 0.35
        )
        vertical = sum(
            1
            for x in range(width)
            if sum(1 for y in range(height) if pixels[x, y]) >= height * 0.35
        )
        landscape_bonus = 0.08 if width >= height else 0.0
        return horizontal / max(1, height) + vertical / max(1, width) + landscape_bonus

    def _override_for(self, filename: str) -> dict:
        if not self.overrides_path.is_file():
            return {}
        try:
            payload = json.loads(self.overrides_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            self.logger.warning(
                "Ignoring invalid figure display overrides: %s",
                self.overrides_path,
            )
            return {}
        value = payload.get(filename)
        return value if isinstance(value, dict) else {}

    def _manual_rotation(self, override: dict) -> int | None:
        if "rotation" not in override:
            return None
        return self._rotation_value(override.get("rotation"))

    def _rotation_value(self, value: object) -> int | None:
        try:
            rotation = int(value)
        except (TypeError, ValueError):
            return None
        return rotation if rotation in {0, 90, 180, 270} else None

    def _file_sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
