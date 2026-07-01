import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageStat

from app import main
from app.services.figure_preview import FigurePreviewService


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_service(tmp_path: Path, *, version: str = "test-v1") -> FigurePreviewService:
    return FigurePreviewService(
        tmp_path / "previews",
        tmp_path / "overrides.json",
        algorithm_version=version,
    )


def make_pdf_fixture(
    tmp_path: Path,
    *,
    page_rotation: int = 0,
    portrait_source: bool = False,
    image_rotation: int = 0,
    at_page_edge: bool = False,
):
    import fitz

    raw_dir = tmp_path / "raw"
    metadata_dir = tmp_path / "metadata"
    figures_dir = tmp_path / "figures"
    raw_dir.mkdir()
    metadata_dir.mkdir()
    figures_dir.mkdir()

    source_size = (80, 140) if portrait_source else (160, 100)
    source = Image.new("RGB", source_size, "black")
    source_bytes = io.BytesIO()
    source.save(source_bytes, format="PNG")

    document = fitz.open()
    page = document.new_page(width=320, height=240)
    rect = fitz.Rect(0, 0, 230, 140) if at_page_edge else fitz.Rect(40, 45, 280, 190)
    page.insert_image(rect, stream=source_bytes.getvalue(), rotate=image_rotation)
    page.draw_rect(rect, color=(0, 0, 0), fill=(1, 1, 1), overlay=True)
    page.draw_line(rect.tl + (12, rect.height - 18), rect.br - (12, 18), color=(0, 0, 0), width=2, overlay=True)
    page.draw_line((rect.x0 + 14, rect.y1 - 18), (rect.x1 - 12, rect.y1 - 18), color=(0, 0, 0), width=1, overlay=True)
    page.insert_text((rect.x0 + 18, rect.y0 + 22), "Test Plot", fontsize=12, overlay=True)
    if page_rotation:
        page.set_rotation(page_rotation)

    temporary_pdf = tmp_path / "fixture.tmp.pdf"
    document.save(temporary_pdf)
    document.close()
    document_id = file_sha256(temporary_pdf)
    pdf_path = raw_dir / f"{document_id}_fixture.pdf"
    temporary_pdf.replace(pdf_path)
    (metadata_dir / f"{document_id}.json").write_text(
        json.dumps({"document_id": document_id, "filename": "fixture.pdf"}),
        encoding="utf-8",
    )

    with fitz.open(pdf_path) as saved:
        xref = saved[0].get_images(full=True)[0][0]
        extracted = saved.extract_image(xref)
    original = figures_dir / f"{document_id}_p1_fig1.{extracted['ext']}"
    original.write_bytes(extracted["image"])
    service = FigurePreviewService(
        tmp_path / "previews",
        tmp_path / "overrides.json",
        raw_dir=raw_dir,
        metadata_dir=metadata_dir,
        algorithm_version="pdf-test",
        render_dpi=180,
    )
    return service, original, pdf_path, document_id, tuple(rect)


def test_preview_preserves_original_and_enhances_dark_lines(tmp_path):
    original = tmp_path / "dark.png"
    image = Image.new("RGB", (320, 180), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.line((10, 150, 300, 20), fill=(20, 75, 75), width=2)
    draw.line((10, 130, 300, 80), fill=(70, 65, 15), width=2)
    image.save(original)
    before_hash = file_sha256(original)
    before_max = max(channel[1] for channel in image.getextrema())

    preview = make_service(tmp_path).get_or_create_preview(original)

    assert preview is not None
    assert file_sha256(original) == before_hash
    with Image.open(preview) as enhanced:
        assert max(channel[1] for channel in enhanced.getextrema()) > before_max
        assert ImageStat.Stat(enhanced.convert("L")).mean[0] < 100


def test_manual_rotation_90_is_clockwise(tmp_path):
    original = tmp_path / "direction.png"
    image = Image.new("RGB", (80, 40), "blue")
    ImageDraw.Draw(image).rectangle((0, 0, 39, 39), fill="red")
    image.save(original)
    (tmp_path / "overrides.json").write_text(
        json.dumps({original.name: {"rotation": 90, "enhance": False}}),
        encoding="utf-8",
    )

    preview = make_service(tmp_path).get_or_create_preview(original)

    assert preview is not None
    with Image.open(preview) as rotated:
        assert rotated.size == (40, 80)
        assert rotated.getpixel((20, 10))[0] > 200
        assert rotated.getpixel((20, 70))[2] > 200


def test_exif_orientation_is_applied(tmp_path):
    original = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (80, 40), "white")
    exif = image.getexif()
    exif[274] = 6
    image.save(original, exif=exif)

    preview = make_service(tmp_path).get_or_create_preview(original)

    assert preview is not None
    with Image.open(preview) as oriented:
        assert oriented.size == (40, 80)


def test_cache_reuse_and_invalidation(tmp_path):
    original = tmp_path / "cache.png"
    Image.new("RGB", (80, 40), "white").save(original)
    service = make_service(tmp_path)

    first = service.get_or_create_preview(original)
    second = service.get_or_create_preview(original)
    assert first == second

    (tmp_path / "overrides.json").write_text(
        json.dumps({original.name: {"rotation": 90}}),
        encoding="utf-8",
    )
    rotated = service.get_or_create_preview(original)
    versioned = make_service(tmp_path, version="test-v2").get_or_create_preview(original)

    assert rotated != first
    assert versioned not in {first, rotated}


def test_pdf_page_crop_is_preferred_and_preserves_white_composite(tmp_path):
    service, original, pdf_path, document_id, _ = make_pdf_fixture(tmp_path)
    pdf_hash = file_sha256(pdf_path)
    extracted_hash = file_sha256(original)

    result = service.get_or_create_preview(
        original,
        document_id=document_id,
        document_name="fixture.pdf",
        page=1,
        image_index=1,
        image_type="graph",
    )

    assert result is not None
    assert result.source_type == "pdf_page_crop"
    assert result.enhanced is False
    with Image.open(original) as extracted, Image.open(result) as preview:
        assert ImageStat.Stat(extracted.convert("L")).mean[0] < 5
        assert ImageStat.Stat(preview.convert("L")).mean[0] > 200
        assert preview.convert("RGB").getpixel((0, 0))[0] > 240
    assert file_sha256(pdf_path) == pdf_hash
    assert file_sha256(original) == extracted_hash


def test_pdf_crop_handles_rotated_page_and_clamped_padding(tmp_path):
    service, original, _, document_id, bbox = make_pdf_fixture(
        tmp_path,
        page_rotation=90,
        at_page_edge=True,
    )

    result = service.get_or_create_preview(
        original,
        document_id=document_id,
        page=1,
        image_index=1,
        bbox=bbox,
        image_type="graph",
    )

    assert result is not None
    assert result.source_type == "pdf_page_crop"
    with Image.open(result) as preview:
        assert preview.width >= 64
        assert preview.height >= 64
        assert ImageStat.Stat(preview.convert("L")).mean[0] > 200


def test_pdf_crop_does_not_reapply_extracted_rotation_override(tmp_path):
    service, original, _, document_id, _ = make_pdf_fixture(
        tmp_path,
        portrait_source=True,
        image_rotation=90,
    )
    service.overrides_path.write_text(
        json.dumps({original.name: {"rotation": 90, "enhance": True}}),
        encoding="utf-8",
    )

    pdf_result = service.get_or_create_preview(
        original,
        document_id=document_id,
        page=1,
        image_index=1,
        image_type="graph",
    )
    fallback_service = FigurePreviewService(
        tmp_path / "fallback-previews",
        service.overrides_path,
        raw_dir=tmp_path / "missing-raw",
        metadata_dir=tmp_path / "missing-metadata",
    )
    fallback_result = fallback_service.get_or_create_preview(
        original,
        document_id=document_id,
        page=1,
        image_index=1,
        image_type="graph",
    )

    assert pdf_result is not None
    assert pdf_result.source_type == "pdf_page_crop"
    assert pdf_result.rotation_applied == 0
    assert fallback_result is not None
    assert fallback_result.source_type == "extracted_image"
    assert fallback_result.rotation_applied == 90


def test_pdf_failure_falls_back_and_missing_extracted_returns_none(tmp_path):
    service, original, _, document_id, _ = make_pdf_fixture(tmp_path)

    fallback = service.get_or_create_preview(
        original,
        document_id="f" * 64,
        page=1,
        image_index=1,
        image_type="graph",
    )
    original.unlink()
    missing = service.get_or_create_preview(
        original,
        document_id=document_id,
        page=1,
        image_index=1,
        image_type="graph",
    )

    assert fallback is not None
    assert fallback.source_type == "extracted_image"
    assert missing is None


@pytest.mark.parametrize(
    "filename",
    ["../secret.png", r"..\secret.png", r"C:\secret.png", "secret.txt"],
)
def test_preview_endpoint_rejects_unsafe_paths(tmp_path, monkeypatch, filename):
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(data_dir=tmp_path, figures_dir=tmp_path / "figures"),
    )

    with pytest.raises(HTTPException) as caught:
        main.get_figure_preview(filename)

    assert caught.value.status_code in {400, 404}


def test_preview_endpoint_serves_only_preview_root(tmp_path, monkeypatch):
    preview_root = tmp_path / "figure_display_previews"
    preview_root.mkdir()
    preview = preview_root / "preview.png"
    Image.new("RGB", (8, 8), "white").save(preview)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(data_dir=tmp_path, figures_dir=tmp_path / "figures"),
    )

    response = main.get_figure_preview(preview.name)

    assert Path(response.path) == preview


def test_explicit_pdf_crop_rotation_override_is_applied(tmp_path):
    service, original, _, document_id, _ = make_pdf_fixture(tmp_path)
    service.overrides_path.write_text(
        json.dumps({
            original.name: {
                "pdf_crop_rotation": 90,
                "rotation": 0,
                "enhance": True,
            }
        }),
        encoding="utf-8",
    )

    result = service.get_or_create_preview(
        original,
        document_id=document_id,
        document_name="fixture.pdf",
        page=1,
        image_index=1,
        image_type="graph",
    )

    assert result is not None
    assert result.source_type == "pdf_page_crop"
    assert result.rotation_applied == 90



def test_explicit_pdf_crop_rotation_270_override_is_applied(tmp_path):
    service, original, _, document_id, _ = make_pdf_fixture(tmp_path)
    service.overrides_path.write_text(
        json.dumps({
            original.name: {
                "pdf_crop_rotation": 270,
                "rotation": 90,
                "enhance": True,
            }
        }),
        encoding="utf-8",
    )

    result = service.get_or_create_preview(
        original,
        document_id=document_id,
        document_name="fixture.pdf",
        page=1,
        image_index=1,
        image_type="graph",
    )

    assert result is not None
    assert result.source_type == "pdf_page_crop"
    assert result.rotation_applied == 270
