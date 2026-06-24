from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageStat, UnidentifiedImageError

from app.core.config import Settings
from app.models.schemas import DocumentRecord
from app.services.chunking import chunk_pages

if TYPE_CHECKING:
    from fastapi import UploadFile

    from app.services.ollama import OllamaClient


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".ppt", ".pptx"}
MAX_FIGURE_ANALYSES = 20


class DocumentProcessor:
    def __init__(self, settings: Settings, ollama: OllamaClient):
        self.settings = settings
        self.ollama = ollama

    async def save_upload(self, upload: UploadFile) -> tuple[Path, str]:
        suffix = Path(upload.filename or "uploaded").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        content = await upload.read()
        digest = hashlib.sha256(content).hexdigest()
        safe_name = Path(upload.filename or f"{digest}{suffix}").name
        target = self.settings.raw_dir / f"{digest}_{safe_name}"

        if not target.exists():
            target.write_bytes(content)

        return target, digest

    def metadata_path(self, digest: str) -> Path:
        return self.settings.metadata_dir / f"{digest}.json"

    def load_existing(self, digest: str) -> DocumentRecord | None:
        path = self.metadata_path(digest)
        if not path.exists():
            return None
        return DocumentRecord(**json.loads(path.read_text(encoding="utf-8")))

    def persist_record(self, record: DocumentRecord) -> None:
        self.metadata_path(record.sha256).write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def classify_image_candidate(image_path: Path) -> dict[str, object]:
        """Classify obvious non-figure images before spending a vision call."""
        try:
            file_size = image_path.stat().st_size

            with Image.open(image_path) as image:
                width, height = image.size
                rgb_image = image.convert("RGB").resize((64, 64))
                stat = ImageStat.Stat(rgb_image)

            area = width * height
            aspect_ratio = width / max(height, 1)
            contrast = sum(stat.stddev) / max(len(stat.stddev), 1)

            result: dict[str, object] = {
                "image_type": "unknown",
                "confidence": 0.6,
                "should_analyze": True,
                "width": width,
                "height": height,
                "file_size": file_size,
                "contrast": round(float(contrast), 2),
                "reason": "",
            }

            if file_size < 20_000 or width < 350 or height < 250 or area < 150_000:
                result.update(
                    {
                        "image_type": "decorative",
                        "confidence": 0.1,
                        "should_analyze": False,
                        "reason": "too small or low resolution",
                    }
                )
                return result

            if contrast < 8:
                result.update(
                    {
                        "image_type": "decorative",
                        "confidence": 0.1,
                        "should_analyze": False,
                        "reason": "nearly monochrome",
                    }
                )
                return result

            if aspect_ratio > 8 or aspect_ratio < 0.125:
                result.update(
                    {
                        "image_type": "decorative",
                        "confidence": 0.2,
                        "should_analyze": False,
                        "reason": "extreme aspect ratio",
                    }
                )
                return result

            if height < 320 and width < 900:
                result.update(
                    {
                        "image_type": "logo",
                        "confidence": 0.35,
                        "should_analyze": False,
                        "reason": "small banner/logo-like image",
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

    @staticmethod
    def should_analyze_image(image_path: Path) -> bool:
        """Return True only for images large enough to contain a useful figure."""
        return bool(DocumentProcessor.classify_image_candidate(image_path).get("should_analyze"))

    def build_figure_note(
        self,
        *,
        document_name: str,
        page_number: int,
        image_index: int,
        image_path: Path,
        note: str,
        page_text: str,
        candidate: dict[str, object],
    ) -> str:
        metadata = self._classify_figure_note(note, candidate)
        related_text = re.sub(r"\s+", " ", page_text).strip()[:800]
        return "\n".join(
            [
                "[Figure Note Metadata]",
                f"Document: {document_name}",
                f"Page: {page_number}",
                f"Image Path: {image_path}",
                f"image_path: {image_path}",
                f"Image Number: {image_index}",
                f"Image Type: {metadata['image_type']}",
                f"image_type: {metadata['image_type']}",
                f"Confidence: {metadata['confidence']}",
                f"confidence: {metadata['confidence']}",
                f"X Axis Confirmed: {metadata['x_axis_confirmed']}",
                f"x_axis_verified: {metadata['x_axis_confirmed']}",
                f"Y Axis Confirmed: {metadata['y_axis_confirmed']}",
                f"y_axis_verified: {metadata['y_axis_confirmed']}",
                f"Units Confirmed: {metadata['units_confirmed']}",
                f"units_verified: {metadata['units_confirmed']}",
                f"Analysis Model: {self.settings.vision_model}",
                f"Created At: {datetime.now(timezone.utc).isoformat()}",
                f"Related Page Text: {related_text}",
                "",
                "[Analysis]",
                note,
            ]
        )

    def _classify_figure_note(self, note: str, candidate: dict[str, object]) -> dict[str, object]:
        lower = note.lower()
        image_type = str(candidate.get("image_type") or "unknown")
        if image_type in {"logo", "decorative"}:
            base_confidence = float(candidate.get("confidence") or 0.0)
        else:
            base_confidence = 0.55

        if any(term in lower for term in ["x-axis", "y-axis", "axis", "legend", "trend", "plot", "graph", "x축", "y축"]):
            image_type = "graph"
            base_confidence = 0.75
        elif any(term in lower for term in ["chart", "bar chart", "line chart"]):
            image_type = "chart"
            base_confidence = 0.7
        elif "table" in lower or "표" in note:
            image_type = "table"
            base_confidence = 0.65
        elif any(term in lower for term in ["equation", "formula", "수식"]):
            image_type = "equation"
            base_confidence = 0.65
        elif any(term in lower for term in ["logo", "decorative", "장식"]):
            image_type = "logo"
            base_confidence = 0.2
        elif any(term in lower for term in ["photograph", "photo", "image shows"]):
            image_type = "photograph"
            base_confidence = 0.45

        uncertainty_penalty = lower.count("확인할 수 없음") * 0.08 + lower.count("cannot determine") * 0.08
        x_axis = any(term in lower for term in ["x-axis", "x axis", "x축"])
        y_axis = any(term in lower for term in ["y-axis", "y axis", "y축"])
        units = any(term in lower for term in ["unit", "units", "단위", "psi", "ft", "ppg", "sg"])
        confidence = max(0.0, min(1.0, base_confidence - uncertainty_penalty))

        return {
            "image_type": image_type,
            "confidence": round(confidence, 2),
            "x_axis_confirmed": x_axis,
            "y_axis_confirmed": y_axis,
            "units_confirmed": units,
        }

    @staticmethod
    def _figure_note_confidence(note: str) -> float:
        match = re.search(r"^Confidence:\s*([0-9.]+)", note, flags=re.MULTILINE)
        if not match:
            return 0.5
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0

    async def extract(
        self,
        file_path: Path,
        digest: str,
        analyze_figures: bool = True,
    ) -> tuple[DocumentRecord, list[dict[str, object]]]:
        suffix = file_path.suffix.lower()

        # 실제 저장 파일명에서 SHA-256 접두사 제거
        display_filename = file_path.name
        digest_prefix = f"{digest}_"

        if display_filename.startswith(digest_prefix):
            display_filename = display_filename[len(digest_prefix):]

        if suffix == ".pdf":
            pages, figure_count = await self._extract_pdf(
                file_path,
                digest,
                analyze_figures,
            )
        elif suffix == ".txt":
            pages, figure_count = self._extract_text(file_path, digest), 0
        elif suffix in {".png", ".jpg", ".jpeg"}:
            pages, figure_count = await self._extract_image(file_path, digest), 1
        else:
            pages, figure_count = [
                {
                    "page": 1,
                    "text": (
                        f"Uploaded {suffix} document. "
                        "PPT text extraction is reserved for the Phase 2 parser."
                    ),
                }
            ], 0

        document_info = self.detect_document_info(pages, display_filename,)
        for page in pages:
            page["document_type"] = document_info.get("document_type", "")

        chunks = self.chunk_pages(pages, display_filename, digest)
        record = DocumentRecord(
            document_id=digest,
            filename=display_filename,
            sha256=digest,
            status="processed",
            pages=len(pages),
            chunks=len(chunks),
            figures=figure_count,
            title=document_info.get("title"),
            document_type=document_info.get("document_type"),
            contents_pages=document_info.get("contents_pages", []),
            title_pages=document_info.get("title_pages", []),
        )

        (self.settings.extracted_dir / f"{digest}.json").write_text(
            json.dumps(
                {"pages": pages, "chunks": chunks},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.persist_record(record)
        return record, chunks

    def _extract_text(
        self,
        file_path: Path,
        digest: str,
    ) -> list[dict[str, object]]:
        return [
            {
                "page": 1,
                "text": file_path.read_text(encoding="utf-8", errors="ignore"),
                "document_id": digest,
            }
        ]

    async def _extract_image(
        self,
        file_path: Path,
        digest: str,
    ) -> list[dict[str, object]]:
        target = self.settings.figures_dir / f"{digest}_{file_path.name}"
        if not target.exists():
            shutil.copyfile(file_path, target)

        note = await self.ollama.describe_image(target)
        if note.strip():
            candidate = self.classify_image_candidate(target)
            note = self.build_figure_note(
                document_name=file_path.name,
                page_number=1,
                image_index=1,
                image_path=target,
                note=note,
                page_text="",
                candidate=candidate,
            )
            note_path = self.settings.figure_notes_dir / f"{target.stem}.md"
            note_path.write_text(note, encoding="utf-8")

        return [
            {
                "page": 1,
                "text": f"Image analysis for {file_path.name}:\n{note}",
                "document_id": digest,
            }
        ]

    async def _extract_pdf(
        self,
        file_path: Path,
        digest: str,
        analyze_figures: bool,
    ) -> tuple[list[dict[str, object]], int]:
        import fitz
        import pdfplumber

        pages: list[dict[str, object]] = []
        figure_count = 0
        vision_calls = 0
        note_cache: dict[int, str] = {}
        plumber_pdf = None

        doc = fitz.open(file_path)

        # pdfplumber는 보조 추출기로만 사용합니다.
        # 일부 손상되거나 비표준인 PDF는 pdfminer 단계에서 예외가 발생할 수 있습니다.
        try:
            plumber_pdf = pdfplumber.open(file_path)
        except Exception as exc:
            print(
                f"[pdfplumber 비활성화] {file_path.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            plumber_pdf = None

        try:
            total_pages = len(doc)

            for index, page in enumerate(doc):
                page_number = index + 1

                if page_number == 1 or page_number % 25 == 0:
                    print(f"[PDF 텍스트 추출] {page_number}/{total_pages}페이지")

                # PyMuPDF를 기본 텍스트 추출기로 사용
                try:
                    text = page.get_text("text").strip()
                except Exception as exc:
                    print(
                        f"[PyMuPDF 텍스트 추출 실패] "
                        f"{page_number}페이지: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    text = ""

                # PyMuPDF 결과가 없을 때만 pdfplumber를 보조로 사용
                plumber_text = ""
                if (
                    not text
                    and plumber_pdf is not None
                    and index < len(plumber_pdf.pages)
                ):
                    try:
                        plumber_text = (
                            plumber_pdf.pages[index].extract_text() or ""
                        ).strip()
                    except Exception as exc:
                        print(
                            f"[pdfplumber 건너뜀] "
                            f"{page_number}페이지: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        plumber_text = ""

                combined = text or plumber_text
                figure_notes: list[str] = []

                try:
                    page_images = page.get_images(full=True)
                except Exception as exc:
                    print(
                        f"[페이지 이미지 목록 건너뜀] "
                        f"{page_number}페이지: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    page_images = []

                for image_index, image_info in enumerate(
                    page_images,
                    start=1,
                ):
                    xref = image_info[0]

                    try:
                        extracted_image = doc.extract_image(xref)
                        image_bytes = extracted_image.get("image")
                        if not image_bytes:
                            continue

                        extension = extracted_image.get("ext", "png")
                        figure_path = (
                            self.settings.figures_dir
                            / f"{digest}_p{page_number}_fig{image_index}.{extension}"
                        )

                        if not figure_path.exists():
                            figure_path.write_bytes(image_bytes)

                        figure_count += 1

                    except Exception as exc:
                        print(
                            f"[PDF 이미지 추출 건너뜀] "
                            f"{page_number}페이지 Figure {image_index}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue

                    if not analyze_figures:
                        continue

                    note = note_cache.get(xref, "")
                    candidate = self.classify_image_candidate(figure_path)
                    note_path = (
                        self.settings.figure_notes_dir
                        / f"{figure_path.stem}.md"
                    )

                    if not note and note_path.exists():
                        try:
                            note = note_path.read_text(
                                encoding="utf-8"
                            ).strip()
                        except OSError:
                            note = ""

                    if not note:
                        if vision_calls >= MAX_FIGURE_ANALYSES:
                            continue

                        if not candidate.get("should_analyze"):
                            continue

                        vision_calls += 1
                        print(
                            f"[PDF 이미지 분석] {vision_calls}/"
                            f"{MAX_FIGURE_ANALYSES}: {figure_path.name}"
                        )

                        try:
                            raw_note = (
                                await self.ollama.describe_image(
                                    figure_path
                                )
                            ).strip()
                        except Exception as exc:
                            print(
                                f"[PDF 이미지 분석 건너뜀] "
                                f"{figure_path.name}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            raw_note = ""

                        # 실패한 빈 결과는 저장하지 않음
                        if raw_note:
                            note = self.build_figure_note(
                                document_name=file_path.name,
                                page_number=page_number,
                                image_index=image_index,
                                image_path=figure_path,
                                note=raw_note,
                                page_text=combined,
                                candidate=candidate,
                            )
                            note_path.write_text(
                                note,
                                encoding="utf-8",
                            )

                    note_cache[xref] = note

                    if note and self._figure_note_confidence(note) >= self.settings.figure_note_min_confidence:
                        figure_notes.append(
                            f"Figure {image_index}: {note}"
                        )

                if figure_notes:
                    combined += (
                        "\n\n[Extracted figure notes]\n"
                        + "\n".join(figure_notes)
                    )

                pages.append(
                    {
                        "page": page_number,
                        "text": combined,
                        "document_id": digest,
                    }
                )

        finally:
            if plumber_pdf is not None:
                try:
                    plumber_pdf.close()
                except Exception:
                    pass

            doc.close()

        print(
            f"[PDF 처리 완료] 추출 이미지 {figure_count}개, "
            f"새 Vision 호출 {vision_calls}개"
        )
        return pages, figure_count

    def detect_document_info(
        self,
        pages: list[dict[str, object]],
        filename: str,
    ) -> dict[str, object]:
        contents_pages: list[int] = []
        title_pages: list[int] = []
        title = Path(filename).stem
        document_type = "unknown"
        contents_terms = [
            "master author index",
            "master subject index",
            "spe symbols standard",
            "si metric conversion factors",
            "alphabetical list of units",
            "tables of recommended si units",
            "contents",
            "table of contents",
        ]

        for page in pages:
            page_num = int(page.get("page", 0))
            text = str(page.get("text", ""))
            lower = text.lower()
            hits = sum(1 for term in contents_terms if term in lower)
            is_contents = (
                ("contents" in lower and hits >= 2)
                or hits >= 4
            )

            if is_contents:
                page["is_contents"] = True
                page["heading"] = "Contents"
                contents_pages.append(page_num)

            if page_num <= 3:
                page["is_title_page"] = True
                title_pages.append(page_num)
                lines = [
                    line.strip()
                    for line in text.splitlines()
                    if len(line.strip()) > 5
                ]

                for line in lines[:8]:
                    if (
                        "petroleum engineering handbook" in line.lower()
                        or "indexes and standards" in line.lower()
                    ):
                        title = line
                        break

            if (
                "master author index" in lower
                or "master subject index" in lower
                or "spe symbols standard" in lower
            ):
                document_type = "indexes_and_standards"

        return {
            "title": title,
            "document_type": document_type,
            "contents_pages": sorted(set(contents_pages)),
            "title_pages": sorted(set(title_pages)),
        }

    def chunk_pages(
        self,
        pages: list[dict[str, object]],
        filename: str,
        digest: str,
    ) -> list[dict[str, object]]:
        return chunk_pages(
            pages,
            filename,
            digest,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
