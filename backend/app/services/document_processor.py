from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, UnidentifiedImageError

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
    def should_analyze_image(image_path: Path) -> bool:
        """Return True only for images large enough to contain a useful figure."""
        try:
            if image_path.stat().st_size < 20_000:
                return False

            with Image.open(image_path) as image:
                width, height = image.size

            if width < 350 or height < 250:
                return False

            if width * height < 150_000:
                return False

            return True

        except (OSError, ValueError, UnidentifiedImageError):
            return False

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

                        if not self.should_analyze_image(figure_path):
                            continue

                        vision_calls += 1
                        print(
                            f"[PDF 이미지 분석] {vision_calls}/"
                            f"{MAX_FIGURE_ANALYSES}: {figure_path.name}"
                        )

                        try:
                            note = (
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
                            note = ""

                        # 실패한 빈 결과는 저장하지 않음
                        if note:
                            note_path.write_text(
                                note,
                                encoding="utf-8",
                            )

                    note_cache[xref] = note

                    if note:
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
