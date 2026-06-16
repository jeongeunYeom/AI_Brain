from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.models.schemas import DocumentRecord
from app.services.chunking import chunk_pages

if TYPE_CHECKING:
    from fastapi import UploadFile
    from app.services.ollama import OllamaClient

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".ppt", ".pptx"}


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
        self.metadata_path(record.sha256).write_text(record.model_dump_json(indent=2), encoding="utf-8")

    async def extract(self, file_path: Path, digest: str, analyze_figures: bool = True) -> tuple[DocumentRecord, list[dict[str, object]]]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            pages, figure_count = await self._extract_pdf(file_path, digest, analyze_figures)
        elif suffix == ".txt":
            pages, figure_count = self._extract_text(file_path, digest), 0
        elif suffix in {".png", ".jpg", ".jpeg"}:
            pages, figure_count = await self._extract_image(file_path, digest), 1
        else:
            pages, figure_count = [{"page": 1, "text": f"Uploaded {suffix} document. PPT text extraction is reserved for the Phase 2 parser."}], 0

        chunks = self.chunk_pages(pages, file_path.name, digest)
        record = DocumentRecord(
            document_id=digest,
            filename=file_path.name,
            sha256=digest,
            status="processed",
            pages=len(pages),
            chunks=len(chunks),
            figures=figure_count,
        )
        (self.settings.extracted_dir / f"{digest}.json").write_text(
            json.dumps({"pages": pages, "chunks": chunks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.persist_record(record)
        return record, chunks

    def _extract_text(self, file_path: Path, digest: str) -> list[dict[str, object]]:
        return [{"page": 1, "text": file_path.read_text(encoding="utf-8", errors="ignore"), "document_id": digest}]

    async def _extract_image(self, file_path: Path, digest: str) -> list[dict[str, object]]:
        target = self.settings.figures_dir / f"{digest}_{file_path.name}"
        if not target.exists():
            shutil.copyfile(file_path, target)
        note = await self.ollama.describe_image(target)
        note_path = self.settings.figure_notes_dir / f"{target.stem}.md"
        note_path.write_text(note, encoding="utf-8")
        return [{"page": 1, "text": f"Image analysis for {file_path.name}:\n{note}", "document_id": digest}]

    async def _extract_pdf(self, file_path: Path, digest: str, analyze_figures: bool) -> tuple[list[dict[str, object]], int]:
        pages: list[dict[str, object]] = []
        figure_count = 0
        import fitz
        import pdfplumber

        doc = fitz.open(file_path)
        plumber_pdf = pdfplumber.open(file_path)
        try:
            for index, page in enumerate(doc):
                page_number = index + 1
                text = page.get_text("text").strip()
                plumber_text = (plumber_pdf.pages[index].extract_text() or "").strip() if index < len(plumber_pdf.pages) else ""
                combined = "\n".join(part for part in [text, plumber_text] if part)
                figure_notes: list[str] = []
                for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                    xref = image_info[0]
                    image = doc.extract_image(xref)
                    extension = image.get("ext", "png")
                    figure_path = self.settings.figures_dir / f"{digest}_p{page_number}_fig{image_index}.{extension}"
                    if not figure_path.exists():
                        figure_path.write_bytes(image["image"])
                    figure_count += 1
                    if analyze_figures:
                        note_path = self.settings.figure_notes_dir / f"{figure_path.stem}.md"
                        if note_path.exists():
                            note = note_path.read_text(encoding="utf-8")
                        else:
                            note = await self.ollama.describe_image(figure_path)
                            note_path.write_text(note, encoding="utf-8")
                        figure_notes.append(f"Figure {image_index}: {note}")
                if figure_notes:
                    combined = combined + "\n\n[Extracted figure notes]\n" + "\n".join(figure_notes)
                pages.append({"page": page_number, "text": combined, "document_id": digest})
        finally:
            plumber_pdf.close()
            doc.close()
        return pages, figure_count

    def chunk_pages(self, pages: list[dict[str, object]], filename: str, digest: str) -> list[dict[str, object]]:
        return chunk_pages(pages, filename, digest, self.settings.chunk_size, self.settings.chunk_overlap)
