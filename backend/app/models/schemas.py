from pydantic import BaseModel, Field
from typing import Any


class Source(BaseModel):
    document: str
    page: int | None = None
    chunk_id: str
    score: float | None = None
    excerpt: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    sha256: str
    status: str
    pages: int = 0
    chunks: int = 0
    figures: int = 0


class UploadResponse(BaseModel):
    document: DocumentRecord
    skipped: bool


class VisionResponse(BaseModel):
    filename: str
    analysis: str


class PlotRequest(BaseModel):
    title: str
    x: list[float | int | str]
    y: list[float | int]
    x_label: str = "x"
    y_label: str = "y"
    chart_type: str = "line"


class PlotResponse(BaseModel):
    figure: dict[str, Any]
