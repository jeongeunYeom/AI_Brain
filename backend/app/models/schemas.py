from pydantic import BaseModel, Field
from typing import Any


class Source(BaseModel):
    document: str
    page: int | None = None
    chunk_id: str
    score: float | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    excerpt: str
    preview: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    query_type: str | None = None


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    sha256: str
    status: str
    pages: int = 0
    chunks: int = 0
    figures: int = 0
    title: str | None = None
    document_type: str | None = None
    contents_pages: list[int] = Field(default_factory=list)
    title_pages: list[int] = Field(default_factory=list)


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
