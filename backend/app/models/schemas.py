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



class FigureReference(BaseModel):
    document: str
    page: int | None = None
    title: str | None = None
    image_type: str | None = None
    filename: str
    url: str
    preview_url: str | None = None
    preview_source: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.preview_source is not None or not self.preview_url:
            return
        if "_display_pdf_page_crop_" in self.preview_url:
            self.preview_source = "pdf_page_crop"
        elif "_display_extracted_image_" in self.preview_url:
            self.preview_source = "extracted_image"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    query_type: str | None = None
    figures: list[FigureReference] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    sha256: str
    status: str
    pages: int = 0
    chunks: int = 0
    figures: int = 0
    figures_analyzed: int = 0
    figures_valid: int = 0
    figures_review_required: int = 0
    figures_failed: int = 0
    figures_ignored: int = 0
    figure_vision_calls: int = 0
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


class FigureReviewUpdateRequest(BaseModel):
    status: str | None = None
    title: str | None = None
    analysis: str | None = None
    x_axis: str | None = None
    x_axis_unit: str | None = None
    y_axis: str | None = None
    y_axis_unit: str | None = None
    trend_summary: str | None = None
    engineering_meaning: str | None = None
    series_descriptions: list[str] | None = None
    reference_lines: list[str] | None = None


class FigureRotationUpdateRequest(BaseModel):
    rotation: int | None = None
    pdf_crop_rotation: int | None = None
    enhance: bool = True
    regenerate: bool = True
