from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import asyncio
from PIL import Image, ImageDraw

from app.services.figure_analysis import FigureAnalysisService


@dataclass
class DummySettings:
    data_dir: Path
    vision_model: str = "qwen2.5vl:7b"
    figure_note_min_confidence: float = 0.5

    @property
    def figure_notes_dir(self) -> Path:
        path = self.data_dir / "figure_notes"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def figure_candidates_dir(self) -> Path:
        path = self.data_dir / "figure_candidates"
        path.mkdir(parents=True, exist_ok=True)
        return path


class FakeOllama:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def describe_image(
        self,
        image_path: Path,
        prompt: str | None = None,
        num_predict: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "image_path": image_path,
                "prompt": prompt,
                "num_predict": num_predict,
            }
        )
        if not self.responses:
            raise AssertionError("Unexpected Vision call")
        return self.responses.pop(0)


def make_graph_image(path: Path) -> None:
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.line((80, 520, 820, 520), fill="black", width=4)
    draw.line((80, 520, 80, 60), fill="black", width=4)
    for index in range(8):
        x = 130 + index * 80
        y = 470 - index * 45
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="black")
    image.save(path, format="PNG")


def test_valid_graph_is_saved_and_indexable(tmp_path: Path) -> None:
    image_path = tmp_path / "sample_p1_fig1.png"
    make_graph_image(image_path)

    ollama = FakeOllama(
        [
            "\n".join(
                [
                    "image_type: graph",
                    "confidence: 0.98",
                    "readable_labels: true",
                    "reason: axes and plotted markers are visible",
                ]
            ),
            "\n".join(
                [
                    "image_type: graph",
                    "confidence: 0.96",
                    "title: Pressure response",
                    "analysis: One marker series increases continuously across the graph.",
                    "x_axis: Time",
                    "x_axis_unit: hours",
                    "x_axis_scale: Linear",
                    "y_axis: Pressure",
                    "y_axis_unit: psi",
                    "y_axis_scale: Linear",
                    "series_count: 1",
                    "series_count_verified: true",
                    "series_descriptions: Circular markers forming a continuously rising series.",
                    "legend: unknown",
                    "reference_lines: unknown",
                    "plateau: unknown",
                    "peak: unknown",
                    "decline: unknown",
                    "slope_changes: unknown",
                    "trend: The series rises continuously as time increases.",
                    "engineering_meaning: unknown",
                ]
            ),
        ]
    )
    settings = DummySettings(tmp_path)
    service = FigureAnalysisService(settings, ollama)  # type: ignore[arg-type]

    result = asyncio.run(service.analyze_figure(
        document_name="sample.pdf",
        document_id="abc123",
        page_number=1,
        image_index=1,
        image_path=image_path,
        remaining_vision_calls=10,
    ))

    assert result.status == "valid"
    assert result.should_index is True
    assert result.vision_calls == 2
    assert result.note_path and result.note_path.is_file()
    assert result.candidate_path and result.candidate_path.is_file()
    assert result.candidate["apply_ready"] is True
    assert "x_axis: Time" in result.note_text
    assert "trend_verified: true" in result.note_text


def test_unresolved_multiseries_graph_is_held_for_review(tmp_path: Path) -> None:
    image_path = tmp_path / "dark_p1_fig1.png"
    make_graph_image(image_path)

    ollama = FakeOllama(
        [
            "\n".join(
                [
                    "image_type: graph",
                    "confidence: 0.80",
                    "readable_labels: false",
                    "reason: multiple marker clusters are visible but labels are unreadable",
                ]
            ),
            "\n".join(
                [
                    "image_type: graph",
                    "confidence: 0.65",
                    "title: unknown",
                    "analysis: Several marker clusters are visible.",
                    "x_axis: unknown",
                    "x_axis_unit: unknown",
                    "x_axis_scale: unknown",
                    "y_axis: unknown",
                    "y_axis_unit: unknown",
                    "y_axis_scale: unknown",
                    "series_count: unknown",
                    "series_count_verified: false",
                    "series_descriptions: Multiple marker clusters are visible but cannot be separated reliably.",
                    "legend: unknown",
                    "reference_lines: unknown",
                    "plateau: unknown",
                    "peak: unknown",
                    "decline: unknown",
                    "slope_changes: unknown",
                    "trend: unknown",
                    "engineering_meaning: unknown",
                ]
            ),
        ]
    )
    settings = DummySettings(tmp_path)
    service = FigureAnalysisService(settings, ollama)  # type: ignore[arg-type]

    result = asyncio.run(service.analyze_figure(
        document_name="sample.pdf",
        document_id="abc123",
        page_number=1,
        image_index=1,
        image_path=image_path,
        remaining_vision_calls=10,
    ))

    assert result.status == "review_required"
    assert result.should_index is False
    assert result.note_path is None
    assert result.candidate_path and result.candidate_path.name.endswith(
        ".review_required.json"
    )
    assert result.candidate["apply_ready"] is False
    assert result.candidate["manual_review_required"] is True


def test_dark_source_can_use_page_render_fallback(tmp_path: Path) -> None:
    source_path = tmp_path / "source_p1_fig1.png"
    fallback_path = tmp_path / "source_p1_fig1_page_render.png"
    make_graph_image(source_path)
    make_graph_image(fallback_path)

    ollama = FakeOllama(
        [
            "image_type: graph\nconfidence: 0.7\nreadable_labels: false\nreason: labels unreadable",
            "\n".join(
                [
                    "image_type: graph",
                    "confidence: 0.6",
                    "title: unknown",
                    "analysis: Several points are visible.",
                    "x_axis: unknown",
                    "x_axis_unit: unknown",
                    "x_axis_scale: unknown",
                    "y_axis: unknown",
                    "y_axis_unit: unknown",
                    "y_axis_scale: unknown",
                    "series_count: unknown",
                    "series_count_verified: false",
                    "series_descriptions: Points are visible but cannot be separated reliably.",
                    "legend: unknown",
                    "reference_lines: unknown",
                    "plateau: unknown",
                    "peak: unknown",
                    "decline: unknown",
                    "slope_changes: unknown",
                    "trend: unknown",
                    "engineering_meaning: unknown",
                ]
            ),
            "image_type: graph\nconfidence: 0.98\nreadable_labels: true\nreason: rendered labels and markers are clear",
            "\n".join(
                [
                    "image_type: graph",
                    "confidence: 0.97",
                    "title: Drawdown response",
                    "analysis: Two marker series are visible and are described separately.",
                    "x_axis: Time",
                    "x_axis_unit: hours",
                    "x_axis_scale: Logarithmic",
                    "y_axis: Delta P",
                    "y_axis_unit: psi",
                    "y_axis_scale: Logarithmic",
                    "series_count: 2",
                    "series_count_verified: true",
                    "series_descriptions: Upper square-marker series rises continuously; Lower circle-marker series rises and approaches a plateau.",
                    "legend: unknown",
                    "reference_lines: unknown",
                    "plateau: Lower circle-marker series approaches a plateau.",
                    "peak: unknown",
                    "decline: unknown",
                    "slope_changes: The lower series changes from rising to nearly horizontal.",
                    "trend: The upper series rises continuously while the lower series approaches a plateau.",
                    "engineering_meaning: unknown",
                ]
            ),
        ]
    )
    settings = DummySettings(tmp_path)
    service = FigureAnalysisService(settings, ollama)  # type: ignore[arg-type]

    result = asyncio.run(service.analyze_figure(
        document_name="sample.pdf",
        document_id="abc123",
        page_number=1,
        image_index=1,
        image_path=source_path,
        fallback_image_path=fallback_path,
        remaining_vision_calls=10,
    ))

    assert result.status == "valid"
    assert result.vision_calls == 4
    assert result.candidate["fallback_used"] is True
    assert result.should_index is True


def test_forced_graph_uses_one_vision_call(tmp_path: Path) -> None:
    image_path = tmp_path / "forced_graph_p1_fig1.png"
    make_graph_image(image_path)

    ollama = FakeOllama(
        [
            "\n".join(
                [
                    "image_type: graph",
                    "confidence: 0.96",
                    "title: unknown",
                    "analysis: One circular-marker series increases continuously across the plot.",
                    "x_axis: unknown",
                    "x_axis_unit: unknown",
                    "x_axis_scale: unknown",
                    "y_axis: unknown",
                    "y_axis_unit: unknown",
                    "y_axis_scale: unknown",
                    "series_count: 1",
                    "series_count_verified: true",
                    "series_descriptions: Circular markers forming one continuously rising series.",
                    "legend: unknown",
                    "reference_lines: unknown",
                    "plateau: unknown",
                    "peak: unknown",
                    "decline: unknown",
                    "slope_changes: unknown",
                    "trend: The visible series rises continuously.",
                    "engineering_meaning: unknown",
                ]
            )
        ]
    )
    settings = DummySettings(tmp_path)
    service = FigureAnalysisService(settings, ollama)  # type: ignore[arg-type]

    result = asyncio.run(
        service.analyze_figure(
            document_name="sample.pdf",
            document_id="abc123",
            page_number=1,
            image_index=1,
            image_path=image_path,
            remaining_vision_calls=1,
            forced_classification="graph",
        )
    )

    assert result.status == "valid"
    assert result.vision_calls == 1
    assert len(ollama.calls) == 1
    assert result.should_index is True


def test_graph_priority_beats_generic_candidate(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.png"
    generic_path = tmp_path / "generic.png"
    make_graph_image(graph_path)

    generic = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(generic)
    draw.rectangle((150, 120, 750, 480), outline="black", width=3)
    generic.save(generic_path, format="PNG")

    settings = DummySettings(tmp_path)
    service = FigureAnalysisService(settings, FakeOllama([]))  # type: ignore[arg-type]

    graph_priority = service.priority_for_image(
        graph_path,
        "Pressure transient log-log plot and derivative response.",
    )
    generic_priority = service.priority_for_image(
        generic_path,
        "General introductory page.",
    )

    assert graph_priority["score"] > generic_priority["score"]
    assert graph_priority["forced_classification"] == "graph"


def test_semicolon_collapsed_graph_response_is_parsed(tmp_path: Path) -> None:
    image_path = tmp_path / "collapsed_graph_p1_fig1.png"
    make_graph_image(image_path)

    ollama = FakeOllama(
        [
            (
                "image_type: graph; confidence: 0.95; title: Log-Log Plot; "
                "analysis: Two visible series rise across the plot; x_axis: Equivalent Time; "
                "x_axis_unit: hours; x_axis_scale: Logarithmic; y_axis: Delta P; "
                "y_axis_unit: psi; y_axis_scale: Logarithmic; series_count: 2; "
                "series_count_verified: true; series_descriptions: solid line with square markers, "
                "dashed line without markers; legend: unknown; reference_lines: horizontal line at 100 psi; "
                "plateau: unknown; peak: unknown; decline: unknown; slope_changes: present in both lines; "
                "trend: both visible series rise continuously; engineering_meaning: unknown"
            )
        ]
    )
    settings = DummySettings(tmp_path)
    service = FigureAnalysisService(settings, ollama)  # type: ignore[arg-type]

    result = asyncio.run(
        service.analyze_figure(
            document_name="sample.pdf",
            document_id="abc123",
            page_number=1,
            image_index=1,
            image_path=image_path,
            remaining_vision_calls=1,
            forced_classification="graph",
        )
    )

    assert result.status == "valid"
    assert result.vision_calls == 1
    assert result.candidate["final_note_data"]["confidence"] == 0.95
    assert result.candidate["final_note_data"]["analysis"].startswith("Two visible series")
    assert result.candidate["final_note_data"]["series_descriptions"] == [
        "solid line with square markers",
        "dashed line without markers",
    ]


def test_semicolon_collapsed_diagram_response_is_parsed(tmp_path: Path) -> None:
    image_path = tmp_path / "collapsed_diagram_p1_fig1.png"
    make_graph_image(image_path)

    ollama = FakeOllama(
        [
            (
                "image_type: Engineering diagram; confidence: 0.95; title: Aquifer system; "
                "analysis: Schematic of producing and observation wells; "
                "components: Aquifer; Oil Zone; Producing Well; Observation Well; "
                "component_labels: WOC; OWOC; connections: Producing Well to Aquifer; "
                "Observation Well to Aquifer; flow_directions: Vertical water flux; "
                "annotations: u_z; legend: u_z is vertical water flux; "
                "engineering_meaning: unknown"
            )
        ]
    )
    settings = DummySettings(tmp_path)
    service = FigureAnalysisService(settings, ollama)  # type: ignore[arg-type]

    result = asyncio.run(
        service.analyze_figure(
            document_name="sample.pdf",
            document_id="abc123",
            page_number=1,
            image_index=1,
            image_path=image_path,
            remaining_vision_calls=1,
            forced_classification="diagram",
        )
    )

    assert result.status == "valid"
    assert result.vision_calls == 1
    assert result.candidate["final_note_data"]["confidence"] == 0.95
    assert result.candidate["final_note_data"]["components"] == [
        "Aquifer",
        "Oil Zone",
        "Producing Well",
        "Observation Well",
    ]
    assert result.candidate["final_note_data"]["analysis"].startswith("Schematic")
