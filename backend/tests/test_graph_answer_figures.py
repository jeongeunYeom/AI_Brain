from types import SimpleNamespace

from app.services.qa import QAService


class Dummy:
    pass


def make_service(tmp_path):
    settings = SimpleNamespace(figures_dir=tmp_path)
    return QAService(settings, Dummy(), Dummy())


def figure_hit(image_path: str, *, title: str = "Log-Log Plot"):
    return {
        "id": "doc:p10:c0",
        "metadata": {
            "document": "doc.pdf",
            "page": 10,
        },
        "text": (
            "[Extracted figure notes]\n"
            "Figure 2: [Figure Note Metadata]\n"
            "document_name: doc.pdf\n"
            "page_number: 10\n"
            "image_index: 2\n"
            f"image_path: {image_path}\n"
            "image_type: graph\n"
            f"title: {title}\n"
            "title_verified: true\n"
            "analysis: Two series are visible.\n"
            "x_axis: Time\n"
        ),
    }


def test_returns_existing_evidence_figure(tmp_path):
    image = tmp_path / "doc_p10_fig2.jpeg"
    image.write_bytes(b"image")

    service = make_service(tmp_path)
    figures = service._figure_references(
        [figure_hit(r"D:\AI_Brain\data\figures\doc_p10_fig2.jpeg")]
    )

    assert len(figures) == 1
    assert figures[0].filename == "doc_p10_fig2.jpeg"
    assert figures[0].document == "doc.pdf"
    assert figures[0].page == 10
    assert figures[0].title == "Log-Log Plot"
    assert figures[0].image_type == "graph"
    assert figures[0].url.endswith("/doc_p10_fig2.jpeg")
    assert figures[0].preview_url is None


def test_returns_preview_url_when_preview_creation_succeeds(tmp_path):
    from PIL import Image

    image = tmp_path / "doc_p10_fig2.jpeg"
    Image.new("RGB", (80, 40), "black").save(image)

    service = make_service(tmp_path)
    figures = service._figure_references(
        [figure_hit(r"D:\AI_Brain\data\figures\doc_p10_fig2.jpeg")]
    )

    assert len(figures) == 1
    assert figures[0].url.endswith("/doc_p10_fig2.jpeg")
    assert figures[0].preview_url is not None
    assert figures[0].preview_url.startswith("/api/figure-previews/")
    assert figures[0].preview_source == "extracted_image"


def test_deduplicates_and_limits_to_three(tmp_path):
    hits = []
    for index in range(5):
        filename = f"doc_p{index}_fig1.png"
        (tmp_path / filename).write_bytes(b"image")
        hits.append(
            figure_hit(
                rf"D:\AI_Brain\data\figures\{filename}",
                title=f"Figure {index}",
            )
        )

    service = make_service(tmp_path)
    figures = service._figure_references(
        [hits[0], hits[0], *hits[1:]],
        limit=3,
    )

    assert [item.filename for item in figures] == [
        "doc_p0_fig1.png",
        "doc_p1_fig1.png",
        "doc_p2_fig1.png",
    ]


def test_skips_missing_and_decoration_images(tmp_path):
    decoration = tmp_path / "decor.png"
    decoration.write_bytes(b"image")

    decoration_hit = figure_hit(
        r"D:\AI_Brain\data\figures\decor.png",
        title="null",
    )
    decoration_hit["text"] = decoration_hit["text"].replace(
        "image_type: graph",
        "image_type: page_decoration",
    )

    missing_hit = figure_hit(
        r"D:\AI_Brain\data\figures\missing.png"
    )

    service = make_service(tmp_path)
    figures = service._figure_references(
        [decoration_hit, missing_hit]
    )

    assert figures == []


def test_rft_comparison_orders_before_and_after_figures(tmp_path):
    from PIL import Image

    before = tmp_path / "doc_p440_fig2.jpeg"
    after = tmp_path / "doc_p440_fig3.jpeg"
    Image.new("RGB", (80, 40), "white").save(before)
    Image.new("RGB", (80, 40), "white").save(after)

    before_hit = figure_hit(
        r"D:\AI_Brain\data\figures\doc_p440_fig2.jpeg",
        title="Appraisal Well RFT Survey",
    )
    after_hit = figure_hit(
        r"D:\AI_Brain\data\figures\doc_p440_fig3.jpeg",
        title="RFT Survey after Significant Production",
    )

    service = make_service(tmp_path)
    figures = service._figure_references(
        [after_hit, before_hit],
        question=(
            "RFT Survey before significant production과 "
            "after significant production을 비교해줘."
        ),
    )

    assert [item.title for item in figures[:2]] == [
        "Appraisal Well RFT Survey",
        "RFT Survey after Significant Production",
    ]


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows

    def get(self, include=None):
        return {
            "ids": [row[0] for row in self.rows],
            "documents": [row[1] for row in self.rows],
            "metadatas": [row[2] for row in self.rows],
        }


class FakeVectorStore:
    def __init__(self, rows):
        self.collection = FakeCollection(rows)


def test_rft_comparison_fetches_missing_before_companion_from_collection(tmp_path):
    from PIL import Image

    before = tmp_path / "doc_p440_fig2.jpeg"
    after = tmp_path / "doc_p440_fig3.jpeg"
    Image.new("RGB", (80, 40), "white").save(before)
    Image.new("RGB", (80, 40), "white").save(after)

    before_text = figure_hit(
        r"D:\AI_Brain\data\figures\doc_p440_fig2.jpeg",
        title="Appraisal Well RFT Survey",
    )["text"]
    after_hit = figure_hit(
        r"D:\AI_Brain\data\figures\doc_p440_fig3.jpeg",
        title="RFT Survey after Significant Production",
    )
    rows = [
        (
            "doc:p440:c1",
            before_text,
            {"document": "doc.pdf", "page": 440},
        ),
        (
            "doc:p440:c2",
            after_hit["text"],
            {"document": "doc.pdf", "page": 440},
        ),
    ]

    settings = SimpleNamespace(figures_dir=tmp_path)
    service = QAService(settings, FakeVectorStore(rows), Dummy())
    figures = service._figure_references(
        [after_hit],
        question=(
            "RFT Survey before significant production과 "
            "after significant production을 비교해줘."
        ),
    )

    assert [item.title for item in figures[:2]] == [
        "Appraisal Well RFT Survey",
        "RFT Survey after Significant Production",
    ]


def test_interpreted_rft_partial_answer_uses_supported_legend_evidence(tmp_path):
    service = make_service(tmp_path)
    hit = {
        "id": "doc:p441:c2",
        "metadata": {
            "document": "Well_Test_Analysis.pdf",
            "page": 441,
        },
        "text": (
            "analysis: The pressure-versus-TVD figure divides the interpreted "
            "RFT data into Zones 1 through 7.\n"
            "series_descriptions:\n"
            "  - Open-circle points identified in the legend as supercharged points.\n"
            "  - Open-square points identified in the legend as double pretest sequence points.\n"
            "  - Filled pressure-observation points, some shown with vertical connecting or uncertainty lines.\n"
            "  - X-shaped points aligned with the mud-gradient line.\n"
            "reference_lines:\n"
            "  - Mud-gradient line labeled 1.11 g/cc and 1.58 psi/m.\n"
            "trend_summary: Pressure observations vary among seven interpreted depth zones "
            "and lie at substantially lower pressures than the mud-gradient trend; "
            "the figure does not present one continuous formation-pressure curve across all zones.\n"
        ),
    }

    answer = service._supported_partial_answer(
        (
            "Interpreted RFT Data 그래프에서 Zone 1부터 Zone 7까지 각각 어떤 "
            "압력 거동을 보이며, supercharged points와 double pretest sequence "
            "points는 어디에 해당하는가?"
        ),
        [hit],
    )

    assert answer is not None
    assert "Zone별 세부 압력 거동은 확인할 수 없습니다" in answer
    assert "Open-circle points는 supercharged points" in answer
    assert "Open-square points는 double pretest sequence points" in answer
    assert "1.11 g/cc" in answer
    assert "[Well_Test_Analysis.pdf, p.441]" in answer
