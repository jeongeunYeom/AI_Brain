from app.services.chunking import chunk_pages


def test_chunk_pages_preserves_document_page_metadata():
    chunks = chunk_pages(
        [{"page": 3, "text": "permeability porosity saturation pressure gradient", "document_id": "abc"}],
        "reservoir.pdf",
        "abc",
        chunk_size=20,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert chunks[0]["metadata"]["document"] == "reservoir.pdf"
    assert chunks[0]["metadata"]["page"] == 3
    assert chunks[0]["id"].startswith("abc:p3:c0")


def test_chunk_pages_adds_ontology_metadata():
    chunks = chunk_pages(
        [
            {
                "page": 1,
                "text": (
                    "Porosity and permeability control radial flow. "
                    "BHP means bottomhole pressure."
                ),
            }
        ],
        "well-test.pdf",
        "doc1",
        chunk_size=200,
        chunk_overlap=0,
    )

    metadata = chunks[0]["metadata"]

    assert metadata["ontology_version"] == "v0.1"
    assert metadata["domain"] == "petroleum_engineering"
    assert metadata["concept_count"] >= 4
    assert "porosity" in metadata["concepts"]
    assert "absolute_permeability" in metadata["concepts"]
    assert "radial_flow" in metadata["concepts"]
    assert "bottomhole_pressure" in metadata["concepts"]
