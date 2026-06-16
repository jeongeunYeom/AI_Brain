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
