from __future__ import annotations

import json

from app.services.ontology import (
    extract_concepts,
    relation_edges,
    write_relation_graph,
)


def test_extract_concepts_uses_petroleum_synonyms():
    concepts = extract_concepts(
        "BHP and bottom-hole pressure are linked to mud weight and pore pressure."
    )

    assert "bottomhole_pressure" in concepts
    assert "mud_weight" in concepts
    assert "pore_pressure" in concepts


def test_relation_graph_includes_aliases_and_chunk_cooccurrence(tmp_path):
    chunks = [
        {
            "id": "doc:p1:c0",
            "metadata": {
                "document_id": "doc",
                "document": "well-test.pdf",
                "page": 1,
                "concepts": (
                    "porosity|absolute_permeability|radial_flow|"
                    "bottomhole_pressure"
                ),
            },
        }
    ]

    edges = relation_edges(chunks)

    assert {
        "source": "bhp",
        "relation": "alias_of",
        "target": "bottomhole_pressure",
        "source_type": "ontology",
        "ontology_version": "v0.1",
    } in edges
    assert any(
        edge["relation"] == "co_occurs_with"
        and edge["source"] == "absolute_permeability"
        and edge["target"] == "porosity"
        and edge["chunk_id"] == "doc:p1:c0"
        for edge in edges
    )

    graph_path = tmp_path / "ontology" / "doc.jsonl"
    write_relation_graph(graph_path, chunks)

    saved = [
        json.loads(line)
        for line in graph_path.read_text(encoding="utf-8").splitlines()
    ]
    assert saved == edges
