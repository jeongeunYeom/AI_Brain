from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any


ONTOLOGY_VERSION = "v0.1"
DOMAIN = "petroleum_engineering"

CONCEPT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "absolute_permeability": (
        "absolute permeability",
        "intrinsic permeability",
        "permeability",
        "투과도",
    ),
    "archie_equation": (
        "archie's equation",
        "archie equation",
        "archie law",
        "archie",
        "아치 식",
    ),
    "bottomhole_pressure": (
        "bottomhole pressure",
        "bottom-hole pressure",
        "bottom hole pressure",
        "bhp",
        "wellbore pressure",
        "공저압",
    ),
    "capillary_pressure": (
        "capillary pressure",
        "pc",
        "모세관압",
    ),
    "co2_storage": (
        "co2 storage",
        "co₂ storage",
        "carbon storage",
        "geological storage",
        "지중저장",
    ),
    "formation_resistivity_factor": (
        "formation resistivity factor",
        "resistivity factor",
        "formation factor",
    ),
    "fracture_pressure": (
        "fracture pressure",
        "fracture gradient",
        "파쇄압",
    ),
    "free_co2": (
        "free co2",
        "free co₂",
        "mobile co2",
        "mobile co₂",
        "자유 co2",
        "자유 co₂",
    ),
    "mud_weight": (
        "mud weight",
        "mud density",
        "ppg",
        "specific gravity",
        "sg",
        "이수 밀도",
    ),
    "porosity": (
        "porosity",
        "pore volume",
        "공극률",
    ),
    "pore_pressure": (
        "pore pressure",
        "formation pressure",
        "reservoir pressure",
        "공극압",
    ),
    "pressure_buildup": (
        "pressure buildup",
        "pressure build-up",
        "build-up test",
        "buildup test",
        "압력 상승",
    ),
    "radial_flow": (
        "radial flow",
        "radial-flow",
        "방사형 유동",
    ),
    "residual_co2_saturation": (
        "residual co2 saturation",
        "residual co₂ saturation",
        "residual gas saturation",
        "srco2",
        "srco₂",
        "잔류 co2 포화도",
        "잔류 co₂ 포화도",
    ),
    "trapped_co2": (
        "trapped co2",
        "trapped co₂",
        "residual trapping",
        "structural trapping",
        "포획 co2",
        "포획 co₂",
    ),
    "water_saturation": (
        "water saturation",
        "sw",
        "수포화도",
    ),
    "wellbore_storage": (
        "wellbore storage",
        "well bore storage",
        "well-bore storage",
        "저류공 저장",
    ),
}

ALIAS_RELATIONS: tuple[tuple[str, str], ...] = (
    ("bhp", "bottomhole_pressure"),
    ("ppg", "mud_weight"),
    ("sg", "mud_weight"),
    ("srco2", "residual_co2_saturation"),
    ("sw", "water_saturation"),
)


def extract_concepts(text: str) -> list[str]:
    normalized = _normalize_text(text)
    matches: list[tuple[int, str]] = []
    seen: set[str] = set()

    for concept, synonyms in CONCEPT_SYNONYMS.items():
        positions = [
            position
            for synonym in synonyms
            if (position := _find_term(normalized, synonym)) >= 0
        ]
        if positions and concept not in seen:
            seen.add(concept)
            matches.append((min(positions), concept))

    return [
        concept
        for _, concept in sorted(matches, key=lambda item: (item[0], item[1]))
    ]


def concept_metadata(text: str) -> dict[str, str | int]:
    concepts = extract_concepts(text)
    return {
        "ontology_version": ONTOLOGY_VERSION,
        "domain": DOMAIN,
        "concepts": "|".join(concepts),
        "concept_count": len(concepts),
    }


def write_relation_graph(
    path: Path,
    chunks: list[dict[str, Any]],
) -> None:
    edges = relation_edges(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for edge in edges:
            handle.write(json.dumps(edge, ensure_ascii=False) + "\n")


def relation_edges(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    present_concepts = {
        concept
        for chunk in chunks
        for concept in str(
            (chunk.get("metadata") or {}).get("concepts") or ""
        ).split("|")
        if concept
    }
    edges: list[dict[str, Any]] = [
        {
            "source": alias,
            "relation": "alias_of",
            "target": concept,
            "source_type": "ontology",
            "ontology_version": ONTOLOGY_VERSION,
        }
        for alias, concept in ALIAS_RELATIONS
        if concept in present_concepts
    ]

    seen: set[tuple[str, str, str, str]] = set()
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        concepts = [
            value
            for value in str(metadata.get("concepts") or "").split("|")
            if value
        ]
        if len(concepts) < 2:
            continue

        for left, right in combinations(sorted(set(concepts)), 2):
            key = (
                str(chunk.get("id") or ""),
                left,
                "co_occurs_with",
                right,
            )
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "source": left,
                    "relation": "co_occurs_with",
                    "target": right,
                    "chunk_id": str(chunk.get("id") or ""),
                    "document_id": metadata.get("document_id"),
                    "document": metadata.get("document"),
                    "page": metadata.get("page"),
                    "ontology_version": ONTOLOGY_VERSION,
                }
            )

    return edges


def _normalize_text(text: str) -> str:
    return (
        str(text)
        .replace("’", "'")
        .replace("`", "'")
        .replace("₂", "2")
        .lower()
    )


def _find_term(normalized_text: str, term: str) -> int:
    normalized_term = _normalize_text(term)
    if re.fullmatch(r"[a-z0-9_./-]+", normalized_term):
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])"
        match = re.search(pattern, normalized_text)
        return match.start() if match else -1
    return normalized_text.find(normalized_term)
