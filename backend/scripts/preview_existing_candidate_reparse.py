from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.figure_analysis import (
    DIAGRAM_TYPES,
    GRAPH_TYPES,
    diagram_metadata,
    graph_metadata,
    parse_bool,
    parse_key_values,
)


def predict_status(
    candidate: dict[str, Any],
    settings: Any,
) -> tuple[str, list[str], list[str]]:
    raw_text = str(candidate.get("raw_vision_text") or "").strip()
    current_status = str(candidate.get("candidate_status") or "")
    if not raw_text:
        return current_status or "review_required", [], list(
            candidate.get("manual_review_reasons") or []
        )

    classification = str(
        candidate.get("effective_classification")
        or candidate.get("automatic_classification")
        or "unknown"
    ).strip().lower()
    class_fields = parse_key_values(str(candidate.get("classification_response") or ""))
    readable_labels = parse_bool(class_fields.get("readable_labels"))

    if classification in GRAPH_TYPES:
        metadata, errors, reasons = graph_metadata(
            raw_text,
            settings=settings,
            readable_labels=readable_labels,
        )
    elif classification in DIAGRAM_TYPES:
        metadata, errors, reasons = diagram_metadata(
            raw_text,
            settings=settings,
            readable_labels=readable_labels,
        )
    else:
        return current_status or "review_required", [], list(
            candidate.get("manual_review_reasons") or []
        )

    schema_valid = bool(metadata.pop("_schema_valid"))
    information_quality = bool(metadata.pop("_information_quality"))
    semantic_grounding = bool(metadata.pop("_semantic_grounding"))
    trend_grounding = bool(metadata.pop("_trend_grounding"))

    if errors:
        predicted = "failed"
    elif (
        schema_valid
        and information_quality
        and semantic_grounding
        and trend_grounding
        and not reasons
    ):
        predicted = "valid"
    else:
        predicted = "review_required"
    return predicted, errors, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--document-id",
        default="01155892bc533f0bcead8df919b991f143103406ac5f138350fe32296fe15462",
    )
    args = parser.parse_args()

    settings = get_settings()
    candidate_dir = settings.figure_candidates_dir / args.document_id
    if not candidate_dir.is_dir():
        raise SystemExit(f"candidate directory not found: {candidate_dir}")

    rows: list[dict[str, Any]] = []
    for path in sorted(candidate_dir.glob("*.json")):
        candidate = json.loads(path.read_text(encoding="utf-8"))
        predicted, errors, reasons = predict_status(candidate, settings)
        rows.append(
            {
                "file": path.name,
                "page_number": candidate.get("page_number"),
                "image_index": candidate.get("image_index"),
                "classification": candidate.get("effective_classification")
                or candidate.get("automatic_classification"),
                "current_status": candidate.get("candidate_status"),
                "predicted_status": predicted,
                "vision_call_count": candidate.get("vision_call_count", 0),
                "validation_errors": "; ".join(errors),
                "review_reasons": "; ".join(reasons),
            }
        )

    output_dir = settings.data_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"candidate_reparse_preview_{stamp}.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    current_counts = Counter(str(row["current_status"]) for row in rows)
    predicted_counts = Counter(str(row["predicted_status"]) for row in rows)

    print("===== CANDIDATE REPARSE PREVIEW =====")
    print(f"candidate_count={len(rows)}")
    print("current=" + ", ".join(f"{k}:{v}" for k, v in sorted(current_counts.items())))
    print("predicted=" + ", ".join(f"{k}:{v}" for k, v in sorted(predicted_counts.items())))
    print(f"csv={output_path}")
    print("NO_FILES_MODIFIED=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
