import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.core.config import Settings
from scripts.migrate_figure_assets import (
    ANALYSIS_PROMPT_VERSION,
    CANDIDATE_SCHEMA_VERSION,
    CLASSIFIER_VERSION,
    DIAGRAM_REQUIRED_FIELDS,
    extract_top_level_keys,
    GRAPH_REQUIRED_FIELDS,
    analyze_reanalysis_operation,
    apply_operation_grounding,
    apply_plan,
    build_plan,
    classify_asset,
    create_verified_backup,
    export_review_report,
    export_candidate_inventory,
    export_classification_delta,
    failed_candidate_path,
    failed_candidate_payload,
    group_summaries,
    load_candidate,
    make_candidate_payload,
    normalize_schematic_components,
    normalize_schematic_labels,
    main_async,
    parse_serialized_note_data,
    parse_serialized_note_raw,
    prompt_for_classification,
    restore_manifest,
    safe_document_stem,
    SERIALIZER_VERSION,
    strict_graph_note_text,
    target_stem,
    update_note_text,
    validate_note_equivalence,
    validate_candidate_output,
    validate_serialized_note_data,
    validate_vision_candidate,
    write_candidate_output,
    write_failed_candidate,
    write_manifest,
)


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=tmp_path)
    for path in [
        settings.figures_dir,
        settings.figure_notes_dir,
        settings.metadata_dir,
        settings.raw_dir,
        settings.vector_db_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return settings


def write_metadata(settings: Settings, doc_id: str, filename: str) -> None:
    (settings.metadata_dir / f"{doc_id}.json").write_text(
        json.dumps({"document_id": doc_id, "filename": filename}),
        encoding="utf-8",
    )


def write_asset(settings: Settings, stem: str, *, note: str = "", image_color: str = "white") -> None:
    Image.new("RGB", (500, 400), image_color).save(settings.figures_dir / f"{stem}.jpeg")
    (settings.figure_notes_dir / f"{stem}.md").write_text(note or "[Figure Note Metadata]\n", encoding="utf-8")


def test_safe_name_and_target_stem_formatting():
    doc_id = "01155892bc533f0bcead8df919b991f143103406ac5f138350fe32296fe15462"

    assert safe_document_stem("Bad:Name / Test?.pdf", doc_id) == "Bad_Name_Test"
    assert target_stem("Heriot-Watt_University_-_Well_Test_Analysis.pdf", doc_id, 261, 2).endswith(
        "_p0261_fig02"
    )


def test_resolves_pdf_name_and_updates_note_paths(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "a" * 64
    stem = f"{doc_id}_p261_fig2"
    write_metadata(settings, doc_id, "Heriot-Watt_University_-_Well_Test_Analysis.pdf")
    write_asset(
        settings,
        stem,
        note=(
            "[Figure Note Metadata]\n"
            f"document_name: {stem}.md\n"
            f"document_id: {doc_id}\n"
            f"image_path: {settings.figures_dir / f'{stem}.jpeg'}\n"
        ),
    )

    plan = build_plan(
        settings,
        document="Well_Test_Analysis.pdf",
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=False,
        update_chroma=False,
    )
    op = plan["operations"][0]
    new_note = Path(op["new_note"])
    text = update_note_text(
        op["old_note_text"],
        {"document_name": op["document_name"], "document_id": op["document_id"], "image_path": op["new_image"]},
    )

    assert plan["stats"]["resolved_documents"] == 1
    assert new_note.name == "Heriot-Watt_University_-_Well_Test_Analysis_p0261_fig02.md"
    assert "document_name: Heriot-Watt_University_-_Well_Test_Analysis.pdf" in text
    assert "document_id: " + doc_id in text
    assert "image_path: " in text and "_p0261_fig02.jpeg" in text


def test_resolves_already_renamed_pdf_name_asset(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "a" * 64
    write_metadata(settings, doc_id, "Heriot-Watt_University_-_Well_Test_Analysis.pdf")
    stem = "Heriot-Watt_University_-_Well_Test_Analysis_p0261_fig02"
    write_asset(
        settings,
        stem,
        note=(
            "[Figure Note Metadata]\n"
            "document_name: Heriot-Watt_University_-_Well_Test_Analysis.pdf\n"
            f"document_id: {doc_id}\n"
            "page_number: 261\n"
            "image_index: 2\n"
        ),
    )

    plan = build_plan(
        settings,
        document="Heriot-Watt_University_-_Well_Test_Analysis.pdf",
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=False,
        update_chroma=False,
    )

    assert plan["stats"]["resolved_asset_count"] == 1
    assert plan["stats"]["already_renamed_images"] == 1
    assert plan["stats"]["already_renamed_notes"] == 1
    assert plan["stats"]["rename_image_candidates"] == 0
    assert plan["stats"]["rename_active_note_candidates"] == 0
    assert plan["operations"][0]["document_id"] == doc_id


def test_dry_run_does_not_change_files(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "b" * 64
    stem = f"{doc_id}_p1_fig1"
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, stem)

    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=False,
        update_chroma=False,
    )

    assert plan["stats"]["rename_candidates"] == 1
    assert (settings.figures_dir / f"{stem}.jpeg").exists()
    assert not (settings.figures_dir / "Doc_p0001_fig01.jpeg").exists()


def test_collision_is_reported_without_overwrite(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "c" * 64
    stem = f"{doc_id}_p1_fig1"
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, stem)
    Image.new("RGB", (500, 400), "white").save(settings.figures_dir / "Doc_p0001_fig01.jpeg")

    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=False,
        update_chroma=False,
    )

    assert plan["stats"]["name_collisions"] == 1


def test_logo_quarantine_but_graph_is_kept(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "d" * 64
    write_metadata(settings, doc_id, "Doc.pdf")
    for page in [1, 2, 3]:
        write_asset(settings, f"{doc_id}_p{page}_fig1", note="image_type: logo\nconfidence: 0.95\n")
        Image.new("RGB", (120, 120), "white").save(settings.figures_dir / f"{doc_id}_p{page}_fig1.jpeg")
    write_asset(settings, f"{doc_id}_p4_fig1", note="image_type: graph\nconfidence: 0.9\n", image_color="black")

    dry_plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=True,
        reanalyze_graphs=True,
        update_chroma=False,
    )
    decision_group = dry_plan["operations"][0]["group_id"]
    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=True,
        reanalyze_graphs=True,
        update_chroma=False,
        decisions={decision_group: {"action": "quarantine", "classification": "logo", "reason": "reviewed"}},
    )

    assert dry_plan["stats"]["would_quarantine"] == 0
    assert plan["stats"]["approved_quarantine_groups"] == 1
    assert plan["stats"]["final_logo_candidates"] == 3
    assert plan["stats"]["graph_candidates"] == 1
    assert plan["stats"]["would_quarantine"] == 3
    assert plan["stats"]["would_reanalyze"] == 1
    assert any(op["quarantine"] for op in plan["operations"])
    assert not [op for op in plan["operations"] if op["classification"] == "graph" and op["quarantine"]]
    assert all("Doc_p000" in Path(op["new_image"]).name for op in plan["operations"] if op["quarantine"])


def test_decision_still_applies_after_rename(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "9" * 64
    write_metadata(settings, doc_id, "Doc.pdf")
    for page in [1, 2, 3]:
        stem = f"Doc_p{page:04d}_fig01"
        write_asset(settings, stem, note=f"document_name: Doc.pdf\ndocument_id: {doc_id}\npage_number: {page}\nimage_index: 1\n")
        Image.new("RGB", (120, 120), "white").save(settings.figures_dir / f"{stem}.jpeg")
    dry_plan = build_plan(settings, document="Doc.pdf", limit=None, quarantine_logos=True, reanalyze_graphs=False, update_chroma=False)
    group_id = dry_plan["operations"][0]["canonical_group_id"]
    plan = build_plan(
        settings,
        document="Doc.pdf",
        limit=None,
        quarantine_logos=True,
        reanalyze_graphs=False,
        update_chroma=False,
        decisions={group_id: {"action": "quarantine", "classification": "logo"}},
    )

    assert plan["stats"]["resolved_asset_count"] == 3
    assert plan["stats"]["already_renamed_images"] == 3
    assert plan["stats"]["would_quarantine"] == 3
    assert plan["stats"]["rename_image_candidates"] == 0


def test_low_confidence_logo_is_not_quarantined(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "e" * 64
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, f"{doc_id}_p1_fig1", note="image_type: logo\nconfidence: 0.2\n")
    Image.new("RGB", (120, 120), "white").save(settings.figures_dir / f"{doc_id}_p1_fig1.jpeg")

    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=True,
        reanalyze_graphs=False,
        update_chroma=False,
    )

    assert plan["stats"]["uncertain_logo_candidates"] == 1
    assert plan["stats"]["would_quarantine"] == 0
    assert plan["stats"]["unknown_candidates"] == 1


def test_backup_note_is_not_active_and_classification_matches_images(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "2" * 64
    stem = f"{doc_id}_p1_fig1"
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, stem, note="image_type: graph\nconfidence: 0.9\n")
    (settings.figure_notes_dir / f"{stem}.20260625_120000.bak.md").write_text("backup", encoding="utf-8")

    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=False,
        update_chroma=False,
    )

    assert plan["stats"]["active_notes"] == 1
    assert plan["stats"]["backup_notes_excluded"] == 1
    assert plan["stats"]["classified_total"] == plan["stats"]["filtered_document_images"] == 1
    assert plan["stats"]["classification_count_matches"] is True


def test_dark_log_log_graph_is_protected(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "3" * 64
    stem = f"{doc_id}_p261_fig2"
    write_metadata(settings, doc_id, "Doc.pdf")
    Image.new("RGB", (1024, 640), "black").save(settings.figures_dir / f"{stem}.jpeg")
    (settings.figure_notes_dir / f"{stem}.md").write_text(
        "image_type: log-log plot\nanalysis: pressure delta time psi hours\nconfidence: 0.67\n",
        encoding="utf-8",
    )

    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=True,
        reanalyze_graphs=True,
        update_chroma=False,
    )
    op = plan["operations"][0]

    assert op["page"] == 261
    assert op["figure"] == 2
    assert op["classification"] == "graph"
    assert op["quarantine"] is False
    assert plan["stats"]["graph_candidates"] == 1
    assert plan["stats"]["logo_candidates"] == 0


def test_logo_requires_three_distinct_pages_and_high_confidence(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "4" * 64
    write_metadata(settings, doc_id, "Doc.pdf")
    for page in [1, 2]:
        write_asset(settings, f"{doc_id}_p{page}_fig1", note="image_type: logo\nconfidence: 0.95\n")
    for page in [3, 4, 5]:
        write_asset(settings, f"{doc_id}_p{page}_fig1", note="image_type: logo\nconfidence: 0.89\n", image_color="blue")

    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=True,
        reanalyze_graphs=False,
        update_chroma=False,
    )

    assert plan["stats"]["logo_candidates"] == 0
    assert plan["stats"]["would_quarantine"] == 0


def test_html_review_report_does_not_modify_sources(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "5" * 64
    stem = f"{doc_id}_p1_fig1"
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, stem, note="image_type: graph\nconfidence: 0.9\n")
    image = settings.figures_dir / f"{stem}.jpeg"
    before = image.read_bytes()
    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=False,
        update_chroma=False,
    )

    report = tmp_path / "review.html"
    export_review_report(plan, report)

    assert report.exists()
    assert "data:image/jpeg;base64" in report.read_text(encoding="utf-8")
    assert report.with_suffix(".json").exists()
    assert report.with_suffix(".csv").exists()
    assert "Showing" in report.read_text(encoding="utf-8")
    assert image.read_bytes() == before


def test_stable_group_id_and_canonical_summary(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "6" * 64
    write_metadata(settings, doc_id, "Doc.pdf")
    for page in [1, 2]:
        write_asset(settings, f"{doc_id}_p{page}_fig1", note="image_type: logo\nconfidence: 0.95\n")

    first = build_plan(settings, document=None, limit=None, quarantine_logos=False, reanalyze_graphs=False, update_chroma=False)
    second = build_plan(settings, document=None, limit=None, quarantine_logos=False, reanalyze_graphs=False, update_chroma=False)

    assert first["operations"][0]["group_id"] == second["operations"][0]["group_id"]
    assert first["operations"][0]["canonical_group_id"] == second["operations"][0]["canonical_group_id"]
    assert group_summaries(first)[0]["near_duplicate_count"] == 2


def test_manual_decision_refuses_group_with_graph(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "7" * 64
    write_metadata(settings, doc_id, "Doc.pdf")
    for page in [1, 2, 3]:
        write_asset(settings, f"{doc_id}_p{page}_fig1", note="image_type: graph\nconfidence: 0.9\n")
    plan_without = build_plan(settings, document=None, limit=None, quarantine_logos=True, reanalyze_graphs=False, update_chroma=False)
    group_id = plan_without["operations"][0]["canonical_group_id"]
    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=True,
        reanalyze_graphs=False,
        update_chroma=False,
        decisions={group_id: {"action": "quarantine", "classification": "logo"}},
    )

    assert plan["stats"]["would_quarantine"] == 0


def test_unknown_can_be_renamed_but_not_quarantined_without_decision(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "8" * 64
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, f"{doc_id}_p1_fig1", note="")
    image = Image.new("RGB", (500, 500), "white")
    draw = ImageDraw.Draw(image)
    for x in range(0, 500, 20):
        draw.line((x, 0, 499 - x, 499), fill=(x % 255, 100, 50), width=2)
    image.save(settings.figures_dir / f"{doc_id}_p1_fig1.jpeg")

    plan = build_plan(settings, document=None, limit=None, quarantine_logos=True, reanalyze_graphs=False, update_chroma=False)

    assert plan["operations"][0]["classification"] == "unknown"
    assert plan["operations"][0]["quarantine"] is False
    assert plan["stats"]["rename_image_candidates"] == 1
    assert plan["stats"]["rename_active_note_candidates"] == 1


def test_prompt_for_graph_and_diagram_are_separate():
    assert "series_count" in prompt_for_classification("graph")
    assert "components" in prompt_for_classification("diagram")
    assert "equation_text" in prompt_for_classification("equation")


def test_graph_and_schematic_reanalysis_are_separate(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "a1" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, f"{doc_id}_p1_fig1", note="image_type: log-log plot\nanalysis: x_axis y_axis psi hours\n")
    write_asset(settings, f"{doc_id}_p2_fig1", note="analysis: fault block boundary reservoir schematic arrows\n")

    graphs = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=True,
        update_chroma=False,
    )
    diagrams = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=False,
        reanalyze_diagrams=True,
        update_chroma=False,
    )

    assert graphs["stats"]["graph_reanalysis_candidates"] == 1
    assert graphs["stats"]["schematic_reanalysis_candidates"] == 1
    assert graphs["stats"]["would_reanalyze"] == 1
    assert diagrams["stats"]["would_reanalyze"] == 1
    assert [op for op in diagrams["operations"] if op["reanalyze"]][0]["classification"] == "schematic"


def test_unknown_engineering_is_protected_but_not_reanalyzed(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "a2" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, f"{doc_id}_p1_fig1", note="")

    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=True,
        reanalyze_diagrams=True,
        update_chroma=False,
    )

    assert plan["stats"]["unknown_engineering_figures"] == 1
    assert plan["stats"]["protected_engineering_figures"] == 1
    assert plan["stats"]["would_reanalyze"] == 0


def test_orphan_graph_gets_new_note_plan(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "a3" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    stem = f"{doc_id}_p261_fig2"
    image = Image.new("RGB", (1024, 640), "black")
    draw = ImageDraw.Draw(image)
    draw.line((80, 560, 960, 560), fill="white", width=3)
    draw.line((80, 560, 80, 60), fill="white", width=3)
    draw.line((100, 520, 900, 120), fill="cyan", width=4)
    image.save(settings.figures_dir / f"{stem}.jpeg")

    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=True,
        update_chroma=False,
    )

    assert plan["operations"][0]["classification"] == "graph"
    assert plan["operations"][0]["reanalyze"] is True
    assert plan["stats"]["would_create_new_notes"] == 1


def test_dark_marker_graph_precedes_schematic_heuristic():
    result = classify_asset(
        {"image_type": "schematic"},
        {
            "width": 1024,
            "height": 640,
            "file_size": 60_372,
            "aspect_ratio": 1.6,
            "contrast": 0.61,
            "brightness": 0.06,
            "edge_density": 0.0032,
            "marker_series_detected": True,
        },
        1,
        1,
        "document_name: Well_Test_Analysis.pdf",
    )

    classification, _confidence, _reasons, protections, bucket, initial, overridden, reason = result
    assert classification == "graph"
    assert bucket == "graph"
    assert "dark_graph" in protections
    assert initial == "schematic"
    assert overridden is True
    assert reason == "dark_graph_precedence"


def test_document_name_metadata_does_not_create_schematic_evidence():
    result = classify_asset(
        {},
        {"width": 500, "height": 400, "file_size": 40_000, "aspect_ratio": 1.25, "contrast": 10, "brightness": 120, "edge_density": 0.001},
        1,
        1,
        "document_name: Well_Test_Analysis.pdf\nimage_path: C:/figures/well_plot.jpeg",
    )

    assert result[0] != "schematic"


def test_single_asset_image_type_override_preserves_automatic_classification(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "d1" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    stem = f"{doc_id}_p10_fig1"
    write_asset(settings, stem, note="image_type: schematic\ncomponents: tank\n", image_color="white")
    asset = f"{stem}.jpeg"

    plan = build_plan(
        settings,
        document="Doc.pdf",
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=True,
        update_chroma=False,
        asset=asset,
        force_image_type="graph",
    )
    op = plan["operations"][0]

    assert op["automatic_classification"] == "schematic"
    assert op["effective_classification"] == "graph"
    assert op["classification_forced"] is True
    assert op["classification_force_reason"] == "manual visual review"
    assert op["reanalyze"] is True


def test_force_image_type_without_asset_is_rejected(monkeypatch):
    monkeypatch.setattr("sys.argv", ["migrate_figure_assets.py", "--force-image-type", "graph", "--reanalyze-graphs"])

    assert asyncio.run(main_async()) == 2


def test_dark_graph_rejects_unsupported_circuit_semantics():
    candidate = validate_vision_candidate(
        valid_graph_vision().replace("analysis: Log-log pressure plot", "analysis: Electrical circuit with resistor and capacitor"),
        classification="graph",
        settings=Settings(),
    )

    grounded = apply_operation_grounding(
        candidate,
        {"classification": "graph", "dark_graph_candidate": True, "detected_text": ""},
    )

    assert grounded["schema_valid"] is False
    assert "diagram_semantics_unsupported_by_visual_structure" in grounded["validation_errors"]


def test_dark_graph_without_readable_text_keeps_conservative_fields():
    candidate = validate_vision_candidate(valid_graph_vision(), classification="graph", settings=Settings())

    grounded = apply_operation_grounding(
        candidate,
        {"classification": "graph", "dark_graph_candidate": True, "detected_text": ""},
    )
    metadata = grounded["metadata"]

    assert metadata["title"] is None and metadata["title_verified"] is False
    assert metadata["x_axis"] is None and metadata["x_axis_verified"] is False
    assert metadata["y_axis"] is None and metadata["y_axis_verified"] is False
    assert metadata["legend"] is None and metadata["legend_verified"] is False
    assert metadata["engineering_meaning_verified"] is False
    assert grounded["semantic_grounding_passed"] is False


def test_dark_graph_override_requires_distinct_series():
    candidate = validate_vision_candidate(
        valid_graph_vision().replace("series_count: 2", "series_count: 1").replace(
            "cyan curve rises continuously; yellow curve rises then plateaus near 500 psi; white diagonal reference line is visible",
            "one dotted series rises",
        ),
        classification="graph",
        settings=Settings(),
    )

    grounded = apply_operation_grounding(
        candidate,
        {
            "classification": "graph",
            "dark_graph_candidate": True,
            "detected_text": "",
            "classification_override_reason": "dark_graph_precedence",
        },
    )

    assert grounded["information_quality_passed"] is False
    assert "dark_graph_distinct_series_not_resolved" in grounded["validation_errors"]


def test_fault_block_image_is_schematic(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "a4" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, f"{doc_id}_p1_fig1", note="analysis: fault block reservoir communication boundary arrows\n")

    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=False,
        reanalyze_diagrams=True,
        update_chroma=False,
    )

    assert plan["operations"][0]["classification"] == "schematic"
    assert plan["stats"]["schematic_reanalysis_candidates"] == 1


def test_limit_applies_after_reanalysis_candidates(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "a5" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, f"{doc_id}_p1_fig1", note="")
    Image.new("RGB", (100, 100), "white").save(settings.figures_dir / f"{doc_id}_p1_fig1.jpeg")
    write_asset(settings, f"{doc_id}_p2_fig1", note="image_type: log-log plot\nanalysis: x_axis y_axis psi hours\n", image_color="black")
    write_asset(settings, f"{doc_id}_p3_fig1", note="image_type: log-log plot\nanalysis: x_axis y_axis psi hours\n", image_color="blue")

    plan = build_plan(
        settings,
        document="Doc.pdf",
        limit=1,
        quarantine_logos=False,
        reanalyze_graphs=True,
        update_chroma=False,
    )

    assert plan["stats"]["resolved_asset_count"] == 3
    assert plan["stats"]["filtered_document_images"] == 3
    assert plan["stats"]["page_decoration_candidates"] == 1
    assert plan["stats"]["graph_reanalysis_candidates_total"] == 2
    assert plan["stats"]["selected_reanalysis_targets"] == 1
    assert plan["stats"]["limited_out_reanalysis_targets"] == 1
    assert plan["stats"]["would_reanalyze"] == 1
    assert plan["selected_reanalysis_files"] == [f"{doc_id}_p2_fig1.jpeg"]


def test_asset_selects_exact_graph_candidate(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "a6" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, f"{doc_id}_p1_fig1", note="")
    write_asset(settings, f"{doc_id}_p2_fig1", note="image_type: log-log plot\nanalysis: x_axis y_axis psi hours\n")

    plan = build_plan(
        settings,
        document="Doc.pdf",
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=True,
        update_chroma=False,
        asset=f"{doc_id}_p2_fig1.jpeg",
    )

    assert plan["stats"]["matched_requested_assets"] == 1
    assert plan["stats"]["selected_reanalysis_targets"] == 1
    assert plan["stats"]["would_reanalyze"] == 1
    assert plan["stats"]["would_update_existing_notes"] == 1
    assert plan["stats"]["would_create_new_notes"] == 0
    assert plan["selected_reanalysis_files"] == [f"{doc_id}_p2_fig1.jpeg"]


def test_asset_rejects_non_graph_for_graph_reanalysis(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "a7" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, f"{doc_id}_p1_fig1", note="analysis: fault block boundary schematic arrows\n")

    plan = build_plan(
        settings,
        document="Doc.pdf",
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=True,
        update_chroma=False,
        asset=f"{doc_id}_p1_fig1.jpeg",
    )

    assert plan["stats"]["matched_requested_assets"] == 1
    assert plan["stats"]["asset_error"].startswith("asset_not_valid_for_requested_reanalysis")
    assert plan["stats"]["selected_reanalysis_targets"] == 0
    assert plan["stats"]["would_reanalyze"] == 0


def valid_graph_vision() -> str:
    return "\n".join(
        [
            "image_type: Log-Log Plot",
            "confidence: 0.91",
            "analysis: Log-log pressure plot with multiple visible series and a diagonal reference line.",
            "x_axis: Equivalent Time (CRH)",
            "x_axis_unit: hours",
            "x_axis_scale: logarithmic",
            "y_axis: Delta P",
            "y_axis_unit: psi",
            "y_axis_scale: logarithmic",
            "series_count: 3",
            "series_descriptions: cyan curve rises continuously; yellow curve rises then plateaus near 500 psi; white diagonal reference line is visible",
            "legend: 확인할 수 없음",
            "reference_lines: white diagonal reference line",
            "plateau: yellow curve plateaus near 500 psi",
            "trend: multiple trends: continuous rise, rise then plateau, diagonal reference line",
            "engineering_meaning: 확인할 수 없음",
        ]
    )


def test_strict_schema_rejects_missing_confidence():
    candidate = validate_vision_candidate(
        "image_type: Log-Log Plot\nconfidence: 확인할 수 없음\nanalysis: title only\n",
        classification="graph",
        settings=Settings(),
    )

    assert candidate["schema_valid"] is False
    assert "confidence" in candidate["validation_errors"][0]


def test_normalizes_graph_metadata_types():
    candidate = validate_vision_candidate(valid_graph_vision(), classification="graph", settings=Settings())
    metadata = candidate["metadata"]

    assert candidate["normalized_schema_valid"] is True
    assert isinstance(metadata["series_count"], int)
    assert metadata["series_count"] == 3
    assert metadata["series_count_verified"] is False
    assert isinstance(metadata["series_descriptions"], list)
    assert len(metadata["series_descriptions"]) == 2
    assert isinstance(metadata["reference_lines"], list)
    assert metadata["plateaus"] == ["yellow curve plateaus near 500 psi"]
    assert metadata["legend"] is None
    assert metadata["engineering_meaning_verified"] is False


def test_multiline_series_descriptions_become_array():
    candidate = validate_vision_candidate(
        valid_graph_vision().replace(
            "series_descriptions: cyan curve rises continuously; yellow curve rises then plateaus near 500 psi; white diagonal reference line is visible",
            "series_descriptions:\n1. cyan curve rises continuously\n2. yellow curve rises then plateaus near 500 psi",
        ),
        classification="graph",
        settings=Settings(),
    )

    assert candidate["metadata"]["series_descriptions"] == [
        "One data series rises continuously as Equivalent Time (CRH) increases.",
        "Another data series rises initially and then becomes approximately horizontal.",
    ]


def test_rejects_unparseable_series_count():
    candidate = validate_vision_candidate(
        valid_graph_vision().replace("series_count: 3", "series_count: two"),
        classification="graph",
        settings=Settings(),
    )

    assert candidate["normalized_schema_valid"] is False
    assert "series_count must be an integer" in candidate["validation_errors"][0]


def test_axis_unit_split_keeps_meaningful_parentheses():
    candidate = validate_vision_candidate(
        valid_graph_vision().replace("x_axis: Equivalent Time (CRH)", "x_axis: Equivalent Time (CRH) (hours)").replace("y_axis: Delta P", "y_axis: Delta P (psi)"),
        classification="graph",
        settings=Settings(),
    )
    metadata = candidate["metadata"]

    assert metadata["x_axis"] == "Equivalent Time (CRH)"
    assert metadata["x_axis_unit"] == "hours"
    assert metadata["y_axis"] == "Delta P"
    assert metadata["y_axis_unit"] == "psi"


def test_serialized_note_preserves_array_types(tmp_path: Path):
    settings = make_settings(tmp_path)
    candidate = validate_vision_candidate(valid_graph_vision(), classification="graph", settings=settings)
    text = strict_graph_note_text(
        document_name="Doc.pdf",
        document_id="d" * 64,
        page=261,
        fig=2,
        image_path=tmp_path / "graph.jpeg",
        candidate=candidate,
        settings=settings,
        backup=None,
    )
    parsed = parse_serialized_note_data(text)

    assert isinstance(parsed["series_descriptions"], list)
    assert isinstance(parsed["reference_lines"], list)
    assert isinstance(parsed["plateaus"], list)
    assert isinstance(parsed["peaks"], list)
    assert isinstance(parsed["declines"], list)
    assert isinstance(parsed["slope_changes"], list)
    assert parsed["legend"] is None
    assert parsed["legacy_note_backup"] is None
    assert "peaks: []" in text
    assert "declines: []" in text
    assert "slope_changes: []" in text
    assert "legend: null" in text
    assert "legacy_note_backup: null" in text
    assert validate_serialized_note_data(parsed) == []


def test_serialized_note_rejects_verified_missing_value():
    data = {
        "image_type": "graph",
        "confidence": 0.8,
        "series_count": 2,
        "series_descriptions": [],
        "reference_lines": [],
        "plateaus": [],
        "peaks": [],
        "declines": [],
        "slope_changes": [],
        "legend": None,
        "title_verified": True,
        "title": "Log-Log Plot",
        "x_axis_verified": True,
        "x_axis": "Equivalent Time",
        "y_axis_verified": True,
        "y_axis": None,
        "legacy_note_backup": None,
    }

    assert "y_axis_verified true but y_axis is missing" in validate_serialized_note_data(data)


def test_schematic_candidate_handles_missing_analysis_without_graph_inference():
    candidate = validate_vision_candidate(
        "\n".join(
            [
                "image_type: schematic",
                "confidence: 0.82",
                "components: fault block; wellbore",
                "connections: arrow from block to wellbore",
                "legend: ?뺤씤?????놁쓬",
            ]
        ),
        classification="schematic",
        settings=Settings(),
    )

    assert candidate["schema_valid"] is True
    assert candidate["information_quality_passed"] is True
    assert candidate["metadata"]["analysis"] is None
    assert "trend_summary" not in candidate["metadata"]


def test_schematic_candidate_rejects_empty_information():
    candidate = validate_vision_candidate(
        "image_type: schematic\nconfidence: 0.82\nanalysis: ?뺤씤?????놁쓬\n",
        classification="schematic",
        settings=Settings(),
    )

    assert candidate["schema_valid"] is False
    assert candidate["information_quality_passed"] is False
    assert "schematic lacks visible components" in candidate["validation_errors"][0]


def test_normalize_schematic_components_splits_top_level_components():
    assert normalize_schematic_components("Flow Rate (q), Bottom-hole Pressure (p_wf)") == [
        "Flow Rate (q)",
        "Bottom-hole Pressure (p_wf)",
    ]
    assert normalize_schematic_components("Separator (oil, gas), Pressure Gauge (p)") == [
        "Separator (oil, gas)",
        "Pressure Gauge (p)",
    ]
    assert normalize_schematic_components("Pressure increases, then stabilizes at late time.") == [
        "Pressure increases, then stabilizes at late time."
    ]


def test_normalize_schematic_labels_splits_symbols_and_keeps_name_symbol_pairs():
    assert normalize_schematic_labels("Time, t; q_1, q_2, p_wf, T_1") == [
        "Time (t)",
        "q_1",
        "q_2",
        "p_wf",
        "T_1",
    ]
    assert normalize_schematic_labels("Pressure, p") == ["Pressure (p)"]
    assert normalize_schematic_labels(" q_1, q_2, q_1 ") == ["q_1", "q_2"]


def test_p174_schematic_lists_are_normalized_before_serialization(tmp_path: Path):
    settings = make_settings(tmp_path)
    candidate = validate_vision_candidate(
        "\n".join(
            [
                "- image_type: Diagram",
                "- confidence: High",
                "- title: Flow Schedule for a Two-Rate Test",
                "- components: Flow Rate (q), Bottom-hole Pressure (p_wf)",
                "- labels: Time, t; q_1, q_2, p_wf, T_1",
                "- engineering_purpose: To illustrate flow rate and bottom-hole pressure during a two-rate test",
            ]
        ),
        classification="schematic",
        settings=settings,
    )
    text = strict_graph_note_text(
        document_name="Doc.pdf",
        document_id="d" * 64,
        page=174,
        fig=1,
        image_path=tmp_path / "diagram.jpeg",
        candidate=candidate,
        settings=settings,
        backup=None,
    )
    parsed = parse_serialized_note_data(text)

    assert "confidence: 0.8" in text
    assert candidate["metadata"]["components"] == ["Flow Rate (q)", "Bottom-hole Pressure (p_wf)"]
    assert candidate["metadata"]["component_labels"] == ["Time (t)", "q_1", "q_2", "p_wf", "T_1"]
    assert parsed["components"] == ["Flow Rate (q)", "Bottom-hole Pressure (p_wf)"]
    assert parsed["component_labels"] == ["Time (t)", "q_1", "q_2", "p_wf", "T_1"]
    assert isinstance(parsed["confidence"], float)
    assert parsed["confidence"] == 0.8


def test_missing_confidence_is_presence_failure(tmp_path: Path):
    settings = make_settings(tmp_path)
    candidate = validate_vision_candidate(
        "\n".join(
            [
                "image_type: diagram",
                "confidence: 0.8",
                "components: Flow Rate (q), Bottom-hole Pressure (p_wf)",
            ]
        ),
        classification="schematic",
        settings=settings,
    )
    text = strict_graph_note_text(
        document_name="Doc.pdf",
        document_id="d" * 64,
        page=174,
        fig=1,
        image_path=tmp_path / "diagram.jpeg",
        candidate=candidate,
        settings=settings,
        backup=None,
    )
    broken = "\n".join(line for line in text.splitlines() if not line.startswith("confidence:"))
    parsed, errors, equivalent, diff = validate_note_equivalence(broken, parse_serialized_note_data(text))

    assert "confidence" not in parse_serialized_note_raw(broken)
    assert "confidence" not in extract_top_level_keys(broken)
    assert "confidence" not in parsed
    assert "confidence must parse as float" in errors
    assert "confidence" in diff["missing_serialized_fields"]
    assert equivalent is False


def test_missing_null_and_false_are_distinct(tmp_path: Path):
    settings = make_settings(tmp_path)
    candidate = validate_vision_candidate(
        "image_type: diagram\nconfidence: 0.8\ncomponents: pump\nlegend: 확인할 수 없음\n",
        classification="schematic",
        settings=settings,
    )
    text = strict_graph_note_text(
        document_name="Doc.pdf",
        document_id="d" * 64,
        page=174,
        fig=1,
        image_path=tmp_path / "diagram.jpeg",
        candidate=candidate,
        settings=settings,
        backup=None,
    )
    missing_legend = "\n".join(line for line in text.splitlines() if not line.startswith("legend:"))
    missing_verified = "\n".join(line for line in text.splitlines() if not line.startswith("legend_verified:"))

    assert validate_note_equivalence(missing_legend, parse_serialized_note_data(text))[3]["missing_serialized_fields"] == ["legend"]
    assert validate_note_equivalence(missing_verified, parse_serialized_note_data(text))[3]["missing_serialized_fields"] == ["legend_verified"]


def test_extract_top_level_keys_ignores_list_items():
    text = "title_verified: true\ncomponents:\n  - title_verified: not a key\nlegend_verified: false\n"

    assert extract_top_level_keys(text) == {"title_verified", "components", "legend_verified"}


@pytest.mark.parametrize("missing_field", sorted(DIAGRAM_REQUIRED_FIELDS))
def test_diagram_missing_required_field_fails(missing_field: str, tmp_path: Path):
    settings = make_settings(tmp_path)
    candidate = validate_vision_candidate(
        "image_type: diagram\nconfidence: 0.8\ntitle: Flow Schedule\ntitle_verified: true\ncomponents: Flow Rate (q)\n",
        classification="schematic",
        settings=settings,
    )
    text = strict_graph_note_text(
        document_name="Doc.pdf",
        document_id="d" * 64,
        page=174,
        fig=1,
        image_path=tmp_path / "diagram.jpeg",
        candidate=candidate,
        settings=settings,
        backup=None,
    )
    broken = "\n".join(line for line in text.splitlines() if not line.startswith(f"{missing_field}:"))
    _parsed, errors, equivalent, diff = validate_note_equivalence(broken, parse_serialized_note_data(text))

    assert missing_field in diff["missing_serialized_fields"]
    assert any(missing_field in error for error in errors)
    assert equivalent is False


@pytest.mark.parametrize("missing_field", sorted(GRAPH_REQUIRED_FIELDS))
def test_graph_missing_required_field_fails(missing_field: str, tmp_path: Path):
    settings = make_settings(tmp_path)
    candidate = validate_vision_candidate(valid_graph_vision(), classification="graph", settings=settings)
    text = strict_graph_note_text(
        document_name="Doc.pdf",
        document_id="d" * 64,
        page=261,
        fig=2,
        image_path=tmp_path / "graph.jpeg",
        candidate=candidate,
        settings=settings,
        backup=None,
    )
    broken = "\n".join(line for line in text.splitlines() if not line.startswith(f"{missing_field}:"))
    _parsed, errors, equivalent, diff = validate_note_equivalence(broken, parse_serialized_note_data(text))

    assert missing_field in diff["missing_serialized_fields"]
    assert any(missing_field in error for error in errors)
    assert equivalent is False


def test_title_verified_false_is_serialized(tmp_path: Path):
    settings = make_settings(tmp_path)
    candidate = validate_vision_candidate(
        "image_type: diagram\nconfidence: 0.8\ncomponents: pump\n",
        classification="schematic",
        settings=settings,
    )
    text = strict_graph_note_text(
        document_name="Doc.pdf",
        document_id="d" * 64,
        page=174,
        fig=1,
        image_path=tmp_path / "diagram.jpeg",
        candidate=candidate,
        settings=settings,
        backup=None,
    )

    assert "title_verified: false" in text


def test_graph_equivalence_detects_missing_verified_and_array_loss(tmp_path: Path):
    settings = make_settings(tmp_path)
    candidate = validate_vision_candidate(valid_graph_vision(), classification="graph", settings=settings)
    final_data = final_note_data = strict_graph_note_text(
        document_name="Doc.pdf",
        document_id="d" * 64,
        page=261,
        fig=2,
        image_path=tmp_path / "graph.jpeg",
        candidate=candidate,
        settings=settings,
        backup=None,
    )
    parsed_data = parse_serialized_note_data(final_data)
    parsed_data["series_descriptions"] = parsed_data["series_descriptions"][1:]
    parsed_data.pop("legend_verified")
    parsed_data.pop("trend_verified")
    broken_text = "\n".join(
        [f"{key}: {value}" for key, value in parsed_data.items() if key != "series_descriptions"]
        + ["series_descriptions:", f"  - {parsed_data['series_descriptions'][0]}"]
    )

    _parsed, _errors, equivalent, diff = validate_note_equivalence(broken_text, parse_serialized_note_data(final_data))

    assert equivalent is False
    assert "legend_verified" in diff["missing_serialized_fields"]
    assert "trend_verified" in diff["missing_serialized_fields"]
    assert diff["array_length_mismatches"]["series_descriptions"] == {"candidate": 2, "serialized": 1}


def test_incomplete_reference_line_is_safened():
    candidate = validate_vision_candidate(
        valid_graph_vision().replace("reference_lines: white diagonal reference line", "reference_lines: diagonal lines at y=100 and y"),
        classification="graph",
        settings=Settings(),
    )

    assert candidate["incomplete_strings_detected"] == 1
    assert candidate["metadata"]["reference_lines"] == [
        "Two diagonal reference lines are visible; their exact labels or values cannot be confirmed."
    ]


def test_truncated_trend_ending_in_for_is_rejected():
    candidate = validate_vision_candidate(
        valid_graph_vision().replace("trend: multiple trends: continuous rise, rise then plateau, diagonal reference line", "trend: continuous rise for"),
        classification="graph",
        settings=Settings(),
    )

    assert candidate["incomplete_strings_detected"] >= 1


def test_plateau_does_not_copy_full_analysis():
    text = valid_graph_vision().replace("plateau: yellow curve plateaus near 500 psi", "plateau: 확인할 수 없음")
    candidate = validate_vision_candidate(text, classification="graph", settings=Settings())

    assert candidate["metadata"]["plateaus"] == ["One data series becomes approximately horizontal at higher Equivalent Time (CRH)."]
    assert candidate["metadata"]["plateaus"][0] != candidate["metadata"]["analysis"]


def test_duplicate_analysis_removed_from_detail_arrays():
    analysis = "One curve rises continuously while another appears to plateau."
    candidate = validate_vision_candidate(
        valid_graph_vision()
        .replace("analysis: Log-log pressure plot with multiple visible series and a diagonal reference line.", f"analysis: {analysis}")
        .replace("series_descriptions: cyan curve rises continuously; yellow curve rises then plateaus near 500 psi; white diagonal reference line is visible", f"series_descriptions: {analysis}"),
        classification="graph",
        settings=Settings(),
    )

    assert analysis not in candidate["metadata"]["series_descriptions"]


def test_incomplete_semantics_trigger_one_retry(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path)
    image = tmp_path / "graph.jpeg"
    Image.new("RGB", (500, 400), "black").save(image)
    calls = {"count": 0}

    async def fake_describe(self, image_path, prompt):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 1:
            return valid_graph_vision().replace("reference_lines: white diagonal reference line", "reference_lines: diagonal lines at y=100 and y")
        return valid_graph_vision().replace("reference_lines: white diagonal reference line", "reference_lines: two diagonal reference lines are visible")

    monkeypatch.setattr("app.services.ollama.OllamaClient.describe_image", fake_describe)
    op = {"old_image": str(image), "new_image": str(image), "classification": "graph"}
    candidate = asyncio.run(analyze_reanalysis_operation(settings, __import__("app.services.ollama", fromlist=["OllamaClient"]).OllamaClient(settings), op))

    assert calls["count"] == 2
    assert candidate["semantic_retry_attempts"] == 1
    assert candidate["semantic_retry_successes"] == 1


def test_backup_result_requires_real_verified_file(tmp_path: Path):
    source = tmp_path / "note.md"
    source.write_text("old note", encoding="utf-8")
    result = create_verified_backup(source, tmp_path / "note.bak.md")

    assert result["created"] is True
    assert result["verified"] is True
    assert result["size_bytes"] > 0
    assert result["source_sha256"] == result["backup_sha256"]


def test_candidate_input_validation_rejects_asset_hash_mismatch(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "ab" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    stem = f"{doc_id}_p261_fig2"
    write_asset(settings, stem, note="image_type: log-log plot\nanalysis: x_axis y_axis psi hours\n", image_color="black")
    plan = build_plan(settings, document="Doc.pdf", limit=None, quarantine_logos=False, reanalyze_graphs=True, update_chroma=False)
    op = plan["operations"][0]
    candidate = validate_vision_candidate(valid_graph_vision(), classification="graph", settings=settings)
    payload = make_candidate_payload(settings, op, candidate)
    payload["asset_sha256"] = "bad"
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_candidate(path, op)
    except ValueError as exc:
        assert "asset sha256 mismatch" in str(exc)
    else:
        raise AssertionError("candidate mismatch was not rejected")


def make_valid_candidate_case(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "c1" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    stem = f"{doc_id}_p261_fig2"
    write_asset(settings, stem, note="image_type: graph\nanalysis: pressure plot\n", image_color="black")
    plan = build_plan(settings, document="Doc.pdf", limit=None, quarantine_logos=False, reanalyze_graphs=True, update_chroma=False)
    op = plan["operations"][0]
    candidate = validate_vision_candidate(valid_graph_vision(), classification="graph", settings=settings)
    candidate["semantic_grounding_passed"] = True
    candidate["trend_grounding_passed"] = True
    return settings, op, make_candidate_payload(settings, op, candidate)


def test_valid_candidate_is_written_atomically(tmp_path: Path):
    _settings, op, payload = make_valid_candidate_case(tmp_path)
    output = tmp_path / "candidate.json"

    write_candidate_output(output, payload, op)

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["candidate_status"] == "valid"
    assert validate_candidate_output(stored, op) == []
    assert not output.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_valid", False, "candidate schema_valid is not true"),
        ("information_quality_passed", False, "candidate information_quality_passed is not true"),
        ("final_note_data", None, "candidate final_note_data missing"),
    ],
)
def test_invalid_candidate_payload_is_rejected(field: str, value: object, message: str, tmp_path: Path):
    _settings, op, payload = make_valid_candidate_case(tmp_path)
    payload[field] = value
    output = tmp_path / "candidate.json"

    assert message in validate_candidate_output(payload, op)
    with pytest.raises(ValueError):
        write_candidate_output(output, payload, op)
    assert not output.exists()


@pytest.mark.parametrize("field", ["asset_path", "asset_sha256", "document_id"])
def test_candidate_metadata_mismatch_blocks_output(field: str, tmp_path: Path):
    _settings, op, payload = make_valid_candidate_case(tmp_path)
    payload[field] = "wrong"
    output = tmp_path / "candidate.json"

    errors = validate_candidate_output(payload, op)

    expected = {
        "asset_path": "candidate asset path does not match requested asset",
        "asset_sha256": "candidate asset sha256 mismatch",
        "document_id": "candidate document_id mismatch",
    }
    assert expected[field] in errors
    with pytest.raises(ValueError):
        write_candidate_output(output, payload, op)
    assert not output.exists()


def test_failed_candidate_uses_separate_path(tmp_path: Path):
    _settings, op, _payload = make_valid_candidate_case(tmp_path)
    output = tmp_path / "candidate.json"
    failure = failed_candidate_payload(op, {"schema_valid": False, "validation_errors": ["bad schema"]})

    write_failed_candidate(failed_candidate_path(output), failure)

    assert not output.exists()
    assert json.loads((tmp_path / "candidate.failed.json").read_text(encoding="utf-8"))["candidate_status"] == "failed"


def test_load_candidate_identifies_failed_artifact_first(tmp_path: Path):
    _settings, op, _payload = make_valid_candidate_case(tmp_path)
    path = tmp_path / "candidate.failed.json"
    write_failed_candidate(path, failed_candidate_payload(op, {"schema_valid": False}))

    with pytest.raises(ValueError, match="candidate file is a failed analysis artifact"):
        load_candidate(path, op)


def test_candidate_temp_revalidation_failure_does_not_replace(monkeypatch, tmp_path: Path):
    _settings, op, payload = make_valid_candidate_case(tmp_path)
    output = tmp_path / "candidate.json"
    output.write_text("existing valid bytes", encoding="utf-8")
    calls = {"count": 0}
    real_validate = validate_candidate_output

    def fail_second_validation(data, operation, asset_path=None, *, require_status=True):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 2:
            return ["forced revalidation failure"]
        return real_validate(data, operation, asset_path, require_status=require_status)

    monkeypatch.setattr("scripts.migrate_figure_assets.validate_candidate_output", fail_second_validation)

    with pytest.raises(ValueError, match="forced revalidation failure"):
        write_candidate_output(output, payload, op)

    assert output.read_text(encoding="utf-8") == "existing valid bytes"
    assert not output.with_suffix(".json.tmp").exists()


def test_candidate_output_failure_returns_nonzero_and_preserves_existing_valid(monkeypatch, tmp_path: Path):
    settings, op, payload = make_valid_candidate_case(tmp_path)
    output = tmp_path / "candidate.json"
    write_candidate_output(output, payload, op)
    original_hash = __import__("scripts.migrate_figure_assets", fromlist=["file_sha256"]).file_sha256(output)

    async def invalid_describe(self, image_path, prompt):  # noqa: ANN001
        return "image_type: graph\nconfidence: unknown\nanalysis: null\n"

    monkeypatch.setattr("scripts.migrate_figure_assets.Settings", lambda: settings)
    monkeypatch.setattr("app.services.ollama.OllamaClient.describe_image", invalid_describe)
    monkeypatch.setattr(
        "sys.argv",
        [
            "migrate_figure_assets.py",
            "--document",
            "Doc.pdf",
            "--asset",
            Path(op["old_image"]).name,
            "--reanalyze-graphs",
            "--analyze-dry-run",
            "--candidate-output",
            str(output),
        ],
    )

    assert asyncio.run(main_async()) == 2
    assert __import__("scripts.migrate_figure_assets", fromlist=["file_sha256"]).file_sha256(output) == original_hash
    assert json.loads(failed_candidate_path(output).read_text(encoding="utf-8"))["candidate_status"] == "failed"


def test_information_quality_failure_returns_nonzero(monkeypatch, tmp_path: Path):
    settings, op, _payload = make_valid_candidate_case(tmp_path)
    output = tmp_path / "candidate.json"

    async def low_information(*args, **kwargs):  # noqa: ANN002, ANN003
        return {
            "schema_valid": True,
            "information_quality_passed": False,
            "validation_errors": ["insufficient visible information"],
            "raw_vision_text": "image_type: graph\nconfidence: 0.8",
            "vision_call_count": 1,
        }

    monkeypatch.setattr("scripts.migrate_figure_assets.Settings", lambda: settings)
    monkeypatch.setattr("scripts.migrate_figure_assets.analyze_reanalysis_operation", low_information)
    monkeypatch.setattr(
        "sys.argv",
        [
            "migrate_figure_assets.py",
            "--document",
            "Doc.pdf",
            "--asset",
            Path(op["old_image"]).name,
            "--reanalyze-graphs",
            "--analyze-dry-run",
            "--candidate-output",
            str(output),
        ],
    )

    assert asyncio.run(main_async()) == 2
    assert not output.exists()
    assert json.loads(failed_candidate_path(output).read_text(encoding="utf-8"))["information_quality_passed"] is False


def test_valid_candidate_output_returns_zero(monkeypatch, tmp_path: Path):
    settings, op, _payload = make_valid_candidate_case(tmp_path)
    output = tmp_path / "candidate.json"

    async def valid_describe(self, image_path, prompt):  # noqa: ANN001
        return valid_graph_vision()

    monkeypatch.setattr("scripts.migrate_figure_assets.Settings", lambda: settings)
    monkeypatch.setattr("app.services.ollama.OllamaClient.describe_image", valid_describe)
    monkeypatch.setattr(
        "sys.argv",
        [
            "migrate_figure_assets.py",
            "--document",
            "Doc.pdf",
            "--asset",
            Path(op["old_image"]).name,
            "--reanalyze-graphs",
            "--analyze-dry-run",
            "--candidate-output",
            str(output),
        ],
    )

    assert asyncio.run(main_async()) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["candidate_status"] == "valid"


def test_diagram_prompt_requires_numeric_confidence():
    prompt = prompt_for_classification("diagram")

    assert "ASCII English" in prompt
    assert "number from 0.0 to 1.0" in prompt


def test_stale_candidate_can_be_inspected_but_not_applied(tmp_path: Path):
    _settings, op, payload = make_valid_candidate_case(tmp_path)
    payload["classifier_version"] = "old-classifier"
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    inspected = load_candidate(path, op, allow_stale=True)

    assert inspected["candidate_status"] == "stale"
    assert inspected["apply_ready"] is False
    assert "classifier_version mismatch" in inspected["stale_reason"]
    with pytest.raises(ValueError, match="classifier_version mismatch"):
        load_candidate(path, op, require_apply_ready=True)


def test_candidate_effective_classification_mismatch_blocks_apply(tmp_path: Path):
    _settings, op, payload = make_valid_candidate_case(tmp_path)
    payload["effective_classification"] = "diagram"
    path = tmp_path / "wrong-type.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="effective classification"):
        load_candidate(path, op, require_apply_ready=True)


def test_semantic_grounding_failure_requires_review_and_blocks_apply(tmp_path: Path):
    settings, op, _payload = make_valid_candidate_case(tmp_path)
    candidate = validate_vision_candidate(valid_graph_vision(), classification="graph", settings=settings)
    candidate["semantic_grounding_passed"] = False
    candidate["trend_grounding_passed"] = True
    payload = make_candidate_payload(settings, op, candidate)
    path = tmp_path / "review.json"
    write_candidate_output(path, payload, op)

    assert payload["candidate_status"] == "review_required"
    assert payload["manual_review_required"] is True
    assert payload["apply_ready"] is False
    with pytest.raises(ValueError, match="not apply-ready"):
        load_candidate(path, op, require_apply_ready=True)


def test_dark_graph_uses_enhanced_input_without_changing_original(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path)
    image = tmp_path / "dark.jpeg"
    Image.new("CMYK", (500, 300), (255, 255, 255, 255)).save(image)
    original_sha = __import__("scripts.migrate_figure_assets", fromlist=["file_sha256"]).file_sha256(image)
    seen: list[Path] = []

    async def fake_describe(self, image_path, prompt):  # noqa: ANN001
        seen.append(Path(image_path))
        return valid_graph_vision().replace("series_count: 3", "series_count: 2")

    monkeypatch.setattr("app.services.ollama.OllamaClient.describe_image", fake_describe)
    op = {
        "old_image": str(image),
        "new_image": str(image),
        "classification": "graph",
        "dark_graph_candidate": True,
        "detected_text": "",
    }
    candidate = asyncio.run(
        analyze_reanalysis_operation(
            settings,
            __import__("app.services.ollama", fromlist=["OllamaClient"]).OllamaClient(settings),
            op,
        )
    )

    assert seen[0] != image
    assert seen[0].suffix == ".png"
    assert candidate["analysis_input_path"] == str(seen[0])
    assert candidate["analysis_input_sha256"]
    assert candidate["analysis_transform"]["autocontrast"] is True
    assert __import__("scripts.migrate_figure_assets", fromlist=["file_sha256"]).file_sha256(image) == original_sha


def test_p0305_plateau_only_description_requires_manual_review():
    candidate = validate_vision_candidate(
        valid_graph_vision()
        .replace("series_count: 3", "series_count: 2")
        .replace("trend: multiple trends: continuous rise, rise then plateau, diagonal reference line", "trend: upper rises; lower rises then plateaus"),
        classification="graph",
        settings=Settings(),
    )
    grounded = apply_operation_grounding(
        candidate,
        {
            "old_image": "Heriot-Watt_University_-_Well_Test_Analysis_p0305_fig02.jpeg",
            "classification": "graph",
            "dark_graph_candidate": True,
            "detected_text": "",
            "classification_override_reason": "dark_graph_precedence",
        },
    )

    assert grounded["trend_grounding_passed"] is False
    assert grounded["manual_review_reasons"] == ["lower series peak and decline not represented in model description"]


def test_candidate_provenance_constants_are_current():
    assert CANDIDATE_SCHEMA_VERSION == 2
    assert CLASSIFIER_VERSION == "dark-marker-series-v2"
    assert ANALYSIS_PROMPT_VERSION == "graph-grounding-v2"
    assert SERIALIZER_VERSION == "figure-note-v2"


def test_apply_reanalysis_creates_verified_backup_and_manifest(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "a8" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    stem = f"{doc_id}_p261_fig2"
    write_asset(settings, stem, note="image_type: log-log plot\nanalysis: x_axis y_axis psi hours\nold: keep me\n", image_color="black")

    async def fake_describe(self, image_path, prompt):  # noqa: ANN001
        return valid_graph_vision()

    monkeypatch.setattr("app.services.ollama.OllamaClient.describe_image", fake_describe)
    plan = build_plan(settings, document="Doc.pdf", limit=None, quarantine_logos=False, reanalyze_graphs=True, update_chroma=False)
    manifest_path = write_manifest(settings, plan, tmp_path / "manifest.json")

    asyncio.run(apply_plan(settings, plan, manifest_path))

    op = plan["operations"][0]
    note_text = Path(op["note_path"]).read_text(encoding="utf-8")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert op["backup_created"] is True
    assert op["backup_verified"] is True
    assert op["note_replaced"] is True
    assert Path(op["backup_path"]).is_file()
    assert f"legacy_note_backup: {op['backup_path']}" in note_text
    assert "backup_created" in manifest_text and "note_replaced" in manifest_text
    assert op["serialization_comparison"]["missing_serialized_fields"] == []
    assert "missing_serialized_fields" not in op


def test_candidate_input_apply_does_not_call_vision(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "ac" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    stem = f"{doc_id}_p261_fig2"
    write_asset(settings, stem, note="image_type: log-log plot\nanalysis: x_axis y_axis psi hours\n", image_color="black")
    plan = build_plan(settings, document="Doc.pdf", limit=None, quarantine_logos=False, reanalyze_graphs=True, update_chroma=False)
    op = plan["operations"][0]
    candidate = validate_vision_candidate(valid_graph_vision(), classification="graph", settings=settings)
    payload = make_candidate_payload(settings, op, candidate)

    async def fail_describe(self, image_path, prompt):  # noqa: ANN001
        raise AssertionError("Vision should not be called")

    monkeypatch.setattr("app.services.ollama.OllamaClient.describe_image", fail_describe)
    asyncio.run(apply_plan(settings, plan, tmp_path / "manifest.json", payload))

    assert op["note_replaced"] is True
    assert op["vision_call_count"] == 0
    assert op["post_write_validation_passed"] is True


def test_temporary_note_missing_confidence_prevents_replace(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "ae" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    stem = f"{doc_id}_p261_fig2"
    original = "image_type: log-log plot\nanalysis: original\n"
    write_asset(settings, stem, note=original, image_color="black")

    async def fake_describe(self, image_path, prompt):  # noqa: ANN001
        return valid_graph_vision()

    real_serialize = __import__("scripts.migrate_figure_assets", fromlist=["serialize_note_data"]).serialize_note_data

    def omit_confidence(metadata):  # noqa: ANN001
        return "\n".join(line for line in real_serialize(metadata).splitlines() if not line.startswith("confidence:"))

    def fail_replace(source, target):  # noqa: ANN001
        raise AssertionError("os.replace should not run when temp note is invalid")

    monkeypatch.setattr("app.services.ollama.OllamaClient.describe_image", fake_describe)
    monkeypatch.setattr("scripts.migrate_figure_assets.serialize_note_data", omit_confidence)
    monkeypatch.setattr("scripts.migrate_figure_assets.os.replace", fail_replace)
    plan = build_plan(settings, document="Doc.pdf", limit=None, quarantine_logos=False, reanalyze_graphs=True, update_chroma=False)
    asyncio.run(apply_plan(settings, plan))

    op = plan["operations"][0]
    assert op["status"] == "temporary_validation_failed"
    assert op["note_replaced"] is False
    assert op["serialization_comparison"]["missing_serialized_fields"] == ["confidence"]
    assert Path(op["note_path"]).read_text(encoding="utf-8") == original


def test_post_write_validation_failure_rolls_back(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "ad" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    stem = f"{doc_id}_p261_fig2"
    original = "image_type: log-log plot\nanalysis: original\n"
    write_asset(settings, stem, note=original, image_color="black")
    note_path = settings.figure_notes_dir / f"{stem}.md"
    original_sha = __import__("scripts.migrate_figure_assets", fromlist=["file_sha256"]).file_sha256(note_path)
    plan = build_plan(settings, document="Doc.pdf", limit=None, quarantine_logos=False, reanalyze_graphs=True, update_chroma=False)

    async def fake_describe(self, image_path, prompt):  # noqa: ANN001
        return valid_graph_vision()

    real_validate = __import__("scripts.migrate_figure_assets", fromlist=["validate_note_equivalence"]).validate_note_equivalence
    calls = {"count": 0}

    def flaky_validate(note_text, final_data):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 2:
            parsed, _errors, _equivalent, diff = real_validate(note_text, final_data)
            return parsed, ["forced post-write mismatch"], False, diff
        return real_validate(note_text, final_data)

    monkeypatch.setattr("app.services.ollama.OllamaClient.describe_image", fake_describe)
    monkeypatch.setattr("scripts.migrate_figure_assets.validate_note_equivalence", flaky_validate)
    asyncio.run(apply_plan(settings, plan, tmp_path / "manifest.json"))

    op = plan["operations"][0]
    assert op["rollback_attempted"] is True
    assert op["rollback_succeeded"] is True
    assert op["note_replaced"] is False
    assert Path(op["note_path"]).read_text(encoding="utf-8") == original
    assert __import__("scripts.migrate_figure_assets", fromlist=["file_sha256"]).file_sha256(Path(op["note_path"])) == original_sha


def test_post_write_missing_title_verified_rolls_back(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "af" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    stem = f"{doc_id}_p174_fig1"
    original = "image_type: diagram\nanalysis: original\n"
    write_asset(settings, stem, note=original, image_color="black")
    note_path = settings.figure_notes_dir / f"{stem}.md"
    original_sha = __import__("scripts.migrate_figure_assets", fromlist=["file_sha256"]).file_sha256(note_path)

    async def fake_describe(self, image_path, prompt):  # noqa: ANN001
        return "\n".join(
            [
                "image_type: diagram",
                "confidence: 0.8",
                "title: Flow Schedule",
                "components: Flow Rate (q)",
            ]
        )

    real_replace = __import__("scripts.migrate_figure_assets", fromlist=["os"]).os.replace

    def corrupt_after_replace(source, target):  # noqa: ANN001
        real_replace(source, target)
        path = Path(target)
        text = path.read_text(encoding="utf-8")
        path.write_text("\n".join(line for line in text.splitlines() if not line.startswith("title_verified:")), encoding="utf-8")

    monkeypatch.setattr("app.services.ollama.OllamaClient.describe_image", fake_describe)
    monkeypatch.setattr("scripts.migrate_figure_assets.os.replace", corrupt_after_replace)
    plan = build_plan(settings, document="Doc.pdf", limit=None, quarantine_logos=False, reanalyze_graphs=False, reanalyze_diagrams=True, update_chroma=False)
    asyncio.run(apply_plan(settings, plan))

    op = plan["operations"][0]
    assert op["post_write_comparison"]["missing_serialized_fields"] == ["title_verified"]
    assert op["post_write_bytes_equal_temporary"] is False
    assert op["rollback_attempted"] is True
    assert op["rollback_succeeded"] is True
    assert op["note_replaced"] is False
    assert __import__("scripts.migrate_figure_assets", fromlist=["file_sha256"]).file_sha256(Path(op["note_path"])) == original_sha


def test_apply_reanalysis_preserves_note_when_schema_invalid(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "a9" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    stem = f"{doc_id}_p261_fig2"
    original = "image_type: log-log plot\nanalysis: original\n"
    write_asset(settings, stem, note=original, image_color="black")

    async def bad_describe(self, image_path, prompt):  # noqa: ANN001
        return "image_type: Log-Log Plot\nconfidence: 확인할 수 없음\nanalysis: title only\n"

    monkeypatch.setattr("app.services.ollama.OllamaClient.describe_image", bad_describe)
    plan = build_plan(settings, document="Doc.pdf", limit=None, quarantine_logos=False, reanalyze_graphs=True, update_chroma=False)
    asyncio.run(apply_plan(settings, plan, tmp_path / "manifest.json"))

    op = plan["operations"][0]
    assert Path(op["note_path"]).read_text(encoding="utf-8") == original
    assert op["note_replaced"] is False
    assert op["existing_note_preserved"] is True
    assert op["status"] == "rejected_invalid_schema"


def test_apply_reanalysis_preserves_note_when_backup_missing(monkeypatch, tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "aa" * 32
    write_metadata(settings, doc_id, "Doc.pdf")
    stem = f"{doc_id}_p261_fig2"
    original = "image_type: log-log plot\nanalysis: original\n"
    write_asset(settings, stem, note=original, image_color="black")

    async def fake_describe(self, image_path, prompt):  # noqa: ANN001
        return valid_graph_vision()

    def no_copy(source, backup):  # noqa: ANN001
        return backup

    monkeypatch.setattr("app.services.ollama.OllamaClient.describe_image", fake_describe)
    monkeypatch.setattr("scripts.migrate_figure_assets.shutil.copy2", no_copy)
    plan = build_plan(settings, document="Doc.pdf", limit=None, quarantine_logos=False, reanalyze_graphs=True, update_chroma=False)
    asyncio.run(apply_plan(settings, plan, tmp_path / "manifest.json"))

    op = plan["operations"][0]
    assert Path(op["note_path"]).read_text(encoding="utf-8") == original
    assert op["backup_created"] is False
    assert op["note_replaced"] is False
    assert op["status"] == "backup_failed"


def test_manifest_restore_round_trip(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "f" * 64
    stem = f"{doc_id}_p1_fig1"
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, stem, note="document_name: old.md\n")
    plan = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=False,
        update_chroma=False,
    )
    manifest = write_manifest(settings, plan, tmp_path / "manifest.json")
    op = plan["operations"][0]
    Path(op["old_image"]).rename(op["new_image"])
    Path(op["old_note"]).rename(op["new_note"])

    restore_manifest(manifest)

    assert Path(op["old_image"]).exists()
    assert Path(op["old_note"]).exists()
    assert not Path(op["new_image"]).exists()


def test_chroma_count_only_when_requested(tmp_path: Path):
    settings = make_settings(tmp_path)
    doc_id = "1" * 64
    write_metadata(settings, doc_id, "Doc.pdf")
    write_asset(settings, f"{doc_id}_p1_fig1")

    off = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=False,
        update_chroma=False,
    )
    on = build_plan(
        settings,
        document=None,
        limit=None,
        quarantine_logos=False,
        reanalyze_graphs=False,
        update_chroma=True,
    )

    assert off["stats"]["would_update_chroma"] == 0
    assert on["stats"]["would_update_chroma"] == 1


def test_restore_cli_requires_apply(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("sys.argv", ["migrate_figure_assets.py", "--restore", str(tmp_path / "x.json")])

    assert asyncio.run(main_async()) == 2
