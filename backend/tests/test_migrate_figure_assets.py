import asyncio
import json
from pathlib import Path

from PIL import Image, ImageDraw

from app.core.config import Settings
from scripts.migrate_figure_assets import (
    build_plan,
    export_review_report,
    group_summaries,
    main_async,
    prompt_for_classification,
    restore_manifest,
    safe_document_stem,
    target_stem,
    update_note_text,
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
    assert plan["stats"]["final_logo_candidates"] == 3
    assert plan["stats"]["graph_candidates"] == 1
    assert plan["stats"]["would_quarantine"] == 3
    assert plan["stats"]["would_reanalyze"] == 1
    assert any(op["quarantine"] for op in plan["operations"])
    assert not [op for op in plan["operations"] if op["classification"] == "graph" and op["quarantine"]]
    assert all("Doc_p000" in Path(op["new_image"]).name for op in plan["operations"] if op["quarantine"])


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
