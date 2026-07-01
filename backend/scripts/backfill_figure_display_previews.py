from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings
from app.services.figure_preview import (  # noqa: E402
    FigurePreviewService,
    SUPPORTED_IMAGE_SUFFIXES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-generate display-only Figure previews. Defaults to dry-run.",
    )
    parser.add_argument("--document-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings()
    service = FigurePreviewService(
        settings.data_dir / "figure_display_previews",
        settings.data_dir / "figure_display_overrides.json",
        raw_dir=settings.raw_dir,
        metadata_dir=settings.metadata_dir,
    )
    document_name = None
    if args.document_id:
        metadata_path = settings.metadata_dir / f"{args.document_id}.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                document_name = str(metadata.get("filename") or "") or None
            except (OSError, json.JSONDecodeError):
                document_name = None
    assets = sorted(
        path
        for path in settings.figures_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and (not args.document_id or path.name.startswith(args.document_id))
    )
    if args.limit is not None:
        assets = assets[: max(0, args.limit)]

    dry_run = args.dry_run or not args.overwrite
    generated = 0
    failed = 0
    source_counts = {
        "pdf_page_crop": 0,
        "extracted_image": 0,
    }
    for asset in assets:
        if dry_run:
            source_type = service.preferred_source_type(
                asset,
                document_id=args.document_id,
                document_name=document_name,
            )
            if source_type in source_counts:
                source_counts[source_type] += 1
            print(f"WOULD GENERATE: {asset.name} source={source_type}")
            continue
        result = service.get_or_create_preview(
            asset,
            document_id=args.document_id,
            document_name=document_name,
            overwrite=True,
        )
        if result is None:
            failed += 1
        else:
            generated += 1
            source_counts[result.source_type] += 1
            print(f"GENERATED: {result.name} source={result.source_type}")

    print(
        f"scanned={len(assets)} dry_run={str(dry_run).lower()} "
        f"generated={generated} pdf_page_crop={source_counts['pdf_page_crop']} "
        f"extracted_image_fallback={source_counts['extracted_image']} "
        f"failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
