from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.benchmark_comparison import (  # noqa: E402
    build_benchmark_comparison,
)


def read_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark 결과 JSON 형식이 아닙니다: {path}")
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine multiple Well Test benchmark runs into a paper-ready table."
    )
    parser.add_argument("runs", nargs="+", help="Benchmark result JSON paths.")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    paths = [Path(value).resolve() for value in args.runs]
    comparison = build_benchmark_comparison(
        read_payload(path) for path in paths
    )
    settings = get_settings()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(settings.evaluation_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"benchmark_comparison_{run_id}.json"
    csv_path = output_dir / f"benchmark_comparison_{run_id}.csv"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(csv_path, comparison["rows"])

    print("BENCHMARK_COMPARISON_COMPLETED=True")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
