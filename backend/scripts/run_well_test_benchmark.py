from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.benchmark_evaluator import (  # noqa: E402
    STRICT_REFUSAL,
    evaluate_benchmark_answer,
)


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 900.0,
) -> Any:
    body = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        url=url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Cannot reach {url}: {exc.reason}"
        ) from exc


def filter_items(
    items: list[dict[str, Any]],
    *,
    ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    selected = [
        item
        for item in items
        if not ids or str(item.get("id")) in ids
    ]
    if limit > 0:
        selected = selected[:limit]
    return selected


def find_new_agent_run(
    agent_runs_dir: Path,
    before_files: set[Path],
    *,
    benchmark_id: str,
    question: str,
    timeout_seconds: float = 10.0,
) -> tuple[Path, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        current = set(agent_runs_dir.glob("*.json"))
        candidates = sorted(
            current - before_files,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        for path in candidates:
            try:
                record = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue

            if (
                record.get("benchmark_id") == benchmark_id
                and record.get("question") == question
            ):
                return path, record

        time.sleep(0.1)

    raise RuntimeError(
        f"Agent run JSON not found for {benchmark_id}."
    )


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def category_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(
            str(result.get("category") or "unknown"),
            [],
        ).append(result)

    summary: dict[str, Any] = {}
    for category, rows in sorted(grouped.items()):
        final_passes = sum(
            1
            for row in rows
            if row.get("final_benchmark_passed") is True
        )
        summary[category] = {
            "total": len(rows),
            "final_passed": final_passes,
            "final_pass_rate": ratio(
                final_passes,
                len(rows),
            ),
        }
    return summary


def build_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [
        row for row in results
        if row.get("infrastructure_error") is None
    ]
    initial_benchmark_passes = sum(
        1
        for row in completed
        if row.get("initial_benchmark_passed") is True
    )
    final_benchmark_passes = sum(
        1
        for row in completed
        if row.get("final_benchmark_passed") is True
    )
    initial_validator_passes = sum(
        1
        for row in completed
        if row.get("initial_validator_passed") is True
    )
    final_validator_passes = sum(
        1
        for row in completed
        if row.get("final_validator_passed") is True
    )

    initial_benchmark_failures = [
        row
        for row in completed
        if row.get("initial_benchmark_passed") is False
    ]
    rewrite_successes = sum(
        1
        for row in initial_benchmark_failures
        if row.get("final_benchmark_passed") is True
    )
    validator_detections = sum(
        1
        for row in initial_benchmark_failures
        if row.get("initial_validator_passed") is False
    )

    initially_correct = [
        row
        for row in completed
        if row.get("initial_benchmark_passed") is True
    ]
    validator_false_positives = sum(
        1
        for row in initially_correct
        if row.get("initial_validator_passed") is False
    )

    refusal_rows = [
        row
        for row in completed
        if row.get("expected_behavior") == "refuse"
    ]
    exact_refusals = sum(
        1
        for row in refusal_rows
        if row.get("final_answer") == STRICT_REFUSAL
    )

    page_rows = [
        row
        for row in completed
        if row.get("preferred_page_hit") is not None
    ]
    page_hits = sum(
        1
        for row in page_rows
        if row.get("preferred_page_hit") is True
    )

    document_rows = [
        row
        for row in completed
        if row.get("expected_document_hit") is not None
    ]
    document_hits = sum(
        1
        for row in document_rows
        if row.get("expected_document_hit") is True
    )

    return {
        "questions_total": len(results),
        "questions_completed": len(completed),
        "infrastructure_errors": (
            len(results) - len(completed)
        ),
        "initial_benchmark_pass_rate": ratio(
            initial_benchmark_passes,
            len(completed),
        ),
        "final_benchmark_pass_rate": ratio(
            final_benchmark_passes,
            len(completed),
        ),
        "initial_validator_pass_rate": ratio(
            initial_validator_passes,
            len(completed),
        ),
        "final_validator_pass_rate": ratio(
            final_validator_passes,
            len(completed),
        ),
        "rewrite_success_rate": ratio(
            rewrite_successes,
            len(initial_benchmark_failures),
        ),
        "validator_detection_rate": ratio(
            validator_detections,
            len(initial_benchmark_failures),
        ),
        "validator_false_positive_rate": ratio(
            validator_false_positives,
            len(initially_correct),
        ),
        "exact_refusal_rate": ratio(
            exact_refusals,
            len(refusal_rows),
        ),
        "preferred_page_hit_rate": ratio(
            page_hits,
            len(page_rows),
        ),
        "expected_document_hit_rate": ratio(
            document_hits,
            len(document_rows),
        ),
        "average_attempts": safe_mean(
            [
                float(row["attempts"])
                for row in completed
            ]
        ),
        "average_retrieval_seconds": safe_mean(
            [
                float(row["retrieval_seconds"])
                for row in completed
            ]
        ),
        "average_generation_seconds": safe_mean(
            [
                float(row["generation_seconds"])
                for row in completed
            ]
        ),
        "average_total_seconds": safe_mean(
            [
                float(row["total_seconds"])
                for row in completed
            ]
        ),
        "category_metrics": category_summary(completed),
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(
    path: Path,
    results: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "id",
        "category",
        "model",
        "expected_behavior",
        "initial_validator_passed",
        "final_validator_passed",
        "initial_benchmark_passed",
        "final_benchmark_passed",
        "rewrite_success",
        "attempts",
        "final_status",
        "expected_document_hit",
        "preferred_page_hit",
        "source_pages",
        "retrieval_seconds",
        "generation_seconds",
        "total_seconds",
        "initial_required_failures",
        "initial_forbidden_hits",
        "final_required_failures",
        "final_forbidden_hits",
        "infrastructure_error",
        "question",
        "final_answer",
        "agent_run_file",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    key: csv_value(result.get(key))
                    for key in columns
                }
            )


def print_rate(label: str, value: float | None) -> None:
    if value is None:
        print(f"{label}=n/a")
    else:
        print(f"{label}={value:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic Well Test agent benchmark "
            "against the local FastAPI backend."
        )
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/api",
    )
    parser.add_argument(
        "--benchmark",
        default=str(
            PROJECT_ROOT
            / "evaluation"
            / "well_test_agent_benchmark.json"
        ),
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated benchmark IDs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--fail-on-benchmark-failure",
        action="store_true",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark).resolve()
    benchmark_items = read_json(benchmark_path)
    if not isinstance(benchmark_items, list):
        raise RuntimeError(
            "Benchmark JSON must contain a list."
        )

    requested_ids = {
        item.strip()
        for item in args.ids.split(",")
        if item.strip()
    }
    items = filter_items(
        benchmark_items,
        ids=requested_ids,
        limit=args.limit,
    )
    if not items:
        raise RuntimeError(
            "No benchmark questions were selected."
        )

    print(f"benchmark={benchmark_path}")
    print(f"model={args.model}")
    print(f"questions={len(items)}")

    for item in items:
        print(
            f"  {item['id']} "
            f"[{item.get('category', 'unknown')}] "
            f"{item['question']}"
        )

    if args.dry_run:
        print("DRY_RUN=True")
        return 0

    api_base = args.api_url.rstrip("/")
    health = http_json(
        "GET",
        f"{api_base}/health",
        timeout=30.0,
    )
    print(
        "backend_status="
        f"{health.get('status', 'unknown')}"
    )

    settings = get_settings()
    agent_runs_dir = Path(settings.agent_runs_dir)
    evaluation_dir = Path(settings.evaluation_dir)
    agent_runs_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    run_id = utc_run_id()
    results: list[dict[str, Any]] = []
    overall_started = time.perf_counter()

    for index, item in enumerate(items, start=1):
        item_id = str(item["id"])
        question = str(item["question"])
        print(
            f"\n[{index}/{len(items)}] "
            f"{item_id}: {question}"
        )

        before_files = set(
            agent_runs_dir.glob("*.json")
        )
        request_payload: dict[str, Any] = {
            "question": question,
            "model": args.model,
            "benchmark_id": item_id,
        }
        if args.top_k is not None:
            request_payload["top_k"] = args.top_k

        result: dict[str, Any] = {
            "id": item_id,
            "category": item.get("category"),
            "question": question,
            "model": args.model,
            "expected_behavior": item.get(
                "expected_behavior"
            ),
            "infrastructure_error": None,
        }

        try:
            response = http_json(
                "POST",
                f"{api_base}/chat",
                payload=request_payload,
                timeout=args.timeout,
            )
            run_path, record = find_new_agent_run(
                agent_runs_dir,
                before_files,
                benchmark_id=item_id,
                question=question,
            )

            attempts = record.get("attempts") or []
            initial_answer = (
                str(attempts[0].get("answer") or "")
                if attempts
                else ""
            )
            final_answer = str(
                response.get("answer") or ""
            )
            sources = (
                record.get("retrieved_sources") or []
            )

            initial_evaluation = (
                evaluate_benchmark_answer(
                    item,
                    initial_answer,
                    sources=sources,
                )
            )
            final_evaluation = (
                evaluate_benchmark_answer(
                    item,
                    final_answer,
                    sources=sources,
                )
            )

            generation_seconds = sum(
                float(attempt.get("elapsed_seconds") or 0.0)
                for attempt in attempts
            )
            initial_validator_passed = (
                attempts[0].get("validation_passed")
                if attempts
                else None
            )
            final_validator_passed = record.get(
                "final_passed"
            )

            result.update(
                {
                    "initial_answer": initial_answer,
                    "final_answer": final_answer,
                    "sources": sources,
                    "source_pages": (
                        final_evaluation.source_pages
                    ),
                    "initial_validator_passed": (
                        initial_validator_passed
                    ),
                    "final_validator_passed": (
                        final_validator_passed
                    ),
                    "initial_benchmark_passed": (
                        initial_evaluation.passed
                    ),
                    "final_benchmark_passed": (
                        final_evaluation.passed
                    ),
                    "rewrite_success": (
                        not initial_evaluation.passed
                        and final_evaluation.passed
                    ),
                    "attempts": len(attempts),
                    "final_status": record.get(
                        "final_status"
                    ),
                    "expected_document_hit": (
                        final_evaluation
                        .expected_document_hit
                    ),
                    "preferred_page_hit": (
                        final_evaluation
                        .preferred_page_hit
                    ),
                    "retrieval_seconds": float(
                        record.get(
                            "retrieval_elapsed_seconds"
                        )
                        or 0.0
                    ),
                    "generation_seconds": (
                        generation_seconds
                    ),
                    "total_seconds": float(
                        record.get(
                            "total_elapsed_seconds"
                        )
                        or 0.0
                    ),
                    "initial_required_failures": (
                        initial_evaluation
                        .required_failures
                    ),
                    "initial_forbidden_hits": (
                        initial_evaluation
                        .forbidden_hits
                    ),
                    "final_required_failures": (
                        final_evaluation
                        .required_failures
                    ),
                    "final_forbidden_hits": (
                        final_evaluation
                        .forbidden_hits
                    ),
                    "agent_run_file": str(run_path),
                }
            )

            print(
                "  initial="
                f"{initial_evaluation.passed} "
                "final="
                f"{final_evaluation.passed} "
                f"attempts={len(attempts)} "
                "preferred_page="
                f"{final_evaluation.preferred_page_hit}"
            )
        except Exception as exc:
            result.update(
                {
                    "infrastructure_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "initial_validator_passed": None,
                    "final_validator_passed": None,
                    "initial_benchmark_passed": None,
                    "final_benchmark_passed": None,
                    "rewrite_success": False,
                    "attempts": 0,
                    "final_status": "infrastructure_error",
                    "expected_document_hit": None,
                    "preferred_page_hit": None,
                    "source_pages": [],
                    "retrieval_seconds": 0.0,
                    "generation_seconds": 0.0,
                    "total_seconds": 0.0,
                    "initial_required_failures": [],
                    "initial_forbidden_hits": [],
                    "final_required_failures": [],
                    "final_forbidden_hits": [],
                    "final_answer": "",
                    "agent_run_file": "",
                }
            )
            print(
                "  ERROR: "
                f"{result['infrastructure_error']}"
            )

        results.append(result)

        if args.pause_seconds > 0:
            time.sleep(args.pause_seconds)

    summary = build_summary(results)
    elapsed = time.perf_counter() - overall_started

    payload = {
        "benchmark_name": "well_test_agent_benchmark",
        "benchmark_file": str(benchmark_path),
        "run_id": run_id,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": args.model,
        "api_url": api_base,
        "question_count": len(items),
        "wall_clock_seconds": elapsed,
        "summary": summary,
        "results": results,
    }

    json_path = (
        evaluation_dir
        / f"well_test_benchmark_{run_id}.json"
    )
    csv_path = (
        evaluation_dir
        / f"well_test_benchmark_{run_id}.csv"
    )
    latest_json_path = (
        evaluation_dir
        / "well_test_benchmark_latest.json"
    )
    latest_csv_path = (
        evaluation_dir
        / "well_test_benchmark_latest.csv"
    )

    write_json(json_path, payload)
    write_json(latest_json_path, payload)
    write_csv(csv_path, results)
    write_csv(latest_csv_path, results)

    print("\nBENCHMARK_COMPLETED=True")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print_rate(
        "initial_benchmark_pass_rate",
        summary["initial_benchmark_pass_rate"],
    )
    print_rate(
        "final_benchmark_pass_rate",
        summary["final_benchmark_pass_rate"],
    )
    print_rate(
        "rewrite_success_rate",
        summary["rewrite_success_rate"],
    )
    print_rate(
        "validator_detection_rate",
        summary["validator_detection_rate"],
    )
    print_rate(
        "exact_refusal_rate",
        summary["exact_refusal_rate"],
    )
    print_rate(
        "preferred_page_hit_rate",
        summary["preferred_page_hit_rate"],
    )
    print(
        "average_attempts="
        f"{summary['average_attempts'] or 0.0:.3f}"
    )

    has_infrastructure_error = (
        summary["infrastructure_errors"] > 0
    )
    has_benchmark_failure = any(
        row.get("final_benchmark_passed") is False
        for row in results
    )

    if has_infrastructure_error:
        return 2
    if (
        args.fail_on_benchmark_failure
        and has_benchmark_failure
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
