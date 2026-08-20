#!/usr/bin/env python3
"""Docker-free MRST/CO2 Agent end-to-end test.

This test does not require Ollama. It starts a real FastAPI process with temporary
data/workspace directories, creates a representative Johansen-style CSV through
the Agent API, approves the MRST analysis, validates CSV/PNG/Markdown outputs and
downloads, restarts the backend, and confirms task/conversation persistence.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
BACKEND_DIR = ROOT / "backend"

INPUT_PATH = "co2_result.csv"
SUMMARY_PATH = "results/e2e_mrst_analysis.csv"
CHART_PATH = "results/e2e_mrst_analysis.png"
REPORT_PATH = "results/e2e_mrst_analysis.md"
EXPECTED_OUTPUTS = {SUMMARY_PATH, CHART_PATH, REPORT_PATH}
INPUT_CSV = (
    "srco2,trapped_ratio,free_ratio,total_storage_mt\n"
    "0.10,94.9,5.1,5.2\n"
    "0.20,88.7,11.3,5.2\n"
    "0.30,81.2,18.8,5.2\n"
    "0.40,71.8,28.2,5.2\n"
)


def start_backend(
    port: int,
    data_dir: Path,
    workspace_dir: Path,
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    env = os.environ.copy()
    env["DATA_DIR"] = str(data_dir)
    env["AGENT_WORKSPACE_DIR"] = str(workspace_dir)
    env.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    print("$", " ".join(command), f"(cwd={BACKEND_DIR})", flush=True)
    log_handle = log_path.open("ab")
    process = subprocess.Popen(
        command,
        cwd=BACKEND_DIR,
        env=env,
        text=False,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_handle


def stop_backend(process: subprocess.Popen[bytes], log_handle: BinaryIO) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    log_handle.close()


def wait_for_agent_api(
    base_url: str,
    process: subprocess.Popen[bytes],
    timeout: int = 60,
) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    last_error = ""
    with httpx.Client(trust_env=False, timeout=5) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Backend exited early with code {process.returncode}.")
            try:
                response = client.get(
                    f"{base_url}/agent/workspace",
                    params={"path": "."},
                )
                if response.status_code == 200:
                    return
                last_error = response.text
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(0.25)
    raise RuntimeError(f"Agent API did not become ready: {last_error}")


def request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    import httpx

    with httpx.Client(trust_env=False, timeout=30) as client:
        response = client.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()


def wait_for_task(base_url: str, task_id: str, timeout: int = 60) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = request_json("GET", f"{base_url}/agent/tasks/{task_id}")
        if task["status"] in {"completed", "failed", "canceled"}:
            return task
        time.sleep(0.1)
    raise RuntimeError(f"Agent task did not finish within {timeout}s: {task_id}")


def create_and_execute(
    base_url: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    planned = request_json("POST", f"{base_url}/agent/plan", json=payload)
    assert planned["status"] == "planned", planned
    assert planned["requires_approval"] is True, planned
    started = request_json(
        "POST",
        f"{base_url}/agent/tasks/{planned['task_id']}/execute",
        json={"approved": True},
    )
    assert started["status"] == "running", started
    completed = wait_for_task(base_url, planned["task_id"])
    assert completed["status"] == "completed", completed
    return planned, completed


def validate_previews_and_downloads(base_url: str) -> None:
    import httpx

    summary = request_json(
        "GET",
        f"{base_url}/agent/files/preview",
        params={"path": SUMMARY_PATH},
    )
    assert summary["kind"] == "csv", summary
    assert summary["columns"] == [
        "source_file",
        "row_index",
        "srco2",
        "time_years",
        "trapped_ratio_pct",
        "free_ratio_pct",
        "trapped_amount",
        "free_amount",
        "total_storage_value",
        "storage_unit",
        "calculation_basis",
    ], summary
    assert len(summary["rows"]) == 4, summary
    assert float(summary["rows"][0][2]) == 0.1, summary
    assert float(summary["rows"][0][4]) == 94.9, summary
    assert float(summary["rows"][-1][5]) == 28.2, summary

    chart = request_json(
        "GET",
        f"{base_url}/agent/files/preview",
        params={"path": CHART_PATH},
    )
    assert chart == {"path": CHART_PATH, "kind": "image"}, chart

    report = request_json(
        "GET",
        f"{base_url}/agent/files/preview",
        params={"path": REPORT_PATH},
    )
    assert report["kind"] == "text", report
    assert "# MRST / CO2 Storage Analysis" in report["content"], report
    assert "Pearson r=" in report["content"], report
    assert "does not establish causality" in report["content"], report

    with httpx.Client(trust_env=False, timeout=30) as client:
        for path in EXPECTED_OUTPUTS:
            response = client.get(
                f"{base_url}/agent/files/content",
                params={"path": path, "download": "true"},
            )
            response.raise_for_status()
            assert response.content, path
            assert "attachment" in response.headers.get("content-disposition", ""), path
            if path.endswith(".png"):
                assert response.content.startswith(b"\x89PNG\r\n\x1a\n"), path


def run_scenario(base_url: str) -> tuple[str, str, str]:
    import httpx

    conversation = request_json(
        "POST",
        f"{base_url}/agent/conversations",
        json={"title": "MRST CO2 실제 E2E"},
    )
    conversation_id = conversation["conversation_id"]

    _, created = create_and_execute(
        base_url,
        {
            "request": "실제 E2E 입력 CSV 파일을 생성해줘.",
            "conversation_id": conversation_id,
            "output_path": INPUT_PATH,
            "content": INPUT_CSV,
            "permission_level": 2,
        },
    )
    assert INPUT_PATH in created["created_files"], created

    columns = request_json(
        "GET",
        f"{base_url}/agent/csv-columns",
        params={"path": INPUT_PATH},
    )
    assert columns["columns"] == [
        "srco2",
        "trapped_ratio",
        "free_ratio",
        "total_storage_mt",
    ], columns

    planned, analyzed = create_and_execute(
        base_url,
        {
            "request": "co2_result.csv의 srCO2 조건별 trapped/free 비율을 MRST CO2 전용 분석해줘.",
            "conversation_id": conversation_id,
            "target_path": INPUT_PATH,
            "output_path": REPORT_PATH,
            "analysis_profile": "mrst_co2",
            "permission_level": 3,
        },
    )
    run_action = next(
        action for action in planned["actions"] if action["tool"] == "run_python"
    )
    assert set(run_action["arguments"]["expected_outputs"]) == EXPECTED_OUTPUTS
    assert analyzed["validation_passed"] is True, analyzed
    assert EXPECTED_OUTPUTS.issubset(set(analyzed["created_files"])), analyzed
    assert INPUT_PATH in analyzed["read_files"], analyzed
    assert INPUT_PATH not in analyzed["modified_files"], analyzed

    validate_previews_and_downloads(base_url)
    with httpx.Client(trust_env=False, timeout=30) as client:
        input_response = client.get(
            f"{base_url}/agent/files/content",
            params={"path": INPUT_PATH},
        )
    input_response.raise_for_status()
    assert input_response.text == INPUT_CSV

    runs = request_json("GET", f"{base_url}/agent/runs", params={"limit": 10})
    run_ids = {run["task_id"] for run in runs["runs"]}
    assert {created["task_id"], analyzed["task_id"]}.issubset(run_ids), runs

    detail = request_json(
        "GET",
        f"{base_url}/agent/conversations/{conversation_id}",
    )
    assert detail["task_count"] == 2, detail
    assert [task["task_id"] for task in detail["tasks"]] == [
        created["task_id"],
        analyzed["task_id"],
    ], detail
    return conversation_id, created["task_id"], analyzed["task_id"]


def validate_after_restart(
    base_url: str,
    conversation_id: str,
    create_task_id: str,
    analysis_task_id: str,
) -> None:
    detail = request_json(
        "GET",
        f"{base_url}/agent/conversations/{conversation_id}",
    )
    assert detail["task_count"] == 2, detail
    assert [task["status"] for task in detail["tasks"]] == [
        "completed",
        "completed",
    ], detail
    assert [task["task_id"] for task in detail["tasks"]] == [
        create_task_id,
        analysis_task_id,
    ], detail

    restored = request_json(
        "GET",
        f"{base_url}/agent/tasks/{analysis_task_id}",
    )
    assert restored["validation_passed"] is True, restored
    assert EXPECTED_OUTPUTS.issubset(set(restored["created_files"])), restored
    validate_previews_and_downloads(base_url)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    base_url = f"http://127.0.0.1:{args.port}/api"

    with tempfile.TemporaryDirectory(prefix="ai-brain-mrst-e2e-") as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        workspace_dir = root / "workspace"
        log_path = root / "backend.log"
        process, log_handle = start_backend(
            args.port,
            data_dir,
            workspace_dir,
            log_path,
        )
        try:
            wait_for_agent_api(base_url, process)
            conversation_id, create_task_id, analysis_task_id = run_scenario(base_url)

            stop_backend(process, log_handle)
            process, log_handle = start_backend(
                args.port,
                data_dir,
                workspace_dir,
                log_path,
            )
            wait_for_agent_api(base_url, process)
            validate_after_restart(
                base_url,
                conversation_id,
                create_task_id,
                analysis_task_id,
            )
        except Exception:
            if log_path.is_file():
                print("\n--- backend log ---", file=sys.stderr)
                print(log_path.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
            raise
        finally:
            stop_backend(process, log_handle)

    print("MRST/CO2 Agent E2E passed, including restart persistence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
