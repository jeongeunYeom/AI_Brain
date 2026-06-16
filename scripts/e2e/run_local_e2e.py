#!/usr/bin/env python3
"""Docker-free E2E smoke test.

Starts the backend exactly like local development (`uvicorn app.main:app --reload`),
checks health, uploads a generated PDF, asks a question, verifies ChromaDB files,
restarts the backend, asks again without reuploading, and confirms citation metadata
is still returned from the same document/page.
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
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
BACKEND_DIR = ROOT / "backend"
DATA_DIR = ROOT / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
sys.path.insert(0, str(SCRIPT_DIR))

from run_e2e import ask, make_pdf, upload_pdf, wait_for_health  # noqa: E402


def start_backend(port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.setdefault("DATA_DIR", str(DATA_DIR))
    env.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    command = ["uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", str(port)]
    print("$", " ".join(command), f"(cwd={BACKEND_DIR})", flush=True)
    return subprocess.Popen(
        command,
        cwd=BACKEND_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop_backend(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def assert_same_source(first: dict[str, Any], second: dict[str, Any]) -> None:
    assert first["sources"], first
    assert second["sources"], second
    first_source = first["sources"][0]
    second_source = second["sources"][0]
    assert first_source["document"] == second_source["document"], (first_source, second_source)
    assert first_source["page"] == second_source["page"] == 1, (first_source, second_source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    base_url = f"http://127.0.0.1:{args.port}/api"

    process = start_backend(args.port)
    try:
        wait_for_health(base_url)
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "darcy_local_test.pdf"
            make_pdf(pdf_path)
            upload = upload_pdf(base_url, pdf_path)
            assert upload["document"]["chunks"] >= 1, upload

            vector_files_before = sorted(p.relative_to(VECTOR_DB_DIR).as_posix() for p in VECTOR_DB_DIR.rglob("*") if p.is_file())
            assert vector_files_before, "Expected ChromaDB files under /data/vector_db after upload."

            first_chat = ask(base_url)
            assert first_chat["sources"][0]["document"].endswith("darcy_local_test.pdf"), first_chat
            assert first_chat["sources"][0]["page"] == 1, first_chat

            stop_backend(process)
            time.sleep(2)
            process = start_backend(args.port)
            wait_for_health(base_url)

            vector_files_after = sorted(p.relative_to(VECTOR_DB_DIR).as_posix() for p in VECTOR_DB_DIR.rglob("*") if p.is_file())
            assert vector_files_after == vector_files_before, "Vector DB file set changed unexpectedly after backend restart."

            second_chat = ask(base_url)
            assert_same_source(first_chat, second_chat)
            print("Docker-free E2E passed with source:", second_chat["sources"][0])
    finally:
        stop_backend(process)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
