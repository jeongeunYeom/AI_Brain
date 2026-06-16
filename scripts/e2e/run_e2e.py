#!/usr/bin/env python3
"""End-to-end smoke test for the Petroleum Engineering AI Agent.

The script validates the operational requirements that need real services:
- docker compose backend/frontend health
- persistent /data/vector_db across backend restart
- upload one PDF, restart server, then ask without reuploading
- citation document/page metadata in chat response
- image graph upload through the vision endpoint
- friendly Ollama error handling when Ollama/model is unavailable

Usage:
  python scripts/e2e/run_e2e.py --base-url http://localhost:8000/api --use-compose
  python scripts/e2e/run_e2e.py --skip-vision
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=check)


def wait_for_health(base_url: str, timeout: int = 120) -> dict[str, Any]:
    import httpx

    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=10)
            if response.status_code == 200:
                return response.json()
            last_error = response.text
        except Exception as exc:  # noqa: BLE001 - printed for diagnostics
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Backend did not become healthy within {timeout}s: {last_error}")


def make_pdf(path: Path) -> None:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for the E2E PDF fixture. Install backend requirements first.") from exc

    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Petroleum engineering test document.\n"
        "Darcy law states that flow rate increases with permeability and pressure gradient.\n"
        "Reservoir permeability is measured in millidarcy.\n"
        "This sentence is on page 1 for citation validation."
    )
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()


def make_png(path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for the E2E image fixture. Install backend requirements first.") from exc

    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.line((80, 340, 580, 340), fill="black", width=3)
    draw.line((80, 340, 80, 60), fill="black", width=3)
    draw.line((100, 310, 220, 250, 340, 190, 460, 130, 560, 90), fill="blue", width=4)
    draw.text((250, 370), "Flow rate q (STB/day)", fill="black")
    draw.text((10, 40), "Pressure (psi)", fill="black")
    draw.text((390, 95), "increasing trend", fill="blue")
    image.save(path)


def upload_pdf(base_url: str, pdf_path: Path) -> dict[str, Any]:
    import httpx

    with pdf_path.open("rb") as file:
        response = httpx.post(
            f"{base_url}/documents/upload",
            params={"analyze_figures": "false"},
            files={"file": (pdf_path.name, file, "application/pdf")},
            timeout=240,
        )
    response.raise_for_status()
    return response.json()


def ask(base_url: str) -> dict[str, Any]:
    import httpx

    response = httpx.post(
        f"{base_url}/chat",
        json={"question": "According to the uploaded document, how does Darcy law relate flow rate to permeability?"},
        timeout=240,
    )
    response.raise_for_status()
    return response.json()


def analyze_image(base_url: str, image_path: Path) -> dict[str, Any]:
    import httpx

    with image_path.open("rb") as file:
        response = httpx.post(
            f"{base_url}/vision/analyze",
            files={"file": (image_path.name, file, "image/png")},
            timeout=240,
        )
    response.raise_for_status()
    return response.json()


def assert_friendly_ollama_error_shape(base_url: str, image_path: Path) -> None:
    import httpx

    bad_base_url = os.getenv("E2E_BAD_OLLAMA_API")
    if not bad_base_url:
        print("Skipping live Ollama-down check; set E2E_BAD_OLLAMA_API to a backend configured with an unavailable Ollama URL.")
        return
    with image_path.open("rb") as file:
        response = httpx.post(
            f"{bad_base_url.rstrip('/')}/vision/analyze",
            files={"file": (image_path.name, file, "image/png")},
            timeout=60,
        )
    assert response.status_code == 503, response.text
    detail = response.json().get("detail", {})
    assert "message" in detail and ("Ollama" in detail["message"] or "모델" in detail["message"]), detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("E2E_BASE_URL", "http://localhost:8000/api"))
    parser.add_argument("--use-compose", action="store_true", help="Start/restart services with docker compose.")
    parser.add_argument("--skip-vision", action="store_true", help="Skip Qwen2.5-VL image endpoint validation.")
    args = parser.parse_args()

    if args.use_compose:
        run(["docker", "compose", "up", "-d", "--build"])

    health = wait_for_health(args.base_url)
    print("health:", json.dumps(health, ensure_ascii=False))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdf_path = tmp_path / "darcy_test.pdf"
        image_path = tmp_path / "graph_test.png"
        make_pdf(pdf_path)
        make_png(image_path)

        upload = upload_pdf(args.base_url, pdf_path)
        assert upload["document"]["chunks"] >= 1, upload
        assert (DATA_DIR / "metadata" / f"{upload['document']['sha256']}.json").exists()

        vector_files_before = sorted(p.relative_to(VECTOR_DB_DIR).as_posix() for p in VECTOR_DB_DIR.rglob("*") if p.is_file())
        assert vector_files_before, "Expected ChromaDB files under /data/vector_db after upload."

        if args.use_compose:
            run(["docker", "compose", "restart", "backend"])
            wait_for_health(args.base_url)

        vector_files_after = sorted(p.relative_to(VECTOR_DB_DIR).as_posix() for p in VECTOR_DB_DIR.rglob("*") if p.is_file())
        assert vector_files_after == vector_files_before, "Vector DB file set changed unexpectedly after restart."

        chat = ask(args.base_url)
        assert chat["sources"], chat
        first_source = chat["sources"][0]
        assert first_source["document"].endswith("darcy_test.pdf"), first_source
        assert first_source["page"] == 1, first_source
        print("chat source:", first_source)

        if not args.skip_vision:
            vision = analyze_image(args.base_url, image_path)
            assert vision["analysis"].strip(), vision
            print("vision analysis length:", len(vision["analysis"]))

        assert_friendly_ollama_error_shape(args.base_url, image_path)

    print("E2E smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
