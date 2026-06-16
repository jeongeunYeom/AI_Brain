import base64
from pathlib import Path

import httpx

from app.core.config import Settings
from app.core.error_mapping import ollama_error_from_exception


class OllamaClient:
    """Small local Ollama client inspired by Connect-AI's offline /api/chat pattern."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ollama_error_from_exception(exc) from exc
        return [model["name"] for model in response.json().get("models", [])]

    async def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        payload = {
            "model": model or self.settings.text_model,
            "messages": messages,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ollama_error_from_exception(exc, payload["model"]) from exc
        return response.json().get("message", {}).get("content", "").strip()

    async def describe_image(self, image_path: Path, prompt: str | None = None) -> str:
        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        graph_prompt = prompt or (
            "Analyze this petroleum engineering graph, plot, map, table, or technical drawing. "
            "Describe axes, units, legend, increasing/decreasing trends, important numeric values, "
            "engineering meaning, and any uncertainty. Answer in Korean unless labels require English."
        )
        payload = {
            "model": self.settings.vision_model,
            "messages": [{"role": "user", "content": graph_prompt, "images": [encoded]}],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=240) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ollama_error_from_exception(exc, payload["model"]) from exc
        return response.json().get("message", {}).get("content", "").strip()
