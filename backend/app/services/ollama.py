import base64
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

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

        return [
            model["name"]
            for model in response.json().get("models", [])
        ]

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> str:
        payload = {
            "model": model or self.settings.text_model,
            "messages": messages,
            "stream": False,
            "think" : False,
            "keep_alive": "30m",
            "options": {
                "temperature": self.settings.ollama_temperature,
                "num_ctx": 8192,
                "num_predict": 1024,
            },
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.settings.ollama_timeout_seconds
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

        except httpx.HTTPError as exc:
            raise ollama_error_from_exception(
                exc,
                payload["model"],
            ) from exc
        
        data = response.json()
        message = data.get("message", {})

        content = message.get("content", "").strip()
        thinking = message.get("thinking", "").strip()

        if not content:
            print(
                "[Ollama 빈 답변] "
                f"done_reason={data.get('done_reason')}, "
                f"eval_count={data.get('eval_count')}, "
                f"thinking_length={len(thinking)}"
            )

        return content

    async def describe_image(
        self,
        image_path: Path,
        prompt: str | None = None,
    ) -> str:
        # PDF에서 추출된 이미지를 일반 RGB PNG로 재변환
        try:
            with Image.open(image_path) as image:
                image.load()

                if image.width < 2 or image.height < 2:
                    print(
                        f"[이미지 생략] 크기가 너무 작음: "
                        f"{image_path}"
                    )
                    return ""

                # 투명 배경이 있는 이미지 처리
                if "A" in image.getbands():
                    rgba_image = image.convert("RGBA")

                    rgb_image = Image.new(
                        "RGB",
                        rgba_image.size,
                        (255, 255, 255),
                    )

                    rgb_image.paste(
                        rgba_image,
                        mask=rgba_image.getchannel("A"),
                    )

                    image = rgb_image
                else:
                    image = image.convert("RGB")

                # 너무 큰 이미지는 축소
                image.thumbnail((1280, 1280))

                buffer = BytesIO()
                image.save(
                    buffer,
                    format="PNG",
                    optimize=True,
                )

                image_bytes = buffer.getvalue()

                if len(image_bytes) < 100:
                    print(
                        f"[이미지 생략] 변환 결과가 비정상적으로 작음: "
                        f"{image_path}"
                    )
                    return ""

                encoded = base64.b64encode(
                    image_bytes
                ).decode("ascii")

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            print(
                f"[이미지 생략] 읽을 수 없는 이미지: "
                f"{image_path} / {exc}"
            )
            return ""

        graph_prompt = prompt or (
            "Analyze only what is directly visible in this image. "
            "Do not assume it is a petroleum engineering figure. "
            "If it is a logo, decorative image, page ornament, or non-technical photo, say it is not suitable for graph analysis. "
            "If it is not a graph or chart, do not invent x-axis or y-axis values. "
            "For graphs/charts, describe only readable axes, units, legend, trends, and numeric values. "
            "If axes, units, or values are not readable, write '확인할 수 없음'. "
            "Do not guess units or engineering meaning beyond the visible evidence. "
            "Answer in Korean unless labels require English."
        )

        payload = {
            "model": self.settings.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": graph_prompt,
                    "images": [encoded],
                }
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 160,
            },
        }

        image_timeout = httpx.Timeout(
            connect=15.0,
            read=120.0,
            write=60.0,
            pool=30.0,
        )

        try:
            async with httpx.AsyncClient(
                timeout=image_timeout
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

        except httpx.ReadTimeout:
            print(
                f"[이미지 분석 생략] 120초 초과: "
                f"{image_path.name}"
            )
            return ""

        except httpx.HTTPStatusError as exc:
            # 특정 이미지만 읽지 못한 경우 PDF 전체 작업은 계속 진행
            if exc.response.status_code == 400:
                print(
                    f"[이미지 분석 생략] Ollama가 이미지를 읽지 못함: "
                    f"{image_path.name} / {exc.response.text}"
                )
                return ""

            raise ollama_error_from_exception(
                exc,
                payload["model"],
            ) from exc

        except httpx.HTTPError as exc:
            raise ollama_error_from_exception(
                exc,
                payload["model"],
            ) from exc

        return (
            response.json()
            .get("message", {})
            .get("content", "")
            .strip()
        )
