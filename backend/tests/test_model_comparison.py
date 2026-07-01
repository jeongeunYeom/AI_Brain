from __future__ import annotations

import asyncio

from app.core.config import Settings
from app.services.qa import QAService


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls = 0

    def hybrid_search(self, *args, **kwargs):
        self.calls += 1
        return [
            {
                "id": "chunk-1",
                "text": (
                    "Porosity is the ratio of pore volume "
                    "to bulk volume."
                ),
                "metadata": {
                    "document": "Reservoir_Engineering.pdf",
                    "page": 12,
                },
                "score": 0.91,
                "vector_score": 0.88,
                "keyword_score": 0.72,
            }
        ]


class FakeOllama:
    def __init__(self) -> None:
        self.calls: list[
            tuple[str, list[dict[str, str]]]
        ] = []

    async def chat(self, messages, model=None):
        copied = [
            dict(message)
            for message in messages
        ]
        self.calls.append(
            (str(model), copied)
        )
        return (
            f"{model} answer "
            "[Reservoir_Engineering.pdf, p.12]"
        )


def test_compare_retrieves_once_and_reuses_identical_context(
    tmp_path,
):
    async def run_test():
        settings = Settings(data_dir=tmp_path)
        vector_store = FakeVectorStore()
        ollama = FakeOllama()
        service = QAService(
            settings,
            vector_store,
            ollama,
        )

        response = await service.compare(
            "What is porosity?",
            ["qwen3:8b", "gemma4:latest"],
        )

        assert vector_store.calls == 1
        assert [
            item.model
            for item in response.answers
        ] == [
            "qwen3:8b",
            "gemma4:latest",
        ]
        assert response.shared_context is True
        assert len(ollama.calls) == 2
        assert (
            ollama.calls[0][1]
            == ollama.calls[1][1]
        )
        assert response.sources[0].page == 12

    asyncio.run(run_test())


def test_single_model_selection_is_forwarded(
    tmp_path,
):
    async def run_test():
        settings = Settings(data_dir=tmp_path)
        vector_store = FakeVectorStore()
        ollama = FakeOllama()
        service = QAService(
            settings,
            vector_store,
            ollama,
        )

        response = await service.answer(
            "What is porosity?",
            model="gemma4:latest",
        )

        assert response.model == "gemma4:latest"
        assert (
            ollama.calls[0][0]
            == "gemma4:latest"
        )

    asyncio.run(run_test())


def test_default_comparison_models(tmp_path):
    settings = Settings(data_dir=tmp_path)
    assert settings.comparison_models == (
        "qwen3:8b",
        "gemma4:latest",
    )
