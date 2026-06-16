from app.core.error_mapping import OllamaModelMissingError, OllamaUnavailableError, ollama_error_from_exception, ollama_error_from_status


class ConnectError(Exception):
    pass


def test_ollama_connect_error_returns_friendly_message():
    error = ollama_error_from_exception(ConnectError("connection refused"))

    assert isinstance(error, OllamaUnavailableError)
    assert "Ollama" in error.user_message
    assert "실행" in error.user_message


def test_ollama_missing_model_returns_pull_hint():
    error = ollama_error_from_status(404, '{"error":"model not found"}', "qwen3:8b")

    assert isinstance(error, OllamaModelMissingError)
    assert "ollama pull qwen3:8b" in error.user_message
