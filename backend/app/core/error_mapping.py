from typing import Any


class ExternalServiceError(RuntimeError):
    def __init__(self, user_message: str, detail: str | None = None):
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail or user_message


class OllamaUnavailableError(ExternalServiceError):
    pass


class OllamaModelMissingError(ExternalServiceError):
    pass


def ollama_error_from_status(status_code: int, response_text: str, model: str | None = None) -> ExternalServiceError:
    normalized = response_text.lower()
    if status_code == 404 or ("model" in normalized and "not found" in normalized):
        model_hint = f" `{model}`" if model else ""
        return OllamaModelMissingError(
            f"Ollama 모델{model_hint}을 찾을 수 없습니다. `ollama pull {model}` 명령으로 모델을 설치해 주세요."
            if model
            else "Ollama 모델을 찾을 수 없습니다. 필요한 모델을 설치해 주세요.",
            response_text,
        )
    return OllamaUnavailableError(f"Ollama 호출이 실패했습니다. 상태 코드: {status_code}", response_text)


def ollama_error_from_exception(exc: Any, model: str | None = None) -> ExternalServiceError:
    name = exc.__class__.__name__
    if name == "ConnectError":
        return OllamaUnavailableError(
            "Ollama 서버에 연결할 수 없습니다. Ollama를 실행한 뒤 필요한 모델을 설치해 주세요.",
            str(exc),
        )
    if name.endswith("TimeoutException") or name in {"ReadTimeout", "ConnectTimeout", "PoolTimeout"}:
        return OllamaUnavailableError(
            "Ollama 응답 시간이 초과되었습니다. 모델이 로딩 중인지 또는 서버 상태를 확인해 주세요.",
            str(exc),
        )
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "status_code"):
        return ollama_error_from_status(int(response.status_code), getattr(response, "text", ""), model)
    return OllamaUnavailableError("Ollama 호출 중 알 수 없는 오류가 발생했습니다.", str(exc))
