from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
import re

from app.agents.permission_manager import AgentSecurityError
from app.models.agent_schemas import AgentPlanRequest


_URL_PATTERN = re.compile(r"(?i)\b(?:https?|ftp|file)://")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(
    r"(?i)(?:^|[\s\"'`(])(?:[a-z]:[\\/]|\\\\)"
)
_POSIX_SYSTEM_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'`(])/(?:etc|usr|var|home|root|opt|proc|sys|dev|tmp)(?:[\\/\s]|$)",
    flags=re.IGNORECASE,
)
_TRAVERSAL_PATTERN = re.compile(r"(?:^|[\s\"'`(])\.\.[\\/]")
_HOME_PATTERN = re.compile(r"(?:^|[\s\"'`(])~[\\/]")
_SENSITIVE_PATTERN = re.compile(
    r"(?:^|[\\/:\s\"'`(])"
    r"(?:"
    r"\.env(?:\.[a-z0-9_.-]+)?"
    r"|id_rsa|id_ed25519"
    r"|credentials\.json|secrets\.json"
    r"|[^\\/\s]+\.(?:pem|key|p12|pfx)"
    r")"
    r"(?=$|[\\/\s\"'`),.!?])",
    flags=re.IGNORECASE,
)


def validate_agent_plan_request(request: AgentPlanRequest) -> None:
    """Reject paths the planner must never reinterpret as workspace work."""

    _validate_request_text(request.request)
    _validate_explicit_path("대상 경로", request.target_path)
    _validate_explicit_path("결과 경로", request.output_path)


def _validate_request_text(value: str) -> None:
    text = _normalize(value)
    if _URL_PATTERN.search(text):
        raise AgentSecurityError(
            "인터넷 URL에는 접근할 수 없습니다. workspace 내부 파일을 사용하세요."
        )
    if (
        _WINDOWS_ABSOLUTE_PATTERN.search(text)
        or _POSIX_SYSTEM_PATH_PATTERN.search(text)
        or _TRAVERSAL_PATTERN.search(text)
        or _HOME_PATTERN.search(text)
    ):
        raise AgentSecurityError(
            "Agent workspace 밖의 경로에는 접근할 수 없습니다. "
            "workspace 내부의 상대경로만 사용하세요."
        )
    if _SENSITIVE_PATTERN.search(text):
        raise AgentSecurityError(
            "환경설정, 인증키 또는 비밀정보 파일에는 접근할 수 없습니다."
        )


def _validate_explicit_path(label: str, value: str | None) -> None:
    if value is None or not value.strip():
        return

    path_text = _normalize(value.strip())
    if _URL_PATTERN.search(path_text):
        raise AgentSecurityError(f"{label}에는 인터넷 URL을 사용할 수 없습니다.")

    windows_path = PureWindowsPath(path_text)
    posix_path = PurePosixPath(path_text.replace("\\", "/"))
    path_parts = [part.lower() for part in re.split(r"[\\/]+", path_text) if part]

    if (
        windows_path.is_absolute()
        or posix_path.is_absolute()
        or path_text.startswith(("\\\\", "~\\", "~/"))
        or ".." in path_parts
    ):
        raise AgentSecurityError(
            "Agent workspace 밖의 경로에는 접근할 수 없습니다. "
            "workspace 내부의 상대경로만 사용하세요."
        )

    if _SENSITIVE_PATTERN.search(f"/{path_text}"):
        raise AgentSecurityError(
            "환경설정, 인증키 또는 비밀정보 파일에는 접근할 수 없습니다."
        )


def _normalize(value: str) -> str:
    # Korean Windows fonts sometimes display a backslash as a won sign.
    return value.replace("₩", "\\")
