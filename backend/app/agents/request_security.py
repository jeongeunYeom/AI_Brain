from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
import re

from app.agents.permission_manager import AgentSecurityError
from app.models.agent_schemas import AgentPlanRequest


class AgentRequestRejected(AgentSecurityError):
    """An unsafe user-supplied path was rejected before planning."""


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
_DELETE_INTENT_PATTERN = re.compile(
    r"(?i)(?:"
    r"(?:파일|폴더|디렉터리|자료|결과물).{0,12}(?:삭제|지워|제거)"
    r"|(?:삭제|지워|제거).{0,12}(?:파일|폴더|디렉터리|자료|결과물)"
    r"|\b(?:delete|remove|unlink|rmdir|rm)\b"
    r")"
)
_SHELL_INTENT_PATTERN = re.compile(
    r"(?i)(?:"
    r"(?:shell|쉘|셸|터미널|명령\s*프롬프트|powershell|cmd)"
    r".{0,16}(?:명령|command|실행|run)"
    r"|(?:명령|command).{0,16}(?:shell|쉘|셸|터미널|powershell|cmd)"
    r")"
)


def validate_agent_plan_request(request: AgentPlanRequest) -> None:
    """Reject paths the planner must never reinterpret as workspace work."""

    _validate_request_text(request.request)
    _validate_explicit_path("대상 경로", request.target_path)
    for value in request.target_paths:
        _validate_explicit_path("비교 대상 경로", value)
    _validate_explicit_path("결과 경로", request.output_path)


def _validate_request_text(value: str) -> None:
    text = _normalize(value)
    if _DELETE_INTENT_PATTERN.search(text):
        raise AgentRequestRejected("파일 또는 폴더 삭제 작업은 허용되지 않습니다.")
    if _SHELL_INTENT_PATTERN.search(text):
        raise AgentRequestRejected("임의 shell 명령 실행은 허용되지 않습니다.")
    if _URL_PATTERN.search(text):
        raise AgentRequestRejected(
            "인터넷 URL에는 접근할 수 없습니다. workspace 내부 파일을 사용하세요."
        )
    if (
        _WINDOWS_ABSOLUTE_PATTERN.search(text)
        or _POSIX_SYSTEM_PATH_PATTERN.search(text)
        or _TRAVERSAL_PATTERN.search(text)
        or _HOME_PATTERN.search(text)
    ):
        raise AgentRequestRejected(
            "Agent workspace 밖의 경로에는 접근할 수 없습니다. "
            "workspace 내부의 상대경로만 사용하세요."
        )
    if _SENSITIVE_PATTERN.search(text):
        raise AgentRequestRejected(
            "환경설정, 인증키 또는 비밀정보 파일에는 접근할 수 없습니다."
        )


def _validate_explicit_path(label: str, value: str | None) -> None:
    if value is None or not value.strip():
        return

    path_text = _normalize(value.strip())
    if _URL_PATTERN.search(path_text):
        raise AgentRequestRejected(f"{label}에는 인터넷 URL을 사용할 수 없습니다.")

    windows_path = PureWindowsPath(path_text)
    posix_path = PurePosixPath(path_text.replace("\\", "/"))
    path_parts = [part.lower() for part in re.split(r"[\\/]+", path_text) if part]

    if (
        windows_path.is_absolute()
        or posix_path.is_absolute()
        or path_text.startswith(("\\\\", "~\\", "~/"))
        or ".." in path_parts
    ):
        raise AgentRequestRejected(
            "Agent workspace 밖의 경로에는 접근할 수 없습니다. "
            "workspace 내부의 상대경로만 사용하세요."
        )

    if _SENSITIVE_PATTERN.search(f"/{path_text}"):
        raise AgentRequestRejected(
            "환경설정, 인증키 또는 비밀정보 파일에는 접근할 수 없습니다."
        )


def _normalize(value: str) -> str:
    # Korean Windows fonts sometimes display a backslash as a won sign.
    return value.replace("₩", "\\")
