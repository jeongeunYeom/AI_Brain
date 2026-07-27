from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.models.agent_schemas import AgentPermissionLevel, AgentToolName


class AgentSecurityError(ValueError):
    pass


class AgentPermissionError(PermissionError):
    pass


_ALLOWED_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".json",
    ".csv",
    ".yaml",
    ".yml",
    ".log",
    ".png",
}

_SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".git",
    ".ssh",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "secrets.json",
}

_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}

_TOOL_LEVELS = {
    AgentToolName.LIST_DIRECTORY: AgentPermissionLevel.READ_ONLY,
    AgentToolName.READ_FILE: AgentPermissionLevel.READ_ONLY,
    AgentToolName.CREATE_FILE: AgentPermissionLevel.SAFE_CREATE,
    AgentToolName.EDIT_FILE: AgentPermissionLevel.APPROVED_EXECUTION,
    AgentToolName.RUN_PYTHON: AgentPermissionLevel.APPROVED_EXECUTION,
}


class PermissionManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.workspace = settings.agent_workspace_dir.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def require_tool_level(
        self,
        permission_level: AgentPermissionLevel,
        tool: AgentToolName,
    ) -> None:
        required = _TOOL_LEVELS[tool]
        if permission_level < required:
            raise AgentPermissionError(
                f"{tool.value} requires permission level {required.value}."
            )

    def resolve_path(
        self,
        raw_path: str | Path,
        *,
        must_exist: bool = False,
        allow_directory: bool = True,
        allowed_extensions: set[str] | None = None,
    ) -> Path:
        value = str(raw_path or ".").strip()
        if not value:
            value = "."

        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate

        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise AgentSecurityError(
                "Agent workspace 밖의 경로에는 접근할 수 없습니다."
            ) from exc

        self._reject_sensitive_path(resolved)

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"File or directory not found: {value}")

        if resolved.exists() and resolved.is_dir():
            if not allow_directory:
                raise AgentSecurityError("A file path is required.")
            return resolved

        extensions = allowed_extensions or _ALLOWED_EXTENSIONS
        if resolved.suffix.lower() not in extensions:
            raise AgentSecurityError(
                f"Unsupported file type: {resolved.suffix or '(no extension)'}"
            )

        return resolved

    def to_relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.workspace).as_posix()

    def find_first_file(self, suffix: str) -> Path:
        suffix = suffix.lower()
        for path in sorted(self.workspace.rglob(f"*{suffix}")):
            if not path.is_file():
                continue
            try:
                self._reject_sensitive_path(path.resolve())
            except AgentSecurityError:
                continue
            return path.resolve()
        raise FileNotFoundError(
            f"No {suffix} file was found inside the agent workspace."
        )

    def _reject_sensitive_path(self, path: Path) -> None:
        relative_parts = path.relative_to(self.workspace).parts
        for part in relative_parts:
            lowered = part.lower()
            if lowered in _SENSITIVE_NAMES:
                raise AgentSecurityError(
                    "환경설정, 인증키 또는 비밀정보 파일에는 접근할 수 없습니다."
                )
            if Path(part).suffix.lower() in _SENSITIVE_SUFFIXES:
                raise AgentSecurityError(
                    "인증서 또는 키 파일에는 접근할 수 없습니다."
                )
