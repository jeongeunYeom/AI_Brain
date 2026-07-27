from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.permission_manager import AgentSecurityError, PermissionManager
from app.core.config import Settings
from app.tools.file_tools import FileTools


def test_sensitive_files_are_blocked(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        agent_workspace_dir=tmp_path / "workspace",
    )
    settings.agent_workspace_dir.mkdir(parents=True)
    (settings.agent_workspace_dir / ".env").write_text("TOKEN=secret", encoding="utf-8")
    files = FileTools(settings, PermissionManager(settings))

    with pytest.raises(AgentSecurityError):
        files.read_file(".env")


def test_create_file_never_overwrites(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        agent_workspace_dir=tmp_path / "workspace",
    )
    settings.agent_workspace_dir.mkdir(parents=True)
    target = settings.agent_workspace_dir / "report.txt"
    target.write_text("original", encoding="utf-8")
    files = FileTools(settings, PermissionManager(settings))

    with pytest.raises(FileExistsError):
        files.create_file("report.txt", "replacement")
    assert target.read_text(encoding="utf-8") == "original"
