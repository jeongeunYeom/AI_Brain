from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.permission_manager import AgentSecurityError, PermissionManager
from app.core.config import Settings
from app.tools.file_tools import FileTools
from app.tools.python_tools import PythonTools


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


def test_python_blocks_indirect_module_access(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        agent_workspace_dir=tmp_path / "workspace",
    )
    settings.agent_workspace_dir.mkdir(parents=True)
    python = PythonTools(settings, PermissionManager(settings))

    with pytest.raises(AgentSecurityError):
        python.validate(
            "import matplotlib.pyplot as plt\n"
            "print(plt.sys.modules['subprocess'])"
        )


def test_python_blocks_absolute_paths_and_urls(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        agent_workspace_dir=tmp_path / "workspace",
    )
    settings.agent_workspace_dir.mkdir(parents=True)
    python = PythonTools(settings, PermissionManager(settings))

    with pytest.raises(AgentSecurityError):
        python.validate("open('/etc/passwd').read()")
    with pytest.raises(AgentSecurityError):
        python.validate("print('https://example.com/data.csv')")


def test_python_blocks_unapproved_modules(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        agent_workspace_dir=tmp_path / "workspace",
    )
    settings.agent_workspace_dir.mkdir(parents=True)
    python = PythonTools(settings, PermissionManager(settings))

    with pytest.raises(AgentSecurityError, match="Blocked Python import"):
        python.validate("import subprocess\nsubprocess.run(['echo', 'no'])")


def test_python_timeout_stops_execution(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        agent_workspace_dir=tmp_path / "workspace",
        agent_python_timeout_seconds=0.1,
    )
    settings.agent_workspace_dir.mkdir(parents=True)
    python = PythonTools(settings, PermissionManager(settings))

    result = python.run_python("while True:\n    pass", task_id="timeout-test")

    assert result["success"] is False
    assert result["timed_out"] is True
    assert result["exit_code"] == 124
    assert "timed out" in result["stderr"]
