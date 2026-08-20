from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.permission_manager import AgentSecurityError, PermissionManager
from app.agents.tool_registry import ToolRegistry
from app.core.config import Settings
from app.models.agent_schemas import AgentAction, AgentToolName
from app.tools.file_tools import FileTools
from app.tools.knowledge_tools import KnowledgeTools
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


def test_knowledge_tools_return_sources_and_figures_without_generation(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        agent_workspace_dir=tmp_path / "workspace",
    )

    class FakeQAService:
        def __init__(self):
            self.calls = []

        def retrieve_evidence(self, question: str, top_k: int) -> dict:
            self.calls.append((question, top_k))
            return {
                "query": question,
                "query_type": "graph_analysis",
                "sources": [{"document": "well_test.pdf", "page": 22}],
                "figures": [{"filename": "figure_22.png"}],
                "retrieval_elapsed_seconds": 0.01,
            }

    fake_qa = FakeQAService()
    tools = KnowledgeTools(settings, qa_service=fake_qa)

    result = tools.get_related_figures("wellbore storage Figure 찾아줘", top_k=3)

    assert fake_qa.calls == [("wellbore storage Figure 찾아줘", 3)]
    assert result["source_count"] == 1
    assert result["figure_count"] == 1
    assert result["figures"][0]["filename"] == "figure_22.png"


def test_tool_registry_executes_injected_read_only_knowledge_tool(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        agent_workspace_dir=tmp_path / "workspace",
    )
    permissions = PermissionManager(settings)
    registry = ToolRegistry(settings, permissions)

    class FakeKnowledgeTools:
        @staticmethod
        def search_knowledge_base(query: str, top_k: int) -> dict:
            return {"query": query, "top_k": top_k, "sources": []}

    registry._knowledge_tools = FakeKnowledgeTools()
    action = AgentAction(
        action_id="A-test",
        tool=AgentToolName.SEARCH_KNOWLEDGE_BASE,
        description="문헌 검색",
        arguments={"query": "pressure buildup", "top_k": 5},
    )

    permissions.require_tool_level(1, action.tool)
    result = registry.execute(action, task_id="AT-test")

    assert result == {
        "query": "pressure buildup",
        "top_k": 5,
        "sources": [],
    }
