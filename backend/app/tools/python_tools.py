from __future__ import annotations

import ast
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from app.agents.permission_manager import AgentSecurityError, PermissionManager
from app.core.config import Settings


_ALLOWED_IMPORT_ROOTS = {
    "csv",
    "json",
    "math",
    "statistics",
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "itertools",
    "functools",
    "numpy",
    "pandas",
    "matplotlib",
}

_BLOCKED_NAMES = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
}

_BLOCKED_ATTRIBUTES = {
    "system",
    "popen",
    "remove",
    "unlink",
    "rmdir",
    "rmtree",
    "chmod",
    "chown",
    "kill",
    "fork",
    "spawn",
    "getenv",
    "modules",
    "run",
    "call",
    "check_call",
    "check_output",
    "Popen",
    "communicate",
    "open",
    "imread",
    "urlopen",
}


class PythonTools:
    def __init__(self, settings: Settings, permissions: PermissionManager):
        self.settings = settings
        self.permissions = permissions

    def validate(self, code: str) -> None:
        if not code.strip():
            raise ValueError("Python code is empty.")
        if len(code) > self.settings.agent_python_max_code_characters:
            raise AgentSecurityError("Python code exceeds the configured size limit.")

        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise ValueError(f"Invalid Python syntax: {exc}") from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._check_import(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise AgentSecurityError("Relative imports are not allowed.")
                self._check_import(node.module or "")
            elif isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                raise AgentSecurityError(f"Blocked Python operation: {node.id}")
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") or node.attr in _BLOCKED_ATTRIBUTES:
                    raise AgentSecurityError(
                        f"Blocked Python attribute access: {node.attr}"
                    )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                self._check_string_literal(node.value)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                raise AgentSecurityError("Global and nonlocal statements are not allowed.")

    def run_python(self, code: str, *, task_id: str) -> dict:
        self.validate(code)

        run_dir = self.settings.agent_runs_dir / task_id
        run_dir.mkdir(parents=True, exist_ok=True)
        script_path = run_dir / "python_task.py"
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        mpl_dir = run_dir / "mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)

        before = self._workspace_snapshot()
        guarded_code = self._guard_prelude() + "\n\n" + code
        script_path.write_text(guarded_code, encoding="utf-8")

        env = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(mpl_dir),
            "HOME": str(self.permissions.workspace),
        }
        for key in ("PATH", "SYSTEMROOT", "WINDIR"):
            if os.environ.get(key):
                env[key] = os.environ[key]

        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        timed_out = False
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=self.permissions.workspace,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.settings.agent_python_timeout_seconds,
                check=False,
                shell=False,
            )
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout = self._to_text(exc.stdout)
            stderr = self._to_text(exc.stderr) + (
                f"\nExecution timed out after "
                f"{self.settings.agent_python_timeout_seconds} seconds."
            )

        duration = time.monotonic() - started
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        after = self._workspace_snapshot()
        created, modified = self._snapshot_diff(before, after)

        return {
            "started_at": started_at,
            "duration_seconds": round(duration, 3),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout": stdout[-self.settings.agent_python_output_characters :],
            "stderr": stderr[-self.settings.agent_python_output_characters :],
            "code_record": script_path.relative_to(self.settings.data_dir).as_posix(),
            "stdout_record": stdout_path.relative_to(self.settings.data_dir).as_posix(),
            "stderr_record": stderr_path.relative_to(self.settings.data_dir).as_posix(),
            "created_files": created,
            "modified_files": modified,
            "success": exit_code == 0,
        }

    @staticmethod
    def _check_string_literal(value: str) -> None:
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered.startswith(("http://", "https://", "ftp://", "file://")):
            raise AgentSecurityError("Network and file URLs are not allowed.")

        normalized = stripped.replace("\\", "/")
        if (
            normalized.startswith("/")
            or re.match(r"^[A-Za-z]:/", normalized)
            or ".." in normalized.split("/")
        ):
            raise AgentSecurityError(
                "Absolute paths and parent-directory traversal are not allowed in Python code."
            )

    def _check_import(self, module_name: str) -> None:
        root = module_name.split(".", 1)[0]
        if root not in _ALLOWED_IMPORT_ROOTS:
            raise AgentSecurityError(f"Blocked Python import: {module_name}")

    def _guard_prelude(self) -> str:
        workspace = repr(str(self.permissions.workspace))
        return f'''# Petroleum RAG Agent sandbox guard
import builtins as _agent_builtins
import socket as _agent_socket
import sys as _agent_sys
from pathlib import Path as _AgentPath

_AGENT_WORKSPACE = _AgentPath({workspace}).resolve()
_AGENT_LIBRARY_ROOTS = tuple(
    _AgentPath(value).resolve()
    for value in {{_agent_sys.prefix, _agent_sys.base_prefix}}
    if value
)
_AGENT_ORIGINAL_OPEN = _agent_builtins.open


def _agent_safe_path(value, *, write=False):
    candidate = _AgentPath(value)
    if not candidate.is_absolute():
        candidate = _AGENT_WORKSPACE / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(_AGENT_WORKSPACE)
    except ValueError as exc:
        if not write:
            for library_root in _AGENT_LIBRARY_ROOTS:
                try:
                    resolved.relative_to(library_root)
                    return resolved
                except ValueError:
                    continue
        raise PermissionError("Agent workspace 밖의 파일 접근이 차단되었습니다.") from exc
    lowered_parts = {{part.lower() for part in resolved.parts}}
    if ".env" in lowered_parts or ".git" in lowered_parts or ".ssh" in lowered_parts:
        raise PermissionError("민감한 파일 또는 폴더 접근이 차단되었습니다.")
    if resolved.suffix.lower() in {{".pem", ".key", ".p12", ".pfx"}}:
        raise PermissionError("인증키 파일 접근이 차단되었습니다.")
    return resolved


def _agent_open(file, *args, **kwargs):
    if isinstance(file, int):
        return _AGENT_ORIGINAL_OPEN(file, *args, **kwargs)
    mode = args[0] if args else kwargs.get("mode", "r")
    write = any(flag in mode for flag in ("w", "a", "x", "+"))
    resolved = _agent_safe_path(file, write=write)
    if write:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return _AGENT_ORIGINAL_OPEN(resolved, *args, **kwargs)


class _AgentBlockedSocket:
    def __init__(self, *args, **kwargs):
        raise PermissionError("Agent Python 실행의 인터넷 접근은 차단되어 있습니다.")


def _agent_block_network(*args, **kwargs):
    raise PermissionError("Agent Python 실행의 인터넷 접근은 차단되어 있습니다.")


_agent_builtins.open = _agent_open
_agent_socket.socket = _AgentBlockedSocket
_agent_socket.create_connection = _agent_block_network
'''

    def _workspace_snapshot(self) -> dict[str, tuple[int, int]]:
        snapshot: dict[str, tuple[int, int]] = {}
        for path in self.permissions.workspace.rglob("*"):
            if not path.is_file():
                continue
            try:
                relative = self.permissions.to_relative(path)
                stat = path.stat()
            except (OSError, ValueError):
                continue
            snapshot[relative] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    @staticmethod
    def _snapshot_diff(
        before: dict[str, tuple[int, int]],
        after: dict[str, tuple[int, int]],
    ) -> tuple[list[str], list[str]]:
        created = sorted(path for path in after if path not in before)
        modified = sorted(
            path
            for path, state in after.items()
            if path in before and before[path] != state
        )
        return created, modified

    @staticmethod
    def _to_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
