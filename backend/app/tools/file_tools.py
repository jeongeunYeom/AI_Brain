from __future__ import annotations

import csv
from datetime import datetime, timezone
from difflib import unified_diff
from pathlib import Path
import shutil

from app.agents.permission_manager import AgentSecurityError, PermissionManager
from app.core.config import Settings


_TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml", ".log"}


class FileTools:
    def __init__(self, settings: Settings, permissions: PermissionManager):
        self.settings = settings
        self.permissions = permissions

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict:
        target = self.permissions.resolve_path(
            path,
            must_exist=True,
            allow_directory=False,
            allowed_extensions=_TEXT_EXTENSIONS,
        )
        if target.stat().st_size > self.settings.agent_max_file_bytes:
            raise AgentSecurityError(
                f"File exceeds the {self.settings.agent_max_file_bytes} byte Agent limit."
            )

        start = max(1, start_line)
        requested_end = end_line if end_line is not None else start + 199
        stop = max(start, min(requested_end, start + 499))

        selected: list[str] = []
        total_lines = 0
        total_characters = 0
        truncated = False

        with target.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                total_lines = line_number
                if line_number < start:
                    continue
                if line_number > stop:
                    truncated = True
                    break
                if total_characters + len(line) > self.settings.agent_max_read_characters:
                    truncated = True
                    break
                selected.append(line.rstrip("\n"))
                total_characters += len(line)

        result = {
            "path": self.permissions.to_relative(target),
            "start_line": start,
            "end_line": start + len(selected) - 1 if selected else start,
            "total_lines_seen": total_lines,
            "content": "\n".join(selected),
            "truncated": truncated,
            "size_bytes": target.stat().st_size,
        }

        if target.suffix.lower() == ".csv":
            result["csv"] = self._inspect_csv(target)

        return result

    def create_file(self, path: str, content: str) -> dict:
        target = self.permissions.resolve_path(
            path,
            must_exist=False,
            allow_directory=False,
            allowed_extensions=_TEXT_EXTENSIONS,
        )
        if target.exists():
            raise FileExistsError(
                "The target already exists. Use edit_file after reviewing and approving the change."
            )

        encoded = content.encode("utf-8")
        if len(encoded) > self.settings.agent_max_file_bytes:
            raise AgentSecurityError("Content exceeds the Agent file size limit.")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "path": self.permissions.to_relative(target),
            "size_bytes": len(encoded),
            "created": True,
        }

    def preview_edit(self, path: str, old_text: str, new_text: str) -> dict:
        target, before, after = self._prepare_edit(path, old_text, new_text)
        return {
            "path": self.permissions.to_relative(target),
            "diff": self._diff(self.permissions.to_relative(target), before, after),
        }

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        task_id: str,
    ) -> dict:
        target, before, after = self._prepare_edit(path, old_text, new_text)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        relative = self.permissions.to_relative(target)
        backup = self.settings.agent_backups_dir / task_id / f"{timestamp}_{relative}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        target.write_text(after, encoding="utf-8")

        return {
            "path": relative,
            "backup": backup.relative_to(self.settings.data_dir).as_posix(),
            "diff": self._diff(relative, before, after),
            "modified": True,
        }

    def _prepare_edit(
        self,
        path: str,
        old_text: str,
        new_text: str,
    ) -> tuple[Path, str, str]:
        if not old_text:
            raise ValueError("old_text is required for a limited edit.")

        target = self.permissions.resolve_path(
            path,
            must_exist=True,
            allow_directory=False,
            allowed_extensions=_TEXT_EXTENSIONS,
        )
        if target.stat().st_size > self.settings.agent_max_file_bytes:
            raise AgentSecurityError("File exceeds the Agent edit size limit.")

        before = target.read_text(encoding="utf-8-sig", errors="replace")
        count = before.count(old_text)
        if count == 0:
            raise ValueError("The exact text to replace was not found.")
        if count > 1:
            raise ValueError(
                "The text to replace occurs more than once. Provide a more specific selection."
            )
        after = before.replace(old_text, new_text, 1)
        return target, before, after

    @staticmethod
    def _diff(path: str, before: str, after: str) -> str:
        return "\n".join(
            unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"{path} (before)",
                tofile=f"{path} (after)",
                lineterm="",
            )
        )

    @staticmethod
    def _inspect_csv(path: Path) -> dict:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                return {"columns": [], "row_count": 0, "sample_rows": []}

            sample_rows: list[list[str]] = []
            row_count = 0
            for row in reader:
                row_count += 1
                if len(sample_rows) < 5:
                    sample_rows.append(row)

        return {
            "columns": header,
            "row_count": row_count,
            "sample_rows": sample_rows,
        }
