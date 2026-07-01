from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class AgentRunLogger:
    """Write one atomic JSON record per agent run."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def new_task_id(self, prefix: str = "WT") -> str:
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"

    def write(self, record: dict[str, Any]) -> Path:
        task_id = self._safe_task_id(
            str(record.get("task_id") or self.new_task_id())
        )
        payload = dict(record)
        payload["task_id"] = task_id
        payload.setdefault(
            "saved_at",
            datetime.now(timezone.utc).isoformat(),
        )

        target = self.root_dir / f"{task_id}.json"
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{task_id}.",
            suffix=".tmp",
            dir=str(self.root_dir),
        )
        temp_path = Path(temp_name)

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        return target

    @staticmethod
    def _safe_task_id(value: str) -> str:
        safe = _SAFE_ID_RE.sub("_", value).strip("._")
        return safe[:160] or uuid4().hex
