from __future__ import annotations

from datetime import datetime, timezone

from app.agents.permission_manager import PermissionManager


class DirectoryTools:
    def __init__(self, permissions: PermissionManager):
        self.permissions = permissions

    def list_directory(self, path: str = ".", *, limit: int = 500) -> dict:
        target = self.permissions.resolve_path(
            path,
            must_exist=True,
            allow_directory=True,
        )
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        entries: list[dict] = []
        for child in sorted(
            target.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.lower()),
        ):
            try:
                relative = self.permissions.to_relative(child)
                self.permissions.resolve_path(
                    relative,
                    must_exist=True,
                    allow_directory=True,
                )
            except (ValueError, PermissionError):
                continue

            stat = child.stat()
            entries.append(
                {
                    "name": child.name,
                    "path": relative,
                    "kind": "directory" if child.is_dir() else "file",
                    "extension": None if child.is_dir() else child.suffix.lower() or None,
                    "size_bytes": None if child.is_dir() else stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ).isoformat(),
                }
            )
            if len(entries) >= limit:
                break

        return {
            "path": self.permissions.to_relative(target) or ".",
            "entries": entries,
            "truncated": len(entries) >= limit,
        }
