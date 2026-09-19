"""In-memory or on-disk offload store for large tool results."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class OffloadStore(Protocol):
    def put(self, content: str) -> str: ...
    def get(self, key: str) -> str | None: ...


class MemoryOffloadStore:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def put(self, content: str) -> str:
        key = "off_" + hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]
        self._data[key] = content
        return key

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def as_dict(self) -> dict[str, str]:
        return dict(self._data)

    def load_dict(self, data: dict[str, str]) -> None:
        self._data.update(data)


class DiskOffloadStore:
    """Persist offloaded blobs under a directory (survives process restart)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, content: str) -> str:
        key = "off_" + hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]
        path = self.root / f"{key}.txt"
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        return key

    def get(self, key: str) -> str | None:
        path = self.root / f"{key}.txt"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
