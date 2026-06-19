"""Minimal in-memory TTL cache (Phase 4 hardening).

Just enough to make repeated demo queries instant. Not a production cache — no
eviction beyond TTL, process-local only.
"""

from __future__ import annotations

import time
from typing import Any


class TTLCache:
    def __init__(self, ttl_seconds: float = 60.0):
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time() + self.ttl, value)

    def clear(self) -> None:
        self._store.clear()
