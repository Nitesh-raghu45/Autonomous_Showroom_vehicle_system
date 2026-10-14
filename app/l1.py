"""Tiny in-process TTL + LRU cache (the L1 tier).

Reads cost ~1 microsecond instead of a Redis round trip. Values handed out are shared
references: callers must treat them as read-only (mutations build a new dict).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

_MISS = object()


class TTLCache:
    def __init__(self, maxsize: int, ttl: float) -> None:
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key: str, default: Any = None) -> Any:
        item = self._data.get(key)
        if item is None:
            return default
        expires, value = item
        if expires < time.monotonic():
            self._data.pop(key, None)
            return default
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._data[key] = (time.monotonic() + (self._ttl if ttl is None else ttl), value)
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def pop(self, key: str) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
