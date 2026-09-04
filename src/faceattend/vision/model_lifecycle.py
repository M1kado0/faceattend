"""Headless, lazy model lifecycle shared by inference adapters and tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class LazyModel(Generic[T]):
    """Load one model on first use and reuse it for the application lifetime."""

    factory: Callable[[], T]
    _instance: T | None = None
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def get(self) -> T:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self.factory()
        return self._instance
