"""Headless, lazy model lifecycle shared by inference adapters and tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


def file_sha256(path: str | Path) -> str:
    """Return a model-file checksum, or ``unknown`` for injected/missing files."""
    model_path = Path(path)
    if not model_path.is_file():
        return "unknown"
    digest = sha256()
    with model_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
