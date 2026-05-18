"""VectorStore protocol — every vector DB backend implements this."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Match:
    embedding_id: str
    score: float
    metadata: dict


class VectorStore(Protocol):
    async def add(
        self,
        *,
        embedding_id: str,
        embedding: np.ndarray,
        metadata: dict,
    ) -> None: ...

    async def search(
        self,
        *,
        embedding: np.ndarray,
        top_k: int,
        filter: dict | None = None,  # noqa: A002
    ) -> list[Match]: ...

    async def delete(self, *, embedding_id: str) -> None: ...

    # GDPR: right-to-erasure entrypoint. Returns count of deleted vectors.
    async def delete_by_user(self, *, user_id: str) -> int: ...


_STORE_CACHE: dict[tuple[str, str], VectorStore] = {}


def get_store() -> VectorStore:
    """Factory selecting the backend by VECTOR_DB_BACKEND env var."""
    backend = os.environ.get("VECTOR_DB_BACKEND", "faiss")
    if backend == "faiss":
        from backend.indexer.faiss_store import FAISSStore

        path = os.getenv("VECTOR_DB_INDEX_PATH", "./data/faiss.idx")
        cache_key = (backend, path)
        if cache_key not in _STORE_CACHE:
            _STORE_CACHE[cache_key] = FAISSStore(path)
        return _STORE_CACHE[cache_key]
    raise ValueError(f"Unsupported VECTOR_DB_BACKEND: {backend}")


def clear_store_cache() -> None:
    """Clear cached store instances for tests and one-off maintenance scripts."""
    _STORE_CACHE.clear()
