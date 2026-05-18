"""Faiss implementation of VectorStore."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.indexer.store import Match, VectorStore

_DIM = 512


class FAISSStore(VectorStore):
    def __init__(self, index_path: str) -> None:
        self._path = Path(index_path)
        self._meta_path = self._path.with_suffix(".meta.json")

        if self._path.exists():
            self._index = faiss.read_index(str(self._path))
            raw = json.loads(self._meta_path.read_text())
            self._id_map: list[str] = raw["id_map"]
            self._metadata: dict[str, dict[str, Any]] = raw["metadata"]
        else:
            self._index = faiss.IndexFlatIP(_DIM)
            self._id_map = []
            self._metadata = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._path))
        self._meta_path.write_text(json.dumps({"id_map": self._id_map, "metadata": self._metadata}))

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vec = embedding.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Zero-norm embedding")
        return vec / norm

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], filter: dict | None) -> bool:  # noqa: A002
        if not filter:
            return True
        return all(metadata.get(key) == value for key, value in filter.items())

    def _rebuild(self, kept_ids: list[str], vectors: list[np.ndarray]) -> None:
        self._index = faiss.IndexFlatIP(_DIM)
        self._id_map = kept_ids
        if vectors:
            self._index.add(np.vstack(vectors).astype(np.float32))
        self._save()

    async def add(self, *, embedding_id: str, embedding: np.ndarray, metadata: dict) -> None:
        if embedding_id in self._metadata:
            await self.delete(embedding_id=embedding_id)
        vec = self._normalize(embedding).reshape(1, _DIM)
        self._index.add(vec)
        self._id_map.append(embedding_id)
        self._metadata[embedding_id] = metadata
        self._save()

    async def search(
        self,
        *,
        embedding: np.ndarray,
        top_k: int,
        filter: dict | None = None,  # noqa: A002
    ) -> list[Match]:
        if self._index.ntotal == 0:
            return []
        vec = self._normalize(embedding).reshape(1, _DIM)
        candidate_count = self._index.ntotal if filter else min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, candidate_count)
        results = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index == -1:
                continue
            embedding_id = self._id_map[index]
            metadata = self._metadata[embedding_id]
            if not self._matches_filter(metadata, filter):
                continue
            results.append(Match(embedding_id=embedding_id, score=float(score), metadata=metadata))
            if len(results) == top_k:
                break
        return results

    async def delete(self, *, embedding_id: str) -> None:
        if embedding_id not in self._metadata:
            return
        kept_ids = [existing_id for existing_id in self._id_map if existing_id != embedding_id]
        vectors = [
            self._index.reconstruct(index)
            for index, existing_id in enumerate(self._id_map)
            if existing_id != embedding_id
        ]
        del self._metadata[embedding_id]
        self._rebuild(kept_ids, vectors)

    async def delete_by_user(self, *, user_id: str) -> int:
        ids_to_delete = {
            embedding_id
            for embedding_id, metadata in self._metadata.items()
            if metadata.get("user_id") == user_id
        }
        if not ids_to_delete:
            return 0

        kept_ids = [
            embedding_id
            for embedding_id in self._id_map
            if embedding_id not in ids_to_delete
        ]
        vectors = [
            self._index.reconstruct(index)
            for index, embedding_id in enumerate(self._id_map)
            if embedding_id not in ids_to_delete
        ]
        for embedding_id in ids_to_delete:
            del self._metadata[embedding_id]
        self._rebuild(kept_ids, vectors)
        return len(ids_to_delete)
