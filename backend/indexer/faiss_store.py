"""Faiss implementation of VectorStore."""
from __future__ import annotations

from backend.indexer.store import Match, VectorStore
import faiss
import json
from pathlib import Path
import numpy as np

_DIM = 512

class FAISSStore(VectorStore):
    def __init__(self, index_path: str) -> None:
        self._path = Path(index_path)
        self._meta_path = self._path.with_suffix('.meta.json')

        if self._path.exists():
            self._index = faiss.read_index(str(self._path))
            raw = json.loads(self._meta_path.read_text())
            self._id_map: list[str] = raw['id_map']
            self._metadata: dict[str, dict] = raw['metadata']
        else:
            self._index = faiss.IndexFlatIP(_DIM)
            self._id_map = []
            self._metadata = {}
    
    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._path))
        self._meta_path.write_text(
            json.dumps({"id_map": self._id_map, "metadata": self._metadata})
        )
    
    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vec = embedding.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Zero-norm embedding")
        return vec / norm
    
    async def add(self, *, image_id, embedding, metadata) -> None:
        vec = self._normalize(embedding).reshape(1, _DIM) 
        self._index.add(vec)
        self._id_map.append(image_id)
        self._metadata[image_id] = metadata
        self._save()

    async def search(self, *, embedding, top_k, filter=None) -> list[Match]:
        if self._index.ntotal == 0:
            return []
        vec = self._normalize(embedding).reshape(1, _DIM)
        scores, indices = self._index.search(vec, top_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            image_id = self._id_map[index]
            results.append(Match(image_id=image_id, score=score, metadata=self._metadata[image_id]))
        return results


    async def delete(self, *, image_id) -> None:
        raise NotImplementedError

    async def delete_by_user(self, *, user_id) -> int:
        raise NotImplementedError
