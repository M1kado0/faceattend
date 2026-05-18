from __future__ import annotations

import numpy as np
import pytest

from backend.indexer.faiss_store import FAISSStore
from backend.indexer.store import clear_store_cache, get_store


def _embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(512, dtype=np.float32)


@pytest.mark.asyncio
async def test_faiss_store_add_search_filter_and_reload(tmp_path) -> None:
    path = tmp_path / "faces.idx"
    store = FAISSStore(str(path))

    alice_embedding = _embedding(1)
    bob_embedding = _embedding(2)
    await store.add(
        embedding_id="alice-1",
        embedding=alice_embedding,
        metadata={"user_id": "alice", "embedding_model_version": "arcface-r100-v1"},
    )
    await store.add(
        embedding_id="bob-1",
        embedding=bob_embedding,
        metadata={"user_id": "bob", "embedding_model_version": "arcface-r100-v2"},
    )

    matches = await store.search(embedding=alice_embedding, top_k=5, filter={"user_id": "alice"})

    assert [match.embedding_id for match in matches] == ["alice-1"]
    assert matches[0].metadata["embedding_model_version"] == "arcface-r100-v1"

    reloaded = FAISSStore(str(path))
    reloaded_matches = await reloaded.search(
        embedding=bob_embedding,
        top_k=5,
        filter={"embedding_model_version": "arcface-r100-v2"},
    )

    assert [match.embedding_id for match in reloaded_matches] == ["bob-1"]


@pytest.mark.asyncio
async def test_faiss_store_delete_and_delete_by_user(tmp_path) -> None:
    store = FAISSStore(str(tmp_path / "faces.idx"))
    await store.add(embedding_id="a-1", embedding=_embedding(1), metadata={"user_id": "alice"})
    await store.add(embedding_id="a-2", embedding=_embedding(2), metadata={"user_id": "alice"})
    await store.add(embedding_id="b-1", embedding=_embedding(3), metadata={"user_id": "bob"})

    await store.delete(embedding_id="a-1")
    remaining_alice = await store.search(
        embedding=_embedding(2),
        top_k=5,
        filter={"user_id": "alice"},
    )
    assert [match.embedding_id for match in remaining_alice] == ["a-2"]

    deleted = await store.delete_by_user(user_id="alice")

    assert deleted == 1
    assert await store.search(embedding=_embedding(2), top_k=5, filter={"user_id": "alice"}) == []
    remaining_bob = await store.search(embedding=_embedding(3), top_k=5, filter={"user_id": "bob"})
    assert [match.embedding_id for match in remaining_bob] == ["b-1"]


@pytest.mark.asyncio
async def test_faiss_store_rejects_zero_vector(tmp_path) -> None:
    store = FAISSStore(str(tmp_path / "faces.idx"))

    with pytest.raises(ValueError, match="Zero-norm embedding"):
        await store.add(embedding_id="bad", embedding=np.zeros(512, dtype=np.float32), metadata={})


@pytest.mark.asyncio
async def test_get_store_reuses_faiss_instance_for_same_path(monkeypatch, tmp_path) -> None:
    clear_store_cache()
    monkeypatch.setenv("VECTOR_DB_BACKEND", "faiss")
    monkeypatch.setenv("VECTOR_DB_INDEX_PATH", str(tmp_path / "faces.idx"))

    writer = get_store()
    reader = get_store()

    assert writer is reader

    embedding = _embedding(10)
    await writer.add(
        embedding_id="shared-1",
        embedding=embedding,
        metadata={"embedding_model_version": "arcface-r100-v1"},
    )

    matches = await reader.search(
        embedding=embedding,
        top_k=1,
        filter={"embedding_model_version": "arcface-r100-v1"},
    )

    assert [match.embedding_id for match in matches] == ["shared-1"]
    clear_store_cache()
