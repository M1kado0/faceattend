from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
import pytest
from starlette.datastructures import UploadFile

from backend.api.ml_client import EmbeddingResult, LivenessCheck
from backend.api.routes import enroll as enroll_module
from backend.api.routes import search as search_module
from backend.db.models.enrollment import Enrollment
from backend.db.models.user import User
from backend.indexer.store import Match


class FakeScalarResult:
    def __init__(self, rows) -> None:
        self.rows = rows

    def all(self):
        return self.rows


class FakeExecuteResult:
    def __init__(self, *, rows=None, one=None) -> None:
        self.rows = rows or []
        self.one = one

    def scalars(self) -> FakeScalarResult:
        return FakeScalarResult(self.rows)

    def scalar_one_or_none(self):
        return self.one


class FakeSession:
    def __init__(self, execute_results=None, commit_error: Exception | None = None) -> None:
        self.added = []
        self.deleted = []
        self.executed = []
        self.commit_count = 0
        self.rollback_count = 0
        self.execute_results = list(execute_results or [])
        self.commit_error = commit_error

    def add(self, obj) -> None:
        self.added.append(obj)

    async def execute(self, statement):
        self.executed.append(statement)
        if self.execute_results:
            return self.execute_results.pop(0)
        return FakeExecuteResult()

    async def delete(self, obj) -> None:
        self.deleted.append(obj)

    async def commit(self) -> None:
        self.commit_count += 1
        if self.commit_error is not None:
            raise self.commit_error

    async def refresh(self, obj) -> None:
        pass

    async def rollback(self) -> None:
        self.rollback_count += 1


@dataclass
class RecordingIndex:
    added: list[tuple[str, np.ndarray, dict]]
    searched: list[tuple[np.ndarray, int, dict | None]]
    deleted: list[str] = field(default_factory=list)

    async def add(self, *, embedding_id: str, embedding: np.ndarray, metadata: dict) -> None:
        self.added.append((embedding_id, embedding, metadata))

    async def search(
        self,
        *,
        embedding: np.ndarray,
        top_k: int,
        filter: dict | None = None,  # noqa: A002
    ) -> list[Match]:
        self.searched.append((embedding, top_k, filter))
        return [
            Match(
                embedding_id="match-1",
                score=0.9,
                metadata={
                    "source_url": "https://example.test/image.jpg",
                    "source_page": "https://example.test/page",
                    "embedding_model_version": "arcface-r100-v1",
                },
            )
        ]

    async def delete(self, *, embedding_id: str) -> None:
        self.deleted.append(embedding_id)

    async def delete_by_user(self, *, user_id: str) -> int:
        raise NotImplementedError


def _upload(name: str) -> UploadFile:
    return UploadFile(filename=name, file=BytesIO(b"image-bytes"))


@pytest.fixture
def user() -> User:
    return User(id="user-1", email="user@example.com", hashed_password="hash", role="user")


@pytest.mark.asyncio
async def test_enroll_rejects_failed_liveness_before_embedding(monkeypatch, user) -> None:
    calls = []

    async def fake_log(**kwargs) -> None:
        calls.append(("log", kwargs["action"]))

    async def fake_liveness(**kwargs) -> LivenessCheck:
        calls.append(("liveness", kwargs["filename"]))
        return LivenessCheck(passed=False, score=0.2, label="Fake", reason="spoof")

    async def fake_embed(**kwargs) -> EmbeddingResult:
        raise AssertionError("embedding must not run after failed liveness")

    monkeypatch.setattr(enroll_module, "log", fake_log)
    monkeypatch.setattr(enroll_module, "verify_passive_liveness", fake_liveness)
    monkeypatch.setattr(enroll_module, "embed_image", fake_embed)

    session = FakeSession()
    with pytest.raises(Exception) as exc_info:
        await enroll_module.enroll(
            photo=_upload("photo.jpg"),
            liveness_blob=_upload("live.jpg"),
            user=user,
            session=session
        )

    assert exc_info.value.status_code == 403
    assert calls == [
        ("log", "enroll.attempt"),
        ("liveness", "live.jpg"),
        ("log", "enroll.liveness_failed"),
    ]


@pytest.mark.asyncio
async def test_enroll_indexes_only_after_passed_liveness(monkeypatch, user) -> None:
    calls = []
    index = RecordingIndex(added=[], searched=[])

    async def fake_log(**kwargs) -> None:
        calls.append(("log", kwargs["action"]))

    async def fake_liveness(**kwargs) -> LivenessCheck:
        calls.append(("liveness", kwargs["filename"]))
        return LivenessCheck(passed=True, score=0.99, label="Real")

    async def fake_embed(**kwargs) -> EmbeddingResult:
        calls.append(("embed", kwargs["filename"]))
        return EmbeddingResult(
            embedding=np.ones(512, dtype=np.float32),
            model_version="arcface-r100-v1",
        )

    monkeypatch.setattr(enroll_module, "log", fake_log)
    monkeypatch.setattr(enroll_module, "verify_passive_liveness", fake_liveness)
    monkeypatch.setattr(enroll_module, "embed_image", fake_embed)
    monkeypatch.setattr(enroll_module, "index", index)

    session = FakeSession()
    response = await enroll_module.enroll(
        photo=_upload("photo.jpg"),
        liveness_blob=_upload("live.jpg"),
        user=user,
        session=session
    )
    assert len(session.added) == 1
    enrollment = session.added[0]

    assert response.embedding_model_version == "arcface-r100-v1"
    assert index.added[0][2] == {
        "user_id": "user-1",
        "embedding_model_version": "arcface-r100-v1",
    }
    assert calls[:3] == [
        ("log", "enroll.attempt"),
        ("liveness", "live.jpg"),
        ("embed", "photo.jpg"),
    ]

    assert enrollment.user_id == "user-1"
    assert enrollment.embedding_model_version == "arcface-r100-v1"
    assert enrollment.embedding_id == index.added[0][0]
    assert response.enrollment_id == enrollment.id


@pytest.mark.asyncio
async def test_list_enrollments_returns_current_user_enrollments(user) -> None:
    enrollment = Enrollment(
        id="enrollment-1",
        user_id=user.id,
        embedding_id="embedding-1",
        embedding_model_version="arcface-r100-v1",
    )
    session = FakeSession(execute_results=[FakeExecuteResult(rows=[enrollment])])

    response = await enroll_module.list_enrollments(user=user, session=session)

    assert response == [
        {
            "id": "enrollment-1",
            "embedding_id": "embedding-1",
            "embedding_model_version": "arcface-r100-v1",
            "created_at": enrollment.created_at,
        }
    ]
    assert len(session.executed) == 1


@pytest.mark.asyncio
async def test_delete_enrollment_deletes_index_and_db_row(monkeypatch, user) -> None:
    logs = []
    index = RecordingIndex(added=[], searched=[])
    enrollment = Enrollment(
        id="enrollment-1",
        user_id=user.id,
        embedding_id="embedding-1",
        embedding_model_version="arcface-r100-v1",
    )
    session = FakeSession(execute_results=[FakeExecuteResult(one=enrollment)])

    async def fake_log(**kwargs) -> None:
        logs.append(kwargs)

    monkeypatch.setattr(enroll_module, "log", fake_log)
    monkeypatch.setattr(enroll_module, "index", index)

    response = await enroll_module.delete_enrollment(
        enrollment_id="enrollment-1",
        user=user,
        session=session,
    )

    assert response == {"status": "deleted"}
    assert index.deleted == ["embedding-1"]
    assert session.deleted == [enrollment]
    assert session.commit_count == 1
    assert session.rollback_count == 0
    assert [entry["action"] for entry in logs] == [
        "enroll.delete.attempt",
        "enroll.delete.success",
    ]


@pytest.mark.asyncio
async def test_delete_enrollment_returns_404_when_missing(monkeypatch, user) -> None:
    logs = []
    index = RecordingIndex(added=[], searched=[])
    session = FakeSession(execute_results=[FakeExecuteResult(one=None)])

    async def fake_log(**kwargs) -> None:
        logs.append(kwargs)

    monkeypatch.setattr(enroll_module, "log", fake_log)
    monkeypatch.setattr(enroll_module, "index", index)

    with pytest.raises(Exception) as exc_info:
        await enroll_module.delete_enrollment(
            enrollment_id="missing-enrollment",
            user=user,
            session=session,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "enrollment_not_found"
    assert index.deleted == []
    assert session.deleted == []
    assert session.commit_count == 0
    assert [entry["action"] for entry in logs] == [
        "enroll.delete.attempt",
        "enroll.delete.not_found",
    ]


@pytest.mark.asyncio
async def test_search_filters_by_embedding_model_after_liveness(monkeypatch, user) -> None:
    index = RecordingIndex(added=[], searched=[])

    async def fake_log(**kwargs) -> None:
        return None

    async def fake_liveness(**kwargs) -> LivenessCheck:
        return LivenessCheck(passed=True, score=0.99, label="Real")

    async def fake_embed(**kwargs) -> EmbeddingResult:
        return EmbeddingResult(
            embedding=np.ones(512, dtype=np.float32),
            model_version="arcface-r100-v1",
        )

    monkeypatch.setattr(search_module, "log", fake_log)
    monkeypatch.setattr(search_module, "verify_passive_liveness", fake_liveness)
    monkeypatch.setattr(search_module, "embed_image", fake_embed)
    monkeypatch.setattr(search_module, "index", index)

    session = FakeSession()
    response = await search_module.search(
        photo=_upload("photo.jpg"),
        liveness_blob=_upload("live.jpg"),
        user=user,
        session=session,
    )

    assert len(session.added) == 1
    match_row = session.added[0]
    assert response.matches[0].match_id == match_row.id
    assert match_row.user_id == user.id
    assert match_row.image_id == "match-1"
    assert session.commit_count == 1
    assert index.searched[0][2] == {"embedding_model_version": "arcface-r100-v1"}
