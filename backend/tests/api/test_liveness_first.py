from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO

import numpy as np
import pytest
from starlette.datastructures import UploadFile

from backend.api.ml_client import EmbeddingResult, LivenessCheck
from backend.api.routes import check_ins as check_in_module
from backend.api.routes import face_registrations as face_registration_module
from backend.api.services import attendance_record_scan as attendance_record_scan_module
from backend.api.services.video_liveness import VideoLivenessSummary
from backend.db.models.attendance_record import AttendanceRecordRow
from backend.db.models.face_registration import FaceRegistration
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
                embedding_id="registration-1",
                score=0.9,
                metadata={
                    "embedding_model_version": "arcface-r100-v1",
                    "user_id": "user-1",
                },
            )
        ]

    async def delete(self, *, embedding_id: str) -> None:
        self.deleted.append(embedding_id)

    async def delete_by_user(self, *, user_id: str) -> int:
        raise NotImplementedError


def _upload(name: str) -> UploadFile:
    return UploadFile(filename=name, file=BytesIO(b"image-bytes"))


def _passing_video_summary() -> VideoLivenessSummary:
    return VideoLivenessSummary(
        sampled_frames=10,
        visible_faces=10,
        passive_passes=9,
        passive_liveness_pass_ratio=0.9,
        face_visible_ratio=1.0,
        embedding_frame=b"live-frame",
        embedding_frames=(b"live-frame-1", b"live-frame-2", b"live-frame-3"),
    )


@pytest.fixture
def user() -> User:
    return User(id="user-1", email="user@example.com", hashed_password="hash", role="user")


@pytest.mark.asyncio
async def test_face_registration_rejects_failed_liveness_before_embedding(
    monkeypatch, user
) -> None:
    calls = []

    async def fake_log(**kwargs) -> None:
        calls.append(("log", kwargs["action"]))

    async def fake_liveness(**kwargs) -> LivenessCheck:
        calls.append(("liveness", kwargs["filename"]))
        return LivenessCheck(passed=False, score=0.2, label="blink_twice", reason="spoof")

    async def fake_video_liveness(**kwargs) -> VideoLivenessSummary:
        raise AssertionError("passive frame liveness must not run after failed active liveness")

    async def fake_embed(**kwargs) -> EmbeddingResult:
        raise AssertionError("embedding must not run after failed liveness")

    monkeypatch.setattr(face_registration_module, "log", fake_log)
    monkeypatch.setattr(face_registration_module, "verify_active_liveness", fake_liveness)
    monkeypatch.setattr(
        face_registration_module,
        "analyze_video_passive_liveness",
        fake_video_liveness,
    )
    monkeypatch.setattr(face_registration_module, "embed_image", fake_embed)

    session = FakeSession()
    with pytest.raises(Exception) as exc_info:
        await face_registration_module.create_face_registration(
            liveness_blob=_upload("live.webm"),
            user=user,
            session=session,
        )

    assert exc_info.value.status_code == 403
    assert calls == [
        ("log", "face_registration.attempt"),
        ("liveness", "live.webm"),
        ("log", "face_registration.liveness_failed"),
    ]


@pytest.mark.asyncio
async def test_enroll_indexes_only_after_passed_liveness(monkeypatch, user) -> None:
    calls = []
    index = RecordingIndex(added=[], searched=[])

    async def fake_log(**kwargs) -> None:
        calls.append(("log", kwargs["action"]))

    async def fake_liveness(**kwargs) -> LivenessCheck:
        calls.append(("liveness", kwargs["filename"]))
        return LivenessCheck(passed=True, score=0.99, label="blink_twice")

    async def fake_video_liveness(**kwargs) -> VideoLivenessSummary:
        calls.append(("video_passive", kwargs["max_frames"]))
        return _passing_video_summary()

    async def fake_embed(**kwargs) -> EmbeddingResult:
        calls.append(("embed", kwargs["filename"]))
        return EmbeddingResult(
            embedding=np.ones(512, dtype=np.float32),
            model_version="arcface-r100-v1",
        )

    monkeypatch.setattr(face_registration_module, "log", fake_log)
    monkeypatch.setattr(face_registration_module, "verify_active_liveness", fake_liveness)
    monkeypatch.setattr(
        face_registration_module,
        "analyze_video_passive_liveness",
        fake_video_liveness,
    )
    monkeypatch.setattr(face_registration_module, "embed_image", fake_embed)
    monkeypatch.setattr(face_registration_module, "index", index)
    monkeypatch.setattr(attendance_record_scan_module, "log", fake_log)
    monkeypatch.setattr(attendance_record_scan_module, "index", index)

    session = FakeSession()
    response = await face_registration_module.create_face_registration(
        liveness_blob=_upload("live.webm"), user=user, session=session
    )
    assert len(session.added) == 2
    face_registration = next(obj for obj in session.added if isinstance(obj, FaceRegistration))

    assert response.embedding_model_version == "arcface-r100-v1"
    assert index.added[0][2] == {
        "user_id": "user-1",
        "embedding_model_version": "arcface-r100-v1",
    }
    assert calls[:3] == [
        ("log", "face_registration.attempt"),
        ("liveness", "live.webm"),
        ("video_passive", 10),
    ]
    assert ("embed", "liveness-frame.jpg") in calls

    assert face_registration.user_id == "user-1"
    assert face_registration.embedding_model_version == "arcface-r100-v1"
    assert face_registration.embedding_id == index.added[0][0]
    assert response.registration_id == face_registration.id


@pytest.mark.asyncio
async def test_face_registration_rejects_when_face_registration_limit_reached(
    monkeypatch, user
) -> None:
    logs = []
    existing_face_registrations = [
        FaceRegistration(
            id=f"face_registration-{index}",
            user_id=user.id,
            embedding_id=f"embedding-{index}",
            embedding_model_version="arcface-r100-v1",
        )
        for index in range(3)
    ]

    async def fake_log(**kwargs) -> None:
        logs.append(kwargs["action"])

    async def fake_liveness(**kwargs) -> LivenessCheck:
        raise AssertionError("liveness must not run when face registration limit is reached")

    monkeypatch.setattr(face_registration_module, "log", fake_log)
    monkeypatch.setattr(face_registration_module, "verify_active_liveness", fake_liveness)

    session = FakeSession(execute_results=[FakeExecuteResult(rows=existing_face_registrations)])
    with pytest.raises(Exception) as exc_info:
        await face_registration_module.create_face_registration(
            liveness_blob=_upload("live.webm"),
            user=user,
            session=session,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "face_registration_limit_reached"
    assert logs == ["face_registration.attempt", "face_registration.limit_reached"]


@pytest.mark.asyncio
async def test_list_face_registrations_returns_current_user_registrations(user) -> None:
    face_registration = FaceRegistration(
        id="face_registration-1",
        user_id=user.id,
        embedding_id="embedding-1",
        embedding_model_version="arcface-r100-v1",
    )
    session = FakeSession(execute_results=[FakeExecuteResult(rows=[face_registration])])

    response = await face_registration_module.list_face_registrations(user=user, session=session)

    assert response == [
        {
            "id": "face_registration-1",
            "embedding_id": "embedding-1",
            "embedding_model_version": "arcface-r100-v1",
            "created_at": face_registration.created_at,
        }
    ]
    assert len(session.executed) == 1


@pytest.mark.asyncio
async def test_delete_face_registration_deletes_index_and_db_row(monkeypatch, user) -> None:
    logs = []
    index = RecordingIndex(added=[], searched=[])
    face_registration = FaceRegistration(
        id="face_registration-1",
        user_id=user.id,
        embedding_id="embedding-1",
        embedding_model_version="arcface-r100-v1",
    )
    session = FakeSession(execute_results=[FakeExecuteResult(one=face_registration)])

    async def fake_log(**kwargs) -> None:
        logs.append(kwargs)

    monkeypatch.setattr(face_registration_module, "log", fake_log)
    monkeypatch.setattr(face_registration_module, "index", index)

    response = await face_registration_module.delete_face_registration(
        registration_id="face_registration-1",
        user=user,
        session=session,
    )

    assert response == {"status": "deleted"}
    assert index.deleted == ["embedding-1"]
    assert session.deleted == [face_registration]
    assert session.commit_count == 1
    assert session.rollback_count == 0
    assert [entry["action"] for entry in logs] == [
        "face_registration.delete.attempt",
        "face_registration.delete.success",
    ]


@pytest.mark.asyncio
async def test_delete_face_registration_returns_404_when_missing(monkeypatch, user) -> None:
    logs = []
    index = RecordingIndex(added=[], searched=[])
    session = FakeSession(execute_results=[FakeExecuteResult(one=None)])

    async def fake_log(**kwargs) -> None:
        logs.append(kwargs)

    monkeypatch.setattr(face_registration_module, "log", fake_log)
    monkeypatch.setattr(face_registration_module, "index", index)

    with pytest.raises(Exception) as exc_info:
        await face_registration_module.delete_face_registration(
            registration_id="missing-registration",
            user=user,
            session=session,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "face_registration_not_found"
    assert index.deleted == []
    assert session.deleted == []
    assert session.commit_count == 0
    assert [entry["action"] for entry in logs] == [
        "face_registration.delete.attempt",
        "face_registration.delete.not_found",
    ]


@pytest.mark.asyncio
async def test_check_in_filters_by_embedding_model_after_liveness(monkeypatch, user) -> None:
    index = RecordingIndex(added=[], searched=[])

    async def fake_log(**kwargs) -> None:
        return None

    async def fake_liveness(**kwargs) -> LivenessCheck:
        assert kwargs["challenge"] == "blink_twice"
        return LivenessCheck(passed=True, score=0.99, label="Real")

    async def fake_embed(**kwargs) -> EmbeddingResult:
        index = int(kwargs["filename"].removesuffix(".jpg").split("-")[-1])
        return EmbeddingResult(
            embedding=np.full(512, index + 1, dtype=np.float32),
            model_version="arcface-r100-v1",
        )

    async def fake_video_liveness(**kwargs) -> VideoLivenessSummary:
        return _passing_video_summary()

    monkeypatch.setattr(check_in_module, "log", fake_log)
    monkeypatch.setattr(check_in_module, "verify_active_liveness", fake_liveness)
    monkeypatch.setattr(
        check_in_module,
        "analyze_video_passive_liveness",
        fake_video_liveness,
    )
    monkeypatch.setattr(check_in_module, "embed_image", fake_embed)
    monkeypatch.setattr(attendance_record_scan_module, "log", fake_log)
    monkeypatch.setattr(attendance_record_scan_module, "index", index)

    session = FakeSession()
    response = await check_in_module.check_in(
        liveness_blob=_upload("live.webm"),
        user=user,
        session=session,
    )

    assert len(session.added) == 1
    attendance_record = session.added[0]
    assert response.attendance_records[0].record_id == attendance_record.id
    assert attendance_record.user_id == user.id
    assert attendance_record.face_registration_id == "registration-1"
    assert session.commit_count == 1
    assert len(index.searched) == 3
    assert index.searched[0][2] == {
        "embedding_model_version": "arcface-r100-v1",
        "user_id": user.id,
    }


@pytest.mark.asyncio
async def test_check_in_reuses_existing_persisted_match(monkeypatch, user) -> None:
    index = RecordingIndex(added=[], searched=[])
    existing_record = AttendanceRecordRow(
        id="persisted-record-1",
        user_id=user.id,
        face_registration_id="registration-1",
        score=0.9,
        checked_in_at=datetime.utcnow(),
        created_at=datetime.utcnow(),
    )

    async def fake_log(**kwargs) -> None:
        return None

    async def fake_liveness(**kwargs) -> LivenessCheck:
        assert kwargs["challenge"] == "blink_twice"
        return LivenessCheck(passed=True, score=0.99, label="Real")

    async def fake_embed(**kwargs) -> EmbeddingResult:
        return EmbeddingResult(
            embedding=np.ones(512, dtype=np.float32),
            model_version="arcface-r100-v1",
        )

    async def fake_video_liveness(**kwargs) -> VideoLivenessSummary:
        return _passing_video_summary()

    monkeypatch.setattr(check_in_module, "log", fake_log)
    monkeypatch.setattr(check_in_module, "verify_active_liveness", fake_liveness)
    monkeypatch.setattr(
        check_in_module,
        "analyze_video_passive_liveness",
        fake_video_liveness,
    )
    monkeypatch.setattr(check_in_module, "embed_image", fake_embed)
    monkeypatch.setattr(attendance_record_scan_module, "log", fake_log)
    monkeypatch.setattr(attendance_record_scan_module, "index", index)

    session = FakeSession(execute_results=[FakeExecuteResult(one=existing_record)])
    response = await check_in_module.check_in(
        liveness_blob=_upload("live.webm"),
        user=user,
        session=session,
    )

    assert session.added == []
    assert session.commit_count == 0
    assert response.attendance_records[0].record_id == "persisted-record-1"
    assert response.attendance_records[0].score == 0.9


@pytest.mark.asyncio
async def test_check_in_rejects_failed_video_passive_liveness(monkeypatch, user) -> None:
    async def fake_log(**kwargs) -> None:
        return None

    async def fake_liveness(**kwargs) -> LivenessCheck:
        assert kwargs["challenge"] == "blink_twice"
        return LivenessCheck(passed=True, score=0.99, label="blink_twice")

    async def fake_video_liveness(**kwargs) -> VideoLivenessSummary:
        summary = _passing_video_summary()
        return VideoLivenessSummary(
            sampled_frames=summary.sampled_frames,
            visible_faces=summary.visible_faces,
            passive_passes=5,
            passive_liveness_pass_ratio=0.5,
            face_visible_ratio=0.9,
            embedding_frame=b"live-frame",
        )

    async def fake_embed(**kwargs) -> EmbeddingResult:
        raise AssertionError("embedding must not run after failed passive video liveness")

    monkeypatch.setattr(check_in_module, "log", fake_log)
    monkeypatch.setattr(check_in_module, "verify_active_liveness", fake_liveness)
    monkeypatch.setattr(
        check_in_module,
        "analyze_video_passive_liveness",
        fake_video_liveness,
    )
    monkeypatch.setattr(check_in_module, "embed_image", fake_embed)

    session = FakeSession()
    with pytest.raises(Exception) as exc_info:
        await check_in_module.check_in(
            liveness_blob=_upload("live.webm"),
            user=user,
            session=session,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "passive_liveness_failed"


@pytest.mark.asyncio
async def test_check_in_rejects_low_face_visible_ratio(monkeypatch, user) -> None:
    async def fake_log(**kwargs) -> None:
        return None

    async def fake_liveness(**kwargs) -> LivenessCheck:
        assert kwargs["challenge"] == "blink_twice"
        return LivenessCheck(passed=True, score=0.99, label="blink_twice")

    async def fake_video_liveness(**kwargs) -> VideoLivenessSummary:
        summary = _passing_video_summary()
        return VideoLivenessSummary(
            sampled_frames=summary.sampled_frames,
            visible_faces=5,
            passive_passes=5,
            passive_liveness_pass_ratio=0.9,
            face_visible_ratio=0.5,
            embedding_frame=b"live-frame",
        )

    async def fake_embed(**kwargs) -> EmbeddingResult:
        raise AssertionError("embedding must not run after low face-visible ratio")

    monkeypatch.setattr(check_in_module, "log", fake_log)
    monkeypatch.setattr(check_in_module, "verify_active_liveness", fake_liveness)
    monkeypatch.setattr(
        check_in_module,
        "analyze_video_passive_liveness",
        fake_video_liveness,
    )
    monkeypatch.setattr(check_in_module, "embed_image", fake_embed)

    session = FakeSession()
    with pytest.raises(Exception) as exc_info:
        await check_in_module.check_in(
            liveness_blob=_upload("live.webm"),
            user=user,
            session=session,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "face_not_visible_enough"


@pytest.mark.asyncio
async def test_check_in_rejects_identity_mismatch(monkeypatch, user) -> None:
    async def fake_log(**kwargs) -> None:
        return None

    async def fake_liveness(**kwargs) -> LivenessCheck:
        assert kwargs["challenge"] == "blink_twice"
        return LivenessCheck(passed=True, score=0.99, label="blink_twice")

    async def fake_video_liveness(**kwargs) -> VideoLivenessSummary:
        return _passing_video_summary()

    async def fake_embed(**kwargs) -> EmbeddingResult:
        return EmbeddingResult(
            embedding=np.ones(512, dtype=np.float32),
            model_version="arcface-r100-v1",
        )

    async def fake_scan_best(**kwargs) -> list:
        return []

    monkeypatch.setattr(check_in_module, "log", fake_log)
    monkeypatch.setattr(check_in_module, "verify_active_liveness", fake_liveness)
    monkeypatch.setattr(
        check_in_module,
        "analyze_video_passive_liveness",
        fake_video_liveness,
    )
    monkeypatch.setattr(check_in_module, "embed_image", fake_embed)
    monkeypatch.setattr(
        check_in_module,
        "scan_best_and_persist_attendance_record",
        fake_scan_best,
    )

    session = FakeSession()
    with pytest.raises(Exception) as exc_info:
        await check_in_module.check_in(
            liveness_blob=_upload("live.webm"),
            user=user,
            session=session,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "identity_not_matched"


@pytest.mark.asyncio
async def test_check_in_ignores_low_confidence_match(monkeypatch, user) -> None:
    index = RecordingIndex(added=[], searched=[])

    async def fake_search(**kwargs) -> list[Match]:
        index.searched.append((kwargs["embedding"], kwargs["top_k"], kwargs["filter"]))
        return [
            Match(
                embedding_id="registration-1",
                score=0.02,
                metadata={
                    "embedding_model_version": "arcface-r100-v1",
                    "user_id": user.id,
                },
            )
        ]

    async def fake_log(**kwargs) -> None:
        return None

    index.search = fake_search
    monkeypatch.setattr(attendance_record_scan_module, "log", fake_log)
    monkeypatch.setattr(attendance_record_scan_module, "index", index)

    session = FakeSession()
    response = await attendance_record_scan_module.scan_and_persist_attendance_records(
        user=user,
        embedding=np.ones(512, dtype=np.float32),
        model_version="arcface-r100-v1",
        session=session,
    )

    assert response == []
    assert session.added == []
    assert session.commit_count == 0


@pytest.mark.asyncio
async def test_check_in_persists_best_match_across_live_frame_embeddings(monkeypatch, user) -> None:
    index = RecordingIndex(added=[], searched=[])

    async def fake_search(**kwargs) -> list[Match]:
        index.searched.append((kwargs["embedding"], kwargs["top_k"], kwargs["filter"]))
        score_by_marker = {1.0: 0.81, 2.0: 0.93, 3.0: 0.88}
        marker = float(kwargs["embedding"][0])
        return [
            Match(
                embedding_id="registration-1",
                score=score_by_marker[marker],
                metadata={
                    "embedding_model_version": "arcface-r100-v1",
                    "user_id": user.id,
                },
            )
        ]

    async def fake_log(**kwargs) -> None:
        return None

    index.search = fake_search
    monkeypatch.setattr(attendance_record_scan_module, "log", fake_log)
    monkeypatch.setattr(attendance_record_scan_module, "index", index)

    session = FakeSession()
    response = await attendance_record_scan_module.scan_best_and_persist_attendance_record(
        user=user,
        embeddings=[
            np.ones(512, dtype=np.float32),
            np.full(512, 2, dtype=np.float32),
            np.full(512, 3, dtype=np.float32),
        ],
        model_version="arcface-r100-v1",
        session=session,
    )

    assert len(index.searched) == 3
    assert len(session.added) == 1
    assert session.added[0].score == 0.93
    assert response[0].score == 0.93
