"""Behavior tests for the headless local registration workflow."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faceattend.application.registration_service import (
    RegistrationCoordinator,
    RegistrationRequest,
)
from faceattend.application.registration_session import (
    RegistrationDesktopSessionProcessor,
    _registration_failure_reason,
)
from faceattend.application.runtime import RuntimeStatus
from faceattend.persistence.database import SQLiteDatabase
from faceattend.persistence.repositories import LocalRepository
from faceattend.vision.types import (
    BoundingBox,
    EvidenceDecision,
    FaceObservation,
    FaceQuality,
    Frame,
    FrameEvidence,
    HeadPose,
    LivenessEvidence,
    LivenessKind,
    ModelMetadata,
    RegistrationStatus,
)

NOW = datetime(2026, 9, 7, 16, 0, tzinfo=UTC)
MODEL = ModelMetadata("buffalo_l_recognition", "buffalo_l", "embedding-checksum")


def _repository(tmp_path: Path) -> tuple[LocalRepository, str, str]:
    database = SQLiteDatabase(tmp_path / "faceattend.sqlite3")
    database.initialize()
    repository = LocalRepository(database)
    configuration_id = repository.add_configuration({"policy": "phase-5"}, created_at=NOW)
    model_id = repository.add_model("embedding", MODEL, created_at=NOW)
    return repository, configuration_id, model_id


def _evidence(sequence_id: int, pose: HeadPose | None = None) -> FrameEvidence:
    pixels = np.full((64, 64, 3), sequence_id, dtype=np.uint8)
    quality = FaceQuality(True, 100.0 + sequence_id, 0.5, 0.1)
    face = FaceObservation(
        BoundingBox(8, 8, 56, 56),
        0.99,
        np.zeros((5, 2), dtype=np.float32),
        quality,
        "track-1",
    )
    return FrameEvidence(
        Frame(pixels, sequence_id * 100_000_000, sequence_id),
        1,
        face,
        "track-1",
        pose or HeadPose(float(sequence_id), -5.0, 0.0),
        quality=quality,
        lighting_score=0.5,
        smile_score=0.0,
    )


class PassedLivenessSession:
    def __init__(self, candidates: list[FrameEvidence]) -> None:
        self.candidates = candidates
        self.phase = SimpleNamespace(value="active_challenge")
        self.failure_reason: str | None = None
        self.active = SimpleNamespace(
            evaluator=SimpleNamespace(
                phases=("turn_left", "blink"),
                result=SimpleNamespace(completed_phases=()),
            )
        )
        self.result = SimpleNamespace(
            decision=EvidenceDecision.INCONCLUSIVE,
            passive=None,
        )
        self.cancelled = False

    @property
    def instruction(self) -> str:
        return "TURN LEFT" if self.phase.value == "active_challenge" else "COMPLETED"

    def __call__(self, frame: Frame) -> FrameEvidence:
        evidence = self.candidates[-1]
        self.phase = SimpleNamespace(value="completed")
        self.active.evaluator.result = SimpleNamespace(
            completed_phases=self.active.evaluator.phases
        )
        passive = LivenessEvidence(
            LivenessKind.PASSIVE,
            EvidenceDecision.PASSED,
            0.99,
            0.85,
            "MiniFASNetV2",
            median_score=0.99,
            minimum_score=0.98,
        )
        self.result = SimpleNamespace(decision=EvidenceDecision.PASSED, passive=passive)
        return replace(evidence, frame=frame)

    def take_embedding_candidates(
        self, *, min_candidates: int = 3, max_candidates: int = 5
    ) -> tuple[FrameEvidence, ...]:
        if len(self.candidates) < min_candidates:
            raise RuntimeError("insufficient template candidates")
        selected = tuple(self.candidates[:max_candidates])
        self.candidates.clear()
        return selected

    def cancel(self) -> object:
        self.cancelled = True
        self.candidates.clear()
        self.phase = SimpleNamespace(value="cancelled")
        self.failure_reason = "cancelled"
        return self.result


class FailedLivenessSession(PassedLivenessSession):
    def __init__(self, reason: str, candidates: list[FrameEvidence] | None = None) -> None:
        super().__init__(candidates or [])
        self.reason = reason

    def __call__(self, frame: Frame) -> FrameEvidence:
        self.phase = SimpleNamespace(value="failed")
        self.failure_reason = self.reason
        self.result = SimpleNamespace(decision=EvidenceDecision.FAILED, passive=None)
        return FrameEvidence(frame, 0, None, None, None, failure_reason=self.reason)


class PassiveFailedSession(PassedLivenessSession):
    def __call__(self, frame: Frame) -> FrameEvidence:
        item = super().__call__(frame)
        self.result = SimpleNamespace(
            decision=EvidenceDecision.FAILED,
            passive=LivenessEvidence(
                LivenessKind.PASSIVE,
                EvidenceDecision.FAILED,
                0.2,
                0.85,
                "MiniFASNetV2",
                reason="passive_liveness_failed",
            ),
        )
        return item


class Embedder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def extract_embedding(
        self,
        evidence: FrameEvidence,
        *,
        active: LivenessEvidence,
        passive: LivenessEvidence,
    ) -> np.ndarray:
        assert active.decision is EvidenceDecision.PASSED
        assert passive.decision is EvidenceDecision.PASSED
        self.calls += 1
        if self.fail:
            raise RuntimeError("model failed")
        vector = np.zeros(512, dtype=np.float32)
        vector[(evidence.frame.sequence_id - 1) % 512] = 1.0
        return vector


def _request(configuration_id: str, model_id: str, *, consent: bool = True) -> RegistrationRequest:
    return RegistrationRequest(
        display_name="Ada",
        consent_granted=consent,
        consent_purpose="attendance",
        actor_id="operator",
        configuration_id=configuration_id,
        embedding_model_id=model_id,
        embedding_model=MODEL,
    )


def test_registration_persists_consent_liveness_and_diverse_templates_atomically(
    tmp_path: Path,
) -> None:
    repository, configuration_id, model_id = _repository(tmp_path)
    candidates = [
        _evidence(1, HeadPose(-8.0, -5.0, 0.0)),
        _evidence(2, HeadPose(8.0, -5.0, 0.0)),
        _evidence(3, HeadPose(0.0, -12.0, 0.0)),
        _evidence(4, HeadPose(0.0, 2.0, 0.0)),
        _evidence(5, HeadPose(0.0, -5.0, 4.0)),
        _evidence(6, HeadPose(0.0, -5.0, -4.0)),
    ]
    session = PassedLivenessSession(candidates)
    embedder = Embedder()
    coordinator = RegistrationCoordinator(
        _request(configuration_id, model_id),
        session=session,
        face_analyzer=embedder,
        repository=repository,
        now=lambda: NOW,
    )

    result = coordinator.process(_evidence(99).frame)

    assert result is not None and result.status is RegistrationStatus.COMPLETED
    assert result.person_id is not None
    assert len(result.template_ids) == 5
    assert embedder.calls == 5
    templates = repository.load_compatible_templates(MODEL)
    assert len(templates) == 5
    assert all(template.person_id == result.person_id for template in templates)
    assert all(template.pose is not None for template in templates)
    assert repository.get_liveness_attempt(coordinator.liveness_attempt_id) is not None
    assert repository.audit_actions()[-5:] == (
        "person.created",
        "consent.granted",
        "liveness.attempt",
        "enrollment.started",
        "enrollment.completed",
    )
    assert session.candidates == []


def test_registration_requires_explicit_consent_before_capture(tmp_path: Path) -> None:
    repository, configuration_id, model_id = _repository(tmp_path)
    session = PassedLivenessSession([_evidence(index) for index in range(1, 6)])

    with pytest.raises(ValueError, match="explicit consent"):
        RegistrationCoordinator(
            _request(configuration_id, model_id, consent=False),
            session=session,
            face_analyzer=Embedder(),
            repository=repository,
            now=lambda: NOW,
        )

    assert session.candidates == []
    assert repository.audit_actions() == ()


@pytest.mark.parametrize("reason", ["no_face", "multiple_faces", "active_liveness_failed"])
def test_registration_rejects_failed_liveness_without_embedding_or_persistence(
    tmp_path: Path, reason: str
) -> None:
    repository, configuration_id, model_id = _repository(tmp_path)
    session = FailedLivenessSession(reason, [_evidence(index) for index in range(1, 6)])
    embedder = Embedder()
    coordinator = RegistrationCoordinator(
        _request(configuration_id, model_id),
        session=session,
        face_analyzer=embedder,
        repository=repository,
        now=lambda: NOW,
    )

    result = coordinator.process(_evidence(99).frame)

    assert result is not None and result.status is RegistrationStatus.FAILED
    assert result.reason == reason
    assert embedder.calls == 0
    assert session.candidates == []
    assert repository.person_exists(coordinator.person_id) is False
    assert repository.audit_actions() == ()


def test_registration_cancellation_clears_frames_and_fresh_retry_can_succeed(
    tmp_path: Path,
) -> None:
    repository, configuration_id, model_id = _repository(tmp_path)
    first = PassedLivenessSession([_evidence(index) for index in range(1, 6)])
    coordinator = RegistrationCoordinator(
        _request(configuration_id, model_id),
        session=first,
        face_analyzer=Embedder(),
        repository=repository,
        now=lambda: NOW,
    )

    assert coordinator.cancel().status is RegistrationStatus.CANCELLED
    assert first.candidates == []
    retry = PassedLivenessSession([_evidence(index) for index in range(6, 11)])
    coordinator.retry(retry)

    retried = coordinator.process(_evidence(100).frame)
    assert retried is not None and retried.status is RegistrationStatus.COMPLETED


def test_model_failure_leaves_no_partial_identity_and_clears_frames(tmp_path: Path) -> None:
    repository, configuration_id, model_id = _repository(tmp_path)
    session = PassedLivenessSession([_evidence(index) for index in range(1, 6)])
    coordinator = RegistrationCoordinator(
        _request(configuration_id, model_id),
        session=session,
        face_analyzer=Embedder(fail=True),
        repository=repository,
        now=lambda: NOW,
    )

    result = coordinator.process(_evidence(99).frame)

    assert result is not None and result.status is RegistrationStatus.FAILED
    assert repository.person_exists(coordinator.person_id) is False
    assert repository.audit_actions() == ()
    assert session.candidates == []


def test_database_failure_rolls_back_registration_and_clears_frames(tmp_path: Path) -> None:
    repository, configuration_id, _model_id = _repository(tmp_path)
    session = PassedLivenessSession([_evidence(index) for index in range(1, 6)])
    coordinator = RegistrationCoordinator(
        _request(configuration_id, "missing-model"),
        session=session,
        face_analyzer=Embedder(),
        repository=repository,
        now=lambda: NOW,
    )

    result = coordinator.process(_evidence(99).frame)

    assert result is not None and result.status is RegistrationStatus.FAILED
    assert result.reason == "database_failure"
    assert repository.person_exists(coordinator.person_id) is False
    assert repository.audit_actions() == ()
    assert session.candidates == []


def test_completed_active_with_failed_pad_never_extracts_templates(tmp_path: Path) -> None:
    repository, configuration_id, model_id = _repository(tmp_path)
    session = PassiveFailedSession([_evidence(index) for index in range(1, 6)])
    embedder = Embedder()
    coordinator = RegistrationCoordinator(
        _request(configuration_id, model_id),
        session=session,
        face_analyzer=embedder,
        repository=repository,
        now=lambda: NOW,
    )
    result = coordinator.process(_evidence(99).frame)

    assert result is not None and result.status is RegistrationStatus.FAILED
    assert embedder.calls == 0
    assert repository.person_exists(coordinator.person_id) is False


def test_registration_processor_reports_persistence_outcome(tmp_path: Path) -> None:
    repository, configuration_id, model_id = _repository(tmp_path)
    session = PassedLivenessSession([_evidence(index) for index in range(1, 6)])
    coordinator = RegistrationCoordinator(
        _request(configuration_id, model_id),
        session=session,
        face_analyzer=Embedder(),
        repository=repository,
        now=lambda: NOW,
    )
    processor = RegistrationDesktopSessionProcessor(coordinator)
    frame = _evidence(99).frame

    evidence = processor(frame)
    presentation = processor.presentation(evidence)

    assert presentation.status is RuntimeStatus.COMPLETED
    assert presentation.instruction == "Registration complete"


def test_registration_pad_failure_exposes_aggregate_scores_without_biometric_data() -> None:
    passive = LivenessEvidence(
        LivenessKind.PASSIVE,
        EvidenceDecision.FAILED,
        0.52,
        0.85,
        "MiniFASNetV2",
        reason="liveness_score_below_threshold",
        median_score=0.74,
        minimum_score=0.52,
        suspicious_frame_count=2,
    )

    message = _registration_failure_reason("liveness_incomplete", passive)

    assert message == "PAD score below threshold (min 0.520, median 0.740, threshold 0.850)"
