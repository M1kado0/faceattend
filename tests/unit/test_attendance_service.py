"""Behavior tests for the headless local attendance workflow."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faceattend.application.attendance_service import AttendanceCoordinator, AttendanceRequest
from faceattend.application.attendance_session import AttendanceDesktopSessionProcessor
from faceattend.application.runtime import RuntimeStatus
from faceattend.persistence.database import SQLiteDatabase
from faceattend.persistence.repositories import LocalRepository
from faceattend.vision.matcher import ExactNumpyMatcher
from faceattend.vision.types import (
    AttendanceStatus,
    BoundingBox,
    EmbeddingTemplate,
    EvidenceDecision,
    FaceObservation,
    FaceQuality,
    Frame,
    FrameEvidence,
    HeadPose,
    LivenessEvidence,
    LivenessKind,
    ModelMetadata,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
MODEL = ModelMetadata("buffalo_l_recognition", "buffalo_l", "embedding-checksum")
QUALITY = FaceQuality(True, 200.0, 0.5, 0.1)


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _evidence(sequence_id: int = 1) -> FrameEvidence:
    frame = Frame(np.full((64, 64, 3), sequence_id, np.uint8), sequence_id * 1_000_000, sequence_id)
    face = FaceObservation(
        BoundingBox(8, 8, 56, 56), 0.99, np.zeros((5, 2), np.float32), QUALITY, "track-1"
    )
    return FrameEvidence(frame, 1, face, "track-1", HeadPose(0.0, -5.0, 0.0), quality=QUALITY)


class Session:
    def __init__(self, *, failure: str | None = None, passive_passed: bool = True) -> None:
        self.phase = SimpleNamespace(value="active_challenge")
        self.failure_reason: str | None = None
        self.active = SimpleNamespace(
            evaluator=SimpleNamespace(
                phases=("turn_left", "blink"),
                result=SimpleNamespace(completed_phases=()),
            )
        )
        self.result = SimpleNamespace(decision=EvidenceDecision.INCONCLUSIVE, passive=None)
        self.failure = failure
        self.passive_passed = passive_passed
        self.cancelled = False

    @property
    def instruction(self) -> str:
        return str(self.phase.value).replace("_", " ").upper()

    def __call__(self, frame: Frame) -> FrameEvidence:
        evidence = replace(_evidence(frame.sequence_id), frame=frame)
        if self.failure is not None:
            self.phase = SimpleNamespace(value="failed")
            self.failure_reason = self.failure
            self.result = SimpleNamespace(decision=EvidenceDecision.FAILED, passive=None)
            return replace(evidence, failure_reason=self.failure)
        passive = LivenessEvidence(
            LivenessKind.PASSIVE,
            EvidenceDecision.PASSED if self.passive_passed else EvidenceDecision.FAILED,
            0.99 if self.passive_passed else 0.1,
            0.85,
            "MiniFASNetV2",
            None if self.passive_passed else "passive_liveness_failed",
            median_score=0.99 if self.passive_passed else 0.1,
            minimum_score=0.98 if self.passive_passed else 0.05,
        )
        self.phase = SimpleNamespace(value="completed" if self.passive_passed else "failed")
        self.failure_reason = passive.reason
        self.active.evaluator.result = SimpleNamespace(
            completed_phases=self.active.evaluator.phases
        )
        self.result = SimpleNamespace(decision=passive.decision, passive=passive)
        return evidence

    def take_embedding_candidate(self) -> FrameEvidence:
        if self.result.decision is not EvidenceDecision.PASSED:
            raise RuntimeError("liveness incomplete")
        return _evidence(10)

    def cancel(self) -> object:
        self.cancelled = True
        return self.result


class RaisingSession(Session):
    def __call__(self, frame: Frame) -> FrameEvidence:
        raise OSError("camera/inference boundary failed")


class Analyzer:
    def __init__(self, vector: np.ndarray, *, fail: bool = False) -> None:
        self.vector = vector
        self.fail = fail
        self.calls = 0

    def extract_embedding(
        self, evidence: FrameEvidence, *, active: LivenessEvidence, passive: LivenessEvidence
    ) -> np.ndarray:
        assert active.decision is EvidenceDecision.PASSED
        assert passive.decision is EvidenceDecision.PASSED
        self.calls += 1
        if self.fail:
            raise RuntimeError("model failed")
        return self.vector


def _setup(tmp_path: Path) -> tuple[LocalRepository, str, str, str]:
    database = SQLiteDatabase(tmp_path / "attendance.sqlite3")
    database.initialize()
    repository = LocalRepository(database)
    configuration_id = repository.add_configuration({"match": "calibrated"}, created_at=NOW)
    model_id = repository.add_model("embedding", MODEL, created_at=NOW)
    attendance_session_id = repository.add_attendance_session("Morning", opened_at=NOW)
    return repository, configuration_id, model_id, attendance_session_id


def _enroll(
    repository: LocalRepository,
    configuration_id: str,
    model_id: str,
    person_id: str,
    vector: np.ndarray,
) -> None:
    person, consent = repository.add_person_with_consent(
        person_id, purpose="attendance", actor_id="operator", created_at=NOW
    )
    enrollment = repository.start_enrollment(
        person, consent_id=consent, configuration_id=configuration_id, started_at=NOW
    )
    repository.complete_enrollment(
        enrollment,
        EmbeddingTemplate(
            "template-" + person_id,
            person,
            vector,
            MODEL.name,
            MODEL.version,
            MODEL.checksum,
            True,
            "neutral",
            QUALITY,
            NOW,
        ),
        model_id=model_id,
        actor_id="operator",
        completed_at=NOW,
    )


def _coordinator(
    tmp_path: Path,
    vector: np.ndarray,
    *,
    session: Session | None = None,
    analyzer: Analyzer | None = None,
) -> tuple[AttendanceCoordinator, LocalRepository, Analyzer, str]:
    repository, configuration_id, model_id, attendance_session_id = _setup(tmp_path)
    _enroll(repository, configuration_id, model_id, "Ada", _unit(1, 0, 0))
    matcher = ExactNumpyMatcher(match_threshold=0.8, ambiguity_margin=0.05)
    actual_analyzer = analyzer or Analyzer(vector)
    coordinator = AttendanceCoordinator(
        AttendanceRequest(attendance_session_id, "operator", configuration_id, MODEL),
        session=session or Session(),
        face_analyzer=actual_analyzer,
        matcher=matcher,
        repository=repository,
        now=lambda: NOW,
    )
    return coordinator, repository, actual_analyzer, attendance_session_id


def test_passed_liveness_matches_and_records_attendance_atomically(tmp_path: Path) -> None:
    coordinator, repository, analyzer, _ = _coordinator(tmp_path, _unit(1, 0, 0))

    result = coordinator.process(_evidence().frame)

    assert result is not None and result.status is AttendanceStatus.RECORDED
    assert result.person_id is not None and result.attendance_record_id is not None
    assert analyzer.calls == 1
    assert repository.get_liveness_attempt(coordinator.liveness_attempt_id) is not None
    assert repository.audit_actions()[-2:] == ("liveness.attempt", "attendance.recorded")


def test_duplicate_check_in_returns_existing_record(tmp_path: Path) -> None:
    first, repository, _, session_id = _coordinator(tmp_path, _unit(1, 0, 0))
    result1 = first.process(_evidence().frame)
    second = AttendanceCoordinator(
        first.request,
        session=Session(),
        face_analyzer=Analyzer(_unit(1, 0, 0)),
        matcher=ExactNumpyMatcher(match_threshold=0.8, ambiguity_margin=0.05),
        repository=repository,
        now=lambda: NOW,
    )
    result2 = second.process(_evidence(2).frame)

    assert result1 is not None and result2 is not None
    assert result2.status is AttendanceStatus.DUPLICATE
    assert result2.attendance_record_id == result1.attendance_record_id
    assert session_id == second.request.attendance_session_id


def test_unknown_and_ambiguous_never_write_attendance(tmp_path: Path) -> None:
    unknown, repository, _, _ = _coordinator(tmp_path, _unit(0, 1, 0))
    unknown_result = unknown.process(_evidence().frame)
    assert unknown_result is not None and unknown_result.status is AttendanceStatus.UNKNOWN

    configuration_id = unknown.request.configuration_id
    model_id = repository.add_model("embedding", MODEL, created_at=NOW)
    _enroll(repository, configuration_id, model_id, "Grace", _unit(0.995, 0.1, 0))
    ambiguous = AttendanceCoordinator(
        unknown.request,
        session=Session(),
        face_analyzer=Analyzer(_unit(1, 0, 0)),
        matcher=ExactNumpyMatcher(match_threshold=0.8, ambiguity_margin=0.02),
        repository=repository,
        now=lambda: NOW,
    )
    ambiguous_result = ambiguous.process(_evidence(2).frame)

    assert ambiguous_result is not None and ambiguous_result.status is AttendanceStatus.AMBIGUOUS
    assert repository.attendance_count() == 0
    assert repository.audit_actions()[-1] == "check_in.ambiguous"


def test_passive_failure_after_active_pass_never_extracts_embedding(tmp_path: Path) -> None:
    coordinator, repository, analyzer, _ = _coordinator(
        tmp_path, _unit(1, 0, 0), session=Session(passive_passed=False)
    )

    result = coordinator.process(_evidence().frame)

    assert result is not None and result.status is AttendanceStatus.FAILED
    assert result.reason == "passive_liveness_failed"
    assert analyzer.calls == 0 and repository.attendance_count() == 0
    attempt = repository.get_liveness_attempt(coordinator.liveness_attempt_id)
    assert attempt is not None
    assert attempt.active_decision == "passed"
    assert attempt.passive_decision == "failed"


@pytest.mark.parametrize(
    "reason",
    ("no_face", "multiple_faces", "face_substitution", "implausible_bbox_jump"),
)
def test_liveness_failures_never_embed_or_record(tmp_path: Path, reason: str) -> None:
    failed_session = Session(failure=reason)
    coordinator, repository, analyzer, _ = _coordinator(
        tmp_path, _unit(1, 0, 0), session=failed_session
    )
    result = coordinator.process(_evidence().frame)
    assert result is not None and result.status is AttendanceStatus.FAILED
    assert result.reason == reason and analyzer.calls == 0
    assert repository.attendance_count() == 0

    model_analyzer = Analyzer(_unit(1, 0, 0), fail=True)
    retry = AttendanceCoordinator(
        coordinator.request,
        session=Session(),
        face_analyzer=model_analyzer,
        matcher=ExactNumpyMatcher(match_threshold=0.8, ambiguity_margin=0.05),
        repository=repository,
        now=lambda: NOW,
    )
    model_result = retry.process(_evidence(2).frame)
    assert model_result is not None and model_result.reason == "model_failure"
    assert repository.attendance_count() == 0


def test_camera_failure_is_explicit_and_records_no_attendance(tmp_path: Path) -> None:
    coordinator, repository, analyzer, _ = _coordinator(
        tmp_path, _unit(1, 0, 0), session=RaisingSession()
    )

    result = coordinator.process(_evidence().frame)

    assert result is not None and result.reason == "model_failure"
    assert analyzer.calls == 0
    assert repository.attendance_count() == 0


def test_database_failure_rolls_back_liveness_and_attendance(tmp_path: Path) -> None:
    coordinator, repository, _, _ = _coordinator(tmp_path, _unit(1, 0, 0))
    coordinator.request = replace(coordinator.request, attendance_session_id="missing-session")

    result = coordinator.process(_evidence().frame)

    assert result is not None and result.reason == "database_failure"
    assert repository.attendance_count() == 0
    assert repository.get_liveness_attempt(coordinator.liveness_attempt_id) is None


def test_withdrawn_attendance_consent_excludes_identity_from_matching(tmp_path: Path) -> None:
    coordinator, repository, _, _ = _coordinator(tmp_path, _unit(1, 0, 0))
    templates = repository.load_compatible_templates(MODEL)
    assert templates
    with repository.database.transaction() as connection:
        connection.execute(
            "UPDATE consent_records SET withdrawn_at = ? WHERE person_id = ?",
            (NOW.isoformat(), templates[0].person_id),
        )

    matcher = ExactNumpyMatcher(match_threshold=0.8, ambiguity_margin=0.05)
    second = AttendanceCoordinator(
        coordinator.request,
        session=Session(),
        face_analyzer=Analyzer(_unit(1, 0, 0)),
        matcher=matcher,
        repository=repository,
        now=lambda: NOW,
    )
    result = second.process(_evidence().frame)

    assert result is not None and result.status is AttendanceStatus.UNKNOWN
    assert repository.attendance_count() == 0


def test_desktop_processor_presents_recorded_outcome(tmp_path: Path) -> None:
    coordinator, _, _, _ = _coordinator(tmp_path, _unit(1, 0, 0))
    processor = AttendanceDesktopSessionProcessor(coordinator)
    frame = _evidence().frame

    evidence = processor(frame)
    presentation = processor.presentation(evidence)

    assert presentation.status is RuntimeStatus.COMPLETED
    assert presentation.instruction == "Attendance recorded"
