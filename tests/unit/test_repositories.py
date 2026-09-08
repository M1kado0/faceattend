"""Public behavior tests for local FaceAttend persistence."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from faceattend.persistence.database import SQLiteDatabase
from faceattend.persistence.records import LivenessAttempt
from faceattend.persistence.repositories import LocalRepository, PersistenceError
from faceattend.vision.types import EmbeddingTemplate, FaceQuality, ModelMetadata

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
MODEL = ModelMetadata("buffalo_l_recognition", "buffalo_l", "abc123")


def _repository(tmp_path: Path) -> LocalRepository:
    database = SQLiteDatabase(tmp_path / "faceattend.sqlite3")
    database.initialize()
    return LocalRepository(database)


def _embedding(seed: int = 1) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_registration_records_consent_template_metadata_and_survives_restart(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    configuration_id = repository.add_configuration(
        {"recognition_threshold": "validation-required"}, created_at=NOW
    )
    model_id = repository.add_model("embedding", MODEL, created_at=NOW)
    person_id, consent_id = repository.add_person_with_consent(
        "Ada", purpose="attendance", actor_id="operator", created_at=NOW
    )
    enrollment_id = repository.start_enrollment(
        person_id,
        consent_id=consent_id,
        configuration_id=configuration_id,
        started_at=NOW,
    )
    template = EmbeddingTemplate(
        template_id="template-1",
        person_id=person_id,
        embedding=_embedding(),
        model_name=MODEL.name,
        model_version=MODEL.version,
        model_checksum=MODEL.checksum,
        normalized=True,
        pose_bin="neutral",
        quality=FaceQuality(True, 320.0, 0.45, 0.08),
        created_at=NOW,
    )

    pose_template = EmbeddingTemplate(
        template_id="template-2",
        person_id=person_id,
        embedding=_embedding(9),
        model_name=MODEL.name,
        model_version=MODEL.version,
        model_checksum=MODEL.checksum,
        normalized=True,
        pose_bin="left",
        quality=FaceQuality(True, 290.0, 0.43, 0.075),
        created_at=NOW,
    )

    repository.complete_enrollment(
        enrollment_id,
        (template, pose_template),
        model_id=model_id,
        actor_id="operator",
        completed_at=NOW,
    )

    reopened = _repository(tmp_path)
    loaded = reopened.load_compatible_templates(MODEL)
    assert len(loaded) == 2
    assert loaded[0].template_id == "template-1"
    assert loaded[0].embedding.dtype == np.float32
    assert np.allclose(loaded[0].embedding, template.embedding)
    assert loaded[0].quality == template.quality
    assert loaded[1].template_id == "template-2"
    assert loaded[1].pose_bin == "left"
    assert reopened.audit_actions() == (
        "person.created",
        "consent.granted",
        "enrollment.started",
        "enrollment.completed",
    )


def test_atomic_registration_rolls_back_identity_consent_templates_and_audit(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    configuration_id = repository.add_configuration({}, created_at=NOW)
    model_id = repository.add_model("embedding", MODEL, created_at=NOW)
    person_id = "atomic-person"
    templates = tuple(
        EmbeddingTemplate(
            "duplicate-template-id",
            person_id,
            _embedding(index),
            MODEL.name,
            MODEL.version,
            MODEL.checksum,
            True,
            "neutral",
            FaceQuality(True, 200.0 + index, 0.5, 0.1),
            NOW,
        )
        for index in range(3)
    )
    attempt = LivenessAttempt(
        "atomic-attempt",
        person_id,
        "registration",
        ("blink", "turn_left"),
        ("blink", "turn_left"),
        "passed",
        "passed",
        None,
        0.99,
        0.98,
        0,
        0,
        configuration_id,
        NOW,
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.register_person_atomically(
            "Atomic",
            purpose="attendance",
            actor_id="operator",
            configuration_id=configuration_id,
            model_id=model_id,
            templates=templates,
            liveness_attempt=attempt,
            completed_at=NOW,
        )

    assert repository.person_exists(person_id) is False
    assert repository.get_liveness_attempt("atomic-attempt") is None
    assert repository.audit_actions() == ()


def test_liveness_attempt_and_sanitized_audit_preserve_decision_evidence(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    configuration_id = repository.add_configuration({"active_policy": "v1"}, created_at=NOW)
    attempt = LivenessAttempt(
        attempt_id="attempt-1",
        person_id=None,
        operation="check_in",
        challenge_sequence=("turn_left", "blink"),
        completed_challenges=("turn_left",),
        active_decision="failed",
        passive_decision="not_run",
        failure_reason="challenge_timeout",
        passive_median=None,
        passive_minimum=None,
        suspicious_frame_count=0,
        processing_failure_count=0,
        configuration_version_id=configuration_id,
        created_at=NOW,
    )

    repository.record_liveness_attempt(attempt, actor_id="operator")

    assert repository.get_liveness_attempt("attempt-1") == attempt
    audit = repository.audit_metadata("liveness.attempt")
    assert audit == {
        "active_decision": "failed",
        "challenge_sequence": ["turn_left", "blink"],
        "completed_challenges": ["turn_left"],
        "configuration_version_id": configuration_id,
        "failure_reason": "challenge_timeout",
        "liveness_attempt_id": "attempt-1",
        "operation": "check_in",
        "passive_decision": "not_run",
    }
    with pytest.raises(PersistenceError, match="raw biometric"):
        repository.record_audit(
            actor_id="operator",
            action="unsafe",
            target_id=None,
            metadata={"frame_pixels": b"secret"},
            created_at=NOW,
        )


def test_attendance_is_idempotent_and_person_erasure_removes_templates(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    configuration_id = repository.add_configuration({"match_policy": "candidate"}, created_at=NOW)
    model_id = repository.add_model("embedding", MODEL, created_at=NOW)
    person_id, consent_id = repository.add_person_with_consent(
        "Grace", purpose="attendance", actor_id="operator", created_at=NOW
    )
    enrollment_id = repository.start_enrollment(
        person_id,
        consent_id=consent_id,
        configuration_id=configuration_id,
        started_at=NOW,
    )
    repository.complete_enrollment(
        enrollment_id,
        EmbeddingTemplate(
            "template-delete",
            person_id,
            _embedding(2),
            MODEL.name,
            MODEL.version,
            MODEL.checksum,
            True,
            "neutral",
            FaceQuality(True, 250.0, 0.5, 0.09),
            NOW,
        ),
        model_id=model_id,
        actor_id="operator",
        completed_at=NOW,
    )
    session_id = repository.add_attendance_session("Morning", opened_at=NOW)
    repository.record_liveness_attempt(
        LivenessAttempt(
            "checkin-live-1",
            person_id,
            "check_in",
            ("blink", "turn_left"),
            ("blink", "turn_left"),
            "passed",
            "passed",
            None,
            0.99,
            0.98,
            0,
            0,
            configuration_id,
            NOW,
        ),
        actor_id="operator",
    )

    first = repository.record_attendance(
        session_id,
        person_id,
        liveness_attempt_id="checkin-live-1",
        match_score=0.82,
        match_threshold=0.75,
        ambiguity_margin=0.08,
        configuration_id=configuration_id,
        actor_id="operator",
        checked_in_at=NOW,
    )
    duplicate = repository.record_attendance(
        session_id,
        person_id,
        liveness_attempt_id="checkin-live-1",
        match_score=0.84,
        match_threshold=0.75,
        ambiguity_margin=0.09,
        configuration_id=configuration_id,
        actor_id="operator",
        checked_in_at=NOW,
    )

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.record_id == first.record_id
    deletion = repository.erase_person(person_id, actor_id="operator", deleted_at=NOW)
    assert deletion.template_count == 1
    assert repository.load_compatible_templates(MODEL) == ()
    assert repository.person_exists(person_id) is False
    assert repository.audit_actions()[-3:] == (
        "attendance.recorded",
        "attendance.duplicate",
        "person.erased",
    )


def test_compatibility_filter_and_database_guards_fail_closed(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    configuration_id = repository.add_configuration({}, created_at=NOW)
    person_id, _consent_id = repository.add_person_with_consent(
        "Lin", purpose="attendance", actor_id="operator", created_at=NOW
    )
    repository.record_liveness_attempt(
        LivenessAttempt(
            "failed-live",
            person_id,
            "check_in",
            ("blink",),
            (),
            "failed",
            "not_run",
            "timeout",
            None,
            None,
            0,
            0,
            configuration_id,
            NOW,
        ),
        actor_id="operator",
    )
    session_id = repository.add_attendance_session("Afternoon", opened_at=NOW)

    with pytest.raises(PersistenceError, match="passed check-in liveness"):
        repository.record_attendance(
            session_id,
            person_id,
            liveness_attempt_id="failed-live",
            match_score=0.9,
            match_threshold=0.8,
            ambiguity_margin=0.1,
            configuration_id=configuration_id,
            actor_id="operator",
            checked_in_at=NOW,
        )
    assert (
        repository.load_compatible_templates(
            ModelMetadata(MODEL.name, "other-version", MODEL.checksum)
        )
        == ()
    )

    with repository.database.transaction() as connection:
        event_id = connection.execute("SELECT id FROM audit_events LIMIT 1").fetchone()[0]
    with (
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
        repository.database.transaction() as connection,
    ):
        connection.execute("DELETE FROM audit_events WHERE id = ?", (event_id,))
