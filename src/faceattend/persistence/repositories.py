"""Transactional repositories for local FaceAttend records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from faceattend.persistence.database import SQLiteDatabase
from faceattend.persistence.records import (
    AttendanceWrite,
    EnrollmentWrite,
    ErasureResult,
    LivenessAttempt,
)
from faceattend.vision.types import EmbeddingTemplate, FaceQuality, HeadPose, ModelMetadata


class PersistenceError(ValueError):
    """Raised when a record violates a biometric persistence contract."""


class LocalRepository:
    """Small synchronous repository used by application services.

    Methods own their transactions. Callers must not pass camera frames, face
    crops, or other raw biometric payloads into audit metadata.
    """

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def add_configuration(self, configuration: dict[str, Any], *, created_at: datetime) -> str:
        configuration_id = str(uuid4())
        payload = _json(configuration)
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO configuration_versions(id, configuration_json, created_at) "
                "VALUES (?, ?, ?)",
                (configuration_id, payload, _timestamp(created_at)),
            )
        return configuration_id

    def add_model(self, role: str, model: ModelMetadata, *, created_at: datetime) -> str:
        if not all(
            (role.strip(), model.name.strip(), model.version.strip(), model.checksum.strip())
        ):
            raise PersistenceError("model metadata must be non-empty")
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT id FROM model_versions WHERE role = ? AND name = ? AND version = ? "
                "AND checksum = ?",
                (role, model.name, model.version, model.checksum),
            ).fetchone()
            if existing:
                return str(existing[0])
            model_id = str(uuid4())
            connection.execute(
                "INSERT INTO model_versions(id, role, name, version, checksum, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    role,
                    model.name,
                    model.version,
                    model.checksum,
                    _timestamp(created_at),
                ),
            )
        return model_id

    def add_person_with_consent(
        self,
        display_name: str,
        *,
        purpose: str,
        actor_id: str,
        created_at: datetime,
    ) -> tuple[str, str]:
        if not display_name.strip() or not purpose.strip():
            raise PersistenceError("display name and consent purpose must be non-empty")
        person_id, consent_id = str(uuid4()), str(uuid4())
        timestamp = _timestamp(created_at)
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO people(id, display_name, created_at) VALUES (?, ?, ?)",
                (person_id, display_name.strip(), timestamp),
            )
            connection.execute(
                "INSERT INTO consent_records(id, person_id, purpose, granted_at) "
                "VALUES (?, ?, ?, ?)",
                (consent_id, person_id, purpose, timestamp),
            )
            self._audit(
                connection,
                actor_id=actor_id,
                action="person.created",
                target_id=person_id,
                metadata={"display_name_recorded": True},
                created_at=created_at,
            )
            self._audit(
                connection,
                actor_id=actor_id,
                action="consent.granted",
                target_id=person_id,
                metadata={"consent_record_id": consent_id, "purpose": purpose},
                created_at=created_at,
            )
        return person_id, consent_id

    def register_person_atomically(
        self,
        display_name: str,
        *,
        purpose: str,
        actor_id: str,
        configuration_id: str,
        model_id: str,
        templates: Sequence[EmbeddingTemplate],
        liveness_attempt: LivenessAttempt,
        completed_at: datetime,
    ) -> EnrollmentWrite:
        """Commit identity, consent, liveness, templates, and audits together.

        The caller may generate the person identifier before inference so every
        template is bound to one subject, but no partial biometric record is
        visible unless this transaction commits.
        """
        if not display_name.strip() or not purpose.strip():
            raise PersistenceError("display name and consent purpose must be non-empty")
        enrollment_templates = tuple(templates)
        if not 3 <= len(enrollment_templates) <= 5:
            raise PersistenceError("registration requires 3 to 5 embedding templates")
        person_ids = {template.person_id for template in enrollment_templates}
        if len(person_ids) != 1:
            raise PersistenceError("all templates must belong to one person")
        person_id = next(iter(person_ids))
        if liveness_attempt.person_id != person_id or liveness_attempt.operation != "registration":
            raise PersistenceError("registration liveness must belong to the enrollment person")
        if (
            liveness_attempt.active_decision != "passed"
            or liveness_attempt.passive_decision != "passed"
            or liveness_attempt.failure_reason is not None
        ):
            raise PersistenceError("registration requires passed active and passive liveness")
        if liveness_attempt.configuration_version_id != configuration_id:
            raise PersistenceError("liveness configuration does not match enrollment")
        validated = tuple(
            (
                template,
                _validate_embedding(template),
                _json(asdict(template.quality)),
                _json(asdict(template.pose)) if template.pose is not None else None,
            )
            for template in enrollment_templates
        )
        consent_id, enrollment_id = str(uuid4()), str(uuid4())
        timestamp = _timestamp(completed_at)
        with self.database.transaction() as connection:
            configuration = connection.execute(
                "SELECT 1 FROM configuration_versions WHERE id = ?", (configuration_id,)
            ).fetchone()
            model = connection.execute(
                "SELECT role, name, version, checksum FROM model_versions WHERE id = ?",
                (model_id,),
            ).fetchone()
            expected_models = {
                ("embedding", item.model_name, item.model_version, item.model_checksum)
                for item in enrollment_templates
            }
            if configuration is None or model is None or expected_models != {tuple(model)}:
                raise PersistenceError("registration configuration or model is incompatible")

            connection.execute(
                "INSERT INTO people(id, display_name, created_at) VALUES (?, ?, ?)",
                (person_id, display_name.strip(), timestamp),
            )
            connection.execute(
                "INSERT INTO consent_records(id, person_id, purpose, granted_at) "
                "VALUES (?, ?, ?, ?)",
                (consent_id, person_id, purpose.strip(), timestamp),
            )
            connection.execute(
                "INSERT INTO liveness_attempts(id, person_id, operation, "
                "challenge_sequence_json, completed_challenges_json, active_decision, "
                "passive_decision, failure_reason, passive_median, passive_minimum, "
                "suspicious_frame_count, processing_failure_count, configuration_version_id, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _liveness_values(liveness_attempt),
            )
            connection.execute(
                "INSERT INTO enrollment_sessions(id, person_id, consent_record_id, "
                "configuration_version_id, status, started_at, completed_at) "
                "VALUES (?, ?, ?, ?, 'completed', ?, ?)",
                (enrollment_id, person_id, consent_id, configuration_id, timestamp, timestamp),
            )
            for template, vector, quality_json, pose_json in validated:
                connection.execute(
                    "INSERT INTO embedding_templates(id, person_id, enrollment_session_id, "
                    "model_version_id, embedding, dimension, dtype, normalized, pose_bin, "
                    "quality_json, created_at, pose_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'float32', 1, ?, ?, ?, ?)",
                    (
                        template.template_id,
                        person_id,
                        enrollment_id,
                        model_id,
                        vector.tobytes(order="C"),
                        int(vector.size),
                        template.pose_bin,
                        quality_json,
                        _timestamp(template.created_at),
                        pose_json,
                    ),
                )

            self._audit(
                connection,
                actor_id=actor_id,
                action="person.created",
                target_id=person_id,
                metadata={"display_name_recorded": True},
                created_at=completed_at,
            )
            self._audit(
                connection,
                actor_id=actor_id,
                action="consent.granted",
                target_id=person_id,
                metadata={"consent_record_id": consent_id, "purpose": purpose.strip()},
                created_at=completed_at,
            )
            self._audit(
                connection,
                actor_id=actor_id,
                action="liveness.attempt",
                target_id=person_id,
                metadata=_liveness_audit_metadata(liveness_attempt),
                created_at=completed_at,
            )
            self._audit(
                connection,
                actor_id=actor_id,
                action="enrollment.started",
                target_id=person_id,
                metadata={
                    "enrollment_id": enrollment_id,
                    "configuration_version_id": configuration_id,
                },
                created_at=completed_at,
            )
            self._audit(
                connection,
                actor_id=actor_id,
                action="enrollment.completed",
                target_id=person_id,
                metadata={
                    "enrollment_id": enrollment_id,
                    "template_ids": [item.template_id for item in enrollment_templates],
                    "model_version_id": model_id,
                },
                created_at=completed_at,
            )
        return EnrollmentWrite(person_id, consent_id, enrollment_id)

    def start_enrollment(
        self,
        person_id: str,
        *,
        consent_id: str,
        configuration_id: str,
        started_at: datetime,
        actor_id: str = "system",
    ) -> str:
        enrollment_id = str(uuid4())
        with self.database.transaction() as connection:
            consent = connection.execute(
                "SELECT person_id, withdrawn_at FROM consent_records WHERE id = ?",
                (consent_id,),
            ).fetchone()
            if consent is None or consent[0] != person_id or consent[1] is not None:
                raise PersistenceError("active consent for this person is required")
            connection.execute(
                "INSERT INTO enrollment_sessions(id, person_id, consent_record_id, "
                "configuration_version_id, status, started_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    enrollment_id,
                    person_id,
                    consent_id,
                    configuration_id,
                    "in_progress",
                    _timestamp(started_at),
                ),
            )
            self._audit(
                connection,
                actor_id=actor_id,
                action="enrollment.started",
                target_id=person_id,
                metadata={
                    "enrollment_id": enrollment_id,
                    "configuration_version_id": configuration_id,
                },
                created_at=started_at,
            )
        return enrollment_id

    def complete_enrollment(
        self,
        enrollment_id: str,
        templates: EmbeddingTemplate | Sequence[EmbeddingTemplate],
        *,
        model_id: str,
        actor_id: str,
        completed_at: datetime,
    ) -> None:
        enrollment_templates = (
            (templates,) if isinstance(templates, EmbeddingTemplate) else tuple(templates)
        )
        if not enrollment_templates:
            raise PersistenceError("at least one embedding template is required")
        validated = tuple(
            (
                template,
                _validate_embedding(template),
                _json(asdict(template.quality)),
            )
            for template in enrollment_templates
        )
        with self.database.transaction() as connection:
            enrollment = connection.execute(
                "SELECT person_id, status FROM enrollment_sessions WHERE id = ?",
                (enrollment_id,),
            ).fetchone()
            person_ids = {template.person_id for template in enrollment_templates}
            if enrollment is None or person_ids != {str(enrollment[0])}:
                raise PersistenceError("template does not belong to enrollment person")
            if enrollment[1] != "in_progress":
                raise PersistenceError("enrollment is not in progress")
            model = connection.execute(
                "SELECT role, name, version, checksum FROM model_versions WHERE id = ?",
                (model_id,),
            ).fetchone()
            expected_models = {
                (
                    "embedding",
                    template.model_name,
                    template.model_version,
                    template.model_checksum,
                )
                for template in enrollment_templates
            }
            if model is None or expected_models != {tuple(model)}:
                raise PersistenceError("embedding model metadata is incompatible")
            for template, vector, quality_json in validated:
                connection.execute(
                    "INSERT INTO embedding_templates(id, person_id, enrollment_session_id, "
                    "model_version_id, embedding, dimension, dtype, normalized, pose_bin, "
                    "quality_json, created_at) VALUES (?, ?, ?, ?, ?, ?, 'float32', 1, ?, ?, ?)",
                    (
                        template.template_id,
                        template.person_id,
                        enrollment_id,
                        model_id,
                        vector.tobytes(order="C"),
                        int(vector.size),
                        template.pose_bin,
                        quality_json,
                        _timestamp(template.created_at),
                    ),
                )
            connection.execute(
                "UPDATE enrollment_sessions SET status = 'completed', completed_at = ? "
                "WHERE id = ?",
                (_timestamp(completed_at), enrollment_id),
            )
            self._audit(
                connection,
                actor_id=actor_id,
                action="enrollment.completed",
                target_id=enrollment_templates[0].person_id,
                metadata={
                    "enrollment_id": enrollment_id,
                    "template_ids": [item.template_id for item in enrollment_templates],
                    "model_version_id": model_id,
                },
                created_at=completed_at,
            )

    def load_compatible_templates(self, model: ModelMetadata) -> tuple[EmbeddingTemplate, ...]:
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT t.id, t.person_id, t.embedding, t.dimension, t.dtype, t.normalized, "
                "t.pose_bin, t.quality_json, t.created_at, m.name, m.version, m.checksum, "
                "t.pose_json "
                "FROM embedding_templates AS t JOIN model_versions AS m "
                "ON m.id = t.model_version_id JOIN people AS p ON p.id = t.person_id "
                "WHERE p.deleted_at IS NULL AND EXISTS (SELECT 1 FROM consent_records AS c "
                "WHERE c.person_id = p.id AND c.withdrawn_at IS NULL) "
                "AND m.role = 'embedding' AND m.name = ? "
                "AND m.version = ? AND m.checksum = ? ORDER BY t.created_at, t.id",
                (model.name, model.version, model.checksum),
            ).fetchall()
        return tuple(_template_from_row(row) for row in rows)

    def record_check_in_rejection(
        self,
        attempt: LivenessAttempt,
        *,
        outcome: str,
        actor_id: str,
        match_score: float | None = None,
        match_threshold: float | None = None,
        ambiguity_margin: float | None = None,
    ) -> None:
        """Atomically retain a sanitized failed/unknown/ambiguous check-in decision."""
        if outcome not in {"failed", "unknown", "ambiguous"}:
            raise PersistenceError("invalid rejected check-in outcome")
        if attempt.operation != "check_in" or attempt.person_id is not None:
            raise PersistenceError("rejected check-in must not assert an identity")
        metadata: dict[str, Any] = {
            "liveness_attempt_id": attempt.attempt_id,
            "configuration_version_id": attempt.configuration_version_id,
            "outcome": outcome,
            "failure_reason": attempt.failure_reason,
        }
        if match_score is not None:
            metadata.update(
                match_score=match_score,
                match_threshold=match_threshold,
                ambiguity_margin=ambiguity_margin,
            )
        with self.database.transaction() as connection:
            self._insert_liveness(connection, attempt)
            self._audit(
                connection,
                actor_id=actor_id,
                action="liveness.attempt",
                target_id=None,
                metadata=_liveness_audit_metadata(attempt),
                created_at=attempt.created_at,
            )
            self._audit(
                connection,
                actor_id=actor_id,
                action=f"check_in.{outcome}",
                target_id=None,
                metadata=metadata,
                created_at=attempt.created_at,
            )

    def record_check_in_atomically(
        self,
        attendance_session_id: str,
        person_id: str,
        *,
        attempt: LivenessAttempt,
        match_score: float,
        match_threshold: float,
        ambiguity_margin: float,
        actor_id: str,
        checked_in_at: datetime,
    ) -> AttendanceWrite:
        """Commit passed liveness, attendance, and audits as one transaction."""
        if (
            attempt.operation != "check_in"
            or attempt.person_id != person_id
            or attempt.active_decision != "passed"
            or attempt.passive_decision != "passed"
            or attempt.failure_reason is not None
        ):
            raise PersistenceError("attendance requires passed check-in liveness")
        values = (match_score, match_threshold, ambiguity_margin)
        if not all(np.isfinite(value) for value in values):
            raise PersistenceError("attendance match values must be finite")
        proposed_id = str(uuid4())
        with self.database.transaction() as connection:
            active_consent = connection.execute(
                "SELECT 1 FROM people AS p WHERE p.id = ? AND p.deleted_at IS NULL "
                "AND EXISTS (SELECT 1 FROM consent_records AS c WHERE c.person_id = p.id "
                "AND c.withdrawn_at IS NULL)",
                (person_id,),
            ).fetchone()
            session = connection.execute(
                "SELECT 1 FROM attendance_sessions WHERE id = ? AND closed_at IS NULL",
                (attendance_session_id,),
            ).fetchone()
            if active_consent is None or session is None:
                raise PersistenceError("active session and attendance consent are required")
            self._insert_liveness(connection, attempt)
            self._audit(
                connection,
                actor_id=actor_id,
                action="liveness.attempt",
                target_id=person_id,
                metadata=_liveness_audit_metadata(attempt),
                created_at=checked_in_at,
            )
            cursor = connection.execute(
                "INSERT INTO attendance_records(id, attendance_session_id, person_id, "
                "liveness_attempt_id, checked_in_at, match_score, match_threshold, "
                "ambiguity_margin, configuration_version_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(attendance_session_id, person_id) DO NOTHING",
                (
                    proposed_id,
                    attendance_session_id,
                    person_id,
                    attempt.attempt_id,
                    _timestamp(checked_in_at),
                    match_score,
                    match_threshold,
                    ambiguity_margin,
                    attempt.configuration_version_id,
                ),
            )
            duplicate = cursor.rowcount == 0
            if duplicate:
                row = connection.execute(
                    "SELECT id FROM attendance_records WHERE attendance_session_id = ? "
                    "AND person_id = ?",
                    (attendance_session_id, person_id),
                ).fetchone()
                assert row is not None
                record_id = str(row[0])
            else:
                record_id = proposed_id
            self._audit(
                connection,
                actor_id=actor_id,
                action="attendance.duplicate" if duplicate else "attendance.recorded",
                target_id=person_id,
                metadata={
                    "attendance_record_id": record_id,
                    "attendance_session_id": attendance_session_id,
                    "liveness_attempt_id": attempt.attempt_id,
                    "configuration_version_id": attempt.configuration_version_id,
                    "match_score": match_score,
                    "match_threshold": match_threshold,
                    "ambiguity_margin": ambiguity_margin,
                },
                created_at=checked_in_at,
            )
        return AttendanceWrite(record_id, duplicate)

    def attendance_count(self) -> int:
        with self.database.connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM attendance_records").fetchone()[0])

    @staticmethod
    def _insert_liveness(connection: sqlite3.Connection, attempt: LivenessAttempt) -> None:
        if attempt.suspicious_frame_count < 0 or attempt.processing_failure_count < 0:
            raise PersistenceError("liveness counts must be non-negative")
        connection.execute(
            "INSERT INTO liveness_attempts(id, person_id, operation, challenge_sequence_json, "
            "completed_challenges_json, active_decision, passive_decision, failure_reason, "
            "passive_median, passive_minimum, suspicious_frame_count, processing_failure_count, "
            "configuration_version_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _liveness_values(attempt),
        )

    def audit_actions(self) -> tuple[str, ...]:
        with self.database.connection() as connection:
            rows = connection.execute("SELECT action FROM audit_events ORDER BY rowid").fetchall()
        return tuple(str(row[0]) for row in rows)

    def record_liveness_attempt(self, attempt: LivenessAttempt, *, actor_id: str) -> None:
        if attempt.suspicious_frame_count < 0 or attempt.processing_failure_count < 0:
            raise PersistenceError("liveness counts must be non-negative")
        challenge_sequence = list(attempt.challenge_sequence)
        completed_challenges = list(attempt.completed_challenges)
        audit_metadata = {
            "liveness_attempt_id": attempt.attempt_id,
            "operation": attempt.operation,
            "challenge_sequence": challenge_sequence,
            "completed_challenges": completed_challenges,
            "active_decision": attempt.active_decision,
            "passive_decision": attempt.passive_decision,
            "failure_reason": attempt.failure_reason,
            "configuration_version_id": attempt.configuration_version_id,
        }
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO liveness_attempts(id, person_id, operation, "
                "challenge_sequence_json, completed_challenges_json, active_decision, "
                "passive_decision, failure_reason, passive_median, passive_minimum, "
                "suspicious_frame_count, processing_failure_count, configuration_version_id, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    attempt.attempt_id,
                    attempt.person_id,
                    attempt.operation,
                    _json(challenge_sequence),
                    _json(completed_challenges),
                    attempt.active_decision,
                    attempt.passive_decision,
                    attempt.failure_reason,
                    attempt.passive_median,
                    attempt.passive_minimum,
                    attempt.suspicious_frame_count,
                    attempt.processing_failure_count,
                    attempt.configuration_version_id,
                    _timestamp(attempt.created_at),
                ),
            )
            self._audit(
                connection,
                actor_id=actor_id,
                action="liveness.attempt",
                target_id=attempt.person_id,
                metadata=audit_metadata,
                created_at=attempt.created_at,
            )

    def add_attendance_session(
        self, name: str, *, opened_at: datetime, actor_id: str = "system"
    ) -> str:
        if not name.strip():
            raise PersistenceError("attendance session name must be non-empty")
        session_id = str(uuid4())
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO attendance_sessions(id, name, opened_at) VALUES (?, ?, ?)",
                (session_id, name.strip(), _timestamp(opened_at)),
            )
            self._audit(
                connection,
                actor_id=actor_id,
                action="attendance_session.created",
                target_id=session_id,
                metadata={"attendance_session_id": session_id},
                created_at=opened_at,
            )
        return session_id

    def open_attendance_sessions(self) -> tuple[tuple[str, str], ...]:
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT id, name FROM attendance_sessions WHERE closed_at IS NULL "
                "ORDER BY opened_at DESC, id DESC"
            ).fetchall()
        return tuple((str(row[0]), str(row[1])) for row in rows)

    def record_attendance(
        self,
        attendance_session_id: str,
        person_id: str,
        *,
        liveness_attempt_id: str,
        match_score: float,
        match_threshold: float,
        ambiguity_margin: float,
        configuration_id: str,
        actor_id: str,
        checked_in_at: datetime,
    ) -> AttendanceWrite:
        values = (match_score, match_threshold, ambiguity_margin)
        if not all(np.isfinite(value) for value in values):
            raise PersistenceError("attendance match values must be finite")
        if not -1.0 <= match_score <= 1.0 or not -1.0 <= match_threshold <= 1.0:
            raise PersistenceError("cosine scores and threshold must be between -1 and 1")
        if ambiguity_margin < 0.0:
            raise PersistenceError("ambiguity margin must be non-negative")
        proposed_id = str(uuid4())
        with self.database.transaction() as connection:
            liveness = connection.execute(
                "SELECT person_id, operation, active_decision, passive_decision "
                "FROM liveness_attempts WHERE id = ?",
                (liveness_attempt_id,),
            ).fetchone()
            if (
                liveness is None
                or liveness[0] != person_id
                or liveness[1] != "check_in"
                or liveness[2] != "passed"
                or liveness[3] != "passed"
            ):
                raise PersistenceError("attendance requires a passed check-in liveness attempt")
            cursor = connection.execute(
                "INSERT INTO attendance_records(id, attendance_session_id, person_id, "
                "liveness_attempt_id, "
                "checked_in_at, match_score, match_threshold, ambiguity_margin, "
                "configuration_version_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(attendance_session_id, person_id) DO NOTHING",
                (
                    proposed_id,
                    attendance_session_id,
                    person_id,
                    liveness_attempt_id,
                    _timestamp(checked_in_at),
                    match_score,
                    match_threshold,
                    ambiguity_margin,
                    configuration_id,
                ),
            )
            duplicate = cursor.rowcount == 0
            if duplicate:
                existing = connection.execute(
                    "SELECT id FROM attendance_records WHERE attendance_session_id = ? "
                    "AND person_id = ?",
                    (attendance_session_id, person_id),
                ).fetchone()
                assert existing is not None
                record_id = str(existing[0])
            else:
                record_id = proposed_id
            self._audit(
                connection,
                actor_id=actor_id,
                action="attendance.duplicate" if duplicate else "attendance.recorded",
                target_id=person_id,
                metadata={
                    "attendance_record_id": record_id,
                    "attendance_session_id": attendance_session_id,
                    "liveness_attempt_id": liveness_attempt_id,
                    "configuration_version_id": configuration_id,
                    "match_score": match_score,
                    "match_threshold": match_threshold,
                    "ambiguity_margin": ambiguity_margin,
                },
                created_at=checked_in_at,
            )
        return AttendanceWrite(record_id, duplicate)

    def erase_person(self, person_id: str, *, actor_id: str, deleted_at: datetime) -> ErasureResult:
        """Hard-delete a person and biometric templates while preserving audit history."""
        with self.database.transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM people WHERE id = ?", (person_id,)
            ).fetchone()
            if exists is None:
                raise PersistenceError("person does not exist")
            template_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM embedding_templates WHERE person_id = ?", (person_id,)
                ).fetchone()[0]
            )
            attendance_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM attendance_records WHERE person_id = ?", (person_id,)
                ).fetchone()[0]
            )
            # GDPR: biometric templates and linked subject records are erased atomically.
            connection.execute("DELETE FROM people WHERE id = ?", (person_id,))
            # AUDIT: event contains identifiers and counts, never biometric payloads.
            self._audit(
                connection,
                actor_id=actor_id,
                action="person.erased",
                target_id=person_id,
                metadata={
                    "deleted_template_count": template_count,
                    "deleted_attendance_count": attendance_count,
                },
                created_at=deleted_at,
            )
        return ErasureResult(person_id, template_count, attendance_count)

    def person_exists(self, person_id: str) -> bool:
        with self.database.connection() as connection:
            return (
                connection.execute("SELECT 1 FROM people WHERE id = ?", (person_id,)).fetchone()
                is not None
            )

    def get_liveness_attempt(self, attempt_id: str) -> LivenessAttempt | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT id, person_id, operation, challenge_sequence_json, "
                "completed_challenges_json, active_decision, passive_decision, failure_reason, "
                "passive_median, passive_minimum, suspicious_frame_count, "
                "processing_failure_count, configuration_version_id, created_at "
                "FROM liveness_attempts WHERE id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            return None
        return LivenessAttempt(
            attempt_id=str(row[0]),
            person_id=row[1],
            operation=str(row[2]),
            challenge_sequence=tuple(json.loads(row[3])),
            completed_challenges=tuple(json.loads(row[4])),
            active_decision=str(row[5]),
            passive_decision=str(row[6]),
            failure_reason=row[7],
            passive_median=row[8],
            passive_minimum=row[9],
            suspicious_frame_count=int(row[10]),
            processing_failure_count=int(row[11]),
            configuration_version_id=str(row[12]),
            created_at=datetime.fromisoformat(row[13]),
        )

    def record_audit(
        self,
        *,
        actor_id: str,
        action: str,
        target_id: str | None,
        metadata: dict[str, Any],
        created_at: datetime,
    ) -> str:
        with self.database.transaction() as connection:
            return self._audit(
                connection,
                actor_id=actor_id,
                action=action,
                target_id=target_id,
                metadata=metadata,
                created_at=created_at,
            )

    def audit_metadata(self, action: str) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT metadata_json FROM audit_events WHERE action = ? "
                "ORDER BY rowid DESC LIMIT 1",
                (action,),
            ).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def _audit(
        self,
        connection: sqlite3.Connection,
        *,
        actor_id: str,
        action: str,
        target_id: str | None,
        metadata: dict[str, Any],
        created_at: datetime,
    ) -> str:
        _reject_raw_biometrics(metadata)
        event_id = str(uuid4())
        connection.execute(
            "INSERT INTO audit_events(id, actor_id, action, target_id, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, actor_id, action, target_id, _json(metadata), _timestamp(created_at)),
        )
        return event_id


def _validate_embedding(template: EmbeddingTemplate) -> np.ndarray:
    vector = np.asarray(template.embedding)
    if vector.dtype != np.float32 or vector.ndim != 1 or vector.size == 0:
        raise PersistenceError("embedding must be a non-empty float32 vector")
    if not np.isfinite(vector).all():
        raise PersistenceError("embedding must contain finite values")
    if not template.normalized or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5):
        raise PersistenceError("embedding must be L2 normalized")
    return np.ascontiguousarray(vector)


def _template_from_row(row: tuple[Any, ...]) -> EmbeddingTemplate:
    blob, dimension, dtype, normalized = row[2], int(row[3]), row[4], bool(row[5])
    if (
        dtype != "float32"
        or not normalized
        or len(blob) != dimension * np.dtype(np.float32).itemsize
    ):
        raise PersistenceError("stored embedding metadata is invalid")
    vector = np.frombuffer(blob, dtype=np.float32).copy()
    if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5):
        raise PersistenceError("stored embedding is invalid")
    quality_data = json.loads(row[7])
    quality_data["warnings"] = tuple(quality_data.get("warnings", ()))
    pose_data = json.loads(row[12]) if len(row) > 12 and row[12] is not None else None
    return EmbeddingTemplate(
        template_id=str(row[0]),
        person_id=str(row[1]),
        embedding=vector,
        model_name=str(row[9]),
        model_version=str(row[10]),
        model_checksum=str(row[11]),
        normalized=True,
        pose_bin=row[6],
        quality=FaceQuality(**quality_data),
        created_at=datetime.fromisoformat(row[8]),
        pose=HeadPose(**pose_data) if pose_data is not None else None,
    )


def _liveness_values(attempt: LivenessAttempt) -> tuple[Any, ...]:
    return (
        attempt.attempt_id,
        attempt.person_id,
        attempt.operation,
        _json(list(attempt.challenge_sequence)),
        _json(list(attempt.completed_challenges)),
        attempt.active_decision,
        attempt.passive_decision,
        attempt.failure_reason,
        attempt.passive_median,
        attempt.passive_minimum,
        attempt.suspicious_frame_count,
        attempt.processing_failure_count,
        attempt.configuration_version_id,
        _timestamp(attempt.created_at),
    )


def _liveness_audit_metadata(attempt: LivenessAttempt) -> dict[str, Any]:
    return {
        "liveness_attempt_id": attempt.attempt_id,
        "operation": attempt.operation,
        "challenge_sequence": list(attempt.challenge_sequence),
        "completed_challenges": list(attempt.completed_challenges),
        "active_decision": attempt.active_decision,
        "passive_decision": attempt.passive_decision,
        "failure_reason": attempt.failure_reason,
        "configuration_version_id": attempt.configuration_version_id,
    }


def _reject_raw_biometrics(value: Any, path: str = "metadata") -> None:
    if isinstance(value, (bytes, bytearray, memoryview, np.ndarray)):
        raise PersistenceError(f"raw biometric payload forbidden at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(
                term in normalized
                for term in ("raw_image", "face_crop", "frame_pixels", "embedding")
            ):
                raise PersistenceError(f"raw biometric audit key forbidden: {key}")
            _reject_raw_biometrics(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_raw_biometrics(item, f"{path}[{index}]")


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PersistenceError("record must be valid JSON") from exc


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise PersistenceError("timestamps must include a timezone")
    return value.isoformat()
