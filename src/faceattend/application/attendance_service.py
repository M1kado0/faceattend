"""Headless liveness-gated attendance orchestration for the local application."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

import numpy as np

from faceattend.persistence.records import LivenessAttempt
from faceattend.persistence.repositories import LocalRepository
from faceattend.vision.types import (
    AttendanceResult,
    AttendanceStatus,
    EmbeddingTemplate,
    EvidenceDecision,
    Frame,
    FrameEvidence,
    LivenessEvidence,
    LivenessKind,
    MatchDecision,
    MatchStatus,
    ModelMetadata,
)


class AttendanceLivenessSession(Protocol):
    @property
    def phase(self) -> Any: ...

    @property
    def result(self) -> Any: ...

    @property
    def failure_reason(self) -> str | None: ...

    @property
    def active(self) -> Any: ...

    @property
    def instruction(self) -> str: ...

    def __call__(self, frame: Frame) -> FrameEvidence: ...

    def take_embedding_candidate(self) -> FrameEvidence: ...

    def cancel(self) -> object: ...


class AttendanceFaceAnalyzer(Protocol):
    def extract_embedding(
        self,
        evidence: FrameEvidence,
        *,
        active: LivenessEvidence,
        passive: LivenessEvidence,
    ) -> np.ndarray: ...


class AttendanceMatcher(Protocol):
    match_threshold: float
    ambiguity_margin: float

    def rebuild(self, templates: Sequence[EmbeddingTemplate]) -> None: ...

    def match(self, embedding: np.ndarray) -> MatchDecision: ...


@dataclass(frozen=True, slots=True)
class AttendanceRequest:
    attendance_session_id: str
    actor_id: str
    configuration_id: str
    embedding_model: ModelMetadata
    active_model_version: str = "face_landmarker"


class AttendanceCoordinator:
    """Consume one explicit check-in; liveness always precedes matching and writes."""

    def __init__(
        self,
        request: AttendanceRequest,
        *,
        session: AttendanceLivenessSession,
        face_analyzer: AttendanceFaceAnalyzer,
        matcher: AttendanceMatcher,
        repository: LocalRepository,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not request.attendance_session_id.strip() or not request.actor_id.strip():
            session.cancel()
            raise ValueError("attendance session and actor are required")
        self.request = request
        self.session = session
        self.face_analyzer = face_analyzer
        self.matcher = matcher
        self.repository = repository
        self.now = now or (lambda: datetime.now(UTC))
        self.liveness_attempt_id = str(uuid4())
        self.last_evidence: FrameEvidence | None = None
        self._result: AttendanceResult | None = None
        # PRIVACY: only active-consent templates matching the exact model identity load.
        self.matcher.rebuild(repository.load_compatible_templates(request.embedding_model))

    @property
    def result(self) -> AttendanceResult | None:
        return self._result

    def process(self, frame: Frame) -> AttendanceResult | None:
        if self._result is not None:
            return self._result
        try:
            evidence = self.session(frame)
            self.last_evidence = evidence
        except (ValueError, TypeError, RuntimeError, OSError):
            return self._reject(AttendanceStatus.FAILED, "model_failure")

        phase = str(getattr(self.session.phase, "value", self.session.phase))
        passive = getattr(self.session.result, "passive", None)
        if phase in {"failed", "cancelled"}:
            status = AttendanceStatus.CANCELLED if phase == "cancelled" else AttendanceStatus.FAILED
            return self._reject(
                status,
                self.session.failure_reason or evidence.failure_reason or phase,
                passive=passive if isinstance(passive, LivenessEvidence) else None,
            )
        if phase != "completed":
            return None

        if (
            self.session.result.decision is not EvidenceDecision.PASSED
            or not isinstance(passive, LivenessEvidence)
            or passive.kind is not LivenessKind.PASSIVE
            or passive.decision is not EvidenceDecision.PASSED
        ):
            return self._reject(AttendanceStatus.FAILED, "liveness_incomplete")
        active = LivenessEvidence(
            LivenessKind.ACTIVE,
            EvidenceDecision.PASSED,
            None,
            None,
            self.request.active_model_version,
        )
        try:
            candidate = self.session.take_embedding_candidate()
            embedding = np.asarray(
                self.face_analyzer.extract_embedding(
                    candidate,
                    active=active,
                    passive=passive,
                ),
                dtype=np.float32,
            )
            match = self.matcher.match(embedding)
        except (ValueError, TypeError, RuntimeError, OSError):
            return self._reject(AttendanceStatus.FAILED, "model_failure", passive=passive)

        if match.status is MatchStatus.UNKNOWN:
            return self._reject(AttendanceStatus.UNKNOWN, "unknown_identity", passive, match)
        if match.status is MatchStatus.AMBIGUOUS:
            return self._reject(AttendanceStatus.AMBIGUOUS, "ambiguous_identity", passive, match)
        assert match.best is not None
        completed_at = self.now()
        attempt = self._liveness_attempt(passive, completed_at, person_id=match.best.person_id)
        try:
            # AUDIT: passed liveness, attendance decision, and audit records share one commit.
            write = self.repository.record_check_in_atomically(
                self.request.attendance_session_id,
                match.best.person_id,
                attempt=attempt,
                match_score=match.best.score,
                match_threshold=match.match_threshold,
                ambiguity_margin=match.ambiguity_margin,
                actor_id=self.request.actor_id,
                checked_in_at=completed_at,
            )
        except (ValueError, TypeError, RuntimeError, OSError, sqlite3.Error):
            self.session.cancel()
            self._result = AttendanceResult(
                AttendanceStatus.FAILED, None, None, match, "database_failure"
            )
            return self._result
        self._result = AttendanceResult(
            AttendanceStatus.DUPLICATE if write.duplicate else AttendanceStatus.RECORDED,
            match.best.person_id,
            write.record_id,
            match,
        )
        return self._result

    def cancel(self) -> AttendanceResult:
        if self._result is None:
            self.session.cancel()
            self._result = AttendanceResult(
                AttendanceStatus.CANCELLED, None, None, None, "cancelled"
            )
        return self._result

    def _reject(
        self,
        status: AttendanceStatus,
        reason: str,
        passive: LivenessEvidence | None = None,
        match: MatchDecision | None = None,
    ) -> AttendanceResult:
        completed_at = self.now()
        liveness_reason = reason if status is AttendanceStatus.FAILED else None
        attempt = self._liveness_attempt(
            passive, completed_at, person_id=None, reason=liveness_reason
        )
        if status is not AttendanceStatus.CANCELLED:
            try:
                self.repository.record_check_in_rejection(
                    attempt,
                    outcome=(
                        status.value
                        if status in {AttendanceStatus.UNKNOWN, AttendanceStatus.AMBIGUOUS}
                        else "failed"
                    ),
                    actor_id=self.request.actor_id,
                    match_score=match.best.score if match and match.best else None,
                    match_threshold=match.match_threshold if match else None,
                    ambiguity_margin=match.ambiguity_margin if match else None,
                )
            except (ValueError, TypeError, RuntimeError, OSError, sqlite3.Error):
                reason = "database_failure"
                status = AttendanceStatus.FAILED
        self.session.cancel()
        self._result = AttendanceResult(status, None, None, match, reason)
        return self._result

    def _liveness_attempt(
        self,
        passive: LivenessEvidence | None,
        created_at: datetime,
        *,
        person_id: str | None,
        reason: str | None = None,
    ) -> LivenessAttempt:
        evaluator = self.session.active.evaluator
        sequence = tuple(_action_name(item) for item in evaluator.phases)
        completed = tuple(_action_name(item) for item in evaluator.result.completed_phases)
        active_passed = bool(sequence) and completed == sequence
        passive_decision = passive.decision.value if passive is not None else "not_run"
        return LivenessAttempt(
            self.liveness_attempt_id,
            person_id,
            "check_in",
            sequence,
            completed,
            "passed" if active_passed else "failed",
            passive_decision,
            reason,
            passive.median_score if passive else None,
            passive.minimum_score if passive else None,
            passive.suspicious_frame_count if passive else 0,
            passive.failure_to_process_count if passive else 0,
            self.request.configuration_id,
            created_at,
        )


def _action_name(value: object) -> str:
    return str(getattr(value, "value", value))
