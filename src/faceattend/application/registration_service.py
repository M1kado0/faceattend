"""Headless registration orchestration for the local desktop application."""

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
from faceattend.vision.matcher import ExactNumpyMatcher
from faceattend.vision.types import (
    EmbeddingTemplate,
    EvidenceDecision,
    Frame,
    FrameEvidence,
    HeadPose,
    LivenessEvidence,
    LivenessKind,
    MatchDecision,
    MatchStatus,
    ModelMetadata,
    RegistrationResult,
    RegistrationStatus,
)


class EnrollmentSession(Protocol):
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

    def take_embedding_candidates(
        self, *, min_candidates: int = 3, max_candidates: int = 5
    ) -> tuple[FrameEvidence, ...]: ...

    def cancel(self) -> object: ...


class EnrollmentFaceAnalyzer(Protocol):
    def extract_embedding(
        self,
        evidence: FrameEvidence,
        *,
        active: LivenessEvidence,
        passive: LivenessEvidence,
    ) -> np.ndarray: ...


class RegistrationMatcher(Protocol):
    """The local matcher boundary used to block likely duplicate enrollment."""

    def rebuild(self, templates: Sequence[EmbeddingTemplate]) -> None: ...

    def match(self, embedding: np.ndarray) -> MatchDecision: ...


@dataclass(frozen=True, slots=True)
class RegistrationRequest:
    display_name: str
    consent_granted: bool
    consent_purpose: str
    actor_id: str
    configuration_id: str
    embedding_model_id: str
    embedding_model: ModelMetadata
    active_model_version: str = "face_landmarker"


class RegistrationCoordinator:
    """Coordinate one consented registration without owning camera or GUI state."""

    def __init__(
        self,
        request: RegistrationRequest,
        *,
        session: EnrollmentSession,
        face_analyzer: EnrollmentFaceAnalyzer,
        matcher: RegistrationMatcher | None = None,
        repository: LocalRepository,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.request = request
        self.session = session
        self.face_analyzer = face_analyzer
        # This is intentionally the same policy shape as local check-in. Its
        # numeric values remain provisional while Phase 7 is deferred.
        self.matcher = matcher or ExactNumpyMatcher(match_threshold=0.75, ambiguity_margin=0.05)
        self.repository = repository
        self.now = now or (lambda: datetime.now(UTC))
        self.person_id = str(uuid4())
        self.liveness_attempt_id = str(uuid4())
        self._result: RegistrationResult | None = None
        self.last_evidence: FrameEvidence | None = None
        if not request.consent_granted:
            # PRIVACY: no camera processing begins without explicit consent.
            session.cancel()
            raise ValueError("explicit consent is required before registration")
        if not request.display_name.strip() or not request.consent_purpose.strip():
            session.cancel()
            raise ValueError("display name and consent purpose are required")

    @property
    def result(self) -> RegistrationResult | None:
        return self._result

    def process(self, frame: Frame) -> RegistrationResult | None:
        """Consume a frame and complete persistence only after both liveness gates."""
        if self._result is not None:
            return self._result
        try:
            evidence = self.session(frame)
            self.last_evidence = evidence
        except (ValueError, TypeError, RuntimeError, OSError):
            return self._fail("model_failure")

        phase = str(getattr(self.session.phase, "value", self.session.phase))
        if phase in {"failed", "cancelled"}:
            status = (
                RegistrationStatus.CANCELLED if phase == "cancelled" else RegistrationStatus.FAILED
            )
            self._result = RegistrationResult(
                status,
                None,
                reason=self.session.failure_reason or evidence.failure_reason or phase,
            )
            self.session.cancel()
            return self._result
        if phase != "completed":
            return None

        passive = getattr(self.session.result, "passive", None)
        if (
            self.session.result.decision is not EvidenceDecision.PASSED
            or not isinstance(passive, LivenessEvidence)
            or passive.kind is not LivenessKind.PASSIVE
            or passive.decision is not EvidenceDecision.PASSED
        ):
            return self._fail("liveness_incomplete")
        active = LivenessEvidence(
            LivenessKind.ACTIVE,
            EvidenceDecision.PASSED,
            None,
            None,
            self.request.active_model_version,
        )

        try:
            candidates = self.session.take_embedding_candidates(min_candidates=3, max_candidates=5)
            selected = select_diverse_template_evidence(candidates, maximum=5)
            if len(selected) < 3:
                raise ValueError("insufficient template candidates")
            completed_at = self.now()
            templates = tuple(
                self._template_from_evidence(item, active, passive, completed_at)
                for item in selected
            )
            attempt = self._liveness_attempt(passive, completed_at)
        except (ValueError, TypeError, RuntimeError, OSError):
            return self._fail("template_extraction_failed")

        try:
            self.matcher.rebuild(self.repository.load_compatible_templates(self.request.embedding_model))
            if any(
                self.matcher.match(template.embedding).status
                in {MatchStatus.MATCHED, MatchStatus.AMBIGUOUS}
                for template in templates
            ):
                return self._block_duplicate(completed_at)
        except (ValueError, TypeError, RuntimeError, OSError, sqlite3.Error):
            return self._fail("duplicate_enrollment_check_failed")

        try:
            # AUDIT: repository commits consent, liveness, templates, and audit
            # records together; any failure rolls the entire identity back.
            write = self.repository.register_person_atomically(
                self.request.display_name,
                purpose=self.request.consent_purpose,
                actor_id=self.request.actor_id,
                configuration_id=self.request.configuration_id,
                model_id=self.request.embedding_model_id,
                templates=templates,
                liveness_attempt=attempt,
                completed_at=completed_at,
            )
        except (ValueError, TypeError, RuntimeError, OSError, sqlite3.Error):
            return self._fail("database_failure")

        self._result = RegistrationResult(
            RegistrationStatus.COMPLETED,
            write.person_id,
            tuple(item.template_id for item in templates),
        )
        return self._result

    def _block_duplicate(self, occurred_at: datetime) -> RegistrationResult:
        """Reject a likely duplicate without disclosing the matched identity.

        # PRIVACY: the audit intentionally has no target identity, match score,
        # embedding, or template reference. The current policy is provisional,
        # so this protects against simple duplicate enrollment but is not a
        # calibrated identity-proofing guarantee.
        """
        try:
            self.repository.record_audit(
                actor_id=self.request.actor_id,
                action="enrollment.duplicate_blocked",
                target_id=None,
                metadata={
                    "outcome": "face_already_registered",
                    "policy": "provisional_current_matching_policy",
                },
                created_at=occurred_at,
            )
        except (TypeError, ValueError, OSError, sqlite3.Error):
            return self._fail("database_failure")
        self.session.cancel()
        self._result = RegistrationResult(
            RegistrationStatus.DUPLICATE,
            None,
            reason="face_already_registered",
        )
        return self._result

    def cancel(self) -> RegistrationResult:
        if self._result is None:
            self.session.cancel()
            self._result = RegistrationResult(
                RegistrationStatus.CANCELLED, None, reason="cancelled"
            )
        return self._result

    def retry(self, session: EnrollmentSession) -> None:
        """Replace a terminal attempt with a fresh liveness session."""
        if self._result is None or self._result.status is RegistrationStatus.COMPLETED:
            raise RuntimeError("only a failed or cancelled registration can be retried")
        self.session.cancel()
        self.session = session
        self.liveness_attempt_id = str(uuid4())
        self._result = None
        self.last_evidence = None

    def _fail(self, reason: str) -> RegistrationResult:
        self.session.cancel()
        self._result = RegistrationResult(RegistrationStatus.FAILED, None, reason=reason)
        return self._result

    def _template_from_evidence(
        self,
        evidence: FrameEvidence,
        active: LivenessEvidence,
        passive: LivenessEvidence,
        created_at: datetime,
    ) -> EmbeddingTemplate:
        if evidence.face is None or evidence.quality is None or evidence.pose is None:
            raise ValueError("template evidence must contain one quality-approved face and pose")
        vector = np.asarray(
            self.face_analyzer.extract_embedding(
                evidence,
                active=active,
                passive=passive,
            ),
            dtype=np.float32,
        )
        return EmbeddingTemplate(
            template_id=str(uuid4()),
            person_id=self.person_id,
            embedding=np.ascontiguousarray(vector),
            model_name=self.request.embedding_model.name,
            model_version=self.request.embedding_model.version,
            model_checksum=self.request.embedding_model.checksum,
            normalized=True,
            pose_bin=_pose_bin(evidence.pose),
            quality=evidence.quality,
            created_at=created_at,
            pose=evidence.pose,
        )

    def _liveness_attempt(self, passive: LivenessEvidence, created_at: datetime) -> LivenessAttempt:
        evaluator = self.session.active.evaluator
        sequence = tuple(_action_name(item) for item in evaluator.phases)
        completed = tuple(_action_name(item) for item in evaluator.result.completed_phases)
        return LivenessAttempt(
            self.liveness_attempt_id,
            self.person_id,
            "registration",
            sequence,
            completed,
            "passed",
            "passed",
            None,
            passive.median_score,
            passive.minimum_score,
            passive.suspicious_frame_count,
            passive.failure_to_process_count,
            self.request.configuration_id,
            created_at,
        )


def select_diverse_template_evidence(
    candidates: Sequence[FrameEvidence], *, maximum: int = 5
) -> tuple[FrameEvidence, ...]:
    """Prefer sharp frames, then greedily spread selections across pose space."""
    valid = [
        item
        for item in candidates
        if item.failure_reason is None
        and item.face_count == 1
        and item.face is not None
        and item.quality is not None
        and item.quality.passed
        and item.pose is not None
    ]
    if not valid or maximum <= 0:
        return ()
    first = max(valid, key=lambda item: item.quality.sharpness if item.quality else 0.0)
    selected = [first]
    remaining = [item for item in valid if item is not first]
    while remaining and len(selected) < maximum:
        next_item = max(
            remaining,
            key=lambda item: (
                min(_pose_distance(item.pose, chosen.pose) for chosen in selected),
                item.quality.sharpness if item.quality else 0.0,
            ),
        )
        selected.append(next_item)
        remaining = [item for item in remaining if item is not next_item]
    return tuple(selected)


def _pose_distance(left: HeadPose | None, right: HeadPose | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(
        np.linalg.norm(
            np.array(
                [
                    left.yaw_degrees - right.yaw_degrees,
                    left.pitch_degrees - right.pitch_degrees,
                    left.roll_degrees - right.roll_degrees,
                ]
            )
        )
    )


def _pose_bin(pose: HeadPose) -> str:
    axes = {
        "left" if pose.yaw_degrees < 0 else "right": abs(pose.yaw_degrees),
        "up" if pose.pitch_degrees < 0 else "down": abs(pose.pitch_degrees),
        "roll_left" if pose.roll_degrees < 0 else "roll_right": abs(pose.roll_degrees),
    }
    direction, magnitude = max(axes.items(), key=lambda item: item[1])
    return direction if magnitude >= 5.0 else "neutral"


def _action_name(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)
