"""Typed records stored by the local persistence layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class LivenessAttempt:
    attempt_id: str
    person_id: str | None
    operation: str
    challenge_sequence: tuple[str, ...]
    completed_challenges: tuple[str, ...]
    active_decision: str
    passive_decision: str
    failure_reason: str | None
    passive_median: float | None
    passive_minimum: float | None
    suspicious_frame_count: int
    processing_failure_count: int
    configuration_version_id: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AttendanceWrite:
    record_id: str
    duplicate: bool


@dataclass(frozen=True, slots=True)
class ErasureResult:
    person_id: str
    template_count: int
    attendance_count: int


@dataclass(frozen=True, slots=True)
class EnrollmentWrite:
    person_id: str
    consent_id: str
    enrollment_id: str
