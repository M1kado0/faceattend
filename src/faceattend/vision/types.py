"""Typed values exchanged across capture, CV, application, and evaluation layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Float32Array: TypeAlias = NDArray[np.float32]
UInt8Array: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Frame:
    """Transient camera frame; ordinary workflows must not persist its pixels."""

    pixels: UInt8Array
    captured_at_ns: int
    sequence_id: int


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True, slots=True)
class FaceQuality:
    passed: bool
    sharpness: float
    brightness: float
    face_area_ratio: float
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class HeadPose:
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float


@dataclass(frozen=True, slots=True)
class FaceObservation:
    bbox: BoundingBox
    detector_score: float
    landmarks: Float32Array
    quality: FaceQuality
    track_id: str | None = None


@dataclass(frozen=True, slots=True)
class PoseEvidence:
    pose: HeadPose
    pose_bin: str | None
    stable_for_ms: int
    accepted: bool
    reason: str | None = None


class EvidenceDecision(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class LivenessKind(StrEnum):
    ACTIVE = "active"
    PASSIVE = "passive"


@dataclass(frozen=True, slots=True)
class LivenessEvidence:
    kind: LivenessKind
    decision: EvidenceDecision
    score: float | None
    threshold: float | None
    model_version: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingTemplate:
    template_id: str
    person_id: str
    embedding: Float32Array
    model_name: str
    model_version: str
    model_checksum: str
    normalized: bool
    pose_bin: str | None
    quality: FaceQuality
    created_at: datetime


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    person_id: str
    template_id: str
    score: float


class MatchStatus(StrEnum):
    MATCHED = "matched"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class MatchDecision:
    status: MatchStatus
    best: MatchCandidate | None
    second_best: MatchCandidate | None
    match_threshold: float
    ambiguity_margin: float


class RegistrationStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    status: RegistrationStatus
    person_id: str | None
    template_ids: tuple[str, ...] = ()
    reason: str | None = None


class AttendanceStatus(StrEnum):
    RECORDED = "recorded"
    DUPLICATE = "duplicate"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AttendanceResult:
    status: AttendanceStatus
    person_id: str | None
    attendance_record_id: str | None
    match: MatchDecision | None
    reason: str | None = None
