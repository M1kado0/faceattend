"""Typed values exchanged across capture, CV, application, and evaluation layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from faceattend.vision.challenge_session import ChallengeAction

Float32Array: TypeAlias = NDArray[np.float32]
UInt8Array: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Identity of loaded weights; embeddings from different versions are incomparable."""

    name: str
    version: str
    checksum: str


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
    dark_fraction: float = 0.0
    bright_fraction: float = 0.0
    center_offset: float = 0.0
    warnings: tuple[str, ...] = ()


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
class FrameEvidence:
    """Immutable per-frame CV evidence shared by active and passive gates.

    This is data only: model inference, GUI updates, persistence, and Qt
    signals remain outside the evidence boundary.
    """

    frame: Frame
    face_count: int
    face: FaceObservation | None
    track_id: str | None
    pose: HeadPose | None
    quality: FaceQuality | None = None
    lighting_score: float | None = None
    motion_score: float | None = None
    embedding: Float32Array | None = None
    frame_fingerprint: str | None = None
    action: ChallengeAction | None = None
    smile_score: float | None = None
    passive: LivenessEvidence | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.face_count < 0:
            raise ValueError("face_count must be non-negative")
        if self.face_count == 1 and self.face is None:
            raise ValueError("one face count requires a face observation")
        if self.face_count != 1 and self.face is not None:
            raise ValueError("face observation requires exactly one face")
        if self.lighting_score is not None and not 0.0 <= self.lighting_score <= 1.0:
            raise ValueError("lighting_score must be between 0 and 1")
        if self.motion_score is not None and self.motion_score < 0.0:
            raise ValueError("motion_score must be non-negative")
        if self.smile_score is not None and not 0.0 <= self.smile_score <= 1.0:
            raise ValueError("smile_score must be between 0 and 1")


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
    median_score: float | None = None
    minimum_score: float | None = None
    suspicious_frame_count: int = 0
    failure_to_process_count: int = 0


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
    pose: HeadPose | None = None


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
    DUPLICATE = "duplicate"
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
