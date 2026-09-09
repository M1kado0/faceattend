"""Headless frame-evidence utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from math import isfinite
from time import monotonic_ns

import numpy as np

from faceattend.vision.continuity import ContinuityStatus, FaceContinuityTracker
from faceattend.vision.types import Frame, FrameEvidence


def evidence_failure(evidence: FrameEvidence) -> str | None:
    """Common structural/quality policy for both liveness consumers."""
    if evidence.failure_reason:
        return evidence.failure_reason
    frame = evidence.frame
    if not isinstance(frame.captured_at_ns, int) or frame.captured_at_ns < 0:
        return "invalid_timestamp"
    if (
        frame.pixels.dtype != np.uint8
        or frame.pixels.ndim != 3
        or frame.pixels.shape[2] != 3
        or not frame.pixels.size
    ):
        return "invalid_frame"
    if evidence.face_count != 1 or evidence.face is None:
        return "no_face" if evidence.face_count == 0 else "multiple_faces"
    face, quality, pose = evidence.face, evidence.quality, evidence.pose
    if pose is None:
        return "pose_unavailable"
    if not all(
        isfinite(v) and abs(v) <= 180
        for v in (pose.yaw_degrees, pose.pitch_degrees, pose.roll_degrees)
    ):
        return "invalid_pose"
    if quality is None:
        return "quality_unavailable"
    if quality != face.quality:
        return "quality_source_mismatch"
    if not all(
        isfinite(v)
        for v in (
            quality.sharpness,
            quality.brightness,
            quality.face_area_ratio,
            quality.dark_fraction,
            quality.bright_fraction,
            quality.center_offset,
            face.detector_score,
        )
    ):
        return "nonfinite_quality"
    if (
        not 0 <= face.detector_score <= 1
        or quality.sharpness < 0
        or not all(
            0 <= v <= 1
            for v in (
                quality.brightness,
                quality.face_area_ratio,
                quality.dark_fraction,
                quality.bright_fraction,
            )
        )
    ):
        return "invalid_quality"
    if not quality.passed or quality.reason:
        return quality.reason or "poor_quality"
    if evidence.lighting_score is None or evidence.lighting_score != quality.brightness:
        return "lighting_source_mismatch"
    if evidence.motion_score is not None and not isfinite(evidence.motion_score):
        return "invalid_motion"
    if face.landmarks.shape != (5, 2) or not np.isfinite(face.landmarks).all():
        return "invalid_alignment_landmarks"
    if face.track_id is not None and face.track_id != evidence.track_id:
        return "track_source_mismatch"
    box = face.bbox
    if not all(isfinite(v) for v in (box.x_min, box.y_min, box.x_max, box.y_max)):
        return "invalid_face_box"
    if box.x_max <= box.x_min or box.y_max <= box.y_min:
        return "invalid_face_box"
    return None


class EvidenceStreamGuard:
    """One continuity/timestamp owner per attempt; exact duplicates fail closed.

    This detects frozen/repeated pixels, not general replay or identity-preserving
    substitution. Geometric tracking is not biometric identity verification.
    """

    def __init__(
        self,
        *,
        tracker: FaceContinuityTracker | None = None,
        clock_ns: Callable[[], int] = monotonic_ns,
        max_age_ms: int = 1000,
    ) -> None:
        if max_age_ms <= 0:
            raise ValueError("max_age_ms must be positive")
        self.tracker = tracker or FaceContinuityTracker()
        self.clock_ns = clock_ns
        self.max_age_ns = max_age_ms * 1_000_000
        self.duplicates = DuplicateFrameDetector()
        self._last_ns: int | None = None
        self._last_sequence: int | None = None
        self._failure: str | None = None

    def observe(self, evidence: FrameEvidence) -> FrameEvidence:
        frame = evidence.frame
        reason = self._failure
        if reason is None:
            now = self.clock_ns()
            if not isinstance(frame.captured_at_ns, int) or frame.captured_at_ns < 0:
                reason = "invalid_timestamp"
            elif self._last_ns is not None and frame.captured_at_ns <= self._last_ns:
                reason = "timestamp_not_monotonic"
            elif frame.captured_at_ns > now:
                reason = "future_frame"
            elif now - frame.captured_at_ns > self.max_age_ns:
                reason = "stale_frame"
            elif not isinstance(frame.sequence_id, int) or frame.sequence_id < 0:
                reason = "invalid_sequence_id"
            elif self._last_sequence is not None and frame.sequence_id <= self._last_sequence:
                reason = "sequence_not_monotonic"
            else:
                reason = evidence_failure(evidence)
        if reason is None:
            assert evidence.face is not None
            face = replace(evidence.face, track_id=evidence.track_id)
            continuity = self.tracker.observe(frame, [face])
            if continuity.status is not ContinuityStatus.ACCEPTED:
                reason = continuity.reason or "continuity_failed"
            elif self.duplicates.observe(frame):
                reason = "duplicate_frame"
        self._last_ns, self._last_sequence = frame.captured_at_ns, frame.sequence_id
        self._failure = reason
        return replace(evidence, failure_reason=reason)

    def cancel(self) -> None:
        self._failure = "cancelled"


class DuplicateFrameDetector:
    """Detect repeated pixel frames as a camera/replay warning signal."""

    def __init__(self, *, history_size: int = 3) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self.history_size = history_size
        self._recent: list[str] = []

    def reset(self) -> None:
        self._recent.clear()

    def observe(self, frame: Frame) -> bool:
        fingerprint = self.fingerprint(frame)
        duplicate = fingerprint in self._recent
        self._recent.append(fingerprint)
        if len(self._recent) > self.history_size:
            self._recent.pop(0)
        return duplicate

    @staticmethod
    def fingerprint(frame: Frame) -> str:
        return hashlib.sha256(np.ascontiguousarray(frame.pixels).tobytes()).hexdigest()
