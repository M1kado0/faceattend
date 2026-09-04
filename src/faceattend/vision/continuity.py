"""Fail-closed temporal continuity checks for active-liveness sequences."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from faceattend.vision.types import FaceObservation, Frame


class ContinuityStatus(StrEnum):
    ACCEPTED = "accepted"
    WAITING = "waiting"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ContinuityResult:
    status: ContinuityStatus
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FaceContinuityConfig:
    max_gap_ms: int = 400
    max_center_jump_ratio: float = 0.35
    max_area_change_ratio: float = 0.75


def _center(face: FaceObservation) -> tuple[float, float]:
    box = face.bbox
    return ((box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0)


def _area(face: FaceObservation) -> float:
    box = face.bbox
    return max(0.0, box.x_max - box.x_min) * max(0.0, box.y_max - box.y_min)


class FaceContinuityTracker:
    """Track one face; no-face gaps are temporary only within configured bounds."""

    def __init__(self, config: FaceContinuityConfig | None = None) -> None:
        self.config = config or FaceContinuityConfig()
        self._last_frame_ns: int | None = None
        self._last_valid_timestamp_ns: int | None = None
        self._last_face: FaceObservation | None = None

    def reset(self) -> None:
        self._last_frame_ns = None
        self._last_valid_timestamp_ns = None
        self._last_face = None

    def observe(
        self,
        frame: Frame,
        faces: tuple[FaceObservation, ...] | list[FaceObservation],
    ) -> ContinuityResult:
        if self._last_frame_ns is not None and frame.captured_at_ns <= self._last_frame_ns:
            return ContinuityResult(ContinuityStatus.FAILED, "timestamp_not_monotonic")
        self._last_frame_ns = frame.captured_at_ns

        if len(faces) == 0:
            if self._last_face is None:
                return ContinuityResult(ContinuityStatus.FAILED, "no_face")
            assert self._last_valid_timestamp_ns is not None
            gap_ms = (frame.captured_at_ns - self._last_valid_timestamp_ns) / 1_000_000
            if gap_ms <= self.config.max_gap_ms:
                return ContinuityResult(ContinuityStatus.WAITING, "face_temporarily_missing")
            return ContinuityResult(ContinuityStatus.FAILED, "face_missing_too_long")
        if len(faces) > 1:
            return ContinuityResult(ContinuityStatus.FAILED, "multiple_faces")

        face = faces[0]
        if self._last_face is not None:
            if (
                self._last_face.track_id is not None
                and face.track_id is not None
                and self._last_face.track_id != face.track_id
            ):
                return ContinuityResult(ContinuityStatus.FAILED, "track_changed")
            previous_center = _center(self._last_face)
            current_center = _center(face)
            height, width = frame.pixels.shape[:2]
            diagonal = max(1.0, (width**2 + height**2) ** 0.5)
            jump = (
                (current_center[0] - previous_center[0]) ** 2
                + (current_center[1] - previous_center[1]) ** 2
            ) ** 0.5 / diagonal
            if jump > self.config.max_center_jump_ratio:
                return ContinuityResult(ContinuityStatus.FAILED, "implausible_face_jump")
            previous_area = _area(self._last_face)
            current_area = _area(face)
            if previous_area <= 0 or current_area <= 0:
                return ContinuityResult(ContinuityStatus.FAILED, "invalid_face_area")
            area_change = abs(current_area - previous_area) / previous_area
            if area_change > self.config.max_area_change_ratio:
                return ContinuityResult(ContinuityStatus.FAILED, "implausible_face_scale_change")

        self._last_face = face
        self._last_valid_timestamp_ns = frame.captured_at_ns
        return ContinuityResult(ContinuityStatus.ACCEPTED)
