"""Desktop-session adapter for the headless attendance coordinator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from faceattend.application.attendance_service import AttendanceCoordinator
from faceattend.application.mediapipe_session import MediaPipeDesktopSessionProcessor
from faceattend.application.runtime import DesktopMode, RuntimeStatus, SessionPresentation
from faceattend.vision.types import AttendanceStatus, Frame, FrameEvidence


class AttendanceDesktopSessionProcessor:
    """Expose one explicit check-in to the existing inference-worker contract."""

    def __init__(
        self,
        coordinator: AttendanceCoordinator,
        *,
        close: Callable[[], object] | None = None,
    ) -> None:
        self.coordinator = coordinator
        self._presentation = MediaPipeDesktopSessionProcessor(
            DesktopMode.ATTENDANCE, coordinator.session, close=close
        )

    def __call__(self, frame: Frame) -> FrameEvidence:
        self.coordinator.process(frame)
        evidence = self.coordinator.last_evidence
        if evidence is None:
            raise RuntimeError("attendance produced no frame evidence")
        return evidence

    def presentation(self, evidence: FrameEvidence) -> SessionPresentation:
        presentation = self._presentation.presentation(evidence)
        result = self.coordinator.result
        if result is None:
            return presentation
        status = (
            RuntimeStatus.COMPLETED
            if result.status
            in {
                AttendanceStatus.RECORDED,
                AttendanceStatus.DUPLICATE,
            }
            else RuntimeStatus.CANCELLED
            if result.status is AttendanceStatus.CANCELLED
            else RuntimeStatus.FAILED
        )
        instruction = {
            AttendanceStatus.RECORDED: "Attendance recorded",
            AttendanceStatus.DUPLICATE: "Already checked in",
            AttendanceStatus.UNKNOWN: "Unknown face",
            AttendanceStatus.AMBIGUOUS: "Face match is ambiguous",
        }.get(result.status, "Retry")
        return replace(
            presentation,
            status=status,
            instruction=instruction,
            failure_reason=result.reason,
        )

    def cancel(self) -> None:
        self.coordinator.cancel()

    def close(self) -> None:
        if self.coordinator.result is None:
            self.coordinator.cancel()
        self._presentation.close()
