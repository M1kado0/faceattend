"""Desktop-session adapter for the headless registration coordinator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from faceattend.application.mediapipe_session import MediaPipeDesktopSessionProcessor
from faceattend.application.registration_service import RegistrationCoordinator
from faceattend.application.runtime import DesktopMode, RuntimeStatus, SessionPresentation
from faceattend.vision.types import Frame, FrameEvidence, RegistrationStatus


class RegistrationDesktopSessionProcessor:
    """Expose registration to the Phase 4 inference-worker contract."""

    def __init__(
        self,
        coordinator: RegistrationCoordinator,
        *,
        close: Callable[[], object] | None = None,
    ) -> None:
        self.coordinator = coordinator
        self._presentation = MediaPipeDesktopSessionProcessor(
            DesktopMode.REGISTRATION,
            coordinator.session,
            close=close,
        )

    def __call__(self, frame: Frame) -> FrameEvidence:
        self.coordinator.process(frame)
        evidence = self.coordinator.last_evidence
        if evidence is None:
            raise RuntimeError("registration produced no frame evidence")
        return evidence

    def presentation(self, evidence: FrameEvidence) -> SessionPresentation:
        presentation = self._presentation.presentation(evidence)
        result = self.coordinator.result
        if result is None:
            return presentation
        status = {
            RegistrationStatus.COMPLETED: RuntimeStatus.COMPLETED,
            RegistrationStatus.FAILED: RuntimeStatus.FAILED,
            RegistrationStatus.CANCELLED: RuntimeStatus.CANCELLED,
        }[result.status]
        return replace(
            presentation,
            status=status,
            instruction=("Registration complete" if status is RuntimeStatus.COMPLETED else "Retry"),
            failure_reason=result.reason,
        )

    def cancel(self) -> None:
        self.coordinator.cancel()

    def close(self) -> None:
        if self.coordinator.result is None:
            self.coordinator.cancel()
        self._presentation.close()
