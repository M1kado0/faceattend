"""Desktop-session adapter for the headless registration coordinator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from faceattend.application.mediapipe_session import MediaPipeDesktopSessionProcessor
from faceattend.application.registration_service import RegistrationCoordinator
from faceattend.application.runtime import DesktopMode, RuntimeStatus, SessionPresentation
from faceattend.vision.types import Frame, FrameEvidence, LivenessEvidence, RegistrationStatus


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
            RegistrationStatus.DUPLICATE: RuntimeStatus.FAILED,
            RegistrationStatus.FAILED: RuntimeStatus.FAILED,
            RegistrationStatus.CANCELLED: RuntimeStatus.CANCELLED,
        }[result.status]
        failure_reason = _registration_failure_reason(
            result.reason,
            getattr(self.coordinator.session.result, "passive", None),
        )
        return replace(
            presentation,
            status=status,
            instruction=(
                "Registration complete"
                if status is RuntimeStatus.COMPLETED
                else "Face already registered"
                if result.reason == "face_already_registered"
                else "Retry"
            ),
            failure_reason=failure_reason,
        )

    def cancel(self) -> None:
        self.coordinator.cancel()

    def close(self) -> None:
        if self.coordinator.result is None:
            self.coordinator.cancel()
        self._presentation.close()


def _registration_failure_reason(
    reason: str | None,
    passive: object,
) -> str | None:
    """Give the operator PAD numbers needed to diagnose a failed local attempt.

    # PRIVACY: this deliberately exposes only aggregate model scores, never
    # frames, landmarks, embeddings, or any other biometric payload.
    """
    if not isinstance(passive, LivenessEvidence) or passive.reason is None:
        return reason
    if passive.reason != "liveness_score_below_threshold":
        return passive.reason
    minimum = "n/a" if passive.minimum_score is None else f"{passive.minimum_score:.3f}"
    median = "n/a" if passive.median_score is None else f"{passive.median_score:.3f}"
    threshold = "n/a" if passive.threshold is None else f"{passive.threshold:.3f}"
    return f"PAD score below threshold (min {minimum}, median {median}, threshold {threshold})"
