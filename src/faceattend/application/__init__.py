"""Application workflows that coordinate headless FaceAttend interfaces."""

from faceattend.application.states import (
    AttendanceState,
    RegistrationState,
    StateTransitionError,
    WorkflowStateMachine,
)

__all__ = [
    "AttendanceState",
    "RegistrationState",
    "StateTransitionError",
    "WorkflowStateMachine",
]
