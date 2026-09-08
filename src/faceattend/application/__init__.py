"""Application workflows that coordinate headless FaceAttend interfaces."""

from faceattend.application.registration_service import (
    RegistrationCoordinator,
    RegistrationRequest,
)
from faceattend.application.registration_session import RegistrationDesktopSessionProcessor
from faceattend.application.runtime import (
    CallbackSessionProcessor,
    DesktopLifecycle,
    DesktopMode,
    RuntimeStatus,
    SessionPresentation,
    SessionProcessor,
)
from faceattend.application.states import (
    AttendanceState,
    RegistrationState,
    StateTransitionError,
    WorkflowStateMachine,
)

__all__ = [
    "AttendanceState",
    "CallbackSessionProcessor",
    "DesktopLifecycle",
    "DesktopMode",
    "RegistrationState",
    "RegistrationCoordinator",
    "RegistrationDesktopSessionProcessor",
    "RegistrationRequest",
    "RuntimeStatus",
    "SessionPresentation",
    "SessionProcessor",
    "StateTransitionError",
    "WorkflowStateMachine",
]
