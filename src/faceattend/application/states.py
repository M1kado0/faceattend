"""Explicit states for local registration and attendance workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Generic, TypeVar


class RegistrationState(StrEnum):
    IDLE = "idle"
    ACQUIRING = "acquiring"
    ACTIVE_LIVENESS = "active_liveness"
    PASSIVE_LIVENESS = "passive_liveness"
    CAPTURING_TEMPLATES = "capturing_templates"
    PERSISTING = "persisting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AttendanceState(StrEnum):
    IDLE = "idle"
    ACQUIRING = "acquiring"
    ACTIVE_LIVENESS = "active_liveness"
    PASSIVE_LIVENESS = "passive_liveness"
    MATCHING = "matching"
    PERSISTING = "persisting"
    RECORDED = "recorded"
    DUPLICATE = "duplicate"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"
    CANCELLED = "cancelled"


StateT = TypeVar("StateT", bound=Enum)


class StateTransitionError(ValueError):
    """Raised when a workflow attempts an undefined transition."""


@dataclass(slots=True)
class WorkflowStateMachine(Generic[StateT]):
    """Small headless state machine shared by GUI and test workflows."""

    state: StateT
    transitions: Mapping[StateT, frozenset[StateT]]

    def can_transition_to(self, target: StateT) -> bool:
        return target in self.transitions.get(self.state, frozenset())

    def transition_to(self, target: StateT) -> None:
        if not self.can_transition_to(target):
            raise StateTransitionError(f"invalid transition: {self.state.value} -> {target.value}")
        self.state = target


REGISTRATION_TRANSITIONS: Mapping[RegistrationState, frozenset[RegistrationState]] = {
    RegistrationState.IDLE: frozenset({RegistrationState.ACQUIRING}),
    RegistrationState.ACQUIRING: frozenset(
        {
            RegistrationState.ACTIVE_LIVENESS,
            RegistrationState.FAILED,
            RegistrationState.CANCELLED,
        }
    ),
    RegistrationState.ACTIVE_LIVENESS: frozenset(
        {
            RegistrationState.PASSIVE_LIVENESS,
            RegistrationState.FAILED,
            RegistrationState.CANCELLED,
        }
    ),
    RegistrationState.PASSIVE_LIVENESS: frozenset(
        {
            RegistrationState.CAPTURING_TEMPLATES,
            RegistrationState.FAILED,
            RegistrationState.CANCELLED,
        }
    ),
    RegistrationState.CAPTURING_TEMPLATES: frozenset(
        {
            RegistrationState.PERSISTING,
            RegistrationState.FAILED,
            RegistrationState.CANCELLED,
        }
    ),
    RegistrationState.PERSISTING: frozenset(
        {RegistrationState.COMPLETED, RegistrationState.FAILED}
    ),
}


ATTENDANCE_TRANSITIONS: Mapping[AttendanceState, frozenset[AttendanceState]] = {
    AttendanceState.IDLE: frozenset({AttendanceState.ACQUIRING}),
    AttendanceState.ACQUIRING: frozenset(
        {
            AttendanceState.ACTIVE_LIVENESS,
            AttendanceState.FAILED,
            AttendanceState.CANCELLED,
        }
    ),
    AttendanceState.ACTIVE_LIVENESS: frozenset(
        {
            AttendanceState.PASSIVE_LIVENESS,
            AttendanceState.FAILED,
            AttendanceState.CANCELLED,
        }
    ),
    AttendanceState.PASSIVE_LIVENESS: frozenset(
        {
            AttendanceState.MATCHING,
            AttendanceState.FAILED,
            AttendanceState.CANCELLED,
        }
    ),
    AttendanceState.MATCHING: frozenset(
        {
            AttendanceState.PERSISTING,
            AttendanceState.UNKNOWN,
            AttendanceState.AMBIGUOUS,
            AttendanceState.FAILED,
        }
    ),
    AttendanceState.PERSISTING: frozenset(
        {
            AttendanceState.RECORDED,
            AttendanceState.DUPLICATE,
            AttendanceState.FAILED,
        }
    ),
}


def new_registration_state_machine() -> WorkflowStateMachine[RegistrationState]:
    return WorkflowStateMachine(RegistrationState.IDLE, REGISTRATION_TRANSITIONS)


def new_attendance_state_machine() -> WorkflowStateMachine[AttendanceState]:
    return WorkflowStateMachine(AttendanceState.IDLE, ATTENDANCE_TRANSITIONS)
