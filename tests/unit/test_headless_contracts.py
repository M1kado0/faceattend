"""Contract tests for the first local-first package boundary."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from faceattend.application.states import (
    AttendanceState,
    RegistrationState,
    StateTransitionError,
    new_attendance_state_machine,
    new_registration_state_machine,
)
from faceattend.vision.types import BoundingBox, FaceObservation, FaceQuality, Frame


def test_registration_requires_active_then_passive_liveness() -> None:
    workflow = new_registration_state_machine()

    workflow.transition_to(RegistrationState.ACQUIRING)
    workflow.transition_to(RegistrationState.ACTIVE_LIVENESS)
    with pytest.raises(StateTransitionError):
        workflow.transition_to(RegistrationState.CAPTURING_TEMPLATES)
    workflow.transition_to(RegistrationState.PASSIVE_LIVENESS)
    workflow.transition_to(RegistrationState.CAPTURING_TEMPLATES)


def test_attendance_cannot_match_before_both_liveness_stages() -> None:
    workflow = new_attendance_state_machine()

    workflow.transition_to(AttendanceState.ACQUIRING)
    with pytest.raises(StateTransitionError):
        workflow.transition_to(AttendanceState.MATCHING)
    workflow.transition_to(AttendanceState.ACTIVE_LIVENESS)
    workflow.transition_to(AttendanceState.PASSIVE_LIVENESS)
    workflow.transition_to(AttendanceState.MATCHING)


@pytest.mark.parametrize("terminal", [AttendanceState.UNKNOWN, AttendanceState.AMBIGUOUS])
def test_unknown_or_ambiguous_match_cannot_be_recorded(
    terminal: AttendanceState,
) -> None:
    workflow = new_attendance_state_machine()
    workflow.transition_to(AttendanceState.ACQUIRING)
    workflow.transition_to(AttendanceState.ACTIVE_LIVENESS)
    workflow.transition_to(AttendanceState.PASSIVE_LIVENESS)
    workflow.transition_to(AttendanceState.MATCHING)
    workflow.transition_to(terminal)

    with pytest.raises(StateTransitionError):
        workflow.transition_to(AttendanceState.PERSISTING)


def test_frame_and_observation_are_typed_transient_values() -> None:
    frame = Frame(
        pixels=np.zeros((2, 2, 3), dtype=np.uint8),
        captured_at_ns=10,
        sequence_id=1,
    )
    observation = FaceObservation(
        bbox=BoundingBox(0.0, 0.0, 1.0, 1.0),
        detector_score=0.99,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        quality=FaceQuality(
            passed=True,
            sharpness=1.0,
            brightness=0.5,
            face_area_ratio=0.25,
        ),
    )

    assert frame.pixels.dtype == np.uint8
    assert observation.landmarks.dtype == np.float32
    with pytest.raises(FrozenInstanceError):
        frame.sequence_id = 2  # type: ignore[misc]
