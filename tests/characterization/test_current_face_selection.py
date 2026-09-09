"""Characterize face-count behavior that the desktop pipeline must preserve."""

import pytest

from ml.pipeline.face import (
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    get_single_face,
)


def test_current_pipeline_rejects_no_face() -> None:
    with pytest.raises(NoFaceDetectedError) as exc_info:
        get_single_face([])

    assert exc_info.value.args == (422, "no_faces_detected")


def test_current_pipeline_rejects_multiple_faces() -> None:
    with pytest.raises(MultipleFacesDetectedError) as exc_info:
        get_single_face([object(), object()])

    assert exc_info.value.args == (422, "multiple_faces_detected")


def test_current_pipeline_accepts_exactly_one_face() -> None:
    face = object()

    assert get_single_face([face]) is face
