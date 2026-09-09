import pytest

from faceattend.vision.active_liveness import (
    ActiveLivenessCalibration,
    ActiveLivenessChallengeEvaluator,
    ChallengePhase,
    ChallengeStatus,
)
from faceattend.vision.types import HeadPose


def _pose(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> HeadPose:
    return HeadPose(yaw, pitch, roll)


def test_ordered_turn_challenge_completes_after_dwell() -> None:
    evaluator = ActiveLivenessChallengeEvaluator(
        (ChallengePhase.NEUTRAL, ChallengePhase.LEFT, ChallengePhase.NEUTRAL, ChallengePhase.RIGHT),
        ActiveLivenessCalibration(dwell_ms=100),
    )
    observations = [
        (0, _pose()),
        (100, _pose()),
        (200, _pose(yaw=-40)),
        (300, _pose(yaw=-40)),
        (400, _pose()),
        (500, _pose()),
        (600, _pose(yaw=40)),
        (700, _pose(yaw=40)),
    ]
    for timestamp, pose in observations:
        result = evaluator.observe(pose, timestamp, track_id="person")

    assert result.status is ChallengeStatus.COMPLETED


def test_wrong_direction_and_insufficient_dwell_fail_closed() -> None:
    wrong = ActiveLivenessChallengeEvaluator((ChallengePhase.LEFT,))
    assert wrong.observe(_pose(yaw=40), 0).status is ChallengeStatus.WRONG_DIRECTION

    short = ActiveLivenessChallengeEvaluator(
        (ChallengePhase.LEFT,), ActiveLivenessCalibration(dwell_ms=300, timeout_ms=400)
    )
    short.observe(_pose(), 0, track_id="person")
    short.observe(_pose(yaw=-40), 100, track_id="person")
    result = short.observe(_pose(), 500, track_id="person")
    assert result.status is ChallengeStatus.INSUFFICIENT_DWELL


def test_challenge_timeout_is_explicit() -> None:
    evaluator = ActiveLivenessChallengeEvaluator(
        (ChallengePhase.NEUTRAL,), ActiveLivenessCalibration(timeout_ms=100)
    )
    evaluator.observe(_pose(yaw=20), 0, track_id="person")
    result = evaluator.observe(_pose(yaw=20), 101, track_id="person")

    assert result.status is ChallengeStatus.TIMEOUT


def test_direction_requires_neutral_return_before_opposite_turn() -> None:
    evaluator = ActiveLivenessChallengeEvaluator(
        (ChallengePhase.LEFT, ChallengePhase.RIGHT),
        ActiveLivenessCalibration(dwell_ms=100, min_valid_frames=2),
    )
    for timestamp, yaw in ((0, 0), (100, -40), (200, -40)):
        evaluator.observe(_pose(yaw=yaw), timestamp, track_id="person")
    assert evaluator.result.completed_phases == (ChallengePhase.LEFT,)
    evaluator.observe(_pose(yaw=40), 300, track_id="person")
    assert evaluator.result.completed_phases == (ChallengePhase.LEFT,)
    evaluator.observe(_pose(), 400, track_id="person")
    evaluator.observe(_pose(yaw=40), 500, track_id="person")
    result = evaluator.observe(_pose(yaw=40), 600, track_id="person")
    assert result.status is ChallengeStatus.COMPLETED


def test_hysteresis_band_counts_toward_existing_dwell_but_cannot_enter() -> None:
    calibration = ActiveLivenessCalibration(
        yaw_enter=35, yaw_exit=25, dwell_ms=200, min_valid_frames=3
    )
    evaluator = ActiveLivenessChallengeEvaluator((ChallengePhase.LEFT,), calibration)
    evaluator.observe(_pose(), 0)
    evaluator.observe(_pose(yaw=-30), 100)
    assert evaluator.result.completed_phases == ()
    evaluator.observe(_pose(yaw=-36), 200)
    evaluator.observe(_pose(yaw=-30), 300)
    result = evaluator.observe(_pose(yaw=-30), 400)
    assert result.status is ChallengeStatus.COMPLETED


def test_evaluator_reset_clears_terminal_and_dwell_state() -> None:
    evaluator = ActiveLivenessChallengeEvaluator(
        (ChallengePhase.LEFT,), ActiveLivenessCalibration(dwell_ms=100)
    )
    assert evaluator.observe(_pose(yaw=40), 0).status is ChallengeStatus.WRONG_DIRECTION
    evaluator.reset()
    evaluator.observe(_pose(), 10)
    evaluator.observe(_pose(yaw=-40), 110)
    assert evaluator.observe(_pose(yaw=-40), 210).status is ChallengeStatus.COMPLETED


@pytest.mark.parametrize(
    ("face_count", "track_id", "expected"),
    [(0, "person", ChallengeStatus.NO_FACE), (2, "person", ChallengeStatus.MULTIPLE_FACES)],
)
def test_face_presence_failures_are_explicit(
    face_count: int, track_id: str, expected: ChallengeStatus
) -> None:
    evaluator = ActiveLivenessChallengeEvaluator((ChallengePhase.NEUTRAL,))
    result = evaluator.observe(_pose(), 0, face_count=face_count, track_id=track_id)
    assert result.status is expected


def test_face_substitution_is_rejected() -> None:
    evaluator = ActiveLivenessChallengeEvaluator((ChallengePhase.NEUTRAL,))
    evaluator.observe(_pose(), 0, track_id="person-a")
    result = evaluator.observe(_pose(), 100, track_id="person-b")
    assert result.status is ChallengeStatus.FACE_SUBSTITUTION
