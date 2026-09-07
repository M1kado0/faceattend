import random

import numpy as np
import pytest

from faceattend.vision.active_liveness import (
    ActiveLivenessCalibration,
    ActiveLivenessChallengeEvaluator,
    ChallengePhase,
    ChallengeStatus,
)
from faceattend.vision.challenge_session import (
    ChallengeAction,
    ChallengeSession,
    ChallengeSessionConfig,
    ChallengeSessionStatus,
)
from faceattend.vision.types import (
    BoundingBox,
    FaceObservation,
    FaceQuality,
    Frame,
    FrameEvidence,
    HeadPose,
)


def _pose(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> HeadPose:
    return HeadPose(yaw, pitch, roll)


def _evidence(
    *,
    face_count: int = 1,
    pose: HeadPose | None = None,
    timestamp_ns: int = 1_000_000,
    action: ChallengeAction | None = None,
) -> FrameEvidence:
    face = (
        FaceObservation(
            BoundingBox(0, 0, 1, 1),
            1.0,
            np.zeros((5, 2), dtype=np.float32),
            FaceQuality(True, 1.0, 0.5, 0.2),
        )
        if face_count == 1
        else None
    )
    return FrameEvidence(
        frame=Frame(np.zeros((2, 2, 3), dtype=np.uint8), timestamp_ns, timestamp_ns // 1_000_000),
        face_count=face_count,
        face=face,
        track_id="person",
        pose=pose,
        action=action,
        quality=face.quality if face else None,
        lighting_score=face.quality.brightness if face else None,
    )


def test_session_generates_unique_auditable_sequence() -> None:
    session = ChallengeSession(
        ChallengeSessionConfig(min_challenges=3, max_challenges=3),
        random_source=random.Random(7),
    )

    state = session.start(0)

    assert state.status is ChallengeSessionStatus.IN_PROGRESS
    assert len(state.sequence) == 3
    assert len(set(state.sequence)) == 3
    assert state.current is state.sequence[0]


def test_session_supports_every_action_type() -> None:
    actions = [
        ChallengeAction.BLINK,
        ChallengeAction.TURN_LEFT,
        ChallengeAction.TURN_RIGHT,
        ChallengeAction.LOOK_UP,
        ChallengeAction.SMILE,
    ]
    session = ChallengeSession(
        ChallengeSessionConfig(
            min_challenges=len(actions),
            max_challenges=len(actions),
            challenge_pool=tuple(actions),
        ),
        random_source=_FixedSequenceRandom(actions),
    )

    state = session.start(0)

    assert state.sequence == tuple(actions)


def test_session_advances_only_when_current_challenge_is_completed() -> None:
    session = ChallengeSession(
        ChallengeSessionConfig(min_challenges=2, max_challenges=2),
        random_source=random.Random(2),
    )
    started = session.start(100)

    next_state = session.complete_current(500)

    assert next_state.completed == (started.sequence[0],)
    assert next_state.current is started.sequence[1]
    assert next_state.current_started_at_ms == 500


def test_session_requires_exact_sequence_completion() -> None:
    session = ChallengeSession(
        ChallengeSessionConfig(min_challenges=2, max_challenges=2),
        random_source=random.Random(2),
    )
    session.start(0)
    session.complete_current(100)
    result = session.complete_current(200)

    assert result.status is ChallengeSessionStatus.COMPLETED
    assert result.completed == result.sequence
    assert result.current is None


def test_challenge_and_session_timeouts_are_explicit() -> None:
    config = ChallengeSessionConfig(
        min_challenges=2,
        max_challenges=2,
        challenge_timeout_ms=100,
        session_timeout_ms=500,
    )
    challenge = ChallengeSession(config, random_source=random.Random(2))
    challenge.start(0)
    assert challenge.observe(101).status is ChallengeSessionStatus.CHALLENGE_TIMEOUT

    session = ChallengeSession(config, random_source=random.Random(2))
    session.start(0)
    session.complete_current(100)
    assert session.observe(501).status is ChallengeSessionStatus.SESSION_TIMEOUT


def test_timestamps_must_be_strictly_increasing() -> None:
    session = ChallengeSession(random_source=random.Random(2))
    session.start(100)

    with pytest.raises(ValueError, match="strictly increasing"):
        session.observe(100)


def test_invalid_session_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="challenge_pool"):
        ChallengeSessionConfig(
            max_challenges=4,
            challenge_pool=(ChallengeAction.BLINK,),
        )


class _FixedSequenceRandom:
    def __init__(self, sequence: list[ChallengeAction]) -> None:
        self.sequence = sequence

    def randint(self, start: int, stop: int) -> int:
        return len(self.sequence)

    def sample(self, population: list[ChallengeAction], k: int) -> list[ChallengeAction]:
        return self.sequence[:k]


def test_session_backed_evaluator_requires_neutral_between_turns() -> None:
    session = ChallengeSession(
        ChallengeSessionConfig(min_challenges=2, max_challenges=2),
        random_source=_FixedSequenceRandom([ChallengeAction.TURN_LEFT, ChallengeAction.TURN_RIGHT]),
    )
    session.start(0)
    evaluator = ActiveLivenessChallengeEvaluator(
        calibration=ActiveLivenessCalibration(dwell_ms=100), session=session
    )

    evaluator.observe(_pose(), 1, track_id="person")
    evaluator.observe(_pose(yaw=-40), 101, track_id="person")
    evaluator.observe(_pose(yaw=-40), 201, track_id="person")
    assert evaluator.result.phase is ChallengePhase.RIGHT
    assert evaluator.result.status is ChallengeStatus.IN_PROGRESS

    evaluator.observe(_pose(), 202, track_id="person")
    evaluator.observe(_pose(yaw=40), 203, track_id="person")
    result = evaluator.observe(_pose(yaw=40), 303, track_id="person")

    assert result.status is ChallengeStatus.COMPLETED
    assert session.snapshot.completed == session.snapshot.sequence


def test_session_backed_evaluator_rejects_track_substitution() -> None:
    session = ChallengeSession(
        ChallengeSessionConfig(min_challenges=1, max_challenges=1),
        random_source=_FixedSequenceRandom([ChallengeAction.LOOK_UP]),
    )
    session.start(0)
    evaluator = ActiveLivenessChallengeEvaluator(session=session)

    evaluator.observe(_pose(), 1, track_id="person-a")
    evaluator.observe(_pose(pitch=-30), 2, track_id="person-a")
    result = evaluator.observe(_pose(pitch=-30), 3, track_id="person-b")

    assert result.status is ChallengeStatus.FACE_SUBSTITUTION


def test_session_backed_evaluator_supports_look_up_and_smile_actions() -> None:
    session = ChallengeSession(
        ChallengeSessionConfig(min_challenges=2, max_challenges=2),
        random_source=_FixedSequenceRandom([ChallengeAction.LOOK_UP, ChallengeAction.SMILE]),
    )
    session.start(0)
    evaluator = ActiveLivenessChallengeEvaluator(
        calibration=ActiveLivenessCalibration(dwell_ms=100), session=session
    )

    evaluator.observe(_pose(), 1, track_id="person")
    evaluator.observe(_pose(pitch=-30), 101, track_id="person")
    evaluator.observe(_pose(pitch=-30), 201, track_id="person")
    result = evaluator.observe(_pose(), 202, track_id="person", action=ChallengeAction.SMILE)

    assert result.status is ChallengeStatus.COMPLETED


def test_session_timeout_is_checked_while_waiting_for_neutral() -> None:
    session = ChallengeSession(
        ChallengeSessionConfig(
            min_challenges=1, max_challenges=1, challenge_timeout_ms=100, session_timeout_ms=200
        ),
        random_source=_FixedSequenceRandom([ChallengeAction.TURN_LEFT]),
    )
    session.start(0)
    evaluator = ActiveLivenessChallengeEvaluator(session=session)
    assert (
        evaluator.observe_evidence(_evidence(pose=_pose(), timestamp_ns=1_000_000)).status
        is ChallengeStatus.IN_PROGRESS
    )
    result = evaluator.observe_evidence(_evidence(pose=_pose(yaw=20), timestamp_ns=201_000_000))
    assert result.status is ChallengeStatus.TIMEOUT
    assert result.reason == "session_timeout"
    assert session.snapshot.status is ChallengeSessionStatus.SESSION_TIMEOUT


def test_challenge_timeout_is_checked_during_directional_dwell() -> None:
    session = ChallengeSession(
        ChallengeSessionConfig(
            min_challenges=1, max_challenges=1, challenge_timeout_ms=100, session_timeout_ms=1_000
        ),
        random_source=_FixedSequenceRandom([ChallengeAction.TURN_LEFT]),
    )
    session.start(0)
    evaluator = ActiveLivenessChallengeEvaluator(
        calibration=ActiveLivenessCalibration(dwell_ms=500), session=session
    )
    evaluator.observe_evidence(_evidence(pose=_pose(), timestamp_ns=1_000_000))
    evaluator.observe_evidence(_evidence(pose=_pose(yaw=-40), timestamp_ns=50_000_000))
    result = evaluator.observe_evidence(_evidence(pose=_pose(yaw=-40), timestamp_ns=101_000_000))
    assert result.status is ChallengeStatus.TIMEOUT
    assert result.reason == "challenge_timeout"


@pytest.mark.parametrize(
    ("action", "phase", "pose"),
    [
        (ChallengeAction.BLINK, ChallengePhase.BLINK, _pose()),
        (ChallengeAction.SMILE, ChallengePhase.SMILE, _pose()),
        (ChallengeAction.TURN_LEFT, ChallengePhase.LEFT, _pose(yaw=-40)),
        (ChallengeAction.TURN_RIGHT, ChallengePhase.RIGHT, _pose(yaw=40)),
        (ChallengeAction.LOOK_UP, ChallengePhase.UP, _pose(pitch=-30)),
    ],
)
def test_supported_actions_complete_through_frame_evidence(
    action: ChallengeAction, phase: ChallengePhase, pose: HeadPose
) -> None:
    evaluator = ActiveLivenessChallengeEvaluator((phase,))
    evidence = _evidence(pose=_pose(), timestamp_ns=1_000_000)
    if phase in {ChallengePhase.LEFT, ChallengePhase.RIGHT, ChallengePhase.UP}:
        evaluator.observe_evidence(evidence)
        evidence = _evidence(pose=pose, timestamp_ns=101_000_000)
        evaluator.observe_evidence(evidence)
        evidence = _evidence(pose=pose, timestamp_ns=401_000_000)
    else:
        evidence = _evidence(pose=pose, timestamp_ns=1_000_000, action=action)
    result = evaluator.observe_evidence(evidence)
    assert result.status is ChallengeStatus.COMPLETED


def test_evidence_boundary_reports_missing_face() -> None:
    evaluator = ActiveLivenessChallengeEvaluator((ChallengePhase.NEUTRAL,))

    result = evaluator.observe_evidence(_evidence(face_count=0, pose=_pose()))

    assert result.status is ChallengeStatus.NO_FACE


def test_evidence_boundary_reports_multiple_faces() -> None:
    evaluator = ActiveLivenessChallengeEvaluator((ChallengePhase.NEUTRAL,))

    result = evaluator.observe_evidence(_evidence(face_count=2, pose=_pose()))

    assert result.status is ChallengeStatus.MULTIPLE_FACES


def test_evidence_boundary_reports_invalid_pose() -> None:
    evaluator = ActiveLivenessChallengeEvaluator((ChallengePhase.NEUTRAL,))

    result = evaluator.observe_evidence(_evidence(pose=None))

    assert result.status is ChallengeStatus.INVALID_EVIDENCE
    assert result.reason == "pose_unavailable"


def test_evidence_boundary_completes_action_challenge() -> None:
    evaluator = ActiveLivenessChallengeEvaluator((ChallengePhase.BLINK,))
    evidence = _evidence(pose=_pose())
    evidence = FrameEvidence(
        frame=evidence.frame,
        face_count=evidence.face_count,
        face=evidence.face,
        track_id=evidence.track_id,
        pose=evidence.pose,
        action=ChallengeAction.BLINK,
        quality=evidence.quality,
        lighting_score=evidence.lighting_score,
    )

    result = evaluator.observe_evidence(evidence)

    assert result.status is ChallengeStatus.COMPLETED
