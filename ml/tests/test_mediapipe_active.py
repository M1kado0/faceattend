"""Tests for MediaPipe active liveness helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

import ml.liveness.mediapipe_active as mediapipe_active
from faceattend.vision.active_liveness import (
    ActiveLivenessCalibration,
    ActiveLivenessChallengeEvaluator,
    ChallengeStatus,
)
from faceattend.vision.challenge_session import (
    ChallengeAction,
    ChallengeSession,
    ChallengeSessionConfig,
)
from faceattend.vision.evidence import EvidenceStreamGuard
from faceattend.vision.passive_liveness import TemporalPassiveLivenessSession
from faceattend.vision.types import (
    BoundingBox,
    EvidenceDecision,
    FaceObservation,
    FaceQuality,
    Frame,
    HeadPose,
    LivenessEvidence,
    LivenessKind,
    ModelMetadata,
)
from ml.liveness.mediapipe_active import (
    ActiveLivenessConfig,
    BlinkCounter,
    BlinkTurnLeftRightChallenge,
    MediaPipeActionEvidence,
    MediaPipeActiveLivenessChecker,
    MediaPipeChallengeSessionAdapter,
    MediaPipeLivenessSession,
    VideoDecodeError,
    VideoFrame,
    _challenge_yaw,
    _decode_video,
    _frame_timestamp_ms,
    _matrix_pose_from_result,
)


def test_decode_video_empty_bytes_raises_video_decode_error() -> None:
    with pytest.raises(VideoDecodeError, match="empty_video"):
        _decode_video(b"")


def _blink_counter(*, cooldown_ms: int = 250) -> BlinkCounter:
    return BlinkCounter(
        closed_eye_threshold=0.20,
        open_eye_threshold=0.24,
        min_closed_frames=2,
        min_open_frames=2,
        blink_cooldown_ms=cooldown_ms,
    )


def _observe_sequence(counter: BlinkCounter, values: list[float], *, start_ms: int = 0) -> None:
    for index, eye_aspect_ratio in enumerate(values):
        counter.observe(eye_aspect_ratio, timestamp_ms=start_ms + index * 100)


def _face_landmarks_for_ear(ear: float) -> list[SimpleNamespace]:
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
    for indices in (mediapipe_active.LEFT_EYE, mediapipe_active.RIGHT_EYE):
        p1, p2, p3, p4, p5, p6 = indices
        landmarks[p1] = SimpleNamespace(x=0.4, y=0.5)
        landmarks[p4] = SimpleNamespace(x=0.6, y=0.5)
        # Horizontal eye width is 0.2; four vertical offsets divided by the
        # 0.4 horizontal denominator gives the requested EAR.
        offset = ear * 0.1
        landmarks[p2] = SimpleNamespace(x=0.45, y=0.5 + offset)
        landmarks[p6] = SimpleNamespace(x=0.45, y=0.5 - offset)
        landmarks[p3] = SimpleNamespace(x=0.55, y=0.5 + offset)
        landmarks[p5] = SimpleNamespace(x=0.55, y=0.5 - offset)
    return landmarks


def _result(*, ear: float = 0.30, smile: float | None = None) -> SimpleNamespace:
    categories = []
    if smile is not None:
        categories = [
            SimpleNamespace(category_name="mouthSmileLeft", score=smile),
            SimpleNamespace(category_name="mouthSmileRight", score=smile),
        ]
    return SimpleNamespace(
        face_landmarks=[_face_landmarks_for_ear(ear)],
        face_blendshapes=[categories],
        facial_transformation_matrixes=[np.eye(4)],
    )


def _pose() -> HeadPose:
    return HeadPose(yaw_degrees=0.0, pitch_degrees=0.0, roll_degrees=0.0)


def test_action_evidence_emits_blink_from_landmark_transition() -> None:
    detector = MediaPipeActionEvidence(
        ActiveLivenessConfig(
            closed_eye_threshold=0.20,
            open_eye_threshold=0.24,
            min_closed_frames=2,
            min_open_frames=2,
            blink_cooldown_ms=0,
        )
    )
    actions = [
        detector.observe(_result(ear=0.30), 0),
        detector.observe(_result(ear=0.30), 50),
        detector.observe(_result(ear=0.18), 100),
        detector.observe(_result(ear=0.17), 200),
        detector.observe(_result(ear=0.25), 300),
        detector.observe(_result(ear=0.26), 400),
    ]

    assert actions[-1] is ChallengeAction.BLINK


def test_action_evidence_emits_smile_from_real_blendshapes() -> None:
    detector = MediaPipeActionEvidence(
        ActiveLivenessConfig(smile_score_threshold=0.6, min_smile_frames=2)
    )

    assert detector.observe(_result(smile=0.0), 0) is None
    assert detector.observe(_result(smile=0.8), 100) is None
    assert detector.observe(_result(smile=0.8), 200) is ChallengeAction.SMILE


def test_action_evidence_ignores_missing_or_weak_smile_blendshapes() -> None:
    detector = MediaPipeActionEvidence(
        ActiveLivenessConfig(smile_score_threshold=0.6, min_smile_frames=2)
    )

    assert detector.observe(_result(smile=0.5), 0) is None
    assert detector.observe(_result(), 100) is None


def test_media_pipe_action_evidence_completes_session_smile() -> None:
    session = ChallengeSession(
        ChallengeSessionConfig(
            min_challenges=1,
            max_challenges=1,
            challenge_pool=(ChallengeAction.SMILE,),
        ),
        random_source=_FixedRandom(),
    )
    session.start(0)
    evaluator = ActiveLivenessChallengeEvaluator(
        session=session,
        calibration=ActiveLivenessCalibration(timeout_ms=2_000),
    )
    adapter = MediaPipeChallengeSessionAdapter(
        evaluator,
        ActiveLivenessConfig(smile_score_threshold=0.6, min_smile_frames=1),
    )

    adapter.observe(_result(smile=0), _pose(), 50, track_id="face-1")
    result = adapter.observe(_result(smile=0.9), _pose(), 100, track_id="face-1")

    assert result.status is ChallengeStatus.COMPLETED


def test_runtime_session_finalizes_passive_pad_after_active_completion() -> None:
    session = ChallengeSession(
        ChallengeSessionConfig(
            min_challenges=1,
            max_challenges=1,
            challenge_pool=(ChallengeAction.SMILE,),
        ),
        random_source=_FixedRandom(),
    )
    session.start(0)
    evaluator = ActiveLivenessChallengeEvaluator(session=session)

    class _Passive:
        model_metadata = ModelMetadata("test", "1", "checksum")

        def evaluate(self, frames, faces):
            return LivenessEvidence(
                kind=LivenessKind.PASSIVE,
                decision=EvidenceDecision.PASSED,
                score=0.95,
                threshold=0.85,
                model_version="1",
            )

        evaluate_temporal = evaluate

    runtime = MediaPipeLivenessSession(
        evaluator,
        TemporalPassiveLivenessSession(
            _Passive(),
            max_frames=2,
            min_frames=1,
            min_duration_ms=0,
            sample_interval_ms=0,
        ),
        ActiveLivenessConfig(smile_score_threshold=0.6, min_smile_frames=1),
        evidence_guard=EvidenceStreamGuard(clock_ns=lambda: 300_000_000),
    )
    frame = Frame(
        np.random.default_rng(1).integers(40, 210, (112, 112, 3), np.uint8), 100_000_000, 1
    )
    face = FaceObservation(
        BoundingBox(15, 15, 100, 108),
        0.9,
        np.array([[38, 52], [74, 52], [56, 72], [42, 92], [70, 92]], dtype=np.float32),
        FaceQuality(True, 1.0, 0.5, 0.2),
    )

    runtime.observe(_result(smile=0), _pose(), frame, face, 100, track_id="face-1")
    next_frame = Frame(frame.pixels.copy(), 200_000_000, 2)
    next_frame.pixels[0, 0, 0] = 1
    active_result = runtime.observe(
        _result(smile=0.9), _pose(), next_frame, face, 200, track_id="face-1"
    )

    assert active_result.status is ChallengeStatus.COMPLETED
    neutral_frame = Frame(frame.pixels.copy(), 300_000_000, 3)
    neutral_frame.pixels[0, 0, 0] = 2
    runtime.observe(_result(smile=0), _pose(), neutral_frame, face, 300, track_id="face-1")
    assert runtime.finalize_passive().decision is EvidenceDecision.PASSED


def test_runtime_session_cannot_finalize_pad_before_active_completion() -> None:
    session = ChallengeSession(
        ChallengeSessionConfig(
            min_challenges=1,
            max_challenges=1,
            challenge_pool=(ChallengeAction.SMILE,),
        ),
        random_source=_FixedRandom(),
    )
    session.start(0)
    evaluator = ActiveLivenessChallengeEvaluator(session=session)

    class _Passive:
        model_metadata = ModelMetadata("test", "1", "checksum")

        def evaluate(self, frames, faces):
            raise AssertionError("must not run")

    runtime = MediaPipeLivenessSession(
        evaluator, TemporalPassiveLivenessSession(_Passive(), max_frames=2)
    )

    result = runtime.finalize_passive()
    assert result.decision is EvidenceDecision.FAILED
    assert result.reason == "active_liveness_incomplete"


class _FixedRandom:
    def randint(self, start: int, stop: int) -> int:
        return start

    def sample(self, population, k: int):
        return list(population[:k])


class _FakeCapture:
    def __init__(self, timestamp_ms: float) -> None:
        self.timestamp_ms = timestamp_ms

    def get(self, _property_id: int) -> float:
        return self.timestamp_ms


def test_frame_timestamp_ms_clamps_duplicate_metadata_timestamps() -> None:
    timestamp_ms = _frame_timestamp_ms(
        _FakeCapture(timestamp_ms=100.0),
        frame_index=4,
        fps=30.0,
        last_timestamp_ms=100,
    )

    assert timestamp_ms == 101


def test_blink_counter_counts_stable_closed_to_open_transitions() -> None:
    counter = _blink_counter()

    _observe_sequence(
        counter,
        [0.30, 0.30, 0.18, 0.17, 0.25, 0.26, 0.30, 0.19, 0.18, 0.25, 0.26],
    )

    assert counter.blinks == 2


def test_blink_counter_ignores_non_consecutive_closed_noise() -> None:
    counter = _blink_counter()

    _observe_sequence(counter, [0.30, 0.18, 0.30, 0.17, 0.25, 0.26])

    assert counter.blinks == 0


def test_blink_counter_ignores_non_consecutive_open_noise() -> None:
    counter = _blink_counter()

    _observe_sequence(counter, [0.30, 0.18, 0.17, 0.25, 0.18, 0.25])

    assert counter.blinks == 0


def test_blink_requires_open_evidence_after_counter_start() -> None:
    counter = _blink_counter()
    _observe_sequence(counter, [0.18, 0.17, 0.25, 0.26])
    assert counter.blinks == 0
    _observe_sequence(counter, [0.30, 0.30, 0.18, 0.17, 0.25, 0.26], start_ms=500)
    assert counter.blinks == 1


def test_blink_counter_respects_cooldown() -> None:
    counter = _blink_counter(cooldown_ms=500)

    _observe_sequence(counter, [0.30, 0.30, 0.18, 0.17, 0.25, 0.26, 0.18, 0.17, 0.25, 0.26])

    assert counter.blinks == 1


def test_sustained_smile_needs_neutral_arming_and_emits_once() -> None:
    actions = MediaPipeActionEvidence(
        ActiveLivenessConfig(smile_score_threshold=0.6, min_smile_frames=2)
    )
    assert actions.observe(_result(smile=0.9), 0) is None
    assert actions.observe(_result(smile=0.9), 1) is None
    assert actions.observe(_result(smile=0.0), 2) is None
    assert actions.observe(_result(smile=0.9), 3) is None
    assert actions.observe(_result(smile=0.9), 4) is ChallengeAction.SMILE
    assert actions.observe(_result(smile=0.9), 5) is None
    assert actions.observe(_result(smile=0.9), 6) is None


def test_action_reset_requires_fresh_blink_and_smile_evidence() -> None:
    actions = MediaPipeActionEvidence(
        ActiveLivenessConfig(
            min_open_frames=1,
            min_closed_frames=1,
            min_smile_frames=1,
            closed_eye_threshold=0.2,
            open_eye_threshold=0.24,
        )
    )
    actions.observe(_result(ear=0.3, smile=0), 0)
    actions.observe(_result(ear=0.1, smile=0), 1)
    assert actions.observe(_result(ear=0.3, smile=0), 2) is ChallengeAction.BLINK
    actions.reset()
    assert actions.observe(_result(ear=0.1, smile=0.9), 3) is None
    assert actions.observe(_result(ear=0.3, smile=0.9), 4) is None


def test_blink_turn_left_right_challenge_completes_in_order() -> None:
    challenge = BlinkTurnLeftRightChallenge(
        ActiveLivenessConfig(
            closed_eye_threshold=0.20,
            open_eye_threshold=0.24,
            min_closed_frames=2,
            min_open_frames=2,
            min_head_turn_frames=2,
            head_turn_yaw_threshold=10.0,
        )
    )

    observations = [
        (0.30, 0.0),
        (0.30, 0.0),
        (0.18, 0.0),
        (0.17, 0.0),
        (0.25, 0.0),
        (0.26, 0.0),
        (0.18, 0.0),
        (0.17, 0.0),
        (0.25, 0.0),
        (0.26, 0.0),
        (0.30, -12.0),
        (0.30, -13.0),
        (0.30, 12.0),
        (0.30, 13.0),
    ]
    for index, (ear, yaw) in enumerate(observations):
        challenge.observe(
            eye_aspect_ratio=ear,
            yaw=yaw,
            timestamp_ms=index * 100,
        )

    assert challenge.completed is True
    assert challenge.score == 1.0


def test_blink_turn_left_right_requires_order() -> None:
    challenge = BlinkTurnLeftRightChallenge(
        ActiveLivenessConfig(
            closed_eye_threshold=0.20,
            open_eye_threshold=0.24,
            min_closed_frames=2,
            min_open_frames=2,
            min_head_turn_frames=2,
            head_turn_yaw_threshold=10.0,
        )
    )

    observations = [
        (0.30, -13.0),
        (0.30, -13.0),
        (0.18, 0.0),
        (0.17, 0.0),
        (0.25, 0.0),
        (0.26, 0.0),
        (0.18, 0.0),
        (0.17, 0.0),
        (0.25, 0.0),
        (0.26, 0.0),
    ]
    for index, (ear, yaw) in enumerate(observations):
        challenge.observe(
            eye_aspect_ratio=ear,
            yaw=yaw,
            timestamp_ms=index * 100,
        )

    assert challenge.completed is False
    assert challenge.reason == "left_turn_not_completed"


def test_blink_turn_left_right_requires_directional_dwell() -> None:
    challenge = BlinkTurnLeftRightChallenge(
        ActiveLivenessConfig(
            closed_eye_threshold=0.20,
            open_eye_threshold=0.24,
            min_closed_frames=2,
            min_open_frames=2,
            min_head_turn_frames=3,
            head_turn_yaw_threshold=10.0,
        )
    )
    blink_values = [0.30, 0.30, 0.18, 0.17, 0.25, 0.26, 0.18, 0.17, 0.25, 0.26]
    for index, ear in enumerate(blink_values):
        challenge.observe(eye_aspect_ratio=ear, yaw=0.0, timestamp_ms=index * 100)
    for index in range(2):
        challenge.observe(
            eye_aspect_ratio=0.30,
            yaw=-12.0,
            timestamp_ms=(len(blink_values) + index) * 100,
        )

    assert challenge.completed is False
    assert challenge.reason == "left_turn_not_completed"


def test_blink_turn_left_right_rejects_near_threshold_noise() -> None:
    challenge = BlinkTurnLeftRightChallenge(
        ActiveLivenessConfig(
            closed_eye_threshold=0.20,
            open_eye_threshold=0.24,
            min_closed_frames=2,
            min_open_frames=2,
            min_head_turn_frames=3,
            head_turn_yaw_threshold=10.0,
        )
    )
    blink_values = [0.30, 0.30, 0.18, 0.17, 0.25, 0.26, 0.18, 0.17, 0.25, 0.26]
    for index, ear in enumerate(blink_values):
        challenge.observe(eye_aspect_ratio=ear, yaw=0.0, timestamp_ms=index * 100)
    for index in range(5):
        challenge.observe(
            eye_aspect_ratio=0.30,
            yaw=-9.9,
            timestamp_ms=(len(blink_values) + index) * 100,
        )

    assert challenge.completed is False
    assert challenge.reason == "left_turn_not_completed"


def test_check_accepts_string_challenge_before_model_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = MediaPipeActiveLivenessChecker(
        model_path="missing.task",
        config=ActiveLivenessConfig(min_seconds=0.0, max_seconds=6.0),
    )
    monkeypatch.setattr(
        mediapipe_active,
        "_decode_video",
        lambda _video: ([VideoFrame(image_rgb=object(), timestamp_ms=0)], 1.0),
    )

    result = checker.check(b"fake-video", challenge="blink_twice")

    assert result.passed is False
    assert result.label == "model_not_found"


def test_check_accepts_composite_string_challenge_before_model_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = MediaPipeActiveLivenessChecker(
        model_path="missing.task",
        config=ActiveLivenessConfig(min_seconds=0.0, max_seconds=9.0),
    )
    monkeypatch.setattr(
        mediapipe_active,
        "_decode_video",
        lambda _video: ([VideoFrame(image_rgb=object(), timestamp_ms=0)], 1.0),
    )

    result = checker.check(b"fake-video", challenge="blink_turn_left_right")

    assert result.passed is False
    assert result.label == "model_not_found"


def test_check_rejects_unknown_string_challenge_before_decoding() -> None:
    checker = MediaPipeActiveLivenessChecker(model_path="missing.task")

    result = checker.check(b"", challenge="turn_around")

    assert result.passed is False
    assert result.label == "unsupported_challenge"


def test_validate_duration_rejects_too_short_video() -> None:
    checker = MediaPipeActiveLivenessChecker(
        model_path="missing.task",
        config=ActiveLivenessConfig(min_seconds=2.0, max_seconds=6.0),
    )

    result = checker._validate_duration(1.5)

    assert result is not None
    assert result.passed is False
    assert result.label == "video_too_short"
    assert result.challenge_completed is False


def test_validate_duration_rejects_too_long_video() -> None:
    checker = MediaPipeActiveLivenessChecker(
        model_path="missing.task",
        config=ActiveLivenessConfig(min_seconds=2.0, max_seconds=6.0),
    )

    result = checker._validate_duration(6.5)

    assert result is not None
    assert result.passed is False
    assert result.label == "video_too_long"
    assert result.challenge_completed is False


def test_validate_duration_accepts_allowed_video_duration() -> None:
    checker = MediaPipeActiveLivenessChecker(
        model_path="missing.task",
        config=ActiveLivenessConfig(min_seconds=2.0, max_seconds=6.0),
    )

    assert checker._validate_duration(3.0) is None


def test_timestamp_offset_for_next_video_advances_between_requests() -> None:
    checker = MediaPipeActiveLivenessChecker(model_path="missing.task")
    frames = [
        VideoFrame(image_rgb=object(), timestamp_ms=0),
        VideoFrame(image_rgb=object(), timestamp_ms=100),
    ]

    first_offset = checker._timestamp_offset_for_next_video(frames)
    second_offset = checker._timestamp_offset_for_next_video(frames)

    assert first_offset == 0
    assert second_offset == 101


@pytest.mark.parametrize("yaw", [-20.0, 20.0])
def test_matrix_pose_drives_left_and_right_turn_values(yaw: float) -> None:
    rotation_vector = np.deg2rad(np.array([0.0, yaw, 0.0])).reshape(3, 1)
    matrix, _ = mediapipe_active.cv2.Rodrigues(rotation_vector)
    result = SimpleNamespace(facial_transformation_matrixes=[matrix])

    pose = _matrix_pose_from_result(result)

    assert pose.yaw_degrees == pytest.approx(yaw, abs=0.01)


@pytest.mark.parametrize("pitch", [-15.0, 15.0])
def test_matrix_pose_drives_up_and_down_pitch_values(pitch: float) -> None:
    rotation_vector = np.deg2rad(np.array([pitch, 0.0, 0.0])).reshape(3, 1)
    matrix, _ = mediapipe_active.cv2.Rodrigues(rotation_vector)
    result = SimpleNamespace(facial_transformation_matrixes=[matrix])

    pose = _matrix_pose_from_result(result)

    assert pose.pitch_degrees == pytest.approx(pitch, abs=0.01)


def test_matrix_pose_fails_closed_when_matrix_is_missing_or_invalid() -> None:
    with pytest.raises(mediapipe_active.HeadPoseError, match="missing"):
        _matrix_pose_from_result(SimpleNamespace(facial_transformation_matrixes=[]))

    with pytest.raises(mediapipe_active.HeadPoseError, match="matrix"):
        _matrix_pose_from_result(SimpleNamespace(facial_transformation_matrixes=[np.zeros((2, 2))]))


def test_challenge_yaw_inversion_is_explicit_at_boundary() -> None:
    assert _challenge_yaw(58.0, invert=True) == -58.0
    assert _challenge_yaw(-54.0, invert=True) == 54.0
    assert _challenge_yaw(58.0, invert=False) == 58.0
