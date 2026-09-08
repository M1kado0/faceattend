"""Exercise real evidence guards and active/PAD composition with synthetic pixels."""

from dataclasses import replace

import numpy as np
import pytest

from faceattend.vision.active_liveness import (
    ActiveLivenessChallengeEvaluator,
    ChallengePhase,
    ChallengeStatus,
)
from faceattend.vision.evidence import EvidenceStreamGuard
from faceattend.vision.liveness_gate import LivenessGateError, extract_and_match_if_live
from faceattend.vision.passive_liveness import TemporalPassiveLivenessSession
from faceattend.vision.quality import measure_quality
from faceattend.vision.types import (
    BoundingBox,
    EvidenceDecision,
    FaceObservation,
    Frame,
    FrameEvidence,
    HeadPose,
    LivenessEvidence,
    LivenessKind,
    ModelMetadata,
)
from ml.liveness.mediapipe_active import (
    LivenessRuntimePhase,
    MediaPipeLivenessSession,
    NeutralPadConfig,
)


class PAD:
    model_metadata = ModelMetadata("test", "v1", "test")

    def __init__(self, *, raises: bool = False) -> None:
        self.calls = 0
        self.raises = raises

    def evaluate_temporal(
        self, frames: list[Frame], faces: list[FaceObservation]
    ) -> LivenessEvidence:
        self.calls += 1
        if self.raises:
            raise RuntimeError("native model failure")
        assert frames and len(frames) == len(faces)
        return LivenessEvidence(LivenessKind.PASSIVE, EvidenceDecision.PASSED, 0.95, 0.85, "v1")


def evidence(index: int = 1) -> FrameEvidence:
    frame = Frame(
        np.random.default_rng(index).integers(40, 210, (200, 200, 3), np.uint8),
        index * 100_000_000,
        index,
    )
    box = BoundingBox(40, 40, 160, 160)
    points = np.array([[70, 80], [130, 80], [100, 100], [80, 130], [120, 130]], np.float32)
    quality = measure_quality(frame, box, points)
    face = FaceObservation(box, 0.9, points, quality, "a")
    return FrameEvidence(
        frame, 1, face, "a", HeadPose(0, 0, 0), quality, quality.brightness, smile_score=0.0
    )


def runtime(*, raises: bool = False) -> tuple[MediaPipeLivenessSession, PAD]:
    pad = PAD(raises=raises)
    session = MediaPipeLivenessSession(
        ActiveLivenessChallengeEvaluator((ChallengePhase.NEUTRAL,)),
        TemporalPassiveLivenessSession(
            pad, max_frames=3, min_frames=3, min_duration_ms=200, sample_interval_ms=100
        ),
        post_active_config=NeutralPadConfig(neutral_dwell_ms=200),
        evidence_guard=EvidenceStreamGuard(clock_ns=lambda: 1_000_000_000),
    )
    return session, pad


@pytest.mark.parametrize("reason", ["duplicate_frame", "head_pose_invalid", "landmarker_no_face"])
def test_failure_reason_blocks_both_gates_and_is_sticky(reason: str) -> None:
    session, pad = runtime()
    session.observe_evidence(replace(evidence(), failure_reason=reason))
    assert session.failure_reason == reason
    session.observe_evidence(evidence(2))
    assert session.finalize_passive().reason == reason
    assert pad.calls == 0


@pytest.mark.parametrize("count,reason", [(0, "no_face"), (2, "multiple_faces")])
def test_face_count_errors_are_terminal_without_pad_exception(count: int, reason: str) -> None:
    session, pad = runtime()
    session.observe_evidence(evidence())
    session.observe_evidence(replace(evidence(2), face_count=count, face=None))
    assert session.failure_reason == reason
    assert session.finalize_passive().decision is EvidenceDecision.FAILED
    assert not session.passive._frames and pad.calls == 0


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("track", "track_changed"),
        ("jump", "implausible_face_jump"),
        ("duplicate", "duplicate_frame"),
        ("backward", "timestamp_not_monotonic"),
        ("repeated", "timestamp_not_monotonic"),
        ("sequence", "sequence_not_monotonic"),
        ("pose", "invalid_pose"),
        ("quality", "nonfinite_quality"),
        ("missing_quality", "quality_unavailable"),
        ("lighting", "lighting_source_mismatch"),
    ],
)
def test_composed_stream_rejects_invalid_evidence(kind: str, reason: str) -> None:
    session, pad = runtime()
    first, second = evidence(), evidence(2)
    session.observe_evidence(first)
    assert second.face is not None and second.quality is not None
    if kind == "track":
        second = replace(second, track_id="b", face=replace(second.face, track_id="b"))
    elif kind == "jump":
        second = replace(second, face=replace(second.face, bbox=BoundingBox(140, 140, 260, 260)))
    elif kind == "duplicate":
        second = replace(second, frame=replace(second.frame, pixels=first.frame.pixels.copy()))
    elif kind in {"backward", "repeated"}:
        second = replace(
            second,
            frame=replace(
                second.frame, captured_at_ns=0 if kind == "backward" else first.frame.captured_at_ns
            ),
        )
    elif kind == "sequence":
        second = replace(second, frame=replace(second.frame, sequence_id=1))
    elif kind == "pose":
        second = replace(second, pose=HeadPose(float("nan"), 0, 0))
    elif kind == "quality":
        q = replace(second.quality, sharpness=float("inf"))
        second = replace(second, quality=q, face=replace(second.face, quality=q))
    elif kind == "missing_quality":
        second = replace(second, quality=None)
    elif kind == "lighting":
        second = replace(second, lighting_score=0.9)
    session.observe_evidence(second)
    assert session.failure_reason == reason
    assert session.finalize_passive().reason == reason and pad.calls == 0


def test_stale_and_future_frames_fail_before_active() -> None:
    for now, reason in ((2_000_000_000, "stale_frame"), (0, "future_frame")):
        guard = EvidenceStreamGuard(clock_ns=lambda now=now: now)
        assert guard.observe(evidence()).failure_reason == reason


def test_active_then_pad_success_and_native_pad_failure() -> None:
    for raises in (False, True):
        session, pad = runtime(raises=raises)
        for i in range(1, 5):
            session.observe_evidence(evidence(i))
            assert pad.calls == 0
        assert session.active.evaluator.result.status is ChallengeStatus.COMPLETED
        for i in range(5, 10):
            session.observe_evidence(evidence(i))
        result = session.finalize_passive()
        assert result.decision is (EvidenceDecision.FAILED if raises else EvidenceDecision.PASSED)
        assert pad.calls == 1 and not session.passive._frames
        assert session.finalize_passive() is result


def test_pad_collects_only_after_active_then_stable_neutral_hold() -> None:
    pad = PAD()
    session = MediaPipeLivenessSession(
        ActiveLivenessChallengeEvaluator((ChallengePhase.NEUTRAL,)),
        TemporalPassiveLivenessSession(
            pad, max_frames=3, min_frames=3, min_duration_ms=200, sample_interval_ms=100
        ),
        post_active_config=NeutralPadConfig(neutral_dwell_ms=200, max_smile_score=0.3),
        evidence_guard=EvidenceStreamGuard(clock_ns=lambda: 1_000_000_000),
    )

    for index in range(1, 5):
        session.observe_evidence(evidence(index))
    assert session.phase is LivenessRuntimePhase.FACE_CAMERA
    assert session.passive.frames == () and pad.calls == 0

    session.observe_evidence(replace(evidence(5), pose=HeadPose(0, -30, 0)))
    assert session.phase is LivenessRuntimePhase.FACE_CAMERA
    assert session.passive.frames == ()

    for index in range(6, 9):
        session.observe_evidence(replace(evidence(index), smile_score=0.1))
    assert session.phase is LivenessRuntimePhase.COMPLETED
    assert session.result.decision is EvidenceDecision.PASSED
    assert session.passive.frames == () and pad.calls == 1


def test_non_neutral_pose_or_expression_restarts_post_active_window() -> None:
    session, pad = runtime()
    for index in range(1, 5):
        session.observe_evidence(evidence(index))
    session.observe_evidence(evidence(5))
    assert session.phase is LivenessRuntimePhase.HOLD_STILL
    session.observe_evidence(replace(evidence(6), smile_score=0.9))
    assert session.phase is LivenessRuntimePhase.FACE_CAMERA
    assert session.passive.frames == () and pad.calls == 0


def test_embedding_candidate_is_best_neutral_frame_and_single_use() -> None:
    session, _pad = runtime()
    for index in range(1, 5):
        session.observe_evidence(evidence(index))
    neutral = [evidence(index) for index in range(5, 10)]
    assert neutral[1].quality is not None and neutral[1].face is not None
    best_quality = replace(neutral[1].quality, sharpness=10_000)
    neutral[1] = replace(
        neutral[1], quality=best_quality, face=replace(neutral[1].face, quality=best_quality)
    )
    for item in neutral:
        session.observe_evidence(item)
    assert session.take_embedding_candidate().frame.sequence_id == neutral[1].frame.sequence_id
    with pytest.raises(RuntimeError, match="no neutral"):
        session.take_embedding_candidate()


def test_registration_can_transfer_bounded_neutral_candidates_only_after_both_passes() -> None:
    session, _pad = runtime()
    for index in range(1, 5):
        session.observe_evidence(evidence(index))
    for index in range(5, 12):
        session.observe_evidence(evidence(index))

    candidates = session.take_embedding_candidates(min_candidates=3, max_candidates=5)

    assert len(candidates) == 3
    assert all(item.face_count == 1 and item.pose is not None for item in candidates)
    assert session.retained_candidate_count == 0
    with pytest.raises(RuntimeError, match="insufficient neutral"):
        session.take_embedding_candidates(min_candidates=3, max_candidates=5)


def test_embedding_extraction_runs_only_after_post_active_pad_pass() -> None:
    pad = PAD()
    calls: list[int] = []
    session = MediaPipeLivenessSession(
        ActiveLivenessChallengeEvaluator((ChallengePhase.NEUTRAL,)),
        TemporalPassiveLivenessSession(
            pad, max_frames=3, min_frames=3, min_duration_ms=200, sample_interval_ms=100
        ),
        post_active_config=NeutralPadConfig(neutral_dwell_ms=200),
        evidence_guard=EvidenceStreamGuard(clock_ns=lambda: 1_000_000_000),
        embedding_extractor=lambda item: (
            calls.append(item.frame.sequence_id) or np.ones(512, np.float32)
        ),
    )
    for index in range(1, 5):
        session.observe_evidence(evidence(index))
    assert calls == []
    for index in range(5, 10):
        session.observe_evidence(evidence(index))
    assert len(calls) == 1
    assert session.embedding is not None and session.embedding.shape == (512,)


def test_post_active_timeout_and_cancellation_clear_transient_frames() -> None:
    session, _pad = runtime()
    for index in range(1, 5):
        session.observe_evidence(evidence(index))
    session.observe_evidence(evidence(5))
    timed_out = replace(evidence(6), frame=replace(evidence(6).frame, captured_at_ns=9_000_000_000))
    session.guard.clock_ns = lambda: 9_000_000_000
    session.observe_evidence(timed_out)
    assert session.phase is LivenessRuntimePhase.FAILED
    assert session.result.reason == "neutral_pad_timeout"
    assert session.passive.frames == ()

    cancelled, _pad = runtime()
    for index in range(1, 6):
        cancelled.observe_evidence(evidence(index))
    cancelled.cancel()
    assert cancelled.phase is LivenessRuntimePhase.CANCELLED
    assert cancelled.passive.frames == ()
    assert cancelled.retained_candidate_count == 0
    with pytest.raises(RuntimeError, match="completed liveness"):
        cancelled.take_embedding_candidate()


def test_camera_callable_converts_processor_exception_to_terminal_failure() -> None:
    session, pad = runtime()

    def broken(frame: Frame) -> FrameEvidence:
        raise ValueError("bad model input")

    session.processor = broken
    result = session(evidence().frame)
    assert result.failure_reason == "evidence_processing_failed"
    assert session.finalize_passive().decision is EvidenceDecision.FAILED and pad.calls == 0


def test_standalone_pad_consumer_honors_failure_flag() -> None:
    pad = PAD()
    passive = TemporalPassiveLivenessSession(pad)
    passive.observe_evidence(replace(evidence(), failure_reason="duplicate_frame"))
    assert passive.finalize().reason == "duplicate_frame" and pad.calls == 0


@pytest.mark.parametrize("failure", ["active", "passive"])
def test_embedding_and_matching_callbacks_never_run_after_liveness_failure(
    failure: str,
) -> None:
    session, pad = runtime(raises=failure == "passive")
    if failure == "active":
        session.observe_evidence(replace(evidence(), failure_reason="active_failed"))
    else:
        for index in range(1, 5):
            session.observe_evidence(evidence(index))
        session.finalize_passive()
    calls = {"embedding": 0, "matching": 0}

    def embedding() -> np.ndarray:
        calls["embedding"] += 1
        return np.ones(512, np.float32)

    def matching(_embedding: np.ndarray) -> str:
        calls["matching"] += 1
        return "matched"

    with pytest.raises(LivenessGateError):
        extract_and_match_if_live(session.result, embedding, matching)
    assert calls == {"embedding": 0, "matching": 0} and pad.calls <= 1


def test_embedding_and_matching_run_in_order_only_after_both_liveness_gates_pass() -> None:
    session, _pad = runtime()
    for index in range(1, 5):
        session.observe_evidence(evidence(index))
    for index in range(5, 10):
        session.observe_evidence(evidence(index))
    session.finalize_passive()
    calls: list[str] = []

    def embedding() -> np.ndarray:
        calls.append("embedding")
        return np.ones(512, np.float32)

    def matching(_embedding: np.ndarray) -> str:
        calls.append("matching")
        return "matched"

    assert extract_and_match_if_live(session.result, embedding, matching) == "matched"
    assert calls == ["embedding", "matching"]


def test_runtime_cancellation_is_sticky_and_clears_pad_frames() -> None:
    session, pad = runtime()
    session.observe_evidence(evidence())
    result = session.cancel()
    assert result.decision is EvidenceDecision.FAILED and result.reason == "cancelled"
    assert session.passive.frames == () and session.finalize_passive() is result.passive
    session.observe_evidence(evidence(2))
    assert pad.calls == 0 and session.result.reason == "cancelled"


def test_processing_gap_between_two_valid_faces_does_not_infer_missing_face() -> None:
    guard = EvidenceStreamGuard(clock_ns=lambda: 900_000_000)
    assert guard.observe(evidence()).failure_reason is None
    assert guard.observe(evidence(8)).failure_reason is None


def test_same_geometry_cannot_prove_same_identity() -> None:
    # Different pixels with identical geometry and no identity label are accepted:
    # this documents a substitution limitation, not a biometric guarantee.
    guard = EvidenceStreamGuard(clock_ns=lambda: 500_000_000)
    for index in (1, 2):
        item = evidence(index)
        assert item.face is not None
        item = replace(item, track_id=None, face=replace(item.face, track_id=None))
        assert guard.observe(item).failure_reason is None


def test_raw_yaw_is_mapped_once_at_session_boundary() -> None:
    session, _pad = runtime()
    session.active.evaluator = ActiveLivenessChallengeEvaluator((ChallengePhase.LEFT,))
    session.observe_evidence(evidence())
    for index in range(2, 6):
        item = replace(evidence(index), pose=HeadPose(40, 0, 0))
        session.observe_evidence(item)
        assert item.pose.yaw_degrees == 40
    assert session.active.evaluator.result.status is ChallengeStatus.COMPLETED
