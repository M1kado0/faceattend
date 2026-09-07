"""Runtime composition tests use synthetic outputs at model SDK boundaries."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest

from faceattend.camera.worker import CameraEvidenceWorker
from faceattend.vision.active_liveness import ActiveLivenessChallengeEvaluator, ChallengePhase
from faceattend.vision.alignment import InsightFaceAligner
from faceattend.vision.embeddings import InsightFaceEmbedder
from faceattend.vision.evidence import EvidenceStreamGuard
from faceattend.vision.face_analyzer import (
    HeadlessFaceAnalyzer,
    MediaPipeFrameLandmarker,
    ModelArtifact,
    ModelCompatibilityError,
    RuntimeModelManifest,
    validate_model_files,
)
from faceattend.vision.face_detector import InsightFaceDetector
from faceattend.vision.passive_liveness import (
    MiniFASNetPassiveLivenessDetector,
    TemporalPassiveLivenessSession,
)
from faceattend.vision.types import (
    EvidenceDecision,
    Frame,
    LivenessEvidence,
    LivenessKind,
    ModelMetadata,
)
from ml.liveness.mediapipe_active import MediaPipeLivenessSession, NeutralPadConfig


def _pixels() -> np.ndarray:
    """Synthetic textured, exposed input; a black image must fail quality."""
    return np.random.default_rng(42).integers(40, 210, (112, 112, 3), dtype=np.uint8)


def test_camera_to_real_adapters_to_session_and_pad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor, calls = _runtime(tmp_path)
    ticks = [0]

    class Capture:
        released = False

        def isOpened(self) -> bool:  # noqa: N802 - OpenCV capture protocol
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            ticks[0] += 100_000_000
            pixels = _pixels()
            pixels[0, 0, 0] = ticks[0] // 100_000_000
            return True, pixels

        def release(self) -> None:
            self.released = True

    capture = Capture()
    monkeypatch.setattr("faceattend.camera.worker.time.monotonic_ns", lambda: ticks[0])
    session = MediaPipeLivenessSession(
        ActiveLivenessChallengeEvaluator((ChallengePhase.NEUTRAL,)),
        TemporalPassiveLivenessSession(
            processor.passive,
            max_frames=3,
            min_frames=3,
            min_duration_ms=200,
            sample_interval_ms=100,
        ),
        post_active_config=NeutralPadConfig(neutral_dwell_ms=200),
        evidence_guard=EvidenceStreamGuard(clock_ns=lambda: ticks[0]),
        processor=processor,
    )
    frames = CameraEvidenceWorker(capture_factory=lambda _: capture).run(session, max_frames=7)
    assert capture.released
    assert all(frame.failure_reason is None for frame in frames)
    assert calls == {"embedding": 0, "passive": 3}
    assert session.finalize_passive().decision is EvidenceDecision.PASSED
    assert session.result.decision is EvidenceDecision.PASSED
    assert calls == {"embedding": 0, "passive": 3}
    processor.close()


def test_black_frame_fails_composed_quality_before_action_or_pad(tmp_path: Path) -> None:
    processor, calls = _runtime(tmp_path)
    frame = Frame(np.zeros((112, 112, 3), np.uint8), 100_000_000, 1)
    session = MediaPipeLivenessSession(
        ActiveLivenessChallengeEvaluator((ChallengePhase.NEUTRAL,)),
        TemporalPassiveLivenessSession(processor.passive),
        evidence_guard=EvidenceStreamGuard(clock_ns=lambda: frame.captured_at_ns),
        processor=processor,
    )
    result = session(frame)
    assert result.failure_reason == "underexposed"
    assert session.result.decision is EvidenceDecision.FAILED
    assert session.finalize_passive().decision is EvidenceDecision.FAILED
    assert calls == {"embedding": 0, "passive": 0}
    processor.close()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_blendshape_cannot_complete_action(tmp_path: Path, score: float) -> None:
    result = SimpleNamespace(
        face_landmarks=[[SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]],
        facial_transformation_matrixes=[np.eye(4)],
        face_blendshapes=[[SimpleNamespace(category_name="mouthSmileLeft", score=score)]],
    )
    processor, calls = _runtime(tmp_path, result=result)
    item = processor(Frame(_pixels(), 1_000_000, 1))
    assert item.failure_reason == "invalid_action_evidence"
    assert item.action is None and calls == {"embedding": 0, "passive": 0}
    processor.close()


def test_artifact_rejects_changed_weights_even_if_metadata_still_matches(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"selected weights")
    metadata = ModelMetadata("recognition", "v1", sha256(path.read_bytes()).hexdigest())
    artifact = ModelArtifact(path, metadata)
    artifact.verify(metadata)
    path.write_bytes(b"different weights")
    with pytest.raises(ModelCompatibilityError, match="checksum"):
        artifact.verify(metadata)


def _runtime(
    tmp_path: Path,
    *,
    result: SimpleNamespace | None = None,
    face_count: int = 1,
    embedding_size: int = 512,
) -> tuple[HeadlessFaceAnalyzer, dict[str, int]]:
    artifacts = []
    for role in ("detector", "embedding", "passive", "landmarker"):
        path = tmp_path / role
        if role == "landmarker":
            path.write_bytes(role.encode())
        else:
            size = {"detector": None, "embedding": 112, "passive": 80}[role]
            widths = {
                "detector": [1, 1, 1, 4, 4, 4, 10, 10, 10],
                "embedding": [512],
                "passive": [3],
            }[role]
            graph = onnx.helper.make_graph(
                [],
                role,
                [
                    onnx.helper.make_tensor_value_info(
                        "image", onnx.TensorProto.FLOAT, [1, 3, size, size]
                    )
                ],
                [
                    onnx.helper.make_tensor_value_info(
                        f"out{i}", onnx.TensorProto.FLOAT, [1, width]
                    )
                    for i, width in enumerate(widths)
                ],
            )
            onnx.save(onnx.helper.make_model(graph), path)
        artifacts.append(
            ModelArtifact(path, ModelMetadata(role, "v1", sha256(path.read_bytes()).hexdigest()))
        )
    manifest = RuntimeModelManifest(*artifacts)
    calls = {"embedding": 0, "passive": 0}

    def embed(_image: np.ndarray) -> np.ndarray:
        calls["embedding"] += 1
        return np.ones((1, embedding_size), dtype=np.float32)

    def predict(_image: np.ndarray, _bbox: list[float]) -> dict[str, float]:
        calls["passive"] += 1
        return {"score": 0.99}

    def options(artifact: ModelArtifact) -> dict[str, str]:
        return {
            "model_path": str(artifact.path),
            "model_name": artifact.metadata.name,
            "model_version": artifact.metadata.version,
            "model_checksum": artifact.metadata.checksum,
        }

    points = np.array([[[38, 52], [74, 52], [56, 72], [42, 92], [70, 92]]], dtype=np.float32)
    detector = InsightFaceDetector(
        **options(manifest.detector),
        model_loader=lambda: SimpleNamespace(
            detect=lambda *_a, **_kw: (
                np.tile([[15, 15, 100, 108, 0.99]], (face_count, 1)),
                np.tile(points, (face_count, 1, 1)),
            )
        ),
    )
    embedder = InsightFaceEmbedder(
        **options(manifest.embedding), model_loader=lambda: SimpleNamespace(get_feat=embed)
    )
    passive = MiniFASNetPassiveLivenessDetector(
        **options(manifest.passive), model_loader=lambda: SimpleNamespace(predict=predict)
    )
    landmarks = [SimpleNamespace(x=(i % 20) / 20, y=(i // 20) / 24) for i in range(478)]
    sdk_result = result or SimpleNamespace(
        face_landmarks=[landmarks], facial_transformation_matrixes=[np.eye(4)], face_blendshapes=[]
    )
    sdk = SimpleNamespace(detect_for_video=lambda *_a: sdk_result, close=lambda: None)
    landmarker = MediaPipeFrameLandmarker(manifest.landmarker, model_loader=lambda: sdk)
    return HeadlessFaceAnalyzer(
        manifest,
        detector=detector,
        aligner=InsightFaceAligner(),
        embedder=embedder,
        passive=passive,
        landmarker=landmarker,
    ), calls


def test_processor_builds_evidence_without_running_pad_or_embedding(tmp_path: Path) -> None:
    runtime, calls = _runtime(tmp_path)
    frame = Frame(_pixels(), 1_000_000, 1)
    evidence = runtime(frame)
    assert evidence.face_count == 1
    assert evidence.pose is not None
    assert evidence.pose.yaw_degrees == pytest.approx(0)
    assert evidence.embedding is None and evidence.passive is None
    assert calls == {"embedding": 0, "passive": 0}
    runtime.close()


def test_incomplete_mediapipe_landmarks_return_failure_not_action(tmp_path: Path) -> None:
    runtime, _calls = _runtime(
        tmp_path,
        result=SimpleNamespace(
            face_landmarks=[[]], facial_transformation_matrixes=[np.eye(4)], face_blendshapes=[]
        ),
    )
    evidence = runtime(Frame(_pixels(), 1_000_000, 1))
    assert evidence.failure_reason == "invalid_mediapipe_landmarks"
    assert evidence.action is None
    runtime.close()


def test_model_role_swap_is_rejected_even_with_matching_file_hash(tmp_path: Path) -> None:
    runtime, _calls = _runtime(tmp_path)
    path = runtime.manifest.detector.path
    # A recognition model cannot serve as SCRFD, even if its digest was pinned.
    graph = onnx.helper.make_graph(
        [],
        "wrong-role",
        [onnx.helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, 112, 112])],
        [onnx.helper.make_tensor_value_info("embedding", onnx.TensorProto.FLOAT, [1, 512])],
    )
    onnx.save(onnx.helper.make_model(graph), path)

    detector = ModelArtifact(
        path, ModelMetadata("detector", "v1", sha256(path.read_bytes()).hexdigest())
    )
    manifest = replace(runtime.manifest, detector=detector)
    with pytest.raises(ModelCompatibilityError, match="detector.*contract"):
        validate_model_files(manifest)


def test_embedding_requires_both_passes_and_returns_contract_vector(tmp_path: Path) -> None:
    runtime, calls = _runtime(tmp_path)
    evidence = runtime(Frame(_pixels(), 1_000_000, 1))
    active = LivenessEvidence(LivenessKind.ACTIVE, EvidenceDecision.PASSED, 1, 1, "v1")
    passive = LivenessEvidence(LivenessKind.PASSIVE, EvidenceDecision.FAILED, 0.1, 0.85, "v1")
    with pytest.raises(ValueError, match="liveness"):
        runtime.extract_embedding(evidence, active=active, passive=passive)
    assert calls["embedding"] == 0

    vector = runtime.extract_embedding(
        evidence, active=active, passive=replace(passive, decision=EvidenceDecision.PASSED)
    )
    assert vector.shape == (512,)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1)
    runtime.close()


@pytest.mark.parametrize("count,reason", [(0, "no_face"), (2, "multiple_faces")])
def test_detector_face_count_failure_is_explicit(tmp_path: Path, count: int, reason: str) -> None:
    runtime, calls = _runtime(tmp_path, face_count=count)
    evidence = runtime(Frame(_pixels(), 1_000_000, 1))
    assert evidence.failure_reason == reason
    assert evidence.face_count == count and evidence.face is None
    assert calls == {"embedding": 0, "passive": 0}
    runtime.close()


@pytest.mark.parametrize(
    "matrix,reason",
    [(None, "head_pose_unavailable"), (np.full((4, 4), np.nan), "head_pose_invalid")],
)
def test_invalid_matrix_never_falls_back_to_solvepnp(
    tmp_path: Path, matrix: np.ndarray | None, reason: str
) -> None:
    result = SimpleNamespace(
        face_landmarks=[[SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]],
        facial_transformation_matrixes=[] if matrix is None else [matrix],
        face_blendshapes=[],
    )
    runtime, _calls = _runtime(tmp_path, result=result)
    evidence = runtime(Frame(_pixels(), 1_000_000, 1))
    assert evidence.failure_reason == reason and evidence.pose is None
    runtime.close()


def test_runtime_rechecks_adapter_identity(tmp_path: Path) -> None:
    runtime, _calls = _runtime(tmp_path)
    runtime.manifest = replace(
        runtime.manifest,
        embedding=replace(
            runtime.manifest.embedding,
            metadata=replace(runtime.manifest.embedding.metadata, version="incompatible"),
        ),
    )
    with pytest.raises(ModelCompatibilityError, match="metadata"):
        runtime.verify_models()
    runtime.close()


@pytest.mark.parametrize("checksum", ["unknown", "abc", "G" * 64])
def test_unpinned_manifest_is_rejected(tmp_path: Path, checksum: str) -> None:
    path = tmp_path / "model"
    path.write_bytes(b"weights")
    with pytest.raises(ModelCompatibilityError, match="checksum"):
        ModelArtifact(path, ModelMetadata("model", "v1", checksum)).verify()


def test_missing_artifact_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelCompatibilityError, match="missing"):
        ModelArtifact(tmp_path / "absent", ModelMetadata("model", "v1", "a" * 64)).verify()


def test_wrong_embedding_dimension_is_rejected_after_gates(tmp_path: Path) -> None:
    runtime, _calls = _runtime(tmp_path, embedding_size=256)
    evidence = runtime(Frame(_pixels(), 1_000_000, 1))
    active = LivenessEvidence(LivenessKind.ACTIVE, EvidenceDecision.PASSED, 1, 1, "v1")
    passive = replace(active, kind=LivenessKind.PASSIVE)
    with pytest.raises(ModelCompatibilityError, match="512D"):
        runtime.extract_embedding(evidence, active=active, passive=passive)
    runtime.close()


def test_timestamps_and_close_are_enforced(tmp_path: Path) -> None:
    runtime, _calls = _runtime(tmp_path)
    frame = Frame(_pixels(), 1_000_000, 1)
    runtime(frame)
    assert runtime(frame).failure_reason == "timestamp_not_monotonic"
    runtime.close()
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime(replace(frame, captured_at_ns=2_000_000))
