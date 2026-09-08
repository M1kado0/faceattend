"""Headless face processing with an explicit, pinned model contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from faceattend.vision.head_pose import HeadPoseError, MediaPipeMatrixHeadPoseEstimator
from faceattend.vision.interfaces import (
    FaceAligner,
    FaceDetector,
    FaceEmbedder,
    PassiveLivenessDetector,
)
from faceattend.vision.model_lifecycle import LazyModel, file_sha256
from faceattend.vision.quality import QualityConfig, measure_quality
from faceattend.vision.types import (
    EvidenceDecision,
    FaceObservation,
    Float32Array,
    Frame,
    FrameEvidence,
    LivenessEvidence,
    LivenessKind,
    ModelMetadata,
)


class ModelCompatibilityError(ValueError):
    """The configured weights or adapter do not match the selected contract."""


class _ModelOptions(TypedDict):
    model_path: str
    model_name: str
    model_version: str
    model_checksum: str


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Expected identity, pinned independently of an adapter's self-reported hash.

    A matching hash proves file identity, not trustworthy provenance or licensing.
    Keep the selected files immutable while the runtime owns loaded models.
    """

    path: Path
    metadata: ModelMetadata

    def verify(self, actual: ModelMetadata | None = None) -> None:
        expected = self.metadata
        if not expected.name.strip() or not expected.version.strip():
            raise ModelCompatibilityError("model name and version must be explicit")
        if len(expected.checksum) != 64 or any(
            character not in "0123456789abcdef" for character in expected.checksum
        ):
            raise ModelCompatibilityError("expected checksum must be a lowercase SHA-256 digest")
        if not self.path.is_file():
            raise ModelCompatibilityError(f"model file is missing: {self.path}")
        if file_sha256(self.path) != expected.checksum:
            raise ModelCompatibilityError(f"model checksum mismatch: {self.path}")
        if actual is not None and actual != expected:
            raise ModelCompatibilityError("adapter metadata does not match selected model")


@dataclass(frozen=True, slots=True)
class RuntimeModelManifest:
    """Explicit selected artifacts; never infer trusted hashes from filenames."""

    detector: ModelArtifact
    embedding: ModelArtifact
    passive: ModelArtifact
    landmarker: ModelArtifact
    alignment_size: int = 112
    embedding_dimension: int = 512

    def __post_init__(self) -> None:
        # This composition implements the existing buffalo_l alignment contract.
        if self.alignment_size != 112 or self.embedding_dimension != 512:
            raise ModelCompatibilityError(
                "this runtime requires 112px alignment and 512D embeddings"
            )


class MediaPipeFrameLandmarker:
    """Synchronous VIDEO-mode matrix/action adapter; one instance per capture stream.

    Native resources load once and must be closed by their owning worker. Raw
    matrix pose is retained; physical direction mapping belongs to the challenge
    boundary, not to evidence collection. There is no solvePnP fallback.
    """

    def __init__(
        self, artifact: ModelArtifact, *, model_loader: Callable[[], Any] | None = None
    ) -> None:
        self.artifact = artifact
        self.model_metadata = artifact.metadata
        self._loaded: Any = None
        self._closed = False
        self._last_ms: int | None = None
        self._actions: Any = None

        def load() -> Any:
            artifact.verify(self.model_metadata)
            if model_loader is not None:
                model = model_loader()
            else:
                vision = import_module("mediapipe.tasks.python.vision")
                tasks = import_module("mediapipe.tasks.python")
                model = vision.FaceLandmarker.create_from_options(
                    vision.FaceLandmarkerOptions(
                        base_options=tasks.BaseOptions(model_asset_path=str(artifact.path)),
                        running_mode=vision.RunningMode.VIDEO,
                        num_faces=2,
                        output_facial_transformation_matrixes=True,
                        output_face_blendshapes=True,
                    )
                )
            self._loaded = model
            return model

        self._model = LazyModel(load)

    def observe(self, frame: Frame) -> Any:
        if self._closed:
            raise RuntimeError("landmarker is closed")
        timestamp_ms = frame.captured_at_ns // 1_000_000
        if self._last_ms is not None and timestamp_ms <= self._last_ms:
            raise ValueError("MediaPipe timestamps must increase at millisecond resolution")
        self._last_ms = timestamp_ms
        mp = import_module("mediapipe")
        rgb = np.ascontiguousarray(frame.pixels[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._model.get().detect_for_video(image, timestamp_ms)

    def action(self, result: Any, timestamp_ms: int) -> Any:
        if self._actions is None:
            self._actions = import_module("ml.liveness.mediapipe_active").MediaPipeActionEvidence()
        return self._actions.observe(result, timestamp_ms)

    def smile_score(self, result: Any) -> float:
        """Expose raw expression evidence without coupling core types to MediaPipe."""
        module = import_module("ml.liveness.mediapipe_active")
        return float(module.mediapipe_smile_score(result))

    def reset_actions(self) -> None:
        self._actions = None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._actions = None
            if self._loaded is not None:
                self._loaded.close()
                self._loaded = None


class HeadlessFaceAnalyzer:
    """Compose existing adapters into a camera-worker evidence processor.

    This is collection, not a liveness decision: callers must handle failure_reason
    and enforce continuity/quality/session policy. No embeddings or PAD inference
    run in __call__. Injected adapters are trusted dependencies; metadata checks
    cannot establish that arbitrary injected Python code uses its declared file.
    """

    def __init__(
        self,
        manifest: RuntimeModelManifest,
        *,
        detector: FaceDetector,
        aligner: FaceAligner,
        embedder: FaceEmbedder,
        passive: PassiveLivenessDetector,
        landmarker: MediaPipeFrameLandmarker,
        quality_config: QualityConfig | None = None,
    ) -> None:
        self.manifest = manifest
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.passive = passive
        self.landmarker = landmarker
        self.quality_config = quality_config or QualityConfig()
        self._closed = False
        self._last_ns: int | None = None
        self._latest: FrameEvidence | None = None
        self.verify_models()

    def verify_models(self) -> None:
        """Rehash selected files and compare adapter identities at runtime startup."""
        validate_model_files(self.manifest)
        self.manifest.detector.verify(self.detector.model_metadata)
        self.manifest.embedding.verify(
            ModelMetadata(
                self.embedder.model_name,
                self.embedder.model_version,
                self.embedder.model_checksum,
            )
        )
        self.manifest.passive.verify(self.passive.model_metadata)
        self.manifest.landmarker.verify(self.landmarker.model_metadata)
        if self.landmarker.artifact.path.resolve() != self.manifest.landmarker.path.resolve():
            raise ModelCompatibilityError("landmarker artifact path mismatch")

    def analyze(self, frame: Frame) -> Sequence[FaceObservation]:
        """Detection only, satisfying FaceAnalyzer; liveness is not implied."""
        if self._closed:
            raise RuntimeError("face analyzer is closed")
        pixels = frame.pixels
        if (
            pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape[2] != 3
            or not pixels.shape[0]
            or not pixels.shape[1]
        ):
            raise ValueError("frame must be a nonempty HWC uint8 BGR image")
        return self.detector.detect(frame)

    def __call__(self, frame: Frame) -> FrameEvidence:
        self._latest = None
        if not isinstance(frame.captured_at_ns, int) or frame.captured_at_ns < 0:
            self.landmarker.reset_actions()
            return FrameEvidence(frame, 0, None, None, None, failure_reason="invalid_timestamp")
        if self._last_ns is not None and frame.captured_at_ns <= self._last_ns:
            self.landmarker.reset_actions()
            return FrameEvidence(
                frame, 0, None, None, None, failure_reason="timestamp_not_monotonic"
            )
        self._last_ns = frame.captured_at_ns
        faces = self.analyze(frame)
        count = len(faces)
        face = faces[0] if count == 1 else None

        def failed(reason: str) -> FrameEvidence:
            self.landmarker.reset_actions()
            return FrameEvidence(frame, count, face, None, None, failure_reason=reason)

        if count != 1:
            return failed("no_face" if count == 0 else "multiple_faces")
        assert face is not None
        quality = measure_quality(frame, face.bbox, face.landmarks, self.quality_config)
        face = replace(face, quality=quality)
        if not quality.passed:
            return failed(quality.reason or "poor_quality")
        if face.landmarks.shape != (5, 2) or not np.isfinite(face.landmarks).all():
            return failed("invalid_alignment_landmarks")
        result = self.landmarker.observe(frame)
        landmarks: Sequence[Any] = getattr(result, "face_landmarks", ())
        landmark_count = len(landmarks)
        if landmark_count != 1:
            if landmark_count > 1:
                count, face = landmark_count, None
            return failed("multiple_faces" if landmark_count > 1 else "landmarker_no_face")
        try:
            coordinates = np.asarray([(p.x, p.y) for p in landmarks[0]], dtype=np.float64)
            if len(coordinates) < 468 or not np.isfinite(coordinates).all():
                return failed("invalid_mediapipe_landmarks")
        except (AttributeError, TypeError, ValueError):
            return failed("invalid_mediapipe_landmarks")
        matrices: Sequence[Any] = getattr(result, "facial_transformation_matrixes", ())
        if len(matrices) != 1:
            return failed("head_pose_unavailable")
        try:
            pose = MediaPipeMatrixHeadPoseEstimator().estimate(np.asarray(matrices[0]))
        except (HeadPoseError, ValueError, TypeError):
            return failed("head_pose_invalid")
        try:
            action = self.landmarker.action(result, frame.captured_at_ns // 1_000_000)
            smile_score = self.landmarker.smile_score(result)
        except (ValueError, TypeError, AttributeError):
            return failed("invalid_action_evidence")
        evidence = FrameEvidence(
            frame,
            count,
            face,
            face.track_id,
            pose,
            action=action,
            smile_score=smile_score,
            quality=quality,
            lighting_score=quality.brightness,
        )
        self._latest = evidence
        return evidence

    def extract_embedding(
        self,
        evidence: FrameEvidence,
        *,
        active: LivenessEvidence,
        passive: LivenessEvidence,
    ) -> Float32Array:
        """Align/embed only with both passes supplied by trusted orchestration.

        These result objects are not authentication tokens. The application must
        bind them to the same continuous session; that orchestration is separate.
        Earlier in-memory neutral candidates from that completed session are
        valid because registration intentionally extracts several templates.
        """
        if self._closed:
            raise RuntimeError("face analyzer is closed")
        if (
            active.kind is not LivenessKind.ACTIVE
            or passive.kind is not LivenessKind.PASSIVE
            or active.decision is not EvidenceDecision.PASSED
            or passive.decision is not EvidenceDecision.PASSED
        ):
            raise ValueError("both active and passive liveness must pass before embedding")
        if (
            active.model_version != self.manifest.landmarker.metadata.version
            or passive.model_version != self.manifest.passive.metadata.version
        ):
            raise ModelCompatibilityError("liveness model version mismatch")
        if (
            evidence.failure_reason is not None
            or evidence.face_count != 1
            or evidence.face is None
        ):
            raise ValueError("embedding requires the latest valid single-face evidence")
        self.manifest.embedding.verify(
            ModelMetadata(
                self.embedder.model_name,
                self.embedder.model_version,
                self.embedder.model_checksum,
            )
        )
        aligned = self.aligner.align(evidence.frame, evidence.face)
        if aligned.shape != (112, 112, 3) or aligned.dtype != np.uint8:
            raise ModelCompatibilityError("alignment output violates the 112px BGR contract")
        vector = self.embedder.embed(aligned)
        if (
            vector.shape != (512,)
            or vector.dtype != np.float32
            or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
        ):
            raise ModelCompatibilityError("embedding output violates the normalized 512D contract")
        return vector

    def close(self) -> None:
        """Close native MediaPipe resources; no camera ownership or persistence."""
        self._closed = True
        self._latest = None
        self.landmarker.close()


def create_face_analyzer(
    manifest: RuntimeModelManifest, *, quality_config: QualityConfig | None = None
) -> HeadlessFaceAnalyzer:
    """Construct the real local adapters from verified files; no downloads.

    Expected checksums must be selected explicitly by the application/operator.
    Do not populate a trusted manifest automatically from whatever files happen
    to be present. Models remain lazily loaded and must not change during use.
    """
    from faceattend.vision.alignment import InsightFaceAligner
    from faceattend.vision.embeddings import InsightFaceEmbedder
    from faceattend.vision.face_detector import InsightFaceDetector
    from faceattend.vision.passive_liveness import MiniFASNetPassiveLivenessDetector

    def options(artifact: ModelArtifact) -> _ModelOptions:
        artifact.verify()
        return {
            "model_path": str(artifact.path),
            "model_name": artifact.metadata.name,
            "model_version": artifact.metadata.version,
            "model_checksum": artifact.metadata.checksum,
        }

    return HeadlessFaceAnalyzer(
        manifest,
        detector=InsightFaceDetector(**options(manifest.detector)),
        aligner=InsightFaceAligner(image_size=manifest.alignment_size),
        embedder=InsightFaceEmbedder(**options(manifest.embedding)),
        passive=MiniFASNetPassiveLivenessDetector(**options(manifest.passive)),
        landmarker=MediaPipeFrameLandmarker(manifest.landmarker),
        quality_config=quality_config,
    )


def validate_model_files(manifest: RuntimeModelManifest) -> None:
    """Check pinned files and baseline ONNX tensor contracts without ORT sessions.

    Tensor shapes cannot verify training provenance or class-label semantics.
    The selected PAD manifest must refer to the existing Real-at-index-1 export.
    MediaPipe validates its task archive when its native model is loaded.
    """
    onnx = import_module("onnx")
    for role in ("detector", "embedding", "passive"):
        artifact = getattr(manifest, role)
        artifact.verify()
        try:
            model = onnx.load(str(artifact.path), load_external_data=False)
        except Exception as exc:
            raise ModelCompatibilityError(f"{role} is not a readable ONNX model") from exc
        # One pinned file must represent all weights; reject unpinned external data.
        if any(
            tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer
        ):
            raise ModelCompatibilityError(
                f"{role} external weights are not covered by the checksum"
            )

        def shape(value: Any) -> tuple[int, ...]:
            return tuple(int(d.dim_value) for d in value.type.tensor_type.shape.dim)

        inputs, outputs = model.graph.input, model.graph.output
        valid = len(inputs) == 1 and inputs[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        dimensions = shape(inputs[0]) if len(inputs) == 1 else ()
        valid = valid and len(dimensions) == 4 and dimensions[0] in (0, 1) and dimensions[1] == 3
        out_shapes = [shape(value) for value in outputs]
        valid = valid and all(
            v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for v in outputs
        )
        if role == "detector":
            # Existing SCRFD-10G: three strides, score/bbox/five-keypoint heads.
            valid = valid and dimensions[2:] in ((0, 0), (640, 640))
            valid = (
                valid
                and len(out_shapes) == 9
                and [s[-1:] for s in out_shapes]
                == [(1,), (1,), (1,), (4,), (4,), (4,), (10,), (10,), (10,)]
                and all(len(s) == 2 for s in out_shapes)
            )
        elif role == "embedding":
            valid = (
                valid
                and dimensions[2:] == (112, 112)
                and out_shapes
                in (
                    [(1, 512)],
                    [(0, 512)],
                )
            )
        else:
            valid = (
                valid
                and dimensions[2:] == (80, 80)
                and out_shapes
                in (
                    [(1, 3)],
                    [(0, 3)],
                )
            )
        if not valid:
            raise ModelCompatibilityError(f"{role} ONNX tensor contract is incompatible")
    manifest.landmarker.verify()
