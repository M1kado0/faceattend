"""Headless MiniFASNet passive presentation-attack detector adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from statistics import median
from typing import Any

from faceattend.vision.evidence import evidence_failure
from faceattend.vision.model_lifecycle import LazyModel, file_sha256
from faceattend.vision.types import (
    EvidenceDecision,
    FaceObservation,
    Frame,
    FrameEvidence,
    LivenessEvidence,
    LivenessKind,
    ModelMetadata,
)


class PassiveLivenessError(ValueError):
    """Raised when passive-liveness input or model output is invalid."""


class TemporalPassiveLivenessSession:
    """Bounded temporal window for passive PAD during one active session."""

    def __init__(
        self,
        detector: MiniFASNetPassiveLivenessDetector,
        *,
        max_frames: int = 10,
        min_frames: int | None = None,
        min_duration_ms: int = 1_000,
        sample_interval_ms: int = 200,
    ) -> None:
        resolved_min_frames = min(5, max_frames) if min_frames is None else min_frames
        if max_frames <= 0 or resolved_min_frames <= 0 or resolved_min_frames > max_frames:
            raise ValueError("frame limits must be positive and min_frames <= max_frames")
        if min_duration_ms < 0 or sample_interval_ms < 0:
            raise ValueError("PAD durations must be non-negative")
        self.detector = detector
        self.max_frames = max_frames
        self.min_frames = resolved_min_frames
        self.min_duration_ns = min_duration_ms * 1_000_000
        self.sample_interval_ns = sample_interval_ms * 1_000_000
        self._frames: list[Frame] = []
        self._faces: list[FaceObservation] = []
        self._last_timestamp_ns: int | None = None
        self._finalized = False
        self._result: LivenessEvidence | None = None

    def observe(self, frame: Frame, face: FaceObservation) -> None:
        if self._finalized:
            raise RuntimeError("passive session is already finalized")
        if self._last_timestamp_ns is not None and frame.captured_at_ns <= self._last_timestamp_ns:
            raise ValueError("frame timestamps must be strictly increasing")
        self._last_timestamp_ns = frame.captured_at_ns
        if self._frames and (
            frame.captured_at_ns - self._frames[-1].captured_at_ns < self.sample_interval_ns
        ):
            return
        # Retain the first bounded sample set. Earlier suspicious evidence must
        # not disappear merely because the active challenge took longer.
        if len(self._frames) == self.max_frames:
            return
        self._frames.append(frame)
        self._faces.append(face)

    @property
    def frames(self) -> tuple[Frame, ...]:
        return tuple(self._frames)

    @property
    def ready(self) -> bool:
        duration = (
            self._frames[-1].captured_at_ns - self._frames[0].captured_at_ns
            if len(self._frames) >= 2
            else 0
        )
        return len(self._frames) >= self.min_frames and duration >= self.min_duration_ns

    def reset(self) -> None:
        """Discard an incomplete neutral window without terminating the session."""
        if self._finalized:
            raise RuntimeError("passive session is already finalized")
        self._frames.clear()
        self._faces.clear()
        self._last_timestamp_ns = None

    def observe_evidence(self, evidence: FrameEvidence) -> None:
        """Consume the frame and face fields from the shared evidence record."""
        if self._finalized:
            return
        reason = evidence_failure(evidence)
        if reason:
            self.abort(reason)
            return
        assert evidence.face is not None
        try:
            self.observe(evidence.frame, evidence.face)
        except (ValueError, RuntimeError):
            self.abort("invalid_passive_input")

    @property
    def result(self) -> LivenessEvidence | None:
        return self._result

    def abort(self, reason: str) -> LivenessEvidence:
        """Discard transient pixels and preserve a terminal failure."""
        if self._result is None:
            self._result = LivenessEvidence(
                LivenessKind.PASSIVE,
                EvidenceDecision.FAILED,
                None,
                None,
                self.detector.model_metadata.version,
                reason=reason,
                failure_to_process_count=1,
            )
        self._finalized = True
        self._frames.clear()
        self._faces.clear()
        return self._result

    def finalize(self) -> LivenessEvidence:
        if self._result is not None:
            return self._result
        if not self.ready:
            return self.abort("insufficient_passive_evidence")
        self._finalized = True
        try:
            self._result = self.detector.evaluate_temporal(self._frames, self._faces)
        except (ValueError, TypeError, RuntimeError, OSError):
            return self.abort("passive_inference_failed")
        finally:
            self._frames.clear()
            self._faces.clear()
        return self._result

    def cancel(self) -> LivenessEvidence:
        return self.abort("cancelled")


class MiniFASNetPassiveLivenessDetector:
    """Evaluate one or more tracked faces using the existing MiniFASNet model.

    A temporal window is accepted only when every frame's score reaches the
    threshold. This conservative aggregation prevents one good frame from
    hiding a failed frame during registration or check-in.
    """

    def __init__(
        self,
        *,
        model_path: str,
        model_name: str = "MiniFASNetV2",
        model_version: str = "MiniFASNetV2",
        model_checksum: str = "unknown",
        threshold: float = 0.85,
        scale: float = 2.7,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if scale <= 0.0:
            raise ValueError("scale must be positive")

        checksum = model_checksum
        if checksum == "unknown":
            checksum = file_sha256(model_path)
        self._metadata = ModelMetadata(model_name, model_version, checksum)
        self._threshold = threshold

        def load() -> Any:
            if model_loader is not None:
                return model_loader()
            # Import lazily so this headless boundary does not load ONNX Runtime
            # until the detector is actually used.
            minifasnet_module = import_module("faceattend.vision.minifasnet_model")
            return minifasnet_module.MiniFASNetModel(model_path, scale=scale)

        self._model = LazyModel(load)

    @property
    def model_metadata(self) -> ModelMetadata:
        return self._metadata

    def evaluate(
        self,
        frames: Sequence[Frame],
        faces: Sequence[FaceObservation],
    ) -> LivenessEvidence:
        if not frames or not faces:
            raise PassiveLivenessError("at least one frame and face are required")
        if len(frames) != len(faces):
            raise PassiveLivenessError("one face observation is required per frame")

        return self._evaluate_window(frames, faces, strict=True)

    def evaluate_temporal(
        self,
        frames: Sequence[Frame],
        faces: Sequence[FaceObservation],
    ) -> LivenessEvidence:
        """Evaluate a window while recording per-frame processing failures."""
        return self._evaluate_window(frames, faces, strict=False)

    def _evaluate_window(
        self,
        frames: Sequence[Frame],
        faces: Sequence[FaceObservation],
        *,
        strict: bool,
    ) -> LivenessEvidence:
        if not frames or not faces:
            raise PassiveLivenessError("at least one frame and face are required")
        if len(frames) != len(faces):
            raise PassiveLivenessError("one face observation is required per frame")

        scores: list[float] = []
        failures = 0
        for frame, face in zip(frames, faces, strict=True):
            bbox = face.bbox
            bbox_xyxy = [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
            try:
                result = self._model.get().predict(frame.pixels, bbox_xyxy)
                score = float(result["score"])
            except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
                if strict:
                    raise PassiveLivenessError("passive model returned invalid output") from exc
                failures += 1
                continue
            if not 0.0 <= score <= 1.0:
                if strict:
                    raise PassiveLivenessError("passive model score must be between 0 and 1")
                failures += 1
                continue
            scores.append(score)

        if not scores:
            return LivenessEvidence(
                kind=LivenessKind.PASSIVE,
                decision=EvidenceDecision.FAILED,
                score=None,
                threshold=self._threshold,
                model_version=self._metadata.version,
                reason="passive_inference_failed",
                suspicious_frame_count=0,
                failure_to_process_count=failures,
            )
        score = min(scores)
        suspicious = sum(value < self._threshold for value in scores)
        passed = score >= self._threshold and failures == 0
        return LivenessEvidence(
            kind=LivenessKind.PASSIVE,
            decision=EvidenceDecision.PASSED if passed else EvidenceDecision.FAILED,
            score=score,
            threshold=self._threshold,
            model_version=self._metadata.version,
            reason=(
                None
                if passed
                else "passive_inference_failed"
                if failures
                else "liveness_score_below_threshold"
            ),
            median_score=float(median(scores)),
            minimum_score=score,
            suspicious_frame_count=suspicious,
            failure_to_process_count=failures,
        )
