"""Replaceable CV interfaces with no GUI, web, or persistence dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from faceattend.vision.types import (
    EmbeddingTemplate,
    FaceObservation,
    Float32Array,
    Frame,
    HeadPose,
    LivenessEvidence,
    MatchDecision,
    PoseEvidence,
    UInt8Array,
)


class FaceDetector(Protocol):
    def detect(self, frame: Frame) -> Sequence[FaceObservation]: ...


class FaceAligner(Protocol):
    def align(self, frame: Frame, face: FaceObservation) -> UInt8Array: ...


class FaceEmbedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    def embed(self, aligned_face: UInt8Array) -> Float32Array: ...


class HeadPoseEstimator(Protocol):
    def estimate(self, frame: Frame, face: FaceObservation) -> HeadPose: ...


class ActiveLivenessEvaluator(Protocol):
    def reset(self, challenge_steps: tuple[str, ...]) -> None: ...

    def observe(
        self,
        *,
        frame: Frame,
        face: FaceObservation,
        pose: HeadPose,
    ) -> tuple[PoseEvidence, LivenessEvidence]: ...


class PassiveLivenessDetector(Protocol):
    def evaluate(
        self,
        frames: Sequence[Frame],
        faces: Sequence[FaceObservation],
    ) -> LivenessEvidence: ...


class FaceAnalyzer(Protocol):
    def analyze(self, frame: Frame) -> Sequence[FaceObservation]: ...


class Matcher(Protocol):
    def rebuild(self, templates: Sequence[EmbeddingTemplate]) -> None: ...

    def match(self, embedding: Float32Array) -> MatchDecision: ...
