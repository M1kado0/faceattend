"""Framework-independent computer-vision contracts and value types."""

from faceattend.vision.alignment import AlignmentError, InsightFaceAligner
from faceattend.vision.continuity import (
    ContinuityResult,
    ContinuityStatus,
    FaceContinuityConfig,
    FaceContinuityTracker,
)
from faceattend.vision.embeddings import EmbeddingError, InsightFaceEmbedder
from faceattend.vision.face_detector import FaceDetectionError, InsightFaceDetector
from faceattend.vision.head_pose import (
    CanonicalSolvePnPHeadPoseEstimator,
    ExponentialPoseSmoother,
    HeadPoseError,
    PoseHysteresis,
    estimate_matrix_head_pose,
    estimate_solvepnp_head_pose,
)
from faceattend.vision.model_lifecycle import LazyModel

__all__ = [
    "CanonicalSolvePnPHeadPoseEstimator",
    "AlignmentError",
    "InsightFaceAligner",
    "ContinuityResult",
    "ContinuityStatus",
    "ExponentialPoseSmoother",
    "EmbeddingError",
    "FaceContinuityConfig",
    "FaceContinuityTracker",
    "FaceDetectionError",
    "HeadPoseError",
    "InsightFaceDetector",
    "InsightFaceEmbedder",
    "LazyModel",
    "PoseHysteresis",
    "estimate_matrix_head_pose",
    "estimate_solvepnp_head_pose",
]
