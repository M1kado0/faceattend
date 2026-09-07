"""Framework-independent computer-vision contracts and value types."""

from faceattend.vision.active_liveness import (
    ActiveLivenessCalibration,
    ActiveLivenessChallengeEvaluator,
    ChallengePhase,
    ChallengeResult,
    ChallengeStatus,
)
from faceattend.vision.alignment import AlignmentError, InsightFaceAligner
from faceattend.vision.challenge_session import (
    ChallengeAction,
    ChallengeSession,
    ChallengeSessionConfig,
    ChallengeSessionSnapshot,
    ChallengeSessionStatus,
)
from faceattend.vision.continuity import (
    ContinuityResult,
    ContinuityStatus,
    FaceContinuityConfig,
    FaceContinuityTracker,
)
from faceattend.vision.embeddings import EmbeddingError, InsightFaceEmbedder
from faceattend.vision.evidence import DuplicateFrameDetector
from faceattend.vision.face_detector import FaceDetectionError, InsightFaceDetector
from faceattend.vision.head_pose import (
    CanonicalSolvePnPHeadPoseEstimator,
    ExponentialPoseSmoother,
    HeadPoseError,
    PoseHysteresis,
    estimate_matrix_head_pose,
    estimate_solvepnp_head_pose,
)
from faceattend.vision.matcher import ExactNumpyMatcher
from faceattend.vision.model_lifecycle import LazyModel
from faceattend.vision.passive_liveness import (
    MiniFASNetPassiveLivenessDetector,
    PassiveLivenessError,
    TemporalPassiveLivenessSession,
)
from faceattend.vision.types import FrameEvidence

__all__ = [
    "CanonicalSolvePnPHeadPoseEstimator",
    "ChallengeAction",
    "ChallengeSession",
    "ChallengeSessionConfig",
    "ChallengeSessionSnapshot",
    "ChallengeSessionStatus",
    "ActiveLivenessCalibration",
    "ActiveLivenessChallengeEvaluator",
    "ChallengePhase",
    "ChallengeResult",
    "ChallengeStatus",
    "AlignmentError",
    "InsightFaceAligner",
    "ContinuityResult",
    "ContinuityStatus",
    "ExponentialPoseSmoother",
    "EmbeddingError",
    "ExactNumpyMatcher",
    "DuplicateFrameDetector",
    "FaceContinuityConfig",
    "FaceContinuityTracker",
    "FaceDetectionError",
    "FrameEvidence",
    "HeadPoseError",
    "InsightFaceDetector",
    "InsightFaceEmbedder",
    "LazyModel",
    "MiniFASNetPassiveLivenessDetector",
    "PassiveLivenessError",
    "TemporalPassiveLivenessSession",
    "PoseHysteresis",
    "estimate_matrix_head_pose",
    "estimate_solvepnp_head_pose",
]
