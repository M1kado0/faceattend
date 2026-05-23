"""Active liveness challenge/response checks."""

from pathlib import Path

from ml.liveness.base import LivenessResult
from ml.liveness.challenge import ActiveLivenessChallenge
from ml.liveness.mediapipe_active import MediaPipeActiveLivenessChecker


class ActiveLivenessChecker:
    def __init__(self, model_path: str | Path):
        self.model_path = model_path
        self.model = MediaPipeActiveLivenessChecker(model_path)

    def check(
        self,
        blob: bytes,
        challenge: ActiveLivenessChallenge | str,
    ) -> LivenessResult:
        # Verify challenge was performed AND the same face is present throughout.
        return self.model.check(blob, challenge)
