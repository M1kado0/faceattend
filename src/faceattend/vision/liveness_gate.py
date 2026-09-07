"""Single public guard for work that must occur only after full liveness."""

from collections.abc import Callable
from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from faceattend.vision.types import EvidenceDecision

ResultT = TypeVar("ResultT")


class LivenessDecision(Protocol):
    @property
    def decision(self) -> EvidenceDecision: ...


class LivenessGateError(RuntimeError):
    """Identity work was requested without a complete liveness pass."""


def extract_and_match_if_live(
    liveness: LivenessDecision,
    extract_embedding: Callable[[], NDArray[np.float32]],
    match: Callable[[NDArray[np.float32]], ResultT],
) -> ResultT:
    if liveness.decision is not EvidenceDecision.PASSED:
        raise LivenessGateError("active and passive liveness must pass before identity work")
    return match(extract_embedding())
