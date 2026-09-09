"""Behavior tests for exact person-level cosine matching."""

from datetime import UTC, datetime

import numpy as np

from faceattend.vision.matcher import ExactNumpyMatcher
from faceattend.vision.types import (
    EmbeddingTemplate,
    FaceQuality,
    MatchStatus,
)

NOW = datetime(2026, 9, 7, tzinfo=UTC)
QUALITY = FaceQuality(True, 200.0, 0.5, 0.1)


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _template(template_id: str, person_id: str, vector: np.ndarray) -> EmbeddingTemplate:
    return EmbeddingTemplate(
        template_id,
        person_id,
        vector,
        "model",
        "v1",
        "checksum",
        True,
        None,
        QUALITY,
        NOW,
    )


def test_exact_matcher_rejects_unknown_and_ambiguous_people() -> None:
    matcher = ExactNumpyMatcher(match_threshold=0.8, ambiguity_margin=0.02)
    matcher.rebuild(
        (
            _template("a-front", "person-a", _unit(1, 0, 0)),
            _template("a-pose", "person-a", _unit(0.98, -0.2, 0)),
            _template("b-front", "person-b", _unit(0.995, 0.1, 0)),
        )
    )

    ambiguous = matcher.match(_unit(1, 0, 0))
    assert ambiguous.status is MatchStatus.AMBIGUOUS
    assert ambiguous.best is not None and ambiguous.best.person_id == "person-a"
    assert ambiguous.second_best is not None
    assert ambiguous.second_best.person_id == "person-b"

    unknown = matcher.match(_unit(0, 0, 1))
    assert unknown.status is MatchStatus.UNKNOWN


def test_multiple_templates_from_same_person_do_not_create_false_ambiguity() -> None:
    matcher = ExactNumpyMatcher(match_threshold=0.75, ambiguity_margin=0.05)
    matcher.rebuild(
        (
            _template("a-1", "person-a", _unit(1, 0, 0)),
            _template("a-2", "person-a", _unit(0.99, 0.1, 0)),
            _template("b-1", "person-b", _unit(0, 1, 0)),
        )
    )

    decision = matcher.match(_unit(1, 0, 0))
    assert decision.status is MatchStatus.MATCHED
    assert decision.best is not None and decision.best.person_id == "person-a"
