"""Exact in-memory cosine matcher for small local attendance deployments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from faceattend.vision.types import (
    EmbeddingTemplate,
    Float32Array,
    MatchCandidate,
    MatchDecision,
    MatchStatus,
)


class ExactNumpyMatcher:
    """Search normalized templates exactly, then decide at person level."""

    def __init__(self, *, match_threshold: float, ambiguity_margin: float) -> None:
        if not -1.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be between -1 and 1")
        if not 0.0 <= ambiguity_margin <= 2.0:
            raise ValueError("ambiguity_margin must be between 0 and 2")
        self.match_threshold = match_threshold
        self.ambiguity_margin = ambiguity_margin
        self._templates: tuple[EmbeddingTemplate, ...] = ()
        self._matrix: Float32Array | None = None

    def rebuild(self, templates: Sequence[EmbeddingTemplate]) -> None:
        if not templates:
            self._templates = ()
            self._matrix = None
            return
        identities = {
            (item.model_name, item.model_version, item.model_checksum) for item in templates
        }
        if len(identities) != 1:
            raise ValueError("matcher cannot mix embedding model versions")
        dimension = templates[0].embedding.size
        vectors: list[Float32Array] = []
        for template in templates:
            vector = np.asarray(template.embedding)
            if (
                vector.dtype != np.float32
                or vector.ndim != 1
                or vector.size != dimension
                or not template.normalized
                or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
            ):
                raise ValueError("matcher requires compatible normalized float32 templates")
            vectors.append(vector)
        self._templates = tuple(templates)
        self._matrix = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)

    def match(self, embedding: Float32Array) -> MatchDecision:
        query = np.asarray(embedding)
        if (
            query.dtype != np.float32
            or query.ndim != 1
            or not np.isfinite(query).all()
            or not np.isclose(np.linalg.norm(query), 1.0, atol=1e-5)
        ):
            raise ValueError("query must be a normalized finite float32 vector")
        if self._matrix is None:
            return self._decision(MatchStatus.UNKNOWN, None, None)
        if query.size != self._matrix.shape[1]:
            raise ValueError("query dimension does not match templates")

        scores = self._matrix @ query
        best_by_person: dict[str, MatchCandidate] = {}
        for template, score in zip(self._templates, scores, strict=True):
            candidate = MatchCandidate(template.person_id, template.template_id, float(score))
            previous = best_by_person.get(template.person_id)
            if previous is None or candidate.score > previous.score:
                best_by_person[template.person_id] = candidate
        ranked = sorted(best_by_person.values(), key=lambda item: item.score, reverse=True)
        best = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        if best is None or best.score < self.match_threshold:
            return self._decision(MatchStatus.UNKNOWN, best, second)
        if second is not None and best.score - second.score < self.ambiguity_margin:
            return self._decision(MatchStatus.AMBIGUOUS, best, second)
        return self._decision(MatchStatus.MATCHED, best, second)

    def _decision(
        self,
        status: MatchStatus,
        best: MatchCandidate | None,
        second: MatchCandidate | None,
    ) -> MatchDecision:
        return MatchDecision(
            status,
            best,
            second,
            match_threshold=self.match_threshold,
            ambiguity_margin=self.ambiguity_margin,
        )
