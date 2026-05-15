"""End-to-end crawler ingest path: image bytes → [face_embeddings].

Steps: phash → detect → align → embed → cluster → index write.
Liveness is NOT run here — that only applies to user enroll/search.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class IngestResult:
    image_id: str
    phash: str
    embeddings: list[list[float]]
    cluster_ids: list[str]


async def ingest_image(image_bytes: bytes, *, source_url: str) -> IngestResult | None:
    """Returns None if duplicate or no face found."""

    raise NotImplementedError
