"""Small HTTP client helpers for the ML service."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import numpy as np


@dataclass(frozen=True)
class LivenessCheck:
    passed: bool
    score: float
    label: str
    reason: str | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: np.ndarray
    model_version: str


class MLServiceError(Exception):
    """Base class for expected ML service failures."""


class MLServiceUnavailableError(MLServiceError):
    """Raised when the ML service cannot be reached."""


class MLServiceRejectedError(MLServiceError):
    """Raised when the ML service rejects a request with an HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


async def verify_passive_liveness(
    *,
    ml_service_url: str,
    blob: bytes,
    filename: str = "liveness.jpg",
    content_type: str = "application/octet-stream",
) -> LivenessCheck:
    """Call the passive liveness endpoint before any biometric embedding work."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ml_service_url}/v1/liveness/passive",
                files={"blob": (filename, blob, content_type)},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise MLServiceRejectedError(exc.response.status_code, exc.response.text) from exc
    except httpx.RequestError as exc:
        raise MLServiceUnavailableError("ml_service_unavailable") from exc

    data = response.json()
    return LivenessCheck(
        passed=bool(data["passed"]),
        score=float(data["score"]),
        label=str(data.get("label", "")),
        reason=data.get("reason"),
    )

async def verify_active_liveness(
    *,
    ml_service_url: str,
    blob: bytes,
    challenge: str,
    filename: str = "liveness.webm",
    content_type: str = "video/webm",
) -> LivenessCheck:
    """Call the active liveness endpoint before any biometric embedding work."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ml_service_url}/v1/liveness/active",
                data={"challenge": challenge},
                files={"blob": (filename, blob, content_type)},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise MLServiceRejectedError(exc.response.status_code, exc.response.text) from exc
    except httpx.RequestError as exc:
        raise MLServiceUnavailableError("ml_service_unavailable") from exc

    data = response.json()
    return LivenessCheck(
        passed=bool(data["passed"]),
        score=float(data["score"]),
        label=str(data.get("label", "")),
        reason=data.get("reason"),
    )


async def embed_image(
    *,
    ml_service_url: str,
    image: bytes,
    filename: str = "photo.jpg",
    content_type: str = "application/octet-stream",
) -> EmbeddingResult:
    """Call the ML embedding endpoint and normalize the wire response."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ml_service_url}/v1/embed",
                files={"image": (filename, image, content_type)},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise MLServiceRejectedError(exc.response.status_code, exc.response.text) from exc
    except httpx.RequestError as exc:
        raise MLServiceUnavailableError("ml_service_unavailable") from exc

    data = response.json()
    return EmbeddingResult(
        embedding=np.array(data["embedding"], dtype=np.float32),
        model_version=str(data["model_version"]),
    )
