"""Backend JSON API client. Use this from web app routes — never raw httpx."""

from __future__ import annotations

from typing import Any

import httpx


class BackendClient:
    def __init__(self, base_url: str):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- Auth ---

    async def login(self, email: str, password: str) -> dict[str, Any]:
        r = await self._client.post("/v1/auth/login", json={"email": email, "password": password})
        r.raise_for_status()
        return r.json()

    async def register(self, email: str, password: str) -> dict[str, Any]:
        r = await self._client.post(
            "/v1/auth/register", json={"email": email, "password": password}
        )
        r.raise_for_status()
        return r.json()

    # --- User ---

    async def get_me(self, *, token: str) -> dict[str, Any]:
        r = await self._client.get("/v1/users/me", headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        return r.json()

    # --- Search / Enroll ---

    async def search(
        self,
        *,
        photo: bytes,
        liveness_blob: bytes,
        token: str,
        photo_filename: str = "photo.jpg",
        photo_content_type: str = "image/jpeg",
        liveness_filename: str = "liveness.jpg",
        liveness_content_type: str = "image/jpeg",
    ) -> list[dict]:
        r = await self._client.post(
            "/v1/search",
            files={
                "photo": (photo_filename, photo, photo_content_type),
                "liveness_blob": (
                    liveness_filename,
                    liveness_blob,
                    liveness_content_type,
                ),
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()["matches"]

    async def enroll(
        self,
        *,
        photo: bytes,
        liveness_blob: bytes,
        token: str,
        photo_filename: str = "photo.jpg",
        photo_content_type: str = "image/jpeg",
        liveness_filename: str = "liveness.jpg",
        liveness_content_type: str = "image/jpeg",
    ) -> dict[str, Any]:
        r = await self._client.post(
            "/v1/enroll",
            files={
                "photo": (photo_filename, photo, photo_content_type),
                "liveness_blob": (
                    liveness_filename,
                    liveness_blob,
                    liveness_content_type,
                ),
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()

    async def list_matches(self, *, token: str) -> list[dict]:
        r = await self._client.get(
            "/v1/matches/",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()

    async def get_match(self, *, token: str, match_id: str) -> dict:
        r = await self._client.get(
            f"/v1/matches/{match_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()
