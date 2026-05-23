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

    # --- Attendance / Face registration ---

    async def create_face_registration(
        self,
        *,
        liveness_video: bytes,
        token: str,
        liveness_video_filename: str = "liveness.webm",
        liveness_video_content_type: str = "video/webm",
    ) -> dict[str, Any]:
        r = await self._client.post(
            "/v1/face-registrations",
            files={
                "liveness_video": (
                    liveness_video_filename,
                    liveness_video,
                    liveness_video_content_type,
                ),
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()

    async def list_face_registrations(self, *, token: str) -> list[dict]:
        r = await self._client.get(
            "/v1/face-registrations",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()

    async def delete_face_registration(
        self,
        *,
        token: str,
        registration_id: str,
    ) -> dict[str, str]:
        r = await self._client.delete(
            f"/v1/face-registrations/{registration_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()

    async def list_attendance_records(self, *, token: str) -> list[dict]:
        r = await self._client.get(
            "/v1/attendance-records",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()

    async def get_attendance_record(self, *, token: str, record_id: str) -> dict:
        r = await self._client.get(
            f"/v1/attendance-records/{record_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()
