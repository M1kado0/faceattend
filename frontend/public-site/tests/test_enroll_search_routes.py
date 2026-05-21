from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest


def _files() -> dict[str, tuple[str, bytes, str]]:
    return {
        "photo": ("photo.jpg", b"photo-bytes", "image/jpeg"),
        "liveness_blob": ("liveness.jpg", b"liveness-bytes", "image/jpeg"),
    }


def _status_error(status_code: int, detail: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend.test/v1/enroll")
    response = httpx.Response(status_code, json={"detail": detail}, request=request)
    return httpx.HTTPStatusError("backend error", request=request, response=response)


def _request_error() -> httpx.RequestError:
    request = httpx.Request("POST", "http://backend.test/v1/enroll")
    return httpx.RequestError("backend unavailable", request=request)


@dataclass
class FakeEnrollClient:
    result: dict | None = None
    enrollments: list[dict] | None = None
    error: Exception | None = None
    list_error: Exception | None = None
    delete_error: Exception | None = None

    async def enroll(self, **kwargs) -> dict:
        if self.error is not None:
            raise self.error
        return self.result or {
            "enrollment_id": "enrollment-1",
            "embedding_model_version": "arcface-r100-v1",
        }

    async def list_enrollments(self, **kwargs) -> list[dict]:
        if self.list_error is not None:
            raise self.list_error
        if self.enrollments is not None:
            return self.enrollments
        return [
            {
                "id": "enrollment-1",
                "embedding_id": "embedding-1",
                "embedding_model_version": "arcface-r100-v1",
                "created_at": "2026-05-18T00:00:00Z",
            }
        ]

    async def delete_enrollment(self, **kwargs) -> dict[str, str]:
        if self.delete_error is not None:
            raise self.delete_error
        return {"status": "deleted"}


@dataclass
class FakeMatchesClient:
    enrollment_result: list[dict] | None = None
    list_result: list[dict] | None = None
    detail_result: dict | None = None
    enrollment_error: Exception | None = None
    list_error: Exception | None = None
    detail_error: Exception | None = None

    async def list_enrollments(self, **kwargs) -> list[dict]:
        if self.enrollment_error is not None:
            raise self.enrollment_error
        if self.enrollment_result is not None:
            return self.enrollment_result
        return [
            {
                "id": "enrollment-1",
                "embedding_id": "embedding-1",
                "embedding_model_version": "arcface-r100-v1",
                "created_at": "2026-05-18T00:00:00Z",
            }
        ]

    async def list_matches(self, **kwargs) -> list[dict]:
        if self.list_error is not None:
            raise self.list_error
        if self.list_result is not None:
            return self.list_result
        return [
            {
                "match_id": "match-1",
                "source_url": "https://example.test/image.jpg",
                "source_page": "https://example.test/page",
                "score": 0.91,
                "crawled_at": "2026-05-18T00:00:00Z",
                "created_at": "2026-05-18T00:00:00Z",
                "image_thumbnail_url": None,
            }
        ]

    async def get_match(self, **kwargs) -> dict:
        if self.detail_error is not None:
            raise self.detail_error
        return self.detail_result or {
            "match_id": "match-1",
            "source_url": "https://example.test/image.jpg",
            "source_page": "https://example.test/page",
            "score": 0.91,
            "crawled_at": "2026-05-18T00:00:00Z",
            "created_at": "2026-05-18T00:00:00Z",
            "image_thumbnail_url": None,
        }


def test_enroll_without_session_returns_login_required(client) -> None:
    response = client.post("/enroll", files=_files())

    assert response.status_code == 200
    assert "Login required" in response.text


def test_liveness_failure_returns_liveness_partial(client, monkeypatch) -> None:
    enroll_module = pytest.importorskip("routers.enroll")
    monkeypatch.setattr(
        enroll_module,
        "backend_client",
        FakeEnrollClient(error=_status_error(403, "liveness_failed")),
    )
    client.cookies.set("session_token", "token")

    response = client.post("/enroll", files=_files())

    assert response.status_code == 200
    assert "Liveness check failed" in response.text


def test_backend_unavailable_returns_backend_unavailable_partial(client, monkeypatch) -> None:
    enroll_module = pytest.importorskip("routers.enroll")
    monkeypatch.setattr(enroll_module, "backend_client", FakeEnrollClient(error=_request_error()))
    client.cookies.set("session_token", "token")

    response = client.post("/enroll", files=_files())

    assert response.status_code == 200
    assert "Service unavailable" in response.text


def test_successful_enroll_redirects_to_matches(client, monkeypatch) -> None:
    enroll_module = pytest.importorskip("routers.enroll")
    monkeypatch.setattr(enroll_module, "backend_client", FakeEnrollClient())
    client.cookies.set("session_token", "token")

    response = client.post("/enroll", files=_files())

    assert response.status_code == 204
    assert response.headers["HX-Redirect"] == "/matches"


def test_enrollment_limit_reached_returns_limit_partial(client, monkeypatch) -> None:
    enroll_module = pytest.importorskip("routers.enroll")
    monkeypatch.setattr(
        enroll_module,
        "backend_client",
        FakeEnrollClient(error=_status_error(409, "enrollment_limit_reached")),
    )
    client.cookies.set("session_token", "token")

    response = client.post("/enroll", files=_files())

    assert response.status_code == 200
    assert "Delete an existing enrollment before adding another." in response.text


def test_enroll_page_shows_existing_enrollments(client, monkeypatch) -> None:
    enroll_module = pytest.importorskip("routers.enroll")
    monkeypatch.setattr(enroll_module, "backend_client", FakeEnrollClient())
    client.cookies.set("session_token", "token")

    response = client.get("/enroll")

    assert response.status_code == 200
    assert "Active enrollments" in response.text
    assert "arcface-r100-v1" in response.text


def test_enroll_page_disables_form_at_enrollment_limit(client, monkeypatch) -> None:
    enroll_module = pytest.importorskip("routers.enroll")
    enrollments = [
        {
            "id": f"enrollment-{index}",
            "embedding_id": f"embedding-{index}",
            "embedding_model_version": "arcface-r100-v1",
            "created_at": "2026-05-18T00:00:00Z",
        }
        for index in range(3)
    ]
    monkeypatch.setattr(
        enroll_module,
        "backend_client",
        FakeEnrollClient(enrollments=enrollments),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/enroll")

    assert response.status_code == 200
    assert "Delete an existing enrollment before adding another." in response.text
    assert "disabled" in response.text


def test_delete_enrollment_redirects_to_enroll(client, monkeypatch) -> None:
    enroll_module = pytest.importorskip("routers.enroll")
    monkeypatch.setattr(enroll_module, "backend_client", FakeEnrollClient())
    client.cookies.set("session_token", "token")

    response = client.post("/enrollments/enrollment-1/delete")

    assert response.status_code == 204
    assert response.headers["HX-Redirect"] == "/enroll"


def test_matches_list_without_session_returns_login_required(client) -> None:
    response = client.get("/matches", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_successful_matches_list_returns_matches_page(client, monkeypatch) -> None:
    matches_module = pytest.importorskip("routers.matches")
    monkeypatch.setattr(matches_module, "backend_client", FakeMatchesClient())
    client.cookies.set("session_token", "token")

    response = client.get("/matches")

    assert response.status_code == 200
    assert "Potential match" in response.text
    assert "match-1" in response.text


def test_matches_list_without_enrollment_shows_setup_state(client, monkeypatch) -> None:
    matches_module = pytest.importorskip("routers.matches")
    monkeypatch.setattr(
        matches_module,
        "backend_client",
        FakeMatchesClient(enrollment_result=[], list_result=[]),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/matches")

    assert response.status_code == 200
    assert "Enroll your face first to start monitoring matches." in response.text


def test_matches_list_backend_401_returns_login_required(client, monkeypatch) -> None:
    matches_module = pytest.importorskip("routers.matches")
    monkeypatch.setattr(
        matches_module,
        "backend_client",
        FakeMatchesClient(list_error=_status_error(401, "invalid_token")),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/matches")

    assert response.status_code == 200
    assert "Login required" in response.text


def test_matches_list_backend_500_returns_matches_error(client, monkeypatch) -> None:
    matches_module = pytest.importorskip("routers.matches")
    monkeypatch.setattr(
        matches_module,
        "backend_client",
        FakeMatchesClient(list_error=_status_error(500, "db_error")),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/matches")

    assert response.status_code == 200
    assert "Could not load matches" in response.text


def test_match_detail_404_returns_match_not_found(client, monkeypatch) -> None:
    matches_module = pytest.importorskip("routers.matches")
    monkeypatch.setattr(
        matches_module,
        "backend_client",
        FakeMatchesClient(detail_error=_status_error(404, "match_not_found")),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/matches/missing-match")

    assert response.status_code == 200
    assert "Match not found" in response.text


def test_match_detail_backend_500_returns_matches_error(client, monkeypatch) -> None:
    matches_module = pytest.importorskip("routers.matches")
    monkeypatch.setattr(
        matches_module,
        "backend_client",
        FakeMatchesClient(detail_error=_status_error(500, "db_error")),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/matches/match-1")

    assert response.status_code == 200
    assert "Could not load matches" in response.text
