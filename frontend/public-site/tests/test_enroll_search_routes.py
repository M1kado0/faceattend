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
    error: Exception | None = None

    async def enroll(self, **kwargs) -> dict:
        if self.error is not None:
            raise self.error
        return self.result or {
            "enrollment_id": "enrollment-1",
            "embedding_model_version": "arcface-r100-v1",
        }


@dataclass
class FakeSearchClient:
    result: list[dict] | None = None
    error: Exception | None = None

    async def search(self, **kwargs) -> list[dict]:
        if self.error is not None:
            raise self.error
        return self.result or [
            {
                "match_id": "match-1",
                "source_url": "https://example.test/image.jpg",
                "source_page": "https://example.test/page",
                "score": 0.91,
                "crawled_at": "2026-05-18T00:00:00Z",
                "image_thumbnail_url": None,
            }
        ]


@dataclass
class FakeMatchesClient:
    list_result: list[dict] | None = None
    detail_result: dict | None = None
    list_error: Exception | None = None
    detail_error: Exception | None = None

    async def list_matches(self, **kwargs) -> list[dict]:
        if self.list_error is not None:
            raise self.list_error
        return self.list_result or [
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


def test_search_without_session_returns_login_required(client) -> None:
    response = client.post("/search", files=_files())

    assert response.status_code == 200
    assert "Login required" in response.text


@pytest.mark.parametrize(
    ("path", "module_name"),
    [("/enroll", "routers.enroll"), ("/search", "routers.search")],
)
def test_liveness_failure_returns_liveness_partial(client, monkeypatch, path, module_name) -> None:
    module = pytest.importorskip(module_name)
    fake_client = (
        FakeEnrollClient(error=_status_error(403, "liveness_failed"))
        if path == "/enroll"
        else FakeSearchClient(error=_status_error(403, "liveness_failed"))
    )
    monkeypatch.setattr(module, "backend_client", fake_client)
    client.cookies.set("session_token", "token")

    response = client.post(path, files=_files())

    assert response.status_code == 200
    assert "Liveness check failed" in response.text


@pytest.mark.parametrize(
    ("path", "module_name"),
    [("/enroll", "routers.enroll"), ("/search", "routers.search")],
)
def test_backend_unavailable_returns_backend_unavailable_partial(
    client,
    monkeypatch,
    path,
    module_name,
) -> None:
    module = pytest.importorskip(module_name)
    fake_client = (
        FakeEnrollClient(error=_request_error())
        if path == "/enroll"
        else FakeSearchClient(error=_request_error())
    )
    monkeypatch.setattr(module, "backend_client", fake_client)
    client.cookies.set("session_token", "token")

    response = client.post(path, files=_files())

    assert response.status_code == 200
    assert "Service unavailable" in response.text


def test_successful_enroll_returns_success_partial(client, monkeypatch) -> None:
    enroll_module = pytest.importorskip("routers.enroll")
    monkeypatch.setattr(enroll_module, "backend_client", FakeEnrollClient())
    client.cookies.set("session_token", "token")

    response = client.post("/enroll", files=_files())

    assert response.status_code == 200
    assert "Enrollment saved" in response.text
    assert "arcface-r100-v1" in response.text


def test_successful_search_returns_results_partial(client, monkeypatch) -> None:
    search_module = pytest.importorskip("routers.search")
    monkeypatch.setattr(search_module, "backend_client", FakeSearchClient())
    client.cookies.set("session_token", "token")

    response = client.post("/search", files=_files())

    assert response.status_code == 200
    assert "Potential match" in response.text
    assert "match-1" in response.text


def test_matches_list_without_session_returns_login_required(client) -> None:
    response = client.get("/matches")

    assert response.status_code == 200
    assert "Login required" in response.text


def test_successful_matches_list_returns_matches_page(client, monkeypatch) -> None:
    matches_module = pytest.importorskip("routers.matches")
    monkeypatch.setattr(matches_module, "backend_client", FakeMatchesClient())
    client.cookies.set("session_token", "token")

    response = client.get("/matches")

    assert response.status_code == 200
    assert "Potential match" in response.text
    assert "match-1" in response.text


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
