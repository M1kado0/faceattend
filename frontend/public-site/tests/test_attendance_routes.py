from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest


def _files() -> dict[str, tuple[str, bytes, str]]:
    return {
        "liveness_video": ("liveness.webm", b"video-bytes", "video/webm"),
    }


def _check_in_files() -> dict[str, tuple[str, bytes, str]]:
    return {
        "liveness_video": ("liveness.webm", b"video-bytes", "video/webm"),
    }


def _status_error(status_code: int, detail: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend.test/v1/face-registrations")
    response = httpx.Response(status_code, json={"detail": detail}, request=request)
    return httpx.HTTPStatusError("backend error", request=request, response=response)


def _request_error() -> httpx.RequestError:
    request = httpx.Request("POST", "http://backend.test/v1/face-registrations")
    return httpx.RequestError("backend unavailable", request=request)


@dataclass
class FakeFaceRegistrationClient:
    result: dict | None = None
    face_registrations: list[dict] | None = None
    error: Exception | None = None
    list_error: Exception | None = None
    delete_error: Exception | None = None

    async def create_face_registration(self, **kwargs) -> dict:
        if self.error is not None:
            raise self.error
        return self.result or {
            "registration_id": "registration-1",
            "embedding_model_version": "arcface-r100-v1",
        }

    async def list_face_registrations(self, **kwargs) -> list[dict]:
        if self.list_error is not None:
            raise self.list_error
        if self.face_registrations is not None:
            return self.face_registrations
        return [
            {
                "id": "registration-1",
                "embedding_id": "template-1",
                "embedding_model_version": "arcface-r100-v1",
                "created_at": "2026-05-18T00:00:00Z",
            }
        ]

    async def delete_face_registration(self, **kwargs) -> dict[str, str]:
        if self.delete_error is not None:
            raise self.delete_error
        return {"status": "deleted"}


@dataclass
class FakeAttendanceClient:
    face_registration_result: list[dict] | None = None
    list_result: list[dict] | None = None
    detail_result: dict | None = None
    check_in_result: dict | None = None
    face_registration_error: Exception | None = None
    list_error: Exception | None = None
    detail_error: Exception | None = None
    check_in_error: Exception | None = None

    async def list_face_registrations(self, **kwargs) -> list[dict]:
        if self.face_registration_error is not None:
            raise self.face_registration_error
        if self.face_registration_result is not None:
            return self.face_registration_result
        return [
            {
                "id": "registration-1",
                "embedding_id": "template-1",
                "embedding_model_version": "arcface-r100-v1",
                "created_at": "2026-05-18T00:00:00Z",
            }
        ]

    async def list_attendance_records(self, **kwargs) -> list[dict]:
        if self.list_error is not None:
            raise self.list_error
        if self.list_result is not None:
            return self.list_result
        return [
            {
                "record_id": "record-1",
                "face_registration_id": "registration-1",
                "session_id": "session-1",
                "score": 0.91,
                "checked_in_at": "2026-05-18T00:00:00Z",
                "created_at": "2026-05-18T00:00:00Z",
            }
        ]

    async def get_attendance_record(self, **kwargs) -> dict:
        if self.detail_error is not None:
            raise self.detail_error
        return self.detail_result or {
            "record_id": "record-1",
            "face_registration_id": "registration-1",
            "session_id": "session-1",
            "score": 0.91,
            "checked_in_at": "2026-05-18T00:00:00Z",
            "created_at": "2026-05-18T00:00:00Z",
        }

    async def check_in(self, **kwargs) -> dict:
        if self.check_in_error is not None:
            raise self.check_in_error
        return self.check_in_result or {
            "query_id": "query-1",
            "attendance_records": [
                {
                    "record_id": "record-1",
                    "face_registration_id": "registration-1",
                    "session_id": "session-1",
                    "score": 0.91,
                    "checked_in_at": "2026-05-18T00:00:00Z",
                    "created_at": "2026-05-18T00:00:00Z",
                }
            ],
        }


def test_face_registration_without_session_returns_login_required(client) -> None:
    response = client.post("/face-registrations", files=_files())

    assert response.status_code == 200
    assert "Login required" in response.text


def test_liveness_failure_returns_liveness_partial(client, monkeypatch) -> None:
    face_registration_module = pytest.importorskip("routers.face_registrations")
    monkeypatch.setattr(
        face_registration_module,
        "backend_client",
        FakeFaceRegistrationClient(error=_status_error(403, "liveness_failed")),
    )
    client.cookies.set("session_token", "token")

    response = client.post("/face-registrations", files=_files())

    assert response.status_code == 200
    assert "Liveness check failed" in response.text


def test_backend_unavailable_returns_backend_unavailable_partial(client, monkeypatch) -> None:
    face_registration_module = pytest.importorskip("routers.face_registrations")
    monkeypatch.setattr(
        face_registration_module,
        "backend_client",
        FakeFaceRegistrationClient(error=_request_error()),
    )
    client.cookies.set("session_token", "token")

    response = client.post("/face-registrations", files=_files())

    assert response.status_code == 200
    assert "Service unavailable" in response.text


def test_successful_face_registration_redirects_to_check_in(client, monkeypatch) -> None:
    face_registration_module = pytest.importorskip("routers.face_registrations")
    monkeypatch.setattr(face_registration_module, "backend_client", FakeFaceRegistrationClient())
    client.cookies.set("session_token", "token")

    response = client.post("/face-registrations", files=_files())

    assert response.status_code == 204
    assert response.headers["HX-Redirect"] == "/check-in"


def test_face_registration_limit_reached_returns_limit_partial(client, monkeypatch) -> None:
    face_registration_module = pytest.importorskip("routers.face_registrations")
    monkeypatch.setattr(
        face_registration_module,
        "backend_client",
        FakeFaceRegistrationClient(error=_status_error(409, "face_registration_limit_reached")),
    )
    client.cookies.set("session_token", "token")

    response = client.post("/face-registrations", files=_files())

    assert response.status_code == 200
    assert "Delete an existing face registration before adding another." in response.text


def test_face_registration_page_shows_existing_face_registrations(client, monkeypatch) -> None:
    face_registration_module = pytest.importorskip("routers.face_registrations")
    monkeypatch.setattr(face_registration_module, "backend_client", FakeFaceRegistrationClient())
    client.cookies.set("session_token", "token")

    response = client.get("/face-registration")

    assert response.status_code == 200
    assert "Registered face templates" in response.text
    assert "arcface-r100-v1" in response.text


def test_face_registration_page_disables_form_at_face_registration_limit(
    client, monkeypatch
) -> None:
    face_registration_module = pytest.importorskip("routers.face_registrations")
    face_registrations = [
        {
            "id": f"registration-{index}",
            "embedding_id": f"template-{index}",
            "embedding_model_version": "arcface-r100-v1",
            "created_at": "2026-05-18T00:00:00Z",
        }
        for index in range(3)
    ]
    monkeypatch.setattr(
        face_registration_module,
        "backend_client",
        FakeFaceRegistrationClient(face_registrations=face_registrations),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/face-registration")

    assert response.status_code == 200
    assert "Delete an existing face registration before adding another." in response.text
    assert "disabled" in response.text


def test_delete_face_registration_redirects_to_face_registration(client, monkeypatch) -> None:
    face_registration_module = pytest.importorskip("routers.face_registrations")
    monkeypatch.setattr(face_registration_module, "backend_client", FakeFaceRegistrationClient())
    client.cookies.set("session_token", "token")

    response = client.post("/face-registrations/registration-1/delete")

    assert response.status_code == 204
    assert response.headers["HX-Redirect"] == "/face-registration"


def test_attendance_list_without_session_returns_login_required(client) -> None:
    response = client.get("/attendance", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_check_in_page_shows_registration_prompt_without_face_registration(
    client, monkeypatch
) -> None:
    attendance_module = pytest.importorskip("routers.attendance")
    monkeypatch.setattr(
        attendance_module,
        "backend_client",
        FakeAttendanceClient(face_registration_result=[]),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/check-in")

    assert response.status_code == 200
    assert "Register your face before checking in." in response.text


def test_check_in_without_session_returns_login_required(client) -> None:
    response = client.post("/check-in", files=_check_in_files())

    assert response.status_code == 200
    assert "Login required" in response.text


def test_successful_check_in_redirects_to_attendance(client, monkeypatch) -> None:
    attendance_module = pytest.importorskip("routers.attendance")
    monkeypatch.setattr(attendance_module, "backend_client", FakeAttendanceClient())
    client.cookies.set("session_token", "token")

    response = client.post("/check-in", files=_check_in_files())

    assert response.status_code == 204
    assert response.headers["HX-Redirect"] == "/attendance"


def test_check_in_liveness_failure_returns_liveness_partial(client, monkeypatch) -> None:
    attendance_module = pytest.importorskip("routers.attendance")
    monkeypatch.setattr(
        attendance_module,
        "backend_client",
        FakeAttendanceClient(check_in_error=_status_error(403, "liveness_failed")),
    )
    client.cookies.set("session_token", "token")

    response = client.post("/check-in", files=_check_in_files())

    assert response.status_code == 200
    assert "Liveness check failed" in response.text


def test_check_in_identity_mismatch_returns_warning_partial(client, monkeypatch) -> None:
    attendance_module = pytest.importorskip("routers.attendance")
    monkeypatch.setattr(
        attendance_module,
        "backend_client",
        FakeAttendanceClient(check_in_error=_status_error(403, "identity_not_matched")),
    )
    client.cookies.set("session_token", "token")

    response = client.post("/check-in", files=_check_in_files())

    assert response.status_code == 200
    assert "Face not recognized" in response.text
    assert "not recorded" in response.text


def test_sessions_page_shows_attendance_session(client) -> None:
    response = client.get("/sessions")

    assert response.status_code == 200
    assert "Attendance sessions" in response.text
    assert "Demo Attendance Session" in response.text


def test_reports_page_shows_attendance_reports(client) -> None:
    response = client.get("/reports")

    assert response.status_code == 200
    assert "Attendance reports" in response.text


def test_settings_page_shows_attendance_privacy_settings(client) -> None:
    response = client.get("/settings")

    assert response.status_code == 200
    assert "attendance privacy" in response.text


def test_successful_attendance_list_returns_results_page(client, monkeypatch) -> None:
    attendance_module = pytest.importorskip("routers.attendance")
    monkeypatch.setattr(attendance_module, "backend_client", FakeAttendanceClient())
    client.cookies.set("session_token", "token")

    response = client.get("/attendance")

    assert response.status_code == 200
    assert "Latest check-in recorded" in response.text
    assert "Recent check-ins" in response.text
    assert "record-1" in response.text
    assert "registration-1" in response.text


def test_attendance_list_without_registration_shows_setup_state(client, monkeypatch) -> None:
    attendance_module = pytest.importorskip("routers.attendance")
    monkeypatch.setattr(
        attendance_module,
        "backend_client",
        FakeAttendanceClient(face_registration_result=[], list_result=[]),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/attendance")

    assert response.status_code == 200
    assert "Register your face before checking in to attendance sessions." in response.text


def test_attendance_list_backend_401_returns_login_required(client, monkeypatch) -> None:
    attendance_module = pytest.importorskip("routers.attendance")
    monkeypatch.setattr(
        attendance_module,
        "backend_client",
        FakeAttendanceClient(list_error=_status_error(401, "invalid_token")),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/attendance")

    assert response.status_code == 200
    assert "Login required" in response.text


def test_attendance_list_backend_500_returns_attendance_error(client, monkeypatch) -> None:
    attendance_module = pytest.importorskip("routers.attendance")
    monkeypatch.setattr(
        attendance_module,
        "backend_client",
        FakeAttendanceClient(list_error=_status_error(500, "db_error")),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/attendance")

    assert response.status_code == 200
    assert "Could not load attendance results" in response.text


def test_attendance_detail_404_returns_record_not_found(client, monkeypatch) -> None:
    attendance_module = pytest.importorskip("routers.attendance")
    monkeypatch.setattr(
        attendance_module,
        "backend_client",
        FakeAttendanceClient(detail_error=_status_error(404, "attendance_record_not_found")),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/attendance/missing-record")

    assert response.status_code == 200
    assert "Attendance record not found" in response.text


def test_attendance_detail_shows_attendance_fields(client, monkeypatch) -> None:
    attendance_module = pytest.importorskip("routers.attendance")
    monkeypatch.setattr(attendance_module, "backend_client", FakeAttendanceClient())
    client.cookies.set("session_token", "token")

    response = client.get("/attendance/record-1")

    assert response.status_code == 200
    assert "Checked in" in response.text
    assert "session-1" in response.text
    assert "registration-1" in response.text


def test_attendance_detail_backend_500_returns_attendance_error(client, monkeypatch) -> None:
    attendance_module = pytest.importorskip("routers.attendance")
    monkeypatch.setattr(
        attendance_module,
        "backend_client",
        FakeAttendanceClient(detail_error=_status_error(500, "db_error")),
    )
    client.cookies.set("session_token", "token")

    response = client.get("/attendance/record-1")

    assert response.status_code == 200
    assert "Could not load attendance results" in response.text
