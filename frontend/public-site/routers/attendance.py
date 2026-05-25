"""/attendance — attendance result dashboard + detail."""

import csv
import io
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from services.api_client import BackendClient

load_dotenv()

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8002")
SESSION_COOKIE_NAME = "session_token"

backend_client = BackendClient(BACKEND_API_URL)

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]

templates = Jinja2Templates(
    directory=ROOT / "templates",
)


@router.get("/check-in", response_class=HTMLResponse)
async def check_in_page(request: Request):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return RedirectResponse("/login", status_code=303)
    try:
        face_registrations = await backend_client.list_face_registrations(token=token)
        attendance_sessions = await backend_client.list_attendance_sessions(token=token)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return templates.TemplateResponse(
                request=request,
                name="partials/login_required.html",
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/check_in_error.html",
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="partials/backend_unavailable.html",
        )

    return templates.TemplateResponse(
        request=request,
        name="pages/check_in.html",
        context={
            "has_registration": bool(face_registrations),
            "attendance_sessions": attendance_sessions,
            "selected_session_id": request.query_params.get("session_id"),
        },
    )


@router.post("/check-in", response_class=HTMLResponse)
async def create_check_in(
    request: Request,
    liveness_video: UploadFile,
    session_id: str | None = Form(None),
):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return templates.TemplateResponse(
            request=request,
            name="partials/login_required.html",
        )
    if not session_id:
        return templates.TemplateResponse(
            request=request,
            name="partials/check_in_error.html",
            context={"message": "Choose an open attendance session before checking in."},
        )

    liveness_video_bytes = await liveness_video.read()
    try:
        await backend_client.check_in(
            liveness_video=liveness_video_bytes,
            session_id=session_id,
            token=token,
            liveness_video_filename=liveness_video.filename or "liveness.webm",
            liveness_video_content_type=liveness_video.content_type or "application/octet-stream",
        )
        return Response(
            status_code=204,
            headers={"HX-Redirect": "/attendance"},
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            try:
                data = exc.response.json()
            except ValueError:
                data = {}
            if data.get("detail") in {
                "liveness_failed",
                "passive_liveness_failed",
                "face_not_visible_enough",
            }:
                return templates.TemplateResponse(
                    request=request,
                    name="partials/liveness_failed.html",
                )
            if data.get("detail") == "identity_not_matched":
                return templates.TemplateResponse(
                    request=request,
                    name="partials/identity_not_matched.html",
                )
        if exc.response.status_code == 404:
            try:
                data = exc.response.json()
            except ValueError:
                data = {}
            if data.get("detail") == "attendance_session_not_found":
                return templates.TemplateResponse(
                    request=request,
                    name="partials/check_in_error.html",
                    context={"message": "Choose an open attendance session before checking in."},
                )
        if exc.response.status_code == 401:
            return templates.TemplateResponse(
                request=request,
                name="partials/login_required.html",
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/check_in_error.html",
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="partials/backend_unavailable.html",
        )


@router.get("/attendance", response_class=HTMLResponse)
async def attendance_list(request: Request):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return RedirectResponse("/login", status_code=303)
    try:
        face_registrations = await backend_client.list_face_registrations(token=token)
        attendance_sessions = await backend_client.list_attendance_sessions(token=token)
        selected_session_id = request.query_params.get("session_id")
        attendance_records = await backend_client.list_attendance_records(
            token=token,
            session_id=selected_session_id,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return templates.TemplateResponse(
                request=request,
                name="partials/login_required.html",
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/attendance_error.html",
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="partials/backend_unavailable.html",
        )
    return templates.TemplateResponse(
        request=request,
        name="pages/attendance_results.html",
        context={
            "attendance_records": attendance_records,
            "attendance_sessions": attendance_sessions,
            "session_name_by_id": {
                session["session_id"]: session["name"] for session in attendance_sessions
            },
            "selected_session_id": request.query_params.get("session_id"),
            "has_face_registrations": bool(face_registrations),
        },
    )


@router.get("/attendance.csv")
async def attendance_csv(request: Request):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return RedirectResponse("/login", status_code=303)
    try:
        records = await backend_client.list_attendance_records(
            token=token,
            session_id=request.query_params.get("session_id"),
        )
    except (httpx.HTTPStatusError, httpx.RequestError):
        return templates.TemplateResponse(
            request=request,
            name="partials/attendance_error.html",
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["record_id", "session_id", "status", "score", "checked_in_at"])
    for record in records:
        writer.writerow(
            [
                record.get("record_id", ""),
                record.get("session_id", ""),
                record.get("status", "recorded"),
                record.get("score", ""),
                record.get("checked_in_at", ""),
            ]
        )
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="attendance.csv"'},
    )


@router.get("/sessions", response_class=HTMLResponse)
async def sessions_page(request: Request):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return RedirectResponse("/login", status_code=303)
    try:
        attendance_sessions = await backend_client.list_attendance_sessions(token=token)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return templates.TemplateResponse(
                request=request,
                name="partials/login_required.html",
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/attendance_error.html",
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="partials/backend_unavailable.html",
        )
    return templates.TemplateResponse(
        request=request,
        name="pages/sessions.html",
        context={"sessions": attendance_sessions},
    )


@router.post("/sessions", response_class=HTMLResponse)
async def create_session(request: Request, name: str = Form(...)):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return templates.TemplateResponse(
            request=request,
            name="partials/login_required.html",
        )
    try:
        await backend_client.create_attendance_session(token=token, name=name)
    except (httpx.HTTPStatusError, httpx.RequestError):
        return templates.TemplateResponse(
            request=request,
            name="partials/attendance_error.html",
        )
    return Response(status_code=204, headers={"HX-Redirect": "/sessions"})


@router.get("/attendance/{record_id}", response_class=HTMLResponse)
async def attendance_detail(request: Request, record_id: str):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return templates.TemplateResponse(
            request=request,
            name="partials/login_required.html",
        )
    try:
        attendance_record = await backend_client.get_attendance_record(
            token=token,
            record_id=record_id,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return templates.TemplateResponse(
                request=request,
                name="partials/login_required.html",
            )
        if exc.response.status_code == 404:
            return templates.TemplateResponse(
                request=request,
                name="partials/attendance_record_not_found.html",
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/attendance_error.html",
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="partials/backend_unavailable.html",
        )
    return templates.TemplateResponse(
        request=request,
        name="pages/attendance_record.html",
        context={"attendance_record": attendance_record},
    )
