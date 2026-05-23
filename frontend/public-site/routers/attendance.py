"""/attendance — attendance result dashboard + detail."""

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
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
        context={"has_registration": bool(face_registrations)},
    )


@router.get("/attendance", response_class=HTMLResponse)
async def attendance_list(request: Request):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return RedirectResponse("/login", status_code=303)
    try:
        face_registrations = await backend_client.list_face_registrations(token=token)
        attendance_records = await backend_client.list_attendance_records(token=token)
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
            "has_face_registrations": bool(face_registrations),
        },
    )


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
