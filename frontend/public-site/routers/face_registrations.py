"""/face-registration — register a consenting face for attendance."""

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from services.api_client import BackendClient

load_dotenv()

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8002")
SESSION_COOKIE_NAME = "session_token"
MAX_ACTIVE_FACE_REGISTRATIONS = 3

backend_client = BackendClient(BACKEND_API_URL)

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]

templates = Jinja2Templates(
    directory=ROOT / "templates",
)


@router.get("/face-registration", response_class=HTMLResponse)
async def face_registration_page(request: Request):
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
            name="partials/face_registration_error.html",
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="partials/backend_unavailable.html",
        )

    return templates.TemplateResponse(
        request=request,
        name="pages/face_registration.html",
        context={
            "face_registrations": face_registrations,
            "max_active_face_registrations": MAX_ACTIVE_FACE_REGISTRATIONS,
        },
    )


@router.post("/face-registrations", response_class=HTMLResponse)
async def create_face_registration(request: Request, liveness_video: UploadFile):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return templates.TemplateResponse(
            request=request,
            name="partials/login_required.html",
        )

    liveness_video_bytes = await liveness_video.read()

    try:
        await backend_client.create_face_registration(
            liveness_video=liveness_video_bytes,
            token=token,
            liveness_video_filename=liveness_video.filename or "liveness.webm",
            liveness_video_content_type=(liveness_video.content_type or "application/octet-stream"),
        )
        return Response(
            status_code=204,
            headers={"HX-Redirect": "/check-in"},
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
        if exc.response.status_code == 409:
            return templates.TemplateResponse(
                request=request,
                name="partials/face_registration_limit_reached.html",
            )
        if exc.response.status_code == 401:
            return templates.TemplateResponse(
                request=request,
                name="partials/login_required.html",
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/face_registration_error.html",
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="partials/backend_unavailable.html",
        )


@router.post("/face-registrations/{registration_id}/delete", response_class=HTMLResponse)
async def delete_face_registration(request: Request, registration_id: str):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return templates.TemplateResponse(
            request=request,
            name="partials/login_required.html",
        )

    try:
        await backend_client.delete_face_registration(
            token=token,
            registration_id=registration_id,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return templates.TemplateResponse(
                request=request,
                name="partials/login_required.html",
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/face_registration_error.html",
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="partials/backend_unavailable.html",
        )

    return Response(
        status_code=204,
        headers={"HX-Redirect": "/face-registration"},
    )
