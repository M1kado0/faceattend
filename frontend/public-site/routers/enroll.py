"""/enroll — add a face to be monitored (liveness REQUIRED)."""
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from services.api_client import BackendClient

load_dotenv()

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8002")
SESSION_COOKIE_NAME = "session_token"

backend_client = BackendClient(BACKEND_API_URL)

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT.parent / "shared"

templates = Jinja2Templates(
    directory=[ROOT / "templates", SHARED / "templates"],
)


@router.get("/enroll", response_class=HTMLResponse)
async def enroll_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="pages/enroll.html",
    )


@router.post("/enroll", response_class=HTMLResponse)
async def enroll(request: Request, photo: UploadFile, liveness_blob: UploadFile):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return templates.TemplateResponse(
            request=request,
            name="partials/login_required.html",
        )

    photo_bytes = await photo.read()
    liveness_bytes = await liveness_blob.read()

    try:
        enrollment = await backend_client.enroll(
            photo=photo_bytes,
            liveness_blob=liveness_bytes,
            token=token,
            photo_filename=photo.filename or "photo.jpg",
            photo_content_type=photo.content_type or "application/octet-stream",
            liveness_filename=liveness_blob.filename or "liveness.jpg",
            liveness_content_type=liveness_blob.content_type or "application/octet-stream",
        )
        return templates.TemplateResponse(
            request=request,
            name="partials/enroll_success.html",
            context={"enrollment": enrollment},
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            try:
                data = exc.response.json()
            except ValueError:
                data = {}
            if data.get("detail") == "liveness_failed":
                return templates.TemplateResponse(
                    request=request,
                    name="partials/liveness_failed.html",
                )
        if exc.response.status_code == 401:
            return templates.TemplateResponse(
                request=request,
                name="partials/login_required.html",
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/enroll_error.html",
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="partials/backend_unavailable.html",
        )
