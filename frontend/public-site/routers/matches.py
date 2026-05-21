"""/matches — match dashboard + detail."""

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


@router.get("/matches", response_class=HTMLResponse)
async def matches_list(request: Request):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return RedirectResponse("/login", status_code=303)
    try:
        enrollments = await backend_client.list_enrollments(token=token)
        matches = await backend_client.list_matches(token=token)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return templates.TemplateResponse(
                request=request,
                name="partials/login_required.html",
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/matches_error.html",
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="partials/backend_unavailable.html",
        )
    return templates.TemplateResponse(
        request=request,
        name="pages/matches.html",
        context={"matches": matches, "has_enrollments": bool(enrollments)},
    )


@router.get("/matches/{match_id}", response_class=HTMLResponse)
async def match_detail(request: Request, match_id: str):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return templates.TemplateResponse(
            request=request,
            name="partials/login_required.html",
        )
    try:
        match = await backend_client.get_match(token=token, match_id=match_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return templates.TemplateResponse(
                request=request,
                name="partials/login_required.html",
            )
        if exc.response.status_code == 404:
            return templates.TemplateResponse(
                request=request,
                name="partials/match_not_found.html",
            )
        return templates.TemplateResponse(
            request=request,
            name="partials/matches_error.html",
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="partials/backend_unavailable.html",
        )
    return templates.TemplateResponse(
        request=request,
        name="pages/match_detail.html",
        context={"match": match},
    )
