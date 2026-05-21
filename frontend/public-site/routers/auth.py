"""Public site auth routes — /login, /register, /verify."""

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from services.api_client import BackendClient

load_dotenv()

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8002")
SESSION_COOKIE_NAME = "session_token"
SESSION_MAX_AGE_SECONDS = 15 * 60

ROOT = Path(__file__).resolve().parents[1]

templates = Jinja2Templates(
    directory=ROOT / "templates",
)

backend_client = BackendClient(BACKEND_API_URL)
router = APIRouter()


def _redirect_with_session_token(url: str, token: str) -> RedirectResponse:
    response = RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        max_age=SESSION_MAX_AGE_SECONDS,
        samesite="lax",
    )
    return response


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="pages/login.html",
    )


@router.post("/login", response_class=HTMLResponse)
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    try:
        token_pair = await backend_client.login(email=email, password=password)
    except httpx.HTTPStatusError:
        return templates.TemplateResponse(
            request=request,
            name="pages/login.html",
            context={"error": "Invalid email or password."},
            status_code=401,
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="pages/login.html",
            context={"error": "Backend service is unavailable."},
            status_code=503,
        )

    return _redirect_with_session_token("/enroll", token_pair["access_token"])


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="pages/register.html",
    )


@router.post("/register", response_class=HTMLResponse)
async def register(request: Request, email: str = Form(...), password: str = Form(...)):
    try:
        token_pair = await backend_client.register(email=email, password=password)
    except httpx.HTTPStatusError as exc:
        message = "Could not create account."
        if exc.response.status_code == 409:
            message = "That email is already registered."
        return templates.TemplateResponse(
            request=request,
            name="pages/register.html",
            context={"error": message},
            status_code=exc.response.status_code,
        )
    except httpx.RequestError:
        return templates.TemplateResponse(
            request=request,
            name="pages/register.html",
            context={"error": "Backend service is unavailable."},
            status_code=503,
        )

    return _redirect_with_session_token("/enroll", token_pair["access_token"])


@router.post("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response

@router.get("/verify", response_class=HTMLResponse)
async def verify(request: Request, token: str):
    raise NotImplementedError
