"""/settings — account settings + attendance privacy controls."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]

templates = Jinja2Templates(
    directory=ROOT / "templates",
)


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="pages/settings.html",
    )


# GDPR: right to portability
@router.get("/settings/export", response_class=HTMLResponse)
async def export_data(request: Request):
    raise NotImplementedError


# GDPR: right to erasure
@router.get("/settings/delete", response_class=HTMLResponse)
async def delete_account_page(request: Request):
    raise NotImplementedError


@router.post("/settings/delete", response_class=HTMLResponse)
async def delete_account(request: Request):
    raise NotImplementedError
