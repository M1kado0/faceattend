"""/matches — match dashboard + detail."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT.parent / "shared"

templates = Jinja2Templates(
    directory=[ROOT / "templates", SHARED / "templates"],
)


@router.get("/matches", response_class=HTMLResponse)
async def matches_list(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="pages/matches.html",
    )


@router.get("/matches/{match_id}", response_class=HTMLResponse)
async def match_detail(request: Request, match_id: str):
    return templates.TemplateResponse(
        request=request,
        name="pages/match_detail.html",
        context={"match_id": match_id},
    )
