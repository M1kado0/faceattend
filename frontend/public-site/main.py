"""Public site (port 8000) — end-user-facing FastAPI + Jinja2 + HTMX app."""

# Imported via direct path because the dir uses hyphens (not a Python package).
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).parent))

from routers import auth, billing, enroll, matches, settings  # noqa: E402

app = FastAPI(title="FaceAttend — Public Site", version="0.1.0")

ROOT = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

templates = Jinja2Templates(
    directory=ROOT / "templates",
)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="pages/index.html",
    )

app.include_router(auth.router, tags=["auth"])
app.include_router(enroll.router, tags=["enroll"])
app.include_router(matches.router, tags=["matches"])
app.include_router(settings.router, tags=["settings"])
app.include_router(billing.router, tags=["billing"])
