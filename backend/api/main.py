"""FastAPI entry point for the backend JSON API (port 8002)."""

from fastapi import FastAPI

from backend.api.routes import (
    auth,
    billing,
    enroll,
    matches,
    notifications,
    users,
)

app = FastAPI(
    title="FaceGuard Backend API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.include_router(auth.router, prefix="/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/v1/users", tags=["users"])
app.include_router(enroll.router, prefix="/v1", tags=["enroll"])
app.include_router(matches.router, prefix="/v1/matches", tags=["matches"])
app.include_router(notifications.router, prefix="/v1/notifications", tags=["notifications"])
app.include_router(billing.router, prefix="/v1/webhooks", tags=["billing"])


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
