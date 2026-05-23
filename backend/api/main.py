"""FastAPI entry point for the backend JSON API (port 8002)."""

from fastapi import FastAPI

from backend.api.routes import (
    attendance_records,
    auth,
    billing,
    check_ins,
    face_registrations,
    notifications,
    users,
)

app = FastAPI(
    title="FaceAttend Backend API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.include_router(auth.router, prefix="/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/v1/users", tags=["users"])
app.include_router(
    face_registrations.router,
    prefix="/v1/face-registrations",
    tags=["face-registrations"],
)
app.include_router(
    attendance_records.router,
    prefix="/v1/attendance-records",
    tags=["attendance-records"],
)
app.include_router(check_ins.router, prefix="/v1", tags=["check-ins"])
app.include_router(notifications.router, prefix="/v1/notifications", tags=["notifications"])
app.include_router(billing.router, prefix="/v1/webhooks", tags=["billing"])


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
