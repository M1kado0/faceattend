"""Typed view-models the web apps consume from the backend."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class User(BaseModel):
    id: str
    email: str
    role: Literal["user", "moderator", "admin"]
    plan: Literal["free", "paid"]
    token: str | None = None


class AttendanceRecord(BaseModel):
    record_id: str
    face_registration_id: str
    session_id: str | None = None
    score: float
    checked_in_at: datetime
    created_at: datetime
