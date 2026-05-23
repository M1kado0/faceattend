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
    source_url: str
    source_page: str
    score: float
    checked_in_at: datetime
    image_thumbnail_url: str | None = None
