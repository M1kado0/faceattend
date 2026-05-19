"""Pydantic v2 request/response schemas for all API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

# --- Auth ---


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"


# --- Users ---


class UserOut(BaseModel):
    id: str
    email: EmailStr
    role: Literal["user", "moderator", "admin"]
    plan: Literal["free", "paid"]
    created_at: datetime


# --- Enroll / Search ---


class EnrollResponse(BaseModel):
    enrollment_id: str
    embedding_model_version: str


class Match(BaseModel):
    match_id: str
    source_url: str
    source_page: str
    score: float
    crawled_at: datetime
    created_at: datetime
    image_thumbnail_url: str | None = None


class SearchResponse(BaseModel):
    query_id: str
    matches: list[Match]


# --- Takedown ---


class TakedownRequest(BaseModel):
    match_id: str
    notice_type: Literal["dmca", "gdpr_erasure", "platform_tos"]


class TakedownOut(BaseModel):
    id: str
    match_id: str
    status: Literal["pending", "sent", "resolved", "rejected"]
    created_at: datetime
    updated_at: datetime


# --- Problem details (RFC 7807) ---


class Problem(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
