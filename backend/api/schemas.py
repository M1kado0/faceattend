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


# --- Face registration / Attendance ---


class FaceRegistrationResponse(BaseModel):
    registration_id: str
    embedding_model_version: str


class AttendanceRecord(BaseModel):
    record_id: str
    face_registration_id: str
    session_id: str | None = None
    score: float
    checked_in_at: datetime
    created_at: datetime


class CheckInResponse(BaseModel):
    query_id: str
    attendance_records: list[AttendanceRecord]


# --- Attendance review ---


class AttendanceReviewRequest(BaseModel):
    attendance_record_id: str
    reason: Literal["manual_review", "false_match", "liveness_failure"]


class AttendanceReviewOut(BaseModel):
    id: str
    attendance_record_id: str
    status: Literal["pending", "approved", "rejected"]
    created_at: datetime
    updated_at: datetime


# --- Problem details (RFC 7807) ---


class Problem(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
