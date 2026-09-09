"""SQLModel: attendance_records — liveness-gated attendance check-in results."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class AttendanceRecordRow(SQLModel, table=True):
    __tablename__ = "attendance_records"

    id: str = Field(primary_key=True)
    user_id: str = Field(index=True, foreign_key="users.id")
    face_registration_id: str = Field(index=True)
    session_id: str | None = Field(default=None, index=True)
    score: float
    checked_in_at: datetime
    notified_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
