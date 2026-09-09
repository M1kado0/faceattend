"""SQLModel: attendance_sessions — explicit sessions users can check into."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class AttendanceSessionRow(SQLModel, table=True):
    __tablename__ = "attendance_sessions"

    id: str = Field(primary_key=True)
    user_id: str = Field(index=True, foreign_key="users.id")
    name: str = Field(index=True)
    status: str = Field(default="open", index=True)
    starts_at: datetime | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
