"""SQLModel: takedowns — takedown requests + status."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class Takedown(SQLModel, table=True):
    __tablename__ = "takedowns"

    id: str = Field(primary_key=True)
    user_id: str = Field(index=True, foreign_key="users.id")
    attendance_record_id: str = Field(index=True, foreign_key="attendance_records.id")
    notice_type: str
    status: str = "pending"
    platform_ref: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
