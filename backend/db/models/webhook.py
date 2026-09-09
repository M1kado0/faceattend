"""SQLModel: webhooks — user-configured webhook endpoints."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class Webhook(SQLModel, table=True):
    __tablename__ = "webhooks"

    id: str = Field(primary_key=True)
    user_id: str = Field(index=True, foreign_key="users.id")
    url: str
    secret: str
    events: str  # comma-separated event names
    created_at: datetime = Field(default_factory=datetime.utcnow)
