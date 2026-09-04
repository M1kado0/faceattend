"""SQLModel: api_keys — programmatic access tokens."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class ApiKey(SQLModel, table=True):
    __tablename__ = "api_keys"

    id: str = Field(primary_key=True)
    user_id: str = Field(index=True, foreign_key="users.id")
    hashed_key: str = Field(unique=True)
    label: str
    rate_limit_per_min: int = 60
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
