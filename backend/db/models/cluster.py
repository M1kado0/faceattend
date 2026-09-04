"""SQLModel: clusters — face cluster metadata."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class Cluster(SQLModel, table=True):
    __tablename__ = "clusters"

    id: str = Field(primary_key=True)
    centroid_image_id: str
    member_count: int = 0
    last_merged_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
