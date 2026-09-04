"""SQLModel: face_registrations — user to face embedding ID link."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class FaceRegistration(SQLModel, table=True):
    __tablename__ = "face_registrations"

    id: str = Field(primary_key=True)
    user_id: str = Field(index=True, foreign_key="users.id")
    embedding_id: str = Field(index=True)
    embedding_model_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Raw face crops are NEVER stored — only the embedding (in vector DB) and metadata.
