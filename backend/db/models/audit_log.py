"""SQLModel: audit_log — append-only audit trail.

Row-level retention guarantees enforced via DB triggers/policies (see migration).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"

    id: str = Field(primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    actor_id: str = Field(index=True)
    actor_type: str
    action: str = Field(index=True)
    target_id: str | None = Field(default=None, index=True)
    metadata_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    ip_address: str | None = None
    user_agent: str | None = None
