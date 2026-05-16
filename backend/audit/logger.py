"""Append-only audit log writer. EVERY biometric op MUST call `log()`.

Schema (see backend/CLAUDE.md):
  id, timestamp, actor_id, actor_type, action, target_id, metadata, ip_address, user_agent
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
import uuid

from backend.db.models.audit_log import AuditLog
from backend.db.session import async_session

ActorType = Literal["user", "admin", "system"]


# AUDIT: this is the sole writer; do not mark optional or skip on any biometric op.
async def log(
    *,
    actor_id: str,
    actor_type: ActorType,
    action: str,
    target_id: str | None = None,
    metadata: dict | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> None:
    entry = AuditLog(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        actor_id=actor_id,
        actor_type=actor_type,
        action=action,
        target_id=target_id,
        metadata_json=metadata or {},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    async with async_session() as session:
        session.add(entry)
        await session.commit()
