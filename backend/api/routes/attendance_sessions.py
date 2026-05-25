"""Attendance session routes."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_user
from backend.api.schemas import AttendanceSession, AttendanceSessionCreate
from backend.audit.logger import log
from backend.db.models.attendance_session import AttendanceSessionRow
from backend.db.models.user import User
from backend.db.session import get_session

router = APIRouter()


def _serialize_session(row: AttendanceSessionRow) -> AttendanceSession:
    return AttendanceSession(
        session_id=row.id,
        name=row.name,
        status=row.status,
        starts_at=row.starts_at,
        created_at=row.created_at,
    )


@router.get("/", response_model=list[AttendanceSession])
async def list_attendance_sessions(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[AttendanceSession]:
    result = await session.execute(
        select(AttendanceSessionRow)
        .where(AttendanceSessionRow.user_id == user.id)
        .order_by(AttendanceSessionRow.created_at.desc())
    )
    return [_serialize_session(row) for row in result.scalars().all()]


@router.post("/", response_model=AttendanceSession, status_code=201)
async def create_attendance_session(
    payload: AttendanceSessionCreate,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AttendanceSession:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="session_name_required")

    row = AttendanceSessionRow(
        id=str(uuid.uuid4()),
        user_id=user.id,
        name=name,
        starts_at=payload.starts_at,
    )
    session.add(row)
    try:
        await session.commit()
        await session.refresh(row)
    except Exception as exc:
        await session.rollback()
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="attendance_session.persist_error",
            target_id=user.id,
        )
        raise HTTPException(500, "attendance_session_persist_error") from exc

    await log(
        actor_id=user.id,
        actor_type=user.role,
        action="attendance_session.created",
        target_id=row.id,
    )
    return _serialize_session(row)
