"""Attendance record routes."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_user
from backend.api.schemas import AttendanceRecord
from backend.db.models.attendance_record import AttendanceRecordRow
from backend.db.models.user import User
from backend.db.session import get_session

router = APIRouter()


@router.get("/", response_model=list[AttendanceRecord])
async def list_attendance_records(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    since: datetime | None = None,
    session_id: str | None = None,
) -> list[AttendanceRecord]:
    statement = select(AttendanceRecordRow).where(AttendanceRecordRow.user_id == user.id)
    if since is not None:
        statement = statement.where(AttendanceRecordRow.created_at >= since)
    if session_id:
        statement = statement.where(AttendanceRecordRow.session_id == session_id)
    statement = statement.order_by(
        AttendanceRecordRow.created_at.desc(),
        AttendanceRecordRow.score.desc(),
    )
    result = await session.execute(statement)
    attendance_records = result.scalars().all()
    return [
        AttendanceRecord(
            record_id=record.id,
            face_registration_id=record.face_registration_id,
            session_id=record.session_id,
            status="recorded",
            score=record.score,
            checked_in_at=record.checked_in_at,
            created_at=record.created_at,
        )
        for record in attendance_records
    ]


@router.get("/{record_id}", response_model=AttendanceRecord)
async def get_attendance_record(
    record_id: str,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AttendanceRecord:
    result = await session.execute(
        select(AttendanceRecordRow).where(
            AttendanceRecordRow.id == record_id,
            AttendanceRecordRow.user_id == user.id,
        )
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="attendance_record_not_found")
    return AttendanceRecord(
        record_id=record.id,
        face_registration_id=record.face_registration_id,
        session_id=record.session_id,
        status="recorded",
        score=record.score,
        checked_in_at=record.checked_in_at,
        created_at=record.created_at,
    )
