from __future__ import annotations

import os
import uuid
from datetime import datetime

import numpy as np
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.schemas import AttendanceRecord
from backend.audit.logger import log
from backend.db.models.attendance_record import AttendanceRecordRow
from backend.db.models.user import User
from backend.indexer.store import get_store

index = get_store()
MIN_ATTENDANCE_MATCH_SCORE = float(os.getenv("ATTENDANCE_MATCH_THRESHOLD", "0.75"))


async def scan_and_persist_attendance_records(
    *,
    user: User,
    embedding: np.ndarray,
    model_version: str,
    session: AsyncSession,
    top_k: int = 10,
) -> list[AttendanceRecord]:
    try:
        raw_results = await index.search(
            embedding=embedding,
            top_k=top_k,
            filter={
                "embedding_model_version": model_version,
                "user_id": user.id,
            },
        )
    except Exception as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.index_error",
            target_id=user.id,
        )
        raise HTTPException(500, "index_error") from exc

    results = []
    for vector_result in raw_results:
        if vector_result.score < MIN_ATTENDANCE_MATCH_SCORE:
            continue
        results.append(vector_result)

    embedding_ids = [vector_result.embedding_id for vector_result in results]
    existing_records_by_registration_id = {}
    if embedding_ids:
        existing_result = await session.execute(
            select(AttendanceRecordRow).where(
                AttendanceRecordRow.user_id == user.id,
                AttendanceRecordRow.face_registration_id.in_(embedding_ids),
            )
        )
        existing_records_by_registration_id = {
            record.face_registration_id: record for record in existing_result.scalars().all()
        }

    attendance_records = []
    check_in_created_at = datetime.utcnow()
    has_new_records = False
    for vector_result in results:
        metadata = vector_result.metadata
        checked_in_at = metadata.get("checked_in_at", check_in_created_at)

        attendance_record = existing_records_by_registration_id.get(vector_result.embedding_id)
        if attendance_record is None:
            attendance_record = AttendanceRecordRow(
                id=str(uuid.uuid4()),
                user_id=user.id,
                face_registration_id=vector_result.embedding_id,
                session_id=metadata.get("session_id"),
                score=float(vector_result.score),
                checked_in_at=checked_in_at,
                notified_at=None,
                created_at=check_in_created_at,
            )
            session.add(attendance_record)
            has_new_records = True

        attendance_records.append(
            AttendanceRecord(
                record_id=attendance_record.id,
                face_registration_id=attendance_record.face_registration_id,
                session_id=attendance_record.session_id,
                score=attendance_record.score,
                checked_in_at=attendance_record.checked_in_at,
                created_at=attendance_record.created_at,
            )
        )
    if has_new_records:
        try:
            await session.commit()
        except Exception as exc:
            await session.rollback()
            await log(
                actor_id=user.id,
                actor_type=user.role,
                action="attendance_record.persist_error",
                target_id=user.id,
            )
            raise HTTPException(500, "attendance_record_persist_error") from exc

    await log(actor_id=user.id, actor_type=user.role, action="check_in.success", target_id=user.id)
    return attendance_records


async def scan_best_and_persist_attendance_record(
    *,
    user: User,
    embeddings: list[np.ndarray],
    model_version: str,
    session: AsyncSession,
    top_k: int = 10,
) -> list[AttendanceRecord]:
    """Search several live-frame embeddings and persist only the strongest match."""
    best_result = None
    for embedding in embeddings:
        try:
            raw_results = await index.search(
                embedding=embedding,
                top_k=top_k,
                filter={
                    "embedding_model_version": model_version,
                    "user_id": user.id,
                },
            )
        except Exception as exc:
            await log(
                actor_id=user.id,
                actor_type=user.role,
                action="check_in.index_error",
                target_id=user.id,
            )
            raise HTTPException(500, "index_error") from exc

        for vector_result in raw_results:
            if vector_result.score < MIN_ATTENDANCE_MATCH_SCORE:
                continue
            if best_result is None or vector_result.score > best_result.score:
                best_result = vector_result

    if best_result is None:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="check_in.success",
            target_id=user.id,
            metadata={"attendance_records": 0},
        )
        return []

    existing_result = await session.execute(
        select(AttendanceRecordRow).where(
            AttendanceRecordRow.user_id == user.id,
            AttendanceRecordRow.face_registration_id == best_result.embedding_id,
        )
    )
    existing_record = existing_result.scalar_one_or_none()

    check_in_created_at = datetime.utcnow()
    if existing_record is None:
        metadata = best_result.metadata
        attendance_record = AttendanceRecordRow(
            id=str(uuid.uuid4()),
            user_id=user.id,
            face_registration_id=best_result.embedding_id,
            session_id=metadata.get("session_id"),
            score=float(best_result.score),
            checked_in_at=metadata.get("checked_in_at", check_in_created_at),
            notified_at=None,
            created_at=check_in_created_at,
        )
        session.add(attendance_record)
        try:
            await session.commit()
        except Exception as exc:
            await session.rollback()
            await log(
                actor_id=user.id,
                actor_type=user.role,
                action="attendance_record.persist_error",
                target_id=user.id,
            )
            raise HTTPException(500, "attendance_record_persist_error") from exc
    else:
        attendance_record = existing_record

    await log(
        actor_id=user.id,
        actor_type=user.role,
        action="check_in.success",
        target_id=user.id,
        metadata={"best_score": float(best_result.score)},
    )
    return [
        AttendanceRecord(
            record_id=attendance_record.id,
            face_registration_id=attendance_record.face_registration_id,
            session_id=attendance_record.session_id,
            score=attendance_record.score,
            checked_in_at=attendance_record.checked_in_at,
            created_at=attendance_record.created_at,
        )
    ]
