from __future__ import annotations

import uuid
from datetime import datetime

import numpy as np
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.schemas import Match
from backend.audit.logger import log
from backend.db.models.match import MatchRow
from backend.db.models.user import User
from backend.indexer.store import get_store

index = get_store()


async def scan_and_persist_matches(
    *,
    user: User,
    embedding: np.ndarray,
    model_version: str,
    session: AsyncSession,
    top_k: int = 10,
) -> list[Match]:
    try:
        raw_results = await index.search(
            embedding=embedding,
            top_k=top_k,
            filter={"embedding_model_version": model_version},
        )
    except Exception as exc:
        await log(
            actor_id=user.id,
            actor_type=user.role,
            action="search.index_error",
            target_id=user.id,
        )
        raise HTTPException(500, "index_error") from exc

    results = []
    for result_match in raw_results:
        if result_match.metadata.get("user_id") == user.id:
            continue
        results.append(result_match)

    embedding_ids = [result_match.embedding_id for result_match in results]
    existing_matches_by_image_id = {}
    if embedding_ids:
        existing_result = await session.execute(
            select(MatchRow).where(
                MatchRow.user_id == user.id,
                MatchRow.image_id.in_(embedding_ids),
            )
        )
        existing_matches_by_image_id = {
            match.image_id: match for match in existing_result.scalars().all()
        }

    matches = []
    search_created_at = datetime.utcnow()
    has_new_matches = False
    for result_match in results:
        metadata = result_match.metadata
        crawled_at = metadata.get("crawled_at", search_created_at)

        match_row = existing_matches_by_image_id.get(result_match.embedding_id)
        if match_row is None:
            match_row = MatchRow(
                id=str(uuid.uuid4()),
                user_id=user.id,
                image_id=result_match.embedding_id,
                source_url=str(metadata.get("source_url", "")),
                source_page=str(metadata.get("source_page", "")),
                score=float(result_match.score),
                crawled_at=crawled_at,
                notified_at=None,
                created_at=search_created_at,
            )
            session.add(match_row)
            has_new_matches = True

        matches.append(
            Match(
                match_id=match_row.id,
                source_url=match_row.source_url,
                source_page=match_row.source_page,
                score=match_row.score,
                crawled_at=match_row.crawled_at,
                created_at=match_row.created_at,
                image_thumbnail_url=result_match.metadata.get("image_thumbnail_url"),
            )
        )
    if has_new_matches:
        try:
            await session.commit()
        except Exception as exc:
            await session.rollback()
            await log(
                actor_id=user.id,
                actor_type=user.role,
                action="search.match_persist_error",
                target_id=user.id,
            )
            raise HTTPException(500, "match_persist_error") from exc

    await log(actor_id=user.id, actor_type=user.role, action="search.success", target_id=user.id)
    return matches
