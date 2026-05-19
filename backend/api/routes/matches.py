"""GET /v1/matches — list and read matches the user accumulated."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_user
from backend.api.schemas import Match
from backend.db.models.match import MatchRow
from backend.db.models.user import User
from backend.db.session import get_session

router = APIRouter()


@router.get("/", response_model=list[Match])
async def list_matches(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    since: datetime | None = None,
) -> list[Match]:
    statement = select(MatchRow).where(MatchRow.user_id == user.id)
    if since is not None:
        statement = statement.where(MatchRow.created_at >= since)
    statement = statement.order_by(MatchRow.created_at.desc())
    result = await session.execute(statement)
    matches = result.scalars().all()
    return [
        Match(
            match_id=match.id,
            source_url=match.source_url,
            source_page=match.source_page,
            score=match.score,
            crawled_at=match.crawled_at,
            created_at=match.created_at,
        )
        for match in matches
    ]


@router.get("/{match_id}", response_model=Match)
async def get_match(
    match_id: str,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Match:
    result = await session.execute(
        select(MatchRow).where(
            MatchRow.id == match_id,
            MatchRow.user_id == user.id,
        )
    )
    match = result.scalar_one_or_none()
    if match is None:
        raise HTTPException(status_code=404, detail="match_not_found")
    return Match(
        match_id=match.id,
        source_url=match.source_url,
        source_page=match.source_page,
        score=match.score,
        crawled_at=match.crawled_at,
        created_at=match.created_at,
    )
