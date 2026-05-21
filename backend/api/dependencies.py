"""Shared FastAPI dependencies (current user, DB session, etc.)."""

from __future__ import annotations

import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models.user import User
from backend.db.session import get_session

load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET", "secret")


async def get_current_user(
    session: Annotated[AsyncSession, Depends(get_session)],
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    """Resolve the current user from the Authorization: Bearer <jwt> header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="not_authenticated")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="invalid_token")

    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="invalid_token") from exc

    user_id = decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="invalid_token")

    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    return user
