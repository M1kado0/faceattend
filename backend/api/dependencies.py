"""Shared FastAPI dependencies (current user, DB session, etc.)."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException
from jose import jwt, JWTError
import os
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.session import get_session
from sqlalchemy import select

from backend.db.models.user import User


load_dotenv()
secret = os.getenv("JWT_SECRET", "secret")

async def get_current_user(authorization: Annotated[str | None, Header()] = None, session: AsyncSession = Depends(get_session)):
    """Resolve the current user from the Authorization: Bearer <jwt> header.

    TODO: implement JWT decode + lookup against users table.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="not_authenticated")
    token = authorization.replace("Bearer ", "")
    try:
        decoded = jwt.decode(token, secret, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401, detail="invalid_token")
    user_id = decoded["sub"]
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    return user







async def require_admin(user=Depends(get_current_user)):
    """Gate admin-only routes."""
    if getattr(user, "role", None) not in {"admin", "moderator"}:
        raise HTTPException(status_code=403, detail="forbidden")
    return user
