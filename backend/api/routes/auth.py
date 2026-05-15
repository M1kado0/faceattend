"""POST /v1/auth/* — register, login, refresh, logout."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.session import get_session
from passlib.hash import pbkdf2_sha256
import uuid
from jose import jwt
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone
from sqlalchemy import select

load_dotenv()
secret = os.getenv("JWT_SECRET", "secret")

from backend.api.schemas import LoginRequest, RegisterRequest, TokenPair
from backend.db.models.user import User

router = APIRouter()


@router.post("/register", response_model=TokenPair)
async def register(payload: RegisterRequest, session: AsyncSession = Depends(get_session)) -> TokenPair:
    email = payload.email
    password = payload.password
    hash = pbkdf2_sha256.hash(password)
    user_id = str(uuid.uuid4())
    user = User(id=user_id, email=email, hashed_password=hash)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    token = jwt.encode({"sub": user_id, "role": user.role, "exp": datetime.now(timezone.utc) + timedelta(minutes=15)}, secret, algorithm="HS256")
    return TokenPair(access_token=token, refresh_token=token, token_type="bearer")


@router.post("/login", response_model=TokenPair)
async def login(payload: LoginRequest, session: AsyncSession = Depends(get_session)) -> TokenPair:
    email = payload.email
    password = payload.password
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    if not pbkdf2_sha256.verify(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    token = jwt.encode({"sub": user.id, "role": user.role, "exp": datetime.now(timezone.utc) + timedelta(minutes=15)}, secret, algorithm="HS256")
    return TokenPair(access_token=token, refresh_token=token, token_type="bearer")




@router.post("/refresh", response_model=TokenPair)
async def refresh(refresh_token: str) -> TokenPair:
    raise NotImplementedError


@router.post("/logout")
async def logout() -> dict[str, str]:
    raise NotImplementedError
