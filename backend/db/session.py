"""Async SQLAlchemy engine + session factory."""
import os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from dotenv import load_dotenv

load_dotenv()

DATABASE_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql+asyncpg://user:pass@localhost:5432/faceguard",
)

engine = create_async_engine(DATABASE_URI, echo=False, future=True)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
