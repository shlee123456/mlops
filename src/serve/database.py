"""
Database Configuration - SQLAlchemy 2.0 Async

비동기 SQLAlchemy 엔진 및 세션 설정
+ SQLAdmin용 동기 엔진 추가
"""

import os
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

# 데이터베이스 URL (기본: SQLite)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./mlops_chat.db"
)


def get_sync_database_url() -> str:
    """
    비동기 URL을 동기 URL로 변환 (SQLAdmin용)

    SQLAdmin은 동기 SQLAlchemy 엔진이 필요함
    """
    url = DATABASE_URL
    if url.startswith("sqlite+aiosqlite"):
        return url.replace("sqlite+aiosqlite", "sqlite")
    elif url.startswith("postgresql+asyncpg"):
        return url.replace("postgresql+asyncpg", "postgresql")
    elif url.startswith("mysql+aiomysql"):
        return url.replace("mysql+aiomysql", "mysql+pymysql")
    return url


# 비동기 엔진 생성
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
    future=True,
)

# 동기 엔진 생성 (SQLAdmin용)
sync_engine = create_engine(
    get_sync_database_url(),
    echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
)

# 세션 팩토리
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """SQLAlchemy ORM Base 클래스"""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Dependency Injection용 DB 세션
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """데이터베이스 테이블 생성 (개발용)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """데이터베이스 연결 종료"""
    await engine.dispose()
