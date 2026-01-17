"""
Dependencies

FastAPI 의존성 주입
"""

from typing import AsyncGenerator

from fastapi import Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.serve.database import async_session_maker
from src.serve.core.config import settings
from src.serve.core.llm import LLMClient


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    데이터베이스 세션 의존성
    
    Usage:
        @router.get("/items")
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


async def verify_api_key(
    x_api_key: str = Header(None, alias="X-API-Key"),
) -> bool:
    """
    API 키 검증 의존성
    
    ENABLE_AUTH=true인 경우에만 검증 수행
    
    Usage:
        @router.post("/chat")
        async def chat(authenticated: bool = Depends(verify_api_key)):
            ...
    """
    if not settings.enable_auth:
        return True
    
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "API-Key"},
        )
    
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "API-Key"},
        )
    
    return True


# LLM 클라이언트 싱글톤
_llm_client: LLMClient | None = None


async def get_llm_client() -> LLMClient:
    """
    LLM 클라이언트 의존성
    
    Usage:
        @router.post("/chat")
        async def chat(llm: LLMClient = Depends(get_llm_client)):
            response = await llm.chat_completion(messages)
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


async def close_llm_client() -> None:
    """LLM 클라이언트 종료 (앱 셧다운 시 호출)"""
    global _llm_client
    if _llm_client:
        await _llm_client.close()
        _llm_client = None
