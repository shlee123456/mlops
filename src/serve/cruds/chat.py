"""
Chat CRUD Operations

대화 관련 데이터베이스 CRUD 함수
"""

from typing import Optional, Sequence

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.serve.models.chat import Conversation, ChatMessage, LLMConfig


# ============================================================
# Conversation CRUD
# ============================================================

async def create_conversation(
    db: AsyncSession,
    title: Optional[str] = None,
    llm_config_id: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Conversation:
    """대화 세션 생성"""
    conversation = Conversation(
        title=title,
        llm_config_id=llm_config_id,
        session_id=session_id,
    )
    db.add(conversation)
    await db.flush()
    await db.refresh(conversation)
    return conversation


async def get_conversation(
    db: AsyncSession,
    conversation_id: int,
    include_messages: bool = False,
) -> Optional[Conversation]:
    """대화 세션 조회"""
    query = select(Conversation).where(Conversation.id == conversation_id)
    
    if include_messages:
        query = query.options(selectinload(Conversation.messages))
    
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_conversations(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 20,
    session_id: Optional[str] = None,
) -> Sequence[Conversation]:
    """대화 세션 목록 조회"""
    query = select(Conversation).order_by(desc(Conversation.updated_at))
    
    if session_id:
        query = query.where(Conversation.session_id == session_id)
    
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


async def delete_conversation(
    db: AsyncSession,
    conversation_id: int,
) -> bool:
    """대화 세션 삭제"""
    conversation = await get_conversation(db, conversation_id)
    if conversation:
        await db.delete(conversation)
        return True
    return False


# ============================================================
# ChatMessage CRUD
# ============================================================

async def create_message(
    db: AsyncSession,
    conversation_id: int,
    role: str,
    content: str,
    model: Optional[str] = None,
    tokens_used: Optional[int] = None,
    latency_ms: Optional[int] = None,
    extra_data: Optional[dict] = None,
) -> ChatMessage:
    """채팅 메시지 생성"""
    message = ChatMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        model=model,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
        extra_data=extra_data,
    )
    db.add(message)
    await db.flush()
    await db.refresh(message)
    
    # 대화 updated_at 갱신
    conversation = await get_conversation(db, conversation_id)
    if conversation:
        from datetime import datetime
        conversation.updated_at = datetime.utcnow()
    
    return message


async def get_messages(
    db: AsyncSession,
    conversation_id: int,
    skip: int = 0,
    limit: int = 100,
) -> Sequence[ChatMessage]:
    """대화의 메시지 목록 조회"""
    query = (
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at)
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(query)
    return result.scalars().all()


# ============================================================
# LLMConfig CRUD
# ============================================================

async def create_llm_config(
    db: AsyncSession,
    name: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    is_default: bool = False,
) -> LLMConfig:
    """LLM 설정 생성"""
    # 기본값 설정 시 기존 기본값 해제
    if is_default:
        await _clear_default_llm_config(db)
    
    config = LLMConfig(
        name=name,
        model_name=model_name,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        is_default=is_default,
    )
    db.add(config)
    await db.flush()
    await db.refresh(config)
    return config


async def get_llm_config(
    db: AsyncSession,
    config_id: int,
) -> Optional[LLMConfig]:
    """LLM 설정 조회"""
    result = await db.execute(
        select(LLMConfig).where(LLMConfig.id == config_id)
    )
    return result.scalar_one_or_none()


async def get_llm_configs(
    db: AsyncSession,
) -> Sequence[LLMConfig]:
    """LLM 설정 목록 조회"""
    result = await db.execute(
        select(LLMConfig).order_by(LLMConfig.name)
    )
    return result.scalars().all()


async def get_default_llm_config(
    db: AsyncSession,
) -> Optional[LLMConfig]:
    """기본 LLM 설정 조회"""
    result = await db.execute(
        select(LLMConfig).where(LLMConfig.is_default == True)  # noqa: E712
    )
    return result.scalar_one_or_none()


async def update_llm_config(
    db: AsyncSession,
    config_id: int,
    **kwargs,
) -> Optional[LLMConfig]:
    """LLM 설정 부분 업데이트"""
    config = await get_llm_config(db, config_id)
    if not config:
        return None

    # is_default=True 설정 시 기존 기본값 해제
    if kwargs.get("is_default") is True:
        await _clear_default_llm_config(db)

    for key, value in kwargs.items():
        if value is not None:
            setattr(config, key, value)

    await db.flush()
    await db.refresh(config)
    return config


async def delete_llm_config(
    db: AsyncSession,
    config_id: int,
) -> bool:
    """LLM 설정 삭제 (참조 Conversation의 llm_config_id NULL 처리)"""
    config = await get_llm_config(db, config_id)
    if not config:
        return False

    # 참조하는 Conversation의 llm_config_id를 NULL로 설정
    result = await db.execute(
        select(Conversation).where(Conversation.llm_config_id == config_id)
    )
    for conversation in result.scalars().all():
        conversation.llm_config_id = None

    await db.delete(config)
    return True


async def _clear_default_llm_config(db: AsyncSession) -> None:
    """기존 기본값 해제"""
    result = await db.execute(
        select(LLMConfig).where(LLMConfig.is_default == True)  # noqa: E712
    )
    for config in result.scalars().all():
        config.is_default = False
