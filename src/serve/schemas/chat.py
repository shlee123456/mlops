"""
Chat Schemas

채팅 관련 Pydantic 스키마
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# Message Schemas
# ============================================================

class MessageCreate(BaseModel):
    """메시지 생성 요청"""
    role: str = Field(..., description="메시지 역할 (system/user/assistant)")
    content: str = Field(..., description="메시지 내용")


class MessageResponse(BaseModel):
    """메시지 응답"""
    id: int
    conversation_id: int
    role: str
    content: str
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ============================================================
# Conversation Schemas
# ============================================================

class ConversationCreate(BaseModel):
    """대화 생성 요청"""
    title: Optional[str] = Field(None, description="대화 제목")
    llm_config_id: Optional[int] = Field(None, description="LLM 설정 ID")
    session_id: Optional[str] = Field(None, description="세션 ID")


class ConversationResponse(BaseModel):
    """대화 응답"""
    id: int
    title: Optional[str]
    llm_config_id: Optional[int]
    session_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    """대화 목록 응답"""
    id: int
    title: Optional[str]
    llm_config_id: Optional[int]
    session_id: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============================================================
# Chat Completion Schemas (OpenAI 호환)
# ============================================================

class ChatCompletionRequest(BaseModel):
    """채팅 완성 요청"""
    messages: list[MessageCreate] = Field(..., description="대화 메시지 리스트")
    model: Optional[str] = Field(None, description="모델 이름")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="샘플링 온도")
    max_tokens: int = Field(512, ge=1, le=4096, description="최대 생성 토큰")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p 샘플링")
    stream: bool = Field(False, description="스트리밍 모드")
    
    # 대화 저장 옵션
    conversation_id: Optional[int] = Field(None, description="기존 대화 ID (연속 대화)")
    save_conversation: bool = Field(True, description="대화 저장 여부")


class UsageResponse(BaseModel):
    """토큰 사용량"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """채팅 완성 응답"""
    content: str = Field(..., description="생성된 텍스트")
    model: str = Field(..., description="사용된 모델")
    usage: UsageResponse = Field(default_factory=UsageResponse)
    finish_reason: Optional[str] = None
    conversation_id: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# LLM Config Schemas
# ============================================================

class LLMConfigCreate(BaseModel):
    """LLM 설정 생성 요청"""
    name: str = Field(..., description="설정 이름")
    model_name: str = Field(..., description="모델 이름")
    system_prompt: Optional[str] = Field(None, description="시스템 프롬프트")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=4096)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    is_default: bool = Field(False, description="기본값 여부")


class LLMConfigResponse(BaseModel):
    """LLM 설정 응답"""
    id: int
    name: str
    model_name: str
    system_prompt: Optional[str]
    temperature: float
    max_tokens: int
    top_p: float
    is_default: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ============================================================
# Health Schemas
# ============================================================

class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = Field(..., description="서비스 상태")
    vllm_connected: bool = Field(..., description="vLLM 연결 상태")
    available_models: list[str] = Field(default_factory=list)
    database_connected: bool = Field(True, description="DB 연결 상태")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
