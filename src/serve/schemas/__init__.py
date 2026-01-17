"""
Pydantic Schemas

API 요청/응답 스키마
"""

from src.serve.schemas.chat import (
    # Message
    MessageCreate,
    MessageResponse,
    # Conversation
    ConversationCreate,
    ConversationResponse,
    ConversationListResponse,
    # Chat Completion
    ChatCompletionRequest,
    ChatCompletionResponse,
    # LLM Config
    LLMConfigCreate,
    LLMConfigResponse,
    # Health
    HealthResponse,
)

__all__ = [
    "MessageCreate",
    "MessageResponse",
    "ConversationCreate",
    "ConversationResponse",
    "ConversationListResponse",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "LLMConfigCreate",
    "LLMConfigResponse",
    "HealthResponse",
]
