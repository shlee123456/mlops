"""
Admin Views - SQLAdmin 모델 뷰 정의
"""

from sqladmin import ModelView
from src.serve.models.chat import LLMConfig, Conversation, ChatMessage


class LLMConfigAdmin(ModelView, model=LLMConfig):
    name = "LLM 설정"
    name_plural = "LLM 설정 목록"
    icon = "fa-solid fa-sliders"
    column_list = [LLMConfig.id, LLMConfig.name, LLMConfig.model_name, LLMConfig.temperature, LLMConfig.is_default, LLMConfig.created_at]
    column_searchable_list = [LLMConfig.name, LLMConfig.model_name]
    column_default_sort = ("created_at", True)


class ConversationAdmin(ModelView, model=Conversation):
    name = "대화"
    name_plural = "대화 목록"
    icon = "fa-solid fa-comments"
    column_list = [Conversation.id, Conversation.title, Conversation.session_id, Conversation.created_at, Conversation.updated_at]
    column_searchable_list = [Conversation.title, Conversation.session_id]
    column_default_sort = ("updated_at", True)
    can_delete = True


class ChatMessageAdmin(ModelView, model=ChatMessage):
    name = "메시지"
    name_plural = "메시지 목록"
    icon = "fa-solid fa-message"
    column_list = [ChatMessage.id, ChatMessage.conversation_id, ChatMessage.role, ChatMessage.content, ChatMessage.created_at]
    column_searchable_list = [ChatMessage.content, ChatMessage.role]
    column_default_sort = ("created_at", True)
    can_create = False
    can_edit = False
    can_delete = False
    column_formatters = {
        ChatMessage.content: lambda m, a: (m.content[:50] + "..." if m.content and len(m.content) > 50 else m.content),
    }
