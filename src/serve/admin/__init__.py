"""
Admin Module - SQLAdmin 관리자 인터페이스
"""

from sqladmin import Admin

from src.serve.database import sync_engine
from src.serve.admin.auth import AdminAuthBackend
from src.serve.admin.views import LLMConfigAdmin, ConversationAdmin, ChatMessageAdmin
from src.serve.core.config import settings


def create_admin(app) -> Admin:
    """SQLAdmin 인스턴스 생성"""
    authentication_backend = AdminAuthBackend(secret_key=settings.jwt_secret_key)
    admin = Admin(
        app,
        sync_engine,
        authentication_backend=authentication_backend,
        title="MLOps Admin",
        base_url="/admin",
    )
    admin.add_view(LLMConfigAdmin)
    admin.add_view(ConversationAdmin)
    admin.add_view(ChatMessageAdmin)
    return admin

__all__ = ["create_admin"]
