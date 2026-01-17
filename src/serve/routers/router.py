"""
Router Aggregator

모든 라우터를 하나로 통합
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from starlette.responses import Response

from src.serve.core.config import settings
from src.serve.core.llm import LLMClient
from src.serve.core.metrics import get_metrics
from src.serve.routers.dependency import get_llm_client
from src.serve.routers.chat import router as chat_router
from src.serve.schemas.chat import HealthResponse

# 메인 라우터
router = APIRouter()

# 서브 라우터 포함
router.include_router(chat_router)


# ============================================================
# 공통 엔드포인트
# ============================================================

@router.get("/", tags=["Info"])
async def root():
    """루트 엔드포인트"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "chat": "/v1/chat/completions",
            "conversations": "/v1/conversations",
            "llm_configs": "/v1/llm-configs",
            "models": "/v1/models",
        },
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(
    llm: LLMClient = Depends(get_llm_client),
):
    """헬스 체크"""
    vllm_connected = await llm.health_check()
    models = await llm.list_models() if vllm_connected else []
    
    return HealthResponse(
        status="healthy" if vllm_connected else "degraded",
        vllm_connected=vllm_connected,
        available_models=models,
        database_connected=True,  # DB 연결 체크 추가 가능
        timestamp=datetime.utcnow(),
    )


@router.get("/v1/models", tags=["Models"])
async def list_models(
    llm: LLMClient = Depends(get_llm_client),
):
    """사용 가능한 모델 목록 (OpenAI 호환)"""
    models = await llm.list_models()
    return {
        "object": "list",
        "data": [{"id": model, "object": "model"} for model in models],
    }


@router.get("/metrics", tags=["Monitoring"], include_in_schema=False)
async def metrics() -> Response:
    """
    Prometheus 메트릭 엔드포인트
    
    Prometheus 서버가 스크랩하는 엔드포인트입니다.
    """
    return get_metrics()
