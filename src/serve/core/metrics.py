"""
Prometheus Metrics

FastAPI 메트릭 수집 및 노출
"""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from src.serve.core.config import settings


# ============================================================
# Metrics Definitions
# ============================================================

# 앱 정보
APP_INFO = Info(
    "fastapi_app",
    "FastAPI application information"
)
APP_INFO.info({
    "app_name": settings.app_name,
    "version": settings.app_version,
})

# HTTP 요청 카운터
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"]
)

# HTTP 요청 지속시간
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# 진행 중인 요청 수
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests in progress",
    ["method", "endpoint"]
)

# LLM 요청 메트릭
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total number of LLM completion requests",
    ["model", "status"]
)

LLM_REQUEST_DURATION_SECONDS = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

LLM_TOKENS_GENERATED = Counter(
    "llm_tokens_generated_total",
    "Total number of tokens generated",
    ["model"]
)

# 데이터베이스 메트릭
DB_CONNECTIONS_ACTIVE = Gauge(
    "db_connections_active",
    "Number of active database connections"
)

DB_QUERY_DURATION_SECONDS = Histogram(
    "db_query_duration_seconds",
    "Database query duration in seconds",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)


# ============================================================
# Prometheus Middleware
# ============================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """HTTP 요청 메트릭을 수집하는 미들웨어"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # /metrics 경로는 제외
        if request.url.path == "/metrics":
            return await call_next(request)
        
        method = request.method
        endpoint = self._get_endpoint(request)
        
        # 진행 중인 요청 증가
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            # 요청 완료 시간
            duration = time.time() - start_time
            
            # 메트릭 기록
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            # 진행 중인 요청 감소
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
        
        return response
    
    def _get_endpoint(self, request: Request) -> str:
        """엔드포인트 경로 추출 (경로 파라미터 정규화)"""
        path = request.url.path
        
        # 경로 파라미터 정규화 (예: /v1/conversations/123 → /v1/conversations/{id})
        parts = path.split("/")
        normalized = []
        for part in parts:
            if part.isdigit():
                normalized.append("{id}")
            else:
                normalized.append(part)
        
        return "/".join(normalized)


# ============================================================
# Metrics Endpoint
# ============================================================

def get_metrics() -> StarletteResponse:
    """Prometheus 메트릭 응답 생성"""
    return StarletteResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================
# Helper Functions
# ============================================================

def record_llm_request(model: str, duration: float, tokens: int, success: bool = True):
    """LLM 요청 메트릭 기록"""
    status = "success" if success else "error"
    LLM_REQUESTS_TOTAL.labels(model=model, status=status).inc()
    LLM_REQUEST_DURATION_SECONDS.labels(model=model).observe(duration)
    if tokens > 0:
        LLM_TOKENS_GENERATED.labels(model=model).inc(tokens)


def record_db_query(operation: str, duration: float):
    """DB 쿼리 메트릭 기록"""
    DB_QUERY_DURATION_SECONDS.labels(operation=operation).observe(duration)
