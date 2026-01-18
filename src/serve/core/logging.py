"""
FastAPI Structured Logging

structlog 기반 JSON 로깅 설정 및 미들웨어
"""

import logging
import logging.handlers
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.serve.core.config import settings


# Request ID Context Variable
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


# ============================================================
# Structlog Configuration
# ============================================================

def setup_logging(
    log_dir: str = "/logs",  # Docker 볼륨 매핑과 일치 (./logs/fastapi:/logs)
    json_format: bool = True,
) -> None:
    """
    애플리케이션 로깅 설정
    
    Args:
        log_dir: 로그 파일 디렉토리
        json_format: JSON 형식 출력 여부
    """
    # 로그 디렉토리 생성
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 로깅 레벨
    log_level = logging.DEBUG if settings.debug else logging.INFO
    
    # 공통 전처리 프로세서 (stdlib과 공유)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_request_id,
        _add_app_info,
    ]
    
    # structlog 설정 - stdlib 로거로 렌더링 위임
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if json_format:
        # JSON 출력용 포매터
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
            )
        )
    else:
        # 개발용 컬러 포매터
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(colors=True),
            )
        )
    
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 (JSON 형식으로 항상 기록)
    log_file = Path(log_dir) / "app.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
    )
    root_logger.addHandler(file_handler)
    
    # 서드파티 로거 레벨 조정
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _add_request_id(logger, method_name, event_dict):
    """Request ID 추가"""
    req_id = request_id_var.get()
    if req_id:
        event_dict["request_id"] = req_id
    return event_dict


def _add_app_info(logger, method_name, event_dict):
    """앱 정보 추가"""
    event_dict["service"] = "fastapi"
    event_dict["app_name"] = settings.app_name
    return event_dict


# ============================================================
# Logger Factory
# ============================================================

def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """
    structlog 로거 가져오기
    
    Args:
        name: 로거 이름
        
    Returns:
        structlog BoundLogger
    """
    return structlog.get_logger(name)


# ============================================================
# Request Logging Middleware
# ============================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """HTTP 요청/응답 로깅 미들웨어"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger("http")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Request ID 생성 또는 헤더에서 가져오기
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request_id_var.set(request_id)
        
        # 경로 필터 (metrics, health 제외)
        if request.url.path in ("/metrics", "/health"):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        
        start_time = time.time()
        
        # 요청 로깅
        self.logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params),
            client_ip=request.client.host if request.client else "unknown",
        )
        
        try:
            response = await call_next(request)
            
            # 응답 로깅
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
            
            # Response Header에 Request ID 추가
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise


