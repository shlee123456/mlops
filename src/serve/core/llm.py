"""
LLM Client

vLLM 서버 클라이언트 래퍼
"""

import httpx
from typing import AsyncGenerator, Optional

from src.serve.core.config import settings


class LLMClient:
    """vLLM 서버 비동기 클라이언트"""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 획득"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client
    
    async def close(self):
        """클라이언트 종료"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> bool:
        """vLLM 서버 상태 확인"""
        try:
            client = await self._get_client()
            response = await client.get("/models")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[str]:
        """사용 가능한 모델 목록"""
        try:
            client = await self._get_client()
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception:
            return []
    
    async def chat_completion(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
    ) -> dict:
        """
        채팅 완성 요청
        
        Args:
            messages: [{"role": "user", "content": "Hello"}]
            model: 모델 이름 (없으면 기본값)
            temperature: 샘플링 온도
            max_tokens: 최대 토큰 수
            top_p: Top-p 샘플링
            stream: 스트리밍 모드
            
        Returns:
            {"content": str, "model": str, "usage": dict, ...}
        """
        client = await self._get_client()
        
        payload = {
            "model": model or settings.default_model or "default",
            "messages": messages,
            "temperature": temperature or settings.default_temperature,
            "max_tokens": max_tokens or settings.default_max_tokens,
            "top_p": top_p or settings.default_top_p,
            "stream": stream,
        }
        
        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # OpenAI 형식 응답 파싱
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            
            return {
                "content": message.get("content", ""),
                "model": data.get("model", payload["model"]),
                "usage": data.get("usage", {}),
                "finish_reason": choice.get("finish_reason"),
            }
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def chat_completion_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        채팅 완성 스트리밍
        
        Yields:
            SSE 형식 청크
        """
        client = await self._get_client()
        
        payload = {
            "model": model or settings.default_model or "default",
            "messages": messages,
            "temperature": temperature or settings.default_temperature,
            "max_tokens": max_tokens or settings.default_max_tokens,
            "top_p": top_p or settings.default_top_p,
            "stream": True,
        }
        
        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line[6:]  # "data: " 제거
        except Exception as e:
            yield f'{{"error": "{str(e)}"}}'
    
    async def completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        텍스트 완성 요청
        
        Args:
            prompt: 프롬프트 텍스트
            
        Returns:
            {"content": str, "model": str, "usage": dict}
        """
        client = await self._get_client()
        
        payload = {
            "model": model or settings.default_model or "default",
            "prompt": prompt,
            "temperature": temperature or settings.default_temperature,
            "max_tokens": max_tokens or settings.default_max_tokens,
        }
        
        try:
            response = await client.post("/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            
            choice = data.get("choices", [{}])[0]
            
            return {
                "content": choice.get("text", ""),
                "model": data.get("model", payload["model"]),
                "usage": data.get("usage", {}),
            }
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": str(e)}


# 전역 LLM 클라이언트 (선택적 사용)
llm_client: Optional[LLMClient] = None


async def get_llm_client() -> LLMClient:
    """LLM 클라이언트 의존성"""
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client


async def close_llm_client():
    """LLM 클라이언트 종료"""
    global llm_client
    if llm_client:
        await llm_client.close()
        llm_client = None
