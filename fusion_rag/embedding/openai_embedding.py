"""OpenAI 兼容嵌入适配器。

覆盖 OpenAI / DeepSeek / 通义(dashscope 兼容) / 本地 vLLM、Ollama(/v1)
等所有暴露 ``POST /embeddings`` 的服务。传输层与 LLM 适配器一致：
优先 aiohttp，缺失则回退 urllib + 线程池。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from ..core.exceptions import EmbeddingError
from ..core.logging import get_logger
from ..core.quota import ConcurrencyQuota, get_quota_key
from .base import EmbeddingBase, EmbeddingResponse

logger = get_logger(__name__)

try:
    import aiohttp  # type: ignore

    _HAS_AIOHTTP = True
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore
    _HAS_AIOHTTP = False


class OpenAIEmbedding(EmbeddingBase):
    """OpenAI 兼容 ``/embeddings`` 适配器。"""

    name = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        api_key_env: str | None = None,
        dimensions: int = 1024,
        timeout: float = 30.0,
        max_concurrency: int = 8,
        tenant_max_concurrency: int | None = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimensions = dimensions
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self.api_key = api_key or (os.environ.get(api_key_env, "") if api_key_env else "")
        # ★ 两层并发配额：全局兜底 + 按租户/请求子额度（同 openai_compatible）。
        self._quota = ConcurrencyQuota(max_concurrency, tenant_max_concurrency)
        self._session: Any | None = None
        # ★ session lazy init 锁：同 openai_compatible，防首波并发下双建后
        #   其中一个 session 永远不能 close，泄露一个连接池。
        self._session_lock: Any | None = None

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/embeddings"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(embeddings=[], model=self.model)
        # 分批并发
        batches = [
            texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)
        ]
        results = await asyncio.gather(*(self._embed_batch(b) for b in batches))
        flat: list[list[float]] = []
        tokens = 0
        for vectors, tok in results:
            flat.extend(vectors)
            tokens += tok
        if self._dimensions and flat and len(flat[0]) != self._dimensions:
            # 以实际返回维度为准，纠正声明值
            self._dimensions = len(flat[0])
        return EmbeddingResponse(embeddings=flat, model=self.model, total_tokens=tokens)

    async def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], int]:
        payload = {"model": self.model, "input": texts}
        async with self._quota.acquire(get_quota_key()):
            data = await self._post(payload)
        try:
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"{self.model} 响应格式异常: {str(data)[:200]}") from exc
        tokens = int((data.get("usage", {}) or {}).get("total_tokens", 0))
        return vectors, tokens

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if _HAS_AIOHTTP:
            if self._session is None or self._session.closed:
                if self._session_lock is None:
                    self._session_lock = asyncio.Lock()
                async with self._session_lock:
                    if self._session is None or self._session.closed:
                        self._session = aiohttp.ClientSession(
                            timeout=aiohttp.ClientTimeout(total=self.timeout),
                        )
            try:
                async with self._session.post(
                    self._endpoint, json=payload, headers=self._headers(),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        raise EmbeddingError(f"{self.model} HTTP {resp.status}: {text[:300]}")
                    return json.loads(text)
            except asyncio.TimeoutError as exc:
                raise EmbeddingError(f"{self.model} 嵌入超时") from exc
            except aiohttp.ClientError as exc:
                raise EmbeddingError(f"{self.model} 网络错误: {exc}") from exc

        # urllib 兜底
        import urllib.error
        import urllib.request

        def _call() -> dict[str, Any]:
            req = urllib.request.Request(
                self._endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            return await asyncio.get_running_loop().run_in_executor(None, _call)
        except urllib.error.HTTPError as exc:  # pragma: no cover
            raise EmbeddingError(f"{self.model} HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover
            raise EmbeddingError(f"{self.model} 网络错误: {exc.reason}") from exc

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            # ★ Windows Proactor 下需 sleep 排干 connector，避免 transport.__del__
            #   对已关 loop 调 call_soon 报 Event loop is closed（同 LLM adapter）。
            await asyncio.sleep(0.25)
        self._session = None
