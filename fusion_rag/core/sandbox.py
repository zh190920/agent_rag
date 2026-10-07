"""轻量化沙箱与过载保护（借鉴 DeepSeek-Harness 沙箱 + OpenClaw 本地管控）。

提供三类护栏，Orchestrator 在请求入口统一套用：

- :class:`CpuExecutor`   —— CPU 密集任务线程池隔离，防阻塞事件循环
- :class:`TokenBucket`   —— 每租户令牌桶限流
- :class:`Sandbox`       —— 全局并发闸门 + 请求级超时预算
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, TypeVar

from .exceptions import RateLimitError, TimeoutBudgetError
from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CpuExecutor:
    """CPU 密集任务（嵌入、BM25 打分、余弦相似度）线程池。"""

    def __init__(self, max_workers: int = 8) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="fusion-cpu",
        )

    async def run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        if kwargs:
            func_call = lambda: func(*args, **kwargs)  # noqa: E731
            return await loop.run_in_executor(self._pool, func_call)
        return await loop.run_in_executor(self._pool, func, *args)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


class TokenBucket:
    """令牌桶限流器（线程安全由 asyncio 单线程模型保证）。"""

    def __init__(self, rate: float, burst: int) -> None:
        self.rate = max(rate, 0.001)
        self.burst = max(burst, 1)
        self._tokens = float(self.burst)
        self._updated = time.monotonic()

    def try_acquire(self, tokens: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    async def acquire(self, tokens: float = 1.0, timeout: float = 10.0) -> None:
        """等待获取令牌；超时抛 :class:`RateLimitError`。"""
        deadline = time.monotonic() + timeout
        while True:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            if time.monotonic() >= deadline:
                raise RateLimitError("限流等待超时，请稍后重试")
            wait = min((tokens - self._tokens) / self.rate, 0.2)
            await asyncio.sleep(max(wait, 0.005))

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.burst, self._tokens + (now - self._updated) * self.rate)
        self._updated = now


class Sandbox:
    """请求级护栏：全局并发闸门 + 超时预算 + 每租户限流。"""

    def __init__(
        self,
        max_concurrent: int = 64,
        request_timeout: float = 120.0,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._request_timeout = request_timeout
        self._buckets: dict[str, TokenBucket] = {}
        self._inflight = 0

    def register_tenant(self, tenant_id: str, rate: float, burst: int) -> None:
        self._buckets[tenant_id] = TokenBucket(rate, burst)

    async def acquire_tenant(self, tenant_id: str, timeout: float = 10.0) -> None:
        bucket = self._buckets.get(tenant_id)
        if bucket is not None:
            await bucket.acquire(timeout=timeout)

    @property
    def inflight(self) -> int:
        return self._inflight

    async def run_guarded(
        self,
        coro_factory: Callable[[], Awaitable[T]],
        *,
        timeout: float | None = None,
    ) -> T:
        """在并发闸门与超时预算内执行协程。

        Args:
            coro_factory: 无参工厂（每次调用生成新协程，避免复用告警）。
            timeout: 覆盖默认请求超时。
        """
        async with self._semaphore:
            self._inflight += 1
            started = time.monotonic()
            try:
                return await asyncio.wait_for(
                    coro_factory(), timeout or self._request_timeout,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutBudgetError(
                    f"请求执行超过 {timeout or self._request_timeout}s 预算",
                ) from exc
            finally:
                self._inflight -= 1
                logger.debug("guarded 请求完成，耗时 %.3fs", time.monotonic() - started)
