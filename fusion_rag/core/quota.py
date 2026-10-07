"""按租户/请求的下游并发子配额（ConcurrencyQuota）。

背景：LLM 与 embedding 客户端各自持有一个"全局并发信号量"（如 chat 的
``max_concurrency=16``）。单个请求触发 fanout 时会把一次问答放大成 ≤N 次
模型调用，多用户并发下所有请求 + 所有子 Agent 抢同一个全局闸门，先到的
租户可能占满窗口把别的租户饿死（尾 TTFT 被放大到超时）。

这里提供两层配额治理：

- **全局闸门**：兜底保护下游模型服务不被打爆（等价旧行为）。
- **按 key 子闸门**：key 取自 :data:`quota_key_var`（默认租户 id，可配成
  会话 id = "按请求"），限制单个租户/请求最多同时占用多少全局槽位，从而
  在不同用户之间形成公平份额，并给 fanout 提供天然背压。

获取顺序固定为 **先子闸门后全局闸门**：等待全局槽位期间不占用别的租户的
子额度，避免"抱着全局槽位等子额度"造成的交叉等待与饿死。单线程事件循环下
``dict.get`` 与建 ``Semaphore`` 之间无 ``await``，天然原子，无需额外加锁。
"""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import asynccontextmanager
from typing import AsyncIterator

#: 当前请求的配额键（租户 id 或会话 id）。None = 不参与子配额，仅走全局闸门。
#: 由请求入口（orchestrator.ask）设置；fanout 并发子任务创建时自动继承，
#: 与 tool_agent 的请求起点 ContextVar 同属"跨异步上下文透传"用法。
quota_key_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "fusion_rag_quota_key", default=None,
)


def set_quota_key(value: str | None) -> None:
    """由请求入口设置配额键；空串归一为 None（关闭子配额）。"""
    quota_key_var.set(value or None)


def get_quota_key() -> str | None:
    """返回当前请求配额键；未设置时 None（仅全局闸门）。"""
    return quota_key_var.get()


class ConcurrencyQuota:
    """全局信号量 + 按 key 子信号量的两层并发配额。

    :param global_limit: 全局并发上限（等价旧的单信号量）。
    :param per_key_limit: 单个 key（租户/请求）并发上限；``<=0`` 或 ``>=``
        全局上限时视为关闭子配额（只剩全局闸门，行为与旧实现完全一致）。
    :param max_keys: 跟踪的 key 数量软上限。超出后新 key 回落"仅全局闸门"，
        防止租户近乎无限导致 ``dict`` 无界增长（已知限制：不主动淘汰）。
    """

    def __init__(
        self,
        global_limit: int,
        per_key_limit: int | None = None,
        *,
        max_keys: int = 4096,
    ) -> None:
        self._global_limit = max(1, int(global_limit))
        self._global = asyncio.Semaphore(self._global_limit)
        limit = int(per_key_limit) if per_key_limit else 0
        # 子额度只有在 (0, global) 区间才有意义，否则与全局闸门等价 → 关闭。
        self._per_key_limit = limit if 0 < limit < self._global_limit else 0
        self._max_keys = max(1, int(max_keys))
        self._per_key: dict[str, asyncio.Semaphore] = {}

    @property
    def enabled(self) -> bool:
        """子配额是否生效（否则等价于纯全局闸门）。"""
        return self._per_key_limit > 0

    @property
    def global_limit(self) -> int:
        return self._global_limit

    @property
    def per_key_limit(self) -> int:
        return self._per_key_limit

    @asynccontextmanager
    async def acquire(self, key: str | None = None) -> AsyncIterator[None]:
        """在配额内执行：先取 key 子额度（若生效且有 key），再取全局槽位。"""
        sem: asyncio.Semaphore | None = None
        if self._per_key_limit and key:
            sem = self._per_key.get(key)
            if sem is None and len(self._per_key) < self._max_keys:
                sem = asyncio.Semaphore(self._per_key_limit)
                self._per_key[key] = sem
        if sem is None:
            # 子配额关闭 / 无 key / key 数超限：仅走全局闸门（旧行为）。
            async with self._global:
                yield
            return
        async with sem:
            async with self._global:
                yield
