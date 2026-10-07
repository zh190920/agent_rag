"""进程内异步事件总线（模块解耦的核心）。

问答完成、校验失败、告警触发等均以事件广播，指标统计、问题沉淀、
告警器作为订阅者挂载 —— 新增下游能力无需改动业务代码（微内核思想）。
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .logging import get_logger

logger = get_logger(__name__)

EventHandler = Callable[["Event"], Awaitable[None]]


@dataclass
class Event:
    """一条领域事件。"""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class EventBus:
    """轻量异步 pub/sub。

    - ``publish`` 永不阻塞调用方：事件入队后立即返回，由后台
      dispatcher 协程逐个分发；处理器异常被隔离并记日志。
    - ``start/stop`` 由 Kernel 生命周期统一管理。
    """

    def __init__(self, max_queue: int = 10_000) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._wildcard: list[EventHandler] = []
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task[None] | None = None
        self._dropped = 0

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """订阅指定事件；``"*"`` 订阅全部。"""
        if event_name == "*":
            self._wildcard.append(handler)
        else:
            self._handlers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        bucket = self._wildcard if event_name == "*" else self._handlers.get(event_name, [])
        if handler in bucket:
            bucket.remove(handler)

    def publish(self, event: Event) -> bool:
        """非阻塞发布。队列满时丢弃并计数（背压保护）。"""
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning("事件总线队列已满，丢弃事件 %s（累计丢弃 %d）", event.name, self._dropped)
            return False

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self._dispatch_loop(), name="event-bus",
            )

    async def stop(self, drain_timeout: float = 5.0) -> None:
        if self._task is None:
            return
        await self.publish_and_wait(Event("__shutdown__"), drain_timeout)
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def publish_and_wait(self, event: Event, timeout: float = 5.0) -> None:
        """发布并等待队列排空（用于优雅关闭前 flush）。"""
        self.publish(event)
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

    @property
    def dropped_count(self) -> int:
        return self._dropped

    async def _dispatch_loop(self) -> None:
        while True:
            event = await self._queue.get()
            if event.name == "__shutdown__":
                self._queue.task_done()
                break
            handlers = list(self._handlers.get(event.name, [])) + list(self._wildcard)
            for handler in handlers:
                try:
                    await handler(event)
                except Exception:  # noqa: BLE001 —— 处理器故障隔离
                    logger.exception("事件处理器失败: event=%s handler=%s",
                                     event.name, getattr(handler, "__name__", handler))
            self._queue.task_done()
