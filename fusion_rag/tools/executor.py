"""ToolExecutor：把模型的 tool_calls 调度到具体工具，套超时与埋点。

- **超时**：单次工具调用用 ``asyncio.wait_for`` 施加独立超时。刻意**不**走
  ``Sandbox.run_guarded``——那会再次占用全局并发信号量，而工具调用发生在
  已被 Orchestrator ``run_guarded`` 包裹的请求内，嵌套取槽会在满载时死锁。
- **异常隔离**：越权/超时/IO 失败都转成 ``ToolResult(is_error=True)``，
  回填给模型以便其自我修正，绝不让单个工具炸掉整个循环。
- **可观测**：每次调用埋一个 ``tool.<name>`` span，并上报调用数/错误数/时延。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from typing import Any, Iterator

from ..core.logging import get_logger
from ..llm.base import ToolCall
from .base import ToolContext, ToolResult
from .registry import ToolRegistry

logger = get_logger(__name__)


def _summarize_args(args: dict[str, Any], *, max_len: int = 80) -> dict[str, str]:
    """将工具入参截断为适合写入 trace 的短字符串，方便回放时定位提问。"""
    out: dict[str, str] = {}
    for k, v in (args or {}).items():
        s = str(v)
        out[k] = s if len(s) <= max_len else s[:max_len] + "…"
    return out


class ToolExecutor:
    """工具调度器。"""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        tracer: Any | None = None,
        metrics: Any | None = None,
        call_timeout: float = 15.0,
    ) -> None:
        self.registry = registry
        self.tracer = tracer
        self.metrics = metrics
        self.call_timeout = max(0.5, float(call_timeout))

    # ------------------------------------------------------------------
    async def dispatch(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """执行一次工具调用（永不抛异常，失败以 is_error 结果返回）。"""
        name = getattr(call, "name", None) or (call.get("name") if isinstance(call, dict) else "")
        args = getattr(call, "arguments", None)
        if args is None and isinstance(call, dict):
            args = call.get("arguments", {})
        args = args or {}

        self._count("tool_calls_total", name)
        tool = self.registry.get(name)
        if tool is None:
            self._count("tool_errors_total", name)
            return ToolResult(content=f"未知工具：{name}", is_error=True)

        started = time.monotonic()
        args_summary = _summarize_args(args)
        with self._span(name, tool=name, args_summary=args_summary) as node:
            try:
                result = await asyncio.wait_for(
                    tool.run(dict(args), ctx), timeout=self.call_timeout,
                )
            except asyncio.TimeoutError:
                self._count("tool_errors_total", name)
                if node is not None:
                    node.attributes["timeout"] = True
                    node.attributes["is_error"] = True
                logger.warning("工具 %s 执行超时（%ss）", name, self.call_timeout)
                return ToolResult(
                    content=f"工具 {name} 执行超时（>{self.call_timeout}s）", is_error=True,
                )
            except Exception as exc:  # noqa: BLE001 - 越权/IO 等一律隔离
                self._count("tool_errors_total", name)
                if node is not None:
                    node.attributes["error"] = type(exc).__name__
                    node.attributes["is_error"] = True
                logger.warning("工具 %s 执行失败：%s", name, exc)
                return ToolResult(
                    content=f"工具 {name} 执行失败：{type(exc).__name__}: {exc}",
                    is_error=True,
                )
            finally:
                self._observe("tool_latency_seconds", time.monotonic() - started, name)
            # 成功：写回属性（在 with 退出前，保证能持久化到 trace）
            if node is not None:
                node.attributes["is_error"] = result.is_error
                node.attributes["size"] = len(result.content)
                node.attributes["preview"] = result.content[:200]
        return result

    # ------------------------------------------------------------------
    @contextmanager
    def _span(self, name: str, **attributes: Any) -> Iterator[Any]:
        if self.tracer is None:
            yield None
            return
        with self.tracer.span(f"tool.{name}", **attributes) as node:
            yield node

    def _count(self, metric: str, tool: str) -> None:
        if self.metrics is not None:
            self.metrics.counter(metric).inc(1.0, tool=tool)

    def _observe(self, metric: str, value: float, tool: str) -> None:
        if self.metrics is not None:
            self.metrics.histogram(metric).observe(value, tool=tool)
