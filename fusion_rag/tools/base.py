"""Tool 抽象、执行上下文与结果模型。

工具是「运行时可被大模型自主调用」的最小能力单元。每个工具声明自己的
JSON Schema（对齐 OpenAI function-calling），由 :class:`ToolRegistry` 汇总
后随请求下发给模型；模型返回 ``tool_calls``，:class:`ToolExecutor` 据此
调度到具体工具的 :meth:`Tool.run`。

设计要点：
- 文件 IO 属阻塞操作，工具内部一律经 :meth:`Tool._offload` 投递到 CPU
  线程池执行，避免阻塞事件循环，保障高并发。
- 工具不直接抛异常给上层：可预期的失败（越权、超时、文件不存在）以
  ``ToolResult(is_error=True)`` 返回，让模型能读到错误并自我修正。
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class ToolContext:
    """一次工具调用的作用域上下文（由 Orchestrator/ToolAgent 注入）。

    ``roots`` 是当前租户 + 知识库解析出的授权目录列表，工具的所有路径
    操作都被 jail 在这些根之内。
    """

    tenant_id: str = "default"
    kb_ids: list[str] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)
    trace_id: str = ""
    session_id: str = ""
    #: 会话内规划清单（PlanStore）；由 ToolAgent 在一次问答开始时注入，
    #: 未启用规划时保持 None，plan_update 工具据此判断可用性。
    plan_store: Any | None = None
    #: ★ fanout 并发子 Agent 专用“任务作用域”（非安全边界）：非空时，
    #:   grep/glob/find/read_file 只能访问文件名 basename 命中该集合的文件，
    #:   search_kb 会把它并入 doc_hints。PathJail 目录级沙箱本身不受影响。
    restrict_to_sources: Any | None = None


@dataclass
class ToolResult:
    """工具执行结果。``content`` 为回填给模型的文本（含错误说明）。"""

    content: str
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class Tool(abc.ABC):
    """只读文件工具抽象基类。"""

    #: 工具名（function-calling 中的 function.name）
    name: str = "tool"
    #: 面向模型的能力说明（写清楚何时用、返回什么，直接影响模型调用质量）
    description: str = ""
    #: JSON Schema（OpenAI function.parameters）
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    # --- 能力元数据（驱动“问题类型→工具候选集”筛选，不影响模型自主控制）---
    #: 擅长处理的问题类型（例："概念解释", "精确型号定位"）
    best_for: tuple[str, ...] = ()
    #: 不适合处理的问题类型（引导模型避开误用）
    avoid_for: tuple[str, ...] = ()
    #: 硬性适用意图域（intent.domain）；为空=全适用，非空=仅命中时下发
    applies_to_domains: tuple[str, ...] = ()

    def __init__(self, cpu_executor: Any | None = None) -> None:
        self._cpu = cpu_executor

    @abc.abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """执行工具。实现应把阻塞 IO 包进 :meth:`_offload`。"""

    def schema(self) -> dict[str, Any]:
        """生成 OpenAI function-calling 工具定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # ------------------------------------------------------------------
    async def _offload(self, func: Callable[[], T]) -> T:
        """把阻塞的文件 IO 投递到线程池；无 CPU 池时退化为默认执行器。"""
        if self._cpu is not None:
            return await self._cpu.run(func)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"
