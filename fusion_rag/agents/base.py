"""BaseAgent：多智能体统一基类 + 请求上下文。

所有角色 Agent 共享：模型路由、轨迹埋点、指标上报、租户上下文。
基类只提供「能力手柄」，各 Agent 暴露自己的领域方法（classify/retrieve/
reason/validate），由 Orchestrator 编排，避免为统一而牺牲清晰度。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..llm.router import LLMRouter
from ..observability.metrics import MetricsRegistry
from ..observability.tracer import Span, Tracer
from ..types import Priority


@dataclass
class AgentContext:
    """贯穿一次请求的上下文（租户/会话/知识库作用域/优先级）。"""

    tenant_id: str = "default"
    session_id: str = ""
    user_id: str = ""
    kb_ids: list[str] = field(default_factory=list)
    priority: Priority = Priority.NORMAL
    trace_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def scope_filter(self) -> dict[str, Any]:
        """检索作用域过滤器（tenant + kb），由检索层强制注入。"""
        flt: dict[str, Any] = {"tenant_id": self.tenant_id}
        if self.kb_ids:
            flt["kb_id"] = list(self.kb_ids)
        return flt


class BaseAgent:
    """角色 Agent 基类。"""

    name: str = "base"
    role: str = ""

    def __init__(
        self,
        llm: LLMRouter | None = None,
        tracer: Tracer | None = None,
        metrics: MetricsRegistry | None = None,
        provider: str | None = None,
    ) -> None:
        self.llm = llm
        self.tracer = tracer
        self.metrics = metrics
        self.provider = provider

    # ------------------------------------------------------------------
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        """轨迹埋点；无 tracer 时透明直通。"""
        if self.tracer is None:
            yield Span(span_id="-", name=f"{self.name}.{name}")
            return
        with self.tracer.span(f"{self.name}.{name}", **attributes) as node:
            yield node

    def count(self, metric: str, amount: float = 1.0, **labels: str) -> None:
        if self.metrics is not None:
            self.metrics.counter(metric).inc(amount, **labels)

    def observe(self, metric: str, value: float, **labels: str) -> None:
        if self.metrics is not None:
            self.metrics.histogram(metric).observe(value, **labels)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name} role={self.role!r}>"
