"""Tracer：全链路轨迹记录与回放（借鉴 DeepSeek-Harness 轨迹回放）。

- 用 ``contextvars`` 在异步调用链中传播当前 Trace（跨 ``await``、跨
  ``asyncio.gather`` 子任务自动隔离）。
- ``span(name)`` 上下文管理器记录每个阶段（意图/拆解/检索/推理/校验/脱敏）
  的耗时、属性、状态与异常，形成父子 span 树。
- 结束时序列化 span 树，追加写入 JSONL 并落库，支持按 trace_id 回放。
"""

from __future__ import annotations

import contextvars
import json
import random
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..core.logging import get_logger, session_id_var, trace_id_var

logger = get_logger(__name__)

_current_trace: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar(
    "fusion_current_trace", default=None,
)
_span_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "fusion_span_stack", default=(),
)


@dataclass
class Span:
    """一个执行阶段的轨迹节点。"""

    span_id: str
    name: str
    parent_id: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    status: str = "ok"          # ok / error
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def duration_ms(self) -> int:
        if not self.end_time:
            return 0
        return int((self.end_time - self.start_time) * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": round(self.start_time, 4),
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "error": self.error,
        }


@dataclass
class Trace:
    """一次问答的完整轨迹。"""

    trace_id: str
    session_id: str = ""
    tenant_id: str = ""
    question: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    spans: list[Span] = field(default_factory=list)
    tags: dict[str, Any] = field(default_factory=dict)
    #: begin() 时按 sample_rate 决定的采样结果。False 时 end() 默认
    #: 不写 JSONL / 不落库；除非命中 keep_error_traces 强制保留。
    sampled: bool = True

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "question": self.question,
            "start_time": round(self.start_time, 4),
            "duration_ms": int((self.end_time - self.start_time) * 1000) if self.end_time else 0,
            "tags": self.tags,
            "spans": [s.to_dict() for s in self.spans],
        }


class Tracer:
    """轨迹记录器。"""

    def __init__(
        self,
        trace_path: str | Path | None = None,
        store: Any | None = None,
        *,
        enabled: bool = True,
        sample_rate: float = 1.0,
        keep_error_traces: bool = True,
        metrics: Any | None = None,
    ) -> None:
        self._path = Path(trace_path).expanduser() if trace_path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store = store
        self.enabled = enabled
        # 采样率：[0, 1]。默认 1.0 保持旧行为，无阅量级风险。
        self._sample_rate = max(0.0, min(1.0, float(sample_rate)))
        # 命中异常/降级的 trace 无条件落盘（便于事后回溯）。
        self._keep_error_traces = bool(keep_error_traces)
        self._metrics = metrics

    # ------------------------------------------------------------------
    def begin(
        self,
        *,
        trace_id: str | None = None,
        session_id: str = "",
        tenant_id: str = "",
        question: str = "",
    ) -> Trace:
        """开启一条新轨迹并绑定到当前上下文（含日志 trace_id）。"""
        trace = Trace(
            trace_id=trace_id or uuid.uuid4().hex,
            session_id=session_id,
            tenant_id=tenant_id,
            question=question,
            sampled=(
                self._sample_rate >= 1.0
                or random.random() < self._sample_rate
            ),
        )
        _current_trace.set(trace)
        _span_stack.set(())
        trace_id_var.set(trace.trace_id)
        session_id_var.set(session_id or "-")
        return trace

    def current(self) -> Trace | None:
        return _current_trace.get()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        """记录一个阶段。异常会被捕获、标注到 span 后重新抛出。"""
        trace = _current_trace.get()
        if trace is None or not self.enabled:
            # 无轨迹上下文：透明直通（仍执行业务）
            yield Span(span_id="-", name=name)
            return

        parent = _span_stack.get()[-1] if _span_stack.get() else ""
        node = Span(span_id=uuid.uuid4().hex[:12], name=name, parent_id=parent)
        node.attributes.update(_jsonable(attributes))
        stack = _span_stack.get()
        token = _span_stack.set(stack + (node.span_id,))
        try:
            yield node
        except Exception as exc:  # noqa: BLE001
            node.status = "error"
            node.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            node.end_time = time.time()
            trace.add_span(node)
            _span_stack.reset(token)

    def tag(self, **tags: Any) -> None:
        """给当前轨迹打标签（如最终置信度、是否降级）。"""
        trace = _current_trace.get()
        if trace is not None:
            trace.tags.update(_jsonable(tags))

    async def end(self, trace: Trace | None = None) -> dict[str, Any] | None:
        """结束轨迹并持久化。返回轨迹字典。"""
        trace = trace or _current_trace.get()
        if trace is None:
            return None
        trace.end_time = time.time()

        # ★ 采样门控：未命中时默认直接丢弃；但异常降级 trace 无条件保留
        #   （否则运维只能看到成功回采，故障现场丢失 → 采样反
        #   而拉低可观测度）。与 span.status="error" 与 tags 里的
        #   degraded/validation_passed=False 一起作为保留信号。
        if not trace.sampled and self._keep_error_traces and self._has_issue(trace):
            trace.sampled = True
        if not trace.sampled:
            self._bump("trace_dropped")
            _current_trace.set(None)
            _span_stack.set(())
            return None

        payload = trace.to_dict()
        self._persist_sync(payload)
        if self._store is not None:
            try:
                await self._store.save_trace(
                    trace.trace_id, payload,
                    session_id=trace.session_id, tenant_id=trace.tenant_id,
                    question=trace.question,
                )
            except Exception:  # noqa: BLE001
                logger.exception("轨迹落库失败 trace_id=%s", trace.trace_id)
        # 清理上下文
        _current_trace.set(None)
        _span_stack.set(())
        return payload

    @staticmethod
    def _has_issue(trace: Trace) -> bool:
        """异常降级判定：任一 span 报错 / tags 标降级 / 校验未通过。"""
        for sp in trace.spans:
            if sp.status == "error":
                return True
        tags = trace.tags or {}
        if tags.get("degraded"):
            return True
        if tags.get("validated") is False:
            return True
        if tags.get("llm_timeout"):
            return True
        return False

    def _bump(self, name: str) -> None:
        if self._metrics is None:
            return
        try:
            self._metrics.counter(name).inc()
        except Exception:  # noqa: BLE001 - 指标失败不能阻断主流程
            pass

    def _persist_sync(self, payload: dict[str, Any]) -> None:
        if self._path is None or not self.enabled:
            return
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("轨迹写文件失败 %s", self._path)

    # ------------------------------------------------------------------
    async def replay(self, trace_id: str) -> dict[str, Any] | None:
        """按 trace_id 回放完整轨迹（优先查库，回退扫 JSONL）。"""
        if self._store is not None:
            row = await self._store.get_trace(trace_id)
            if row:
                try:
                    return json.loads(row["payload"])
                except (KeyError, json.JSONDecodeError):
                    pass
        return self._replay_from_file(trace_id)

    def _replay_from_file(self, trace_id: str) -> dict[str, Any] | None:
        if self._path is None or not self._path.exists():
            return None
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("trace_id") == trace_id:
                    return obj
        return None


def _jsonable(data: dict[str, Any]) -> dict[str, Any]:
    """确保属性能被 JSON 序列化。"""
    result: dict[str, Any] = {}
    for key, value in data.items():
        try:
            json.dumps(value)
            result[key] = value
        except (TypeError, ValueError):
            result[key] = str(value)
    return result
