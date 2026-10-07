"""框架共享数据类型（跨模块契约）。

集中定义 Chunk / Document / RetrievedChunk / Citation / Answer /
QaResult 等，避免各层之间重复定义、循环依赖。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def new_id() -> str:
    return uuid.uuid4().hex


# ----------------------------------------------------------------------
# 文档与分块
# ----------------------------------------------------------------------
@dataclass
class Chunk:
    """知识库文档切片（检索与嵌入的最小单元）。"""

    content: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    # 稳定身份：跨重建索引不变（对齐 agentscope (document_id, chunk_index)）
    block_id: str = field(default_factory=new_id)

    @property
    def token_estimate(self) -> int:
        return approx_token_count(self.content)


@dataclass
class Document:
    """一篇源文档。"""

    content: str
    document_id: str = field(default_factory=new_id)
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


@dataclass
class RetrievedChunk:
    """一次检索命中的切片，带来源与得分。"""

    chunk: Chunk
    document_id: str
    score: float
    kb_id: str = ""
    tenant_id: str = ""
    retriever: str = ""       # vector / bm25 / hybrid
    title: str = ""
    source: str = ""

    @property
    def identity(self) -> tuple[str, int]:
        """稳定去重键：(document_id, chunk_index)。"""
        return (self.document_id, self.chunk.chunk_index)

    @property
    def content(self) -> str:
        return self.chunk.content


# ----------------------------------------------------------------------
# 引用与答案
# ----------------------------------------------------------------------
@dataclass
class Citation:
    """答案引用来源。"""

    index: int
    document_id: str
    title: str
    source: str
    chunk_index: int
    snippet: str
    score: float = 0.0


class OutputFormat(str, Enum):
    """自适应输出格式。"""

    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    STEPS = "steps"


@dataclass
class Answer:
    """推理 Agent 产出的答案草稿/终稿。"""

    text: str
    citations: list[Citation] = field(default_factory=list)
    output_format: OutputFormat = OutputFormat.PARAGRAPH
    confidence: float = 0.0
    model: str = ""


@dataclass
class ValidationResult:
    """校验 Agent 结论。"""

    passed: bool
    confidence: float
    issues: list[str] = field(default_factory=list)
    correction: str = ""


@dataclass
class IntentResult:
    """意图识别结论。"""

    intent: str = "qa"          # qa / chitchat / reject / clarify
    domain: str = "general"
    risk: str = "low"           # low / medium / high
    valid: bool = True
    reason: str = ""


class Priority(str, Enum):
    NORMAL = "normal"
    PROFESSIONAL = "professional"
    URGENT = "urgent"


_PRIORITY_ORDER = {
    Priority.URGENT: 0,
    Priority.PROFESSIONAL: 1,
    Priority.NORMAL: 2,
}


def priority_rank(p: Priority) -> int:
    return _PRIORITY_ORDER.get(p, 2)


@dataclass
class QaResult:
    """一次完整问答的最终产物（API/CLI 返回体）。"""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    session_id: str = ""
    trace_id: str = ""
    intent: IntentResult = field(default_factory=IntentResult)
    sub_questions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    validated: bool = False
    output_format: OutputFormat = OutputFormat.PARAGRAPH
    redacted: bool = False
    latency_ms: int = 0
    model: str = ""
    degraded: bool = False      # 是否走了降级/兜底路径
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.__dict__ for c in self.citations],
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "intent": self.intent.__dict__,
            "sub_questions": self.sub_questions,
            "confidence": self.confidence,
            "validated": self.validated,
            "output_format": self.output_format.value,
            "redacted": self.redacted,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "degraded": self.degraded,
            "meta": self.meta,
        }


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def approx_token_count(text: str) -> int:
    """近似 token 数：中文按字、英文按 ~4 字符/词混合估算。

    无需引入 tiktoken；对预算裁剪足够精确。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    # 纯中文时不额外计入英文 token，避免恒定 +1 偏差
    other_tokens = max(1, other // 4) if other else 0
    return cjk + other_tokens


def now_ts() -> float:
    return time.time()
