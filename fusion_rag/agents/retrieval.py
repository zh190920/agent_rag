"""RetrievalAgent：调度混合检索器，输出带来源的候选知识。

职责单一：把「子问题文本 + 租户作用域」翻译为一次隔离检索。多子问题时
并发检索各自证据，再全局去重合并（借鉴 agentscope 多路检索并发合并），
交给上层统一推理，避免逐子问题分别生成造成的答案割裂。

租户隔离是硬约束：``ctx.scope_filter()`` 的 tenant_id + kb_id 恒注入检索
层，任何越界结果在向量库/BM25 过滤阶段即被丢弃。
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.logging import get_logger
from ..retrieval.base import RetrieverBase
from ..types import RetrievedChunk
from .base import AgentContext, BaseAgent

logger = get_logger(__name__)


def merge_chunks(
    groups: list[list[RetrievedChunk]], limit: int | None = None,
) -> list[RetrievedChunk]:
    """跨子问题合并候选：按 (document_id, chunk_index) 去重、保留最高分。

    合并后按分数降序；``limit`` 截断（None 表示不截断）。
    """
    best: dict[tuple[str, int], RetrievedChunk] = {}
    for group in groups:
        for chunk in group:
            key = chunk.identity
            existing = best.get(key)
            if existing is None or chunk.score > existing.score:
                best[key] = chunk
    ordered = sorted(best.values(), key=lambda c: c.score, reverse=True)
    return ordered[:limit] if limit else ordered


class RetrievalAgent(BaseAgent):
    """检索 Agent。"""

    name = "retrieval"
    role = "向量/关键词/跨文档关联检索，输出原始参考知识（多租户隔离）"

    def __init__(
        self,
        retriever: RetrieverBase,
        *args: Any,
        default_top_k: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._retriever = retriever
        self.default_top_k = default_top_k

    # ------------------------------------------------------------------
    async def retrieve(
        self,
        query: str,
        ctx: AgentContext,
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """单子问题检索（强制注入租户作用域）。"""
        top_k = top_k or self.default_top_k
        if not query or not query.strip():
            return []
        with self.span("retrieve", top_k=top_k, qlen=len(query)) as node:
            try:
                hits = await self._retriever.retrieve(
                    query,
                    top_k=top_k,
                    tenant_id=ctx.tenant_id,
                    kb_ids=ctx.kb_ids or None,
                )
            except Exception as exc:  # noqa: BLE001
                node.status = "error"
                node.error = f"{type(exc).__name__}: {exc}"
                self.count("retrieval_errors")
                logger.warning("检索失败，返回空候选：%s", exc)
                return []
            node.attributes["hits"] = len(hits)
            self.observe("retrieval_hits", float(len(hits)))
            return hits

    async def retrieve_for_plan(
        self,
        sub_questions: list[str],
        ctx: AgentContext,
        *,
        top_k: int | None = None,
        final_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """按执行计划并发检索所有子问题，合并去重后返回统一证据集。

        - 单个子问题：直接检索。
        - 多个子问题：``asyncio.gather`` 并发；任一失败降级为空，不影响其余。
        - 复合问题需要更广证据面，``final_k`` 默认放宽到 ``top_k * 2``。
        """
        top_k = top_k or self.default_top_k
        queries = [q for q in (sub_questions or []) if q and q.strip()]
        if not queries:
            return []
        if len(queries) == 1:
            return await self.retrieve(queries[0], ctx, top_k=final_k or top_k)

        with self.span("retrieve_multi", sub_questions=len(queries)) as node:
            groups = await asyncio.gather(
                *(self.retrieve(q, ctx, top_k=top_k) for q in queries),
                return_exceptions=False,
            )
            merged = merge_chunks(list(groups), limit=final_k or top_k * 2)
            node.attributes["merged"] = len(merged)
            self.count("retrieval_multi", subs=str(len(queries)))
            return merged
