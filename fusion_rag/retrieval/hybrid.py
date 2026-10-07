"""HybridRetriever：向量 + BM25 双路并发召回 → RRF 融合 → 精排。

复刻 agentscope 的多路检索合并思路：两路检索用 ``asyncio.gather`` 并发，
按 ``(document_id, chunk_index)`` 稳定身份去重，用倒数排名融合(RRF)综合
两路名次，再交给 Reranker 精排截断到 top_k。

多租户隔离：``retrieve`` 恒将 ``tenant_id`` + ``kb_id`` 注入 metadata
过滤，检索结果永不越界。
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.logging import get_logger
from ..embedding.base import EmbeddingBase
from ..types import RetrievedChunk
from .base import RetrieverBase, VectorStoreBase
from .bm25 import BM25Index
from .reranker import HeuristicReranker, RerankerBase

logger = get_logger(__name__)


class HybridRetriever(RetrieverBase):
    """混合检索器。"""

    name = "hybrid"

    def __init__(
        self,
        embedding: EmbeddingBase,
        vector_store: VectorStoreBase,
        bm25: BM25Index,
        collection: str,
        reranker: RerankerBase | None = None,
        *,
        vector_recall: int = 20,
        bm25_recall: int = 20,
        rrf_k: int = 60,
        score_threshold: float = 0.0,
    ) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._bm25 = bm25
        self._collection = collection
        self._reranker = reranker or HeuristicReranker()
        self.vector_recall = vector_recall
        self.bm25_recall = bm25_recall
        self.rrf_k = rrf_k
        self.score_threshold = score_threshold

    def _build_filter(
        self,
        tenant_id: str,
        kb_ids: list[str] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        flt: dict[str, Any] = {}
        if tenant_id:
            flt["tenant_id"] = tenant_id
        if kb_ids:
            flt["kb_id"] = list(kb_ids)
        if extra:
            flt.update(extra)
        return flt

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 6,
        tenant_id: str = "",
        kb_ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []
        flt = self._build_filter(tenant_id, kb_ids, metadata_filter)

        # 双路并发召回（单路失败不拖垮另一路）
        vector_task = asyncio.create_task(self._vector_recall(query, flt))
        bm25_task = asyncio.create_task(self._bm25_recall(query, flt))
        vector_hits, bm25_hits = await asyncio.gather(vector_task, bm25_task)

        fused = self._rrf_fuse(vector_hits, bm25_hits)
        # ★ 旧实现先拿 RRF 分数过 score_threshold（阈值 0.01）——但 RRF 分天然落在
        #   1/(60+rank) ≈ 0.016–0.03 区间，几乎不会过滤任何候选，属无效配置。
        #   改为在 cross-encoder 给出真实 relevance_score (0–1) 之后再过门槛。
        if not fused:
            return []
        reranked = await self._reranker.rerank(query, fused, top_k)
        if self.score_threshold > 0 and reranked:
            reranked = [c for c in reranked if c.score >= self.score_threshold]
        return reranked

    async def _vector_recall(
        self, query: str, flt: dict[str, Any],
    ) -> list[RetrievedChunk]:
        try:
            response = await self._embedding.embed([query])
            if not response.embeddings:
                return []
            results = await self._vector_store.search(
                collection=self._collection,
                query_vector=response.embeddings[0],
                top_k=self.vector_recall,
                metadata_filter=flt,
            )
            return [
                RetrievedChunk(
                    chunk=r.chunk,
                    document_id=r.document_id,
                    score=r.score,
                    kb_id=str(r.metadata.get("kb_id", "")),
                    tenant_id=str(r.metadata.get("tenant_id", "")),
                    retriever="vector",
                    title=str(r.metadata.get("title", "")),
                    source=str(r.metadata.get("source", "")),
                )
                for r in results
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("向量召回失败，降级仅 BM25：%s", exc)
            return []

    async def _bm25_recall(
        self, query: str, flt: dict[str, Any],
    ) -> list[RetrievedChunk]:
        try:
            return await self._bm25.search(
                query, top_k=self.bm25_recall, metadata_filter=flt,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 召回失败，降级仅向量：%s", exc)
            return []

    def _rrf_fuse(
        self,
        vector_hits: list[RetrievedChunk],
        bm25_hits: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """倒数排名融合：score = Σ 1/(rrf_k + rank)。"""
        fused: dict[tuple[str, int], RetrievedChunk] = {}
        scores: dict[tuple[str, int], float] = {}

        for hits in (vector_hits, bm25_hits):
            for rank, chunk in enumerate(hits):
                key = chunk.identity
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                # 保留信息更全的一份（向量结果通常带来源元数据）
                existing = fused.get(key)
                if existing is None or (not existing.title and chunk.title):
                    fused[key] = chunk

        for key, score in scores.items():
            fused[key].score = round(score, 6)
        ordered = sorted(fused.values(), key=lambda c: c.score, reverse=True)
        return ordered

    async def aclose(self) -> None:
        """释放子资源（如 reranker 的 HTTP 连接池）。"""
        aclose = getattr(self._reranker, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass
