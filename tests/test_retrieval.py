"""检索层测试：离线嵌入确定性、BM25、混合检索排序。"""

from __future__ import annotations

import asyncio
import math
import tempfile

from fusion_rag.chunking.chunker import Chunker
from fusion_rag.embedding.hash_embedding import HashEmbedding
from fusion_rag.retrieval.bm25 import BM25Index
from fusion_rag.retrieval.hybrid import HybridRetriever
from fusion_rag.retrieval.vector_store import LocalVectorStore
from fusion_rag.types import Chunk


def test_hash_embedding_deterministic_and_normalized():
    async def scenario():
        emb = HashEmbedding(dimensions=128)
        a = (await emb.embed(["多智能体协作调度"])).embeddings[0]
        b = (await emb.embed(["多智能体协作调度"])).embeddings[0]
        assert a == b, "同文本嵌入必须确定"
        assert len(a) == 128
        norm = math.sqrt(sum(x * x for x in a))
        assert abs(norm - 1.0) < 1e-6, "向量应 L2 归一化"

    asyncio.run(scenario())


def test_hash_embedding_similarity_ordering():
    async def scenario():
        emb = HashEmbedding(dimensions=256)
        resp = await emb.embed([
            "多智能体分布式编排框架",
            "多智能体分布式调度框架",
            "今天天气很好适合散步",
        ])
        q = (await emb.embed(["多智能体分布式编排框架"])).embeddings[0]
        sims = [sum(x * y for x, y in zip(q, v)) for v in resp.embeddings]
        assert sims[0] > sims[2], "同主题相似度应高于无关文本"
        assert sims[1] > sims[2]

    asyncio.run(scenario())


def test_bm25_keyword_match():
    async def scenario():
        bm25 = BM25Index(path=None)
        chunks = [
            Chunk(content="AgentScope 支持多智能体分布式编排", chunk_index=0),
            Chunk(content="OpenClaw 强调本地化隐私与数据留存", chunk_index=0),
        ]
        await bm25.add("d1", chunks[:1])
        await bm25.add("d2", chunks[1:])
        hits = await bm25.search("分布式编排", top_k=2)
        assert hits, "BM25 应命中关键词"
        assert hits[0].document_id == "d1"

    asyncio.run(scenario())


def test_hybrid_retriever_ranks_relevant_first():
    async def scenario():
        tmp = tempfile.mkdtemp(prefix="fusion-ret-")
        emb = HashEmbedding(dimensions=256)
        vs = LocalVectorStore(tmp, cpu_executor=None)
        bm25 = BM25Index(path=None)
        chunker = Chunker(max_tokens=200, overlap_tokens=0)
        await vs.create_collection("knowledge", emb.dimensions)

        docs = {
            "d_as": "AgentScope 是多智能体分布式编排框架，支持高并发问答处理。",
            "d_oc": "OpenClaw 强调本地化隐私可控与私有知识库安全隔离。",
            "d_da": "DeepAgents 擅长长任务智能规划与复杂问题分层拆解。",
        }
        from fusion_rag.retrieval.base import VectorRecord

        for did, text in docs.items():
            chunks = chunker.split(text, metadata={"tenant_id": "default", "kb_id": "general", "document_id": did})
            vecs = (await emb.embed([c.content for c in chunks])).embeddings
            recs = [
                VectorRecord(vector=v, document_id=did, chunk=c,
                             metadata={"tenant_id": "default", "kb_id": "general"})
                for v, c in zip(vecs, chunks)
            ]
            await vs.insert("knowledge", recs)
            await bm25.add(did, chunks, metadata={"tenant_id": "default", "kb_id": "general"})

        retriever = HybridRetriever(emb, vs, bm25, "knowledge")
        hits = await retriever.retrieve(
            "多智能体分布式编排", top_k=3, tenant_id="default", kb_ids=["general"],
        )
        assert hits, "混合检索应有结果"
        assert hits[0].document_id == "d_as", f"最相关应为 AgentScope，实际 {hits[0].document_id}"
        await vs.aclose()

    asyncio.run(scenario())


def test_hybrid_tenant_isolation():
    """跨租户检索必须被 metadata 过滤挡住（纵深防御）。"""
    async def scenario():
        tmp = tempfile.mkdtemp(prefix="fusion-iso-")
        emb = HashEmbedding(dimensions=128)
        vs = LocalVectorStore(tmp, cpu_executor=None)
        bm25 = BM25Index(path=None)
        await vs.create_collection("knowledge", emb.dimensions)
        from fusion_rag.retrieval.base import VectorRecord

        chunk = Chunk(content="租户A的机密知识内容", chunk_index=0)
        vec = (await emb.embed([chunk.content])).embeddings[0]
        await vs.insert("knowledge", [VectorRecord(
            vector=vec, document_id="secret", chunk=chunk,
            metadata={"tenant_id": "tenantA", "kb_id": "general"},
        )])
        await bm25.add("secret", [chunk], metadata={"tenant_id": "tenantA", "kb_id": "general"})

        retriever = HybridRetriever(emb, vs, bm25, "knowledge")
        hits = await retriever.retrieve(
            "机密知识", top_k=5, tenant_id="tenantB", kb_ids=["general"],
        )
        assert all(h.tenant_id != "tenantA" for h in hits), "不得跨租户泄露"
        await vs.aclose()

    asyncio.run(scenario())
