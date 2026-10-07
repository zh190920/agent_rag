"""KnowledgeIndexer：知识库增量索引（新增/修改/删除，无需全量重建）。

生产级要点：
- **增量去重**：按 ``content_hash`` 判重，未变更文档直接跳过（幂等重建）。
- **入库前脱敏**：隐私数据在向量化前即脱敏（OpenClaw 本地合规）。
- **租户隔离**：每条向量/BM25 记录强制打上 ``tenant_id`` + ``kb_id``，
  与检索层过滤构成纵深防御；写入前校验租户对该 KB 的访问权。
- **并发友好**：嵌入分批 ``asyncio.gather`` 并发；向量库/BM25 各自加锁串行写。
- **双索引一致**：向量库 + BM25 + 文档登记表三处同步更新。
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..chunking.chunker import ChunkerBase
from ..core.logging import get_logger
from ..embedding.base import EmbeddingBase
from ..retrieval.base import VectorRecord, VectorStoreBase
from ..retrieval.bm25 import BM25Index
from ..security.access_control import AccessController
from ..security.redaction import Redactor
from ..storage.sqlite_store import SQLiteStore
from ..types import Chunk, Document, new_id
from .loader import LoadedFile, load_directory

logger = get_logger(__name__)


@dataclass
class IndexResult:
    """单篇文档的索引结果。"""

    document_id: str
    chunks: int = 0
    skipped: bool = False
    reason: str = ""
    tenant_id: str = ""
    kb_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class KnowledgeIndexer:
    """知识库索引器。"""

    def __init__(
        self,
        *,
        embedding: EmbeddingBase,
        vector_store: VectorStoreBase,
        bm25: BM25Index,
        chunker: ChunkerBase,
        store: SQLiteStore,
        collection: str = "knowledge",
        redactor: Redactor | None = None,
        access: AccessController | None = None,
        embed_batch_size: int = 32,
    ) -> None:
        self._embedding = embedding
        self._vector_store = vector_store
        self._bm25 = bm25
        self._chunker = chunker
        self._store = store
        self._redactor = redactor
        self._access = access
        self.collection = collection
        self.embed_batch_size = max(1, embed_batch_size)
        self._ready = False

    # ------------------------------------------------------------------
    async def ensure_collection(self) -> None:
        """幂等创建向量集合并预载 BM25 索引。"""
        if self._ready:
            return
        await self._vector_store.create_collection(self.collection, self._embedding.dimensions)
        # 触发 BM25 落盘内容惰性载入（一次空检索即可）
        await self._bm25.search("", top_k=1)
        self._ready = True
        logger.info(
            "索引集合就绪 collection=%s dim=%d bm25=%d",
            self.collection, self._embedding.dimensions, self._bm25.size,
        )

    # ------------------------------------------------------------------
    async def add_document(
        self, doc: Document, *, tenant_id: str, kb_id: str, force: bool = False,
    ) -> IndexResult:
        return await self.add_text(
            doc.content,
            tenant_id=tenant_id, kb_id=kb_id,
            title=doc.title, source=doc.source,
            document_id=doc.document_id, metadata=doc.metadata,
            content_hash=doc.content_hash, force=force,
        )

    async def add_text(
        self,
        text: str,
        *,
        tenant_id: str,
        kb_id: str,
        title: str = "",
        source: str = "",
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        content_hash: str = "",
        force: bool = False,
    ) -> IndexResult:
        """索引一段文本（增量、幂等）。"""
        await self.ensure_collection()
        if self._access is not None:
            self._access.check_kb_access(tenant_id, kb_id)

        text = text or ""
        if not text.strip():
            return IndexResult(document_id or "", skipped=True, reason="empty",
                               tenant_id=tenant_id, kb_id=kb_id)

        content_hash = content_hash or _sha256(text)
        if not force:
            existing = await self._store.find_document_by_hash(tenant_id, kb_id, content_hash)
            if existing:
                logger.debug("文档未变更，跳过索引：%s", existing.get("document_id"))
                return IndexResult(
                    document_id=str(existing.get("document_id", "")),
                    chunks=int(existing.get("chunk_count", 0)),
                    skipped=True, reason="duplicate-hash",
                    tenant_id=tenant_id, kb_id=kb_id,
                )

        document_id = document_id or new_id()
        # ★ 幂等 reindex：无论文档上一次用什么 document_id 落盘（旧行可能为
        #   随机 UUID），都以 source 为主键先把旧向量与 BM25 记录清掉；
        #   否则 --reindex 会“新 UUID 新行 → 旧行永远不被删”，数据会越重越乱。
        if source:
            await self._purge_by_source(source)
        # 入库前脱敏（隐私合规）
        redacted = False
        if self._redactor is not None and self._redactor.enabled:
            result = self._redactor.redact(text)
            text = result.text
            redacted = result.redacted

        base_meta: dict[str, Any] = {
            "tenant_id": tenant_id, "kb_id": kb_id,
            "title": title, "source": source, "document_id": document_id,
        }
        if metadata:
            base_meta.update({k: v for k, v in metadata.items() if k not in base_meta})

        chunks: list[Chunk] = self._chunker.split(text, metadata=dict(base_meta))
        if not chunks:
            return IndexResult(document_id, skipped=True, reason="no-chunks",
                               tenant_id=tenant_id, kb_id=kb_id)

        # 覆盖旧版本（同 document_id 重复索引时先清理）
        await self._remove_indexes(document_id)

        vectors = await self._embed_all([c.content for c in chunks])
        records = [
            VectorRecord(vector=vec, document_id=document_id, chunk=chunk, metadata=dict(base_meta))
            for vec, chunk in zip(vectors, chunks)
        ]
        await self._vector_store.insert(self.collection, records)
        await self._bm25.add(document_id, chunks, metadata=dict(base_meta))
        await self._store.record_document(
            document_id, tenant_id=tenant_id, kb_id=kb_id, title=title, source=source,
            content_hash=content_hash, chunk_count=len(chunks),
            meta={"redacted": redacted, **(metadata or {})},
        )
        logger.info("已索引文档 %s（%d 块，tenant=%s kb=%s）", document_id, len(chunks), tenant_id, kb_id)
        return IndexResult(document_id=document_id, chunks=len(chunks),
                           tenant_id=tenant_id, kb_id=kb_id, metadata={"redacted": redacted})

    async def _embed_all(self, texts: list[str]) -> list[list[float]]:
        batches = [
            texts[i:i + self.embed_batch_size]
            for i in range(0, len(texts), self.embed_batch_size)
        ]
        responses = await asyncio.gather(*(self._embedding.embed(b) for b in batches))
        vectors: list[list[float]] = []
        for resp in responses:
            vectors.extend(resp.embeddings)
        if len(vectors) != len(texts):
            raise RuntimeError(f"嵌入数量不匹配：期望 {len(texts)} 得到 {len(vectors)}")
        return vectors

    # ------------------------------------------------------------------
    async def delete_document(self, document_id: str) -> None:
        """删除文档的所有索引（向量 + BM25 + 登记）。"""
        await self._remove_indexes(document_id)
        await self._store.delete_document(document_id)
        logger.info("已删除文档索引 %s", document_id)

    async def _remove_indexes(self, document_id: str) -> None:
        try:
            await self._vector_store.delete(self.collection, document_id)
        except Exception:  # noqa: BLE001
            logger.exception("删除向量失败 doc=%s", document_id)
        try:
            await self._bm25.remove_document(document_id)
        except Exception:  # noqa: BLE001
            logger.exception("删除 BM25 失败 doc=%s", document_id)

    async def _purge_by_source(self, source: str) -> None:
        """按 source 路径清掉同一文件的旧索引（reindex 幂等）。"""
        if not source:
            return
        try:
            vec_del = getattr(self._vector_store, "delete_by_metadata", None)
            if callable(vec_del):
                n = await vec_del(self.collection, "source", source)
                if n:
                    logger.info("清理旧向量 %d 条 (source=%s)", n, source)
        except Exception:  # noqa: BLE001
            logger.exception("清理旧向量失败 source=%s", source)
        try:
            bm_del = getattr(self._bm25, "remove_by_metadata", None)
            if callable(bm_del):
                n = await bm_del("source", source)
                if n:
                    logger.info("清理旧 BM25 %d 条 (source=%s)", n, source)
        except Exception:  # noqa: BLE001
            logger.exception("清理旧 BM25 失败 source=%s", source)

    # ------------------------------------------------------------------
    async def ingest_directory(
        self,
        root: str | Path,
        *,
        tenant_id: str,
        kb_id: str,
        globs: Iterable[str] | None = None,
        force: bool = False,
    ) -> list[IndexResult]:
        """批量索引一个本地目录（增量）。"""
        files: list[LoadedFile] = await asyncio.get_running_loop().run_in_executor(
            None, lambda: load_directory(root, globs=globs),
        )
        results: list[IndexResult] = []
        for f in files:
            results.append(await self.add_text(
                f.content, tenant_id=tenant_id, kb_id=kb_id,
                title=f.title, source=f.metadata.get("source", str(f.path)),
                metadata=f.metadata, force=force,
            ))
        indexed = sum(1 for r in results if not r.skipped)
        logger.info("目录索引完成：%s → 新增/更新 %d，跳过 %d", root, indexed, len(results) - indexed)
        return results

    async def stats(self) -> dict[str, Any]:
        docs = await self._vector_store.list_documents(self.collection)
        return {
            "collection": self.collection,
            "documents": len(docs),
            "bm25_records": self._bm25.size,
            "dimensions": self._embedding.dimensions,
        }

    async def aclose(self) -> None:
        """索引器本身无独占资源；确保 BM25/向量落盘由各自 aclose 负责。"""
        return None
