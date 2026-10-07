"""BM25 关键词索引（纯 Python + CJK 分词，借鉴 hermes FTS5 CJK）。

零第三方依赖即可对中文做关键词精准检索，与向量检索互补构成混合检索。
索引在内存维护倒排表，记录以 JSONL 落盘；启动时从落盘内容重建倒排。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..text import tokenize, tokenize_query
from ..types import Chunk, RetrievedChunk

logger = get_logger(__name__)


@dataclass
class _Bm25Doc:
    key: tuple[str, int]          # (document_id, chunk_index)
    document_id: str
    chunk: Chunk
    metadata: dict[str, Any]
    tokens: list[str] = field(default_factory=list)
    tf: dict[str, int] = field(default_factory=dict)
    length: int = 0


def _match_filter(metadata: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    # 代理到共享实现，支持字面/枚举/glob 三种预筛形式
    from .base import match_metadata_filter
    return match_metadata_filter(metadata, flt)


class BM25Index:
    """Okapi BM25 倒排索引。"""

    def __init__(
        self,
        path: str | Path | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        cpu_executor: Any | None = None,
    ) -> None:
        self._path = Path(path).expanduser() if path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self.k1 = k1
        self.b = b
        self._cpu = cpu_executor
        self._lock = asyncio.Lock()

        self._docs: dict[tuple[str, int], _Bm25Doc] = {}
        self._postings: dict[str, set[tuple[str, int]]] = {}
        self._total_len = 0
        self._loaded = False
        # ★ 批量落盘：与 LocalVectorStore 同理，避免每次 add 全量重写。
        self._pending = 0
        self._flush_threshold = 200

    # ------------------------------------------------------------------
    @property
    def avgdl(self) -> float:
        n = len(self._docs)
        return (self._total_len / n) if n else 0.0

    @property
    def size(self) -> int:
        return len(self._docs)

    def _index_doc(self, doc: _Bm25Doc) -> None:
        tokens = tokenize(doc.chunk.content)
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        doc.tokens = tokens
        doc.tf = tf
        doc.length = len(tokens)
        self._docs[doc.key] = doc
        self._total_len += doc.length
        for tok in tf:
            self._postings.setdefault(tok, set()).add(doc.key)

    def _deindex_doc(self, key: tuple[str, int]) -> None:
        doc = self._docs.pop(key, None)
        if doc is None:
            return
        self._total_len -= doc.length
        for tok in doc.tf:
            posting = self._postings.get(tok)
            if posting is not None:
                posting.discard(key)
                if not posting:
                    self._postings.pop(tok, None)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded or self._path is None or not self._path.exists():
            self._loaded = True
            return
        count = 0
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = Chunk(
                    content=obj["content"],
                    chunk_index=int(obj["chunk_index"]),
                    metadata=obj.get("chunk_metadata", {}),
                )
                doc = _Bm25Doc(
                    key=(obj["document_id"], chunk.chunk_index),
                    document_id=obj["document_id"],
                    chunk=chunk,
                    metadata=obj.get("metadata", {}),
                )
                self._index_doc(doc)
                count += 1
        self._loaded = True
        logger.debug("BM25 索引载入 %d 条记录", count)

    def _persist(self) -> None:
        if self._path is None:
            return
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for doc in self._docs.values():
                    fh.write(json.dumps({
                        "document_id": doc.document_id,
                        "chunk_index": doc.chunk.chunk_index,
                        "content": doc.chunk.content,
                        "chunk_metadata": doc.chunk.metadata,
                        "metadata": doc.metadata,
                    }, ensure_ascii=False) + "\n")
            os.replace(tmp, self._path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------------
    # 增删查（异步 + 锁）
    # ------------------------------------------------------------------
    async def add(
        self,
        document_id: str,
        chunks: list[Chunk],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata or {}
        async with self._lock:
            self._ensure_loaded()
            for chunk in chunks:
                key = (document_id, chunk.chunk_index)
                if key in self._docs:
                    self._deindex_doc(key)
                merged = {**metadata, **chunk.metadata}
                self._index_doc(_Bm25Doc(
                    key=key, document_id=document_id, chunk=chunk, metadata=merged,
                ))
            # ★ 累积到阈值才落盘（旧实现每写一批就全量重写、索引 2851 条要写 5000 次）
            self._pending += len(chunks)
            if self._pending >= self._flush_threshold:
                await self._run_cpu(self._persist)
                self._pending = 0

    async def remove_document(self, document_id: str) -> None:
        async with self._lock:
            self._ensure_loaded()
            keys = [k for k in self._docs if k[0] == document_id]
            for key in keys:
                self._deindex_doc(key)
            if keys:
                await self._run_cpu(self._persist)
                self._pending = 0

    async def remove_by_metadata(self, key: str, value: Any) -> int:
        """按元数据字段删除一批记录（reindex 同 source 时清理旧孤儿）。"""
        async with self._lock:
            self._ensure_loaded()
            target = str(value or "")
            doomed = [
                k for k, d in self._docs.items()
                if str(d.metadata.get(key, "")) == target and target
            ]
            for k in doomed:
                self._deindex_doc(k)
            if doomed:
                await self._run_cpu(self._persist)
                self._pending = 0
            return len(doomed)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        async with self._lock:
            self._ensure_loaded()
            if not self._docs:
                return []
            query_tokens = tokenize_query(query)
            if not query_tokens:
                return []
            docs_snapshot = dict(self._docs)
            # ★ set(...) 拷壳：_postings 内的 set 对象会被并发的 add/remove
            #   就地 mutate（.add / .discard），而 _score 已交给 executor 在另一
            #   线程跑 → “dictionary/set changed during iteration” 或错误统计。
            #   dict(self._docs) 拷的是引用（旧 doc 对象不会就地改 tf），安全；
            #   但 posting set 必须拷壳。拷量级 ≈ 命中项 × 查询词（百到千），开销可忽。
            postings_snapshot = {
                t: set(self._postings.get(t, ())) for t in query_tokens
            }
            n_docs = len(self._docs)
            avgdl = self.avgdl or 1.0
        return await self._run_cpu(
            self._score, query_tokens, postings_snapshot, docs_snapshot,
            n_docs, avgdl, top_k, metadata_filter,
        )

    def _score(
        self,
        query_tokens: list[str],
        postings: dict[str, set[tuple[str, int]]],
        docs: dict[tuple[str, int], _Bm25Doc],
        n_docs: int,
        avgdl: float,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        scores: dict[tuple[str, int], float] = {}
        for tok in query_tokens:
            matched = postings.get(tok)
            if not matched:
                continue
            df = len(matched)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for key in matched:
                doc = docs.get(key)
                if doc is None or not _match_filter(doc.metadata, metadata_filter):
                    continue
                tf = doc.tf.get(tok, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * doc.length / avgdl)
                scores[key] = scores.get(key, 0.0) + idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results: list[RetrievedChunk] = []
        for key, score in ranked:
            doc = docs[key]
            results.append(RetrievedChunk(
                chunk=doc.chunk,
                document_id=doc.document_id,
                score=score,
                kb_id=str(doc.metadata.get("kb_id", "")),
                tenant_id=str(doc.metadata.get("tenant_id", "")),
                retriever="bm25",
                title=str(doc.metadata.get("title", "")),
                source=str(doc.metadata.get("source", "")),
            ))
        return results

    async def aclose(self) -> None:
        async with self._lock:
            if self._loaded:
                # ★ 直接同步写：避开 Windows Proactor 下 executor 已拆导致
                #   "Event loop is closed" 异常；且保证挂起的批次能兜底 flush。
                self._persist()
                self._pending = 0

    async def _run_cpu(self, func: Any, *args: Any) -> Any:
        if self._cpu is not None:
            return await self._cpu.run(func, *args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)
