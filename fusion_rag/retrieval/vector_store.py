"""LocalVectorStore：本地 JSONL 落盘向量库（OpenClaw 隐私本地化）。

特性：
- 每集合一个 ``<collection>.jsonl``，全量本地留存，无云端上传。
- 内存镜像 + 惰性加载：首次访问某集合时从磁盘载入。
- 余弦相似度检索：装了 numpy 走向量点积（归一化后），否则纯 Python；
  计算放线程池避免阻塞事件循环。
- 写操作（insert/delete）经 :class:`asyncio.Lock` 串行化并即时持久化。
- ``metadata_filter`` 纵深隔离：search/list 恒过滤；insert 强制覆盖。

接口对齐 :class:`VectorStoreBase`，可无缝替换为 Chroma/Milvus/Qdrant。
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..types import Chunk
from .base import DocumentSummary, VectorRecord, VectorSearchResult, VectorStoreBase

logger = get_logger(__name__)

try:
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False


def _match_filter(metadata: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    # 代理到共享实现，支持字面/枚举/glob 三种预筛形式
    from .base import match_metadata_filter
    return match_metadata_filter(metadata, flt)


class LocalVectorStore(VectorStoreBase):
    """JSONL 落盘的本地向量库。"""

    def __init__(self, path: str | Path, cpu_executor: Any | None = None) -> None:
        self._root = Path(path).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)
        self._cpu = cpu_executor
        self._lock = asyncio.Lock()
        self._data: dict[str, list[VectorRecord]] = {}
        self._dims: dict[str, int] = {}
        self._loaded: set[str] = set()
        # ★ 批量落盘：insert 不再每条重写全量 JSONL（否则 N 条写 N 次、索引 O(N²)），
        #   而是累积到 _pending 计数 >= 阈值才 flush；aclose 时同步 flush 一次。
        self._pending: dict[str, int] = {}
        self._flush_threshold: int = 200

    # ------------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------------
    def _file(self, collection: str) -> Path:
        safe = collection.replace(os.sep, "_").replace("/", "_")
        return self._root / f"{safe}.jsonl"

    def _meta_file(self, collection: str) -> Path:
        safe = collection.replace(os.sep, "_").replace("/", "_")
        return self._root / f"{safe}.meta.json"

    # ------------------------------------------------------------------
    # 加载与持久化
    # ------------------------------------------------------------------
    def _ensure_loaded(self, collection: str) -> None:
        if collection in self._loaded:
            return
        records: list[VectorRecord] = []
        file = self._file(collection)
        if file.exists():
            with file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("跳过损坏的向量记录行: %s", file)
                        continue
                    chunk = Chunk(
                        content=obj["chunk"]["content"],
                        chunk_index=int(obj["chunk"]["chunk_index"]),
                        metadata=obj["chunk"].get("metadata", {}),
                        block_id=obj["chunk"].get("block_id", ""),
                    )
                    records.append(
                        VectorRecord(
                            vector=obj["vector"],
                            document_id=obj["document_id"],
                            chunk=chunk,
                            metadata=obj.get("metadata", {}),
                        ),
                    )
        # 维度声明（meta 优先，否则从首批记录推）
        meta_file = self._meta_file(collection)
        declared = 0
        if meta_file.exists():
            try:
                declared = int(
                    json.loads(meta_file.read_text(encoding="utf-8")).get("dimensions", 0),
                )
            except (json.JSONDecodeError, ValueError):
                declared = 0
        if not declared and records:
            declared = len(records[0].vector)
        # ★ 维度自我修复：历史上曾以不同 provider 索引时文件里会混进 ragged
        #   记录（如旧 256-d hash + 新 1024-d bge-m3），numpy 直接报 inhomogeneous
        #   shape → hybrid 降级为仅 BM25。创建集合时会把声明维度写回 meta，
        #   此处提前剔除 mismatch 记录，避免 numpy 报错。
        if declared and records:
            stale = [r for r in records if len(r.vector) != declared]
            if stale:
                logger.warning(
                    "集合 %s 剔除 %d/%d 条维度不匹配向量 (declared=%d, seen=%s)；"
                    "如刚切换 embedding provider，请加 --reindex 重建。",
                    collection, len(stale), len(records), declared,
                    sorted({len(r.vector) for r in stale}),
                )
                records = [r for r in records if len(r.vector) == declared]
        self._data[collection] = records
        self._dims[collection] = declared
        self._loaded.add(collection)
        logger.debug("集合 %s 载入 %d 条记录 dim=%d", collection, len(records), declared)

    def _persist(self, collection: str) -> None:
        """原子重写集合文件（临时文件 + replace）。"""
        records = self._data.get(collection, [])
        file = self._file(collection)
        fd, tmp = tempfile.mkstemp(dir=str(self._root), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps({
                        "vector": rec.vector,
                        "document_id": rec.document_id,
                        "metadata": rec.metadata,
                        "chunk": {
                            "content": rec.chunk.content,
                            "chunk_index": rec.chunk.chunk_index,
                            "metadata": rec.chunk.metadata,
                            "block_id": rec.chunk.block_id,
                        },
                    }, ensure_ascii=False) + "\n")
            os.replace(tmp, file)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self._meta_file(collection).write_text(
            json.dumps({"dimensions": self._dims.get(collection, 0)}), encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # VectorStoreBase 实现
    # ------------------------------------------------------------------
    async def create_collection(self, collection: str, dimensions: int) -> None:
        async with self._lock:
            self._ensure_loaded(collection)
            prev = self._dims.get(collection, 0)
            # ★ 切 provider 导致维度变了：旧集合里的向量完全不可用，直接清空，
            #   避免“一半 256-d + 一半 1024-d”把 numpy 拼成 inhomogeneous。
            if prev and prev != dimensions:
                logger.warning(
                    "集合 %s 维度切换 %d → %d；清空旧向量，请 --reindex 重建",
                    collection, prev, dimensions,
                )
                self._data[collection] = []
            self._dims[collection] = dimensions
            if not self._file(collection).exists() or not self._data.get(collection):
                self._persist(collection)

    async def has_collection(self, collection: str) -> bool:
        return self._file(collection).exists() or collection in self._loaded

    async def insert(self, collection: str, records: list[VectorRecord]) -> None:
        if not records:
            return
        async with self._lock:
            self._ensure_loaded(collection)
            self._data[collection].extend(records)
            if collection not in self._dims or not self._dims[collection]:
                self._dims[collection] = len(records[0].vector)
            # ★ 累积到阈值才落盘：避免“每次 insert 全量重写 JSONL”把索引阶段
            #   从秒级拖到分钟级（2851 条 × 2MB 写入 = 5.5GB IO）。
            self._pending[collection] = self._pending.get(collection, 0) + len(records)
            if self._pending[collection] >= self._flush_threshold:
                await self._run_cpu(self._persist, collection)
                self._pending[collection] = 0

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        async with self._lock:
            self._ensure_loaded(collection)
            records = list(self._data.get(collection, []))
        if not records:
            return []
        scored = await self._run_cpu(
            self._cosine_topk, records, query_vector, top_k, metadata_filter,
        )
        return scored

    def _cosine_topk(
        self,
        records: list[VectorRecord],
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[VectorSearchResult]:
        # 先按过滤条件筛，再算相似度
        candidates = [
            r for r in records if _match_filter(r.metadata, metadata_filter)
        ]
        if not candidates:
            return []
        if _HAS_NUMPY:
            matrix = np.asarray([r.vector for r in candidates], dtype=np.float32)
            qv = np.asarray(query_vector, dtype=np.float32)
            qnorm = np.linalg.norm(qv) or 1.0
            mnorms = np.linalg.norm(matrix, axis=1)
            mnorms[mnorms == 0] = 1.0
            sims = (matrix @ qv) / (mnorms * qnorm)
            k = min(top_k, len(candidates))
            # argpartition 取 top-k 再排序
            idx = np.argpartition(-sims, k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]
            results = [
                VectorSearchResult(
                    score=float(sims[i]),
                    document_id=candidates[i].document_id,
                    chunk=candidates[i].chunk,
                    metadata=candidates[i].metadata,
                )
                for i in idx
            ]
        else:
            scored: list[tuple[float, VectorRecord]] = []
            for rec in candidates:
                scored.append((_cosine_py(rec.vector, query_vector), rec))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [
                VectorSearchResult(
                    score=score,
                    document_id=rec.document_id,
                    chunk=rec.chunk,
                    metadata=rec.metadata,
                )
                for score, rec in scored[:top_k]
            ]
        return results

    async def delete(self, collection: str, document_id: str) -> None:
        async with self._lock:
            self._ensure_loaded(collection)
            before = len(self._data[collection])
            self._data[collection] = [
                r for r in self._data[collection] if r.document_id != document_id
            ]
            if len(self._data[collection]) != before:
                # 删除相对稀有 → 依旧全量落盘（保证一致性），但重置挂起计数
                await self._run_cpu(self._persist, collection)
                self._pending[collection] = 0

    async def delete_by_metadata(
        self, collection: str, key: str, value: Any,
    ) -> int:
        """按元数据字段删除一批记录（reindex 同 source 时清理旧孤儿）。"""
        async with self._lock:
            self._ensure_loaded(collection)
            before = len(self._data.get(collection, []))
            self._data[collection] = [
                r for r in self._data.get(collection, [])
                if str(r.metadata.get(key, "")) != str(value or "")
            ]
            removed = before - len(self._data[collection])
            if removed:
                await self._run_cpu(self._persist, collection)
                self._pending[collection] = 0
            return removed

    async def list_documents(
        self, collection: str, metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentSummary]:
        async with self._lock:
            self._ensure_loaded(collection)
            records = list(self._data.get(collection, []))
        grouped: dict[str, dict[str, Any]] = {}
        for rec in records:
            if not _match_filter(rec.metadata, metadata_filter):
                continue
            entry = grouped.setdefault(
                rec.document_id, {"count": 0, "metadata": dict(rec.metadata)},
            )
            entry["count"] += 1
        return [
            DocumentSummary(document_id=did, chunk_count=v["count"], metadata=v["metadata"])
            for did, v in grouped.items()
        ]

    async def aclose(self) -> None:
        # ★ 关库时一次性 flush 所有挂起集合；走同步 _persist 而不经 _run_cpu，
        #   避开 Windows Proactor 下 executor 已拆 → 抛 "Event loop is closed"。
        async with self._lock:
            for collection in list(self._loaded):
                if self._pending.get(collection, 0) == 0 and collection in self._data:
                    # 无挂起变更也要写一次 meta 保证 dims 一致
                    pass
                try:
                    self._persist(collection)
                except Exception:  # noqa: BLE001
                    logger.exception("集合 %s 落盘失败", collection)
            self._pending.clear()

    # ------------------------------------------------------------------
    async def _run_cpu(self, func: Any, *args: Any) -> Any:
        if self._cpu is not None:
            return await self._cpu.run(func, *args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)


def _cosine_py(a: list[float], b: list[float]) -> float:
    """纯 Python 余弦相似度。"""
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
