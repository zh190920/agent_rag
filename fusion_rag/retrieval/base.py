"""检索层抽象：向量库接口与检索器接口（对齐 agentscope VectorStoreBase）。"""

from __future__ import annotations

import abc
import fnmatch
from dataclasses import dataclass, field
from typing import Any

from ..types import Chunk, RetrievedChunk


def _is_glob_pattern(s: str) -> bool:
    return any(ch in s for ch in ("*", "?", "["))


def match_metadata_filter(
    metadata: dict[str, Any], flt: dict[str, Any] | None,
) -> bool:
    """统一的多租户+业务预筛匹配。

    支持三种形式：
    - 字面值 ``{"tenant_id": "t1"}`` → 精确相等
    - 枚举   ``{"kb_id": ["a", "b"]}`` → metadata 在列表中
    - Glob   ``{"title": ["Easy320*", "H5U*"]}`` → 任一模式匹配

    对列表元素：如任一字符串含通配符，则走 fnmatch 路径；否则走 ``in`` 集合判定。
    非字符串列表元素一律当作枚举处理。
    """
    if not flt:
        return True
    for key, value in flt.items():
        actual = metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            str_globs = [
                p for p in items if isinstance(p, str) and _is_glob_pattern(p)
            ]
            if str_globs:
                # glob 路径：任一模式匹配即通过（仅对字符串字段有意义）
                if not isinstance(actual, str):
                    return False
                if not any(fnmatch.fnmatchcase(actual, p) for p in str_globs):
                    return False
            else:
                if actual not in items:
                    return False
        elif isinstance(value, str) and _is_glob_pattern(value):
            if not isinstance(actual, str) or not fnmatch.fnmatchcase(actual, value):
                return False
        else:
            if actual != value:
                return False
    return True


@dataclass
class VectorRecord:
    """一条待写入向量库的记录。"""

    vector: list[float]
    document_id: str
    chunk: Chunk
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """向量检索单条命中。"""

    score: float
    document_id: str
    chunk: Chunk
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSummary:
    """文档概要（list_documents 用）。"""

    document_id: str
    chunk_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStoreBase(abc.ABC):
    """向量库插件接口。

    ``metadata_filter`` 为多租户纵深隔离机制：search/list 永不逃逸该
    过滤；insert 时强制覆盖到每条记录（见 LocalVectorStore）。
    """

    @abc.abstractmethod
    async def create_collection(self, collection: str, dimensions: int) -> None:
        """幂等创建集合。"""

    @abc.abstractmethod
    async def has_collection(self, collection: str) -> bool:
        """集合是否存在。"""

    @abc.abstractmethod
    async def insert(self, collection: str, records: list[VectorRecord]) -> None:
        """批量写入。"""

    @abc.abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """按向量相似度检索。"""

    @abc.abstractmethod
    async def delete(self, collection: str, document_id: str) -> None:
        """按文档 id 删除其全部记录。"""

    @abc.abstractmethod
    async def list_documents(
        self, collection: str, metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentSummary]:
        """列出集合内文档概要。"""

    async def aclose(self) -> None:
        """释放资源（可选）。"""


class RetrieverBase(abc.ABC):
    """检索器接口：输入查询文本，输出带来源的候选切片。"""

    name: str = "base"

    @abc.abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 6,
        tenant_id: str = "",
        kb_ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """检索候选切片（已按相关性排序、去重）。"""
