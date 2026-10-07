"""HashEmbedding：零依赖确定性离线嵌入（特征哈希 / hashing trick）。

原理：把文本分词后，每个 token 经稳定哈希映射到固定维度并做带符号
累加，最后 L2 归一化。这样「词面重叠越多，余弦相似度越高」，即使
没有任何模型/网络，向量检索也能返回词法相关的片段——非常适合本地
演示、CI 与作为真实嵌入不可用时的兜底。

确定性：同一文本永远得到同一向量（用 blake2b，跨进程/版本稳定，
不依赖 Python 的随机 hash 种子）。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from ..text import tokenize
from .base import EmbeddingBase, EmbeddingResponse


class HashEmbedding(EmbeddingBase):
    """特征哈希离线嵌入。"""

    name = "hash"

    def __init__(self, dimensions: int = 256, sublinear_tf: bool = True, **kwargs: Any) -> None:
        self._dimensions = max(8, int(dimensions))
        self._sublinear = sublinear_tf

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed_sync(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = tokenize(text)
        if not tokens:
            return vector

        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1

        for tok, tf in counts.items():
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            idx = value % self._dimensions
            sign = 1.0 if (value >> 63) & 1 == 0 else -1.0
            weight = (1.0 + math.log(tf)) if self._sublinear else float(tf)
            vector[idx] += sign * weight

        # L2 归一化 → 余弦相似度即点积
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        # 纯 CPU、无 IO；量大时由调用方（检索层）决定是否放线程池。
        # 这里对批量做本地计算，维度小、开销低。
        embeddings = [self._embed_sync(t) for t in texts]
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.name,
            total_tokens=sum(len(t) for t in texts),
        )
