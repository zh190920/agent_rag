"""嵌入模型抽象基类。

对齐 agentscope ``EmbeddingModelBase`` 的批量语义：一次调用接收
文本列表，返回等长向量列表；索引期与检索期必须使用同一模型，
否则向量不可比。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddingResponse:
    """批量嵌入结果。"""

    embeddings: list[list[float]]
    model: str = ""
    total_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.embeddings)


class EmbeddingBase(abc.ABC):
    """嵌入模型接口。"""

    name: str = "base"
    supports_multimodal: bool = False

    @property
    @abc.abstractmethod
    def dimensions(self) -> int:
        """向量维度（建集合时用）。"""

    @abc.abstractmethod
    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """批量嵌入。实现方需保证返回向量数 == 输入文本数。"""

    async def embed_one(self, text: str) -> list[float]:
        resp = await self.embed([text])
        if not resp.embeddings:
            raise RuntimeError("嵌入模型返回空结果")
        return resp.embeddings[0]

    async def __call__(self, texts: list[str]) -> EmbeddingResponse:
        return await self.embed(texts)

    async def aclose(self) -> None:
        """释放底层资源（可选）。"""
