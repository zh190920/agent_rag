"""SentenceTransformer 本地嵌入适配器（可选依赖，隐私本地化首选）。

企业私密知识库场景推荐：模型与数据全部本地，无外部调用。首次使用会
下载模型权重（可预先离线放置到 HF 缓存）。计算放线程池，避免阻塞
事件循环。
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.logging import get_logger
from .base import EmbeddingBase, EmbeddingResponse

logger = get_logger(__name__)


class SentenceTransformerEmbedding(EmbeddingBase):
    """基于 sentence-transformers 的本地嵌入。"""

    name = "sentence_transformers"

    def __init__(
        self,
        model: str = "BAAI/bge-small-zh-v1.5",
        dimensions: int = 512,
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        self._model_name = model
        self._declared_dims = dimensions
        self._device = device
        self._normalize = normalize
        self._batch_size = batch_size
        self._model: Any | None = None
        self._lock = asyncio.Lock()

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "使用 SentenceTransformerEmbedding 需安装: "
                    "pip install sentence-transformers",
                ) from exc
            self._model = SentenceTransformer(self._model_name, device=self._device)
            self._declared_dims = int(self._model.get_sentence_embedding_dimension())
            logger.info("已加载本地嵌入模型 %s (dim=%d)", self._model_name, self._declared_dims)
        return self._model

    @property
    def dimensions(self) -> int:
        # 未加载前返回声明值；加载后返回真实维度
        if self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        return self._declared_dims

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=self._normalize,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(embeddings=[], model=self._model_name)
        loop = asyncio.get_running_loop()
        # 模型首次加载 + 编码均为重 CPU/GPU 操作，放线程池并加锁串行化
        async with self._lock:
            embeddings = await loop.run_in_executor(None, self._embed_sync, texts)
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self._model_name,
            total_tokens=sum(len(t) for t in texts),
        )
