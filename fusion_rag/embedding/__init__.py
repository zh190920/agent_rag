"""嵌入模型适配层（插件化，可替换）。"""

from .base import EmbeddingBase, EmbeddingResponse
from .hash_embedding import HashEmbedding
from .openai_embedding import OpenAIEmbedding
from .sentence_transformers_embedding import SentenceTransformerEmbedding

__all__ = [
    "EmbeddingBase", "EmbeddingResponse",
    "HashEmbedding", "OpenAIEmbedding", "SentenceTransformerEmbedding",
]
