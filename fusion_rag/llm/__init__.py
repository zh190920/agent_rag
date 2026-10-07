"""LLM 适配层：模型无绑定（借鉴 DeepSeek-Harness）。"""

from .base import LLMResponse, LLMBase, Message
from .echo import EchoLLM
from .openai_compatible import OpenAICompatibleLLM
from .router import LLMRouter

__all__ = [
    "LLMBase", "LLMResponse", "Message",
    "EchoLLM", "OpenAICompatibleLLM", "LLMRouter",
]
