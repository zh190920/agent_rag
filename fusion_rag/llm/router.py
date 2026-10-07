"""LLM 路由器：模型无绑定 + 重试 + 降级链（借鉴 DeepSeek-Harness）。

- 业务层通过 ``router.chat(...)`` / ``router.complete(...)`` 调用，
  用 ``provider=`` 指定优先使用哪个已注册模型；缺省走 default。
- 失败按 ``fallback_chain`` 依次降级；每个 provider 内部再重试
  ``retries`` 次（指数退避）。
- 统一累计 token 用量，供成本统计与可观测层消费。
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from ..core.exceptions import LLMError
from ..core.logging import get_logger
from .base import LLMBase, LLMResponse, Message

logger = get_logger(__name__)


class LLMRouter(LLMBase):
    """把多个模型适配器聚合成一个带路由/降级能力的逻辑模型。"""

    name = "router"

    def __init__(
        self,
        providers: dict[str, LLMBase],
        default: str = "echo",
        fallback_chain: list[str] | None = None,
        retries: int = 2,
        backoff_base: float = 0.5,
    ) -> None:
        if default not in providers:
            raise LLMError(f"默认模型 {default} 未在 providers 中注册")
        self.providers = providers
        self.default = default
        self.retries = max(retries, 0)
        self.backoff_base = backoff_base
        # 降级链：显式配置 > [default] + 其余
        self.fallback_chain = fallback_chain or (
            [default] + [p for p in providers if p != default]
        )
        # 用量累计
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}

    def provider_names(self) -> list[str]:
        return list(self.providers)

    def supports_tools(self, provider: str | None = None) -> bool:
        """本次将使用的首选 provider 是否支持原生 function-calling。

        ToolAgent/Orchestrator 据此决定是否进入 agentic 文件工具循环；
        离线 EchoLLM 返回 False，循环被跳过。
        """
        name = provider if (provider and provider in self.providers) else self.default
        llm = self.providers.get(name)
        return bool(getattr(llm, "supports_tools", False))

    def _chain_for(self, provider: str | None) -> list[str]:
        """构造本次调用的尝试顺序：指定 provider 优先，其后接降级链。"""
        if provider and provider in self.providers:
            rest = [p for p in self.fallback_chain if p != provider]
            return [provider, *rest]
        return list(self.fallback_chain)

    async def chat(
        self,
        messages: list[Message],
        *,
        provider: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        chain = self._chain_for(provider)
        last_error: Exception | None = None

        for name in chain:
            llm = self.providers.get(name)
            if llm is None:
                continue
            # 非首选 provider 且已知不健康时跳过（default 始终尝试）
            if name != self.default and not llm.health():
                logger.debug("跳过不健康模型 %s", name)
                continue
            for attempt in range(self.retries + 1):
                try:
                    # 向不支持 function-calling 的模型发 tools 会导致非法请求，
                    # 故按当前 provider 能力剔除 tools/tool_choice。
                    call_kwargs = kwargs
                    if kwargs.get("tools") and not getattr(llm, "supports_tools", False):
                        call_kwargs = {
                            k: v for k, v in kwargs.items()
                            if k not in ("tools", "tool_choice")
                        }
                    resp = await llm.chat(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                        stop=stop,
                        **call_kwargs,
                    )
                    self._record_usage(resp, name)
                    return resp
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    self.usage["errors"] += 1
                    if attempt < self.retries:
                        delay = self.backoff_base * (2 ** attempt) + random.uniform(0, 0.1)
                        logger.warning(
                            "模型 %s 第 %d 次调用失败：%s；%.2fs 后重试",
                            name, attempt + 1, exc, delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error("模型 %s 重试耗尽，尝试降级：%s", name, exc)

        raise LLMError(f"所有模型均调用失败（chain={chain}）: {last_error}")

    async def chat_json(
        self,
        messages: list[Message],
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import json

        resp = await self.chat(messages, provider=provider, json_mode=True, **kwargs)
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        # 再兜底：截取首个 { 到末个 }
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]
        return json.loads(text)

    def _record_usage(self, resp: LLMResponse, provider: str) -> None:
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += resp.prompt_tokens
        self.usage["completion_tokens"] += resp.completion_tokens

    def health(self) -> bool:
        return any(p.health() for p in self.providers.values())

    async def aclose(self) -> None:
        for llm in self.providers.values():
            try:
                await llm.aclose()
            except Exception:  # noqa: BLE001
                logger.exception("关闭模型 %s 失败", getattr(llm, "name", llm))


def _since(start: float) -> float:  # pragma: no cover - 工具函数
    return time.monotonic() - start
