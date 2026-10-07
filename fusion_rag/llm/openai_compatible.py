"""OpenAI 兼容协议适配器。

一套代码覆盖所有「OpenAI Chat Completions 兼容」的服务端：
DeepSeek、OpenAI、通义千问(dashscope 兼容模式)、Moonshot、以及本地
vLLM / Ollama(/v1) / LM Studio 等。

- 传输层优先 aiohttp（连接池、真异步）；未安装则回退到 urllib +
  线程池，保证零依赖也能跑。
- 内置两层并发配额（:class:`ConcurrencyQuota`）：全局闸门兜底保护
  下游模型服务，按租户/请求子闸门避免单租户 fanout 占满窗口饿死他人。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from ..core.exceptions import LLMError, LLMTimeoutError
from ..core.logging import get_logger
from ..core.quota import ConcurrencyQuota, get_quota_key
from .base import LLMBase, LLMResponse, Message, parse_tool_calls

logger = get_logger(__name__)

try:  # 可选依赖
    import aiohttp  # type: ignore

    _HAS_AIOHTTP = True
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore
    _HAS_AIOHTTP = False


class OpenAICompatibleLLM(LLMBase):
    """OpenAI 兼容 Chat Completions 适配器。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        api_key_env: str | None = None,
        timeout: float = 60.0,
        max_concurrency: int = 16,
        tenant_max_concurrency: int | None = None,
        name: str | None = None,
        default_max_tokens: int = 2048,
        supports_tools: bool = True,
        **kwargs: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name or model
        self.timeout = timeout
        self.default_max_tokens = default_max_tokens
        self.supports_tools = supports_tools
        self._extra = kwargs
        # API Key：显式参数 > 环境变量名 > 空
        self.api_key = api_key or (os.environ.get(api_key_env, "") if api_key_env else "")
        # ★ 两层并发配额：全局 max_concurrency 兜底；tenant_max_concurrency
        #   为按租户/请求子额度（None 或 >= 全局则关闭子配额，行为同旧单闸门）。
        self._quota = ConcurrencyQuota(max_concurrency, tenant_max_concurrency)
        self._session: Any | None = None
        # ★ session lazy init 锁：并发首波请求下旧写法会双 new ClientSession，
        #   其中一个永远不能 close → 泄漏一个连接池。
        self._session_lock = asyncio.Lock()
        self._healthy = True

    # ------------------------------------------------------------------
    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
        stop: list[str] | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        if stop:
            payload["stop"] = stop
        # json_mode 与 tools 互斥：携带工具时不强制 JSON 输出格式
        if json_mode and "tools" not in extra:
            payload["response_format"] = {"type": "json_object"}
        payload.update(extra)
        return payload

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload = self._payload(
            messages, temperature, max_tokens, json_mode, stop, kwargs,
        )
        _t0 = time.perf_counter()
        async with self._quota.acquire(get_quota_key()):
            data, ttft = await self._post(payload)
        resp = self._parse(data)
        # 优先用流式测出的 TTFT；无流式回退到全量响应时间
        resp.latency_s = ttft if ttft > 0 else time.perf_counter() - _t0
        return resp

    # ------------------------------------------------------------------
    # 传输层
    # ------------------------------------------------------------------
    async def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Return (assembled_response_dict, ttft_seconds). ttft=−1 if unavailable."""
        if _HAS_AIOHTTP:
            return await self._post_aiohttp(payload)
        data = await self._post_urllib(payload)
        return data, -1.0

    async def _post_aiohttp(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Streaming SSE request: returns (assembled_response_dict, ttft_seconds)."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    self._session = aiohttp.ClientSession(timeout=timeout)
        # ★ 开启流式 + 尾 chunk 包含 usage
        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        _t0 = time.perf_counter()
        ttft: float = -1.0
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        model_name = ""
        usage: dict[str, Any] = {}
        try:
            async with self._session.post(
                self._endpoint, json=payload, headers=self._headers(),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    self._healthy = False
                    raise LLMError(f"{self.name} HTTP {resp.status}: {text[:300]}")
                self._healthy = True
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if not model_name:
                        model_name = chunk.get("model", "")
                    choices = chunk.get("choices", [])
                    if not choices:
                        # 尾 chunk 可能只有 usage
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        continue
                    delta = choices[0].get("delta", {}) or {}
                    # ★ 第一个有内容的 chunk = TTFT
                    if ttft < 0 and (delta.get("content") or delta.get("tool_calls")):
                        ttft = time.perf_counter() - _t0
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if delta.get("tool_calls"):
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": "", "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc_delta.get("id"):
                                tool_calls_acc[idx]["id"] = tc_delta["id"]
                            fn = tc_delta.get("function") or {}
                            if fn.get("name"):
                                tool_calls_acc[idx]["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                    if chunk.get("usage"):
                        usage = chunk["usage"]
        except asyncio.TimeoutError as exc:
            self._healthy = False
            raise LLMTimeoutError(f"{self.name} 调用超时（{self.timeout}s）") from exc
        except aiohttp.ClientError as exc:
            self._healthy = False
            raise LLMError(f"{self.name} 网络错误: {exc}") from exc
        # 拼装成标准 OpenAI response 格式，复用 _parse
        message: dict[str, Any] = {"role": "assistant"}
        full_text = "".join(content_parts)
        message["content"] = full_text or None
        if tool_calls_acc:
            message["tool_calls"] = list(tool_calls_acc.values())
        assembled: dict[str, Any] = {
            "model": model_name or self.model,
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
        }
        return assembled, ttft

    async def _post_urllib(self, payload: dict[str, Any]) -> dict[str, Any]:
        """无 aiohttp 时的兜底：urllib 放线程池执行，不阻塞事件循环。"""
        import urllib.error
        import urllib.request

        def _call() -> dict[str, Any]:
            req = urllib.request.Request(
                self._endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, _call),
                timeout=self.timeout + 5,
            )
            self._healthy = True
            return result
        except urllib.error.HTTPError as exc:  # pragma: no cover
            self._healthy = False
            body = exc.read().decode("utf-8", "ignore")[:300]
            raise LLMError(f"{self.name} HTTP {exc.code}: {body}") from exc
        except (asyncio.TimeoutError, TimeoutError) as exc:
            self._healthy = False
            raise LLMTimeoutError(f"{self.name} 调用超时") from exc
        except urllib.error.URLError as exc:  # pragma: no cover
            self._healthy = False
            raise LLMError(f"{self.name} 网络错误: {exc.reason}") from exc

    def _parse(self, data: dict[str, Any]) -> LLMResponse:
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            finish = choice.get("finish_reason", "stop")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name} 响应格式异常: {str(data)[:200]}") from exc
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            finish_reason=finish,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            tool_calls=parse_tool_calls(message.get("tool_calls")),
            raw=data,
        )

    def health(self) -> bool:
        return self._healthy

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            # ★ Windows Proactor 下 session.close() 后 socket transport 仍挂在
            #   GC 环里，下一拍 loop 关停 → _ProactorBasePipeTransport.__del__
            #   对已关 loop 调 call_soon 抱 RuntimeError: Event loop is closed
            #   （非致命但日志会刷一片）。close 后睡 0.25s 让 connector 排干
            #   不活跃 socket，aiohttp 官方文档推荐的优雅关退方式。
            await asyncio.sleep(0.25)
        self._session = None
