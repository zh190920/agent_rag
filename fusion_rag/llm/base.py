"""LLM 抽象基类与消息/响应模型。

统一所有模型适配器的接口，实现「模型无绑定」：业务层只依赖
:class:`LLMBase`，具体走 DeepSeek / OpenAI / 通义 / 本地 Ollama / Echo
由 :class:`~fusion_rag.llm.router.LLMRouter` 在运行期决定。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """一条对话消息（对齐 OpenAI Chat Completions 结构）。

    支持 function-calling：``assistant`` 消息可携带 :attr:`tool_calls`
    （模型请求调用的工具），``tool`` 角色消息用 :attr:`tool_call_id` +
    :attr:`name` 回填对应工具的执行结果。
    """

    role: Role
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls("assistant", content)

    @classmethod
    def tool_result(
        cls, content: str, *, tool_call_id: str, name: str = "",
    ) -> "Message":
        """构造一条工具执行结果消息（回填给模型）。"""
        return cls("tool", content, name=name or None, tool_call_id=tool_call_id)


@dataclass
class ToolCall:
    """模型请求的一次工具调用（已归一化）。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
    """从 OpenAI 兼容响应的 ``message.tool_calls`` 提取并归一化工具调用。

    ``arguments`` 是 JSON 字符串，做容错解析：非法/空则退化为空 dict，
    避免因模型输出瑕疵中断整个 agentic 循环。
    """
    if not raw_calls or not isinstance(raw_calls, list):
        return []
    calls: list[ToolCall] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        name = fn.get("name") or item.get("name") or ""
        if not name:
            continue
        raw_args = fn.get("arguments", item.get("arguments", ""))
        calls.append(ToolCall(
            id=str(item.get("id") or name),
            name=str(name),
            arguments=_coerce_args(raw_args),
        ))
    return calls


def _coerce_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if not raw_args or not isinstance(raw_args, str):
        return {}
    import json

    text = raw_args.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


@dataclass
class LLMResponse:
    """模型响应，携带用量、来源与工具调用请求，便于可观测与成本统计。"""

    text: str
    model: str = ""
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0  # ★ 单次 chat 全量响应时延（非流式下即 TTFT）
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def wants_tools(self) -> bool:
        """模型是否请求了工具调用（agentic 循环据此决定是否继续）。"""
        return bool(self.tool_calls)


class LLMBase(abc.ABC):
    """所有大模型适配器的抽象基类。"""

    #: 适配器标识（用于路由与日志）
    name: str = "base"

    #: 是否支持原生 function-calling（工具调用）。离线兜底模型为 False，
    #: 真实模型适配器按能力置 True；Orchestrator/ToolAgent 据此决定是否
    #: 进入 agentic 文件工具循环。
    supports_tools: bool = False

    @abc.abstractmethod
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
        """发起一次对话补全。"""

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """便捷单轮补全，返回纯文本。"""
        resp = await self.chat([Message.user(prompt)], **kwargs)
        return resp.text

    async def chat_json(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """要求模型输出 JSON 并解析；解析失败抛 :class:`ValueError`。"""
        import json

        resp = await self.chat(messages, json_mode=True, **kwargs)
        text = resp.text.strip()
        # 容错：剥离 ```json ... ``` 包裹
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)

    async def aclose(self) -> None:
        """释放底层连接（可选实现）。"""

    def health(self) -> bool:
        """健康探针，路由降级时参考。默认恒真。"""
        return True
