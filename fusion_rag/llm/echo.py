"""EchoLLM：零依赖离线兜底模型。

用途：
1. 无网络/无 API Key 时让全链路端到端跑通（CI、本地演示）
2. 单元测试中作为确定性桩

它具备「最小可用智能」：能识别框架发出的结构化 JSON 指令并回以合法
JSON，使意图识别、任务拆解、校验等 Agent 在离线路径下仍有确定行为。
"""

from __future__ import annotations

import json
from typing import Any

from .base import LLMBase, LLMResponse, Message

# 各 Agent 系统提示中约定的 JSON 协议标记，EchoLLM 据此返回结构骨架
_MARKERS = {
    "__INTENT__": {
        "intent": "qa",
        "domain": "general",
        "risk": "low",
        "valid": True,
        "reason": "echo-fallback",
    },
    "__DECOMPOSE__": {"need_decompose": False, "sub_questions": []},
    "__VALIDATE__": {
        "passed": True,
        "confidence": 0.6,
        "issues": [],
        "correction": "",
    },
    "__RERANK__": {"order": []},
}


class EchoLLM(LLMBase):
    """确定性回显模型。"""

    name = "echo"
    #: 离线兜底不具备真实 function-calling 能力，不进入 agentic 工具循环
    supports_tools = False

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

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
        joined = "\n".join(m.content for m in messages)
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), "",
        )

        # 命中结构化协议标记 → 返回对应骨架（可能带上下文覆盖）
        for marker, skeleton in _MARKERS.items():
            if marker in joined:
                payload = self._fill(marker, skeleton, last_user)
                text = json.dumps(payload, ensure_ascii=False)
                return self._resp(text, json_mode)

        if json_mode:
            return self._resp(
                json.dumps({"echo": last_user}, ensure_ascii=False), json_mode,
            )

        # 普通问答：给出可追溯的回显答案，明确标注为离线兜底
        text = (
            f"[EchoLLM 离线兜底] 已收到问题：{last_user.strip()}\n"
            "当前未配置真实大模型。请在 config.yaml 的 llm.providers 配置 "
            "DeepSeek/OpenAI/Ollama 等，并把 llm.router.default 指向它，"
            "即可获得真实推理答案。"
        )
        return self._resp(text, json_mode)

    def _fill(
        self, marker: str, skeleton: dict[str, Any], last_user: str,
    ) -> dict[str, Any]:
        payload = dict(skeleton)
        if marker == "__DECOMPOSE__":
            # 离线启发式：按中文/英文连接词粗拆，交给上层再兜底
            payload["sub_questions"] = _naive_split(last_user)
            payload["need_decompose"] = len(payload["sub_questions"]) > 1
        if marker == "__RERANK__":
            payload["order"] = []
        return payload

    def _resp(self, text: str, json_mode: bool) -> LLMResponse:
        return LLMResponse(
            text=text,
            model=self.name,
            finish_reason="stop",
            prompt_tokens=0,
            completion_tokens=len(text),
            raw={"echo": True, "json_mode": json_mode},
        )


_SPLIT_WORDS = (
    " 和 ", " 以及 ", " 还有 ", "，然后", "。然后", " 并且 ", " 同时 ",
    " and ", " also ", ", then ", " besides ",
)


def _naive_split(question: str) -> list[str]:
    parts = [question]
    for sep in _SPLIT_WORDS:
        new_parts: list[str] = []
        for part in parts:
            new_parts.extend(part.split(sep))
        parts = new_parts
    result = [p.strip(" ？?。.") for p in parts if p and p.strip(" ？?。.")]
    return result[:5] if len(result) > 1 else []
