"""测试共享工具。

由于目标环境未强制安装 pytest-asyncio，这里统一用 ``asyncio.run`` 驱动异步
场景，测试函数保持同步，最大化可移植性（pytest>=6 均可运行）。
"""

from __future__ import annotations

import asyncio
import tempfile
from typing import Any, Callable

from fusion_rag.core.config import load_config
from fusion_rag.core.kernel import Kernel
from fusion_rag.llm.base import LLMBase, LLMResponse, parse_tool_calls


class ScriptedToolLLM(LLMBase):
    """测试替身：按预设脚本回放响应（含 tool_calls），驱动 agentic 循环。

    仅用于测试/示例验证机制，**不是**生产离线兵底（生产由真实 function-
    calling 模型驱动）。``supports_tools=True`` 以让 ToolAgent 进入工具循环。

    script: 逐项为 ``{"text": str, "tool_calls": [openai格式]}``；脚本耗尽后
    返回空响应（无 tool_calls），使循环自然终止。
    """

    name = "scripted"
    supports_tools = True

    def __init__(self, script=None) -> None:
        self._script = list(script or [])
        self._i = 0
        self.requests: list[list[dict]] = []

    async def chat(self, messages, **kwargs) -> LLMResponse:  # type: ignore[override]
        self.requests.append([m.to_dict() for m in messages])
        if self._i < len(self._script):
            item = self._script[self._i]
            self._i += 1
        else:
            item = {"text": "", "tool_calls": []}
        raw_calls = item.get("tool_calls") or []
        return LLMResponse(
            text=item.get("text", ""),
            model=self.name,
            finish_reason="tool_calls" if raw_calls else "stop",
            tool_calls=parse_tool_calls(raw_calls),
        )


def make_kernel(**overrides: Any) -> Kernel:
    """构建一个使用独立临时数据目录的 Kernel（互不干扰、可并发跑）。"""
    data_dir = tempfile.mkdtemp(prefix="fusion-test-")
    base: dict[str, Any] = {
        "app": {"data_dir": data_dir, "log_json": False, "log_level": "WARNING"},
    }
    for key, value in overrides.items():
        section = key.split("__")
        node = base
        for part in section[:-1]:
            node = node.setdefault(part, {})
        node[section[-1]] = value
    return Kernel(load_config(None, base))


def run(coro_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """在独立事件循环中运行一个协程函数。"""
    return asyncio.run(coro_func(*args, **kwargs))


SAMPLE_DOCS = [
    ("AgentScope 是阿里巴巴开源的多智能体框架，支持分布式编排、可视化会话管理"
     "与高并发问答处理，适配多角色协作问答场景。", "agentscope"),
    ("DeepSeek Harness 采用全插件化微内核架构，模型无绑定，支持任务轨迹回放、"
     "可插拔工具链与轻量化沙箱，具备高可观测性。", "harness"),
    ("DeepAgents 擅长长任务智能规划与复杂问题分层拆解，提供上下文卸载、超长对话"
     "记忆与人工介入校验机制，解决上下文溢出问题。", "deepagents"),
    ("OpenClaw 强调本地化隐私可控、持久化智能体会话与私有知识库安全隔离，"
     "适配企业私密知识问答场景。", "openclaw"),
    ("Claude Code 具备深度语义理解、长文本精读推理、精细化结果校验与逻辑纠错"
     "能力，可提升问答精准度与答案完整性。", "claudecode"),
]


async def seed(kernel: Kernel, docs=None, tenant="default", kb="general") -> int:
    """把样本文档灌入知识库，返回成功索引篇数。"""
    docs = docs if docs is not None else SAMPLE_DOCS
    count = 0
    for text, title in docs:
        result = await kernel.indexer.add_text(
            text, tenant_id=tenant, kb_id=kb, title=title,
        )
        if not result.skipped:
            count += 1
    return count
