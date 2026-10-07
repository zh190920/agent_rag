"""示例 04：运行时 agentic 文件工具（对标 Claude Code / hermes harness）。

演示大模型如何通过原生 function-calling **自主多轮调用只读文件工具**
（grep → read_file），在授权目录内探查原始知识库，最终给出带
``文件名:行号`` 可核对引用的答案。

生产模式：配置真实 OpenAI 兼容模型（``supports_tools=True``），由模型驱动循环。
离线演示：本示例内置一个脚本化的 function-calling 替身（仅用于展示机制），
按预设顺序请求工具，让没有网络/密钥的读者也能直观看到完整循环。

运行::

    python examples/04_agentic_tools.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# 允许直接运行脚本（未安装为包时）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_rag.agents.base import AgentContext  # noqa: E402
from fusion_rag.agents.tool_agent import ToolAgent  # noqa: E402
from fusion_rag.llm.base import LLMBase, LLMResponse, parse_tool_calls  # noqa: E402
from fusion_rag.tools.executor import ToolExecutor  # noqa: E402
from fusion_rag.tools.file_tools import (  # noqa: E402
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
)
from fusion_rag.tools.registry import ToolRegistry  # noqa: E402

# 原始知识语料（模拟一个可被工具探查的私有文档目录）
CORPUS = {
    "docs/agentscope.md": (
        "# AgentScope 概览\n"
        "AgentScope 支持多智能体分布式编排与高并发问答。\n"
        "它提供可视化会话管理，适配多角色协作场景。\n"
        "DeepAgents 负责复杂问题分层拆解。\n"
    ),
    "docs/openclaw.md": (
        "# OpenClaw\n"
        "OpenClaw 强调本地化隐私可控与私有知识库安全隔离。\n"
    ),
    "notes.txt": "运维备注：索引每日凌晨重建。\n",
}


class ScriptedFunctionCallingLLM(LLMBase):
    """离线演示替身：按脚本回放 tool_calls，模拟真实模型的自主工具调用。

    仅用于在没有网络/密钥时展示 agentic 循环机制；生产中由真实
    function-calling 模型（``OpenAICompatibleLLM``）驱动同一套 ToolAgent。
    """

    name = "scripted"
    supports_tools = True

    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self._i = 0

    async def chat(self, messages, **kwargs) -> LLMResponse:  # type: ignore[override]
        if self._i < len(self._script):
            item = self._script[self._i]
            self._i += 1
        else:
            item = {"text": "", "tool_calls": []}
        raw = item.get("tool_calls") or []
        return LLMResponse(
            text=item.get("text", ""),
            model=self.name,
            finish_reason="tool_calls" if raw else "stop",
            tool_calls=parse_tool_calls(raw),
        )


def _tc(call_id: str, name: str, args: dict) -> dict:
    """构造 OpenAI 线格式的 tool_call。"""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _build_corpus(root: Path) -> None:
    for rel, text in CORPUS.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="fusion-agentic-"))
    _build_corpus(root)
    print(f"授权目录（PathJail 作用域）：{root}")
    print("语料文件：", ", ".join(sorted(CORPUS)))

    # 1) 装配只读工具 + 执行器 + ToolAgent
    registry = ToolRegistry()
    for tool in (GrepTool(), GlobTool(), ListDirTool(), ReadFileTool()):
        registry.register(tool)
    executor = ToolExecutor(registry, call_timeout=15)

    # 2) 脚本化的 function-calling 演示：grep 定位 → read_file 精读 → 作答
    llm = ScriptedFunctionCallingLLM([
        {"tool_calls": [_tc("c1", "grep", {"pattern": "高并发", "regex": False})]},
        {"tool_calls": [_tc("c2", "read_file",
                            {"path": "docs/agentscope.md", "offset": 1, "limit": 2})]},
        {"text": "AgentScope 支持多智能体分布式编排与高并发问答"
                 "（docs/agentscope.md:2）。", "tool_calls": []},
    ])

    agent = ToolAgent(
        llm=llm, registry=registry, executor=executor, max_iters=6,
        roots_resolver=lambda ctx: [str(root)],
    )

    # 3) 运行 agentic 循环
    ctx = AgentContext(tenant_id="default", session_id="agentic-demo",
                       kb_ids=["general"])
    question = "AgentScope 在高并发方面能力如何？"
    print(f"\n问：{question}\n")
    result = await agent.run(question, ctx)

    # 4) 展示每一轮工具调用（transcript）
    print("== 工具调用轨迹 ==")
    for i, step in enumerate(result.transcript, 1):
        args = json.dumps(step["args"], ensure_ascii=False)
        flag = "错误" if step["is_error"] else "成功"
        print(f"[{i}] {step['tool']}({args}) -> {flag}")
        preview = " ".join(step["preview"].split())[:120]
        print(f"     {preview}")

    # 5) 最终答案与文件引用
    print("\n== 最终答案 ==")
    print(result.answer)
    print(f"\n迭代轮数={result.iterations} 工具调用数={result.tool_calls} "
          f"降级={'是' if result.degraded else '否'} 模型={result.model}")
    print("文件引用：")
    for cite in result.file_citations:
        line = cite.get("line")
        print(f"  - {cite['path']}" + (f":{line}" if line else ""))

    # 6) 演示 PathJail 越权拦截（安全纵深防御）
    print("\n== 越权访问演示 ==")
    escape = await executor.dispatch(
        _toolcall("read_file", {"path": "../../../etc/passwd"}),
        _tool_ctx(root),
    )
    print(f"读取 jail 外文件 -> is_error={escape.is_error}")
    print(" ".join(escape.content.split())[:120])


def _toolcall(name: str, args: dict):
    from fusion_rag.llm.base import ToolCall
    return ToolCall(id="demo", name=name, arguments=args)


def _tool_ctx(root: Path):
    from fusion_rag.tools.base import ToolContext
    return ToolContext(tenant_id="default", kb_ids=["general"], roots=[str(root)])


if __name__ == "__main__":
    asyncio.run(main())
