"""示例 05：Skills 渐进式披露（Anthropic Agent Skills 模式）。

演示把「教模型怎么做某类任务的流程说明书（SKILL.md）」接入 agentic 工具循环：

1. 从技能目录加载 ``SKILL.md``（YAML frontmatter：name/description/allowed-tools）；
2. **渐进式披露**：system prompt 只列 name+description+读取路径（省 token）；
3. 模型判断任务匹配后，用**已有的 read_file 工具**按需读取 SKILL.md 全文；
4. 技能目录被自动并入工具 jail 授权根，越权仍被拒（与知识库目录隔离但统一受控）。

生产模式：真实 function-calling 模型自行决定读哪个技能、按流程执行。
离线演示：内置脚本化替身按「先读技能 → 再按流程 grep/read_file → 作答」回放。

运行::

    python examples/05_skills.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_rag.agents.base import AgentContext  # noqa: E402
from fusion_rag.agents.tool_agent import ToolAgent  # noqa: E402
from fusion_rag.llm.base import LLMBase, LLMResponse, parse_tool_calls  # noqa: E402
from fusion_rag.skills.loader import scan_layer  # noqa: E402
from fusion_rag.skills.registry import SkillRegistry  # noqa: E402
from fusion_rag.tools.base import ToolContext  # noqa: E402
from fusion_rag.tools.executor import ToolExecutor  # noqa: E402
from fusion_rag.tools.file_tools import (  # noqa: E402
    FindTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
)
from fusion_rag.tools.registry import ToolRegistry  # noqa: E402

# 知识库语料（可被工具探查的私有文档目录）
CORPUS = {
    "docs/agentscope.md": (
        "# AgentScope 概览\n"
        "AgentScope 支持多智能体分布式编排与高并发问答。\n"
        "它提供可视化会话管理，适配多角色协作场景。\n"
    ),
}

# 一个技能：报告撰写流程（正文是给模型的详细步骤说明书）
SKILL_MD = """\
---
name: report-writer
description: 依据知识库原始文档撰写结构化技术摘要，必须带文件名:行号引用。
license: MIT
allowed-tools: [grep, read_file]
---

# 报告撰写流程（技能正文，模型按需读取）
1. 用 grep 在知识库定位关键论据行号；
2. 用 read_file 精读命中片段所在上下文；
3. 以「结论先行 + `文件名:行号` 引用」组织答案，工作目录 ${SKILL_DIR}。
"""


class ScriptedFunctionCallingLLM(LLMBase):
    """离线演示替身：按脚本回放 tool_calls，模拟真实模型的自主技能读取与执行。"""

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


def _build_skills(skill_root: Path) -> None:
    d = skill_root / "report-writer"
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")


async def main() -> None:
    kb_root = Path(tempfile.mkdtemp(prefix="fusion-skills-kb-"))
    skill_root = Path(tempfile.mkdtemp(prefix="fusion-skills-"))
    _build_corpus(kb_root)
    _build_skills(skill_root)

    # 1) 加载技能层 → SkillRegistry（技能目录将并入 jail 授权根）
    skills = scan_layer(skill_root, source="project")
    skill_registry = SkillRegistry(skills, roots=[str(skill_root)])
    print(f"知识库授权目录：{kb_root}")
    print(f"技能授权目录：  {skill_root}")
    print(f"已加载技能：    {', '.join(skill_registry.names())}")

    # 2) 装配只读工具 + 执行器 + ToolAgent（带 skill_registry）
    registry = ToolRegistry()
    for tool in (GrepTool(), GlobTool(), FindTool(), ListDirTool(), ReadFileTool()):
        registry.register(tool)
    executor = ToolExecutor(registry, call_timeout=15)

    # 3) 脚本化演示：先按渐进披露读 SKILL.md → 依流程 grep → read_file → 作答
    llm = ScriptedFunctionCallingLLM([
        {"tool_calls": [_tc("c1", "read_file",
                            {"path": "report-writer/SKILL.md", "offset": 1, "limit": 20})]},
        {"tool_calls": [_tc("c2", "grep", {"pattern": "高并发", "regex": False})]},
        {"tool_calls": [_tc("c3", "read_file",
                            {"path": "docs/agentscope.md", "offset": 1, "limit": 2})]},
        {"text": "按 report-writer 技能流程：AgentScope 支持多智能体分布式编排与"
                 "高并发问答（docs/agentscope.md:2）。", "tool_calls": []},
    ])

    agent = ToolAgent(
        llm=llm, registry=registry, executor=executor, max_iters=6,
        roots_resolver=lambda ctx: [str(kb_root)],   # 只给知识库根，技能根由 agent 自动并入
        skill_registry=skill_registry,
    )

    # 4) 展示注入到 system prompt 的渐进披露清单
    preview_ctx = ToolContext(tenant_id="default", roots=[str(kb_root)])
    system_prompt = agent._system_prompt(preview_ctx)
    print("\n== System Prompt 中的技能披露段 ==")
    for line in system_prompt.splitlines():
        if "技能" in line or "SKILL.md" in line or line.startswith("- "):
            print(line)

    # 5) 运行 agentic 循环
    ctx = AgentContext(tenant_id="default", session_id="skills-demo", kb_ids=["general"])
    question = "请依据 report-writer 技能，介绍 AgentScope 的能力。"
    print(f"\n问：{question}\n")
    result = await agent.run(question, ctx)

    print("== 工具调用轨迹 ==")
    for i, step in enumerate(result.transcript, 1):
        args = json.dumps(step["args"], ensure_ascii=False)
        flag = "错误" if step["is_error"] else "成功"
        print(f"[{i}] {step['tool']}({args}) -> {flag}")
        print("     " + " ".join(step["preview"].split())[:120])

    print("\n== 最终答案 ==")
    print(result.answer)
    print(f"\n迭代轮数={result.iterations} 工具调用数={result.tool_calls} "
          f"降级={'是' if result.degraded else '否'} 模型={result.model}")
    print("文件引用：")
    for cite in result.file_citations:
        line = cite.get("line")
        print(f"  - {cite['path']}" + (f":{line}" if line else ""))

    # 6) 越权演示：技能目录虽在 jail 内，但 jail 外绝对路径仍被拒
    print("\n== 越权访问演示 ==")
    escape = await executor.dispatch(
        _toolcall("read_file", {"path": str(skill_root.parent / "secret.env")}),
        ToolContext(tenant_id="default", roots=[str(kb_root), str(skill_root)]),
    )
    print(f"读取 jail 外文件 -> is_error={escape.is_error}")
    print(" ".join(escape.content.split())[:120])


def _toolcall(name: str, args: dict):
    from fusion_rag.llm.base import ToolCall
    return ToolCall(id="demo", name=name, arguments=args)


if __name__ == "__main__":
    asyncio.run(main())
