"""ToolAgent agentic 循环与 ToolExecutor 调度测试。

用 ScriptedToolLLM 回放预设 tool_calls，离线验证「模型请求工具→执行→回填→
再请求→作答」的完整循环机制与文件引用抽取（生产由真实 function-calling 模型驱动）。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from conftest import ScriptedToolLLM, run

from fusion_rag.agents.base import AgentContext
from fusion_rag.agents.tool_agent import ToolAgent
from fusion_rag.llm.base import ToolCall
from fusion_rag.skills.loader import scan_layer
from fusion_rag.skills.registry import SkillRegistry
from fusion_rag.tools.base import ToolContext
from fusion_rag.tools.executor import ToolExecutor
from fusion_rag.tools.file_tools import GrepTool, ReadFileTool
from fusion_rag.tools.plan_tool import PlanStore, PlanUpdateTool
from fusion_rag.tools.registry import ToolRegistry

DOC = (
    "# AgentScope 概览\n"
    "AgentScope 支持多智能体分布式编排与高并发问答。\n"
    "DeepAgents 负责复杂问题分层拆解。\n"
)


def _tc(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _corpus_and_registry() -> tuple[Path, ToolRegistry]:
    root = Path(tempfile.mkdtemp())
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "agentscope.md").write_text(DOC, encoding="utf-8")
    reg = ToolRegistry()
    reg.register(GrepTool())
    reg.register(ReadFileTool())
    reg.register(PlanUpdateTool())
    return root, reg


def _agent(root: Path, script: list) -> tuple[ToolAgent, ScriptedToolLLM]:
    _, reg = _corpus_and_registry()
    llm = ScriptedToolLLM(script)
    executor = ToolExecutor(reg, call_timeout=15)
    agent = ToolAgent(
        llm=llm, registry=reg, executor=executor, max_iters=6,
        roots_resolver=lambda ctx: [str(root)],
    )
    return agent, llm


def _ctx(root: Path) -> AgentContext:
    return AgentContext(tenant_id="default", session_id="s1", kb_ids=["general"])


# ----------------------------------------------------------------------
def test_tool_agent_full_loop():
    root, _ = _corpus_and_registry()
    script = [
        {"tool_calls": [_tc("c1", "grep", {"pattern": "高并发", "regex": False})]},
        {"tool_calls": [_tc("c2", "read_file", {"path": "docs/agentscope.md", "offset": 1, "limit": 2})]},
        {"text": "AgentScope 支持高并发问答（docs/agentscope.md:2）。", "tool_calls": []},
    ]
    agent, llm = _agent(root, script)

    async def scenario():
        return await agent.run("AgentScope 能做什么？", _ctx(root))

    result = run(scenario)
    assert result.answer.startswith("AgentScope 支持高并发问答")
    assert result.iterations == 3
    assert result.tool_calls == 2
    assert result.degraded is False
    assert len(result.transcript) == 2
    assert result.transcript[0]["tool"] == "grep"
    # 文件引用被抽取
    cites = {(c["path"], c["line"]) for c in result.file_citations}
    assert ("docs/agentscope.md", 2) in cites
    # 回填给模型的 tool 结果消息确实进入后续请求
    assert any(m["role"] == "tool" for m in llm.requests[-1])


def test_tool_agent_forces_final_on_iter_cap():
    root, _ = _corpus_and_registry()
    # 前两轮一直要工具、永不直接作答；第三轮作为收尾补全
    script = [
        {"tool_calls": [_tc("c1", "grep", {"pattern": "拆解"})]},
        {"tool_calls": [_tc("c2", "grep", {"pattern": "编排"})]},
        {"text": "（收尾）综合以上：AgentScope 支持分布式编排。", "tool_calls": []},
    ]
    _, reg = _corpus_and_registry()
    llm = ScriptedToolLLM(script)
    agent = ToolAgent(
        llm=llm, registry=reg, executor=ToolExecutor(reg, call_timeout=15),
        max_iters=2, roots_resolver=lambda ctx: [str(root)],
    )

    async def scenario():
        return await agent.run("问题？", _ctx(root))

    result = run(scenario)
    assert result.iterations == 2
    assert result.degraded is True          # 耗尽轮数触发强制收尾
    assert "收尾" in result.answer


def test_tool_agent_no_roots_degrades():
    _, reg = _corpus_and_registry()
    llm = ScriptedToolLLM([{"text": "x"}])
    agent = ToolAgent(
        llm=llm, registry=reg, executor=ToolExecutor(reg),
        roots_resolver=lambda ctx: [],
    )

    async def scenario():
        return await agent.run("问题？", _ctx(Path(tempfile.mkdtemp())))

    result = run(scenario)
    assert result.degraded is True and result.answer == ""


# ----------------------------------------------------------------------
def test_executor_dispatch_success():
    root, reg = _corpus_and_registry()
    executor = ToolExecutor(reg, call_timeout=15)
    call = ToolCall(id="c1", name="grep", arguments={"pattern": "高并发", "regex": False})
    tool_ctx = ToolContext(tenant_id="default", kb_ids=["general"], roots=[str(root)])

    async def scenario():
        return await executor.dispatch(call, tool_ctx)

    result = run(scenario)
    assert not result.is_error
    assert "agentscope.md" in result.content


def test_executor_unknown_tool():
    _, reg = _corpus_and_registry()
    executor = ToolExecutor(reg)
    call = ToolCall(id="x", name="no_such", arguments={})
    tool_ctx = ToolContext(roots=["/tmp"])

    async def scenario():
        return await executor.dispatch(call, tool_ctx)

    result = run(scenario)
    assert result.is_error and "未知工具" in result.content


def test_executor_isolates_jail_violation():
    root, reg = _corpus_and_registry()
    executor = ToolExecutor(reg, call_timeout=15)
    call = ToolCall(id="x", name="read_file", arguments={"path": "/etc/passwd"})
    tool_ctx = ToolContext(tenant_id="t", kb_ids=["general"], roots=[str(root)])

    async def scenario():
        return await executor.dispatch(call, tool_ctx)

    result = run(scenario)
    # 越权不抛异常，以错误结果回填给模型
    assert result.is_error


# ----------------------------------------------------------------------
# Skills 渐进式披露集成
# ----------------------------------------------------------------------
def _skill_layer() -> tuple[Path, SkillRegistry]:
    """造一个技能层目录：demo-skill/SKILL.md（含可被 read_file 读到的正文）。"""
    skill_root = Path(tempfile.mkdtemp())
    d = skill_root / "demo-skill"
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: 演示技能的完整流程。\n---\n\n"
        "# 步骤\n先 grep 再 read_file。\n", encoding="utf-8",
    )
    reg = SkillRegistry(scan_layer(skill_root, source="base"), roots=[str(skill_root)])
    return skill_root, reg


def test_tool_agent_injects_skills_into_system_prompt():
    root, _ = _corpus_and_registry()
    _, skill_reg = _skill_layer()
    _, reg = _corpus_and_registry()
    agent = ToolAgent(
        llm=ScriptedToolLLM([{"text": "x"}]), registry=reg,
        executor=ToolExecutor(reg), roots_resolver=lambda ctx: [str(root)],
        skill_registry=skill_reg,
    )
    tool_ctx = ToolContext(tenant_id="default", roots=[str(root)])
    prompt = agent._system_prompt(tool_ctx)
    assert "可用技能" in prompt
    assert "demo-skill" in prompt
    assert "demo-skill/SKILL.md" in prompt


def test_tool_agent_skill_roots_merged_into_jail():
    """roots_resolver 只给知识库根，但技能根应被自动并入，read_file 才读得到 SKILL.md。"""
    kb_root, _ = _corpus_and_registry()
    skill_root, skill_reg = _skill_layer()
    _, reg = _corpus_and_registry()
    llm = ScriptedToolLLM([
        {"tool_calls": [_tc("c1", "read_file", {"path": "demo-skill/SKILL.md", "offset": 1, "limit": 20})]},
        {"text": "已按技能流程作答（demo-skill/SKILL.md:1）。", "tool_calls": []},
    ])
    agent = ToolAgent(
        llm=llm, registry=reg, executor=ToolExecutor(reg, call_timeout=15),
        max_iters=4,
        # 故意只返回知识库根，验证技能根由 _with_skill_roots 自动并入
        roots_resolver=lambda ctx: [str(kb_root)],
        skill_registry=skill_reg,
    )

    async def scenario():
        return await agent.run("按演示技能处理", AgentContext(tenant_id="default", session_id="s1"))

    result = run(scenario)
    assert result.answer.startswith("已按技能流程作答")
    # 读 SKILL.md 未越权（说明技能根已进 jail）
    assert result.transcript[0]["is_error"] is False
    assert any("步骤" in (r.get("preview") or "") for r in [result.transcript[0]])


def test_tool_agent_without_skill_registry_unchanged():
    """非破坏：不传 skill_registry 时系统提示词不含技能段。"""
    root, _ = _corpus_and_registry()
    agent, _ = _agent(root, [{"text": "x"}])
    prompt = agent._system_prompt(ToolContext(tenant_id="default", roots=[str(root)]))
    assert "可用技能" not in prompt


def test_plan_store_lifecycle():
    """PlanStore：写/merge 字段级更新/状态机/all_resolved/render。"""
    store = PlanStore()
    assert store.is_empty() and not store.all_resolved()
    store.write([
        {"id": "1", "content": "核实前提A", "status": "pending"},
        {"id": "2", "content": "诉求B", "status": "pending"},
    ])
    assert store.counts()["total"] == 2 and not store.all_resolved()
    # merge 只给 status，content 不被清空
    store.write(
        [{"id": "1", "status": "completed"}, {"id": "2", "status": "in_progress"}],
        merge=True,
    )
    items = {it["id"]: it for it in store.read()}
    assert items["1"]["status"] == "completed"
    assert items["1"]["content"] == "核实前提A"
    assert not store.all_resolved()  # item2 仍 in_progress
    # 迭代中修订清单：重写某项 content + merge 新增暴露出的新 id 项
    store.write(
        [
            {"id": "2", "content": "修正后的诉求B（前提已改为X系列）"},
            {"id": "3", "content": "新查子问题C", "status": "pending"},
        ],
        merge=True,
    )
    items = {it["id"]: it for it in store.read()}
    assert items["2"]["content"].startswith("修正后的诉求B")
    assert items["2"]["status"] == "in_progress"  # 未给 status 保持原值
    assert items["3"]["status"] == "pending"
    assert store.counts()["total"] == 3
    # 勾完 + 作废一项 → 全 resolved
    store.write(
        [{"id": "2", "status": "completed"}, {"id": "3", "status": "cancelled"}],
        merge=True,
    )
    assert store.all_resolved()
    assert "[x] #1" in store.render() and "[~] #3" in store.render()


def test_plan_update_tool_read_write():
    """plan_update 工具：写→返回全量+summary；省略 todos→读取；store=None → 报错不抛。"""
    store = PlanStore()
    ctx = ToolContext(tenant_id="default", roots=[], plan_store=store)
    tool = PlanUpdateTool()

    async def scenario():
        w = await tool.run(
            {"todos": [{"id": "1", "content": "查X", "status": "pending"}]}, ctx,
        )
        r = await tool.run({}, ctx)
        return w, r

    w, r = run(scenario)
    assert w.is_error is False
    data = json.loads(w.content)
    assert data["summary"]["total"] == 1 and data["all_resolved"] is False
    assert json.loads(r.content)["todos"][0]["content"] == "查X"

    async def scenario_none():
        return await tool.run({}, ToolContext())

    assert run(scenario_none).is_error is True


def _user_msgs(request, keyword):
    return [
        m["content"] for m in request
        if m["role"] == "user" and keyword in m["content"]
    ]


def test_tool_agent_plan_is_advisory_no_gate():
    """问题分解是导航不是任务表：建了分解后，即使子问题没全勾完，模型也能
    直接作答（不设完成门）；为控制端到端时长，末轮才把分解回显一次。
    """
    root, reg = _corpus_and_registry()
    script = [
        # 建分解：两项 pending
        {"tool_calls": [_tc("p1", "plan_update", {"todos": [
            {"id": "1", "content": "核实前提A", "status": "pending"},
            {"id": "2", "content": "诉求B", "status": "pending"},
        ]})]},
        # 分解仍 0/2（未勾完），模型直接给答案 → 应被接受，不再拦
        {"text": "FINAL 基于A与B（docs/agentscope.md:2）。", "tool_calls": []},
    ]
    llm = ScriptedToolLLM(script)
    executor = ToolExecutor(reg, call_timeout=15)
    agent = ToolAgent(
        llm=llm, registry=reg, executor=executor, max_iters=2, plan_first=True,
        roots_resolver=lambda ctx: [str(root)],
    )

    async def scenario():
        return await agent.run("高并发怎么优化？", _ctx(root))

    result = run(scenario)
    # 末轮（iter#2）将拆解作为“问题分解·参考”回显一次；非末轮不再注入
    assert _user_msgs(llm.requests[1], "问题分解·参考")
    # 未勾完也不拦：答案直接返回，只如实记录还剩 2 项没勾
    assert result.answer.startswith("FINAL")
    assert result.plan_unfinished == 2
    assert "[ ] #1" in result.plan
    assert result.iterations == 2  # 没有额外的 reconcile 轮
    assert result.degraded is False


def test_tool_agent_no_gate_when_plan_absent():
    """plan_first=False：不建清单、不注入、不拦，行为与普通 agentic 一致。"""
    root, reg = _corpus_and_registry()
    script = [
        {"text": "直接答案（docs/agentscope.md:2）。", "tool_calls": []},
    ]
    llm = ScriptedToolLLM(script)
    executor = ToolExecutor(reg, call_timeout=15)
    agent = ToolAgent(
        llm=llm, registry=reg, executor=executor, max_iters=6, plan_first=False,
        roots_resolver=lambda ctx: [str(root)],
    )

    async def scenario():
        return await agent.run("简单问题？", _ctx(root))

    result = run(scenario)
    assert result.answer.startswith("直接答案")
    assert result.plan == "" and result.plan_unfinished == 0
    assert not any(_user_msgs(req, "问题分解") for req in llm.requests)


def test_tool_agent_bootstraps_plan_when_absent():
    """为控制端到端时长，旧的“首轮推一把建拆解” bootstrap 提示已移除；
    plan_first=True 但模型不自发建分解时，也不拦作答，行为与不启用分解一致。"""
    root, reg = _corpus_and_registry()
    script = [
        {"tool_calls": [_tc("g1", "grep", {"pattern": "高并发", "regex": False})]},
        {"text": "FINAL 答案（docs/agentscope.md:2）。", "tool_calls": []},
    ]
    llm = ScriptedToolLLM(script)
    executor = ToolExecutor(reg, call_timeout=15)
    agent = ToolAgent(
        llm=llm, registry=reg, executor=executor, max_iters=6, plan_first=True,
        roots_resolver=lambda ctx: [str(root)],
    )

    async def scenario():
        return await agent.run("高并发怎么优化？", _ctx(root))

    result = run(scenario)
    # 旧行为的“先想清楚要解决什么”已彻底不再注入
    assert not any(_user_msgs(req, "先想清楚要解决什么") for req in llm.requests)
    # 未建分解直答：空分解不拦，答案正常返回
    assert result.answer.startswith("FINAL")
    assert result.plan_unfinished == 0


def test_tool_agent_blocks_duplicate_tool_call():
    """模型重发完全相同的工具调用时被熔断：不重跑，回“勿重复”备忘。"""
    root, reg = _corpus_and_registry()
    same = {"pattern": "高并发", "regex": False}
    script = [
        {"tool_calls": [_tc("g1", "grep", dict(same))]},
        {"tool_calls": [_tc("g2", "grep", dict(same))]},
        {"text": "FINAL（docs/agentscope.md:2）。", "tool_calls": []},
    ]
    llm = ScriptedToolLLM(script)
    executor = ToolExecutor(reg, call_timeout=15)
    agent = ToolAgent(
        llm=llm, registry=reg, executor=executor, max_iters=6,
        roots_resolver=lambda ctx: [str(root)],
    )

    async def scenario():
        return await agent.run("高并发在哪？", _ctx(root))

    result = run(scenario)
    greps = [e for e in result.evidence if e["tool"] == "grep"]
    # 两次 grep 均回包，但第二次是熔断备忘而非真实检索结果
    assert any("完全相同的调用" in e["content"] for e in greps)
    assert result.answer.startswith("FINAL")


def test_seed_direct_escapes_when_unsatisfied():
    """①+③ 逃生阀：策略A 直答低置信/“资料未提及”→不收敛，回退工具深挖。"""
    root, reg = _corpus_and_registry()
    script = [
        # iter#1（策略A 直答）：低置信 + 对冲话术 → 应被逃生阀拦下继续挖
        {"text": "扩展模块数量未在提供的资料中明确说明。 [CONFIDENCE:0.5]",
         "tool_calls": []},
        # 回退后真检索到依据，高置信作答
        {"text": "H5U 最多可扩展 8 个模块（docs/agentscope.md:2）。 [CONFIDENCE:0.9]",
         "tool_calls": []},
    ]
    llm = ScriptedToolLLM(script)
    agent = ToolAgent(
        llm=llm, registry=reg, executor=ToolExecutor(reg, call_timeout=15),
        max_iters=6, seed_direct_answer=True, seed_direct_min_conf=0.7,
        roots_resolver=lambda ctx: [str(root)],
    )

    async def scenario():
        return await agent.run(
            "H5U可以带多少个扩展模块", _ctx(root),
            seed_evidence="初步线索 @docs/agentscope.md:2",
        )

    result = run(scenario)
    # 首轮不该收敛：至少跑了 2 轮，且最终采纳高置信的“8 个模块”
    assert result.iterations >= 2
    assert len(llm.requests) >= 2
    assert "8 个模块" in result.answer
    assert (result.self_confidence or 0) >= 0.9


def test_seed_direct_fast_path_when_confident():
    """① 保留快路径：策略A 直答高置信、无对冲 → 仍 1 轮收敛（不多花一轮）。"""
    root, reg = _corpus_and_registry()
    script = [
        {"text": "答案已清楚：支持高并发问答（docs/agentscope.md:2）。 [CONFIDENCE:0.9]",
         "tool_calls": []},
    ]
    llm = ScriptedToolLLM(script)
    agent = ToolAgent(
        llm=llm, registry=reg, executor=ToolExecutor(reg, call_timeout=15),
        max_iters=6, seed_direct_answer=True, seed_direct_min_conf=0.7,
        roots_resolver=lambda ctx: [str(root)],
    )

    async def scenario():
        return await agent.run(
            "AgentScope 支持什么", _ctx(root),
            seed_evidence="初步线索 @docs/agentscope.md:2",
        )

    result = run(scenario)
    # 高置信直答不应触发逃生阀：一轮即收敛
    assert result.iterations == 1
    assert len(llm.requests) == 1
    assert result.answer.startswith("答案已清楚")
