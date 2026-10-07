"""FanoutCoordinator：多手册并发子 Agent 检索 + 满足即停止广播。

用自建的可控脚本 LLM（按实例隔离脚本序列）离线验证：
- ``SolvedSignal`` 先到先得语义；
- 首个高置信度 Worker 达标后，其余 Worker 在迭代边界优雅提前收尾
  （``stopped_early=True``，且实际 LLM 调用数远小于 ``worker_max_iters``）；
- 无人达标时走一次不带工具的合成调用（``mode="merged"``）；
- ``restrict_to_sources`` 任务作用域软过滤不影响 PathJail 安全边界本身；
- ``Orchestrator._group_candidates_by_source`` / ``_fanout_ready`` 的开关
  门控（单候选/未启用时零回归走原单 Agent 路径）。
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from conftest import ScriptedToolLLM, make_kernel, run

from fusion_rag.agents.base import AgentContext
from fusion_rag.agents.fanout import (
    FanoutCoordinator,
    ManualWorkerSpec,
    SolvedSignal,
)
from fusion_rag.agents.orchestrator import Orchestrator
from fusion_rag.agents.tool_agent import ToolAgent, ToolRunResult
from fusion_rag.llm.base import LLMResponse, parse_tool_calls
from fusion_rag.tools.base import ToolContext
from fusion_rag.tools.executor import ToolExecutor
from fusion_rag.tools.file_tools import (
    _SCOPE_DENY_MSG,
    GrepTool,
    ReadFileTool,
)
from fusion_rag.tools.registry import ToolRegistry


# ----------------------------------------------------------------------
# 测试替身
# ----------------------------------------------------------------------
class GateScriptedLLM(ScriptedToolLLM):
    """在 ``ScriptedToolLLM`` 基础上加两个并发编排钩子，专用于扇出/停止测试。

    - ``wait_gate``: 每次 ``chat()`` 前先等待该事件（模拟"慢速 Worker"，让
      测试代码可以精确控制"谁先收敛、谁后看到停止信号"的时序，而不是依赖
      真实线程池完成顺序——这是 asyncio 单线程协作式调度下唯一可靠的定序方式）。
    - ``set_gate_after_last``: 返回脚本里最后一条内容（即最终收敛答案）之前，
      先把该事件置位（模拟"本 Worker 已经收敛出答案，可以放行其他被
      ``wait_gate`` 卡住的 Worker 继续跑到下一个迭代边界去检查停止信号"）。
    """

    def __init__(
        self, script=None, *, wait_gate: asyncio.Event | None = None,
        set_gate_after_last: asyncio.Event | None = None,
    ) -> None:
        super().__init__(script)
        self._wait_gate = wait_gate
        self._set_gate_after_last = set_gate_after_last

    async def chat(self, messages, **kwargs) -> LLMResponse:
        if self._wait_gate is not None:
            await self._wait_gate.wait()
        resp = await super().chat(messages, **kwargs)
        if (
            self._set_gate_after_last is not None
            and self._i >= len(self._script)
        ):
            self._set_gate_after_last.set()
        return resp


class RecordingMergeLLM:
    """合成调用专用替身：记录每次 ``chat()`` 的入参消息，返回固定合并答案。"""

    def __init__(self, text: str = "合并后的最终答案 [CONFIDENCE:0.8]") -> None:
        self.text = text
        self.model = "merge-stub"
        self.calls: list[list] = []

    async def chat(self, messages, **kwargs) -> SimpleNamespace:
        self.calls.append(list(messages))
        return SimpleNamespace(text=self.text, model=self.model)


def _tc(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


# ----------------------------------------------------------------------
# 测试语料：三本手册，内容互不重叠，方便按 basename 区分作用域
# ----------------------------------------------------------------------
def _corpus() -> tuple[Path, ToolRegistry]:
    root = Path(tempfile.mkdtemp())
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "manualA.md").write_text(
        "# 手册A\n这是手册A里关于并发检索的唯一线索描述。\n", encoding="utf-8",
    )
    (docs / "manualB.md").write_text(
        "# 手册B\n这是手册B里关于任务规划的说明文字。\n", encoding="utf-8",
    )
    (docs / "manualC.md").write_text(
        "# 手册C\n这是手册C里关于证据合成的说明文字。\n", encoding="utf-8",
    )
    reg = ToolRegistry()
    reg.register(GrepTool())
    reg.register(ReadFileTool())
    return root, reg


def _ctx(root: Path) -> AgentContext:
    return AgentContext(tenant_id="default", session_id="s1", kb_ids=["general"])


def _make_agent(root: Path, reg: ToolRegistry, llm: ScriptedToolLLM) -> ToolAgent:
    return ToolAgent(
        llm=llm, registry=reg, executor=ToolExecutor(reg, call_timeout=15),
        roots_resolver=lambda _c: [str(root)],
    )


# ----------------------------------------------------------------------
def test_solved_signal_first_writer_wins():
    """SolvedSignal 先到先得语义（需在已运行的事件循环里构造，py3.8 下
    ``asyncio.Event()`` 会立即绑定当前 loop，脱离 loop 创建会报错）。"""
    async def scenario():
        signal = SolvedSignal(threshold=0.7)
        assert signal.is_stopped is False

        r1 = ToolRunResult(answer="A", self_confidence=0.9)
        r2 = ToolRunResult(answer="B", self_confidence=0.95)
        assert signal.mark_solved("w1", r1) is True
        assert signal.is_stopped is True
        # 先到先得：第二个调用者拿不到写权，winner 仍是 w1/r1
        assert signal.mark_solved("w2", r2) is False
        assert signal.winner_id == "w1"
        assert signal.winner_result is r1

    run(scenario)


# ----------------------------------------------------------------------
def test_fanout_first_solved_stops_others():
    """Worker-A 第 2 轮达标广播停止；B/C 因迭代边界检查提前收尾，未跑满预算。"""
    root, reg = _corpus()
    worker_max_iters = 4

    llms: dict[str, GateScriptedLLM] = {}

    async def scenario():
        # ★ asyncio.Event 必须在真正运行的那个事件循环里创建（conftest.run 会
        #   新开一个事件循环），否则跨 loop 绑定会报 "attached to a different loop"。
        release = asyncio.Event()
        llms["manualA.md"] = GateScriptedLLM(
            [
                {"tool_calls": [_tc("a1", "grep", {"pattern": "并发检索", "regex": False})]},
                {"text": "手册A给出了并发检索的完整说明。 [CONFIDENCE:0.9]", "tool_calls": []},
            ],
            set_gate_after_last=release,
        )
        # B/C 脚本设计为"若无干扰应跑满 4 轮工具调用才收敛"——用于反证确实在
        # 第 2 轮迭代边界就被停止信号拦下，而不是自然跑完自己的预算。
        bc_script = [
            {"tool_calls": [_tc(f"b{i}", "grep", {"pattern": "说明", "regex": False})]}
            for i in range(worker_max_iters)
        ]
        llms["manualB.md"] = GateScriptedLLM(bc_script, wait_gate=release)
        llms["manualC.md"] = GateScriptedLLM([dict(x) for x in bc_script], wait_gate=release)

        def factory(spec: ManualWorkerSpec) -> ToolAgent:
            return _make_agent(root, reg, llms[spec.manual_source])

        specs = [
            ManualWorkerSpec(worker_id=f"manual:{n}", manual_source=n, seed_text="")
            for n in ("manualA.md", "manualB.md", "manualC.md")
        ]
        coordinator = FanoutCoordinator(
            tool_agent_factory=factory, llm=None,
            confidence_threshold=0.7, worker_max_iters=worker_max_iters,
            merge_enabled=False,
        )
        return await coordinator.run("并发检索怎么做？", _ctx(root), specs)

    result = run(scenario)

    assert result.mode == "first_solved"
    assert result.winner_id == "manual:manualA.md"
    assert result.answer.startswith("手册A")

    stats = {s["worker_id"]: s for s in result.worker_stats}
    # A 正常收敛，未被打断
    assert stats["manual:manualA.md"]["stopped_early"] is False
    # B/C 各只做了 1 次 chat（第 1 轮），第 2 轮在迭代边界就被停止信号拦下
    assert len(llms["manualB.md"].requests) < worker_max_iters
    assert len(llms["manualC.md"].requests) < worker_max_iters
    assert stats["manual:manualB.md"]["stopped_early"] is True
    assert stats["manual:manualC.md"]["stopped_early"] is True


# ----------------------------------------------------------------------
def test_fanout_merge_when_no_winner():
    """三个 Worker 全部低置信度收敛 → 无人达标，走一次不带工具的合成调用。"""
    root, reg = _corpus()
    low_conf_script = [
        {"tool_calls": [_tc("x1", "grep", {"pattern": "说明", "regex": False})]},
        {"text": "只能给出很不确定的一段描述。 [CONFIDENCE:0.2]", "tool_calls": []},
    ]
    llms = {
        n: GateScriptedLLM([dict(x) for x in low_conf_script])
        for n in ("manualA.md", "manualB.md", "manualC.md")
    }

    def factory(spec: ManualWorkerSpec) -> ToolAgent:
        return _make_agent(root, reg, llms[spec.manual_source])

    merge_llm = RecordingMergeLLM("综合三本手册：并发检索 + 任务规划 + 证据合成。 [CONFIDENCE:0.75]")
    specs = [
        ManualWorkerSpec(worker_id=f"manual:{n}", manual_source=n, seed_text="")
        for n in ("manualA.md", "manualB.md", "manualC.md")
    ]
    coordinator = FanoutCoordinator(
        tool_agent_factory=factory, llm=merge_llm,
        confidence_threshold=0.7, worker_max_iters=4, merge_enabled=True,
    )

    result = run(coordinator.run, "这三本手册分别讲什么？", _ctx(root), specs)

    assert result.mode == "merged"
    assert result.winner_id is None
    # 合成用的 LLM（不带工具的单次调用）恰好被调用 1 次
    assert len(merge_llm.calls) == 1
    # 输入内容包含三份 Worker 的手册证据摘要（按 basename 分组标记）
    merged_input = "\n".join(str(m.content) for m in merge_llm.calls[0])
    assert "manualA.md" in merged_input
    assert "manualB.md" in merged_input
    assert "manualC.md" in merged_input
    assert result.answer.startswith("综合三本手册")
    assert result.self_confidence == 0.75


# ----------------------------------------------------------------------
def test_file_tool_scope_restriction():
    """restrict_to_sources 软过滤：read_file 直接拒绝；grep 静默跳过范围外文件（不泄漏内容）。

    grep 对目录型检索采取“跳过且不计入 scanned”策略（避免误报“扫过但没命中”，
    也不暴露范围外文件的存在），与 read_file 的显式拒绝文案不同——两者都是
    合理的“任务作用域”软拒绝形式，只是粒度不同（单文件目标 vs 批量扫描）。
    """
    root, _reg = _corpus()
    docs_dir = root / "docs"
    assert (docs_dir / "manualA.md").exists()

    def ctx_for(allow: set[str] | None) -> ToolContext:
        return ToolContext(
            tenant_id="default", kb_ids=["general"], roots=[str(root)],
            restrict_to_sources=allow,
        )

    async def scenario():
        grep = GrepTool()
        read = ReadFileTool()
        # 范围外：目标内容只在 manualB.md 里，但只允许看 A —— grep 应扫不到，
        # 且不能把 B 的内容/文件名泄到结果里
        grep_denied = await grep.run(
            {"pattern": "任务规划", "regex": False, "path": "docs"}, ctx_for({"manualA.md"}),
        )
        # 对照组：不设作用域时，同一 grep 应能命中 B
        grep_open = await grep.run(
            {"pattern": "任务规划", "regex": False, "path": "docs"}, ctx_for(None),
        )
        # read_file 显式指向范围外文件：应拿到软拒绝文案（非异常、非真实内容）
        read_denied = await read.run({"path": "docs/manualB.md"}, ctx_for({"manualA.md"}))
        # read_file 指向范围内文件：正常返回真实内容
        read_allowed = await read.run({"path": "docs/manualA.md"}, ctx_for({"manualA.md"}))
        return grep_denied, grep_open, read_denied, read_allowed

    grep_denied, grep_open, read_denied, read_allowed = run(scenario)

    # grep 无命中时会回显自己搜的 pattern，这不算泄漏；真正要断言的是：
    # 不能出现范围外文件的行定位/文件名（即真命中的证据形式）
    assert "manualB" not in grep_denied.content
    assert "说明文字" not in grep_denied.content
    assert "manualB.md:" in grep_open.content  # 对照组确认非检索本身失败（能定位到 B 里的行）
    assert read_denied.content == _SCOPE_DENY_MSG
    assert read_denied.is_error is False
    assert "手册A" in read_allowed.content


# ----------------------------------------------------------------------
def _fake_chunk(source: str, score: float, title: str = "《文档》") -> SimpleNamespace:
    return SimpleNamespace(
        source=source, score=score, title=title, content=f"{source} 内容片段",
        identity=(source, 0), chunk=SimpleNamespace(document_id=source, chunk_index=0, metadata={}),
    )


def test_group_candidates_by_source_ranks_by_max_score():
    seed_data = [
        ("子查询1", [
            _fake_chunk("docs/manualA.md", 0.5),
            _fake_chunk("docs/manualB.md", 0.9),
        ]),
        ("子查询2", [
            _fake_chunk("docs/manualA.md", 0.8),
            _fake_chunk("docs/manualC.md", 0.3),
        ]),
    ]
    candidates = Orchestrator._group_candidates_by_source(seed_data, max_manuals=4)
    bases = [base for base, _ in candidates]
    # 按组内最高分排序：B(0.9) > A(max(0.5,0.8)=0.8) > C(0.3)
    assert bases == ["manualB.md", "manualA.md", "manualC.md"]
    # 截断只取前 2 本
    truncated = Orchestrator._group_candidates_by_source(seed_data, max_manuals=2)
    assert [base for base, _ in truncated] == ["manualB.md", "manualA.md"]


# ----------------------------------------------------------------------
def test_fanout_disabled_uses_single_agent():
    """enabled=False 时，即便候选 >=2 本手册，_fanout_ready 也必须返回空（零回归）。"""
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            orch = kernel.orchestrator
            orch.fanout_config = {"enabled": False, "max_manuals": 4}
            orch.tool_agent = SimpleNamespace()  # 非 None，排除"因缺 tool_agent 而关闭"的干扰
            seed_data = [
                ("q", [_fake_chunk("docs/manualA.md", 0.9), _fake_chunk("docs/manualB.md", 0.8)]),
            ]
            assert orch._fanout_ready("问题", _ctx(Path(".")), seed_data) == []

    run(scenario)


def test_fanout_single_candidate_falls_back():
    """enabled=True，但分组后候选只有 1 本手册 → 不扇出，走原单 Agent 路径。"""
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            orch = kernel.orchestrator
            orch.fanout_config = {"enabled": True, "max_manuals": 4}
            orch.tool_agent = SimpleNamespace()
            seed_data = [
                ("q", [_fake_chunk("docs/manualA.md", 0.9), _fake_chunk("docs/manualA.md", 0.7)]),
            ]
            assert orch._fanout_ready("问题", _ctx(Path(".")), seed_data) == []

    run(scenario)
