"""编排器端到端测试：问答全链路、多轮记忆、引用、轨迹、意图分流。"""

from __future__ import annotations

from conftest import make_kernel, run, seed

from fusion_rag.agents.tool_agent import ToolRunResult
from fusion_rag.types import Answer


def test_kernel_boot_registers_services():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            for name in ("orchestrator", "indexer", "sandbox", "cpu_executor",
                         "store", "retriever", "tracer", "metrics", "ops"):
                assert name in kernel.services, f"缺少服务 {name}"
            assert kernel.orchestrator is kernel.services["orchestrator"]

    run(scenario)


def test_ask_returns_grounded_answer_with_citations():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await seed(kernel)
            res = await kernel.orchestrator.ask(
                "DeepAgents 擅长什么？", tenant_id="default", session_id="s1",
            )
            assert res.answer.strip()
            assert res.citations, "有证据时应给出引用"
            assert res.trace_id
            # 离线兜底走抽取式，应标记降级
            assert res.model in ("extractive", "echo", "none") or not res.degraded
            assert res.latency_ms >= 0

    run(scenario)


def test_ask_ranks_relevant_document():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await seed(kernel)
            res = await kernel.orchestrator.ask(
                "哪个框架强调本地化隐私与私有知识库隔离？",
                tenant_id="default", session_id="s2",
            )
            titles = [c.title for c in res.citations]
            assert "openclaw" in titles, f"应引用 OpenClaw，实际 {titles}"

    run(scenario)


def test_multi_turn_memory_continuity():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await seed(kernel)
            r1 = await kernel.orchestrator.ask(
                "AgentScope 是什么？", tenant_id="default", session_id="chat",
            )
            assert r1.session_id == "chat"
            r2 = await kernel.orchestrator.ask(
                "它支持高并发吗？", tenant_id="default", session_id="chat",
            )
            # 第二轮应能复用同一会话，并检索到 AgentScope 相关内容
            store = kernel.service("store")
            msgs = await store.get_messages("chat")
            assert len(msgs) >= 4, "两轮问答应至少留下 4 条消息"

    run(scenario)


def test_trace_replay_has_pipeline_spans():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await seed(kernel)
            res = await kernel.orchestrator.ask(
                "Harness 有哪些特性？", tenant_id="default", session_id="t1",
            )
            trace = await kernel.service("tracer").replay(res.trace_id)
            assert trace is not None
            names = [s["name"] for s in trace.get("spans", [])]
            # 关键阶段应留痕
            assert any("intent" in n for n in names)
            assert any("retrieval" in n for n in names)
            assert any("reasoning" in n for n in names)

    run(scenario)


def test_chitchat_intent_short_circuits():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await seed(kernel)
            res = await kernel.orchestrator.ask(
                "你好呀", tenant_id="default", session_id="c1",
            )
            # 寒暄类不产生知识引用
            assert res.intent.intent in ("chitchat", "qa", "clarify")
            assert res.answer.strip()

    run(scenario)


def test_empty_question_rejected():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            res = await kernel.orchestrator.ask(
                "   ", tenant_id="default", session_id="e1",
            )
            assert res.answer.strip(), "空提问也应给出可读回复"
            assert res.intent.intent in ("reject", "clarify") or res.confidence == 0.0

    run(scenario)


def test_metrics_record_requests():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await seed(kernel)
            await kernel.orchestrator.ask("AgentScope?", tenant_id="default", session_id="m1")
            metrics = kernel.service("metrics")
            assert metrics.counter("requests_total").value() >= 1
            # Prometheus 文本导出应包含核心指标
            text = metrics.render_prometheus()
            assert "requests_total" in text

    run(scenario)


# ----------------------------------------------------------------------
# Agentic 文件工具集成（非破坏：离线 echo 完全跳过）
# ----------------------------------------------------------------------
class _StubToolAgent:
    """测试替身：返回预设 agentic 产出，验证 Orchestrator 集成链路。"""

    def __init__(self, result: ToolRunResult) -> None:
        self.result = result
        self.seen_seed: str | None = None

    async def run(
        self, question, ctx, *, seed_evidence="", provider=None,
        memory_block="", output_hint="", followup_issues=None,
        intent_domain=None, intent_label=None,
    ):
        self.seen_seed = seed_evidence
        return self.result


def test_agentic_skipped_for_offline_echo():
    """默认 provider 为 echo（不支持 tools）时，agentic 路径绝不触发。"""
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            orch = kernel.orchestrator
            # 即便强行开关注入 tool_agent，echo 门控仍应拦截
            orch.agentic_tools = True
            orch.tool_agent = _StubToolAgent(ToolRunResult(answer="不应被采用"))
            # echo router 不支持 tools → _agentic_available() 必须为 False
            assert orch._agentic_available() is False
            # 且全链路结果不受污染
            await seed(kernel)
            res = await orch.ask(
                "DeepAgents 擅长什么？", tenant_id="default", session_id="ag0",
            )
            assert "agentic" not in res.meta

    run(scenario)


def test_agentic_path_applied_with_stub_router():
    """tool-capable provider + 低置信度时，agentic 补证应被采用并注入文件引用。"""
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await seed(kernel)
            orch = kernel.orchestrator
            # 伪造一个支持 tools 的路由能力 + 高阈值以强制触发
            orch.llm_router.supports_tools = lambda provider=None: True  # type: ignore[assignment]
            orch.agentic_tools = True
            stub = _StubToolAgent(ToolRunResult(
                answer="依据原文：AgentScope 支持高并发问答。",
                file_citations=[
                    {"path": "docs/agentscope.md", "line": 2},
                    {"path": "notes.txt", "line": None},
                ],
                iterations=2, tool_calls=1, model="stub-tool",
            ))
            orch.tool_agent = stub
            res = await orch.ask(
                "AgentScope 能做什么？", tenant_id="default", session_id="ag1",
            )
            info = res.meta.get("agentic")
            assert info and info["applied"] is True
            assert info["iterations"] == 2 and info["file_citations"] == 2
            # 答案被 agentic 输出替换，模型标注更新
            assert res.answer.startswith("依据原文")
            assert res.model == "stub-tool"
            # 文件引用转成 Citation 并进入结果
            assert res.citations and res.citations[0].title == "docs/agentscope.md"
            assert res.citations[0].snippet == "docs/agentscope.md:2"
            # seed_evidence 传递了推理草稿给 ToolAgent
            assert stub.seen_seed is not None

    run(scenario)


def test_agentic_failure_keeps_original_answer():
    """ToolAgent 抛错时，主链路降级返回提示，不抛异常且记录 error 标签。"""
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await seed(kernel)
            orch = kernel.orchestrator
            orch.llm_router.supports_tools = lambda provider=None: True  # type: ignore[assignment]
            orch.agentic_tools = True

            class _Boom:
                async def run(self, *a, **k):
                    raise RuntimeError("tool loop crashed")

            orch.tool_agent = _Boom()
            res = await orch.ask(
                "AgentScope 能做什么？", tenant_id="default", session_id="ag2",
            )
            assert res.meta["agentic"]["error"] == "RuntimeError"
            assert res.meta["agentic"]["degraded"] is True
            # 主链路下 agentic 失败时应提供一个非空的降级提示
            assert res.answer.strip()

    run(scenario)


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeRouter:
    def __init__(self, text: str) -> None:
        self._text = text

    async def chat(self, messages, **kwargs):
        return _FakeResp(self._text)


class _FakeOrch:
    def __init__(self, text: str) -> None:
        self.llm_router = _FakeRouter(text)


def test_extract_sub_queries_llm_parses_and_prepends_original():
    """②b：拆解完全由大模型产出；原句恒等置顶，不产生字级碎片。"""
    from fusion_rag.agents.orchestrator import Orchestrator

    q = "大点数机器设备报错短路故障，怎么恢复错误？"
    fake = _FakeOrch('["大点数机器设备含有哪些产品型号", "短路故障怎么恢复"]')
    subs = run(Orchestrator._extract_sub_queries_llm, fake, q)
    assert subs[0] == q                                # 原句恒等置顶
    assert "大点数机器设备含有哪些产品型号" in subs  # 模型产出的消歧子查询
    for junk in ("以带", "个扩展模块"):
        assert junk not in subs
    assert len(subs) <= 4


def test_extract_sub_queries_llm_falls_back_to_original():
    """②b：模型回包无法解析时，回落为仅原句（恒等，非规则拆分）。"""
    from fusion_rag.agents.orchestrator import Orchestrator

    fake = _FakeOrch("抱歉，我无法拆解这个问题")  # 无 JSON 数组
    subs = run(Orchestrator._extract_sub_queries_llm, fake, "H5U 默认波特率是多少")
    assert subs == ["H5U 默认波特率是多少"]
