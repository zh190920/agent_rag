"""FanoutCoordinator：多候选手册并发子 Agent 检索 + 满足即停止广播。

借鉴范围（均为本仓库已调研过的成熟框架，非自造轮子）：
- ``asyncio.gather(*tasks, return_exceptions=True)`` 扇出：对标
  agentscope ``rag/_knowledge.py`` / ``workspace/_local_workspace.py``。
- 进程内瞬时广播（"错过就错过"）式的"已解决"信号：对标 agentscope
  ``app/message_bus/_base.py`` Mode D publish/subscribe 语义——单请求、单
  事件循环内不需要持久化/重放，用 ``asyncio.Event`` 即可。
- 子 Agent 只上交"干净摘要"而非完整对话历史：对标 deepagents
  ``middleware/subagents.py`` ``atask``——每个 Worker 的
  :class:`ToolRunResult` 已自带 ``evidence`` 摘要，Coordinator 侧只消费
  摘要拼接，不把 Worker 的 messages 细节倒回主链，控制上下文成本。
- 循环边界处轮询检查共享停止标志（而非硬杀 Task）：对标 hermes-agent
  ``acp_adapter/session.py`` / ``gateway/platforms/base.py`` 的
  ``cancel_event``/``interrupt_event`` 惯用法；本场景 Worker 是一次性检索
  任务（对标 deepseek-harness ``subagent`` 的 one-shot 模式），不需要
  "中断当前轮但保留队列可续跑"的复杂语义，直接轮询停止即可。

作用域隔离：PathJail 本身仍是"目录级"安全沙箱（多 Worker 共用同一个授权
根目录，边界不需要按手册拆分）；"单 Worker 只允许看一本手册"是任务作用域
而非安全作用域，通过 :attr:`ToolContext.restrict_to_sources` 单独传给工具
层做软过滤（见 ``fusion_rag/tools/file_tools.py`` / ``search_tool.py``），
不改动 PathJail 语义。
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable

from ..constants import INOVANCE_PERSONA
from ..core.logging import get_logger
from ..llm.base import Message
from .base import AgentContext
from .tool_agent import ToolAgent, ToolRunResult

logger = get_logger(__name__)


# ----------------------------------------------------------------------
class SolvedSignal:
    """并发 Worker 共享的"已解决"广播标志（先到先得）。

    单线程事件循环下"先检查后写入"天然原子，不需要额外加锁；首个调用
    :meth:`mark_solved` 的 Worker 写入结果并置位，其余 Worker 在下一轮
    迭代边界处看到 :attr:`is_stopped` 为真即提前收尾（软停止，不硬杀
    Task，保留已经拿到的中间证据用于兜底合并）。
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self._event = asyncio.Event()
        self.winner_id: str | None = None
        self.winner_result: ToolRunResult | None = None

    @property
    def is_stopped(self) -> bool:
        return self._event.is_set()

    def mark_solved(self, worker_id: str, result: ToolRunResult) -> bool:
        """首个达标 Worker 写入并广播；后续调用返回 False（先到先得）。"""
        if self._event.is_set():
            return False
        self.winner_id = worker_id
        self.winner_result = result
        self._event.set()
        return True


@dataclass
class ManualWorkerSpec:
    """一个并发子 Agent 的任务卡：只负责检索一本候选手册。"""

    worker_id: str
    manual_source: str
    seed_text: str = ""


@dataclass
class FanoutResult:
    """FanoutCoordinator 的产出，供 Orchestrator 直接接入现有后处理链。"""

    answer: str
    file_citations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    self_confidence: float | None = None
    intent_hint: str | None = None
    domain_hint: str | None = None
    risk_hint: str | None = None
    #: "first_solved"=有Worker达标直接采用；"merged"=无人达标走合成；
    #:   "best_effort"=合成也没产出答案，退回最高分Worker原始答案
    mode: str = ""
    worker_count: int = 0
    winner_id: str | None = None
    #: 各 Worker 的迭代/工具调用汇总（供 info 展示，不重复存整个 transcript）
    worker_stats: list[dict[str, Any]] = field(default_factory=list)


# ----------------------------------------------------------------------
class FanoutCoordinator:
    """按候选手册并发派发子 Agent 检索，满足即广播停止，无人达标则合成。"""

    def __init__(
        self,
        *,
        tool_agent: ToolAgent | None = None,
        tool_agent_factory: Callable[[ManualWorkerSpec], ToolAgent] | None = None,
        llm: Any | None,
        max_manuals: int = 5,
        confidence_threshold: float = 0.7,
        worker_max_iters: int = 4,
        merge_enabled: bool = True,
    ) -> None:
        #: 生产路径下多个 Worker 共享同一个 ``ToolAgent`` 实例完全安全（单线程
        #: 事件循环，无数据竞争）；测试路径下每个 Worker 需要各自独立的
        #: ``ScriptedToolLLM`` 脚本序列（否则共享脚本指针会被并发轮次错开打乱，
        #: 无法断言“某个 Worker 第 N 轮收敛”），因此额外支持按 Worker 构造独立 Agent。
        if tool_agent_factory is None:
            if tool_agent is None:
                raise ValueError("必须提供 tool_agent 或 tool_agent_factory 之一")
            tool_agent_factory = lambda _spec: tool_agent  # noqa: E731
        self.tool_agent = tool_agent
        self._agent_factory = tool_agent_factory
        self.llm = llm
        self.max_manuals = max(1, int(max_manuals))
        self.confidence_threshold = float(confidence_threshold)
        self.worker_max_iters = max(1, int(worker_max_iters))
        self.merge_enabled = bool(merge_enabled)

    # ------------------------------------------------------------------
    async def run(
        self,
        question: str,
        ctx: AgentContext,
        specs: list[ManualWorkerSpec],
        *,
        provider: str | None = None,
        memory_block: str = "",
        output_hint: str = "",
        intent_domain: str | None = None,
        intent_label: str | None = None,
    ) -> FanoutResult:
        specs = specs[: self.max_manuals]
        signal = SolvedSignal(self.confidence_threshold)
        logger.info(
            "fanout dispatched",
            extra={
                "event": "fanout_dispatched",
                "worker_count": len(specs),
                "manuals": [s.manual_source for s in specs],
            },
        )

        tasks = [
            self._run_one(
                spec, question, ctx, signal,
                provider=provider, memory_block=memory_block,
                output_hint=output_hint,
                intent_domain=intent_domain, intent_label=intent_label,
            )
            for spec in specs
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        pairs: list[tuple[ManualWorkerSpec, ToolRunResult]] = []
        results: list[ToolRunResult] = []
        stats: list[dict[str, Any]] = []
        for spec, entry in zip(specs, gathered):
            if isinstance(entry, BaseException):
                logger.warning(
                    "fanout worker 异常（%s）：%s", spec.worker_id, entry,
                )
                stats.append({
                    "worker_id": spec.worker_id, "manual": spec.manual_source,
                    "error": type(entry).__name__,
                })
                continue
            results.append(entry)
            pairs.append((spec, entry))
            stats.append({
                "worker_id": spec.worker_id,
                "manual": spec.manual_source,
                "iterations": entry.iterations,
                "tool_calls": entry.tool_calls,
                "stopped_early": entry.stopped_early,
                "self_confidence": entry.self_confidence,
                "degraded": entry.degraded,
            })

        winner = signal.winner_result
        if winner is not None and winner.answer:
            logger.info(
                "fanout solved by first confident worker",
                extra={
                    "event": "fanout_solved",
                    "worker_id": signal.winner_id,
                    "confidence": winner.self_confidence,
                },
            )
            return self._build_result(
                winner, mode="first_solved", specs=specs,
                results=results, stats=stats, winner_id=signal.winner_id,
            )

        if self.merge_enabled and self.llm is not None and pairs:
            merged = await self._merge(
                question, pairs,
                provider=provider, output_hint=output_hint,
            )
            if merged is not None:
                return self._build_result(
                    merged, mode="merged", specs=specs,
                    results=results, stats=stats, winner_id=None,
                )

        best = self._pick_best(results)
        if best is not None:
            return self._build_result(
                best, mode="best_effort", specs=specs,
                results=results, stats=stats, winner_id=None,
            )

        return FanoutResult(
            answer="", mode="best_effort", worker_count=len(specs),
            worker_stats=stats,
        )

    # ------------------------------------------------------------------
    async def _run_one(
        self,
        spec: ManualWorkerSpec,
        question: str,
        ctx: AgentContext,
        signal: SolvedSignal,
        *,
        provider: str | None = None,
        memory_block: str = "",
        output_hint: str = "",
        intent_domain: str | None = None,
        intent_label: str | None = None,
    ) -> ToolRunResult:
        worker_ctx = dataclasses.replace(
            ctx,
            extra={**ctx.extra, "restrict_to_sources": {spec.manual_source}},
        )
        agent = self._agent_factory(spec)
        return await agent.run(
            question, worker_ctx,
            seed_evidence=spec.seed_text,
            provider=provider,
            memory_block=memory_block,
            output_hint=output_hint,
            intent_domain=intent_domain,
            intent_label=intent_label,
            stop_signal=signal,
            worker_id=spec.worker_id,
            max_iters=self.worker_max_iters,
        )

    # ------------------------------------------------------------------
    def _pick_best(self, results: list[ToolRunResult]) -> ToolRunResult | None:
        answered = [r for r in results if r.answer]
        if not answered:
            return None
        return max(
            answered,
            key=lambda r: (r.self_confidence if r.self_confidence is not None else 0.0),
        )

    # ------------------------------------------------------------------
    def _build_result(
        self,
        base: ToolRunResult,
        *,
        mode: str,
        specs: list[ManualWorkerSpec],
        results: list[ToolRunResult],
        stats: list[dict[str, Any]],
        winner_id: str | None,
    ) -> FanoutResult:
        merged_citations: list[dict[str, Any]] = []
        seen_cites: set[Any] = set()
        merged_evidence: list[dict[str, Any]] = []
        for r in results:
            for c in r.file_citations:
                key = (c.get("path"), c.get("line"))
                if key in seen_cites:
                    continue
                seen_cites.add(key)
                merged_citations.append(c)
            merged_evidence.extend(r.evidence)
        return FanoutResult(
            answer=base.answer,
            file_citations=merged_citations,
            evidence=merged_evidence,
            model=base.model,
            self_confidence=base.self_confidence,
            intent_hint=base.intent_hint,
            domain_hint=base.domain_hint,
            risk_hint=base.risk_hint,
            mode=mode,
            worker_count=len(specs),
            winner_id=winner_id,
            worker_stats=stats,
        )

    # ------------------------------------------------------------------
    async def _merge(
        self,
        question: str,
        pairs: list[tuple[ManualWorkerSpec, ToolRunResult]],
        *,
        provider: str | None = None,
        output_hint: str = "",
    ) -> ToolRunResult | None:
        """无人达标时的合成调用：把各 Worker 证据摘要拼一次不带工具的 chat。"""
        sections: list[str] = []
        for spec, r in pairs:
            digest = _evidence_digest(r)
            if not digest:
                continue
            sections.append(f"▸《{spec.manual_source}》检索摘要:\n{digest}")
        if not sections:
            return None

        hint_line = f"\n{output_hint}" if output_hint else ""
        messages = [
            Message.system(
                INOVANCE_PERSONA + "\n"
                "你是知识库问答的合成环节。下面是同一问题在几本不同手册里分别检索到的"
                "证据摘要（可能不完整、也可能互相补充），请只依据这些证据综合给出最终"
                "答案，不要编造摘要里没有的内容。答案末尾追加 "
                "[CONFIDENCE:x.xx][INTENT:...][DOMAIN:...][RISK:...]。"
            ),
            Message.user(
                f"问题：{question}\n\n" + "\n\n".join(sections) + hint_line,
            ),
        ]
        try:
            resp = await self.llm.chat(messages, provider=provider, temperature=0.1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fanout 合成调用失败（退回 best_effort）：%s", exc)
            return None

        from .tool_agent import _parse_self_report

        cleaned, conf, intent, domain, risk = _parse_self_report((resp.text or "").strip())
        if not cleaned:
            return None
        logger.info(
            "fanout merged",
            extra={
                "event": "fanout_merged",
                "worker_count": len(pairs),
                "contributing_sections": len(sections),
                "confidence": conf,
            },
        )
        merged_result = ToolRunResult(
            answer=cleaned, model=resp.model or "",
            self_confidence=conf, intent_hint=intent,
            domain_hint=domain, risk_hint=risk,
        )
        return merged_result


# ----------------------------------------------------------------------
def _evidence_digest(result: ToolRunResult, max_items: int = 3, max_chars: int = 800) -> str:
    """从 Worker 已收集的工具证据里拼一段有界的摘要文本（供合成调用输入）。"""
    items = result.evidence[-max_items:] if result.evidence else []
    lines: list[str] = []
    for e in items:
        content = (e.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"[{e.get('tool', '?')}] {content[:max_chars]}")
    if not lines and result.answer:
        lines.append(result.answer[:max_chars])
    return "\n".join(lines)
