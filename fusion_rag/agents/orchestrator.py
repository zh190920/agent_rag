"""Orchestrator：多智能体协作调度核心（借鉴 AgentScope 编排 + DeepAgents 规划）。

一次问答的完整链路：

    接入护栏(限流/并发/超时) → 意图识别 → 任务拆解 → 记忆装配
      → 并发检索(多子问题) → 精读推理 → 校验纠错(回炉) → 脱敏
      → 结果组装 → 持久化/事件广播 → 轨迹落盘

设计要点：
- **高并发**：请求入口套 Sandbox（全局并发闸门 + 每租户令牌桶 + 超时预算），
  子问题检索 ``asyncio.gather`` 并发；持久化与统计不阻塞主链路。
- **韧性**：意图/拆解/检索/推理/校验每一步都有降级路径，任一环节失败
  退化为「诚实告知」而非抛错；限流/超时/越权等护栏异常向上抛给 API 层。
- **可观测**：全程 tracer span 埋点 + 指标计数 + 事件广播 + LLM 用量同步。
"""

from __future__ import annotations

import asyncio
import os
import json
import re
import time
from typing import Any

from ..constants import INOVANCE_PERSONA
from ..core.events import Event, EventBus
from ..core.exceptions import (
    AccessDeniedError,
    FusionRagError,
    HITLRequiredError,
    RateLimitError,
    TimeoutBudgetError,
)
from ..core.logging import get_logger
from ..core.quota import set_quota_key
from ..core.sandbox import Sandbox
from ..llm.base import Message
from ..llm.router import LLMRouter
from ..observability.metrics import MetricsRegistry
from ..observability.tracer import Tracer
from ..planning.decomposer import PlanResult, TaskDecomposer
from ..security.access_control import AccessController
from ..security.redaction import Redactor
from ..storage.sqlite_store import SQLiteStore
from ..types import (
    Answer,
    Citation,
    IntentResult,
    OutputFormat,
    Priority,
    QaResult,
    ValidationResult,
    approx_token_count,
    new_id,
)
from .base import AgentContext
from .intent import IntentAgent
from .memory_agent import MemoryAgent
from .reasoning import ReasoningAgent
from .retrieval import RetrievalAgent
from .tool_agent import ToolAgent, set_request_origin
from .validator import ValidatorAgent

logger = get_logger(__name__)


_FALLBACK_MODEL_PREFIXES = ("extractive", "none", "echo")


def _is_fallback_model(name: str | None) -> bool:
    """判定一个 model 名是不是“回退链路”的产物（非真实推理）。

    支持带后缀的复合名（例：echo->extractive），方便下游展示时保留变迁信息。
    """
    n = (name or "").lower()
    head = n.split("->", 1)[0].strip() or n
    return any(head.startswith(p) for p in _FALLBACK_MODEL_PREFIXES)

_REFUSAL = "抱歉，该问题涉及不被允许的内容，我无法提供回答。"
_CLARIFY = "请补充更具体的信息（如所属产品、场景或关键词），以便我为你精准检索知识库。"
_CHITCHAT = "你好，我是企业知识库问答助手，可以就知识库中的文档内容为你答疑。"
_HITL = "该问题涉及高风险/高专业度内容，已转人工审核，请稍后由专家复核答复。"


class Orchestrator:
    """问答编排器（对外主入口）。"""

    def __init__(
        self,
        *,
        intent: IntentAgent,
        decomposer: TaskDecomposer,
        retrieval: RetrievalAgent,
        reasoning: ReasoningAgent,
        validator: ValidatorAgent,
        memory: MemoryAgent,
        access: AccessController,
        redactor: Redactor,
        sandbox: Sandbox,
        store: SQLiteStore,
        tracer: Tracer,
        metrics: MetricsRegistry,
        events: EventBus,
        llm_router: LLMRouter | None = None,
        top_k: int = 6,
        validation_enabled: bool = True,
        max_retry: int = 2,
        redaction_enabled: bool = True,
        hitl_enabled: bool = False,
        tool_agent: ToolAgent | None = None,
        agentic_tools: bool = False,
        retrieval_mode: str = "pipeline",
        agentic_retry: int = 1,
        fanout_config: dict[str, Any] | None = None,
        quota_scope: str = "tenant",
        seed_llm_decompose: bool = True,
    ) -> None:
        self.intent = intent
        self.decomposer = decomposer
        self.retrieval = retrieval
        self.reasoning = reasoning
        self.validator = validator
        self.memory = memory
        self.access = access
        self.redactor = redactor
        self.sandbox = sandbox
        self.store = store
        self.tracer = tracer
        self.metrics = metrics
        self.events = events
        self.llm_router = llm_router
        self.top_k = top_k
        self.validation_enabled = validation_enabled
        self.max_retry = max(0, max_retry)
        self.redaction_enabled = redaction_enabled
        self.hitl_enabled = hitl_enabled
        self.tool_agent = tool_agent
        self.agentic_tools = agentic_tools
        #: "pipeline"=预检索+回炉+可选 agentic（旧行为）；"agentic"=模型驱动全流程
        self.retrieval_mode = (retrieval_mode or "pipeline").lower()
        #: agentic 模式下验证失败后额外重试次数（agentic 单次成本高，默认仅重试 1 次）
        self.agentic_retry = max(0, int(agentic_retry))
        #: ★ 多手册并发子 Agent 检索配置（见 fusion_rag/agents/fanout.py）；
        #:   默认 disabled，行为与非 fanout 旧路径完全一致（零回归）。
        self.fanout_config = dict(fanout_config or {})
        #: ★ 下游并发子配额作域：tenant=跨请求按租户公平 / session=按请求
        #:   （会话隔离）/ none=关闭子配额（仅旧全局闸门）。
        self.quota_scope = (quota_scope or "tenant").lower()
        #: ★ seed 子查询是否走大模型拆解（默认开，见 _extract_sub_queries_llm
        #:   注释：会额外发一次 chat、可能顶破 40s）；关闭时只用原句检索。
        self._seed_llm_decompose = bool(seed_llm_decompose)
        self._last_usage = {"calls": 0, "errors": 0}

    # ==================================================================
    # 对外主入口
    # ==================================================================
    async def ask(
        self,
        question: str,
        *,
        tenant_id: str | None = None,
        session_id: str | None = None,
        user_id: str = "",
        kb_ids: list[str] | None = None,
        priority: Priority | None = None,
        output_format: OutputFormat | str | None = None,
    ) -> QaResult:
        """处理一次知识问答请求。"""
        self.metrics.counter("requests_total").inc()
        tenant = self.access.resolve_tenant(tenant_id)
        resolved_kbs = self.access.resolve_kbs(tenant, kb_ids)
        session_id = session_id or new_id()

        trace = self.tracer.begin(
            session_id=session_id, tenant_id=tenant, question=question or "",
        )
        ctx = AgentContext(
            tenant_id=tenant,
            session_id=session_id,
            user_id=user_id,
            kb_ids=resolved_kbs,
            priority=priority or Priority.NORMAL,
            trace_id=trace.trace_id,
        )
        started = time.monotonic()
        # ★ 记录真实请求起点（perf_counter），写入当前异步上下文；fanout 并发
        #   子任务创建时继承，使答案端到端 TTFT 从本次 ask() 入口起算（全链路）。
        set_request_origin(time.perf_counter())
        # ★ 下游并发子配额键：按作域选租户/会话 id（none → None 关闭）；
        #   写入 ContextVar 后 fanout 子任务与 LLM/embedding 客户端自动继承。
        if self.quota_scope == "session":
            set_quota_key(session_id)
        elif self.quota_scope == "tenant":
            set_quota_key(tenant)
        else:
            set_quota_key(None)
        try:
            # 每租户限流（等待令牌，不占用全局并发槽）
            await self.sandbox.acquire_tenant(tenant)
            # 全局并发闸门 + 请求级超时预算
            result = await self.sandbox.run_guarded(
                lambda: self._pipeline(question, ctx, priority, output_format),
            )
            result.latency_ms = int((time.monotonic() - started) * 1000)
            self._observe_success(result)
            return result
        except (RateLimitError, TimeoutBudgetError, AccessDeniedError, HITLRequiredError):
            self.metrics.counter("requests_failed").inc()
            raise
        except FusionRagError as exc:
            self.metrics.counter("requests_failed").inc()
            logger.error("问答处理失败（框架异常）：%s", exc)
            return self._error_result(ctx, started, f"服务处理异常：{exc}")
        except Exception as exc:  # noqa: BLE001 —— 兜底，绝不把裸异常抛给用户
            self.metrics.counter("requests_failed").inc()
            logger.exception("问答处理未预期异常")
            return self._error_result(ctx, started, f"服务内部错误：{type(exc).__name__}")
        finally:
            self._sync_llm_metrics()
            await self.tracer.end(trace)

    async def ask_many(
        self, requests: list[dict[str, Any]],
    ) -> list[QaResult]:
        """并发处理多个问答请求（批量/压测用）。失败隔离，逐个返回。"""
        return list(await asyncio.gather(*(self.ask(**req) for req in requests)))

    # ==================================================================
    # 主链路
    # ==================================================================
    async def _pipeline(
        self,
        question: str,
        ctx: AgentContext,
        priority: Priority | None,
        output_hint: OutputFormat | str | None,
    ) -> QaResult:
        text = (question or "").strip()
        await self._ensure_session(ctx, text)

        # 记忆装配（基建、非决策步骤，任何模式下都要跑）
        reserve = approx_token_count(text) + 512
        logger.info("phase: memory", extra={"event": "phase", "phase": "memory"})
        memory_ctx = await self.memory.build_context(
            ctx.session_id, ctx.tenant_id, reserve_tokens=reserve,
        )
        memory_block = memory_ctx.to_prompt_block()

        agentic_info: dict[str, Any] = {}
        if self._agentic_available():
            # ★ 主链路：完全自主 agentic。前置不再预调 intent/decompose LLM，
            #     模型自己从问题里推断并在尾标 [INTENT:xxx][DOMAIN:xxx][RISK:xxx]。
            #     独立 validator 也不再跑，取模型自报 confidence。
            answer, validation, attempts, agentic_info = await self._agentic_pipeline(
                text, ctx, memory_block, output_hint,
            )
            intent = agentic_info.pop("intent_result", None) or IntentResult(
                intent="qa", valid=True, domain="unknown", risk="low",
            )
            plan = agentic_info.pop("plan_result", None) or PlanResult(
                original=text, sub_questions=[text], decomposed=False,
                priority=ctx.priority,
            )
            chunks: list[Any] = []
        else:
            # 降级链路：无 function-calling 能力（EchoLLM）时的传统
            # intent → decompose → retrieve → reason → validate 预置流水线。
            logger.info(
                "phase: intent (fallback)",
                extra={"event": "phase", "phase": "intent", "mode": "fallback"},
            )
            intent = await self.intent.classify(text, ctx)
            if intent.intent == "reject":
                return await self._short_circuit(
                    ctx, intent, _REFUSAL, question=text, confidence=1.0,
                )
            if not intent.valid:
                reply = _CLARIFY if intent.intent == "clarify" else _CHITCHAT
                return await self._short_circuit(
                    ctx, intent, reply, question=text, confidence=1.0,
                )
            plan = await self.decomposer.plan(text, risk=intent.risk)
            if priority is None:
                ctx.priority = plan.priority
            if plan.need_human and self.hitl_enabled:
                return await self._short_circuit(
                    ctx, intent, _HITL, question=text, confidence=0.0,
                    meta={"need_human": True, "sub_questions": plan.sub_questions},
                    sub_questions=plan.sub_questions,
                )
            chunks = await self.retrieval.retrieve_for_plan(
                plan.sub_questions, ctx, top_k=self.top_k,
            )
            self.tracer.tag(evidence=len(chunks))
            answer, validation, attempts = await self._reason_and_validate(
                text, chunks, ctx, memory_block, output_hint,
            )

        self.tracer.tag(
            intent=intent.intent, domain=intent.domain, risk=intent.risk,
            decomposed=plan.decomposed, sub_questions=len(plan.sub_questions),
            priority=ctx.priority.value,
        )

        # 输出脱敏
        redacted = self._apply_redaction(answer)

        # 组装结果
        result = self._assemble(
            ctx, intent, plan, answer, validation, redacted, attempts, agentic_info,
        )
        if agentic_info:
            result.meta["agentic"] = agentic_info

        # 持久化 + 事件广播
        await self._persist_qa(ctx, text, result, intent)
        return result

    # ==================================================================
    # Agentic 文件工具（对标 Claude Code：运行时自主读原文）
    # ==================================================================
    def _agentic_available(self) -> bool:
        """当前是否具备 agentic 全流程能力（开关/实例/模型 function-calling）。"""
        if not self.agentic_tools or self.tool_agent is None:
            return False
        if self.llm_router is None or not self.llm_router.supports_tools(None):
            return False
        return True

    # ------------------------------------------------------------------
    # 子查询拆解：完全交给大模型（无确定性正则规则辅助）
    # ------------------------------------------------------------------
    async def _extract_sub_queries_llm(self, question: str) -> list[str]:
        """用大模型把问题拆解成若干利于检索的子查询（关系类型驱动，不做字级/句读硬拆）。

        ★ 拆不拆、怎么拆、拆几条，全由模型依下述关系类型清单自行判断，不借助
          任何正则/词表规则。会在 seed 预检索前发一次 chat（≈ +8~15s），受
          retrieval.seed_llm_decompose（默认开）控制；关闭时只用原句检索。
          失败回落为“仅原句”（恒等，非规则拆分）。
        """
        q = (question or "").strip()
        if not q:
            return []
        try:
            resp = await self.llm_router.chat(  # type: ignore[union-attr]
                [Message.system(INOVANCE_PERSONA), Message.user(
                    "你是检索规划助手。请判断下面这个用户问题是否需要在手册/知"
                    "识库中分多路检索；若需要，拆成若干条各自可独立检索的子查询，"
                    "并覆盖以下关系类型（仅当问题确实涉及该类时才拆，不硬凑）：\n"
                    "1. 实体/术语消歧与黑话归一：先把问题里的汇川行业黑话/口语简称"
                    "（如“H5U”“GL20”“AM600”，以及“大点数/小点数”“常用型号”“新一代”这"
                    "类没指明具体对象的相对词）对应到手册标准名词；对含糊限定词补"
                    "一条把它落到具体产品型号/系列的子查询（如“X 包含哪些产品型号/"
                    "如何界定”），子查询里同时写出用户原词与标准名词。\n"
                    "2. 并列枚举：出现“A、B、C 分别…”“以及”“还有”时，为每个并列"
                    "对象各拆一条。\n"
                    "3. 对比/取舍：出现“区别/哪个更好/异同/对比”时，为每个被比较"
                    "对象各拆一条，并为关键比较维度各拆一条，不要把对比合成单句。\n"
                    "4. 前提依赖：回答前须先确认的前提信息，拆成“先厘清前提”的子查询。\n"
                    "5. 因果/多步流程：既问原因又问处置/步骤时，拆成“原因”与“处置"
                    "步骤”等子查询。\n"
                    "纪律：只做追加式扩展，保留用户原词，绝不替换或同义改写原词。"
                    "若问题是单点事实、无需拆解，就只返回原问题本身。\n"
                    "输出：只输出一个 JSON 字符串数组（按检索优先级排序、最多 4 条，"
                    '可不含原句），不要解释或任何多余文字。\n问题：' + q
                )],
                temperature=0.0,
                max_tokens=320,
            )
            text = (resp.text or "").strip()
            start, end = text.find("["), text.rfind("]")
            if start == -1 or end <= start:
                raise ValueError("no json array in llm response")
            items = json.loads(text[start:end + 1])
            subs: list[str] = []
            seen: set[str] = set()
            for it in (items if isinstance(items, list) else []):
                s = str(it).strip()
                if len(s) >= 2 and s not in seen:
                    seen.add(s)
                    subs.append(s)
            # 原句始终置顶，保证全句语义检索不丢（恒等，非规则）
            if q not in seen:
                subs.insert(0, q)
            return subs[:4] or [q]
        except Exception as exc:  # noqa: BLE001
            logger.warning("seed LLM 拆子查询失败，回落仅用原句：%s", exc)
            return [q]

    async def _collect_seed_data(
        self, question: str, ctx: AgentContext, *, top_k: int = 2,
    ) -> list[tuple[str, list]]:
        """预跑多维检索，返回 ``[(子查询, chunks), ...]`` 结构化列表。

        供 :meth:`_format_seed_evidence` 渲染文本，以及 fanout 分组候选
        复用（见 :meth:`_group_candidates_by_source`）——一次检索，两处用。
        失败或无 retrieval 时返回空列表。
        """
        if self.retrieval is None:
            return []
        try:
            if self._seed_llm_decompose and self.llm_router is not None:
                sub_queries = await self._extract_sub_queries_llm(question)
            else:
                # 关闭拆解：只用原句（恒等，非规则拆分）
                _q = (question or "").strip()
                sub_queries = [_q] if _q else []
            # ★ 并发检索各子查询（与 retrieve_for_plan 同款 gather 范式）：旧实现
            #   串行 await 循环使 N 个子查询的 embedding+rerank 网络往返首尾相加，
            #   是答案首 token 前最大的可控耗时；改为并发后压成约一次往返。
            #   保留 (子查询, chunks) 分组结构，供 _format_seed_evidence 渲染与
            #   _group_candidates_by_source 扇出分组两处复用。
            chunk_groups = await asyncio.gather(
                *(self.retrieval.retrieve(sq, ctx, top_k=top_k) for sq in sub_queries),
                return_exceptions=True,
            )
            results: list[tuple[str, list]] = []
            for sq, chunks in zip(sub_queries, chunk_groups):
                if isinstance(chunks, BaseException):
                    logger.warning("seed 子查询检索失败（%s）：%s", sq[:30], chunks)
                    chunks = []
                results.append((sq, list(chunks)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("agentic seed_evidence 预检索失败（继续空 seed）：%s", exc)
            return []
        return results

    @staticmethod
    def _format_seed_evidence(results: list[tuple[str, list]]) -> str:
        """把 :meth:`_collect_seed_data` 的结构化结果渲染成标签式线索文本。

        低分（score<0.3）条目只给摘要不给精确行号，鼓励模型自主
        search/grep 发现具体内容（与工具 schema 重新定位保持一致）。
        """
        import os as _os
        import re as _re
        if not results or not any(chunks for _, chunks in results):
            return ""
        lines: list[str] = []
        seen_keys: set[tuple[str, int]] = set()
        idx = 0
        for sq, chunks in results:
            if not chunks:
                lines.append(f"▸「{sq}」→ 知识库无直接匹配")
                continue
            # 去重后实际新增的条目数
            new_items: list = []
            for c in chunks:
                inner = getattr(c, "chunk", None)
                key = getattr(c, "identity", None) or (
                    getattr(inner, "document_id", ""), getattr(inner, "chunk_index", 0)
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                new_items.append(c)
            if not new_items:
                lines.append(f"▸「{sq}」→ 线索同上方")
                continue
            lines.append(f"▸「{sq}」初步线索:")
            for c in new_items:
                idx += 1
                inner = getattr(c, "chunk", None)
                title = str(getattr(c, "title", "") or "")
                src = str(getattr(c, "source", "") or "")
                score = round(float(getattr(c, "score", 0.0) or 0.0), 4)
                meta = getattr(inner, "metadata", None) or {}
                start_line = meta.get("start_line") if isinstance(meta, dict) else None
                locator = ""
                if src and start_line:
                    try:
                        locator = f"{_os.path.basename(src)}:{int(start_line)}"
                    except (TypeError, ValueError):
                        locator = _os.path.basename(src) if src else ""
                elif src:
                    locator = _os.path.basename(src)
                snippet = _re.sub(r"\s+", " ", (c.content or ""))[:120]
                head = f"  [{idx}] 《{title}》 score={score}"
                # ★ 低置信度时不给精确行号，鼓励模型自主 search/grep 发现内容
                if locator and score >= 0.3:
                    head += f" @ {locator}"
                lines.append(f"{head}— {snippet}")
        return "\n".join(lines)

    async def _build_seed_evidence(
        self, question: str, ctx: AgentContext, *, top_k: int = 2,
    ) -> str:
        """agentic 主链前预跑多维检索，按子查询分组输出线索（组合调用）。"""
        results = await self._collect_seed_data(question, ctx, top_k=top_k)
        return self._format_seed_evidence(results)

    @staticmethod
    def _group_candidates_by_source(
        seed_data: list[tuple[str, list]], max_manuals: int,
    ) -> list[tuple[str, list]]:
        """按 chunk ``source`` basename 聚合候选，取每组最高分降序截断前 N 本。

        复用现有 seed 检索结果分组去重，零额外检索/LLM 调用成本；
        只有 ``>=2`` 本候选手册时才值得扇出（单候选走原单 Agent 路径）。
        """
        groups: dict[str, dict[str, Any]] = {}
        for _sq, chunks in seed_data:
            for c in chunks:
                src = str(getattr(c, "source", "") or "")
                if not src:
                    continue
                base = os.path.basename(src)
                score = float(getattr(c, "score", 0.0) or 0.0)
                g = groups.get(base)
                if g is None:
                    groups[base] = {"max_score": score, "chunks": [c]}
                else:
                    g["max_score"] = max(g["max_score"], score)
                    g["chunks"].append(c)
        ranked = sorted(groups.items(), key=lambda kv: kv[1]["max_score"], reverse=True)
        return [(base, info["chunks"]) for base, info in ranked[:max(1, max_manuals)]]

    async def _run_fanout(
        self,
        question: str,
        ctx: AgentContext,
        candidates: list[tuple[str, list]],
        memory_block: str,
        output_hint: OutputFormat | str | None,
    ) -> Any:
        """按候选手册分片构造子 Agent 任务卡，交给 FanoutCoordinator 并发检索。"""
        from .fanout import FanoutCoordinator, ManualWorkerSpec

        fc = self.fanout_config
        specs: list[ManualWorkerSpec] = []
        for base_name, chunks in candidates:
            seed_text = self._format_seed_evidence([(question, chunks)])
            specs.append(ManualWorkerSpec(
                worker_id=f"manual:{base_name}",
                manual_source=base_name,
                seed_text=seed_text,
            ))
        coordinator = FanoutCoordinator(
            tool_agent=self.tool_agent,
            llm=self.llm_router,
            max_manuals=int(fc.get("max_manuals", 5)),
            confidence_threshold=float(fc.get("confidence_threshold", 0.7)),
            worker_max_iters=int(fc.get("worker_max_iters", 4)),
            merge_enabled=bool(fc.get("merge_enabled", True)),
        )
        hint_text = str(output_hint) if output_hint else ""
        return await coordinator.run(
            question, ctx, specs,
            memory_block=memory_block, output_hint=hint_text,
        )

    def _fanout_ready(
        self, question: str, ctx: AgentContext, seed_data: list[tuple[str, list]],
    ) -> list[tuple[str, list]]:
        """判断是否应走 fanout；返回候选列表（不足 2 本或功能关闭时为空）。"""
        if not self.fanout_config.get("enabled") or self.tool_agent is None:
            return []
        max_manuals = int(self.fanout_config.get("max_manuals", 5))
        candidates = self._group_candidates_by_source(seed_data, max_manuals)
        if len(candidates) < 2:
            return []
        logger.info(
            "fanout candidates grouped",
            extra={
                "event": "fanout_candidates",
                "question": question[:80],
                "manuals": [base for base, _ in candidates],
            },
        )
        return candidates

    def _finalize_fanout(
        self, question: str, ctx: AgentContext, fanout_result: Any,
    ) -> tuple[Answer, ValidationResult | None, int, dict[str, Any]]:
        """把 FanoutResult 接入现有答案合成/引用聚合链（与单 Agent 后处理同构）。"""
        confidence = (
            fanout_result.self_confidence
            if fanout_result.self_confidence is not None else 0.6
        )
        _n_cites = len(fanout_result.file_citations or [])
        if _n_cites < 2 and confidence > 0.7:
            confidence = 0.7
        if _n_cites == 0 and confidence > 0.55:
            confidence = 0.55
        answer = Answer(
            text=fanout_result.answer,
            citations=self._build_file_citations(
                fanout_result.file_citations, [], evidence=fanout_result.evidence,
            ),
            confidence=confidence,
            model=fanout_result.model or "",
        )
        validation = ValidationResult(
            passed=confidence >= 0.5,
            confidence=confidence,
            issues=[] if confidence >= 0.5 else ["fanout low confidence"],
            correction="",
        )
        _stats = fanout_result.worker_stats or []
        info: dict[str, Any] = {
            "iterations": sum(int(s.get("iterations", 0) or 0) for s in _stats),
            "tool_calls": sum(int(s.get("tool_calls", 0) or 0) for s in _stats),
            "degraded": (all(bool(s.get("degraded")) for s in _stats) if _stats else True),
            "file_citations": len(fanout_result.file_citations),
            "applied": bool(fanout_result.answer),
            "mode": f"fanout_{fanout_result.mode}",
            "self_confidence": fanout_result.self_confidence,
            "intent_hint": fanout_result.intent_hint,
            "domain_hint": fanout_result.domain_hint,
            "risk_hint": fanout_result.risk_hint,
            "fanout": {
                "worker_count": fanout_result.worker_count,
                "winner_id": fanout_result.winner_id,
                "worker_stats": _stats,
            },
        }
        info["intent_result"] = IntentResult(
            intent=fanout_result.intent_hint or "qa",
            valid=True,
            domain=fanout_result.domain_hint or "unknown",
            risk=fanout_result.risk_hint or "low",
        )
        info["plan_result"] = PlanResult(
            original=question,
            sub_questions=[question],
            decomposed=False,
            priority=ctx.priority,
        )
        if not answer.text:
            answer = Answer(text="未能从知识库中获取足够信息作答。", confidence=0.0)
        return answer, validation, 1, info

    async def _agentic_pipeline(
        self,
        question: str,
        ctx: AgentContext,
        memory_block: str,
        output_hint: OutputFormat | str | None,
    ) -> tuple[Answer, ValidationResult | None, int, dict[str, Any]]:
        """完全 agentic 主链路：模型自主选工具、自己组织证据、自报信心与意图。

        流程：
          1. ToolAgent.run(question, memory_block, output_hint)
          2. 从 tool_run.self_confidence / intent_hint / domain_hint / risk_hint 合成结果
          3. 不再独立跑 validator（模型自报已可取，且成本十拆中最高）

        失败回干：无答案时统一给到固定提示，不抛异常。
        """
        hint_text = str(output_hint) if output_hint else ""
        # 预算与重试：agentic_retry=0 时 max_attempts=1
        max_attempts = max(1, self.agentic_retry + 1)

        # ★ 预跑一次混合检索，把 top-3 命中拼成 seed_evidence 喂给 ToolAgent：
        #   弱模型 iter#1 不再从零猜 doc_hints（避免塑袋“第 798 页”），
        #   且因为命中已带 `basename.md:line`，首轮就能形成 file_citations。
        #   失败静默降级为空 seed，不阻断主链。
        #   seed_data 结构化结果同时用于 fanout 候选手册分组（见下），一次检索两处用。
        seed_data = await self._collect_seed_data(question, ctx)
        seed_evidence = self._format_seed_evidence(seed_data)
        # ★ 可观测性：把 seed 内容输出给 run_kb 展示，让用户理解模型的起点信息
        logger.info(
            "seed_evidence_built",
            extra={
                "event": "seed_evidence_built",
                "seed": seed_evidence[:600] if seed_evidence else "",
                # ★ 展示“实际用于检索的那批子查询”（而非另算一份），与拆解开关一致
                "sub_queries": [sq for sq, _ in seed_data],
            },
        )

        # ★ fanout 分支：候选命中 >=2 本手册且开关启用时，按手册分片并发子 Agent，
        #   满足即广播停止；单候选/未启用完全走下方原有单 Agent 路径（零回归）。
        fanout_candidates = self._fanout_ready(question, ctx, seed_data)
        if fanout_candidates:
            fanout_result = await self._run_fanout(
                question, ctx, fanout_candidates, memory_block, output_hint,
            )
            return self._finalize_fanout(question, ctx, fanout_result)

        answer = Answer(text="", confidence=0.0, citations=[])
        validation: ValidationResult | None = None
        tool_run = None
        issues: list[str] = []
        attempts_used = 0
        failure_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            self.metrics.counter("agentic_attempt").inc()
            # ★ 单发模（max_attempts=1）下不报 attempt 事件，避免 UI
            #   在开头挂一个“重试 #1/1”的误导横幅；真重试才报。
            if max_attempts > 1:
                logger.info(
                    f"agentic attempt #{attempt}",
                    extra={
                        "event": "agentic_attempt",
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "followup_count": len(issues),
                    },
                )
            try:
                tool_run = await self.tool_agent.run(  # type: ignore[union-attr]
                    question, ctx,
                    seed_evidence=seed_evidence,
                    memory_block=memory_block,
                    output_hint=hint_text,
                    followup_issues=issues if issues else None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("agentic run 失败：%s", exc)
                failure_error = type(exc).__name__
                break

            if not tool_run.answer:
                break

            # ★ 合成答案：信心采用模型自报，未报时默认 0.6（不作可信性信号）
            confidence = (
                tool_run.self_confidence
                if tool_run.self_confidence is not None else 0.6
            )
            # ★ 硬校验：弱模型的 self_confidence 基本就是“幻觉计数器”（旧行为
            #   报 0.95 但一条引用也无）。用实际证据量给它下限：
            #   • 引用不够 2 条时，上限 0.7（不给“自信满格”的假信号）
            #   • 一条引用也无时，直接上限 0.55（“完全没落地”）
            _n_cites = len(tool_run.file_citations or [])
            if _n_cites < 2 and confidence > 0.7:
                confidence = 0.7
            if _n_cites == 0 and confidence > 0.55:
                confidence = 0.55
            answer = Answer(
                text=tool_run.answer,
                citations=self._build_file_citations(
                    tool_run.file_citations, [], evidence=tool_run.evidence,
                ),
                confidence=confidence,
                model=tool_run.model or "",
            )
            # 自报代替外部 validator；若开关强制要求，才回补跑一次
            validation = ValidationResult(
                passed=confidence >= 0.5,
                confidence=confidence,
                issues=[] if confidence >= 0.5 else ["self-report low confidence"],
                correction="",
            )
            # 自报信心低 + 允许重试 → 拉一个 followup 回炉
            if confidence < 0.4 and attempt < max_attempts:
                issues = [
                    "自报信心低于 0.4，请补充更多工具取证后重写答案",
                ]
                continue
            break

        # 回传 agentic_info：包含模型自报的 intent/domain/risk 合成到
        # IntentResult 中，以及 plan_result 从 tool_run.transcript 提炼工具链摘要。
        info: dict[str, Any] = {}
        if tool_run is not None:
            info = {
                "iterations": tool_run.iterations,
                "tool_calls": tool_run.tool_calls,
                "degraded": tool_run.degraded,
                "file_citations": len(tool_run.file_citations),
                "applied": bool(tool_run.answer),
                "mode": "agentic_first",
                "self_confidence": tool_run.self_confidence,
                "intent_hint": tool_run.intent_hint,
                "domain_hint": tool_run.domain_hint,
                "risk_hint": tool_run.risk_hint,
            }
            info["intent_result"] = IntentResult(
                intent=tool_run.intent_hint or "qa",
                valid=True,
                domain=tool_run.domain_hint or "unknown",
                risk=tool_run.risk_hint or "low",
            )
            info["plan_result"] = PlanResult(
                original=question,
                sub_questions=[question],
                decomposed=False,
                priority=ctx.priority,
            )
        else:
            # tool_run 为 None 代表 agentic 主链路本身崩了，记录错误供外部展示
            info = {
                "iterations": 0,
                "tool_calls": 0,
                "degraded": True,
                "applied": False,
                "mode": "agentic_first",
            }
            if failure_error:
                info["error"] = failure_error
        # 无答案（异常情况）：退回一个固定提示，不抛异常
        if not answer.text:
            answer = Answer(text="未能从知识库中获取足够信息作答。", confidence=0.0)
        return answer, validation, attempts_used, info

    @staticmethod
    def _wrap_tool_evidence(evidence: list[dict[str, Any]]) -> list[Any]:
        """将 ToolRunResult.evidence 包装为带 ``.content`` 的 duck-typed 对象。

        Validator 内部只读 ``c.content``；避免额外定义完整 Chunk 体系。
        """
        class _Evi:
            __slots__ = ("content", "score", "title", "source", "document_id")
            def __init__(self, content: str, tool: str) -> None:
                self.content = content
                self.score = 0.5
                self.title = f"tool:{tool}"
                self.source = tool
                self.document_id = ""
        return [_Evi(e.get("content", ""), e.get("tool", "")) for e in evidence]

    @staticmethod
    def _build_file_citations(
        file_citations: list[dict[str, Any]], chunks: list[Any],
        evidence: list[dict[str, Any]] | None = None,
    ) -> list[Citation]:
        """把工具读到的 ``文件:行号`` 转成 Citation，尽量回链到已检索文档 id。

        agentic 主链路下 chunks 往往为空，真正可用的相关性信号在
        evidence[search_kb].content（已含 ``[i] title (score=x.xxxx)``）里。

        ★ **按 path 聚合**：同一手册多行命中会归并为一条引用（snippet
        里列出全部行号），score 取该手册在 search_kb 里的 max，避免
        “同一手册 6 行 → 6 条都是 0.9543”的伪多样。
        """
        by_name: dict[str, Any] = {}
        for rc in chunks or []:
            if rc.source:
                by_name[os.path.basename(rc.source).lower()] = rc
        # 从 search_kb 证据里回捞 score（按 title 取 max）
        score_map: dict[str, float] = {}
        _score_pat = re.compile(r"\(score=([0-9.]+)\)")
        for ev in evidence or []:
            if str(ev.get("tool", "")) != "search_kb":
                continue
            for line in str(ev.get("content", "")).splitlines():
                m = _score_pat.search(line)
                if not m:
                    continue
                head = line.split(" (score=", 1)[0]
                head = re.sub(r"^\s*\[\d+\]\s*", "", head)
                head = re.sub(r"\s@\s.*$", "", head)
                title = head.strip()
                if not title:
                    continue
                try:
                    s = float(m.group(1))
                except ValueError:
                    continue
                key = title.lower()
                if s > score_map.get(key, -1.0):
                    score_map[key] = s
        # ★ 按 path 归并同一手册的多行命中
        grouped: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for fc in file_citations:
            path = str(fc.get("path", ""))
            if not path:
                continue
            line = fc.get("line")
            entry = grouped.get(path)
            if entry is None:
                grouped[path] = {"path": path, "lines": [line] if line else []}
                order.append(path)
            else:
                if line and line not in entry["lines"]:
                    entry["lines"].append(line)
        cites: list[Citation] = []
        for i, path in enumerate(order[:12], 1):
            entry = grouped[path]
            lines: list[Any] = entry["lines"]
            rc = by_name.get(os.path.basename(path).lower())
            if lines:
                snippet = path + ":" + ",".join(str(x) for x in lines[:5])
                if len(lines) > 5:
                    snippet += f",…(+{len(lines) - 5}行)"
            else:
                snippet = path
            score = float(getattr(rc, "score", 0.0) or 0.0) if rc else 0.0
            if score == 0.0:
                pname = os.path.basename(path).lower()
                for k, v in score_map.items():
                    if pname and (pname in k or k in pname):
                        score = v
                        break
            cites.append(Citation(
                index=i,
                document_id=getattr(rc, "document_id", "") if rc else "",
                title=path,
                source=rc.source if rc else path,
                chunk_index=rc.chunk.chunk_index if rc else 0,
                snippet=snippet,
                score=score,
            ))
        return cites

    async def _reason_and_validate(
        self,
        question: str,
        chunks: list[Any],
        ctx: AgentContext,
        memory_block: str,
        output_hint: OutputFormat | str | None,
    ) -> tuple[Answer, ValidationResult | None, int]:
        max_attempts = (self.max_retry + 1) if self.validation_enabled else 1
        correction = ""
        answer: Answer | None = None
        validation: ValidationResult | None = None

        for attempt in range(1, max_attempts + 1):
            answer = await self.reasoning.reason(
                question, chunks, ctx,
                memory_block=memory_block, output_hint=output_hint, correction=correction,
            )
            if not self.validation_enabled:
                break
            validation = await self.validator.validate(question, answer, chunks, ctx)
            if validation.passed and validation.confidence >= self.validator.min_confidence:
                break
            correction = validation.correction or "；".join(validation.issues)
            if attempt < max_attempts:
                self.metrics.counter("answer_retries").inc()
                logger.info("答案校验未通过（第 %d 轮），回炉重写：%s", attempt, correction)

        assert answer is not None
        return answer, validation, attempt

    # ==================================================================
    # 结果组装 / 脱敏
    # ==================================================================
    def _apply_redaction(self, answer: Answer) -> bool:
        if not self.redaction_enabled or self.redactor is None:
            return False
        result = self.redactor.redact(answer.text)
        changed = result.redacted
        if changed:
            answer.text = result.text
        for cite in answer.citations:
            r = self.redactor.redact(cite.snippet)
            if r.redacted:
                cite.snippet = r.text
                changed = True
        if changed:
            self.metrics.counter("redactions", "脱敏命中").inc()
        return changed

    def _assemble(
        self,
        ctx: AgentContext,
        intent: IntentResult,
        plan: PlanResult,
        answer: Answer,
        validation: ValidationResult | None,
        redacted: bool,
        attempts: int,
        agentic_info: dict[str, Any] | None = None,
    ) -> QaResult:
        validated = bool(validation.passed) if validation else False
        if validation is not None:
            confidence = validation.confidence
        else:
            confidence = answer.confidence
        degraded = _is_fallback_model(answer.model)
        # agentic 模式：当实际使用了真实 LLM（非兜底 model）时，采用工具循环
        # 真实 degraded 信号；如果回退到了 echo/extractive，那本身就是降级，保持启发式判定
        if (
            agentic_info
            and agentic_info.get("mode") == "agentic_first"
            and not _is_fallback_model(answer.model)
        ):
            degraded = bool(agentic_info.get("degraded", False))
        self.tracer.tag(
            confidence=round(confidence, 3), validated=validated,
            degraded=degraded, attempts=attempts, redacted=redacted,
        )
        return QaResult(
            answer=answer.text,
            citations=answer.citations,
            session_id=ctx.session_id,
            trace_id=ctx.trace_id,
            intent=intent,
            sub_questions=plan.sub_questions,
            confidence=round(confidence, 3),
            validated=validated,
            output_format=answer.output_format,
            redacted=redacted,
            model=answer.model,
            degraded=degraded,
            meta={
                "domain": intent.domain,
                "risk": intent.risk,
                "priority": ctx.priority.value,
                "attempts": attempts,
                "decomposed": plan.decomposed,
                "issues": (validation.issues if validation else []),
            },
        )

    # ==================================================================
    # 短路与错误路径
    # ==================================================================
    async def _short_circuit(
        self,
        ctx: AgentContext,
        intent: IntentResult,
        reply: str,
        *,
        question: str = "",
        confidence: float,
        meta: dict[str, Any] | None = None,
        sub_questions: list[str] | None = None,
    ) -> QaResult:
        if self.redaction_enabled and self.redactor is not None:
            reply = self.redactor.redact(reply).text
        result = QaResult(
            answer=reply,
            citations=[],
            session_id=ctx.session_id,
            trace_id=ctx.trace_id,
            intent=intent,
            sub_questions=sub_questions or [],
            confidence=confidence,
            validated=False,
            output_format=OutputFormat.PARAGRAPH,
            model="policy",
            meta=meta or {"short_circuit": intent.intent},
        )
        # 仍记录到会话记忆，保证多轮连续性
        if question:
            try:
                await self.memory.record_qa(ctx.session_id, question, reply)
            except Exception:  # noqa: BLE001
                logger.debug("短路结果记忆记录失败（忽略）", exc_info=True)
        return result

    def _error_result(self, ctx: AgentContext, started: float, message: str) -> QaResult:
        return QaResult(
            answer="抱歉，处理该问题时出现异常，请稍后重试或联系管理员。",
            citations=[],
            session_id=ctx.session_id,
            trace_id=ctx.trace_id,
            intent=IntentResult(intent="error", valid=False, reason=message),
            confidence=0.0,
            validated=False,
            latency_ms=int((time.monotonic() - started) * 1000),
            degraded=True,
            model="error",
            meta={"error": message},
        )

    # ==================================================================
    # 持久化 / 指标 / 事件
    # ==================================================================
    async def _ensure_session(self, ctx: AgentContext, question: str) -> None:
        try:
            await self.store.upsert_session(
                ctx.session_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id,
                title=question[:40],
            )
        except Exception:  # noqa: BLE001
            logger.debug("会话登记失败（忽略）", exc_info=True)

    async def _persist_qa(
        self, ctx: AgentContext, question: str, result: QaResult, intent: IntentResult,
    ) -> None:
        # 记忆（多轮上下文）
        try:
            await self.memory.record_qa(
                ctx.session_id, question, result.answer,
                meta={"confidence": result.confidence, "trace_id": result.trace_id},
            )
        except Exception:  # noqa: BLE001
            logger.debug("记忆记录失败（忽略）", exc_info=True)
        # 问答反馈（问题沉淀）
        try:
            await self.store.log_qa(
                question=question, answer=result.answer, tenant_id=ctx.tenant_id,
                session_id=ctx.session_id, trace_id=ctx.trace_id,
                confidence=result.confidence, validated=result.validated,
                latency_ms=result.latency_ms, good=result.confidence >= 0.5,
            )
        except Exception:  # noqa: BLE001
            logger.debug("问答反馈落库失败（忽略）", exc_info=True)
        # 事件广播（下游：指标、问题沉淀、审计订阅者）
        self.events.publish(Event("qa.completed", {
            "trace_id": result.trace_id, "session_id": ctx.session_id,
            "tenant_id": ctx.tenant_id, "intent": intent.intent,
            "confidence": result.confidence, "validated": result.validated,
            "degraded": result.degraded, "latency_ms": result.latency_ms,
            "citations": len(result.citations),
        }))

    def _observe_success(self, result: QaResult) -> None:
        self.metrics.histogram("request_latency_seconds").observe(
            result.latency_ms / 1000.0,
        )
        self.metrics.histogram("answer_confidence").observe(result.confidence)
        if result.degraded:
            self.metrics.counter("requests_degraded").inc()

    def _sync_llm_metrics(self) -> None:
        """把 LLM Router 的累计用量以增量方式同步到指标（供告警规则消费）。"""
        if self.llm_router is None:
            return
        usage = self.llm_router.usage
        calls_delta = usage.get("calls", 0) - self._last_usage["calls"]
        errors_delta = usage.get("errors", 0) - self._last_usage["errors"]
        if calls_delta > 0:
            self.metrics.counter("llm_calls").inc(float(calls_delta))
        if errors_delta > 0:
            self.metrics.counter("llm_errors").inc(float(errors_delta))
        self._last_usage = {"calls": usage.get("calls", 0), "errors": usage.get("errors", 0)}
