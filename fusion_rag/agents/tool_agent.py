"""ToolAgent：运行时自主调用文件工具的 agentic 多轮循环（对标 Claude Code）。

给定问题与租户作用域，让**真实大模型**通过原生 function-calling 自主决定
调用哪些文件工具（grep/glob/list_dir/read_file）、按什么顺序调用，读取原始
知识库文件补齐证据，最终给出带 ``文件:行号`` 引用的答案。

时序（``run``）：
    组装 system（工具清单 + 作用域 + 引用规范）
      → 循环 ≤ max_iters：
          chat(tools=..., tool_choice="auto")
            ├─ 模型请求工具 → 执行 → 以 tool 角色回填结果 → 继续
            └─ 模型直接作答 → 结束
      → 达上限仍无答案 → 去工具再补一轮，强制收敛出答案（标记 degraded）

韧性：任一工具/模型异常都被捕获并降级，绝不抛出打断主链路；离线无
function-calling 能力时由 Orchestrator 事先跳过，不会走到这里。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging import get_logger
from ..constants import INOVANCE_PERSONA, cite_extension_alternation as _cite_alt
from ..llm.base import Message
from ..tools.base import ToolContext, ToolResult
from ..tools.plan_tool import PlanStore
from .base import AgentContext, BaseAgent

logger = get_logger(__name__)

# ★ 满足才停用的“未答上/资料不足”式对冲语（策略 A 逃生阀）：命中即视为
#   未真正满足原问题，触发回退工具深挖。保守列表，宁可漏判不误伤正常答案。
_UNSATISFIED_RE = re.compile(
    r"未.{0,6}(说明|提及|提供|找到|检索到|包含|涉及)"
    r"|不明确|不确定|无法确定|无从判断"
    r"|资料中?(没有|无|未)"
    r"|需参考完整|需查阅完整|信息不足"
)

# ★ 真实请求起点（perf_counter），由 Orchestrator.ask 入口写入当前上下文。
#   fanout 并发的每个子任务在创建时继承该值，使端到端 TTFT 覆盖全链路
#   （含索引加载、意图/规划/seed 检索），而非仅 ToolAgent 内部耗时。
#   取不到（如单元直测）时回退 _t_agent_start，保持旧行为。
_REQUEST_ORIGIN: ContextVar[float] = ContextVar(
    "fusion_rag_request_origin", default=0.0,
)


def set_request_origin(value: float) -> None:
    """由请求入口设置 perf_counter 起点，供端到端 TTFT 计算。"""
    _REQUEST_ORIGIN.set(value)


def get_request_origin() -> float:
    """返回当前请求起点；未设置时返回 0.0。"""
    return _REQUEST_ORIGIN.get()


def _truncate_args(args: Any, max_len: int = 200) -> dict[str, str]:
    """将工具入参截断为适合日志展示的精简字典。"""
    if not isinstance(args, dict):
        return {"raw": str(args)[:max_len]}
    out: dict[str, str] = {}
    for k, v in args.items():
        s = str(v)
        out[k] = s if len(s) <= max_len else s[:max_len] + "…"
    return out


def _tool_dedup_key(call: Any) -> str:
    """为一次工具调用算去重键（工具名 + 规范化参数）。plan_update 是清单读写
    工具、其重复调用属正常推进，不参与去重，返回空串。"""
    name = getattr(call, "name", "")
    if not name or name == "plan_update":
        return ""
    try:
        norm = json.dumps(
            call.arguments or {}, sort_keys=True, ensure_ascii=False, default=str,
        )
    except Exception:  # noqa: BLE001
        norm = str(getattr(call, "arguments", ""))
    return f"{name}\x00{norm}"


# 自报标签形式：[KEY:value]，允许多个行内，均会从正文中剔除
_SELF_TAG_RE = re.compile(
    r"\[\s*(CONFIDENCE|INTENT|DOMAIN|RISK)\s*:\s*([^\]\n]{1,40})\s*\]",
    re.IGNORECASE,
)


def _parse_self_report(
    text: str,
) -> tuple[str, float | None, str | None, str | None, str | None]:
    """从模型最终回答中抽取 ``[CONFIDENCE:0.85] [INTENT:qa] [DOMAIN:tech] [Risk:low]`` 标签。

    未出现时对应位置返回 None，不影响主链路。抽取后的正文会剥除这些标签，
    避免直接递给用户。
    """
    conf: float | None = None
    intent: str | None = None
    domain: str | None = None
    risk: str | None = None

    def _consume(m: re.Match[str]) -> str:
        nonlocal conf, intent, domain, risk
        key = m.group(1).upper()
        val = m.group(2).strip().lower()
        if key == "CONFIDENCE":
            try:
                conf = max(0.0, min(1.0, float(val)))
            except ValueError:
                pass
        elif key == "INTENT":
            intent = val
        elif key == "DOMAIN":
            domain = val
        elif key == "RISK":
            risk = val
        return ""

    cleaned = _SELF_TAG_RE.sub(_consume, text or "").strip()
    # 连续空行归并
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    # ★ 剥除内部反思小标题（【反思失败】【策略调整】【思考必现】【预算卡位】等）
    #   —— 面向用户的答案不应带这些提示注入时用的自定标签。
    cleaned = re.sub(
        r"【\s*(?:反思失败|反思成功|反思|策略调整|思考必现|预算卡位|禁止空壳|自报签名|标签只能出现在最后一轮|路径规范|面向用户的最终答案|检索心法[^】]*|对比类问题硬规则|预筛降流|grep\s*降流|禁无目的\s*glob)\s*】[\s:：、]?",
        "",
        cleaned,
    ).strip()
    # 开头可能的多余标点 / 空行
    cleaned = re.sub(r"^(?:\n+|[，,、：:\s])+", "", cleaned)
    # ★ 剥除面向用户时不合适的工具名提法（常见开头），保留后面内容
    #   例如：“grep 返回结果中，...”→“检索结果中，...”
    _TOOL_REF_MAP = {
        r"search_kb\s*(?:的)?\s*(?:返回结果|命中结果|结果)中[，,、]?\s*": "检索结果中，",
        r"grep\s*(?:的)?\s*(?:返回结果|命中结果|结果)(?:中|显示)?[，,、]?\s*": "检索结果中，",
        r"glob\s*(?:的)?\s*(?:返回结果|命中结果|结果)(?:中|显示)?[，,、]?\s*": "目录探测结果中，",
        r"read_file\s*(?:的)?\s*(?:返回结果|内容|结果)(?:中|显示)?[，,、]?\s*": "手册原文中，",
        r"根据\s*search_kb\s*": "根据检索结果",
        r"根据\s*grep\s*": "根据内容比对",
        r"根据\s*glob\s*": "根据目录探测",
        r"\bsearch_kb\b": "检索",
        r"\bread_file\b": "精读",
        r"\bglob\b": "目录探测",
    }
    for _pat, _rep in _TOOL_REF_MAP.items():
        cleaned = re.sub(_pat, _rep, cleaned, flags=re.IGNORECASE)
    # ★ 剥除“根据策略 X，… / 根据策略 0.1.1… / 下一轮不进一步调用”等内部提示引用
    #   这些属于系统 prompt 里定义的机制名字，不应直接递用户。
    cleaned = re.sub(
        r"[^\n。]{0,20}根据\s*(?:策略|预筛降流|硬规则|规则)\s*[\d\.一二三四五]+[^\n。]*[。.、]?\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"[^\n。]{0,20}(?:证据已(?:足够|充分)|无需(?:进一步)?(?:调用)?(?:新的)?工具(?:调用)?|可直接(?:总结|输出|给出)(?:最终)?结论)[^\n。]*[。.、]?\s*",
        "",
        cleaned,
    )
    cleaned = cleaned.strip()
    return cleaned, conf, intent, domain, risk


# 从工具输出中抽取「文件:行号」引用（扩展名白名单与 file_tools/
# knowledge.loader 统一到 constants.TEXT_SCAN_SUFFIXES，避免加一种语言要
# 同时改 4 处字面量、正则不会自动同步的坑）。
_CITE_RE = re.compile(
    r"([^\s:|\[\]{}()*'\"`]+\.(?:" + _cite_alt() + r"))(?::(\d+))?"
)

# 结构化代码标识符（必须含 `.` 或 `[]` 分段，如系统变量 `_X[0].Field`）
_IDENT_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\[\d*\]|\.[A-Za-z_][A-Za-z0-9_]*)+"
)
_IDENT_SPLIT = re.compile(r"[\[\]\.]+")


def _is_truncation(short: str, full: str) -> bool:
    """判断 short 是否为 full 的「同构尾截断」：分段数相同、每段都是对应段的前缀、
    且至少一段严格更短。仅处理“完整拼完但末尾若干字符被截”这种无歧义情形（
    如 `_RTU_ECT_INF[0].DisableSlotContr` ← `_RTU_ECT_INFO[0].DisableSlotControl`）；
    不处理“整段被丢”（段数不等）以免跨段重建误伤。"""
    if not short or short == full:
        return False
    ss = _IDENT_SPLIT.split(short)
    fs = _IDENT_SPLIT.split(full)
    if len(ss) != len(fs) or len(ss) < 2:
        return False
    # 拒绝幻影空段（如以 `]` 结尾的 `_X[0]` 会 split 出尾部空串）：真实标识符
    # 的分隔符总在内部，空段会骗过段数相等检查、导致误将结构体名补成带成员全名。
    if any(s == "" for s in ss) or any(s == "" for s in fs):
        return False
    strict = False
    for a, b in zip(ss, fs):
        if not b.startswith(a):
            return False
        if len(a) < len(b):
            strict = True
    return strict


@dataclass
class ToolRunResult:
    """ToolAgent 一次 agentic 运行的产出。"""

    answer: str
    file_citations: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    #: 工具返回的原文片段，作为给校验器的“证据”（agentic 模式下取代预检索 chunks）
    evidence: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    degraded: bool = False
    model: str = ""
    #: ★ 模型自报信心与预判（开源自 [CONFIDENCE:0.XX] 与 [INTENT:xxx] 标签提取）
    self_confidence: float | None = None
    intent_hint: str | None = None
    domain_hint: str | None = None
    risk_hint: str | None = None
    #: ★ TODO 清单式规划：最终清单渲染文本与未勾完项数（供展示/trace）
    plan: str = ""
    plan_unfinished: int = 0
    #: ★ fanout 并发子 Agent 专用：本轮属于哪个 Worker（空串=非 fanout 场景）
    #:   以及是否因同伴已解决而提前收尾（区别于正常收敛/预算耗尽降级）
    worker_id: str = ""
    stopped_early: bool = False


class ToolAgent(BaseAgent):
    """自主文件工具 Agent。"""

    name = "tool_agent"
    role = "运行时自主调用只读文件工具，读取原文补证并给出可核对引用"

    def __init__(
        self,
        *args: Any,
        registry: Any,
        executor: Any,
        max_iters: int = 6,
        budget_ms: int = 0,
        roots_resolver: Callable[[AgentContext], list[str]] | None = None,
        skill_registry: Any | None = None,
        plan_first: bool = False,
        seed_direct_answer: bool = False,
        seed_direct_min_conf: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.registry = registry
        self.executor = executor
        #: ★ 强制首轮规划：进工具循环前先让模型做一次拆题规划（真实应用经配置开启；
        #:   默认关，保证离线/测试脚本化应答序列不被打乱）
        self.plan_first = bool(plan_first)
        #: ★ 策略 A：seed 预检索已携 file:line 行级证据时，跳过多余的工具往返，
        #:   首轮直接以 tool_choice=none 作答（省一轮 ~20s chat，将端到端压向 40s）。
        #:   __init__ 默认关（保护脚本化单测轮次不变）；真实应用由 bootstrap 从
        #:   retrieval.agentic_seed_direct（默认 true）驱动开启。
        self.seed_direct_answer = bool(seed_direct_answer)
        #: ★ 满足才停阈值（① + ③）：策略 A 直答的自报置信低于此值（或命中
        #:   对冲语）则不收敛，回退工具循环继续深挖。真实应用由 bootstrap 从
        #:   retrieval.agentic_seed_direct_min_conf（默认 0.7）驱动。
        self.seed_direct_min_conf = float(seed_direct_min_conf)
        self.max_iters = max(1, int(max_iters))
        # ★ 总时间预算（毫秒），0=不限；超出则强制收敛，避免单卡卡死全链
        self.budget_ms = max(0, int(budget_ms))
        self._roots_resolver = roots_resolver or (lambda ctx: [])
        #: 可选技能注册表：注入渐进披露清单并把 skill 目录并入 jail
        self.skill_registry = skill_registry

    # ------------------------------------------------------------------
    async def run(
        self,
        question: str,
        ctx: AgentContext,
        *,
        seed_evidence: str = "",
        provider: str | None = None,
        memory_block: str = "",
        output_hint: str = "",
        followup_issues: list[str] | None = None,
        intent_domain: str | None = None,
        intent_label: str | None = None,
        stop_signal: Any | None = None,
        worker_id: str = "",
        max_iters: int | None = None,
    ) -> ToolRunResult:
        """驱动多轮工具调用；失败一律降级返回，不抛异常。

        ``stop_signal``/``worker_id``/``max_iters`` 仅 fanout 并发场景下由
        :class:`FanoutCoordinator` 注入（见 ``fusion_rag/agents/fanout.py``）；
        默认均为 None/空/沿用 ``self.max_iters``，非 fanout 路径行为完全不变。
        """
        effective_max_iters = self.max_iters if max_iters is None else max(1, int(max_iters))
        with self.span(
            "run", question_len=len(question or ""), max_iters=effective_max_iters,
        ) as node:
            tool_ctx = ToolContext(
                tenant_id=ctx.tenant_id,
                kb_ids=list(ctx.kb_ids),
                roots=self._resolve_roots(ctx),
                trace_id=ctx.trace_id,
                session_id=ctx.session_id,
            )
            # ★ 任务作用域隔离（非安全边界）：fanout 子 Agent 只能看自己被分配
            #   的那本手册，由工具层软过滤（PathJail 目录级安全沙箱不受影响）。
            _restrict = ctx.extra.get("restrict_to_sources") if ctx.extra else None
            if _restrict:
                tool_ctx.restrict_to_sources = set(_restrict)
            # ★ TODO 清单式规划：一次问答一份会话内清单，注入 ctx 供 plan_update 工具读写；
            #   默认关（plan_first=False）时为 None，全套规划行为不触发，行为与旧一致。
            plan_store: PlanStore | None = PlanStore() if self.plan_first else None
            tool_ctx.plan_store = plan_store
            # 技能目录并入授权根，模型才能用 read_file 读取 SKILL.md 全文
            tool_ctx.roots = self._with_skill_roots(tool_ctx.roots)
            if not tool_ctx.roots:
                node.attributes["no_roots"] = True
                return ToolRunResult(answer="", degraded=True, model="", worker_id=worker_id)

            # 根据意图域对工具做硬筛选（applies_to_domains 为空=全适用）
            active_registry = self.registry.filter_for_intent(domain=intent_domain)
            node.attributes["intent_domain"] = intent_domain or ""
            node.attributes["tools_active"] = len(active_registry)

            _t_agent_start = time.perf_counter()  # ★ 端到端计时起点
            messages = [
                Message.system(self._system_prompt(
                    tool_ctx, active_registry,
                    intent_domain=intent_domain, intent_label=intent_label,
                )),
                Message.user(self._user_prompt(
                    question, seed_evidence,
                    memory_block=memory_block, output_hint=output_hint,
                    followup_issues=followup_issues,
                )),
            ]
            tools = active_registry.openai_schemas()
            result = ToolRunResult(answer="", model="", worker_id=worker_id)
            seen_cites: set[str] = set()
            # ★ seed_evidence 里已含 `basename.md:line` 定位（与 search_kb 同格式），
            #   开环前先把它们计入 file_citations：一方面避免“实际已携证据引用但
            #   信心被硬校验到 0.7”的无妄降级，另一方面让早停信号（≥ 2 引用→
            #   下一轮禁工具）在 seed 较强时就能命中，省一轮 chat（≈ 15s）。
            if seed_evidence:
                _seed_view = type("_SV", (), {"is_error": False, "content": seed_evidence})()
                self._collect_citations(_seed_view, seen_cites, result.file_citations)
                # ★ seed 也归入 evidence，让 iter#1 timeout 时 extractive 兑底能
                #   拿到预检索里的真实手册片段，而不是直接“信息不足”。
                result.evidence.append({
                    "tool": "pre_search", "args": {},
                    "content": seed_evidence,
                })
            # ★ 收尾预留：最终答案合成那一轮 chat 是最有价值的，必须给它留够时
            #   间。SiliconFlow Qwen3-8B 实测单轮收尾 chat 7–12s；旧=16 与 40s 预算
            #   + max_iters=2 相冲（iter#1 开完就碰预留线→提前卡不跑工具），调到 8 适中。
            _answer_reserve_s = 5.0
            _deadline_at = (
                time.monotonic() + self.budget_ms / 1000.0
                if self.budget_ms > 0 else None
            )

            # ★ 规划不再是额外的“首轮规划 turn”，而是循环内持续维护的问题分解：
            # TODO 清单式规划：不再跑额外的规划 chat（旧方案会 TimeoutError 跳过），
            #   改由模型在循环内调 plan_update 维护“要解决哪些子问题”的问题分解；
            #   下面每轮把它作为参考注入、驱动模型自主检索。这是导航不是任务表—​—
            #   不设完成门、不要求逐条勾完，证据够了随时可作答。
            # 弱模型（Qwen3-8B）不会自发调 plan_update——旧版本会首轮推一把，现在为了
            # 控制端到端时长（目标 <40s）不再主动推，完全交给模型自发。
            # 重复调用熔断：记录已执行过的 (工具名+参数)，模型重发相同调用时不再
            # 真执行（相同检索反复空命中会白烧预算直至超时）。
            _seen_tool_calls: set[str] = set()
            # ★ 早停：一旦某轮拿到 file:line 级引用（强证据），下一轮 tool_choice 直接
            #   设为 none 强制收敛，不再指望弱模型自觉停手。
            _early_stop_next = False
            # ★ 策略 A：seed 预检索已带 file:line 行级证据（低分条目会隐藏行号，
            #   故出现 @ basename:line 即代表检索侧已判较可信）时，直接武装早停，
            #   让 iter#1 即 tool_choice=none 直答，省一轮 ~20s 工具往返 chat。
            if self.seed_direct_answer and seed_evidence and re.search(
                r"@\s*\S+\:\d+", seed_evidence,
            ):
                _early_stop_next = True
                logger.info(
                    "seed 已含行级证据，策略A 首轮直答（跳过工具轮）",
                    extra={
                        "event": "agentic_seed_direct_answer",
                        "worker_id": worker_id,
                    },
                )
            # ★ 空转回收：模型某轮只调 plan_update、既不检索也不作答时，退回该轮 iter
            #   额度、逼它把「改拆解 + 干活」合到同一次回复；上限 2 次防死循环。
            _plan_only_rewinds = 0
            _max_plan_only_rewinds = 2

            iteration = 0
            _real_iters_allowed = effective_max_iters
            _shell_retries = 0
            _max_shell_retries = 2  # 同一回次里最多回退 2 次防空壳，避免无限循环
            # ★ 循环内已因 LLM 总预算超时退出时，_force_final 再发一次 chat 几乎肯
            #   定同样超时（白白多烧 8s）——直接走 extractive 兜底，至少把已取回
            #   的证据拼成一份摘录式答案交给用户。
            _chat_timed_out = False
            while iteration < _real_iters_allowed:
                iteration += 1
                result.iterations = iteration
                # ★ 本轮开始时分解是否为空——用于区分“首轮建分解”与“已有分解却只回写”，
                #   前者不算空转不退回，后者才回收额度。
                _plan_was_empty = (
                    plan_store is None or plan_store.is_empty()
                )
                # ★ fanout 广播停止：同伴已抢占“已解决”，本轮不再开新 chat，直接带
                #   已收集的证据收尾（软停止，不硬杀 Task，对标 hermes cancel_event
                #   惯用法；区别于本 Worker 自己跑到 max_iters/预算耗尽的正常降级）。
                if stop_signal is not None and stop_signal.is_stopped:
                    result.stopped_early = True
                    logger.info(
                        "agentic 收到 fanout 停止信号，提前结束",
                        extra={
                            "event": "fanout_stopped_early",
                            "worker_id": worker_id,
                            "iter": iteration,
                            "reason": "peer_solved",
                        },
                    )
                    break
                # ★ 预算卡位：下一轮开始前若已超预算，直接强制收敛
                if _deadline_at is not None and time.monotonic() >= _deadline_at:
                    logger.warning(
                        "agentic 预算耗尽，强制收敛",
                        extra={
                            "event": "agentic_budget_exceeded",
                            "iter": iteration,
                            "budget_ms": self.budget_ms,
                        },
                    )
                    result.degraded = True
                    break
                # ★ 收尾预留：剩余预算不够再跑一轮工具+一轮收尾时，若已有证据，
                #   就提前进强制收尾（宁可少一轮工具也不能丢整个答案）。_deadline_at
                #   会在 _force_final 处加上预留时间，避免收尾 chat 被瞬时掐断。
                _remaining_now = (
                    None if _deadline_at is None
                    else _deadline_at - time.monotonic()
                )
                # 预留阈值 = 尾轮 _force_final 至少需要的 8s。低于此线时
                # 不再开新一轮 chat（非末轮时会剪去 8s → chat 不到 0.5s，
                # 一定失败），直接进 _force_final。末轮自己的 _is_last 分支会
                # 全量拿剩余，因此对尾轮无影响。
                _reserve_gate_s = _answer_reserve_s
                if (
                    _remaining_now is not None
                    and _remaining_now < _reserve_gate_s
                    and result.tool_calls > 0
                ):
                    logger.warning(
                        "agentic 逼近收尾预留线，提前强制收敛",
                        extra={
                            "event": "agentic_reserve_early_stop",
                            "iter": iteration,
                            "remaining_s": round(_remaining_now, 1),
                            "reserve_s": _answer_reserve_s,
                        },
                    )
                    break
                logger.info(
                    "agentic iteration",
                    extra={
                        "event": "agentic_iter_start",
                        "iter": iteration,
                        # 报当前有效上界（完成门为勾清单临时抬高后不再是固定 max_iters）
                        "max_iters": _real_iters_allowed,
                        # fanout 并发场景下用于 run_kb 按子 Agent 分组展示（非扇出时为空串，不影响旧行为）
                        "worker_id": worker_id,
                    },
                )
                with self.span("iteration", n=iteration):
                    try:
                        # 先算本轮是否末轮（后面预算分配要参考）
                        _is_last = (
                            iteration >= _real_iters_allowed or _early_stop_next
                        )
                        # ★ 给本轮 chat 的预算：
                        #   • 非末轮 = 剩余 - 预留（至少留 _answer_reserve_s 给
                        #     _force_final 尾轮写答案）。旧实现下 SiliconFlow 慢时，
                        #     chat 会吞完整个 budget → 到 _force_final 已过期 →
                        #     extractive 兑底 → 用户看到摘录而不是综述。
                        #   • 末轮 = 全剩余，因为本轮后不会回到循环，直接写答案；
                        #     本行失败也不需 _force_final（下一行会手动 extractive）。
                        if _deadline_at is None:
                            _remaining = None
                        elif _is_last:
                            _remaining = max(0.5, _deadline_at - time.monotonic())
                        else:
                            _remaining = max(
                                0.5,
                                _deadline_at - time.monotonic() - _answer_reserve_s,
                            )
                        # ★ 尾轮处理：不拆 tools（避免 Qwen3 将调用降级为
                        #   文本形式的 <tool_call>，造成答案泄漏）；而是插一条
                        #   硬提醒 + tool_choice=none 禁卡，模型不能发工具但
                        #   仍能看到列表。真需要时后面 _force_final 仍会无 tools 兼容。
                        #
                        # 非末轮不再注入任何额外的 per-round 提示（旧的【每轮回显
                        # 问题分解】/【首轮先想清楚】/【已收集证据】均已去除）——它
                        # 会每轮把 prompt 变肨、拖慢端到端；模型从自己的工具回包（tool
                        # 消息历史）已足够看到上一轮命中。只保留末轮的回看清单 + 预算卡位。
                        if plan_store is not None and not plan_store.is_empty() and _is_last:
                            # 末轮工具已禁（tool_choice=none）：只回看拆解查漏、直接作答，绝不
                            #   再要求 plan_update——否则模型会把工具调用降级成文本（如 DSML
                            #   标记）泄进最终答案。
                            messages.append(Message.user(
                                "【问题分解·参考】下面是目前已厘清的子问题，只供你回看有"
                                "没有漏答的关键限定：\n" + plan_store.render() + "\n"
                                "这已是最后一轮，请不要再调用任何工具（包括 plan_update），"
                                "直接基于已有证据给出完整答案，确保问题里每个关键限定都被"
                                "交代，末尾接 [CONFIDENCE:x.xx][INTENT:xxx][DOMAIN:xxx][RISK:xxx]。"
                            ))
                        # 末轮（_is_last 已在上方算出）：额外插一条【预算卡位】硬提醒，
                        #   配合 tool_choice=none 强制模型直接给文本答案。
                        if _is_last:
                            messages.append(Message.user(
                                "【预算卡位】这已是最后一轮工具机会。请优先直接给最"
                                "终答案（基于已有证据），不要再发起新的工具调用；"
                                "末尾接 [CONFIDENCE:x.xx][INTENT:xxx][DOMAIN:xxx][RISK:xxx]。"
                            ))
                        _chat_ts = time.perf_counter()  # ★ 记录本轮 chat 开始时间
                        _chat_coro = self.llm.chat(  # type: ignore[union-attr]
                            messages, provider=provider or self.provider,
                            temperature=0.1,
                            # ★ 截断 max_tokens 削减 SiliconFlow 响应时间。
                            #   工具轮只需 ~100 tok JSON tool_calls，256 充给CoT；
                            #   答案轮正文 ~400 tok + 尾注标签，600 够用。
                            max_tokens=600 if _is_last else 256,
                            # ★ 仅 early_stop armed 时不传 tools schema（无 schema
                            #   时模型无从发起）；循环自然到上限那一回仍传 tools，
                            #   避免破坏历史测试里"工具派发后 _force_final 补尾"的行为。
                            tools=None if _early_stop_next else tools,
                            tool_choice="none" if _is_last else "auto",
                        )
                        # ★ 单轮硬上限 40s。SiliconFlow Qwen3-8B 当下实测 21–28s，
                        #   40s 能安全覆盖服务侧抖动，且保证每轮迭代能拿到结果。
                        _ROUND_CAP_S = 40.0
                        _effective_timeout = (
                            _ROUND_CAP_S if _remaining is None
                            else min(_remaining, _ROUND_CAP_S)
                        )
                        resp = (
                            await asyncio.wait_for(_chat_coro, timeout=_effective_timeout)
                            if _remaining is not None else await _chat_coro
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "agentic 单轮 chat 超时（cap=%ss，iter#%d）",
                            _effective_timeout, iteration,
                        )
                        self.count("tool_agent_llm_timeout")
                        # ★ 超时不一定是“没预算了”——可能是 SiliconFlow 单轮抽风。
                        #   如果还有 ≥ 8s 预算 + 非末轮 → 重试本轮。
                        #   否则 break 进 extractive。
                        _can_retry = (
                            _deadline_at is not None
                            and (_deadline_at - time.monotonic()) >= _answer_reserve_s + 8.0
                            and not _is_last
                        )
                        if _can_retry:
                            iteration -= 1  # 退回本轮额度，下一拍重走
                            result.iterations = iteration
                            logger.info("agentic 重试本轮 chat (iter--)")
                            continue
                        result.degraded = True
                        _chat_timed_out = True
                        break
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("ToolAgent 模型调用失败：%s", exc)
                        self.count("tool_agent_llm_errors")
                        result.degraded = True
                        break

                result.model = resp.model or result.model
                # ★ 记录 LLM 单次调用时延 + E2E TTFT
                _llm_event_extra: dict[str, Any] = {
                    "event": "llm_chat_done",
                    "iter": iteration,
                    "latency_s": round(resp.latency_s, 2),
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "worker_id": worker_id,
                }
                # 答案轮（无 tool_calls）：计算端到端 TTFT
                if not resp.wants_tools and (resp.text or "").strip():
                    # 基准优先用 ask() 入口写入的真实请求起点（ContextVar 跨
                    # fanout 子任务继承），取不到才回退 ToolAgent 内部起点。
                    _origin = get_request_origin() or _t_agent_start
                    _e2e_ttft = (_chat_ts - _origin) + resp.latency_s
                    _llm_event_extra["answer_e2e_ttft_s"] = round(_e2e_ttft, 2)
                logger.info("agentic llm_chat_done", extra=_llm_event_extra)
                # 思考过程（assistant 在发起工具前的自然语言）
                if resp.text and resp.text.strip():
                    logger.info(
                        "agentic thinking",
                        extra={
                            "event": "agentic_thinking",
                            "iter": iteration,
                            "thinking": resp.text.strip()[:600],
                            "worker_id": worker_id,
                        },
                    )
                # ★ 硬阻断（仅在证据已够时）：_early_stop_next 下不管模型回什么
                #   都不再派发 → 当最终答案处理。注意“iteration >= max_iters”
                #   不归入硬阻断（归入软卡），否则下一轮工具不能派发 →
                #   历史上 _force_final 补尾的脚本测试会多一个 extractive 回道。
                if _early_stop_next and resp.wants_tools:
                    logger.warning(
                        "agentic 末轮模型仍坚持发 tool_calls → 强制改写为答案",
                        extra={
                            "event": "agentic_last_iter_override",
                            "iter": iteration,
                            "attempts": len(resp.tool_calls or []),
                        },
                    )
                    self.count("tool_agent_last_iter_override")
                    # wants_tools 是 @property（基于 tool_calls）→ 不能直接赋值；
                    # 清空 tool_calls 就能让它变 False，下文会当作文本答案处理。
                    resp.tool_calls = []
                    # 模型工具回包时无文本→ 剥不出生成式答案；直接走
                    # extractive 兜底，至少把已取回的证据交给用户。
                    if not (resp.text or "").strip():
                        result.degraded = True
                        result.answer = self._synthesize_extractive_answer(
                            question, result,
                        )
                        break
                if not resp.wants_tools:
                    raw_answer = (resp.text or "").strip()
                    cleaned, _conf, _intent, _domain, _risk = _parse_self_report(
                        raw_answer,
                    )
                    # ★ 防空壳：模型有时仅吐标签行（尤其 iter1 前几项预筛信息
                    #   不足时）。剥后正文为空 → 不应当作正式答案处理，而是追
                    #   一条提醒后继续下一轮（仍有预算时），避免走 _force_final 时
                    #   因无工具结果而直接“信息不足”收束。
                    if not cleaned and iteration < effective_max_iters:
                        messages.append(Message("assistant", raw_answer))
                        messages.append(Message.user(
                            "上一回只递交了自报标签、没写实质答案。请先调用必要"
                            "工具获取证据（search_kb / grep / read_file），"
                            "或直接在正文中给出对问题的分析，不要只发空标签。"
                        ))
                        logger.info(
                            "agentic empty shell nudged",
                            extra={
                                "event": "agentic_empty_shell",
                                "iter": iteration,
                            },
                        )
                        # ★ 不消耗迭代额度：回退 iteration，下一轮 while 会重
                        #   新取同一序号。_shell_retries 上限 2，避免同一回次
                        #   反复无果。
                        if _shell_retries < _max_shell_retries:
                            _shell_retries += 1
                            result.iterations = max(0, iteration - 1)
                            iteration -= 1
                        continue
                    # ★ 满足才停（① + ③）：策略 A 直答若自报置信不足或呈
                    #   “资料未提及”式对冲，且仍有迭代/时间预算 → 不收敛，撤销
                    #   早停、下一轮恢复 tools，逼模型针对原问题关键限定真检索/精读。
                    #   仅在 _early_stop_next（策略 A 武装）时生效——脚本化单测不武装
                    #   策略 A，行为不变（零回归）。
                    if (
                        _early_stop_next
                        and (
                            _conf is None
                            or _conf < self.seed_direct_min_conf
                            or bool(_UNSATISFIED_RE.search(cleaned))
                        )
                        and iteration < effective_max_iters
                        and (
                            _deadline_at is None
                            or time.monotonic()
                            < _deadline_at - _answer_reserve_s
                        )
                    ):
                        _early_stop_next = False
                        messages.append(Message("assistant", raw_answer))
                        messages.append(Message.user(
                            "仅凭 seed 直答把握不足或结论为「资料未提及」。请勿"
                            "再直接收尾：调用 search_kb / grep / read_file 围绕"
                            f"原问题的关键限定（{question}）实际检索并精读原文，"
                            "找到确切依据后再作答，末尾接 [CONFIDENCE:x.xx]"
                            "[INTENT:xxx][DOMAIN:xxx][RISK:xxx]。"
                        ))
                        logger.info(
                            "策略 A 直答未达满足阈值，回退工具深挖",
                            extra={
                                "event": "agentic_seed_direct_unsatisfied_retry",
                                "iter": iteration,
                                "self_confidence": _conf,
                                "worker_id": worker_id,
                            },
                        )
                        self.count("tool_agent_seed_direct_retry")
                        continue
                    result.answer = cleaned
                    if _conf is not None:
                        result.self_confidence = _conf
                    result.intent_hint = _intent or result.intent_hint
                    result.domain_hint = _domain or result.domain_hint
                    result.risk_hint = _risk or result.risk_hint
                    logger.info(
                        "agentic final answer",
                        extra={
                            "event": "agentic_final",
                            "iter": iteration,
                            "self_confidence": result.self_confidence,
                            "intent_hint": result.intent_hint,
                            "domain_hint": result.domain_hint,
                            "risk_hint": result.risk_hint,
                            "answer_preview": cleaned[:400],
                            "worker_id": worker_id,
                        },
                    )
                    break

                # 记录 assistant 发起的工具调用（回填 wire 结构，供后续消息一致）
                wire_calls = self._wire_tool_calls(resp.tool_calls)
                messages.append(Message(
                    "assistant", resp.text or "", tool_calls=wire_calls,
                ))
                # 一次性先报 开始事件（保持与流式渲染习惯一致）
                for call in resp.tool_calls:
                    logger.info(
                        f"tool call: {call.name}",
                        extra={
                            "event": "tool_call",
                            "iter": iteration,
                            "tool": call.name,
                            "thinking": (resp.text or "").strip()[:600],
                            "tool_args": _truncate_args(call.arguments),
                            "worker_id": worker_id,
                        },
                    )

                # ★ 并行 dispatch：同一回里多个 tool_calls 同时开跑
                _cites_before_batch = len(result.file_citations)
                _t0 = time.monotonic()
                async def _run_one(_c: Any) -> tuple[Any, Any, int]:
                    _key = _tool_dedup_key(_c)
                    _lt = time.monotonic()
                    if _key and _key in _seen_tool_calls:
                        # 完全相同的调用已执行过 → 不重跑，直接回“勿重复”备忘。
                        _r = ToolResult(
                            content=(
                                "⚠ 你已执行过完全相同的调用（相同工具+相同参数），结果同"
                                "上，请勿重复。请更换关键词/工具推进清单，或基于已有证"
                                "据作答。"
                            ),
                            is_error=False,
                        )
                        return _c, _r, 0
                    _r = await self.executor.dispatch(_c, tool_ctx)
                    if _key:
                        _seen_tool_calls.add(_key)
                    _d = int((time.monotonic() - _lt) * 1000)
                    return _c, _r, _d

                paired = await asyncio.gather(
                    *(_run_one(c) for c in resp.tool_calls),
                    return_exceptions=True,
                )
                _batch_ms = int((time.monotonic() - _t0) * 1000)
                logger.info(
                    f"parallel dispatch {len(resp.tool_calls)} tools in {_batch_ms}ms",
                    extra={
                        "event": "tool_batch",
                        "iter": iteration,
                        "count": len(resp.tool_calls),
                        "batch_ms": _batch_ms,
                    },
                )

                # 按原顺序回敿 messages/transcript/evidence（保证 tool_call_id 对应）
                for entry in paired:
                    if isinstance(entry, Exception):
                        logger.warning("工具派发异常：%s", entry)
                        continue
                    call, tool_result, _dur = entry
                    result.tool_calls += 1
                    logger.info(
                        f"tool result: {call.name} -> {len(tool_result.content)}B",
                        extra={
                            "event": "tool_result",
                            "iter": iteration,
                            "tool": call.name,
                            "duration_ms": _dur,
                            "is_error": tool_result.is_error,
                            "size": len(tool_result.content),
                            "preview": tool_result.content[:280],
                            "worker_id": worker_id,
                        },
                    )
                    # ★ plan_update 改完清单后，把当前全量清单作为事件输出，供 run_kb 展示。
                    if (
                        call.name == "plan_update"
                        and plan_store is not None
                        and not tool_result.is_error
                    ):
                        logger.info(
                            "agentic plan updated",
                            extra={
                                "event": "agentic_plan",
                                "iter": iteration,
                                "plan": plan_store.render(),
                                "summary": plan_store.counts(),
                            },
                        )
                    self._collect_citations(
                        tool_result, seen_cites, result.file_citations,
                    )
                    # 收集非错工具输出作为证据（agentic 模式下供校验器参考）
                    if not tool_result.is_error and tool_result.content:
                        result.evidence.append({
                            "tool": call.name,
                            "args": dict(call.arguments or {}),
                            "content": tool_result.content[:4000],
                        })
                    result.transcript.append({
                        "tool": call.name,
                        "args": call.arguments,
                        "is_error": tool_result.is_error,
                        "preview": tool_result.content[:300],
                    })
                    messages.append(Message.tool_result(
                        tool_result.content, tool_call_id=call.id, name=call.name,
                    ))

                # ★ 早停触发条件（新设计）：不再看“citations 数量”，而是看
                #   “剩余预算是否不够再跑一轮”。这样模型可以自由探索多轮工具，
                #   直到预算接近耗尽才强制收敛。
                #   阈值 45s = 留 1 轮 chat(40s) + 收尾(5s)。
                _budget_low = (
                    _deadline_at is not None
                    and (_deadline_at - time.monotonic()) < 45.0
                )
                if (
                    len(result.file_citations) >= 2
                    and result.tool_calls > 0
                    and _budget_low
                ):
                    _early_stop_next = True
                    logger.info(
                        "agentic 预算低位且证据已足，下一轮禁工具强制收敛",
                        extra={
                            "event": "agentic_early_stop_armed",
                            "iter": iteration,
                            "cites": len(result.file_citations),
                            "remaining_s": round(_deadline_at - time.monotonic(), 1),
                        },
                    )
                _plan_only_this_round = (
                    len(resp.tool_calls) == 1
                    and resp.tool_calls[0].name == "plan_update"
                    and not (resp.text and len(resp.text.strip()) >= 30)
                    and not _plan_was_empty  # 首轮建分解不算空转
                )
                if (
                    _plan_only_this_round
                    and _plan_only_rewinds < _max_plan_only_rewinds
                    and iteration < _real_iters_allowed
                ):
                    _plan_only_rewinds += 1
                    _refund_iter = iteration
                    iteration -= 1
                    result.iterations = iteration
                    messages.append(Message.user(
                        "上一轮只更新了问题分解、没有实际检索或作答。拆解更新必须与"
                        "当轮的检索/作答在同一次回复里同批发出（先反思→plan_update"
                        " merge→同批调去查/给答案），不要单独占一轮只改拆解。"
                    ))
                    logger.info(
                        "agentic plan-only round refunded",
                        extra={
                            "event": "agentic_plan_only_refund",
                            "iter": _refund_iter,
                            "rewinds": _plan_only_rewinds,
                        },
                    )

            # 循环内从未收敛出答案（耗尽 max_iters 或异常）→ 去工具强制收尾
            # ★ 但因同伴抢占“已解决”而 stopped_early 的 Worker 不走收尾：答案交给
            #   胜者，自己只提供已收集的中间证据（避免白烧一次无工具 chat）。
            if not result.answer and not result.stopped_early:
                result.degraded = True
                # ★ LLM 已经因预算超时断开 → 再发一次 force_final chat 大概率同样
                #   超时（多烧 8s）；直接走 extractive 兜底。无超时时才尝试收尾 chat。
                if _chat_timed_out:
                    result.answer = self._synthesize_extractive_answer(question, result)
                else:
                    # ★ 给收尾 chat 至少预留 _answer_reserve_s（在原 deadline 基础上外
                    #   放），否则预算刚耗尽时 _remaining2=0.5s，收尾 chat 必被瞬时
                    #   掐断 → 证据全丢、回“信息不足”。收尾轮是无工具的单次 chat，
                    #   比多花 10s 拿不到答案划算。
                    _remaining2 = (
                        None if _deadline_at is None
                        else max(
                            _answer_reserve_s,
                            _deadline_at - time.monotonic(),
                        )
                    )
                    forced = await self._force_final(
                        messages, provider, remaining=_remaining2,
                    )
                    cleaned, _conf, _intent, _domain, _risk = _parse_self_report(forced)
                    result.answer = cleaned or self._synthesize_extractive_answer(
                        question, result,
                    )
                    if _conf is not None:
                        result.self_confidence = _conf
                    result.intent_hint = _intent or result.intent_hint
                    result.domain_hint = _domain or result.domain_hint
                    result.risk_hint = _risk or result.risk_hint

            # ★ Router 链降级到 echo 时（openai 超时后回退），resp 不为空但
            #   answer 是 EchoLLM 兜底句（“当前未配置真实大模型...”），对用户
            #   而言跟“信息不足”一样没用。此时直接换成 extractive 兜底，
            #   把已经搜到的手册证据实实地端上桌。
            if result.answer and (
                (result.model or "").lower() == "echo"
                or "EchoLLM 离线兜底" in result.answer
                or "当前未配置真实大模型" in result.answer
            ):
                result.degraded = True
                result.answer = self._synthesize_extractive_answer(question, result)
                result.model = "echo->extractive"

            self._correct_truncated_identifiers(result)
            # ★ 把最终清单状态落到结果/trace，供 run_kb 展示与事后检查。
            if plan_store is not None:
                result.plan = plan_store.render()
                result.plan_unfinished = len(plan_store.active_items())
            node.attributes.update(
                iterations=result.iterations, tool_calls=result.tool_calls,
                citations=len(result.file_citations), degraded=result.degraded,
                plan_items=(plan_store.counts().get("total", 0)
                            if plan_store is not None else 0),
                plan_unfinished=result.plan_unfinished,
            )
            self.observe("tool_agent_iterations", result.iterations)
            self.count("tool_agent_runs")
            # ★ fanout “先到先得”广播：无论答案来自自然收敛还是 _force_final 兑底，
            #   都统一在退出前拿最终 self_confidence 判定一次（两处分支都能命中，
            #   避免只在循环内检查而漏掉兑底路径达标的情况）。
            if (
                stop_signal is not None and result.answer
                and result.self_confidence is not None
                and result.self_confidence >= stop_signal.threshold
            ):
                if stop_signal.mark_solved(worker_id, result):
                    logger.info(
                        "agentic worker 达标，广播停止其他子 Agent",
                        extra={
                            "event": "fanout_worker_marked_solved",
                            "worker_id": worker_id,
                            "confidence": result.self_confidence,
                            "threshold": stop_signal.threshold,
                        },
                    )
            return result

    # ------------------------------------------------------------------
    def _correct_truncated_identifiers(self, result: ToolRunResult) -> None:
        """确定性「代码标识符截断」订正（不依赖模型自觉）。

        若最终回答里出现结构化代码标识符（含 `.` 或 `[]`，如系统变量
        `_X[0].Field`），但该串并未逐字出现在检索证据中，而证据里存在**唯一**
        一个逐段以它为前缀的更长标识符，则判定用户给的是被截断/写错的符号，
        把回答中的该串订正为手册原文全名，并附一行透明说明。仅处理代码标识符，
        不触碰中文领域名词。
        """
        ans = result.answer or ""
        if not ans or not result.evidence:
            return
        ev = "\n".join(str(e.get("content", "")) for e in result.evidence)
        ev_idents = set(_IDENT_RE.findall(ev))
        if not ev_idents:
            return
        fixes: list[tuple[str, str]] = []
        for tok in dict.fromkeys(_IDENT_RE.findall(ans)):
            if tok in ev:  # 手册里确有此写法，保持不动
                continue
            cands = [e for e in ev_idents if _is_truncation(tok, e)]
            if len(cands) == 1:
                fixes.append((tok, cands[0]))
        if not fixes:
            return
        for tok, full in fixes:
            ans = ans.replace(tok, full)
        note = "；".join(f"`{t}` 手册准确写法为 `{f}`" for t, f in fixes)
        result.answer = (
            ans.rstrip()
            + f"\n\n📖 术语订正：{note}（已按手册原文校正标识符拼写）。"
        )
        logger.info(
            "agentic identifier truncation corrected",
            extra={"event": "agentic_ident_fix", "fixes": dict(fixes)},
        )

    # ------------------------------------------------------------------
    def _resolve_roots(self, ctx: AgentContext) -> list[str]:
        try:
            return list(self._roots_resolver(ctx) or [])
        except Exception:  # noqa: BLE001
            logger.exception("解析工具授权目录失败")
            return []

    def _with_skill_roots(self, roots: list[str]) -> list[str]:
        """把技能层级根目录并入 jail（去重、保序）。"""
        if not self.skill_registry:
            return roots
        merged = list(roots)
        existing = {str(Path(str(r))) for r in merged}
        for skill_root in (self.skill_registry.roots() or []):
            key = str(Path(str(skill_root)))
            if key not in existing:
                existing.add(key)
                merged.append(skill_root)
        return merged

    def _system_prompt(
        self, ctx: ToolContext, registry: Any | None = None, *,
        intent_domain: str | None = None,
        intent_label: str | None = None,
    ) -> str:
        roots = ", ".join(ctx.roots) or "（无）"
        skill_block = ""
        if self.skill_registry:
            rendered = self.skill_registry.render_prompt(
                session_id=ctx.session_id, tenant_id=ctx.tenant_id,
            )
            if rendered:
                skill_block = f"\n{rendered}\n"
        intent_block = ""
        if intent_domain or intent_label:
            intent_block = (
                f"\n【意图】domain={intent_domain or '未知'}"
                f" intent={intent_label or '未知'}。\n"
            )
        return (
            INOVANCE_PERSONA + "\n"
            "通过工具对知识库做深度问答。\n"
            f"{intent_block}"
            f"{skill_block}"
            f"授权目录：{roots}\n"
            "规则：\n"
            "1. 调工具前用1句中文说明理由，不得空content只回tool_calls。\n"
            "2. 结论以原文为准；未核验的行号/参数标注推断。\n"
            "3. 问题含多个关键概念时，先用语义检索发现各概念的相关内容，再用精确匹配确认细节。\n"
            "4. 回包0命中必换招；证据够即停直接给答案。\n"
            "5. 末轮回答末尾追加 [CONFIDENCE:x.xx][INTENT:...][DOMAIN:...][RISK:...]。\n"
            "6. 来源写自然中文“《手册名》第X行”，不得出现工具名/参数名。"
        )

    @staticmethod
    def _format_tool_line(tool: Any) -> str:
        """把工具 + 能力元数据格式化为一行。"""
        parts = [f"- {tool.name}: {tool.description}"]
        if getattr(tool, "best_for", ()):
            parts.append("  ★ 擅长： " + " / ".join(tool.best_for))
        if getattr(tool, "avoid_for", ()):
            parts.append("  ✗ 避免： " + " / ".join(tool.avoid_for))
        return "\n".join(parts)

    def _user_prompt(
        self, question: str, seed_evidence: str, *,
        memory_block: str = "",
        output_hint: str = "",
        followup_issues: list[str] | None = None,
    ) -> str:
        parts: list[str] = []
        if memory_block:
            parts.append(f"【会话记忆】\n{memory_block}")
        if seed_evidence:
            parts.append(
                f"【初步线索（低分条目仅给摘要，请用 search_kb/grep 发现具体内容）】\n{seed_evidence}"
            )
        if followup_issues:
            parts.append(
                "【上一轮答案未通过校验，请重点修正以下问题后重新检索作答】\n"
                + "\n".join(f"  - {i}" for i in followup_issues)
            )
        parts.append(f"【问题】\n{question}")
        if output_hint:
            parts.append(f"【输出格式要求】\n{output_hint}")
        return "\n\n".join(parts)

    @staticmethod
    def _wire_tool_calls(calls: list[Any]) -> list[dict[str, Any]]:
        """把归一化 ToolCall 还原为 OpenAI assistant.tool_calls 线格式。"""
        wire: list[dict[str, Any]] = []
        for c in calls:
            wire.append({
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": json.dumps(c.arguments, ensure_ascii=False),
                },
            })
        return wire

    @staticmethod
    def _collect_citations(
        tool_result: Any, seen: set[str], out: list[dict[str, Any]],
    ) -> None:
        if tool_result.is_error:
            return
        for m in _CITE_RE.finditer(tool_result.content):
            path, line = m.group(1), m.group(2)
            key = f"{path}:{line or ''}"
            if key in seen:
                continue
            seen.add(key)
            out.append({"path": path, "line": int(line) if line else None})

    def _synthesize_extractive_answer(
        self, question: str, result: "ToolRunResult",
    ) -> str:
        """LLM 彻底超时时不冷回“信息不足”：从已取回的工具回包里拼一份
        摘录式答案（每工具首段 + 行级引用），至少把已检索到的原文交给用户。

        无工具回包时仍回退为原来的“信息不足”。
        """
        if not result.evidence:
            return ""
        parts: list[str] = [
            f"（LLM 因预算超时未综述，以下为已检索到的原文摘录，供参考）",
            f"问题：{question}",
            "",
        ]
        _seen_content: set[str] = set()
        for e in result.evidence:
            content = str(e.get("content", "") or "").strip()
            tool = str(e.get("tool", "") or "tool")
            if not content or content in _seen_content:
                continue
            _seen_content.add(content)
            # 每个工具回包取前 800 字（已含行级引用），多工具时不会注卡总长
            excerpt = content if len(content) <= 800 else content[:800] + "…"
            parts.append(f"【{tool}】\n{excerpt}\n")
        cites = result.file_citations or []
        if cites:
            lines = [
                f"- {c.get('path')}{':' + str(c.get('line')) if c.get('line') else ''}"
                for c in cites[:8]
            ]
            parts.append("【行级引用】\n" + "\n".join(lines))
        return "\n".join(parts)

    async def _force_final(
        self, messages: list[Any], provider: str | None,
        *, remaining: float | None = None,
    ) -> str:
        """达迭代上限仍未作答：去工具再问一次，要求基于已获证据直接给答案。

        remaining: 剩余预算（秒）。None 为不限；超出则归为 ""
        """
        try:
            coro = self.llm.chat(  # type: ignore[union-attr]
                messages + [Message.user(
                    "已达到工具调用上限。请仅基于以上已获得的信息，尽快给出带 "
                    "`文件名:行号` 引用的最终答案；信息确实不足时明确说明。"
                    "末尾接 [CONFIDENCE:x.xx][INTENT:xxx][DOMAIN:xxx][RISK:xxx] 标签。",
                )],
                provider=provider, temperature=0.2,
                max_tokens=600,  # ★ 答案轮封顶 600 tok
            )
            # 单轮硬上限 40s，与 _ROUND_CAP_S 对齐
            _cap = min(remaining, 40.0) if remaining is not None else 40.0
            resp = await asyncio.wait_for(coro, timeout=_cap)
            return (resp.text or "").strip()
        except asyncio.TimeoutError:
            logger.warning("ToolAgent 强制收尾因预算耗尽而超时")
            return ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("ToolAgent 强制收尾失败：%s", exc)
            return ""
