"""全局装配：把配置翻译为一整张协作对象图，注入 Kernel.services。

这是微内核「延迟装配」的落点——:meth:`fusion_rag.core.kernel.Kernel.start`
在事件循环内 ``await bootstrap_kernel(self)``，此处按依赖顺序构建：

    配置 → 沙箱/线程池/存储 → 安全 → 可观测 → 模型/嵌入/检索
         → 分块/记忆/规划 → 各角色 Agent → Orchestrator/Indexer
         → 后台巡检与反馈沉淀任务

设计原则：

- **模型无绑定**：LLM / Embedding 全部按 ``config`` 声明式构建，未配置真实
  模型时自动回落到离线兜底（EchoLLM / HashEmbedding），保证零依赖可跑通。
- **插件化**：所有可替换能力注册进 :class:`PluginRegistry`，运维层可巡检、
  后续可热替换。
- **韧性**：任一可选增强（reranker / sentence-transformers / aiohttp）缺失
  只降级、不阻断启动。
- **生命周期**：需要释放的服务实现 ``aclose``，由 Kernel 逆序优雅关闭；
  后台任务登记到 ``services["background_tasks"]`` 供 Kernel 统一取消。
"""

from __future__ import annotations

from typing import Any

from .agents.intent import IntentAgent
from .agents.memory_agent import MemoryAgent
from .agents.ops import OpsAgent
from .agents.orchestrator import Orchestrator
from .agents.reasoning import ReasoningAgent
from .agents.retrieval import RetrievalAgent
from .agents.validator import ValidatorAgent
from .chunking.chunker import Chunker
from .core.config import Config
from .core.logging import get_logger
from .core.sandbox import CpuExecutor, Sandbox
from .embedding.base import EmbeddingBase
from .knowledge.feedback import FeedbackCollector
from .knowledge.indexer import KnowledgeIndexer
from .llm.base import LLMBase
from .llm.router import LLMRouter
from .memory.long_term import LongTermMemory
from .memory.manager import MemoryManager
from .memory.short_term import ShortTermMemory
from .observability.alerter import Alerter
from .observability.metrics import MetricsRegistry
from .observability.tracer import Tracer
from .planning.context_offload import ContextOffloader
from .planning.decomposer import TaskDecomposer
from .retrieval.bm25 import BM25Index
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import (
    CrossEncoderReranker,
    HeuristicReranker,
    LLMReranker,
    RerankerBase,
)
from .retrieval.vector_store import LocalVectorStore
from .security.access_control import AccessController
from .security.redaction import Redactor
from .storage.sqlite_store import SQLiteStore

logger = get_logger(__name__)

if False:  # pragma: no cover - 仅用于类型检查，避免运行时循环导入
    from .core.kernel import Kernel


# ======================================================================
# 入口
# ======================================================================
async def bootstrap_kernel(kernel: "Kernel") -> None:
    """按配置装配全部服务并注入 ``kernel.services``。"""
    cfg = kernel.config
    svc: dict[str, Any] = kernel.services
    data_dir = cfg.data_dir

    # ---- 1) 运行时底座：线程池 / 沙箱 / 存储（最先建，最后关）----------
    cpu = CpuExecutor(int(cfg.get("kernel.cpu_workers", 8)))
    svc["cpu_executor"] = cpu

    sandbox = Sandbox(
        max_concurrent=int(cfg.get("kernel.max_concurrent_requests", 64)),
        request_timeout=float(cfg.get("kernel.request_timeout", 120)),
    )
    svc["sandbox"] = sandbox

    store = SQLiteStore(
        cfg.path("storage.sqlite.path"),
        wal=bool(cfg.get("storage.sqlite.wal", True)),
        busy_timeout_ms=int(cfg.get("storage.sqlite.busy_timeout_ms", 5000)),
    )
    await store.start()
    svc["store"] = store

    # ---- 2) 安全：租户 ACL + 脱敏 --------------------------------------
    access = AccessController.from_config(cfg.section("security"))
    for tenant in access.all_tenants():
        rate, burst = access.rate_limit(tenant)
        sandbox.register_tenant(tenant, rate, burst)
    redactor = Redactor(enabled=bool(cfg.get("security.redaction_enabled", True)))
    svc["access"] = access
    svc["redactor"] = redactor

    # ---- 3) 可观测：指标 / 轨迹 / 告警 ---------------------------------
    metrics = MetricsRegistry()
    tracer = Tracer(
        cfg.get("observability.trace_path"), store,
        sample_rate=float(cfg.get("observability.trace_sample_rate", 1.0)),
        keep_error_traces=bool(cfg.get("observability.trace_keep_error_traces", True)),
        metrics=metrics,
    )
    alert_cfg = cfg.section("observability.alert")
    alerter = Alerter(
        metrics,
        kernel.events,
        rules=Alerter.default_rules(
            error_rate_threshold=float(alert_cfg.get("error_rate_threshold", 0.2)),
            p95_latency_threshold=float(alert_cfg.get("p95_latency_threshold", 30.0)),
        ),
    )
    svc["metrics"] = metrics
    svc["tracer"] = tracer
    svc["alerter"] = alerter

    # ---- 4) 模型层：LLM Router + Embedding（模型无绑定）----------------
    llm_router = _build_llm_router(cfg, kernel)
    svc["llm_router"] = llm_router
    embedding = _build_embedding(cfg)
    svc["embedding"] = embedding

    # ---- 5) 检索层：向量库 + BM25 + 混合检索 + 精排 --------------------
    collection = str(cfg.get("retrieval.collection", "knowledge"))
    vector_store = LocalVectorStore(cfg.path("storage.vector_store.path"), cpu_executor=cpu)
    bm25 = BM25Index(data_dir / "index" / "bm25.json", cpu_executor=cpu)
    svc["vector_store"] = vector_store
    svc["bm25"] = bm25

    reranker = _build_reranker(cfg, llm_router)
    retriever = HybridRetriever(
        embedding, vector_store, bm25, collection, reranker,
        vector_recall=int(cfg.get("retrieval.vector_recall", 20)),
        bm25_recall=int(cfg.get("retrieval.bm25_recall", 20)),
        rrf_k=int(cfg.get("retrieval.rrf_k", 60)),
        score_threshold=float(cfg.get("retrieval.score_threshold", 0.0)),
    )
    svc["retriever"] = retriever

    chunker = Chunker(
        max_tokens=int(cfg.get("retrieval.chunk.max_tokens", 400)),
        overlap_tokens=int(cfg.get("retrieval.chunk.overlap_tokens", 60)),
    )
    svc["chunker"] = chunker

    # ---- 6) 记忆层：短期 + 长期 + 编排 ---------------------------------
    short_term = ShortTermMemory(store, int(cfg.get("planning.short_term_rounds", 6)))
    long_term = LongTermMemory(store)
    memory_manager = MemoryManager(
        short_term, long_term, llm=llm_router,
        token_budget=int(cfg.get("planning.context_token_budget", 6000)),
    )
    svc["memory_manager"] = memory_manager

    # ---- 7) 规划层：任务拆解 + 上下文卸载 ------------------------------
    decomposer = TaskDecomposer(
        llm=llm_router,
        enabled=bool(cfg.get("planning.enabled", True)),
        max_sub_questions=int(cfg.get("planning.max_sub_questions", 5)),
        decompose_threshold=int(cfg.get("planning.decompose_threshold", 30)),
        hitl_enabled=bool(cfg.get("security.hitl_enabled", False)),
    )
    offloader = ContextOffloader(data_dir / "offload")
    svc["decomposer"] = decomposer
    svc["offloader"] = offloader

    # ---- 8) 多智能体角色 ------------------------------------------------
    top_k = int(cfg.get("retrieval.top_k", 6))
    intent = IntentAgent(llm=llm_router, tracer=tracer, metrics=metrics)
    retrieval_agent = RetrievalAgent(
        retriever, llm=llm_router, tracer=tracer, metrics=metrics, default_top_k=top_k,
    )
    reasoning = ReasoningAgent(
        llm=llm_router, tracer=tracer, metrics=metrics, offloader=offloader,
        token_budget=int(cfg.get("planning.context_token_budget", 6000)),
    )
    validator = ValidatorAgent(
        llm=llm_router, tracer=tracer, metrics=metrics,
        enabled=bool(cfg.get("validation.enabled", True)),
        min_confidence=float(cfg.get("validation.min_confidence", 0.5)),
    )
    memory_agent = MemoryAgent(memory_manager, tracer=tracer, metrics=metrics)

    # ---- 8.5) Agentic 文件工具（运行时自主读原文，可选）----------------
    tool_agent = None
    agentic_tools = False
    if bool(cfg.get("tools.enabled", True)):
        tool_agent = _build_tools(
            cfg, kernel, llm_router, tracer, metrics, cpu, retriever=retriever,
        )
        agentic_tools = bool(cfg.get("retrieval.agentic_tools", False))
    svc["tool_agent"] = tool_agent

    # ---- 9) 编排器（问答主链路）---------------------------------------
    orchestrator = Orchestrator(
        intent=intent, decomposer=decomposer, retrieval=retrieval_agent,
        reasoning=reasoning, validator=validator, memory=memory_agent,
        access=access, redactor=redactor, sandbox=sandbox, store=store,
        tracer=tracer, metrics=metrics, events=kernel.events, llm_router=llm_router,
        top_k=top_k,
        validation_enabled=bool(cfg.get("validation.enabled", True)),
        max_retry=int(cfg.get("validation.max_retry", 2)),
        redaction_enabled=bool(cfg.get("security.redaction_enabled", True)),
        hitl_enabled=bool(cfg.get("security.hitl_enabled", False)),
        tool_agent=tool_agent,
        agentic_tools=agentic_tools,
        retrieval_mode=str(cfg.get("retrieval.mode", "pipeline")),
        agentic_retry=int(cfg.get("retrieval.agentic_retry", 1)),
        quota_scope=str(cfg.get("kernel.quota_scope", "tenant")),
        seed_llm_decompose=bool(
            cfg.get("retrieval.seed_llm_decompose", True)
        ),
        fanout_config=cfg.section("retrieval.fanout") if isinstance(
            cfg.get("retrieval.fanout"), dict,
        ) else {},
    )
    svc["orchestrator"] = orchestrator

    # ---- 10) 知识索引器（增量入库）------------------------------------
    indexer = KnowledgeIndexer(
        embedding=embedding, vector_store=vector_store, bm25=bm25,
        chunker=chunker, store=store, collection=collection,
        redactor=redactor, access=access,
    )
    await indexer.ensure_collection()
    svc["indexer"] = indexer

    # ---- 11) 运维 Agent + 反馈沉淀（后台任务）-------------------------
    ops = OpsAgent(
        metrics, alerter, store, sandbox, kernel.registry,
        metrics_path=cfg.get("observability.metrics_path"),
        interval=float(cfg.get("observability.ops_interval", 30.0)),
        tracer=tracer,
        health_probes={
            "llm": llm_router.health,
            "store": lambda: not getattr(store, "_closed", True),
        },
    )
    feedback = FeedbackCollector(
        store, long_term,
        tenants=access.all_tenants(),
        interval=float(cfg.get("observability.feedback_interval", 300.0)),
        low_confidence=float(cfg.get("validation.min_confidence", 0.5)),
    )
    svc["ops"] = ops
    svc["feedback"] = feedback

    background = [ops.start(), feedback.start()]
    svc["background_tasks"] = background

    logger.info(
        "装配完成：collection=%s llm=%s embedding=%s(dim=%d) reranker=%s 租户=%d",
        collection, llm_router.default, embedding.name, embedding.dimensions,
        getattr(reranker, "name", "none"), len(access.all_tenants()),
    )


# ======================================================================
# 组件工厂
# ======================================================================
def _build_llm_router(cfg: Config, kernel: "Kernel") -> LLMRouter:
    """按 ``llm.providers`` 声明构建各模型适配器并注册为插件。"""
    from .llm.echo import EchoLLM
    from .llm.openai_compatible import OpenAICompatibleLLM

    providers_cfg = cfg.section("llm.providers") or {"echo": {"type": "echo"}}
    router_cfg = cfg.section("llm.router")
    default = str(router_cfg.get("default", "echo"))
    # ★ 全局默认的按租户/请求子额度（provider 未自行声明时注入；None/0=关闭）。
    default_tenant_max = cfg.get("llm.tenant_max_concurrency")

    providers: dict[str, LLMBase] = {}
    for name, spec in providers_cfg.items():
        spec = dict(spec or {})
        ptype = str(spec.pop("type", "echo")).lower()
        if (
            ptype != "echo"
            and "tenant_max_concurrency" not in spec
            and default_tenant_max
        ):
            spec["tenant_max_concurrency"] = int(default_tenant_max)
        try:
            if ptype == "echo":
                llm: LLMBase = EchoLLM(**spec)
            elif ptype in ("openai", "openai_compatible", "deepseek", "ollama", "vllm", "qwen"):
                llm = OpenAICompatibleLLM(
                    base_url=str(spec.pop("base_url", "")),
                    model=str(spec.pop("model", name)),
                    name=name,
                    **spec,
                )
            else:
                logger.warning("未知 LLM 类型 %s（provider=%s），跳过", ptype, name)
                continue
        except Exception:  # noqa: BLE001
            logger.exception("构建 LLM provider=%s 失败，跳过", name)
            continue
        providers[name] = llm
        kernel.registry.register("llm", name, lambda _n=name: providers[_n], replace=True)

    # 兜底：确保 default 可用，且至少有一个 provider
    if default not in providers:
        logger.warning("默认模型 %s 未成功构建，回落到 echo", default)
        providers.setdefault("echo", EchoLLM())
        default = "echo"
    if "echo" not in providers:
        providers["echo"] = EchoLLM()

    fallback = router_cfg.get("fallback_chain") or None
    if fallback:
        # 保证降级链末尾始终有离线兜底
        fallback = [p for p in list(fallback) if p in providers]
        if "echo" not in fallback:
            fallback.append("echo")
    return LLMRouter(
        providers, default=default, fallback_chain=fallback,
        retries=int(router_cfg.get("retries", 2)),
    )


def _build_embedding(cfg: Config) -> EmbeddingBase:
    """按 ``embedding.provider`` 构建嵌入器；增强依赖缺失则回落 HashEmbedding。

    provider 专属参数读自嵌套子节（如 ``embedding.openai.*``），并兼容
    扁平写法（``embedding.base_url``）。
    """
    from .embedding.hash_embedding import HashEmbedding

    provider = str(cfg.get("embedding.provider", "hash")).lower()
    dimensions = int(cfg.get("embedding.dimensions", 256))

    def _opt(section: str, key: str, default: Any = None) -> Any:
        """优先取嵌套子节，回落扁平键。"""
        val = cfg.get(f"embedding.{section}.{key}")
        if val is None:
            val = cfg.get(f"embedding.{key}")
        return default if val is None else val

    if provider == "hash":
        return HashEmbedding(dimensions=dimensions)
    if provider in ("openai", "openai_compatible"):
        from .embedding.openai_embedding import OpenAIEmbedding

        base_url = _opt("openai", "base_url")
        model = _opt("openai", "model")
        if not base_url or not model:
            logger.warning("openai 嵌入缺少 base_url/model，回落 hash")
            return HashEmbedding(dimensions=dimensions)
        return OpenAIEmbedding(
            base_url=str(base_url), model=str(model),
            api_key=str(_opt("openai", "api_key", "") or ""),
            api_key_env=_opt("openai", "api_key_env"),
            dimensions=int(_opt("openai", "dimensions", dimensions)),
            timeout=float(_opt("openai", "timeout", 30.0)),
            max_concurrency=int(_opt("openai", "max_concurrency", 8)),
            tenant_max_concurrency=(
                int(_opt("openai", "tenant_max_concurrency",
                         cfg.get("embedding.tenant_max_concurrency", 0) or 0))
                or None
            ),
            batch_size=int(_opt("openai", "batch_size", 64)),
        )
    if provider in ("sentence_transformers", "st", "local"):
        try:
            from .embedding.sentence_transformers_embedding import (
                SentenceTransformerEmbedding,
            )

            return SentenceTransformerEmbedding(
                model=str(_opt("sentence_transformers", "model", "BAAI/bge-small-zh-v1.5")),
                dimensions=int(_opt("sentence_transformers", "dimensions", dimensions)),
                device=_opt("sentence_transformers", "device"),
            )
        except Exception:  # noqa: BLE001
            logger.exception("sentence-transformers 嵌入初始化失败，回落 hash")
            return HashEmbedding(dimensions=dimensions)

    logger.warning("未知嵌入 provider=%s，回落 hash", provider)
    return HashEmbedding(dimensions=dimensions)


def _build_reranker(cfg: Config, llm_router: LLMRouter) -> RerankerBase | None:
    """按 ``retrieval.reranker`` 构建精排器（null/heuristic/llm/cross_encoder）。"""
    kind = cfg.get("retrieval.reranker")
    if not kind:
        return None
    kind = str(kind).lower()
    if kind == "llm":
        return LLMReranker(llm_router, fallback=HeuristicReranker())
    if kind == "heuristic":
        return HeuristicReranker()
    if kind in ("cross_encoder", "bge", "reranker_api"):
        rr_cfg = cfg.get("retrieval.reranker_config") or {}
        # 默认复用 embedding 的 base_url + api_key
        emb_cfg = cfg.section("embedding").get("openai") or {}
        base_url = str(rr_cfg.get("base_url") or emb_cfg.get("base_url") or "")
        api_key = str(rr_cfg.get("api_key") or emb_cfg.get("api_key") or "")
        model = str(rr_cfg.get("model") or "BAAI/bge-reranker-v2-m3")
        timeout = float(rr_cfg.get("timeout") or 15.0)
        if not base_url:
            logger.warning("cross_encoder reranker 缺少 base_url，降级启发式")
            return HeuristicReranker()
        return CrossEncoderReranker(
            base_url=base_url, api_key=api_key, model=model,
            timeout=timeout, fallback=HeuristicReranker(),
        )
    logger.warning("未知 reranker=%s，使用 heuristic", kind)
    return HeuristicReranker()


def _build_tools(
    cfg: Config,
    kernel: "Kernel",
    llm_router: LLMRouter,
    tracer: Tracer,
    metrics: MetricsRegistry,
    cpu: CpuExecutor,
    retriever: Any | None = None,
) -> Any:
    """按 ``tools`` 节构建只读文件工具集、调度器与 :class:`ToolAgent`。

    - 按 ``allow`` 白名单实例化工具，并注册进 PluginRegistry（kind=``tool``）；
    - 由 ``tools.roots``/``tools.kb_roots`` 计算每租户+知识库的 jail 根目录，
      缺省回落到 ``data_dir/knowledge``；
    - ``cfg.section`` 读取时 ``${data_dir}`` 已自动展开。
    """
    from .agents.tool_agent import ToolAgent
    from .tools.executor import ToolExecutor
    from .tools.file_tools import (
        FindTool,
        GlobTool,
        GrepTool,
        ListDirTool,
        ReadFileTool,
    )
    from .tools.registry import ToolRegistry
    from .tools.search_tool import SearchKBTool
    from .tools.plan_tool import PlanUpdateTool

    classes: dict[str, Any] = {
        "grep": GrepTool,
        "glob": GlobTool,
        "find": FindTool,
        "list_dir": ListDirTool,
        "read_file": ReadFileTool,
        "plan_update": PlanUpdateTool,
    }
    # 将混合检索注入为工具（需 retriever 实例，签名不统一故单独处理）
    if retriever is not None:
        classes["search_kb"] = lambda cpu_executor=None: SearchKBTool(
            retriever=retriever, cpu_executor=cpu_executor,
        )
    tools_cfg = cfg.section("tools")
    allow = [str(x) for x in (tools_cfg.get("allow") or list(classes))]

    registry = ToolRegistry()
    for name in allow:
        factory = classes.get(name)
        if factory is None:
            logger.warning("未知工具 %s，跳过", name)
            continue
        # 支持两种登记方式：直接类引用（文件工具）或 lambda 工厂（search_kb）
        if isinstance(factory, type):
            instance = factory(cpu_executor=cpu)
        else:
            instance = factory(cpu_executor=cpu)
        registry.register(instance)
        kernel.registry.register(
            "tool", name,
            (lambda _c=factory, _cpu=cpu: _c(cpu_executor=_cpu))
            if isinstance(factory, type)
            else (lambda _f=factory, _cpu=cpu: _f(cpu_executor=_cpu)),
            replace=True,
        )

    executor = ToolExecutor(
        registry, tracer=tracer, metrics=metrics,
        call_timeout=float(tools_cfg.get("call_timeout", 15)),
    )

    roots_map = {str(k): str(v) for k, v in (tools_cfg.get("roots") or {}).items()}
    kb_roots = {str(k): str(v) for k, v in (tools_cfg.get("kb_roots") or {}).items()}
    default_root = str(tools_cfg.get("default_root") or (cfg.data_dir / "knowledge"))

    def _resolve_roots(ctx: Any) -> list[str]:
        dirs: list[str] = []
        t_root = roots_map.get(ctx.tenant_id)
        if t_root:
            dirs.append(t_root)
        for kb in (ctx.kb_ids or []):
            kb_root = kb_roots.get(kb)
            if kb_root:
                dirs.append(kb_root)
        return dirs or [default_root]

    skill_registry = _build_skill_registry(cfg)
    return ToolAgent(
        llm=llm_router, tracer=tracer, metrics=metrics,
        registry=registry, executor=executor,
        max_iters=int(tools_cfg.get("max_iters", 6)),
        budget_ms=int(cfg.get("retrieval.agentic_budget_ms", 0) or 0),
        roots_resolver=_resolve_roots,
        skill_registry=skill_registry,
        plan_first=bool(cfg.get("retrieval.agentic_plan_first", True)),
        seed_direct_answer=bool(cfg.get("retrieval.agentic_seed_direct", True)),
        seed_direct_min_conf=float(
            cfg.get("retrieval.agentic_seed_direct_min_conf", 0.7) or 0.7
        ),
    )


def _build_skill_registry(cfg: Config) -> Any:
    """按 ``skills`` 节加载分层技能，返回 :class:`SkillRegistry`（无技能时 None）。

    ``skills.dirs`` 为有序根目录列表（相对路径以 ``data_dir`` 为基准），后层同名
    覆盖前层。目录经 ``cfg.section`` 读取时 ``${data_dir}`` 已自动展开。
    """
    from pathlib import Path

    from .skills.loader import load_skills
    from .skills.registry import SkillRegistry

    scfg = cfg.section("skills") or {}
    if not bool(scfg.get("enabled", True)):
        return None
    raw_dirs = scfg.get("dirs") or []
    layers: list[tuple[str, Path]] = []
    for entry in raw_dirs:
        path = Path(str(entry)).expanduser()
        if not path.is_absolute():
            path = cfg.data_dir / path
        layers.append((path.name or str(path), path))
    if not layers:
        return None

    skills = load_skills(layers)
    roots = [str(p.resolve()) for _, p in layers if p.is_dir()]
    if not skills:
        logger.info("skills.dirs 未扫描到任何技能（%s）", [str(p) for _, p in layers])
        return None
    logger.info("已加载 %d 个技能：%s", len(skills), ", ".join(sorted(skills)))
    return SkillRegistry(skills.values(), roots=roots)
