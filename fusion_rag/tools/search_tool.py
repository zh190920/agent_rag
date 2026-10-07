"""把知识库混合检索（BM25 + Vector + Rerank）包装成一个 agentic 工具。

设计意图：让大模型自主决定"什么时候需要重新检索、用什么查询词"，
把检索从**固定前置阶段**降级为**可选能力**，与 grep/read_file 平级。

模型可以在 agentic 循环中：
- 用不同的关键词调 ``search_kb`` 二次召回（比初次检索的固定 query 更灵活）
- 拿到 chunk 后再 ``read_file`` 精读原文，或 ``grep`` 定位具体段落
- 简单问题直接调 ``search_kb`` 拿证据就能答，无需绕远路
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from ..core.logging import get_logger
from .base import Tool, ToolContext, ToolResult

logger = get_logger(__name__)


class SearchKBTool(Tool):
    """调用 HybridRetriever（向量+BM25+Reranker）召回知识库切片。

    构造时注入 ``retriever``（需实现 ``retrieve(query, top_k, tenant_id,
    kb_ids) -> list[RetrievedChunk]``）；每次调用按模型给的 query 检索，
    结果格式化为 ``[i] title (score) 摘要`` 回填给模型。
    """

    name = "search_kb"
    description = (
        "语义检索：根据概念/主题找到相关段落，即使用词与原文不同也能召回。"
        "适合查定义、流程、原因解释、用户口语与文档术语不一致时的桥接。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索查询（自然语言或关键词均可，短词更敏感）",
            },
            "top_k": {
                "type": "integer",
                "description": "返回切片数（默3，上限 4）",
                "default": 3,
            },
            "doc_hints": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "已知手册名/型号子串/glob 时传入以圈定预筛；不确定传空列表 []"
                ),
                "default": [],
            },
        },
        "required": ["query"],
    }
    best_for = (
        "概览/定义类问题（“什么是X”“X怎么用”）",
        "用户口语与文档术语不一致时（语义召回同义改写）",
        "跨文档发现相关内容（不确定答案在哪本手册时）",
    )
    avoid_for = (
        "已知精确标识符/型号/参数的全量出现位置（用 grep）",
    )

    def __init__(
        self,
        retriever: Any,
        cpu_executor: Any | None = None,
        max_preview: int = 100,
        default_top_k: int = 3,
        max_top_k: int = 4,
    ) -> None:
        super().__init__(cpu_executor)
        self._retriever = retriever
        self._max_preview = max_preview
        self._default_top_k = default_top_k
        self._max_top_k = max_top_k

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(content="query 不能为空", is_error=True)
        top_k = int(args.get("top_k") or self._default_top_k)
        top_k = max(1, min(top_k, self._max_top_k))

        # ★ 模型自主预筛：doc_hints → 每个 hint 单独召回、轮转合并，
        #   避免“一个高分手册吃满 top_k 名额”造成其他 hint 候选一个都
        #   看不到。无 hints 时回全库检索。
        raw_hints = args.get("doc_hints") or []
        hints: list[str] = []
        if isinstance(raw_hints, str):
            hints = [raw_hints.strip()] if raw_hints.strip() else []
        elif isinstance(raw_hints, (list, tuple)):
            hints = [str(h).strip() for h in raw_hints if str(h).strip()]
        # ★ fanout 任务作用域：restrict_to_sources 非空时，doc_hints 强制限定在该
        #   集合内（模型自己传的 hints 若与集合无交集会被丢弃，不能越范围）。
        _restrict = getattr(ctx, "restrict_to_sources", None)
        if _restrict:
            _allow = {str(x) for x in _restrict}
            if not hints:
                hints = sorted(_allow)
            else:
                hints = [h for h in hints if h in _allow] or sorted(_allow)
        empty_hints: list[str] = []
        filter_notice = ""
        try:
            if hints:
                # 每 hint 给 max(3, top_k//len(hints)) 个名额（下限 3，上限 top_k）
                per_hint = max(3, top_k // max(1, len(hints)))
                per_hint = min(per_hint, top_k)
                tasks = []
                for h in hints:
                    # ★ hint 归一化：title 元数据是 p.stem（不含扩展名），glob 回包的
                    #   文件名却带 .md；且 fnmatch 是整串匹配（"H5U*" 只匹配前缀）。
                    #   故去扩展名 + 无通配符时子串化，使“拷 glob 文件名进 doc_hints”开箱即用。
                    title_pat, source_pat = self._normalize_hint(h)
                    tasks.append(self._safe_retrieve(
                        query, per_hint, ctx, {"title": [title_pat]},
                    ))
                    tasks.append(self._safe_retrieve(
                        query, per_hint, ctx, {"source": [source_pat]},
                    ))
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                # 按 hint 分组（title 主使、source 补位），再轮转抽取
                grouped: list[list[Any]] = [[] for _ in hints]
                for idx, h in enumerate(hints):
                    title_res = gathered[idx * 2]
                    source_res = gathered[idx * 2 + 1]
                    bucket: list[Any] = []
                    seen_local: set[tuple[str, int]] = set()
                    for res in (title_res, source_res):
                        if isinstance(res, Exception) or not res:
                            continue
                        for c in res:
                            key = self._dedup_key(c)
                            if key in seen_local:
                                continue
                            seen_local.add(key)
                            bucket.append(c)
                    bucket.sort(key=lambda c: -float(getattr(c, "score", 0.0)))
                    grouped[idx] = bucket
                    if not bucket:
                        empty_hints.append(h)
                # 轮转（每个 hint 先取 1 条 → 再取第 2 条 → …）确保均衷
                picked: list[Any] = []
                seen_all: set[tuple[str, int]] = set()
                rank = 0
                while len(picked) < top_k and any(
                    len(g) > rank for g in grouped
                ):
                    for g in grouped:
                        if rank < len(g):
                            c = g[rank]
                            key = self._dedup_key(c)
                            if key not in seen_all:
                                seen_all.add(key)
                                picked.append(c)
                                if len(picked) >= top_k:
                                    break
                    rank += 1
                chunks = picked
                if not chunks:
                    # ★ doc_hints 全部过滤后 0 命中：不再“静默”回退，而是如实告知，
                    #   让模型知道圈定失败（多半是命名不存在），这轮命中并非圈定结果。
                    chunks = await self._safe_retrieve(query, top_k, ctx, None)
                    filter_notice = (
                        "⚠ doc_hints=" + str(hints) +
                        " 在本知识库【未匹配到任何手册】，已回退为全库检索。"
                        "下列命中并非圈定结果——若这些文件名确实不存在，"
                        "说明用户口中的型号可能被合并在别的手册里，请用 glob 核对命名或直接按内容检索。"
                    )
                elif empty_hints:
                    filter_notice = (
                        "⚠ 部分 doc_hints 未匹配到手册：" + str(empty_hints) +
                        " （命中的是来自其他 hint 的结果，请核对这些手册命名是否存在）。"
                    )
            else:
                chunks = await self._safe_retrieve(query, top_k, ctx, None)
        except asyncio.TimeoutError:
            return ToolResult(content="检索超时（>30s）", is_error=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_kb 检索失败：%s", exc)
            return ToolResult(content=f"检索失败：{exc}", is_error=True)

        if not chunks:
            return ToolResult(content=f"未检索到与 \"{query}\" 相关的内容。")

        # 格式化：[编号] 标题 定位 (score)  摘要
        lines = []
        if filter_notice:
            lines.append(filter_notice)
        lines.append(f"共 {len(chunks)} 条命中：")
        for i, c in enumerate(chunks, 1):
            title = getattr(c, "title", "") or getattr(c, "document_id", "?")
            score = round(float(getattr(c, "score", 0.0)), 4)
            locator = self._chunk_locator(c)
            snippet = self._clean_snippet(c.content, self._max_preview)
            head = f"[{i}] {title}"
            if locator:
                head += f" @ {locator}"
            head += f" (score={score})"
            lines.append(head)
            lines.append(f"    {snippet}")
        return ToolResult(content="\n".join(lines), meta={"count": len(chunks)})

    _DOC_EXT_RE = re.compile(
        r"\.(md|markdown|txt|rst|log|py|js|ts|tsx|json|ya?ml|csv|html|xml)$",
        re.IGNORECASE,
    )

    @classmethod
    def _normalize_hint(cls, h: str) -> tuple[str, str]:
        """把手册提示归一化为可命中的 glob pattern。

        返回 (title_pattern, source_pattern)：
        - title 元数据 = p.stem（文件名去扩展名），故对 title 用去扩展名后的核心；
        - source 元数据 = 含扩展名的路径，故对 source 保留原样；
        - 二者若不含通配符，则包成 *core* 做子串匹配，规避 fnmatch 的整串前缀语义。
        """
        h = h.strip()
        core = cls._DOC_EXT_RE.sub("", h)

        def widen(pat: str) -> str:
            # doc_hints 的意图是“找关于 X 的手册”，而本库 title 以文档号开头
            # （19011157-…），型号几乎不会出现在开头 → 强制双侧通配做子串
            # 匹配，避免 "X*" 只匹配前缀导致圈定失败。
            if not pat.startswith("*"):
                pat = "*" + pat
            if not pat.endswith("*"):
                pat = pat + "*"
            return pat

        return widen(core), widen(h)

    async def _safe_retrieve(
        self, query: str, top_k: int, ctx: ToolContext,
        metadata_filter: dict[str, Any] | None,
    ) -> list[Any]:
        return await asyncio.wait_for(
            self._retriever.retrieve(
                query,
                top_k=top_k,
                tenant_id=ctx.tenant_id,
                kb_ids=list(ctx.kb_ids or []),
                metadata_filter=metadata_filter,
            ),
            timeout=30.0,
        )

    @staticmethod
    def _dedup_key(c: Any) -> tuple[str, int]:
        """稳定去重键 (document_id, chunk_index)。

        ★ RetrievedChunk 的 chunk_index 实际在 ``c.chunk.chunk_index``（顶层无此属性），
        旧代码用 ``getattr(c, "chunk_index", 0)`` 会恒为 0 → 同一本手册的所有块被
        折叠成 1 条（doc_hints 圈定时召回塌缩）。优先用 RetrievedChunk 自带的
        ``key`` 属性，否则回退到 ``c.chunk.chunk_index``。
        """
        key = getattr(c, "key", None)
        if isinstance(key, (tuple, list)) and len(key) == 2:
            return (str(key[0]), int(key[1]))
        doc = str(getattr(c, "document_id", "") or "")
        inner = getattr(c, "chunk", None)
        idx = getattr(inner, "chunk_index", None)
        if idx is None:
            idx = getattr(c, "chunk_index", 0)
        return (doc, int(idx or 0))

    @staticmethod
    def _chunk_locator(chunk: Any) -> str:
        """尽量给出一个可 grep/read_file 定位的位置提示。

        ★ 新索引时优先输出 `basename.md:<start_line>`：匹配
        tool_agent._CITE_RE，回包就能自动进入 file_citations（让早停与
        反幻觉校验实际能工作）。旧索引无 start_line 时回退到 chunk#N。
        """
        import os as _os
        # RetrievedChunk 把 metadata 存在内层 Chunk 上，同时顶层有 source/title。
        inner = getattr(chunk, "chunk", None) or chunk
        meta = getattr(inner, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        src = str(
            getattr(chunk, "source", "") or meta.get("source") or ""
        )
        start = meta.get("start_line")
        if src and start:
            try:
                return f"{_os.path.basename(src)}:{int(start)}"
            except (TypeError, ValueError):
                return _os.path.basename(src) or src
        for key in ("source_locator", "locator", "line_start", "page"):
            v = meta.get(key)
            if v:
                return str(v)
        idx = getattr(inner, "chunk_index", None)
        return f"chunk#{idx}" if idx is not None else ""

    @staticmethod
    def _clean_snippet(text: str, limit: int) -> str:
        t = (text or "").replace("\n", " ").strip()
        return t if len(t) <= limit else t[:limit] + "..."
