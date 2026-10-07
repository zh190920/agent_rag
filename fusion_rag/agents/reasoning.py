"""ReasoningAgent：长文精读 + 证据锚定推理 + 带引用答案生成（对标 Claude-Code）。

核心原则「无证据不作答」：答案必须锚定在检索到的编号证据上，引用以
``[n]`` 内联标注，可被校验层核对、被前端渲染为可点击来源。

- 证据装配交给 :class:`ContextOffloader`（预算裁剪 + 超长卸载）。
- LLM 优先；无模型或失败时降级为「抽取式」答案（拼接高分证据要点），
  保证永远有可追溯输出，并标记 degraded。
- 自适应输出格式：步骤/清单/表格/段落，按问题措辞推断或按显式 hint。
"""

from __future__ import annotations

import re
from typing import Any

from ..core.logging import get_logger
from ..llm.base import Message
from ..planning.context_offload import ContextOffloader, EvidenceBlock
from ..text import tokenize_query
from ..types import Answer, Citation, OutputFormat, RetrievedChunk
from .base import AgentContext, BaseAgent

logger = get_logger(__name__)

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")
_INSUFFICIENT_HINTS = (
    "无法回答", "未找到", "没有找到", "知识库中没", "信息不足", "无法确定",
    "抱歉", "insufficient", "not found", "cannot answer",
)
_FORMAT_STEPS = ("步骤", "如何", "怎么", "怎样", "流程", "how to", "steps")
_FORMAT_LIST = ("列出", "有哪些", "清单", "列举", "几个", "list", "enumerate")
_FORMAT_TABLE = ("对比", "比较", "表格", "区别", "versus", "compare", "vs")


def infer_output_format(question: str, hint: OutputFormat | str | None = None) -> OutputFormat:
    """按显式 hint 或问题措辞推断输出格式。"""
    if isinstance(hint, OutputFormat):
        return hint
    if isinstance(hint, str) and hint:
        try:
            return OutputFormat(hint)
        except ValueError:
            pass
    q = (question or "").lower()
    if any(w in q for w in _FORMAT_TABLE):
        return OutputFormat.TABLE
    if any(w in q for w in _FORMAT_STEPS):
        return OutputFormat.STEPS
    if any(w in q for w in _FORMAT_LIST):
        return OutputFormat.LIST
    return OutputFormat.PARAGRAPH


class ReasoningAgent(BaseAgent):
    """推理 Agent。"""

    name = "reasoning"
    role = "长文精读、语义推理、逻辑串联、带引用答案生成"

    def __init__(
        self,
        *args: Any,
        offloader: ContextOffloader | None = None,
        token_budget: int = 3500,
        use_llm: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._offloader = offloader or ContextOffloader()
        self.token_budget = token_budget
        self.use_llm = use_llm

    # ------------------------------------------------------------------
    async def reason(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        ctx: AgentContext,
        *,
        memory_block: str = "",
        output_hint: OutputFormat | str | None = None,
        correction: str = "",
    ) -> Answer:
        """基于证据生成答案。``correction`` 为校验层回炉时给出的纠错指令。"""
        fmt = infer_output_format(question, output_hint)
        with self.span(
            "reason", chunks=len(chunks), fmt=fmt.value, retry=bool(correction),
        ) as node:
            if not chunks:
                node.attributes["grounded"] = False
                return self._no_evidence_answer(fmt)

            evidence = self._offloader.build_evidence(
                chunks, budget=self.token_budget, query=question,
            )
            citation_map = evidence.citation_map()

            answer: Answer | None = None
            if self.use_llm and self.llm is not None:
                answer = await self._llm_reason(
                    question, evidence, citation_map, memory_block, fmt, correction,
                )
            if answer is None:
                answer = self._extractive_reason(question, evidence, citation_map, fmt)
                answer.confidence = min(answer.confidence, 0.55)
                node.attributes["degraded"] = True

            node.attributes["citations"] = len(answer.citations)
            node.attributes["confidence"] = round(answer.confidence, 3)
            self.observe("answer_confidence", answer.confidence)
            self.count("reasoning_done", fmt=fmt.value)
            return answer

    # ------------------------------------------------------------------
    async def _llm_reason(
        self,
        question: str,
        evidence: EvidenceBlock,
        citation_map: dict[int, RetrievedChunk],
        memory_block: str,
        fmt: OutputFormat,
        correction: str,
    ) -> Answer | None:
        system = (
            "你是企业知识库问答推理引擎。严格依据【证据】回答，禁止编造；"
            "证据不足时明确说明「知识库中未找到足够信息」，不要臆测。"
            "在引用证据的句子后用 [编号] 标注来源（编号对应证据前的方括号数字）。"
            f"输出格式要求：{_FORMAT_GUIDE[fmt]}。只输出答案正文，不要额外解释。"
        )
        parts = []
        if memory_block:
            parts.append(memory_block)
        parts.append(f"【证据】\n{evidence.text}")
        parts.append(f"【问题】\n{question}")
        if correction:
            parts.append(f"【上一版答案存在的问题，请修正】\n{correction}")
        user = "\n\n".join(parts)

        try:
            resp = await self.llm.chat(  # type: ignore[union-attr]
                [Message.system(system), Message.user(user)],
                provider=self.provider, temperature=0.2, max_tokens=1200,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("推理 LLM 失败，降级抽取式：%s", exc)
            self.count("reasoning_llm_errors")
            return None

        text = (resp.text or "").strip()
        if not text:
            return None
        # 离线兜底模型（EchoLLM）只会回显提示词，不产出面向用户的答案，
        # 此时改用抽取式推理，保证输出可读且可追溯。
        if (resp.model or "").lower() == "echo":
            return None
        referenced = self._referenced_indices(text, citation_map)
        citations = self._build_citations(referenced, citation_map)
        insufficient = self._is_insufficient(text)
        confidence = self._confidence(citation_map, question, insufficient, bool(referenced))
        return Answer(
            text=text,
            citations=citations,
            output_format=fmt,
            confidence=confidence,
            model=resp.model or "",
        )

    def _extractive_reason(
        self,
        question: str,
        evidence: EvidenceBlock,
        citation_map: dict[int, RetrievedChunk],
        fmt: OutputFormat,
    ) -> Answer:
        """无 LLM 兜底：抽取高分证据要点，逐条标注来源编号。"""
        if not citation_map:
            return self._no_evidence_answer(fmt)
        items = list(citation_map.items())[:5]
        lines: list[str] = []
        for n, chunk in items:
            lead = self._lead_sentence(chunk.content)
            src = chunk.title or chunk.source or chunk.document_id[:8]
            lines.append(f"- {lead}（来源[{n}]：{src}）")
        header = f"根据知识库检索，与「{question}」最相关的要点如下："
        text = header + "\n" + "\n".join(lines)
        if fmt == OutputFormat.PARAGRAPH:
            text = header + " " + "；".join(
                self._lead_sentence(c.content) for _, c in items
            ) + "。"
        citations = self._build_citations([n for n, _ in items], citation_map)
        coverage = self._coverage(question, evidence.text)
        return Answer(
            text=text,
            citations=citations,
            output_format=fmt,
            confidence=round(0.35 + 0.3 * coverage, 3),
            model="extractive",
        )

    def _no_evidence_answer(self, fmt: OutputFormat) -> Answer:
        return Answer(
            text="知识库中未找到与该问题相关的信息，无法给出有依据的回答。"
                 "请补充文档或调整提问后重试。",
            citations=[],
            output_format=fmt,
            confidence=0.0,
            model="none",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _referenced_indices(
        text: str, citation_map: dict[int, RetrievedChunk],
    ) -> list[int]:
        found = [int(m) for m in _CITATION_RE.findall(text)]
        return [n for n in found if n in citation_map]

    @staticmethod
    def _build_citations(
        referenced: list[int], citation_map: dict[int, RetrievedChunk],
    ) -> list[Citation]:
        indices = list(dict.fromkeys(referenced))
        if not indices:
            # 答案未显式标注引用时，回退附上最相关的前 3 条证据作为来源
            indices = sorted(citation_map)[:3]
        citations: list[Citation] = []
        for n in indices:
            chunk = citation_map.get(n)
            if chunk is None:
                continue
            citations.append(Citation(
                index=n,
                document_id=chunk.document_id,
                title=chunk.title,
                source=chunk.source,
                chunk_index=chunk.chunk.chunk_index,
                snippet=chunk.content[:200],
                score=round(chunk.score, 6),
            ))
        return citations

    @staticmethod
    def _is_insufficient(text: str) -> bool:
        lowered = text.lower()
        return any(h in text or h in lowered for h in _INSUFFICIENT_HINTS)

    def _confidence(
        self,
        citation_map: dict[int, RetrievedChunk],
        question: str,
        insufficient: bool,
        has_citation: bool,
    ) -> float:
        if not citation_map:
            return 0.0
        evidence_text = "\n".join(c.content for c in citation_map.values())
        coverage = self._coverage(question, evidence_text)
        conf = 0.45 + 0.35 * coverage
        if has_citation:
            conf += 0.1
        if len(citation_map) >= 3:
            conf += 0.05
        if insufficient:
            conf = min(conf, 0.3)
        return round(max(0.0, min(conf, 0.95)), 3)

    @staticmethod
    def _coverage(question: str, evidence_text: str) -> float:
        q_tokens = set(tokenize_query(question))
        if not q_tokens:
            return 0.0
        e_tokens = set(tokenize_query(evidence_text))
        return len(q_tokens & e_tokens) / len(q_tokens)

    @staticmethod
    def _lead_sentence(text: str, limit: int = 120) -> str:
        first = re.split(r"(?<=[。！？!?\n])", text.strip(), maxsplit=1)[0]
        first = first.strip() or text.strip()
        return first[:limit]


_FORMAT_GUIDE: dict[OutputFormat, str] = {
    OutputFormat.PARAGRAPH: "连贯段落，逻辑清晰",
    OutputFormat.LIST: "要点清单（每行以 - 开头）",
    OutputFormat.TABLE: "Markdown 表格，列出对比维度",
    OutputFormat.STEPS: "有序步骤（1. 2. 3. …），每步含关键操作与注意事项",
}
