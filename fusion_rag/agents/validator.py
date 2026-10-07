"""ValidatorAgent：答案精读校验与逻辑纠错（对标 Claude-Code 结果校验）。

四维校验：
1. 引用忠实性 —— 答案引用的 [n] 是否都对应真实证据来源（防伪造引用）。
2. 完整性     —— 是否实质回答了问题（非空、非敷衍、覆盖问题要点）。
3. 逻辑一致性 —— 结论与证据是否自洽，有无自相矛盾。
4. 幻觉检测   —— 是否存在证据无法支撑的臆断。

LLM 优先（结构化 __VALIDATE__ JSON）；失败或无模型时降级为规则校验。
未通过时产出 ``correction`` 纠错指令，交推理层回炉重写（最多 max_retry 轮）。
"""

from __future__ import annotations

from typing import Any

from ..constants import INOVANCE_PERSONA
from ..core.logging import get_logger
from ..llm.base import Message
from ..types import Answer, RetrievedChunk, ValidationResult
from .base import AgentContext, BaseAgent

logger = get_logger(__name__)

_INSUFFICIENT = (
    "无法回答", "未找到", "没有找到", "信息不足", "无法确定", "知识库中未",
)
_TRIVIAL_LEN = 8


class ValidatorAgent(BaseAgent):
    """校验 Agent。"""

    name = "validator"
    role = "核查答案真实性/完整性/逻辑性，修正幻觉、错误、残缺内容"

    def __init__(
        self,
        *args: Any,
        enabled: bool = True,
        min_confidence: float = 0.5,
        use_llm: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enabled = enabled
        self.min_confidence = min_confidence
        self.use_llm = use_llm

    # ------------------------------------------------------------------
    async def validate(
        self,
        question: str,
        answer: Answer,
        evidence: list[RetrievedChunk],
        ctx: AgentContext,
    ) -> ValidationResult:
        if not self.enabled:
            return ValidationResult(passed=True, confidence=answer.confidence)

        with self.span(
            "validate", citations=len(answer.citations), evidence=len(evidence),
        ) as node:
            result: ValidationResult | None = None
            if self.use_llm and self.llm is not None:
                result = await self._llm_validate(question, answer, evidence)
            if result is None:
                result = self._rule_validate(question, answer, evidence)

            # 与推理层置信度取较保守值，避免校验虚高
            result.confidence = round(
                min(result.confidence, max(answer.confidence, result.confidence * 0.5))
                if answer.confidence else result.confidence, 3,
            )
            node.attributes["passed"] = result.passed
            node.attributes["confidence"] = result.confidence
            node.attributes["issues"] = len(result.issues)
            self.count(
                "validation",
                passed="true" if result.passed else "false",
            )
            return result

    # ------------------------------------------------------------------
    async def _llm_validate(
        self, question: str, answer: Answer, evidence: list[RetrievedChunk],
    ) -> ValidationResult | None:
        ev_text = "\n".join(
            f"[{i + 1}] {c.content[:400]}" for i, c in enumerate(evidence[:8])
        ) or "（无证据）"
        cite_text = "\n".join(
            f"[{c.index}] {c.title or c.source} :: {c.snippet[:120]}"
            for c in answer.citations
        ) or "（答案未标注引用）"
        # 检测是否为 agentic 模式：标题包含文件名 或 source 以 tool: 开头
        is_agentic = any(
            c.snippet and (":" in c.snippet and "." in c.snippet.split(":")[0])
            for c in (answer.citations or [])[:3]
        ) and any(
            str(getattr(c, "source", "")).startswith("tool:")
            for c in (evidence or [])[:3]
        )
        agentic_hint = ""
        if is_agentic:
            agentic_hint = (
                "\n※ 本回合为 agentic 工具驱动模式："
                "【答案引用】采用 `文件名:行号` 格式，来自模型自主调用的"
                "grep/read_file 等工具输出；【证据】为工具返回的原文片段。"
                "判定引用忠实性时，只需确认【证据】内包含【答案】提及的关键事实"
                "（型号/参数/日期等），不要求严格的内容行号对应。"
                "若【答案】实质与【证据】一致即可 passed=true。"
            )
        system = (
            INOVANCE_PERSONA + "\n"
            "你是严格的答案校验器。核对【答案】是否被【证据】充分支撑，从四个维度判断："
            "引用忠实性、完整性、逻辑一致性、是否存在幻觉。只输出 JSON。__VALIDATE__\n"
            '字段：passed(bool), confidence(0~1), issues(问题列表), '
            'correction(若不通过，给推理层的明确修正指令；通过则空串)。'
            f"{agentic_hint}"
        )
        user = (
            f"【问题】\n{question}\n\n【答案】\n{answer.text}\n\n"
            f"【答案引用】\n{cite_text}\n\n【证据】\n{ev_text}"
        )
        try:
            data = await self.llm.chat_json(  # type: ignore[union-attr]
                [Message.system(system), Message.user(user)],
                provider=self.provider, temperature=0.0, max_tokens=1024,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("校验 LLM 失败，降级规则校验：%s", exc)
            self.count("validation_llm_errors")
            return None

        try:
            issues = [str(i) for i in (data.get("issues") or [])]
            return ValidationResult(
                passed=bool(data.get("passed", True)),
                confidence=float(data.get("confidence", 0.6)),
                issues=issues,
                correction=str(data.get("correction", "")),
            )
        except (TypeError, ValueError) as exc:
            logger.warning("校验结果解析失败，降级规则：%s", exc)
            return None

    def _rule_validate(
        self, question: str, answer: Answer, evidence: list[RetrievedChunk],
    ) -> ValidationResult:
        """零依赖规则校验：引用完整性 + 实质性 + 证据一致性。"""
        issues: list[str] = []
        text = (answer.text or "").strip()

        if len(text) < _TRIVIAL_LEN:
            issues.append("答案过短，可能未实质回答问题")

        insufficient = any(h in text for h in _INSUFFICIENT)
        if insufficient and evidence:
            issues.append("存在可用证据却判定信息不足，答案未充分利用检索结果")
        if insufficient and not evidence:
            # 诚实承认无证据：视为通过，但置信度低
            return ValidationResult(
                passed=True, confidence=min(answer.confidence, 0.3),
                issues=["知识库无相关证据，答案诚实告知无法回答"],
            )

        # 引用忠实性：答案引用必须对应真实证据来源
        # 兼容 agentic 模式的 duck-typed evidence（无 .chunk 属性）
        valid_ids: set[tuple[str, int]] = set()
        for c in evidence:
            chunk_obj = getattr(c, "chunk", None)
            idx = getattr(chunk_obj, "chunk_index", 0) if chunk_obj is not None else 0
            valid_ids.add((getattr(c, "document_id", ""), idx))
        # agentic 证据 document_id 均为空→不满足严格对应，回退为宽松判定
        if any(d for d, _ in valid_ids):
            for cite in answer.citations:
                if (cite.document_id, cite.chunk_index) not in valid_ids:
                    issues.append("引用[{}]指向了不存在的证据来源".format(cite.index))
                    break

        # 有实质答案却零引用 → 缺乏可追溯性
        if not insufficient and text and not answer.citations and evidence:
            issues.append("答案未标注任何来源引用，可追溯性不足")

        passed = not issues
        confidence = answer.confidence if passed else max(0.0, answer.confidence - 0.25)
        correction = "" if passed else "；".join(issues) + "。请依据证据重写并正确标注 [编号] 引用。"
        return ValidationResult(
            passed=passed, confidence=round(confidence, 3),
            issues=issues, correction=correction,
        )
