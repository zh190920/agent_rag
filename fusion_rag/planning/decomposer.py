"""TaskDecomposer：复杂问题分层拆解 + 优先级判定 + HITL 触发。

三级降级保证「永远可用」：
1. LLM 语义拆解（结构化 JSON，带 __DECOMPOSE__ 协议标记）
2. LLM 失败 → 启发式连接词拆分
3. 简单问题 → 单任务直通

拆解结果是子问题列表，Orchestrator 会并发检索、再统一推理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..constants import INOVANCE_PERSONA
from ..core.logging import get_logger
from ..llm.base import LLMBase, Message
from ..types import Priority

logger = get_logger(__name__)

# 优先级关键词（可按行业扩展/插件化）
_URGENT_WORDS = ("紧急", "立即", "马上", "尽快", "线上故障", "urgent", "asap", "immediately")
_PRO_WORDS = (
    "合规", "法律", "医疗", "诊断", "合同", "条款", "专利", "审计", "财务",
    "legal", "medical", "compliance", "contract", "audit",
)
# 拆解触发词：出现多个子句连接词，往往意味着复合问题
_DECOMPOSE_HINTS = (
    " 和 ", " 以及 ", " 还有 ", "，然后", "。然后", " 并且 ", " 同时 ",
    " 分别 ", " 对比 ", " 比较 ", "为什么", "怎么办", " and ", " also ",
    " compare ", " respectively ",
)


@dataclass
class PlanResult:
    """任务规划结果。"""

    original: str
    sub_questions: list[str] = field(default_factory=list)
    priority: Priority = Priority.NORMAL
    need_human: bool = False
    decomposed: bool = False
    reason: str = ""

    @property
    def is_complex(self) -> bool:
        return len(self.sub_questions) > 1


class TaskDecomposer:
    """问题拆解器。"""

    def __init__(
        self,
        llm: LLMBase | None = None,
        *,
        enabled: bool = True,
        max_sub_questions: int = 5,
        decompose_threshold: int = 30,
        hitl_enabled: bool = False,
        provider: str | None = None,
    ) -> None:
        self._llm = llm
        self._provider = provider
        self.enabled = enabled
        self.max_sub_questions = max(1, max_sub_questions)
        self.decompose_threshold = decompose_threshold
        self.hitl_enabled = hitl_enabled

    async def plan(self, question: str, *, risk: str = "low") -> PlanResult:
        """产出执行计划。"""
        priority = self._infer_priority(question, risk)
        need_human = self.hitl_enabled and risk == "high"
        result = PlanResult(
            original=question, priority=priority, need_human=need_human,
        )

        # 短问题直接单任务
        if not self.enabled or len(question) < self.decompose_threshold:
            result.sub_questions = [question]
            result.reason = "short-or-disabled"
            return result

        subs: list[str] = []
        if self._llm is not None:
            subs = await self._llm_decompose(question)
        if not subs:
            subs = self._heuristic_decompose(question)

        if len(subs) > 1:
            result.sub_questions = subs[: self.max_sub_questions]
            result.decomposed = True
            result.reason = "decomposed"
        else:
            result.sub_questions = [question]
            result.reason = "atomic"
        return result

    # ------------------------------------------------------------------
    async def _llm_decompose(self, question: str) -> list[str]:
        system = (
            INOVANCE_PERSONA + "\n"
            "你是问题拆解器。先识别问题中的汇川行业黑话/口语简称并归一为手册"
            "标准名词（保留原词、追加标准写法）；再判断用户问题是否为需要分步检索的"
            "复合问题；若是，拆解为若干语义完整、可独立检索的子问题（不超过 "
            f"{self.max_sub_questions} 个）；若否，返回空列表。只输出 JSON。__DECOMPOSE__\n"
            '格式：{"need_decompose": true/false, "sub_questions": ["...", "..."]}'
        )
        try:
            data: dict[str, Any] = await self._llm.chat_json(  # type: ignore[union-attr]
                [Message.system(system), Message.user(question)],
                provider=self._provider, temperature=0.0, max_tokens=400,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 拆解失败，转启发式：%s", exc)
            return []

        if not data.get("need_decompose"):
            return []
        subs = data.get("sub_questions") or []
        cleaned = [str(s).strip() for s in subs if str(s).strip()]
        # 过滤与原问题完全相同的无效拆解
        cleaned = [s for s in cleaned if s != question.strip()]
        return cleaned[: self.max_sub_questions]

    def _heuristic_decompose(self, question: str) -> list[str]:
        """无 LLM 兜底：按连接词/标点粗拆。"""
        parts = [question]
        for sep in _DECOMPOSE_HINTS:
            new_parts: list[str] = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts
        result = [p.strip(" ？?。.，,") for p in parts if p and p.strip(" ？?。.，,")]
        # 去重保序
        seen: set[str] = set()
        unique = [r for r in result if not (r in seen or seen.add(r))]
        if 1 < len(unique) <= self.max_sub_questions:
            return unique
        return []

    def _infer_priority(self, question: str, risk: str) -> Priority:
        lowered = question.lower()
        if risk == "high" or any(w in lowered for w in _URGENT_WORDS):
            return Priority.URGENT
        if any(w.lower() in lowered for w in _PRO_WORDS):
            return Priority.PROFESSIONAL
        return Priority.NORMAL
