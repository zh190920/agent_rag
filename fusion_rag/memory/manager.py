"""MemoryManager：分层记忆装配 + 上下文智能裁剪 + 滚动摘要压缩。

装配优先级（token 预算不足时从低到高裁剪）：
    当前问题 > 检索证据(由 pipeline 注入) > 近期对话 > 长期记忆 > 历史摘要

压缩：当会话消息超出窗口，用 LLM 把溢出部分并入滚动摘要（失败则退化为
截断式摘要，保证永远可用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..constants import INOVANCE_PERSONA
from ..core.logging import get_logger
from ..llm.base import LLMBase, Message
from ..types import approx_token_count
from .long_term import LongTermMemory
from .short_term import ShortTermMemory

logger = get_logger(__name__)


@dataclass
class MemoryContext:
    """一次问答装配好的记忆上下文。"""

    recent_turns: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    long_term: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0

    def to_prompt_block(self) -> str:
        """渲染为可注入系统/用户提示的文本块。"""
        blocks: list[str] = []
        if self.summary:
            blocks.append(f"【历史对话摘要】\n{self.summary}")
        if self.long_term:
            prefs = "\n".join(
                f"- {m['mem_key']}: {m['mem_value']}" for m in self.long_term[:8]
            )
            blocks.append(f"【用户长期偏好/高频关注】\n{prefs}")
        if self.recent_turns:
            lines = [f"{t['role']}: {t['content']}" for t in self.recent_turns]
            blocks.append("【近期对话】\n" + "\n".join(lines))
        return "\n\n".join(blocks)


class MemoryManager:
    """记忆编排器。"""

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        llm: LLMBase | None = None,
        *,
        token_budget: int = 2000,
        provider: str | None = None,
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term
        self._llm = llm
        self._provider = provider
        self.token_budget = token_budget

    # ------------------------------------------------------------------
    async def build_context(
        self,
        session_id: str,
        tenant_id: str,
        *,
        reserve_tokens: int = 0,
    ) -> MemoryContext:
        """按预算装配记忆上下文。``reserve_tokens`` 为问题+证据预留。"""
        budget = max(256, self.token_budget - reserve_tokens)
        ctx = MemoryContext()

        summary = await self.short_term.get_summary(session_id)
        recent = await self.short_term.recent_turns(session_id)
        ltm = await self.long_term.recall(tenant_id=tenant_id, limit=8)

        # 逆序裁剪：先塞近期对话（最重要），再摘要，再长期记忆
        used = 0
        kept_turns: list[dict[str, Any]] = []
        for turn in reversed(recent):
            cost = approx_token_count(turn["content"]) + 8
            if used + cost > budget and kept_turns:
                break
            kept_turns.insert(0, turn)
            used += cost
        ctx.recent_turns = kept_turns

        if summary:
            cost = approx_token_count(summary)
            if used + cost <= budget:
                ctx.summary = summary
                used += cost
            else:
                # 摘要过长则截断保留尾部
                tail = summary[-max(0, (budget - used)) * 2:]
                ctx.summary = tail
                used += approx_token_count(tail)

        if ltm and used < budget:
            kept_ltm = []
            for mem in ltm:
                cost = approx_token_count(str(mem.get("mem_value", ""))) + 12
                if used + cost > budget:
                    break
                kept_ltm.append(mem)
                used += cost
            ctx.long_term = kept_ltm

        ctx.total_tokens = used
        return ctx

    # ------------------------------------------------------------------
    async def record_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        await self.short_term.add_turn(session_id, role, content, meta=meta)

    async def compact_if_needed(self, session_id: str) -> bool:
        """会话超窗时把溢出历史压缩进滚动摘要。返回是否发生压缩。"""
        overflow = await self.short_term.turns_outside_window(session_id)
        if not overflow:
            return False
        old_summary = await self.short_term.get_summary(session_id)
        new_summary = await self._summarize(old_summary, overflow)
        await self.short_term.set_summary(session_id, new_summary)
        logger.debug("会话 %s 压缩 %d 条历史为摘要", session_id, len(overflow))
        return True

    async def _summarize(
        self, old_summary: str, messages: list[dict[str, Any]],
    ) -> str:
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        # 无 LLM 或失败 → 退化摘要（保留要点，截断）
        if self._llm is None:
            return self._fallback_summary(old_summary, messages)
        system = (
            INOVANCE_PERSONA + "\n"
            "你是对话摘要器。把【已有摘要】与【新增对话】合并压缩为简洁中文摘要，"
            "保留关键事实、结论、用户偏好与未决问题，不超过 300 字。直接输出摘要正文。"
        )
        user = f"【已有摘要】\n{old_summary or '（无）'}\n\n【新增对话】\n{convo}"
        try:
            resp = await self._llm.chat(
                [Message.system(system), Message.user(user)],
                provider=self._provider, temperature=0.2, max_tokens=500,
            )
            text = resp.text.strip()
            return text or self._fallback_summary(old_summary, messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("摘要生成失败，退化处理：%s", exc)
            return self._fallback_summary(old_summary, messages)

    @staticmethod
    def _fallback_summary(
        old_summary: str, messages: list[dict[str, Any]],
    ) -> str:
        tail = messages[-6:]
        lines = [f"{m['role']}: {m['content'][:80]}" for m in tail]
        joined = (old_summary + "\n" if old_summary else "") + "\n".join(lines)
        return joined.strip()[-600:]
