"""MemoryAgent：多轮会话记忆与用户偏好管理的角色化封装。

包装 :class:`MemoryManager`，把「装配上下文 / 记录问答 / 压缩历史 /
沉淀偏好」暴露为 Agent 能力，供 Orchestrator 统一编排。记忆当前落盘
SQLite（Windows 友好），后续可无感迁移 Redis（Manager 内 Store 可替换）。
"""

from __future__ import annotations

from typing import Any

from ..core.logging import get_logger
from ..memory.manager import MemoryContext, MemoryManager
from .base import BaseAgent

logger = get_logger(__name__)


class MemoryAgent(BaseAgent):
    """记忆 Agent。"""

    name = "memory"
    role = "管理多轮会话记忆、用户偏好、历史问答，实现连续对话"

    def __init__(
        self,
        manager: MemoryManager,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # 记忆 Agent 不需要 LLM Router（压缩摘要由 Manager 内部持有 LLM）
        kwargs.setdefault("llm", None)
        super().__init__(*args, **kwargs)
        self.manager = manager

    # ------------------------------------------------------------------
    async def build_context(
        self,
        session_id: str,
        tenant_id: str,
        *,
        reserve_tokens: int = 0,
    ) -> MemoryContext:
        """装配本次问答的记忆上下文（按预算裁剪）。"""
        if not session_id:
            return MemoryContext()
        with self.span("build_context", session=session_id) as node:
            ctx = await self.manager.build_context(
                session_id, tenant_id, reserve_tokens=reserve_tokens,
            )
            node.attributes["turns"] = len(ctx.recent_turns)
            node.attributes["tokens"] = ctx.total_tokens
            return ctx

    async def record_qa(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """记录一轮问答并在超窗时压缩历史。"""
        if not session_id:
            return
        with self.span("record_qa", session=session_id):
            await self.manager.record_turn(session_id, "user", question)
            await self.manager.record_turn(session_id, "assistant", answer, meta=meta)
            try:
                await self.manager.compact_if_needed(session_id)
            except Exception:  # noqa: BLE001 —— 压缩失败不影响主流程
                logger.warning("会话 %s 历史压缩失败（忽略）", session_id, exc_info=True)

    async def remember_preference(
        self, key: str, value: str, *, tenant_id: str = "default",
        namespace: str = "preference",
    ) -> None:
        """沉淀用户偏好/高频关注到长期记忆。"""
        await self.manager.long_term.remember(
            key, value, tenant_id=tenant_id, namespace=namespace,
        )
        self.count("memory_remembered", namespace=namespace)

    async def recall(
        self, *, tenant_id: str = "default", namespace: str = "preference", limit: int = 10,
    ) -> list[dict[str, Any]]:
        return await self.manager.long_term.recall(
            tenant_id=tenant_id, namespace=namespace, limit=limit,
        )
