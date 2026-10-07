"""短期会话记忆：滑动窗口 + 滚动摘要（落盘 SQLite）。

- 最近 K 轮对话保留原文；更早轮次压缩为「滚动摘要」存于 session.summary，
  实现超长对话的上下文卸载（借鉴 deepagents summarization + hermes 压缩链）。
"""

from __future__ import annotations

from typing import Any

from ..storage.sqlite_store import SQLiteStore
from ..types import approx_token_count


class ShortTermMemory:
    """单会话的短期记忆句柄。"""

    def __init__(self, store: SQLiteStore, window_rounds: int = 6) -> None:
        self._store = store
        self.window_rounds = max(1, window_rounds)

    async def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        await self._store.append_message(
            session_id, role, content, tokens=approx_token_count(content), meta=meta,
        )

    async def recent_turns(
        self, session_id: str, rounds: int | None = None,
    ) -> list[dict[str, Any]]:
        """取最近 ``rounds`` 轮（1 轮 = user+assistant 两条，故 limit=2*rounds）。"""
        rounds = rounds or self.window_rounds
        msgs = await self._store.get_messages(session_id, limit=rounds * 2)
        return [
            {"role": m["role"], "content": m["content"], "meta": _loads(m.get("meta"))}
            for m in msgs
        ]

    async def turns_outside_window(self, session_id: str) -> list[dict[str, Any]]:
        """返回落在窗口之外、待压缩的历史消息（用于生成滚动摘要）。"""
        keep = self.window_rounds * 2
        total = await self._store.count_messages(session_id)
        if total <= keep:
            return []
        overflow = await self._store.get_messages(
            session_id, limit=total - keep, offset=0,
        )
        return [{"role": m["role"], "content": m["content"]} for m in overflow]

    async def get_summary(self, session_id: str) -> str:
        session = await self._store.get_session(session_id)
        return (session or {}).get("summary", "") or ""

    async def set_summary(self, session_id: str, summary: str) -> None:
        await self._store.touch_session(session_id, summary=summary)

    async def total_messages(self, session_id: str) -> int:
        return await self._store.count_messages(session_id)


def _loads(raw: Any) -> dict[str, Any]:
    import json

    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}
