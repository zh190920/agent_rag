"""长期记忆：用户偏好、高频问题、纠错沉淀（落盘 SQLite）。

按 ``tenant + namespace`` 隔离；``remember`` 幂等累加命中次数，``recall``
按热度返回，实现「越用越懂你」的知识偏好沉淀。
"""

from __future__ import annotations

from typing import Any

from ..storage.sqlite_store import SQLiteStore


class LongTermMemory:
    """长期知识记忆句柄。"""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    async def remember(
        self,
        key: str,
        value: str,
        *,
        tenant_id: str = "default",
        namespace: str = "preference",
    ) -> None:
        await self._store.remember(
            key, value, tenant_id=tenant_id, namespace=namespace,
        )

    async def recall(
        self,
        *,
        tenant_id: str = "default",
        namespace: str = "preference",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return await self._store.recall(
            tenant_id=tenant_id, namespace=namespace, limit=limit,
        )

    async def recall_all_namespaces(
        self, tenant_id: str, namespaces: list[str], limit_each: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for ns in namespaces:
            result[ns] = await self._store.recall(
                tenant_id=tenant_id, namespace=ns, limit=limit_each,
            )
        return result

    async def frequent_questions(
        self, tenant_id: str = "default", limit: int = 10,
    ) -> list[dict[str, Any]]:
        return await self._store.frequent_questions(tenant_id, limit)
