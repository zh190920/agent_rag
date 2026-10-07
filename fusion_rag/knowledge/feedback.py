"""FeedbackCollector：问答反馈沉淀与检索/推理策略迭代（知识更新与迭代模块）。

周期性汇总：
- **高频问题** → 长期记忆 ``hot_question`` 命名空间，供记忆层在装配上下文
  时优先召回，缩短热点问题的检索路径。
- **低质量问答**（置信度低/校验未过/差评）→ ``weak_point`` 命名空间，作为
  知识盲区清单，指导后续文档补充与检索权重调优。

沉淀结果落本地 SQLite，无任何云端上报（OpenClaw 隐私留存）。
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.logging import get_logger
from ..memory.long_term import LongTermMemory
from ..storage.sqlite_store import SQLiteStore

logger = get_logger(__name__)


class FeedbackCollector:
    """问答反馈沉淀器。"""

    def __init__(
        self,
        store: SQLiteStore,
        long_term: LongTermMemory,
        *,
        tenants: list[str] | None = None,
        interval: float = 300.0,
        min_hits: int = 2,
        low_confidence: float = 0.5,
    ) -> None:
        self._store = store
        self._long_term = long_term
        self._tenants = tenants or ["default"]
        self.interval = max(10.0, interval)
        self.min_hits = max(1, min_hits)
        self.low_confidence = low_confidence
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    def start(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self.run_forever(), name="feedback-collector",
            )
        return self._task

    async def run_forever(self) -> None:
        logger.info("FeedbackCollector 启动，间隔 %.0fs", self.interval)
        while True:
            try:
                await asyncio.sleep(self.interval)
                await self.sink_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("反馈沉淀异常（已隔离，继续运行）")

    async def sink_once(self) -> dict[str, int]:
        """对全部租户执行一次沉淀，返回统计。"""
        stats = {"hot": 0, "weak": 0}
        for tenant in self._tenants:
            stats["hot"] += await self._sink_hot(tenant)
            stats["weak"] += await self._sink_weak(tenant)
        if stats["hot"] or stats["weak"]:
            logger.info("反馈沉淀：热点问题 %d，知识盲区 %d", stats["hot"], stats["weak"])
        return stats

    async def _sink_hot(self, tenant: str) -> int:
        try:
            rows = await self._store.frequent_questions(tenant, limit=50)
        except Exception:  # noqa: BLE001
            logger.exception("读取高频问题失败 tenant=%s", tenant)
            return 0
        count = 0
        for row in rows:
            hits = int(row.get("c", 0))
            if hits < self.min_hits:
                continue
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            avg_conf = row.get("avg_conf")
            value = f"命中 {hits} 次" + (f"，平均置信度 {float(avg_conf):.2f}" if avg_conf else "")
            await self._long_term.remember(
                question[:120], value, tenant_id=tenant, namespace="hot_question",
            )
            count += 1
        return count

    async def _sink_weak(self, tenant: str) -> int:
        try:
            rows = await self._store.low_quality_qa(
                tenant, threshold=self.low_confidence, limit=50,
            )
        except Exception:  # noqa: BLE001
            logger.exception("读取低质量问答失败 tenant=%s", tenant)
            return 0
        count = 0
        for row in rows:
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            await self._long_term.remember(
                question[:120],
                f"低质量：置信度 {float(row.get('confidence', 0)):.2f}",
                tenant_id=tenant, namespace="weak_point",
            )
            count += 1
        return count

    async def snapshot(self, tenant: str = "default") -> dict[str, Any]:
        """导出当前沉淀结果（供运维/调优查看）。"""
        return {
            "hot_question": await self._long_term.recall(
                tenant_id=tenant, namespace="hot_question", limit=20,
            ),
            "weak_point": await self._long_term.recall(
                tenant_id=tenant, namespace="weak_point", limit=20,
            ),
        }

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._task = None
