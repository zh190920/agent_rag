"""OpsAgent：运维巡检 Agent（借鉴 DeepSeek-Harness 高可观测性）。

后台周期任务，负责：
- 评估告警规则（错误率/延迟/模型失败率），命中即广播 + 日志。
- 采集指标快照，落盘 JSON 文件 + 写入存储（供趋势分析）。
- 巡检关键服务健康探针（LLM Router / 存储 / 向量库），异常置为告警。
- 采集运行态 Gauge（在途请求数、健康目标数）。

由 Kernel 生命周期托管：``start()`` 返回后台 Task 并登记到 background_tasks，
关闭时被统一 cancel。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable

from ..core.logging import get_logger
from ..core.plugin import PluginRegistry
from ..core.sandbox import Sandbox
from ..observability.alerter import Alerter
from ..observability.metrics import MetricsRegistry
from ..storage.sqlite_store import SQLiteStore
from .base import BaseAgent

logger = get_logger(__name__)

HealthProbe = Callable[[], Any]


class OpsAgent(BaseAgent):
    """运维 Agent。"""

    name = "ops"
    role = "监控模块运行状态、日志统计、异常告警、插件巡检"

    def __init__(
        self,
        metrics: MetricsRegistry,
        alerter: Alerter,
        store: SQLiteStore | None = None,
        sandbox: Sandbox | None = None,
        registry: PluginRegistry | None = None,
        *,
        metrics_path: str | Path | None = None,
        interval: float = 30.0,
        health_probes: dict[str, HealthProbe] | None = None,
        tracer: Any | None = None,
    ) -> None:
        super().__init__(llm=None, tracer=tracer, metrics=metrics)
        self._metrics = metrics
        self._alerter = alerter
        self._store = store
        self._sandbox = sandbox
        self._registry = registry
        self._metrics_path = Path(metrics_path).expanduser() if metrics_path else None
        if self._metrics_path is not None:
            self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.interval = max(1.0, interval)
        self._health_probes = dict(health_probes or {})
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    def add_probe(self, name: str, probe: HealthProbe) -> None:
        self._health_probes[name] = probe

    def start(self) -> asyncio.Task[None]:
        """启动后台巡检循环（幂等）。"""
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self.run_forever(), name="ops-agent",
            )
        return self._task

    async def run_forever(self) -> None:
        logger.info("OpsAgent 巡检启动，间隔 %.0fs", self.interval)
        while True:
            try:
                await asyncio.sleep(self.interval)
                await self.inspect_once()
            except asyncio.CancelledError:
                logger.info("OpsAgent 巡检停止")
                raise
            except Exception:  # noqa: BLE001 —— 巡检绝不能因异常退出
                logger.exception("OpsAgent 巡检异常（已隔离，继续运行）")

    # ------------------------------------------------------------------
    async def inspect_once(self) -> dict[str, Any]:
        """执行一次巡检，返回摘要（也便于 API/CLI 手动触发）。"""
        started = time.monotonic()
        snapshot = self._metrics.snapshot()

        # 运行态 Gauge
        if self._sandbox is not None:
            self._metrics.gauge("inflight_requests").set(float(self._sandbox.inflight))

        health = await self._check_health()
        healthy = sum(1 for ok in health.values() if ok)
        self._metrics.gauge("healthy_targets").set(float(healthy))
        self._metrics.gauge("total_targets").set(float(len(health)))

        alerts = self._alerter.evaluate()
        if alerts:
            self.count("alerts_fired", amount=float(len(alerts)))

        summary = {
            "ts": round(time.time(), 3),
            "cost_ms": int((time.monotonic() - started) * 1000),
            "health": health,
            "alerts": [a.to_dict() for a in alerts],
            "uptime_seconds": snapshot.get("uptime_seconds"),
            "plugins": self._plugin_summary(),
        }

        await self._persist(snapshot, summary)
        return summary

    async def _check_health(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for name, probe in self._health_probes.items():
            try:
                value = probe()
                if inspect.isawaitable(value):
                    value = await value
                result[name] = bool(value)
            except Exception as exc:  # noqa: BLE001
                logger.warning("健康探针 %s 异常：%s", name, exc)
                result[name] = False
        # 任一关键目标不健康即打点，供告警规则消费
        unhealthy = [n for n, ok in result.items() if not ok]
        if unhealthy:
            self.count("unhealthy_targets", amount=float(len(unhealthy)))
        return result

    def _plugin_summary(self) -> dict[str, list[str]]:
        if self._registry is None:
            return {}
        kinds: dict[str, list[str]] = {}
        for spec in self._registry.list_specs():
            kinds.setdefault(spec.kind, []).append(f"{spec.name}@{spec.version}")
        return kinds

    async def _persist(self, snapshot: dict[str, Any], summary: dict[str, Any]) -> None:
        if self._store is not None:
            try:
                await self._store.save_metrics(snapshot)
            except Exception:  # noqa: BLE001
                logger.exception("指标快照落库失败")
        if self._metrics_path is not None:
            payload = {"summary": summary, "snapshot": snapshot}
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, lambda: self._metrics_path.write_text(text, "utf-8"),  # type: ignore[union-attr]
                )
            except OSError:
                logger.exception("指标快照写文件失败 %s", self._metrics_path)

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._task = None
