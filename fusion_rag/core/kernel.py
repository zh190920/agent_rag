"""微内核 Kernel：服务容器 + 生命周期管控 + 全局装配。

借鉴 DeepSeek-Harness「内核轻量化」：Kernel 本身不含业务逻辑，只负责

1. 按配置装配所有插件与服务（依赖注入）
2. 统一管理 start/stop 生命周期（优雅关闭：flush 存储、停事件总线、关线程池）
3. 以服务名暴露给 API / CLI / SDK 层
"""

from __future__ import annotations

import asyncio
from typing import Any

from .config import Config, load_config
from .events import EventBus
from .logging import get_logger, setup_logging
from .plugin import PluginRegistry
from .sandbox import CpuExecutor, Sandbox

logger = get_logger(__name__)


class Kernel:
    """框架运行核心。

    用法::

        kernel = Kernel.from_config_file("config.yaml")
        async with kernel:
            answer = await kernel.orchestrator.ask(...)
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.registry = PluginRegistry()
        self.events = EventBus()
        self.services: dict[str, Any] = {}
        self._started = False

        setup_logging(
            level=str(self.config.get("app.log_level", "INFO")),
            as_json=bool(self.config.get("app.log_json", True)),
        )

    # ------------------------------------------------------------------
    # 构造快捷方式
    # ------------------------------------------------------------------
    @classmethod
    def from_config_file(cls, path: str | None = None, **overrides: Any) -> "Kernel":
        return cls(load_config(path, overrides or None))

    # ------------------------------------------------------------------
    # 服务访问器
    # ------------------------------------------------------------------
    @property
    def orchestrator(self) -> Any:
        return self.services["orchestrator"]

    @property
    def indexer(self) -> Any:
        return self.services["indexer"]

    @property
    def sandbox(self) -> Sandbox:
        return self.services["sandbox"]

    @property
    def cpu(self) -> CpuExecutor:
        return self.services["cpu_executor"]

    def service(self, name: str) -> Any:
        if name not in self.services:
            raise KeyError(f"服务未注册: {name}（kernel 是否已 start？）")
        return self.services[name]

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._started:
            return
        logger.info("Kernel 启动中… data_dir=%s", self.config.data_dir)
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # 延迟导入避免循环依赖，装配逻辑集中在 bootstrap
        from ..bootstrap import bootstrap_kernel

        self.events.start()
        await bootstrap_kernel(self)
        self._started = True
        logger.info(
            "Kernel 启动完成：服务 %d 个，插件 %d 个",
            len(self.services), len(list(self.registry.list_specs())),
        )

    async def stop(self) -> None:
        if not self._started:
            return
        logger.info("Kernel 关闭中…")
        grace = float(self.config.get("kernel.shutdown_grace", 10))

        # 1) 停后台任务（OpsAgent 巡检、指标快照、告警器）
        for task in list(self.services.get("background_tasks", [])):
            task.cancel()
        for task in list(self.services.get("background_tasks", [])):
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass

        # 2) flush / 关闭各服务（实现了 aclose 的按逆序释放）
        for name in reversed(list(self.services)):
            svc = self.services.get(name)
            aclose = getattr(svc, "aclose", None)
            if callable(aclose):
                try:
                    await asyncio.wait_for(aclose(), timeout=grace)
                except Exception:  # noqa: BLE001
                    logger.exception("服务 %s 关闭失败", name)

        # 3) 插件单例、事件总线、线程池
        await self.registry.aclose_all()
        await self.events.stop(drain_timeout=grace)
        cpu: CpuExecutor | None = self.services.get("cpu_executor")
        if cpu is not None:
            cpu.shutdown(wait=False)
        self._started = False
        logger.info("Kernel 已关闭")

    async def __aenter__(self) -> "Kernel":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()
