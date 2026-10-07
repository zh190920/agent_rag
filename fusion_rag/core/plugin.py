"""插件注册中心（微内核底座的核心，借鉴 DeepSeek-Harness 插件化设计）。

一切可替换能力（llm / embedding / vector_store / retriever / reranker /
session_store / memory / security / tracer / metrics / alerter）均以
``(kind, name)`` 注册为插件；支持：

- 工厂延迟实例化（按需加载，避免启动即建连接）
- 版本记录与同名覆盖告警
- 单例缓存（同一插件多次 get 只实例化一次）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .exceptions import PluginNotFoundError
from .logging import get_logger

logger = get_logger(__name__)

PluginFactory = Callable[..., Any]


@dataclass
class PluginSpec:
    """一个插件的注册元信息。"""

    kind: str
    name: str
    factory: PluginFactory
    version: str = "0.1.0"
    kwargs: dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """全局插件注册表。

    用法::

        registry.register("llm", "echo", EchoLLM)
        llm = registry.create("llm", "echo")          # 每次新实例
        llm = registry.get("llm", "echo")             # 单例缓存
    """

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], PluginSpec] = {}
        self._instances: dict[tuple[str, str], Any] = {}

    def register(
        self,
        kind: str,
        name: str,
        factory: PluginFactory,
        *,
        version: str = "0.1.0",
        replace: bool = False,
        **kwargs: Any,
    ) -> None:
        key = (kind, name)
        if key in self._specs and not replace:
            logger.warning(
                "插件 %s/%s 已存在（v%s），使用 replace=True 覆盖为 v%s",
                kind, name, self._specs[key].version, version,
            )
            return
        self._specs[key] = PluginSpec(kind, name, factory, version, kwargs)
        self._instances.pop(key, None)
        logger.debug("注册插件 %s/%s v%s", kind, name, version)

    def create(self, kind: str, name: str, **extra: Any) -> Any:
        """按注册工厂创建新实例。"""
        key = (kind, name)
        spec = self._specs.get(key)
        if spec is None:
            raise PluginNotFoundError(
                f"插件未注册: kind={kind} name={name}；已注册: {self.list_names(kind)}",
            )
        return spec.factory(**{**spec.kwargs, **extra})

    def get(self, kind: str, name: str, **extra: Any) -> Any:
        """单例获取（首次触发实例化）。"""
        key = (kind, name)
        if key not in self._instances:
            self._instances[key] = self.create(kind, name, **extra)
        return self._instances[key]

    def has(self, kind: str, name: str) -> bool:
        return (kind, name) in self._specs

    def list_names(self, kind: str) -> list[str]:
        return [name for (k, name) in self._specs if k == kind]

    def list_specs(self, kind: str | None = None) -> Iterable[PluginSpec]:
        return [
            spec for (k, _), spec in self._specs.items()
            if kind is None or k == kind
        ]

    async def aclose_all(self) -> None:
        """优雅关闭：对所有实现了 ``aclose`` 的单例逆序释放。"""
        for key in reversed(list(self._instances)):
            instance = self._instances.pop(key)
            aclose = getattr(instance, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    logger.exception("插件 %s/%s 关闭失败", *key)

    def clear(self) -> None:
        self._specs.clear()
        self._instances.clear()
