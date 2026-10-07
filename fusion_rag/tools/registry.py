"""ToolRegistry：工具注册表 + OpenAI function schema 生成。

集中管理可被模型调用的工具，按启用白名单裁剪，并统一产出随请求下发给
模型的 ``tools`` 定义列表。
"""

from __future__ import annotations

from typing import Iterable

from .base import Tool


class ToolRegistry:
    """(name -> Tool) 注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        if not tool.name:
            raise ValueError("工具必须声明 name")
        if tool.name in self._tools and not replace:
            raise ValueError(f"工具 {tool.name} 已注册")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def allowed(self, names: Iterable[str] | None) -> "ToolRegistry":
        """按白名单返回仅含这些工具的新注册表（names 为空则原样返回）。"""
        keep = list(names) if names else None
        view = ToolRegistry()
        for name, tool in self._tools.items():
            if keep is None or name in keep:
                view.register(tool)
        return view

    def openai_schemas(self) -> list[dict]:
        """生成 OpenAI Chat Completions ``tools`` 参数。"""
        return [tool.schema() for tool in self._tools.values()]

    def filter_for_intent(
        self, *, domain: str | None = None,
    ) -> "ToolRegistry":
        """基于意图域硬性筛选：保留 ``applies_to_domains`` 为空或命中当前域的工
        具。如果筛选后为空（完全失配），回退到原注册表，避免“无工具可用”情
        况。返回新视图，不修改原表。
        """
        view = ToolRegistry()
        for name, tool in self._tools.items():
            scopes = tuple(tool.applies_to_domains or ())
            if not scopes or (domain and domain in scopes):
                view.register(tool)
        if len(view) == 0:
            return self
        return view

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
