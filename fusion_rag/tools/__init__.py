"""运行时 agentic 工具集（对标 Claude Code / hermes harness）。

让真实大模型在问答运行时通过原生 function-calling 自主多轮调用：
- 知识库检索：``search_kb``（将 HybridRetriever 包装为工具）
- 只读文件工具：``grep`` / ``glob`` / ``find`` / ``list_dir`` / ``read_file``

文件类工具经 :class:`~fusion_rag.tools.path_jail.PathJail` jail 在租户授权目录
内，复用 Sandbox 护栏、Tracer 轨迹与 Metrics 指标。

- :mod:`base`        —— Tool 抽象、ToolContext、ToolResult
- :mod:`path_jail`   —— 路径 jail（拦截越界/遍历/软链逃逸/设备文件）
- :mod:`file_tools`  —— 5 个只读文件工具
- :mod:`search_tool` —— 将混合检索包装为模型可选工具
- :mod:`registry`    —— 工具注册表 + OpenAI function schema 生成
- :mod:`executor`    —— 工具调度（超时/异常隔离/埋点）
"""

from __future__ import annotations

from .base import Tool, ToolContext, ToolResult
from .executor import ToolExecutor
from .file_tools import (
    FindTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
)
from .path_jail import PathJail
from .registry import ToolRegistry
from .search_tool import SearchKBTool

__all__ = [
    "Tool",
    "ToolContext",
    "ToolResult",
    "ToolExecutor",
    "ToolRegistry",
    "PathJail",
    "GrepTool",
    "GlobTool",
    "FindTool",
    "ListDirTool",
    "ReadFileTool",
    "SearchKBTool",
]
