"""fusion-rag-agent：融合五大主流 Agent 框架优势的企业级 RAG 知识问答智能体框架。

- AgentScope        → 多智能体协作调度、知识库多租户隔离、并发检索合并
- DeepSeek-Harness  → 微内核插件底座、模型无绑定、轨迹回放
- DeepAgents        → 任务规划拆解、上下文卸载、人工介入
- OpenClaw          → 本地化隐私留存、持久化会话
- Claude-Code       → 长文精读、答案校验纠错
"""

__version__ = "0.1.0"

from .core.config import Config, load_config
from .core.kernel import Kernel
from .core.win_proactor_patch import apply_windows_proactor_patch

# Windows Proactor 下 aiohttp transport.__del__ 在 loop 关停后仍会刷
# 一片 "Exception ignored: Event loop is closed"（CPython issue #88050）。
# 非致命但污染 stderr，包载入即安装静默补丁。
apply_windows_proactor_patch()

__all__ = ["Kernel", "Config", "load_config", "__version__"]
