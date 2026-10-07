"""统一异常体系。

所有框架内异常继承 :class:`FusionRagError`，便于 API 层统一捕获与
运维层按类型统计告警。
"""

from __future__ import annotations


class FusionRagError(Exception):
    """框架异常基类。"""


class ConfigError(FusionRagError):
    """配置缺失或非法。"""


class PluginError(FusionRagError):
    """插件注册/解析/加载失败。"""


class PluginNotFoundError(PluginError):
    """按 (kind, name) 未找到已注册插件。"""


class LLMError(FusionRagError):
    """模型调用失败（网络、超时、鉴权、协议）。"""


class LLMTimeoutError(LLMError):
    """模型调用超时。"""


class EmbeddingError(FusionRagError):
    """嵌入调用失败。"""


class RetrievalError(FusionRagError):
    """检索失败。"""


class StorageError(FusionRagError):
    """持久化存储失败。"""


class AccessDeniedError(FusionRagError):
    """租户越权访问知识库 / ACL 拒绝。"""

    def __init__(self, tenant_id: str, kb_id: str) -> None:
        super().__init__(f"tenant={tenant_id} 无权访问知识库 kb={kb_id}")
        self.tenant_id = tenant_id
        self.kb_id = kb_id


class RateLimitError(FusionRagError):
    """租户限流触发。"""


class TimeoutBudgetError(FusionRagError):
    """请求级超时护栏触发。"""


class ValidationError(FusionRagError):
    """答案校验不通过（幻觉/引用缺失/逻辑冲突）。"""


class HITLRequiredError(FusionRagError):
    """高风险问题需要人工介入。"""
