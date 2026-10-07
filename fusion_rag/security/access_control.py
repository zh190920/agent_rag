"""访问控制：知识库隔离 + 租户 ACL（借鉴 agentscope metadata_filter 纵深防御）。

核心不变量：**任何检索/写入都必带 tenant_id 作用域**，越权访问在入口即
被拦截，从源头杜绝跨租户数据泄露。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.exceptions import AccessDeniedError
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TenantConfig:
    """单租户配置。"""

    tenant_id: str
    kbs: list[str] = field(default_factory=lambda: ["general"])
    rate_limit_rps: float = 10.0
    burst: int = 20
    meta: dict[str, Any] = field(default_factory=dict)


class AccessController:
    """租户与知识库访问控制器。"""

    def __init__(
        self,
        tenants: dict[str, TenantConfig],
        default_tenant: str = "default",
    ) -> None:
        self._tenants = dict(tenants)
        self._default_tenant = default_tenant
        if default_tenant not in self._tenants:
            # 兜底：至少存在默认租户，避免启动即不可用
            self._tenants[default_tenant] = TenantConfig(tenant_id=default_tenant)

    @classmethod
    def from_config(cls, security_section: dict[str, Any]) -> "AccessController":
        default_tenant = str(security_section.get("default_tenant", "default"))
        raw = security_section.get("tenants", {}) or {}
        tenants: dict[str, TenantConfig] = {}
        for tid, cfg in raw.items():
            cfg = cfg or {}
            tenants[tid] = TenantConfig(
                tenant_id=tid,
                kbs=list(cfg.get("kbs", ["general"])),
                rate_limit_rps=float(cfg.get("rate_limit_rps", 10.0)),
                burst=int(cfg.get("burst", 20)),
                meta={k: v for k, v in cfg.items()
                      if k not in ("kbs", "rate_limit_rps", "burst")},
            )
        return cls(tenants, default_tenant)

    # ------------------------------------------------------------------
    def resolve_tenant(self, tenant_id: str | None) -> str:
        """归一租户 id：未知租户回落到默认租户（可在此改为拒绝）。"""
        if tenant_id and tenant_id in self._tenants:
            return tenant_id
        if tenant_id:
            logger.warning("未知租户 %s，回落到默认租户 %s", tenant_id, self._default_tenant)
        return self._default_tenant

    def tenant(self, tenant_id: str) -> TenantConfig:
        tid = self.resolve_tenant(tenant_id)
        return self._tenants[tid]

    def allowed_kbs(self, tenant_id: str) -> list[str]:
        return list(self.tenant(tenant_id).kbs)

    def check_kb_access(self, tenant_id: str, kb_id: str) -> None:
        """校验租户对单个 KB 的访问权，越权抛 :class:`AccessDeniedError`。"""
        tid = self.resolve_tenant(tenant_id)
        if kb_id not in self._tenants[tid].kbs:
            raise AccessDeniedError(tid, kb_id)

    def resolve_kbs(
        self, tenant_id: str, requested: list[str] | None,
    ) -> list[str]:
        """把请求的 KB 列表裁剪到租户可见范围；未指定则用全部可见 KB。

        对越权请求：记录告警并剔除，而非整体失败（宽松策略，可按需改严格）。
        """
        tid = self.resolve_tenant(tenant_id)
        allowed = self._tenants[tid].kbs
        if not requested:
            return list(allowed)
        resolved: list[str] = []
        for kb in requested:
            if kb in allowed:
                resolved.append(kb)
            else:
                logger.warning("租户 %s 越权请求知识库 %s，已拦截", tid, kb)
        return resolved or list(allowed)

    def scope_filter(
        self, tenant_id: str, kb_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """构造检索作用域过滤器（tenant + kb）。

        该 dict 会被检索层强制注入，确保结果永不逃逸租户边界。
        """
        tid = self.resolve_tenant(tenant_id)
        kbs = self.resolve_kbs(tid, kb_ids)
        return {"tenant_id": tid, "kb_id": kbs}

    def rate_limit(self, tenant_id: str) -> tuple[float, int]:
        cfg = self.tenant(tenant_id)
        return cfg.rate_limit_rps, cfg.burst

    def all_tenants(self) -> list[str]:
        return list(self._tenants)
