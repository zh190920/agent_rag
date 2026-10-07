"""安全层测试：PII 脱敏 + 租户/知识库访问控制。"""

from __future__ import annotations

import pytest

from fusion_rag.core.exceptions import AccessDeniedError
from fusion_rag.security.access_control import AccessController
from fusion_rag.security.redaction import Redactor


def test_redactor_masks_pii():
    redactor = Redactor(enabled=True)
    text = (
        "联系人手机号 13812345678，邮箱 zhangsan@example.com，"
        "身份证 110101199003074258，银行卡 6222021234567890123。"
    )
    result = redactor.redact(text)
    assert result.redacted is True
    assert result.count >= 3
    # 原始敏感串不应再出现
    assert "13812345678" not in result.text
    assert "zhangsan@example.com" not in result.text
    assert "110101199003074258" not in result.text


def test_redactor_disabled_passthrough():
    redactor = Redactor(enabled=False)
    text = "手机号 13812345678"
    result = redactor.redact(text)
    assert result.text == text
    assert result.redacted is False


def test_redactor_masks_api_key():
    redactor = Redactor(enabled=True)
    text = "api_key=sk-abcdefghijklmnopqrstuvwxyz123456"
    result = redactor.redact(text)
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in result.text


def test_access_controller_from_config():
    access = AccessController.from_config({
        "default_tenant": "default",
        "tenants": {
            "default": {"kbs": ["general"], "rate_limit_rps": 5, "burst": 10},
            "acme": {"kbs": ["hr", "legal"], "rate_limit_rps": 20, "burst": 40},
        },
    })
    assert set(access.all_tenants()) == {"default", "acme"}
    assert access.allowed_kbs("acme") == ["hr", "legal"]
    rate, burst = access.rate_limit("acme")
    assert rate == 20 and burst == 40


def test_unknown_tenant_falls_back_to_default():
    access = AccessController.from_config({
        "default_tenant": "default",
        "tenants": {"default": {"kbs": ["general"]}},
    })
    assert access.resolve_tenant("ghost") == "default"


def test_kb_access_denied():
    access = AccessController.from_config({
        "default_tenant": "default",
        "tenants": {"default": {"kbs": ["general"]}},
    })
    with pytest.raises(AccessDeniedError):
        access.check_kb_access("default", "secret-kb")


def test_resolve_kbs_filters_unauthorized():
    access = AccessController.from_config({
        "default_tenant": "default",
        "tenants": {"default": {"kbs": ["general", "public"]}},
    })
    # 越权的 kb 被剔除，只保留可见范围
    resolved = access.resolve_kbs("default", ["general", "secret"])
    assert resolved == ["general"]
    # 未指定则返回全部可见
    assert set(access.resolve_kbs("default", None)) == {"general", "public"}


def test_scope_filter_contains_tenant():
    access = AccessController.from_config({
        "default_tenant": "default",
        "tenants": {"default": {"kbs": ["general"]}},
    })
    flt = access.scope_filter("default")
    assert flt["tenant_id"] == "default"
    assert flt["kb_id"] == ["general"]
