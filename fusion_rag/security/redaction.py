"""数据脱敏：入库前与输出前双向脱敏（OpenClaw 合规输出）。

内置中国大陆常见 PII 与凭据的正则规则，命中即替换为占位符（保留类型
标识，便于审计），并统计命中种类与次数。支持自定义规则扩展。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# 内置识别正则（掩码逻辑见下方 _FUNC_RULES / _TEMPLATE_RULES）
_PATTERNS: dict[str, re.Pattern[str]] = {
    # 中国大陆手机号（避免误伤普通长数字：前后不接数字）
    "phone": re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)"),
    # 身份证号（18 位，末位可为 X）
    "id_card": re.compile(r"(?<!\d)(\d{6})(\d{8})(\d{3}[\dXx])(?!\d)"),
    # 邮箱
    "email": re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})"),
    # 银行卡（16-19 位）
    "bank_card": re.compile(r"(?<!\d)(\d{16,19})(?!\d)"),
    # IPv4
    "ipv4": re.compile(r"(?<!\d)((?:\d{1,3}\.){3}\d{1,3})(?!\d)"),
    # API Key / Token / Secret 赋值
    "secret": re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|access[_-]?key|"
        r"authorization)\b(\s*[:=]\s*)([^\s,;\"']+)",
    ),
    # Bearer / sk- 开头的密钥
    "bearer": re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]+"),
    "sk_key": re.compile(r"\b(sk-[A-Za-z0-9]{10,})\b"),
    # PEM 私钥块（BEGIN/END 中间含 base64 body）—目前 audit 里唯一没接住的
    # 高敏感形式。非贪欲匹配 + DOTALL 保证多行块中一次到位，与 BEGIN 同
    # 名的 END 行一起吞掉（否则脱敏后残留 base64 行会泄泄信息）。
    "pem_key": re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
        r"[\s\S]*?"
        r"-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
    ),
}


@dataclass
class RedactionResult:
    """脱敏结果。"""

    text: str
    count: int = 0
    kinds: list[str] = field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return self.count > 0


def _mask_phone(match: re.Match[str]) -> str:
    value = match.group(1)
    return f"<手机号:{value[:3]}****{value[-4:]}>"


def _mask_id(match: re.Match[str]) -> str:
    return f"<身份证:{match.group(1)}********{match.group(3)[-1]}>"


def _mask_email(match: re.Match[str]) -> str:
    local, domain = match.group(1), match.group(2)
    head = local[0] if local else "*"
    return f"<邮箱:{head}***@{domain}>"


def _mask_bank(match: re.Match[str]) -> str:
    value = match.group(1)
    return f"<银行卡:{value[:4]}****{value[-4:]}>"


# 需要自定义函数的规则（正则替换无法表达掩码逻辑）
_FUNC_RULES: list[tuple[str, re.Pattern[str], object]] = [
    ("phone", _PATTERNS["phone"], _mask_phone),
    ("id_card", _PATTERNS["id_card"], _mask_id),
    ("email", _PATTERNS["email"], _mask_email),
    ("bank_card", _PATTERNS["bank_card"], _mask_bank),
]
# 直接模板替换的规则
_TEMPLATE_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("ipv4", _PATTERNS["ipv4"], "<IP>"),
    ("secret", _PATTERNS["secret"], r"\1\2***"),
    ("bearer", _PATTERNS["bearer"], r"\1***"),
    ("sk_key", _PATTERNS["sk_key"], "<密钥:***>"),
    ("pem_key", _PATTERNS["pem_key"], "<私钥块:已脱敏>"),
]


class Redactor:
    """正则脱敏器。"""

    def __init__(
        self,
        enabled: bool = True,
        extra_rules: Iterable[tuple[str, re.Pattern[str], str]] | None = None,
    ) -> None:
        self.enabled = enabled
        self._extra = list(extra_rules or [])

    def redact(self, text: str) -> RedactionResult:
        if not self.enabled or not text:
            return RedactionResult(text=text, count=0, kinds=[])

        count = 0
        kinds: list[str] = []
        result = text

        for name, pattern, func in _FUNC_RULES:
            result, n = pattern.subn(func, result)  # type: ignore[arg-type]
            if n:
                count += n
                kinds.append(name)

        for name, pattern, template in _TEMPLATE_RULES:
            result, n = pattern.subn(template, result)
            if n:
                count += n
                kinds.append(name)

        for name, pattern, template in self._extra:
            result, n = pattern.subn(template, result)
            if n:
                count += n
                kinds.append(name)

        return RedactionResult(text=result, count=count, kinds=_dedup(kinds))

    def redact_many(self, texts: list[str]) -> list[RedactionResult]:
        return [self.redact(t) for t in texts]


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
