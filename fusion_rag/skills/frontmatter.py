"""SKILL.md YAML frontmatter 解析。

优先使用 PyYAML（项目已把 yaml 作为可选依赖，见 ``core.config``）；**未安装
PyYAML 时**退化到内置的容错解析器，覆盖 Agent Skills 常用子集：

- ``key: value`` 标量（自动去引号）
- 折叠 / 字面块 ``>-`` ``>`` ``|-`` ``|``（缩进续行）
- 行内列表 ``[a, b]`` 与 ``- item`` 列表
- 一层嵌套映射（如 ``metadata:`` 下的 ``k: v``）

目标是零硬依赖也能可靠解析出 name/description/license/allowed-tools/metadata。
"""

from __future__ import annotations

import re
from typing import Any

_KEY_RE = re.compile(r"^([A-Za-z0-9_][\w\-]*):\s*(.*)$")


def split_frontmatter(text: str) -> tuple[str, str]:
    """把 Markdown 文本切分为 (frontmatter 原文, 正文)。无 frontmatter 时返回 ('', text)。"""
    if not text:
        return "", ""
    # 兼容 BOM / CRLF
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n")
    lines = normalized.split("\n")
    if not lines or lines[0].strip() != "---":
        return "", normalized
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            fm = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1:])
            return fm, body
    return "", normalized  # 未闭合，视作无 frontmatter


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """解析 SKILL.md，返回 (metadata dict, 正文)。"""
    fm, body = split_frontmatter(text)
    if not fm.strip():
        return {}, body
    meta = _parse_mapping(fm)
    return (meta if isinstance(meta, dict) else {}), body


def _parse_mapping(fm_text: str) -> dict[str, Any]:
    try:  # 首选 PyYAML
        import yaml  # type: ignore

        data = yaml.safe_load(fm_text)
        if isinstance(data, dict):
            return data
        if data is None:
            return {}
    except ImportError:  # pragma: no cover - 环境相关
        pass
    except Exception:  # noqa: BLE001 - YAML 语法错误 → 容错解析
        pass
    return _fallback_parse(fm_text)


def _fallback_parse(fm_text: str) -> dict[str, Any]:
    lines = fm_text.split("\n")
    result: dict[str, Any] = {}
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        # 仅处理顶格键（无缩进），缩进行属于块/嵌套，跳过由上层收集
        if raw[:1] in (" ", "\t"):
            i += 1
            continue
        m = _KEY_RE.match(raw)
        if not m:
            i += 1
            continue
        key, value = m.group(1), m.group(2).strip()
        i += 1
        if value in (">-", ">", "|-", "|", ""):
            block, i = _collect_block(lines, i)
            if key == "metadata":
                result[key] = _parse_nested_map(block)
            elif value in ("|-", "|"):
                result[key] = "\n".join(block).strip()
            else:  # 折叠：多行并一行
                result[key] = _join_scalar(value, block)
        else:
            result[key] = _scalar(value)
    return result


def _collect_block(lines: list[str], i: int) -> tuple[list[str], int]:
    """收集后续「缩进/空行」构成的块，返回 (去缩进内容行, 新行号)。"""
    block: list[str] = []
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.strip() == "":
            block.append("")
            i += 1
            continue
        if line[:1] in (" ", "\t") or line.lstrip().startswith("- "):
            block.append(line.strip())
            i += 1
        else:
            break
    while block and block[-1] == "":
        block.pop()
    return block, i


def _parse_nested_map(block: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block:
        m = _KEY_RE.match(line)
        if m:
            out[m.group(1)] = _scalar(m.group(2).strip())
    return out


def _join_scalar(style: str, block: list[str]) -> str:
    parts = [x for x in block if x != ""]
    if style in (">-", ">"):
        return " ".join(parts)
    return " ".join(parts)


def _scalar(value: str) -> Any:
    """把标量值转成 str / list（行内列表）。"""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
    return v


def normalize_allowed_tools(raw: Any) -> list[str]:
    """把 allowed-tools 归一为工具名列表：接受 list、空格/逗号分隔字符串。"""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    parts = re.split(r"[,\s]+", text)
    return [p for p in parts if p]
