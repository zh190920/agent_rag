"""Skill 数据模型与 Agent Skills 规范约束。

借鉴 **Anthropic Agent Skills**（deepagents ``skills.py``）与 **hermes**
``skill_preprocessing`` 的做法：一个 skill 就是一个**目录**，内含带 YAML
frontmatter 的 ``SKILL.md``（正文是给模型的**流程说明书**），可选附带脚本 /
参考文档等支撑文件。

与「工具（tool）」的区别：
- **tool** = 可被 function-calling 调用的**函数**（如 grep / read_file）；
- **skill** = 教模型「**怎么做某类任务**」的**知识/流程**，通过渐进式披露按需读入。

本项目采用渐进式披露：只把 name+description+路径注入 system prompt，模型判断
任务匹配后，用**已有的 ``read_file`` 工具**读取 ``SKILL.md`` 全文再照其执行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Agent Skills 规范：name 为小写字母/数字 + 连字符，最长 64；description 最长 1024
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESCRIPTION = 1024

# 模板变量：${SKILL_DIR} / ${SESSION_ID} / ${TENANT_ID}（无值的 token 原样保留）
_TEMPLATE_RE = re.compile(r"\$\{(SKILL_DIR|SESSION_ID|TENANT_ID)\}")


def validate_name(name: str, directory_name: str = "") -> tuple[bool, str]:
    """校验 skill 名是否符合 Agent Skills 规范。返回 (是否合法, 说明)。"""
    if not name:
        return False, "name 不能为空"
    if len(name) > MAX_NAME:
        return False, f"name 超过 {MAX_NAME} 字符"
    if not _NAME_RE.match(name):
        hint = ""
        if directory_name and name != directory_name:
            hint = f"（建议与目录名 {directory_name!r} 一致，且仅用小写字母/数字/连字符）"
        return False, f"name 只允许小写字母、数字与连字符 {hint}"
    return True, ""


@dataclass
class Skill:
    """解析后的一个 skill 元信息 + 正文。"""

    name: str
    description: str
    path: str                       # SKILL.md 展示路径（POSIX）
    dir: str                        # skill 目录绝对路径（用于 ${SKILL_DIR} 与 jail）
    source: str = ""                # 来源层标签（base/user/project...）
    license: str = ""
    compatibility: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    body: str = ""                  # frontmatter 之后的 Markdown 正文

    @property
    def short_description(self) -> str:
        """截断到规范上限的描述（用于列表展示）。"""
        text = (self.description or "").strip()
        return text[:MAX_DESCRIPTION]

    def render_body(
        self, *, session_id: str | None = None, tenant_id: str | None = None,
    ) -> str:
        """把正文里的 ``${...}`` 模板变量替换为实际值（借鉴 hermes）。

        无对应值的 token 原样保留，便于作者发现未替换变量。
        """
        values = {
            "SKILL_DIR": self.dir,
            "SESSION_ID": session_id,
            "TENANT_ID": tenant_id,
        }
        return _TEMPLATE_RE.sub(
            lambda m: values.get(m.group(1)) or m.group(0), self.body,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.short_description,
            "path": self.path,
            "dir": self.dir,
            "source": self.source,
            "license": self.license,
            "compatibility": self.compatibility,
            "metadata": dict(self.metadata),
            "allowed_tools": list(self.allowed_tools),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Skill name={self.name!r} source={self.source!r}>"
