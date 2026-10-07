"""SkillRegistry：技能注册表 + 渐进式披露（progressive disclosure）提示词渲染。

**渐进式披露**（对齐 Anthropic Agent Skills / deepagents）：为节省 token，我们
**不**把所有 ``SKILL.md`` 正文塞进 system prompt，而只列出每个技能的
``name + description + 读取路径``。模型判断某技能与当前任务匹配后，用**已有的
``read_file`` 工具**按需读取该 ``SKILL.md`` 全文，再照其流程执行。

因此注册表对上层只暴露三件事：

1. :meth:`roots` —— 各 skill 层级**根目录**，需并入工具 jail 授权根，模型才读得到；
2. :meth:`render_prompt` —— 注入 ToolAgent system prompt 的技能清单文本；
3. :meth:`get` / :meth:`names` —— 查询已加载技能。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from ..core.logging import get_logger
from .base import Skill

logger = get_logger(__name__)


class SkillRegistry:
    """已加载技能的只读注册表。"""

    def __init__(
        self,
        skills: Iterable[Skill] | None = None,
        *,
        roots: Iterable[os.PathLike | str] | None = None,
    ) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills or []:
            self._skills[skill.name] = skill
        # 层级根目录（绝对、去重、保序）：read_file 需被 jail 授权到这些目录内
        self._roots: list[str] = []
        seen: set[str] = set()
        for root in roots or []:
            key = str(root)
            if key and key not in seen:
                seen.add(key)
                self._roots.append(key)

    # ------------------------------------------------------------------
    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return list(self._skills)

    def all(self) -> list[Skill]:
        """按名称排序返回全部技能，保证提示词渲染稳定可复现。"""
        return [self._skills[n] for n in sorted(self._skills)]

    def roots(self) -> list[str]:
        """需并入工具 jail 的 skill 层级根目录（供 read_file 访问）。"""
        return list(self._roots)

    def __len__(self) -> int:
        return len(self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)

    # ------------------------------------------------------------------
    def display_path(self, skill: Skill) -> str:
        """把 ``SKILL.md`` 表达成「相对某个 jail 根」的可直接 read_file 路径。

        形如 ``<skill-name>/SKILL.md``；找不到匹配层级根时回落绝对路径。
        """
        skill_dir = Path(skill.dir)
        for root in self._roots:
            try:
                rel = skill_dir.relative_to(Path(root)).as_posix()
            except ValueError:
                continue
            return f"{rel}/SKILL.md"
        return skill.path

    def render_prompt(
        self,
        *,
        session_id: str | None = None,
        tenant_id: str | None = None,
    ) -> str:
        """渲染渐进式披露的技能清单段；无技能时返回空串（调用方据此省略）。"""
        if not self._skills:
            return ""
        lines = [
            "可用技能（Skills）——每份都是一类任务的详细流程说明，按需读取：",
        ]
        for skill in self.all():
            desc = skill.short_description.replace("\n", " ").strip()
            route_hint = ""
            if skill.allowed_tools:
                route_hint = f"（建议工具：{', '.join(skill.allowed_tools)}）"
            lines.append(
                f"- {skill.name}: {desc}{route_hint} —— 用 read_file 读取 `{self.display_path(skill)}` 获取完整流程",
            )
        lines.append(
            "使用方式：当某个技能与当前任务匹配时，先用 read_file 读取其 SKILL.md 全文，"
            "再严格照其中的步骤与注意事项执行；无关技能不必读取。",
        )
        return "\n".join(lines)
