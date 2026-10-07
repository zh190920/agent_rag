"""Skills 子系统：教 Agent「怎么做某类任务」的可插拔流程说明书。

借鉴 **Anthropic Agent Skills**（deepagents ``middleware/skills.py``）与
**hermes** ``skill_preprocessing`` 的成熟做法，落地为项目内一等公民：

- 一个 skill = 一个目录，内含带 YAML frontmatter 的 ``SKILL.md``；
- **渐进式披露**：只把 name+description+路径注入 system prompt，模型按需
  用已有的 ``read_file`` 工具读全文再执行（省 token、可组合工具）；
- **分层来源**：base → user → project，后层同名覆盖前层。

与工具（``tools/``）互补：tool 是可被 function-calling 调用的**函数**，skill
是指导模型如何编排工具的**知识/流程**。

- :mod:`base`        —— Skill 数据模型、名称校验、模板替换
- :mod:`frontmatter` —— SKILL.md YAML frontmatter 解析（PyYAML + 容错回退）
- :mod:`loader`      —— 目录扫描、分层来源合并、同名覆盖
- :mod:`registry`    —— SkillRegistry + 渐进披露提示词渲染
"""

from __future__ import annotations

from .base import MAX_DESCRIPTION, MAX_NAME, Skill, validate_name
from .frontmatter import normalize_allowed_tools, parse_frontmatter, split_frontmatter
from .loader import SKILL_FILENAME, build_skill, load_skill_dir, load_skills, scan_layer
from .registry import SkillRegistry

__all__ = [
    "Skill",
    "SkillRegistry",
    "validate_name",
    "MAX_NAME",
    "MAX_DESCRIPTION",
    "parse_frontmatter",
    "split_frontmatter",
    "normalize_allowed_tools",
    "load_skills",
    "load_skill_dir",
    "scan_layer",
    "build_skill",
    "SKILL_FILENAME",
]
