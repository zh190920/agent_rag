"""Skills 子系统测试：frontmatter 解析、loader 分层加载、registry 渐进披露。

覆盖 Anthropic Agent Skills 规范要点：SKILL.md frontmatter（name/description/
allowed-tools/metadata）、名称校验、分层同名覆盖、渐进披露提示词渲染，以及
与已有容错解析器（无 PyYAML 时的回退）的一致性。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fusion_rag.skills import Skill, SkillRegistry, load_skills, validate_name
from fusion_rag.skills.frontmatter import (
    _fallback_parse,
    normalize_allowed_tools,
    parse_frontmatter,
    split_frontmatter,
)
from fusion_rag.skills.loader import load_skill_dir, scan_layer

SKILL_MD = """\
---
name: report-writer
description: 依据检索到的原始文档撰写结构化技术报告。
license: MIT
allowed-tools: [grep, read_file]
metadata:
  team: knowledge
  version: 2
---

# 报告撰写流程
1. 用 grep 定位关键论据行号；
2. 用 read_file 精读相关片段；
3. 按「结论先行 + 文件名:行号 引用」组织正文。
工作目录：${SKILL_DIR}
"""


# ----------------------------------------------------------------------
# frontmatter
# ----------------------------------------------------------------------
def test_split_frontmatter_basic():
    fm, body = split_frontmatter(SKILL_MD)
    assert "name: report-writer" in fm
    assert body.lstrip().startswith("# 报告撰写流程")


def test_split_frontmatter_absent():
    fm, body = split_frontmatter("# 只有正文\n无 frontmatter")
    assert fm == ""
    assert body.startswith("# 只有正文")


def test_parse_frontmatter_fields():
    meta, body = parse_frontmatter(SKILL_MD)
    assert meta["name"] == "report-writer"
    assert meta["description"].startswith("依据检索")
    assert meta["license"] == "MIT"
    assert meta["allowed-tools"] == ["grep", "read_file"]
    assert meta["metadata"]["team"] == "knowledge"
    assert "报告撰写流程" in body


def test_fallback_parser_matches_semantics():
    # 不依赖 PyYAML：内置容错解析器也能取到关键字段
    fm, _ = split_frontmatter(SKILL_MD)
    meta = _fallback_parse(fm)
    assert meta["name"] == "report-writer"
    assert meta["allowed-tools"] == ["grep", "read_file"]
    assert meta["metadata"]["version"] == "2"


def test_normalize_allowed_tools_variants():
    assert normalize_allowed_tools("grep read_file") == ["grep", "read_file"]
    assert normalize_allowed_tools("grep, read_file") == ["grep", "read_file"]
    assert normalize_allowed_tools(["grep", " glob "]) == ["grep", "glob"]
    assert normalize_allowed_tools(None) == []


# ----------------------------------------------------------------------
# base：名称校验 + 模板替换
# ----------------------------------------------------------------------
def test_validate_name_rules():
    assert validate_name("report-writer")[0] is True
    assert validate_name("Bad_Name")[0] is False       # 大写/下划线非法
    assert validate_name("")[0] is False
    assert validate_name("a" * 65)[0] is False         # 超 64 字符


def test_skill_render_body_template():
    skill = Skill(
        name="demo", description="d", path="/x/SKILL.md", dir="/x",
        body="目录 ${SKILL_DIR} 会话 ${SESSION_ID} 缺失 ${TENANT_ID}",
    )
    out = skill.render_body(session_id="s-42")
    assert "/x" in out
    assert "s-42" in out
    assert "${TENANT_ID}" in out                       # 无值 token 原样保留


# ----------------------------------------------------------------------
# loader：目录扫描 + 分层覆盖
# ----------------------------------------------------------------------
def _write_skill(root: Path, name: str, *, description: str = "做某类任务的流程") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# 正文\n步骤。\n",
        encoding="utf-8",
    )
    return d


def test_load_skill_dir_and_scan_layer():
    root = Path(tempfile.mkdtemp())
    _write_skill(root, "report-writer")
    _write_skill(root, "summarizer")
    (root / "not-a-skill").mkdir()  # 无 SKILL.md → 忽略

    skill = load_skill_dir(root / "report-writer", source="base")
    assert skill is not None and skill.name == "report-writer"
    assert skill.source == "base"
    assert "步骤" in skill.body

    found = scan_layer(root, source="base")
    assert {s.name for s in found} == {"report-writer", "summarizer"}


def test_load_skill_dir_requires_description():
    root = Path(tempfile.mkdtemp())
    d = root / "no-desc"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: no-desc\n---\n正文\n", encoding="utf-8")
    assert load_skill_dir(d) is None                   # 缺 description → 跳过


def test_load_skills_layered_override():
    base = Path(tempfile.mkdtemp())
    project = Path(tempfile.mkdtemp())
    _write_skill(base, "report-writer", description="内置版")
    _write_skill(base, "summarizer", description="只有内置有")
    _write_skill(project, "report-writer", description="项目定制版")

    merged = load_skills([("base", base), ("project", project)])
    assert merged["report-writer"].description == "项目定制版"
    assert merged["report-writer"].source == "project"  # 后层覆盖
    assert merged["summarizer"].source == "base"        # 前层保留


# ----------------------------------------------------------------------
# registry：渐进披露渲染 + jail 根 + 展示路径
# ----------------------------------------------------------------------
def _registry_from_layer() -> tuple[SkillRegistry, Path]:
    root = Path(tempfile.mkdtemp())
    _write_skill(root, "report-writer")
    skills = scan_layer(root, source="base")
    return SkillRegistry(skills, roots=[str(root)]), root


def test_registry_roots_and_display_path():
    reg, root = _registry_from_layer()
    assert str(root) in reg.roots()
    skill = reg.get("report-writer")
    assert skill is not None
    # 展示路径相对 jail 根，模型可直接 read_file
    assert reg.display_path(skill) == "report-writer/SKILL.md"


def test_registry_render_prompt_lists_skills():
    reg, _ = _registry_from_layer()
    prompt = reg.render_prompt(session_id="s1", tenant_id="default")
    assert "可用技能" in prompt
    assert "report-writer" in prompt
    assert "read_file" in prompt                        # 指导按需读取
    assert "report-writer/SKILL.md" in prompt


def test_registry_render_prompt_includes_allowed_tools_hint():
    root = Path(tempfile.mkdtemp())
    d = _write_skill(root, "report-writer")
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    reg = SkillRegistry(scan_layer(root, source="base"), roots=[str(root)])
    prompt = reg.render_prompt()
    assert "建议工具：grep, read_file" in prompt


def test_registry_empty_renders_nothing():
    reg = SkillRegistry([], roots=[])
    assert not reg
    assert reg.render_prompt() == ""
