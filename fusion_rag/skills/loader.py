"""Skill 加载：扫描目录、分层来源合并、同名覆盖、allowed-tools 归一。

借鉴 **Anthropic Agent Skills / deepagents** 的目录约定：一个 skill 就是一个
**目录**，内含带 YAML frontmatter 的 ``SKILL.md``；frontmatter 至少要有
``name`` 与 ``description``，可选 ``license`` / ``allowed-tools`` / ``metadata``。

**分层来源（layered sources）**：与 deepagents 的 base→user→project 一致，按
顺序给出多个「skill 根目录」，逐层扫描其下的子目录，**后层同名 skill 覆盖前
层**（便于随部署环境增量覆盖内置技能）。每层可打标签（``source``）用于展示与
排查。

加载是**纯阻塞文件 IO**，由 :class:`~fusion_rag.skills.registry.SkillRegistry`
在启动装配阶段调用一次，不进入请求热路径。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from ..core.logging import get_logger
from .base import Skill, validate_name
from .frontmatter import normalize_allowed_tools, parse_frontmatter

logger = get_logger(__name__)

# skill 目录内的说明书文件名（规范固定）
SKILL_FILENAME = "SKILL.md"


def load_skill_dir(skill_dir: os.PathLike | str, *, source: str = "") -> Skill | None:
    """加载单个 skill 目录（内含 ``SKILL.md``）。非法/缺失返回 ``None``。"""
    directory = Path(str(skill_dir))
    md_path = directory / SKILL_FILENAME
    if not md_path.is_file():
        return None
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:  # pragma: no cover - 权限/IO 异常
        logger.warning("读取 %s 失败：%s", md_path, exc)
        return None
    return build_skill(
        text, skill_dir=directory, path=md_path, source=source,
    )


def build_skill(
    text: str,
    *,
    skill_dir: Path,
    path: Path,
    source: str = "",
) -> Skill | None:
    """从 ``SKILL.md`` 原文构建 :class:`Skill`；校验失败返回 ``None`` 并告警。"""
    meta, body = parse_frontmatter(text)
    dir_name = skill_dir.name
    name = str(meta.get("name") or "").strip() or dir_name
    ok, reason = validate_name(name, directory_name=dir_name)
    if not ok:
        logger.warning("跳过 skill 目录 %s：name=%r 非法（%s）", skill_dir, name, reason)
        return None
    description = str(meta.get("description") or "").strip()
    if not description:
        logger.warning("跳过 skill %s：缺少 description", name)
        return None
    metadata_raw = meta.get("metadata")
    metadata: dict[str, str] = {}
    if isinstance(metadata_raw, dict):
        metadata = {str(k): str(v) for k, v in metadata_raw.items()}
    return Skill(
        name=name,
        description=description,
        path=path.as_posix(),
        dir=str(skill_dir),
        source=source,
        license=str(meta.get("license") or "").strip(),
        compatibility=str(meta.get("compatibility") or "").strip(),
        metadata=metadata,
        allowed_tools=normalize_allowed_tools(meta.get("allowed-tools")),
        body=body.strip(),
    )


def scan_layer(root: os.PathLike | str, *, source: str = "") -> list[Skill]:
    """扫描一个「skill 根目录」下的所有 skill 子目录。

    - 若 ``root`` 自身含 ``SKILL.md``，视作单个 skill 直接返回；
    - 否则遍历其一级子目录，逐个尝试加载。
    """
    root_path = Path(str(root))
    if not root_path.is_dir():
        return []
    if (root_path / SKILL_FILENAME).is_file():
        skill = load_skill_dir(root_path, source=source)
        return [skill] if skill else []
    found: list[Skill] = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        skill = load_skill_dir(child, source=source)
        if skill:
            found.append(skill)
    return found


def load_skills(
    layers: Iterable[tuple[str, os.PathLike | str]],
) -> dict[str, Skill]:
    """按层加载并合并，返回 ``name -> Skill``。

    ``layers`` 为 ``(source_label, root_dir)`` 序列，**顺序即优先级**：后面的层
    同名覆盖前面的层（base → user → project）。跳过不存在/无权限的根目录。
    """
    merged: dict[str, Skill] = {}
    for label, root in layers:
        for skill in scan_layer(root, source=label):
            existing = merged.get(skill.name)
            if existing is not None:
                logger.debug(
                    "skill %s 被来源 %s 覆盖（原 %s）", skill.name, skill.source, existing.source,
                )
            merged[skill.name] = skill
    return merged
