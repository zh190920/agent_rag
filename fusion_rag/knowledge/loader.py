"""文档加载器：把本地文件读入为待索引文本（OpenClaw 本地化隐私留存）。

支持常见纯文本/Markdown 格式；二进制或未知格式跳过并记录。加载全程
本地进行，不上传任何云端。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..core.logging import get_logger
from ..constants import DEFAULT_GLOBS, INDEXABLE_SUFFIXES

logger = get_logger(__name__)

_TEXT_SUFFIXES = INDEXABLE_SUFFIXES
_DEFAULT_GLOBS = DEFAULT_GLOBS


@dataclass
class LoadedFile:
    """一个已加载的本地文件。"""

    path: Path
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_file(path: str | Path, *, title: str | None = None) -> LoadedFile | None:
    """读取单个文本文件；不可读或非文本返回 None。"""
    p = Path(path).expanduser()
    if not p.is_file():
        logger.warning("跳过：文件不存在 %s", p)
        return None
    if p.suffix.lower() not in _TEXT_SUFFIXES:
        logger.warning("跳过：非文本格式 %s", p)
        return None
    try:
        content = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("跳过：读取失败 %s (%s)", p, exc)
        return None
    return LoadedFile(
        path=p,
        title=title or p.stem,
        content=content,
        metadata={"source": str(p), "suffix": p.suffix.lower()},
    )


def load_directory(
    root: str | Path,
    *,
    globs: Iterable[str] | None = None,
    max_files: int = 10_000,
) -> list[LoadedFile]:
    """递归加载目录下所有匹配文件。"""
    base = Path(root).expanduser()
    if not base.is_dir():
        logger.warning("目录不存在：%s", base)
        return []
    patterns = list(globs) if globs else list(_DEFAULT_GLOBS)
    seen: set[Path] = set()
    files: list[LoadedFile] = []
    for pattern in patterns:
        for p in base.glob(pattern):
            rp = p.resolve()
            if rp in seen or not p.is_file():
                continue
            seen.add(rp)
            loaded = load_file(p)
            if loaded is not None:
                files.append(loaded)
            if len(files) >= max_files:
                logger.warning("达到 max_files=%d，停止加载", max_files)
                return files
    logger.info("从 %s 加载 %d 个文件", base, len(files))
    return files
