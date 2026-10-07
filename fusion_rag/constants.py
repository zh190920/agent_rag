"""跨模块共享的常量集合。

放置那些散落在多文件、语义一致但字面容易漂移的小常量。目前主要收敛
文本扩展名白名单——过去在 knowledge/loader / tools/file_tools /
tools/path_jail / agents/tool_agent 里各写各的,加一种语言要改 4 处。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 文本扩展名
# ---------------------------------------------------------------------------
# INDEXABLE: knowledge loader 允许索引的扩展名(保守集,新增会拖慢 reindex)。
INDEXABLE_SUFFIXES: frozenset[str] = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".log", ".py",
})

# DEFAULT_GLOBS: loader 未指定 glob 时的默认扫描模式(与 INDEXABLE 对齐)。
DEFAULT_GLOBS: tuple[str, ...] = (
    "**/*.txt", "**/*.md", "**/*.markdown", "**/*.rst",
)

# PROBE: path_jail 在模型只给文件名不带后缀时,按顺序试探性补全。
PROBE_SUFFIXES: tuple[str, ...] = (".md", ".markdown", ".txt", ".rst")

# TEXT_SCAN: grep / read_file 判定"这个文件是否按文本处理"。宽集,
#   含常见代码/配置格式;命中其一就按 UTF-8 尝试读取,否则视为二进制跳过。
TEXT_SCAN_SUFFIXES: frozenset[str] = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".json",
    ".yaml", ".yml", ".csv", ".log", ".html", ".css", ".xml", ".ini",
    ".conf", ".toml", ".go", ".java", ".c", ".cpp", ".h", ".sh",
})

# CITE: 供 tool_agent._CITE_RE 拼正则,匹配 `basename.ext:line` 行级引用。
#   与 TEXT_SCAN 保持同步(等价集合),但正则里要拆成正则字符类,单独导出。
CITE_SUFFIXES: tuple[str, ...] = tuple(sorted(TEXT_SCAN_SUFFIXES, key=lambda s: s[1:]))


def cite_extension_alternation() -> str:
    """把 CITE_SUFFIXES 拼成 regex 的 alternation(如 `md|markdown|txt|...`)。

    去掉前导点、按长度降序排,保证 `markdown` 会优先匹配在 `md` 之前
    (regex alternation 是最左匹配,不加长序会截短)。
    """
    exts = [s.lstrip(".") for s in CITE_SUFFIXES]
    exts.sort(key=len, reverse=True)
    return "|".join(exts)
