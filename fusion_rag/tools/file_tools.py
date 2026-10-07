"""5 个只读文件工具：grep / glob / find / list_dir / read_file。

对齐 hermes ``file_tools.py`` 的成熟细节，并全部经 :class:`PathJail` jail：

- :class:`GrepTool`     —— 正则/字面量搜索文件内容，输出 ``相对路径:行号: 行内容``（可带前后文/分页）
- :class:`GlobTool`     —— 按 glob 模式定位文件
- :class:`FindTool`     —— 按名称/大小/修改时间/类型定位文件（含排序/分页）
- :class:`ListDirTool`  —— 列出目录一级条目（目录/文件 + 大小）
- :class:`ReadFileTool` —— 分页读取文件（行号前缀 + offset/limit + 字符预算截断到整行）

阻塞 IO 统一经 ``Tool._offload`` 投递到 CPU 线程池，不阻塞事件循环。
所有失败（越权、文件不存在、二进制、过大）返回 ``ToolResult(is_error=True)``，
让模型能读到错误信息并自我修正，而非中断整个 agentic 循环。
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..constants import TEXT_SCAN_SUFFIXES
from .base import Tool, ToolContext, ToolResult
from .path_jail import PathJail

logger = get_logger(__name__)

# 单文件读取上限（字节），超限跳过，防巨型文件拖垮线程
_MAX_FILE_BYTES = 4 * 1024 * 1024
# read_file 单次输出字符预算（对齐 hermes 字符预算截断到整行）
# ★ 从 20000 降到 6000：read_file 回包会全量塞进 messages 作后续每轮 prefill，
#   PLC 手册单行极长（表格/点位表），limit=100 一次就吐 15KB 撑爆尾轮预算导致超时。
_MAX_READ_CHARS = 6000
# 文本扩展名白名单（对齐 knowledge.loader / tool_agent._CITE_RE）
_TEXT_SUFFIXES = TEXT_SCAN_SUFFIXES


def _is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in _TEXT_SUFFIXES:
        return True
    # 无扩展名或小文件：保守尝试（读首块探测空字节）
    try:
        with path.open("rb") as fh:
            head = fh.read(1024)
        return b"\x00" not in head
    except OSError:
        return False


_SCOPE_DENY_MSG = "该文件不在本次检索的任务范围内（子Agent只负责被分配的手册），请换用范围内文件。"


def _scope_denied(path: Path, ctx: ToolContext) -> bool:
    """检查 path 的 basename 是否命中 ``ctx.restrict_to_sources``（未设置限制时永远放行）。

    这是 fanout 并发子 Agent 的“任务作用域”软过滤（非 PathJail 安全边界），
    命中范围外时工具层自己返回软拒绝，不抛异常、不占用 jail 语义。
    """
    allow = getattr(ctx, "restrict_to_sources", None)
    if not allow:
        return False
    return path.name not in allow


class GrepTool(Tool):
    """在授权目录内按内容搜索（对标 hermes search_tool / grep）。"""

    name = "grep"
    description = (
        "精确字符串/正则匹配：逐行扫描文件内容，返回所有包含目标字串的行。"
        "适合查找标识符、型号、参数编号、确切术语的全量出现位置。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "正则或字面量串"},
            "path": {"type": "string", "description": "子目录（相对授权根，默认 '.'）", "default": "."},
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "已知候选文件名列表时传入，只扫这几个文件（大幅提速）",
                "default": [],
            },
            "glob": {"type": "string", "description": "文件名过滤如 *.md", "default": "*"},
            "regex": {"type": "boolean", "description": "true=正则，false=字面量", "default": True},
            "max_count": {"type": "integer", "description": "最多返回条数", "default": 30},
            "skip": {"type": "integer", "description": "跳过前 N 条（分页）", "default": 0},
            "context": {"type": "integer", "description": "前后文行数 0-5", "default": 0},
        },
        "required": ["pattern"],
    }
    best_for = (
        "型号/缩写/参数编号精确匹配（如 FINS/H5U2/Easy320/寄存器号）",
        "已知目标关键词时定位行号",
        "需正则模式（数字+单位、多选项交替）",
    )
    avoid_for = (
        "不确定关键词中文同义词时（先试 search_kb）",
    )

    def __init__(self, cpu_executor: Any | None = None, max_matches: int = 50) -> None:
        super().__init__(cpu_executor)
        self._default_max = max_matches

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        jail = PathJail(ctx.roots, tenant_id=ctx.tenant_id)
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            return ToolResult(content="缺少参数 pattern", is_error=True)
        regex = bool(args.get("regex", True))
        glob = str(args.get("glob", "*"))
        max_count = int(args.get("max_count", self._default_max) or self._default_max)
        max_count = max(1, min(max_count, 500))
        skip = max(0, int(args.get("skip", 0) or 0))
        context = max(0, min(int(args.get("context", 0) or 0), 10))
        # ★ 可选：直接指定候选文件列表（来自上一步 search_kb 结果），把扫描
        # 范围从全库→个位数文件，5000 本手册规模下提速 20× 以上。
        raw_paths = args.get("paths") or []
        target_paths: list[str] = []
        if isinstance(raw_paths, str):
            target_paths = [raw_paths]
        elif isinstance(raw_paths, (list, tuple)):
            target_paths = [str(p).strip() for p in raw_paths if str(p).strip()]
        try:
            if target_paths:
                # 逐个 resolve，失败项不阻断整体（模型偶尔会幻觉文件名）
                files: list[Path] = []
                miss: list[str] = []
                for rel in target_paths:
                    try:
                        files.append(jail.resolve_first_existing(rel))
                    except Exception:  # noqa: BLE001
                        miss.append(rel)
                if not files:
                    return ToolResult(
                        content=f"paths 中没有一个能定位到真实文件：{miss}",
                        is_error=True,
                    )
                bases = files
            else:
                bases = jail.search_bases(args.get("path", "."))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"访问被拒绝：{exc}", is_error=True)
        return await self._offload(
            lambda: self._grep_sync(
                jail, bases, pattern, glob, regex, max_count, skip, context,
                allow=getattr(ctx, "restrict_to_sources", None),
            ),
        )

    def _grep_sync(
        self, jail: PathJail, bases: list[Path], pattern: str, glob: str,
        regex: bool, max_count: int, skip: int, context: int,
        allow: Any | None = None,
    ) -> ToolResult:
        if regex:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                return ToolResult(content=f"非法正则：{exc}", is_error=True)
            match = lambda line, _c=compiled: _c.search(line) is not None  # noqa: E731
        else:
            needle = pattern.lower()
            match = lambda line, _n=needle: _n in line.lower()  # noqa: E731

        out: list[str] = []
        match_seen = 0     # 跨文件累计命中数（含被 skip 跳过的）
        returned = 0       # 本页渲染的命中数
        scanned = 0
        has_more = False
        seen: set[str] = set()
        stop = False
        for base in bases:
            if stop:
                break
            if base.is_file():
                walked = [base]
            else:
                walked = _walk_text_files(base, glob, seen)
            for fp in walked:
                # ★ fanout 任务作用域软过滤：范围外文件直接跳过（不计入 scanned，
                #   避免误报“扫过但没命中”），PathJail 已保证不会越权访问。
                if allow and fp.name not in allow:
                    continue
                scanned += 1
                rel = jail.relative(fp)
                try:
                    lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                matched = [i for i, ln in enumerate(lines) if match(ln)]
                if not matched:
                    continue
                last_end: int | None = None
                for mi in matched:
                    match_seen += 1
                    if match_seen <= skip:
                        continue
                    if returned >= max_count:
                        has_more = True
                        stop = True
                        break
                    start = max(0, mi - context)
                    end = min(len(lines) - 1, mi + context)
                    if context > 0 and last_end is not None and start > last_end + 1:
                        out.append("--")
                    for li in range(start, end + 1):
                        text = lines[li].strip()[:200]
                        if li == mi:
                            out.append(f"{rel}:{li + 1}: {text}")
                        else:
                            out.append(f"{rel}-{li + 1}- {text}")
                    last_end = end
                    returned += 1
                if stop:
                    break

        next_cursor = skip + returned
        if not out:
            if match_seen == 0:
                # ★ 如果 scanned==0 且 bases 明确指定，意味 path 不对 → 直接报错，
                #   避免“扫了但没找到”的误导（模型会自己反思路径拼写）。
                if scanned == 0 and any(b.is_file() or not b.exists() for b in bases):
                    tried = ", ".join(str(getattr(b, "name", b)) for b in bases[:3])
                    return ToolResult(
                        content=(
                            f"❌ paths 里一个可读文件都没扫到（可能已自动补全 .md 仍不存在）。\n"
                            f"尝试过：{tried}\n"
                            f"请确认文件名完全匹配（可从 search_kb 回包的 title 列直接拷取，"
                            f"包含 .md 扩展名），或先调 glob(pattern='<子串>*.md') 定位。"
                        ),
                        is_error=True,
                        meta={"matches": 0, "total_matches": 0, "scanned": 0},
                    )
                return ToolResult(
                    content=f"未找到匹配（pattern={pattern!r}，扫描 {scanned} 个文件）",
                    meta={"matches": 0, "total_matches": 0, "scanned": scanned},
                )
            return ToolResult(
                content=f"skip={skip} 已超出全部命中（共 {match_seen} 条，无更多分页）",
                meta={"matches": 0, "total_matches": match_seen, "scanned": scanned,
                      "has_more": False},
            )
        text = "\n".join(out)
        meta: dict[str, Any] = {
            "matches": returned, "total_matches": match_seen,
            "scanned": scanned, "truncated": has_more, "has_more": has_more,
        }
        if has_more:
            meta["next_cursor"] = next_cursor
            text += f"\n（还有更多命中，本页 {returned} 条；用 skip={next_cursor} 续读）"
        # ★ 字节上限：避免一次 grep 回包 20KB+ 将尾轮 LLM 卡爆预算。
        #   超限时主动截断 + 提示模型用更严的 pattern / 更小 context / skip 分页。
        _MAX_BYTES = 8192
        if len(text.encode("utf-8")) > _MAX_BYTES:
            # 逐行累加到接近阈值，保证不拆断单行命中
            limited: list[str] = []
            total = 0
            for line in text.splitlines():
                lb = len(line.encode("utf-8")) + 1
                if total + lb > _MAX_BYTES - 256:
                    break
                limited.append(line)
                total += lb
            dropped = returned - len(limited)
            meta["byte_truncated"] = True
            meta["dropped_matches"] = dropped
            text = "\n".join(limited) + (
                f"\n\n⚠ 回包已截断到 8KB（少了约 {dropped} 条命中）。"
                "下一轮请收紧：\n"
                "  • 降 context（例如 2→ 0）；\n"
                "  • 缩 pattern（只保留一个关键词）；\n"
                "  • 或直接把目标行号交给 read_file(offset, limit=40) 精读，"
                "不要拿 grep 当“全文回收器”。"
            )
        return ToolResult(content=text, meta=meta)


class GlobTool(Tool):
    """按 glob 模式定位文件（对标 find）。"""

    name = "glob"
    description = (
        "按 glob 模式（如 *H5U*.md）定位知识库文件路径，不扫内容。"
        "【硬约束】禁 `**/*.md` 无差别扫全目录（会白烧 3–5s）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "glob 模式（带型号/主题子串，如 *H5U*.md）",
            },
            "patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "多个 glob 取并集（可选）",
                "default": [],
            },
            "path": {"type": "string", "description": "子目录（相对授权根，默认 '.'）", "default": "."},
        },
        "required": ["pattern"],
    }
    best_for = (
        "刚开始探索时摸清目录结构（例：**/*.md 列出所有手册）",
        "已知文件命名规则时快速定位文件",
    )
    avoid_for = (
        "需要搜文件内容（用 grep）",
        "需按大小/时间筛选（用 find）",
    )

    def __init__(self, cpu_executor: Any | None = None, max_files: int = 200) -> None:
        super().__init__(cpu_executor)
        self._max_files = max_files

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        jail = PathJail(ctx.roots, tenant_id=ctx.tenant_id)
        pattern = str(args.get("pattern", "")).strip()
        extra_patterns = args.get("patterns") or []
        if isinstance(extra_patterns, str):
            extra_patterns = [extra_patterns]
        all_patterns = [p for p in [pattern, *[str(x).strip() for x in extra_patterns]] if p]
        if not all_patterns:
            return ToolResult(content="缺少参数 pattern", is_error=True)
        # ★ 拦下无信息量的探路 glob（例如单一个 `**/*.md` 且没传其他子串），
        #   给模型一次重新机会而不是白烧预算。
        _trivial = {"**/*.md", "**/*", "**", "*.md", "*"}
        if len(all_patterns) == 1 and all_patterns[0] in _trivial:
            return ToolResult(
                content=(
                    f"❌ glob(pattern={all_patterns[0]!r}) 过于宽泛，5000 本手册场景下会"
                    "白返 200 个文件名 + 白烧一轮。请改为：\n"
                    "  1) 直接调 search_kb(query=<主题词>, doc_hints=[<型号 glob>])，"
                    "一次同时完成“定位手册 + 拉相关片段”；\n"
                    "  2) 或传具体子串：glob(pattern='*H5U*.md') / "
                    "glob(pattern='x', patterns=['*A*.md','*B*.md'])。\n"
                    "仅当知识库确实需从零探目录时才允许 **/*.md（本次已拒绝）。"
                ),
                is_error=True,
            )
        try:
            bases = jail.search_bases(args.get("path", "."))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"访问被拒绝：{exc}", is_error=True)
        return await self._offload(
            lambda: self._glob_sync(
                jail, bases, all_patterns,
                allow=getattr(ctx, "restrict_to_sources", None),
            ),
        )

    def _glob_sync(
        self, jail: PathJail, bases: list[Path], patterns: list[str],
        allow: Any | None = None,
    ) -> ToolResult:
        found: list[str] = []
        seen: set[str] = set()
        truncated = False
        for pattern in patterns:
            for base in bases:
                try:
                    it = [base] if base.is_file() else base.glob(pattern)
                except (OSError, ValueError, NotImplementedError) as exc:
                    return ToolResult(content=f"glob 失败：{exc}", is_error=True)
                for p in it:
                    if not p.is_file():
                        continue
                    # ★ fanout 任务作用域软过滤：范围外文件不列入结果，
                    #   避免引导子 Agent 去 read_file 一本不属于它的手册。
                    if allow and p.name not in allow:
                        continue
                    try:
                        real = Path(os.path.realpath(str(p)))
                    except OSError:
                        continue
                    key = str(real)
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(jail.relative(real))
                    if len(found) >= self._max_files:
                        truncated = True
                        break
                if truncated:
                    break
            if truncated:
                break
        if not found:
            return ToolResult(content=f"无文件匹配模式 {patterns}", meta={"count": 0})
        found.sort()
        text = "\n".join(found)
        if truncated:
            text += f"\n（已截断，仅显示前 {self._max_files} 个）"
        return ToolResult(content=text, meta={"count": len(found), "truncated": truncated})


_SIZE_UNITS = {
    "k": 1024, "kb": 1024, "ki": 1024,
    "m": 1024 ** 2, "mb": 1024 ** 2, "mi": 1024 ** 2,
    "g": 1024 ** 3, "gb": 1024 ** 3, "gi": 1024 ** 3,
    "t": 1024 ** 4, "tb": 1024 ** 4,
}


def _parse_size(value: Any) -> int | None:
    """把 "100k"/"2M"/"500"/字节整数 解析为字节数；空值返回 None。"""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return max(0, int(value))
    s = str(value).strip().lower().replace(" ", "")
    if not s:
        return None
    unit = 1
    for suf in ("tib", "gib", "mib", "kib", "tb", "gb", "mb", "kb", "b", "t", "g", "m", "k"):
        if s.endswith(suf):
            num = s[: -len(suf)] or s
            unit = 1 if suf == "b" else _SIZE_UNITS.get(suf.rstrip("b"), 1)
            s = num
            break
    try:
        return max(0, int(float(s) * unit))
    except ValueError:
        return None


def _fmt_size(nbytes: int) -> str:
    """字节转可读大小（用于展示）。"""
    step = float(nbytes)
    for unit in ("B", "K", "M", "G"):
        if step < 1024 or unit == "G":
            return f"{step:.0f}{unit}" if unit == "B" else f"{step:.1f}{unit}"
        step /= 1024
    return f"{step:.1f}G"  # pragma: no cover


class FindTool(Tool):
    """按名称/大小/修改时间/类型定位文件（对标 Unix ``find``）。"""

    name = "find"
    description = (
        "按文件名/大小/时间等元数据筛选文件（不读内容），比 glob 多大小/时间 维度。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "子目录（相对授权根，默认 '.'）", "default": "."},
            "name": {"type": "string", "description": "文件名 glob如 *.md", "default": "*"},
            "type": {
                "type": "string", "enum": ["file", "dir", "any"],
                "description": "文件/目录/不限", "default": "file",
            },
            "min_size": {"type": "string", "description": "最小字节如 1k/500", "default": ""},
            "max_size": {"type": "string", "description": "最大字节如 100k/2M", "default": ""},
            "modified_within_days": {"type": "integer", "description": "仅保留最近 N 天修改", "default": 0},
            "modified_before_days": {"type": "integer", "description": "仅保留早于 N 天前", "default": 0},
            "sort_by": {
                "type": "string", "enum": ["name", "size", "mtime"],
                "description": "排序键", "default": "name",
            },
            "order": {"type": "string", "enum": ["asc", "desc"], "description": "排序方向", "default": "asc"},
            "max_count": {"type": "integer", "description": "最多返回个数", "default": 50},
            "skip": {"type": "integer", "description": "跳过前 N 个（分页）", "default": 0},
        },
    }
    best_for = (
        "按元数据（大小/修改时间/类型）筛选文件",
        "已知文件命名部分，需模糊定位",
    )
    avoid_for = (
        "需要读文件内容（用 grep 或 read_file）",
    )

    def __init__(self, cpu_executor: Any | None = None, max_files: int = 200) -> None:
        super().__init__(cpu_executor)
        self._max_files = max_files

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        jail = PathJail(ctx.roots, tenant_id=ctx.tenant_id)
        try:
            bases = jail.search_bases(args.get("path", "."))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"访问被拒绝：{exc}", is_error=True)
        return await self._offload(
            lambda: self._find_sync(
                jail, bases, args,
                allow=getattr(ctx, "restrict_to_sources", None),
            ),
        )

    def _find_sync(
        self, jail: PathJail, bases: list[Path], args: dict[str, Any],
        allow: Any | None = None,
    ) -> ToolResult:
        name = str(args.get("name", "*") or "*")
        ftype = str(args.get("type", "file") or "file")
        min_size = _parse_size(args.get("min_size"))
        max_size = _parse_size(args.get("max_size"))
        within = int(args.get("modified_within_days", 0) or 0)
        before = int(args.get("modified_before_days", 0) or 0)
        sort_by = str(args.get("sort_by", "name") or "name")
        desc = str(args.get("order", "asc")).lower() == "desc"
        max_count = int(args.get("max_count", self._max_files) or self._max_files)
        max_count = max(1, min(max_count, 1000))
        skip = max(0, int(args.get("skip", 0) or 0))

        now = time.time()
        matches: list[tuple[str, int, float, bool]] = []  # (rel, size, mtime, is_dir)
        seen: set[str] = set()
        for base in bases:
            if base.is_file():
                walked: list[Path] = [base]
            else:
                walked = []
                for dirpath, dirnames, filenames in os.walk(base):
                    dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                    if ftype in ("dir", "any"):
                        for d in dirnames:
                            dp = Path(dirpath) / d
                            walked.append(dp)
                    if ftype in ("file", "any"):
                        for fn in filenames:
                            walked.append(Path(dirpath) / fn)
            for fp in walked:
                try:
                    real = Path(os.path.realpath(str(fp)))
                    st = real.stat()
                except OSError:
                    continue
                key = str(real)
                if key in seen:
                    continue
                is_dir = real.is_dir()
                if is_dir and ftype == "file":
                    continue
                if not is_dir and ftype == "dir":
                    continue
                if not fnmatch.fnmatch(real.name.lower(), name.lower()):
                    continue
                # ★ fanout 任务作用域软过滤（同 glob）：非目录且不命中范围时跳过。
                if allow and not is_dir and real.name not in allow:
                    continue
                if not is_dir:
                    if min_size is not None and st.st_size < min_size:
                        continue
                    if max_size is not None and st.st_size > max_size:
                        continue
                if within and (now - st.st_mtime) > within * 86400:
                    continue
                if before and (now - st.st_mtime) < before * 86400:
                    continue
                seen.add(key)
                matches.append((jail.relative(real), st.st_size, st.st_mtime, is_dir))

        total = len(matches)
        if not matches:
            return ToolResult(
                content=f"无文件符合条件（name={name!r} type={ftype!r}）",
                meta={"count": 0, "total_matches": 0},
            )
        key_fn = {"size": lambda m: (m[1], m[0]), "mtime": lambda m: (m[2], m[0])}.get(
            sort_by, lambda m: m[0].lower(),
        )
        matches.sort(key=key_fn, reverse=desc)
        page = matches[skip:skip + max_count]
        has_more = skip + max_count < total

        lines: list[str] = []
        for rel, size, mtime, is_dir in page:
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            if is_dir:
                lines.append(f"{rel}/  ({stamp})")
            else:
                lines.append(f"{rel}  ({_fmt_size(size)}, {stamp})")
        text = "\n".join(lines)
        meta: dict[str, Any] = {
            "count": len(page), "total_matches": total, "truncated": has_more,
            "has_more": has_more,
        }
        if has_more:
            meta["next_cursor"] = skip + len(page)
            text += f"\n（共 {total} 个，本页 {len(page)} 个；用 skip={skip + len(page)} 续读）"
        return ToolResult(content=text, meta=meta)


class ListDirTool(Tool):
    """列出目录一级条目（对标 ls / list_dir）。"""

    name = "list_dir"
    description = "列出目录一级条目（目录以/结尾，文件附字节大小）。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目录相对授权根，默认 '.'", "default": "."},
        },
    }
    best_for = (
        "逐步探索未知目录层级",
        "确定目录内具体文件列表",
    )
    avoid_for = (
        "已知目录时无需先列表（直接 grep/read_file）",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        jail = PathJail(ctx.roots, tenant_id=ctx.tenant_id)
        try:
            bases = jail.search_bases(args.get("path", "."))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"访问被拒绝：{exc}", is_error=True)
        return await self._offload(lambda: self._list_sync(jail, bases))

    def _list_sync(self, jail: PathJail, bases: list[Path]) -> ToolResult:
        lines: list[str] = []
        for base in bases:
            if base.is_file():
                lines.append(f"{jail.relative(base)}  ({base.stat().st_size} bytes)")
                continue
            try:
                entries = sorted(base.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            except OSError as exc:
                return ToolResult(content=f"无法列出目录：{exc}", is_error=True)
            lines.append(f"[{jail.relative(base) or '.'}]")
            for e in entries:
                if e.is_dir():
                    lines.append(f"  {e.name}/")
                else:
                    try:
                        size = e.stat().st_size
                    except OSError:
                        size = 0
                    lines.append(f"  {e.name}  ({size} bytes)")
            if len(lines) > 500:
                lines.append("（条目过多，已截断）")
                break
        if not lines:
            return ToolResult(content="目录为空或无授权根", meta={"entries": 0})
        return ToolResult(content="\n".join(lines), meta={"entries": len(lines)})


class ReadFileTool(Tool):
    """分页读取文件内容（行号 + offset/limit + 字符预算截断到整行）。"""

    name = "read_file"
    description = (
        "精读已定位区域的上下文：按行号范围读取文件内容（用于 grep/search 命中后查看完整段落）。"
        "★小范围多次读（limit=20–40），单次上限 3000 字。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径（相对授权根）"},
            "offset": {"type": "integer", "description": "起始行号（1基）", "default": 1},
            "limit": {"type": "integer", "description": "最多行数（建议 20–40）", "default": 40},
        },
        "required": ["path"],
    }
    best_for = (
        "已知行号时精读完整上下文（grep 命中后的补充阅读）",
        "需完整句子/段落才能回答的问题",
    )
    avoid_for = (
        "不确定目标文件时先读（应先 search_kb 或 grep 定位）",
    )

    def __init__(self, cpu_executor: Any | None = None, default_limit: int = 40) -> None:
        super().__init__(cpu_executor)
        self._default_limit = default_limit

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        jail = PathJail(ctx.roots, tenant_id=ctx.tenant_id)
        raw = str(args.get("path", "")).strip()
        if not raw:
            return ToolResult(content="缺少参数 path", is_error=True)
        offset = max(1, int(args.get("offset", 1) or 1))
        limit = int(args.get("limit", self._default_limit) or self._default_limit)
        # ★ 硬上限 2000→300（单次字符预算 6000 行不致戒太多，且防模型传 limit=1000 吐巨包）
        limit = max(1, min(limit, 300))
        try:
            resolved = jail.resolve_first_existing(raw)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"访问被拒绝：{exc}", is_error=True)
        # ★ fanout 任务作用域软过滤（非安全边界）：不匹配时不报错、不抛异常，
        #   返回可读提示让模型自己换目标，不污染 PathJail 语义。
        if _scope_denied(resolved, ctx):
            return ToolResult(content=_SCOPE_DENY_MSG, is_error=False)
        return await self._offload(
            lambda: self._read_sync(jail, resolved, raw, offset, limit),
        )

    def _read_sync(
        self, jail: PathJail, resolved: Path, raw: str, offset: int, limit: int,
    ) -> ToolResult:
        if not resolved.is_file():
            return ToolResult(content=f"文件不存在或不是普通文件：{raw}", is_error=True)
        if not _is_probably_text(resolved):
            return ToolResult(content=f"跳过二进制文件：{raw}", is_error=True)
        try:
            if resolved.stat().st_size > _MAX_FILE_BYTES:
                return ToolResult(
                    content=f"文件过大（>{_MAX_FILE_BYTES} bytes），请用 grep 精确定位后小范围读取",
                    is_error=True,
                )
            with resolved.open("r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.read().splitlines()
        except OSError as exc:
            return ToolResult(content=f"读取失败：{exc}", is_error=True)

        total = len(lines)
        start = min(offset - 1, total)
        end = min(start + limit, total)
        window = lines[start:end]

        # 行号 + 字符预算截断到最后一整行（对齐 hermes _truncate_to_char_budget）
        rendered: list[str] = []
        budget = _MAX_READ_CHARS
        used = 0
        cut = False
        for i, line in enumerate(window, start=start + 1):
            row = f"{i:6d} | {line}"
            if used + len(row) + 1 > budget:
                cut = True
                break
            rendered.append(row)
            used += len(row) + 1

        header = f"# {jail.relative(resolved)}  行 {start + 1}-{start + len(rendered)}/{total}"
        body = "\n".join(rendered) if rendered else "（该范围无内容）"
        note = "\n（已达字符预算，可增大 offset 续读）" if (cut or end < total) else ""
        return ToolResult(
            content=f"{header}\n{body}{note}",
            meta={
                "path": jail.relative(resolved),
                "from_line": start + 1,
                "to_line": start + len(rendered),
                "total_lines": total,
                "truncated": cut or end < total,
            },
        )


def _walk_text_files(base: Path, glob: str, seen: set[str]) -> list[Path]:
    """遍历 base 下匹配 glob 的文本文件（跳过隐藏目录与越界软链）。"""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if glob != "*" and not fnmatch.fnmatch(name, glob):
                continue
            fp = Path(dirpath) / name
            try:
                real = Path(os.path.realpath(str(fp)))
            except OSError:
                continue
            if not real.is_file() or str(real) in seen:
                continue
            if real.stat().st_size > _MAX_FILE_BYTES:
                continue
            seen.add(str(real))
            out.append(real)
    return out
