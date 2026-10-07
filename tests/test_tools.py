"""PathJail 与只读文件工具测试：jail 拦截 + grep/glob/list_dir/read_file。"""

from __future__ import annotations

import os
from pathlib import Path

from conftest import run

from fusion_rag.core.exceptions import AccessDeniedError
from fusion_rag.tools.base import ToolContext
from fusion_rag.tools.file_tools import (
    FindTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
)
from fusion_rag.tools.path_jail import PathJail

DOC = (
    "# AgentScope 概览\n"
    "AgentScope 支持多智能体分布式编排与高并发问答。\n"
    "它提供可视化会话管理，适配多角色协作场景。\n"
    "DeepAgents 负责复杂问题分层拆解。\n"
)


def _make_corpus(tmp_root: Path) -> None:
    (tmp_root / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_root / "docs" / "agentscope.md").write_text(DOC, encoding="utf-8")
    (tmp_root / "notes.txt").write_text("OpenClaw 强调本地化隐私可控。", encoding="utf-8")


def _ctx(tmp_root: Path) -> ToolContext:
    return ToolContext(tenant_id="default", kb_ids=["general"], roots=[str(tmp_root)])


# ----------------------------------------------------------------------
# PathJail
# ----------------------------------------------------------------------
def test_jail_resolves_within_root(tmp_path_factory=None):
    import tempfile

    root = Path(tempfile.mkdtemp())
    _make_corpus(root)
    jail = PathJail([root])
    resolved = jail.resolve("docs/agentscope.md")
    assert resolved.is_file()
    assert str(resolved).startswith(str(Path(os.path.realpath(str(root)))))


def test_jail_blocks_traversal():
    import tempfile

    root = Path(tempfile.mkdtemp())
    jail = PathJail([root])
    import pytest

    with pytest.raises(AccessDeniedError):
        jail.resolve("../../etc/passwd")


def test_jail_blocks_absolute_escape():
    import tempfile

    root = Path(tempfile.mkdtemp())
    jail = PathJail([root])
    outside = Path(tempfile.mkdtemp()) / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    import pytest

    with pytest.raises(AccessDeniedError):
        jail.resolve(str(outside))


def test_jail_blocks_when_no_roots():
    import pytest

    jail = PathJail([])
    with pytest.raises(AccessDeniedError):
        jail.resolve("anything")


def test_jail_blocks_device_path():
    import pytest

    jail = PathJail(["/srv/kb"])
    with pytest.raises(AccessDeniedError):
        jail.resolve("/dev/null")


# ----------------------------------------------------------------------
# GrepTool
# ----------------------------------------------------------------------
def test_grep_finds_content():
    import tempfile

    root = Path(tempfile.mkdtemp())
    _make_corpus(root)
    tool = GrepTool()

    async def scenario():
        return await tool.run({"pattern": "高并发", "regex": False}, _ctx(root))

    result = run(scenario)
    assert not result.is_error
    assert "agentscope.md" in result.content
    assert ":2:" in result.content  # 命中在第 2 行


def test_grep_regex_and_glob():
    import tempfile

    root = Path(tempfile.mkdtemp())
    _make_corpus(root)
    tool = GrepTool()

    async def scenario():
        return await tool.run({"pattern": "拆解|编排", "glob": "*.md"}, _ctx(root))

    result = run(scenario)
    assert not result.is_error
    assert result.meta["matches"] >= 2


def test_grep_no_match():
    import tempfile

    root = Path(tempfile.mkdtemp())
    _make_corpus(root)
    tool = GrepTool()

    async def scenario():
        return await tool.run({"pattern": "量子计算"}, _ctx(root))

    result = run(scenario)
    assert result.is_error is False
    assert result.meta["matches"] == 0


def test_grep_blocks_outside_jail():
    import tempfile

    root = Path(tempfile.mkdtemp())
    _make_corpus(root)
    tool = GrepTool()

    async def scenario():
        return await tool.run({"pattern": "root", "path": "../../etc"}, _ctx(root))

    result = run(scenario)
    assert result.is_error and "拒绝" in result.content


# ----------------------------------------------------------------------
# GlobTool / ListDirTool
# ----------------------------------------------------------------------
def test_glob_locates_files():
    import tempfile

    root = Path(tempfile.mkdtemp())
    _make_corpus(root)
    tool = GlobTool()

    async def scenario():
        return await tool.run({"pattern": "docs/*.md"}, _ctx(root))

    result = run(scenario)
    assert not result.is_error
    assert "agentscope.md" in result.content


def test_list_dir_structure():
    import tempfile

    root = Path(tempfile.mkdtemp())
    _make_corpus(root)
    tool = ListDirTool()

    async def scenario():
        return await tool.run({"path": "."}, _ctx(root))

    result = run(scenario)
    assert not result.is_error
    assert "docs/" in result.content      # 目录带斜杠
    assert "notes.txt" in result.content  # 文件


# ----------------------------------------------------------------------
# ReadFileTool
# ----------------------------------------------------------------------
def test_read_file_line_numbers_and_pagination():
    import tempfile

    root = Path(tempfile.mkdtemp())
    _make_corpus(root)
    tool = ReadFileTool()

    async def scenario():
        return await tool.run(
            {"path": "docs/agentscope.md", "offset": 2, "limit": 2}, _ctx(root),
        )

    result = run(scenario)
    assert not result.is_error
    assert "     2 |" in result.content
    assert "     3 |" in result.content
    assert "     4 |" not in result.content  # limit=2 到第 3 行为止
    assert result.meta["total_lines"] == 4


def test_read_file_truncates_to_whole_line():
    import tempfile

    root = Path(tempfile.mkdtemp())
    # 一个超长行文件，触发字符预算截断
    big = "x" * 50000 + "\n第二行内容\n"
    (root / "big.txt").write_text(big, encoding="utf-8")
    tool = ReadFileTool()

    async def scenario():
        return await tool.run({"path": "big.txt"}, _ctx(root))

    result = run(scenario)
    assert not result.is_error
    assert result.meta["truncated"] is True
    assert "第二行内容" not in result.content  # 超预算的后续行被丢弃


def test_read_missing_and_binary():
    import tempfile

    root = Path(tempfile.mkdtemp())
    (root / "bin.dat").write_bytes(b"\x00\x01\x02binary")
    tool = ReadFileTool()

    async def scenario_missing():
        return await tool.run({"path": "nope.md"}, _ctx(root))

    async def scenario_binary():
        return await tool.run({"path": "bin.dat"}, _ctx(root))

    assert run(scenario_missing).is_error
    assert run(scenario_binary).is_error


# ----------------------------------------------------------------------
# FindTool
# ----------------------------------------------------------------------
def test_find_by_name():
    import tempfile

    root = Path(tempfile.mkdtemp())
    _make_corpus(root)
    tool = FindTool()

    async def scenario():
        return await tool.run({"name": "*.md"}, _ctx(root))

    result = run(scenario)
    assert not result.is_error
    assert "docs/agentscope.md" in result.content
    assert "notes.txt" not in result.content


def test_find_by_size_and_sort():
    import tempfile

    root = Path(tempfile.mkdtemp())
    (root / "small.txt").write_text("x", encoding="utf-8")
    (root / "big.txt").write_text("y" * 5000, encoding="utf-8")
    tool = FindTool()

    async def scenario():
        return await tool.run(
            {"name": "*.txt", "min_size": "1k", "sort_by": "size", "order": "desc"},
            _ctx(root),
        )

    result = run(scenario)
    assert not result.is_error
    assert "big.txt" in result.content
    assert "small.txt" not in result.content  # 小于 1k 被过滤


def test_find_type_dir():
    import tempfile

    root = Path(tempfile.mkdtemp())
    _make_corpus(root)
    tool = FindTool()

    async def scenario():
        return await tool.run({"type": "dir", "name": "*"}, _ctx(root))

    result = run(scenario)
    assert not result.is_error
    assert "docs/" in result.content


def test_find_blocks_outside_jail():
    import tempfile

    root = Path(tempfile.mkdtemp())
    tool = FindTool()

    async def scenario():
        return await tool.run({"path": "../../etc"}, _ctx(root))

    result = run(scenario)
    assert result.is_error and "拒绝" in result.content


# ----------------------------------------------------------------------
# grep 上下文行
# ----------------------------------------------------------------------
def test_grep_context_lines():
    import tempfile

    root = Path(tempfile.mkdtemp())
    _make_corpus(root)
    tool = GrepTool()

    async def scenario():
        return await tool.run(
            {"pattern": "高并发", "regex": False, "context": 1}, _ctx(root),
        )

    result = run(scenario)
    assert not result.is_error
    assert "agentscope.md:2:" in result.content     # 命中行（冒号）
    assert "agentscope.md-1-" in result.content      # 上一行（连字符）
    assert "agentscope.md-3-" in result.content      # 下一行


# ----------------------------------------------------------------------
# grep 分页游标
# ----------------------------------------------------------------------
def test_grep_pagination_cursor():
    import tempfile

    root = Path(tempfile.mkdtemp())
    body = "".join(f"line{i} hit\n" for i in range(1, 11))  # 10 条命中
    (root / "many.txt").write_text(body, encoding="utf-8")
    tool = GrepTool()

    async def page1():
        return await tool.run(
            {"pattern": "hit", "regex": False, "max_count": 3}, _ctx(root),
        )

    async def page2():
        return await tool.run(
            {"pattern": "hit", "regex": False, "max_count": 3, "skip": 3}, _ctx(root),
        )

    r1 = run(page1)
    assert r1.meta["matches"] == 3
    assert r1.meta["has_more"] is True
    assert r1.meta["next_cursor"] == 3
    assert "line1 hit" in r1.content and "line4 hit" not in r1.content

    r2 = run(page2)
    assert "line4 hit" in r2.content and "line1 hit" not in r2.content


# ----------------------------------------------------------------------
# 跨 root 搜索合并
# ----------------------------------------------------------------------
def test_jail_search_bases_merges_across_roots():
    import tempfile

    r1 = Path(tempfile.mkdtemp())
    r2 = Path(tempfile.mkdtemp())
    (r1 / "docs").mkdir()
    (r2 / "docs").mkdir()
    jail = PathJail([r1, r2])
    bases = jail.search_bases("docs")
    assert len(bases) == 2  # 两个 root 下的 docs 都被纳入


def test_grep_spans_multiple_roots():
    import tempfile

    r1 = Path(tempfile.mkdtemp())
    r2 = Path(tempfile.mkdtemp())
    (r1 / "a.md").write_text("shared keyword alpha\n", encoding="utf-8")
    (r2 / "b.md").write_text("shared keyword beta\n", encoding="utf-8")
    tool = GrepTool()
    multi_ctx = ToolContext(tenant_id="t", kb_ids=["general"], roots=[str(r1), str(r2)])

    async def scenario():
        return await tool.run({"pattern": "shared keyword", "regex": False}, multi_ctx)

    result = run(scenario)
    assert not result.is_error
    assert result.meta["matches"] == 2
    assert "a.md" in result.content and "b.md" in result.content
