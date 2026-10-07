"""SearchKBTool.doc_hints 归一化与圈定/回退语义测试。

覆盖路子 B 的两项整改：
1. hint 归一化：去扩展名对齐 title(=p.stem)、无通配符时子串化，规避 fnmatch 整串前缀语义；
2. 圈定失败不再静默回退全库，而是如实告知（全量回退 / 部分 hint 未匹配）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from conftest import run

from fusion_rag.retrieval.base import match_metadata_filter
from fusion_rag.tools.base import ToolContext
from fusion_rag.tools.search_tool import SearchKBTool

# 两本手册：title = 文件名去 .md（对齐 loader 的 p.stem），source = 含 .md 的路径
_CORPUS = [
    ("19011157-H5U&Easy系列编程手册", "datas/md/19011157-H5U&Easy系列编程手册.md"),
    ("20000001-独立型号手册", "datas/md/20000001-独立型号手册.md"),
]


def _mk_chunk(title: str, source: str, idx: int, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        document_id=source,
        chunk_index=idx,
        score=score,
        content=f"{title} 正文片段 {idx}",
        title=title,
        source=source,
        metadata={"source": source},
    )


class _FakeRetriever:
    """按 metadata_filter 在固定语料上做真实 match_metadata_filter 过滤。"""

    async def retrieve(
        self, query: str, *, top_k: int = 10, tenant_id: str | None = None,
        kb_ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SimpleNamespace]:
        out: list[SimpleNamespace] = []
        for rank, (title, source) in enumerate(_CORPUS):
            meta = {"title": title, "source": source}
            if not match_metadata_filter(meta, metadata_filter):
                continue
            out.append(_mk_chunk(title, source, rank, 0.9 - rank * 0.1))
        return out[:top_k]


def _ctx() -> ToolContext:
    return ToolContext(tenant_id="default", kb_ids=["general"], roots=[])


def _tool() -> SearchKBTool:
    return SearchKBTool(_FakeRetriever(), default_top_k=6)


# ----------------------------------------------------------------------
# 归一化：_normalize_hint
# ----------------------------------------------------------------------
def test_normalize_hint_becomes_substring():
    # 裸子串 → 双侧 * 子串匹配，规避 "H5U*" 只匹配前缀的坑
    assert SearchKBTool._normalize_hint("H5U") == ("*H5U*", "*H5U*")


def test_normalize_hint_strips_ext_for_title():
    # glob 回包文件名带 .md：title 侧去扩展名（对齐 p.stem），source 侧保留
    t, s = SearchKBTool._normalize_hint("19011157-编程手册.md")
    assert t == "*19011157-编程手册*"
    assert s == "*19011157-编程手册.md*"


def test_normalize_hint_widens_prefix_glob():
    # "Easy320*" 在本库会撞上前缀陷阱（title 以文档号开头）
    # → 强制双侧通配为 "*Easy320*"，回到子串语义
    t, s = SearchKBTool._normalize_hint("Easy320*")
    assert t == "*Easy320*"
    assert s == "*Easy320*"


def test_normalize_hint_preserves_full_glob():
    # 已双侧通配的显式 glob 原样保留
    t, s = SearchKBTool._normalize_hint("*H5U*&Easy*")
    assert t == "*H5U*&Easy*"
    assert s == "*H5U*&Easy*"


# ----------------------------------------------------------------------
# 圈定生效：命中的 hint 只返回该手册，且无告警
# ----------------------------------------------------------------------
def test_doc_hints_scopes_to_single_manual():
    res = run(
        _tool().run,
        {"query": "编程", "doc_hints": ["19011157-H5U&Easy系列编程手册.md"]},
        _ctx(),
    )
    assert not res.is_error
    # 只出现被圈定的那本手册，另一本被过滤掉
    assert "19011157-H5U&Easy系列编程手册" in res.content
    assert "20000001-独立型号手册" not in res.content
    # 圈定成功时不应出现回退告警
    assert "⚠" not in res.content


def test_doc_hints_substring_matches_without_ext():
    # 用户口语型号子串（不带 .md、不带通配符）现在也能命中含该子串的手册
    res = run(
        _tool().run,
        {"query": "编程", "doc_hints": ["H5U&Easy系列编程手册"]},
        _ctx(),
    )
    assert "19011157-H5U&Easy系列编程手册" in res.content
    assert "20000001-独立型号手册" not in res.content
    assert "⚠" not in res.content


def test_doc_hints_wildcard_hint_now_scopes():
    # 模型爱写的 "H5U*" 现在被强制子串化，能真正圈定到含 H5U 的手册（不再回退）
    res = run(
        _tool().run,
        {"query": "编程", "doc_hints": ["H5U*"]},
        _ctx(),
    )
    assert "19011157-H5U&Easy系列编程手册" in res.content
    assert "20000001-独立型号手册" not in res.content
    assert "⚠" not in res.content


# ----------------------------------------------------------------------
# 回退如实告知：hint 完全无匹配 → 全库回退 + ⚠ 头
# ----------------------------------------------------------------------
def test_doc_hints_total_mismatch_falls_back_with_notice():
    res = run(
        _tool().run,
        {"query": "编程", "doc_hints": ["NONEXISTENT999"]},
        _ctx(),
    )
    # 回退后仍有命中（全库），但必须带如实告知
    assert "⚠" in res.content
    assert "未匹配到任何手册" in res.content
    # 回退是全库：另一本无关手册也会出现
    assert "20000001-独立型号手册" in res.content


def test_doc_hints_partial_mismatch_reports_unmatched():
    # 一个 hint 命中、一个不命中：picked 非空，但须列出未匹配的 hint
    res = run(
        _tool().run,
        {"query": "编程", "doc_hints": ["H5U&Easy系列编程手册", "ZZZ999"]},
        _ctx(),
    )
    assert "部分 doc_hints 未匹配到手册" in res.content
    assert "ZZZ999" in res.content


# ----------------------------------------------------------------------
# 回归：同一手册的多个 chunk 不得因去重键相同被折叠成 1 条
# （历史 bug：RetrievedChunk 无顶层 chunk_index，旧 getattr(c,'chunk_index',0)
#   恒为 0 → 同文档所有块 key=(doc,0) 重复 → doc_hints 圈定时召回塔缩）
# ----------------------------------------------------------------------
class _NestedChunk:
    """模拟真实 RetrievedChunk：chunk_index 在 .chunk 上，顶层无此属性。"""

    def __init__(self, doc: str, idx: int) -> None:
        self.document_id = doc
        self.chunk = SimpleNamespace(chunk_index=idx, content=f"片段{idx}")
        self.title = "手册A"
        self.source = "手册A.md"
        self.metadata: dict[str, Any] = {}
        self.score = 0.9

    @property
    def content(self) -> str:
        return self.chunk.content


def _fake_multi_chunk_tool() -> SearchKBTool:
    class _Fake:
        async def retrieve(self, query, *, top_k=10, tenant_id=None, kb_ids=None,
                           metadata_filter=None):
            return [_NestedChunk("docA", i) for i in range(3)]

    return SearchKBTool(_Fake(), default_top_k=12)


def test_dedup_key_uses_nested_chunk_index():
    # 无 .key 属性时回退到 c.chunk.chunk_index
    assert SearchKBTool._dedup_key(_NestedChunk("docA", 2)) == ("docA", 2)


def test_dedup_key_prefers_explicit_key_property():
    obj = SimpleNamespace(key=("docB", 5), document_id="docB", chunk_index=0)
    assert SearchKBTool._dedup_key(obj) == ("docB", 5)


def test_doc_hints_no_chunk_collapse_within_one_manual():
    # 同一手册 3 个块（chunk_index 0/1/2）应全部保留，而非折叠成 1
    res = run(_fake_multi_chunk_tool().run,
              {"query": "x", "doc_hints": ["手册A"]}, _ctx())
    assert "共 3 条命中" in res.content
