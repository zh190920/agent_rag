"""分词与分块单元测试。"""

from __future__ import annotations

from fusion_rag.chunking.chunker import Chunker
from fusion_rag.text import tokenize, tokenize_query
from fusion_rag.types import approx_token_count


def test_tokenize_cjk_bigram():
    toks = tokenize("多智能体框架")
    assert toks, "中文应产生 bigram token"
    # bigram 数量 = 字符数 - 1（无空格时）
    assert "多智" in toks
    assert "框架" in toks


def test_tokenize_mixed_and_english():
    toks = tokenize("RAG framework 支持 hybrid 检索")
    joined = " ".join(toks)
    assert "rag" in joined.lower() or "framework" in joined.lower()
    assert any("\u4e00" <= ch <= "\u9fff" for ch in joined)


def test_tokenize_query_nonempty():
    assert tokenize_query("AgentScope 是什么")


def test_approx_token_count():
    assert approx_token_count("") == 0
    # 中文按字计数
    assert approx_token_count("中文字符") == 4
    assert approx_token_count("hello world") >= 1


def test_chunker_respects_max_tokens():
    text = "。".join(f"这是第{i}句关于知识问答的说明内容" for i in range(80))
    chunker = Chunker(max_tokens=120, overlap_tokens=20)
    chunks = chunker.split(text)
    assert len(chunks) > 1, "长文本应被切成多块"
    for c in chunks:
        assert c.content.strip()
        # 允许重叠导致的轻微超限，但不应离谱
        assert approx_token_count(c.content) <= 120 + 40


def test_chunker_empty_text():
    assert Chunker().split("   ") == []


def test_chunker_keeps_chunk_index_order():
    text = "\n\n".join(f"段落{i}。内容内容内容。" for i in range(30))
    chunks = Chunker(max_tokens=80, overlap_tokens=0).split(text)
    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(indices)
    assert indices[0] == 0
