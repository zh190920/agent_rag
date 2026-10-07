"""文本处理工具：分词、归一化、CJK bigram。

统一分词器供 BM25 关键词检索与 HashEmbedding 离线向量共用，保证
中文在零第三方依赖下也能被合理切分（借鉴 hermes FTS5 CJK 思路）。
"""

from __future__ import annotations

import re
import unicodedata

_LATIN_TOKEN = re.compile(r"[a-z0-9]+")
_CJK_RANGE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# 极简中英文停用词（可被插件覆盖）
STOPWORDS = frozenset({
    "的", "了", "和", "是", "在", "我", "有", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "会", "着", "没有", "看",
    "好", "自己", "这", "那", "他", "她", "它", "吗", "呢", "吧", "啊",
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and",
    "or", "in", "on", "for", "with", "it", "this", "that", "as", "at", "by",
})


def normalize(text: str) -> str:
    """Unicode NFKC 归一 + 转小写。"""
    return unicodedata.normalize("NFKC", text).lower()


def is_cjk(char: str) -> bool:
    return bool(_CJK_RANGE.match(char))


def tokenize(text: str, *, use_stopwords: bool = True) -> list[str]:
    """混合分词：

    - 拉丁/数字：按连续串切词
    - 中文：unigram + bigram（提升召回，兼顾短语匹配）

    返回 token 列表（保留重复，供词频统计）。
    """
    if not text:
        return []
    text = normalize(text)
    tokens: list[str] = []

    # 拉丁词
    tokens.extend(_LATIN_TOKEN.findall(text))

    # 中文：抽取连续 CJK 段，逐段生成 unigram + bigram
    for run in _cjk_runs(text):
        chars = list(run)
        tokens.extend(chars)
        for i in range(len(chars) - 1):
            tokens.append(chars[i] + chars[i + 1])

    if use_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and not t.isspace()]
    return tokens


def _cjk_runs(text: str) -> list[str]:
    """切出所有连续中文片段。"""
    runs: list[str] = []
    buf: list[str] = []
    for ch in text:
        if is_cjk(ch):
            buf.append(ch)
        elif buf:
            runs.append("".join(buf))
            buf = []
    if buf:
        runs.append("".join(buf))
    return runs


def tokenize_query(text: str) -> list[str]:
    """查询侧分词：只保留 **拉丁词 + 中文 bigram 及以上**，弃 unigram。

    单字（如“点”“短”“故”）在数千本手册的 BM25 索引里 tf 巨大、idf 极低，
    作为查询词命中噪声文档、把排名带偏。索引侧仍保 unigram 以维持
    召回（防止多字链写错时一个都匹配不上），但查询侧不再拿它做主排序依据。
    """
    seen: set[str] = set()
    result: list[str] = []
    for tok in tokenize(text):
        # 中文单字跳过（长度 1 且属 CJK 段）；拉丁/数字单字符仍保留
        if len(tok) == 1 and _CJK_RANGE.match(tok) is not None:
            continue
        if tok not in seen:
            seen.add(tok)
            result.append(tok)
    return result
