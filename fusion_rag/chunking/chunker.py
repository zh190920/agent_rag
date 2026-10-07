"""近似 token 分块器（借鉴 agentscope ApproxTokenChunker + 结构感知）。

策略：
1. 先按结构切分为「语义单元」——标题行、段落、列表项，尽量保留
   Markdown 标题路径作为上下文前缀。
2. 贪心合并语义单元，直到逼近 ``max_tokens``；超长单元再按句子/字符
   二次切分。
3. 相邻块保留 ``overlap_tokens`` 重叠，缓解边界信息割裂。
"""

from __future__ import annotations

import abc
import re
from typing import Any

from ..types import Chunk, approx_token_count

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?\.\n])")


class ChunkerBase(abc.ABC):
    """分块器接口。"""

    @abc.abstractmethod
    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """把整篇文档切分为 Chunk 列表。"""


class Chunker(ChunkerBase):
    """结构感知的近似 token 分块器。"""

    def __init__(
        self,
        max_tokens: int = 400,
        overlap_tokens: int = 60,
        keep_heading_context: bool = True,
    ) -> None:
        self.max_tokens = max(64, int(max_tokens))
        self.overlap_tokens = max(0, min(int(overlap_tokens), self.max_tokens // 2))
        self.keep_heading_context = keep_heading_context

    # ------------------------------------------------------------------
    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        if not text or not text.strip():
            return []

        units = self._to_units(text)
        merged = self._merge_units(units)
        chunks: list[Chunk] = []
        for index, (content, heading_path, start_line) in enumerate(merged):
            body = content.strip()
            if not body:
                continue
            if self.keep_heading_context and heading_path:
                prefix = " > ".join(heading_path)
                # 避免重复拼接已在正文里的标题
                if not body.startswith(prefix):
                    body = f"[{prefix}]\n{body}"
            chunk_meta = dict(metadata)
            if heading_path:
                chunk_meta["heading_path"] = " > ".join(heading_path)
            # ★ 记录块在原始文件中的起始行号（1-based），供 search_kb 回包时
            #   直接拼 `basename.md:line` 格式给 ToolAgent 收集 file_citations。
            if start_line and start_line > 0:
                chunk_meta["start_line"] = int(start_line)
            chunks.append(Chunk(content=body, chunk_index=index, metadata=chunk_meta))
        # 重排 chunk_index 保证连续
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
        return chunks

    # ------------------------------------------------------------------
    def _to_units(self, text: str) -> list[tuple[str, list[str], int]]:
        """切分为 (语义单元文本, 当前标题路径, 单元起始行号) 列表。

        行号为 1-based；若单块合并多个单元则取首个单元的起始行作为块定位。
        """
        units: list[tuple[str, list[str], int]] = []
        heading_stack: list[str] = []  # [(level implied by order)]
        levels: list[int] = []
        buf: list[str] = []
        buf_start = 0  # 1-based 行号：当前 buf 首个非空行所在行

        def flush() -> None:
            nonlocal buf_start
            content = "\n".join(buf).strip()
            if content:
                units.append((content, list(heading_stack), buf_start))
            buf.clear()
            buf_start = 0

        for idx, line in enumerate(text.splitlines(), start=1):
            match = _HEADING.match(line.strip())
            if match:
                flush()
                level = len(match.group(1))
                title = match.group(2).strip()
                # 维护标题层级栈
                while levels and levels[-1] >= level:
                    levels.pop()
                    heading_stack.pop()
                levels.append(level)
                heading_stack.append(title)
                buf_start = idx
                buf.append(line)
            elif not line.strip():
                flush()
            else:
                if not buf:
                    buf_start = idx
                buf.append(line)
        flush()

        # 无标题的纯文本：units 可能很大，交给 _merge_units 二次切分
        return units

    def _merge_units(
        self, units: list[tuple[str, list[str], int]],
    ) -> list[tuple[str, list[str], int]]:
        """贪心合并到 token 上限；超长单元按句子二次切分。

        返回项携带起始行号（取当前合并单元链中首个 unit 的 start_line）。
        """
        result: list[tuple[str, list[str], int]] = []
        cur_text: list[str] = []
        cur_tokens = 0
        cur_headings: list[str] = []
        cur_start_line = 0

        def flush_cur() -> None:
            nonlocal cur_text, cur_tokens, cur_start_line
            if cur_text:
                result.append((
                    "\n".join(cur_text).strip(), list(cur_headings), cur_start_line,
                ))
                # 重叠：保留尾部若干句作为下一块前缀
                if self.overlap_tokens > 0:
                    overlap = self._tail_overlap("\n".join(cur_text))
                    cur_text = [overlap] if overlap else []
                    cur_tokens = approx_token_count(overlap)
                    # 下一块起点仍使用当前 start_line（重叠尾部不新行，取同一定位）
                else:
                    cur_text = []
                    cur_tokens = 0
                cur_start_line = 0

        for content, headings, start_line in units:
            unit_tokens = approx_token_count(content)
            # 单元本身超长 → 先按句子拆
            if unit_tokens > self.max_tokens:
                flush_cur()
                cur_headings = list(headings)
                for offset, piece in enumerate(self._split_long(content)):
                    # 超长单元内部行号无法精确到块，粗略按已发块数 +1 估算（保留临近即可）
                    result.append((piece, list(headings), start_line + offset if start_line else 0))
                cur_text = []
                cur_tokens = 0
                cur_start_line = 0
                continue

            if cur_tokens + unit_tokens > self.max_tokens and cur_text:
                flush_cur()
                cur_headings = list(headings)
            else:
                cur_headings = list(headings)
            if not cur_text and start_line:
                cur_start_line = start_line
            cur_text.append(content)
            cur_tokens += unit_tokens

        flush_cur()
        # 清理空块
        return [(t, h, sl) for (t, h, sl) in result if t.strip()]

    def _split_long(self, content: str) -> list[str]:
        """超长单元按句子聚合切分到 max_tokens。"""
        sentences = [s for s in _SENT_SPLIT.split(content) if s and s.strip()]
        pieces: list[str] = []
        buf: list[str] = []
        tokens = 0
        for sent in sentences:
            st = approx_token_count(sent)
            if st > self.max_tokens:
                # 单句仍超长 → 硬切
                if buf:
                    pieces.append("".join(buf))
                    buf, tokens = [], 0
                pieces.extend(self._hard_split(sent))
                continue
            if tokens + st > self.max_tokens and buf:
                pieces.append("".join(buf))
                buf, tokens = [], 0
            buf.append(sent)
            tokens += st
        if buf:
            pieces.append("".join(buf))
        return [p.strip() for p in pieces if p.strip()]

    def _hard_split(self, text: str) -> list[str]:
        """按字符硬切（极端长句兜底）。"""
        # 粗略：每 max_tokens*2 字符切一段（中文≈1字1token）
        step = max(self.max_tokens, 128)
        return [text[i:i + step] for i in range(0, len(text), step)] or [text]

    def _tail_overlap(self, text: str) -> str:
        """取文本尾部约 overlap_tokens 的内容作为重叠前缀。"""
        if self.overlap_tokens <= 0 or not text:
            return ""
        sentences = [s for s in _SENT_SPLIT.split(text) if s and s.strip()]
        tail: list[str] = []
        tokens = 0
        for sent in reversed(sentences):
            st = approx_token_count(sent)
            if tokens + st > self.overlap_tokens and tail:
                break
            tail.insert(0, sent)
            tokens += st
        return "".join(tail).strip()
