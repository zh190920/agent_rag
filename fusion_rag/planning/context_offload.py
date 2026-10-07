"""ContextOffloader：证据装配 + 超长上下文卸载（借鉴 DeepAgents 文件系统卸载）。

职责：
1. 把检索到的候选切片在 token 预算内装配成带编号的证据块（供推理引用）。
2. 预算装不下的证据「卸载」到本地 spill 文件，只在上下文保留引用路径，
   避免上下文溢出，同时保持可追溯。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..types import RetrievedChunk, approx_token_count

logger = get_logger(__name__)


@dataclass
class EvidenceBlock:
    """装配好的证据块。"""

    text: str                                   # 注入提示词的编号证据文本
    used: list[RetrievedChunk] = field(default_factory=list)      # 实际进入上下文的切片
    spilled: list[RetrievedChunk] = field(default_factory=list)   # 被卸载的切片
    spill_path: str = ""                        # 卸载文件路径（可空）
    tokens: int = 0

    def citation_map(self) -> dict[int, RetrievedChunk]:
        """编号 → 切片，供推理层生成引用与校验层核对。"""
        return {i + 1: chunk for i, chunk in enumerate(self.used)}


class ContextOffloader:
    """上下文预算裁剪与卸载器。"""

    def __init__(
        self,
        spill_dir: str | Path | None = None,
        *,
        per_chunk_cap: int = 600,
        enable_spill: bool = True,
    ) -> None:
        self._spill_dir = Path(spill_dir).expanduser() if spill_dir else None
        if self._spill_dir is not None:
            self._spill_dir.mkdir(parents=True, exist_ok=True)
        self.per_chunk_cap = per_chunk_cap
        self.enable_spill = enable_spill

    def build_evidence(
        self,
        chunks: list[RetrievedChunk],
        *,
        budget: int,
        query: str = "",
    ) -> EvidenceBlock:
        """在 ``budget`` token 内装配编号证据；超出的卸载到磁盘。"""
        if not chunks:
            return EvidenceBlock(text="（未检索到相关知识）", tokens=0)

        used: list[RetrievedChunk] = []
        spilled: list[RetrievedChunk] = []
        lines: list[str] = []
        tokens = 0
        header_cost = approx_token_count(query) + 16 if query else 0
        tokens += header_cost

        for idx, chunk in enumerate(chunks):
            body = chunk.content
            if approx_token_count(body) > self.per_chunk_cap:
                body = self._truncate_to_tokens(body, self.per_chunk_cap)
            snippet = f"[{idx + 1}] (来源: {chunk.title or chunk.source or chunk.document_id[:8]})\n{body}"
            cost = approx_token_count(snippet) + 4
            if tokens + cost > budget and used:
                spilled.extend(chunks[idx:])
                break
            lines.append(snippet)
            used.append(chunk)
            tokens += cost

        text = "\n\n".join(lines)
        if query:
            text = f"（围绕问题：{query}）\n\n{text}"

        spill_path = ""
        if spilled and self.enable_spill and self._spill_dir is not None:
            spill_path = self._spill(spilled, query)

        return EvidenceBlock(
            text=text, used=used, spilled=spilled, spill_path=spill_path, tokens=tokens,
        )

    # ------------------------------------------------------------------
    def _spill(self, chunks: list[RetrievedChunk], query: str) -> str:
        """把溢出证据写入本地文件，返回路径（上下文只保留引用）。"""
        try:
            ts = int(time.time() * 1000)
            path = self._spill_dir / f"evidence_{ts}.jsonl"  # type: ignore[union-attr]
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"query": query, "spilled": len(chunks)},
                                    ensure_ascii=False) + "\n")
                for chunk in chunks:
                    fh.write(json.dumps({
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk.chunk_index,
                        "title": chunk.title,
                        "source": chunk.source,
                        "score": chunk.score,
                        "content": chunk.content,
                    }, ensure_ascii=False) + "\n")
            logger.debug("卸载 %d 条证据到 %s", len(chunks), path)
            return str(path)
        except OSError as exc:
            logger.warning("证据卸载失败（忽略）：%s", exc)
            return ""

    @staticmethod
    def _truncate_to_tokens(text: str, cap: int) -> str:
        """按近似 token 上限截断（中文≈1字1token，故用字符数近似）。"""
        if approx_token_count(text) <= cap:
            return text
        # 粗略二分逼近
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if approx_token_count(text[:mid]) <= cap:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + "…"

    @staticmethod
    def estimate(text: str) -> int:
        return approx_token_count(text)

    def budget_for(self, total: int, reserved: dict[str, Any]) -> int:
        """从总预算扣除已占用，得到证据可用预算。"""
        used = sum(int(v) for v in reserved.values())
        return max(256, total - used)
