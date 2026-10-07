"""重排器（Reranker）：混合召回后的精排。

- :class:`CrossEncoderReranker` —— 调用专用 Reranker API（如 bge-reranker-v2-m3）
  对候选切片与问题计算相关性分数，快速且精准。
- :class:`LLMReranker` —— 用大模型对候选与问题的相关性打分排序（高精度，
  成本较高），失败自动降级为原顺序。
- :class:`HeuristicReranker` —— 零成本兜底：按 token 覆盖率 + 位置 + 原分
  数加权，无网络也可用。
"""

from __future__ import annotations

import abc
import asyncio
import json
from typing import Any

from ..core.logging import get_logger
from ..llm.base import LLMBase, Message
from ..text import tokenize_query
from ..types import RetrievedChunk

logger = get_logger(__name__)

try:
    import aiohttp  # type: ignore
    _HAS_AIOHTTP = True
except ImportError:
    aiohttp = None  # type: ignore
    _HAS_AIOHTTP = False


def _head_tail(text: str, n: int) -> str:
    """取前 n 字 + 尾 n 字拼接（总长 ≤ 2n）。中间已含则直接返回全文。"""
    if not text:
        return ""
    if len(text) <= 2 * n:
        return text
    return text[:n] + "\n…\n" + text[-n:]


class RerankerBase(abc.ABC):
    """重排器接口。"""

    name: str = "base"

    @abc.abstractmethod
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int,
    ) -> list[RetrievedChunk]:
        """对候选切片重排并截断到 top_k。"""


class HeuristicReranker(RerankerBase):
    """零依赖启发式重排：token 覆盖率为主，兼顾原始得分与来源多样性。"""

    name = "heuristic"

    def __init__(self, coverage_weight: float = 1.0, score_weight: float = 0.3) -> None:
        self.coverage_weight = coverage_weight
        self.score_weight = score_weight

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        q_tokens = set(tokenize_query(query))
        max_score = max((c.score for c in chunks), default=0.0) or 1.0
        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            if q_tokens:
                c_tokens = set(tokenize_query(chunk.content))
                coverage = len(q_tokens & c_tokens) / len(q_tokens)
            else:
                coverage = 0.0
            norm_score = chunk.score / max_score
            final = self.coverage_weight * coverage + self.score_weight * norm_score
            scored.append((final, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        for final, chunk in scored[:top_k]:
            chunk.score = round(final, 6)
            result.append(chunk)
        return result


class LLMReranker(RerankerBase):
    """大模型精排：让模型输出相关性排序（JSON）。失败降级为启发式。"""

    name = "llm"

    def __init__(
        self,
        llm: LLMBase,
        provider: str | None = None,
        fallback: RerankerBase | None = None,
        max_input: int = 20,
    ) -> None:
        self._llm = llm
        self._provider = provider
        self._fallback = fallback or HeuristicReranker()
        self._max_input = max_input

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int,
    ) -> list[RetrievedChunk]:
        if len(chunks) <= 1:
            return chunks[:top_k]
        candidates = chunks[: self._max_input]
        listing = "\n".join(
            f"[{i}] {c.content[:300]}" for i, c in enumerate(candidates)
        )
        system = (
            "你是检索结果重排器。给定问题与编号候选片段，判断每个片段对回答问题的"
            "相关性，从高到低输出编号顺序。只输出 JSON。__RERANK__\n"
            '格式：{"order": [编号, ...]}'
        )
        user = f"问题：{query}\n\n候选片段：\n{listing}"
        try:
            data = await self._llm.chat_json(
                [Message.system(system), Message.user(user)],
                provider=self._provider,
                temperature=0.0,
                max_tokens=200,
            )
            order = data.get("order") or []
            ordered = [
                candidates[i] for i in order
                if isinstance(i, int) and 0 <= i < len(candidates)
            ]
            # 去重 + 补齐未覆盖的候选（保持原相对顺序）
            seen = {id(c) for c in ordered}
            for c in candidates:
                if id(c) not in seen:
                    ordered.append(c)
            if not ordered:
                return await self._fallback.rerank(query, chunks, top_k)
            return ordered[:top_k]
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 重排失败，降级启发式：%s", exc)
            return await self._fallback.rerank(query, chunks, top_k)


class CrossEncoderReranker(RerankerBase):
    """专用 Cross-Encoder Reranker API（兼容 Jina/Cohere/SiliconFlow /rerank 协议）。

    调用示例：
        POST {base_url}/rerank
        {"model": "BAAI/bge-reranker-v2-m3", "query": "...", "documents": [...], "top_n": N}
    """

    name = "cross_encoder"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "BAAI/bge-reranker-v2-m3",
        timeout: float = 15.0,
        max_input: int = 30,
        fallback: RerankerBase | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_input = max_input
        self._fallback = fallback or HeuristicReranker()
        self._session: Any = None
        # ★ session lazy init 锁（同 openai_compatible）：并发首波 rerank 下
        #   旧写法会双 new ClientSession，其中一个不能 close → 泄露连接池。
        self._session_lock: Any = None  # 延迟到首次使用时建（避在 __init__ 里拿不到 loop）

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if len(chunks) == 1:
            return chunks[:top_k]
        candidates = chunks[: self._max_input]
        # ★ 旧实现只截前 2000 字：PLC 手册里错误码表、参数列表的关键信息
        #   常排在块末尾，前面被无信息量描述占满→rerank 打分低。改成
        #   “前 1000 + 尾 1000”拼接（中间已含则不重复），保证开头主题与
        #   尾部具体编码同时进模型视野。
        documents = [_head_tail(c.content, 1000) for c in candidates]

        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(candidates)),
            "return_documents": False,
        }
        try:
            results = await self._post_rerank(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrossEncoder 重排失败，降级启发式：%s", exc)
            return await self._fallback.rerank(query, chunks, top_k)

        # results: [{"index": 3, "relevance_score": 0.98}, ...]
        ranked: list[RetrievedChunk] = []
        for item in results:
            idx = item.get("index", -1)
            score = item.get("relevance_score", 0.0)
            if 0 <= idx < len(candidates):
                chunk = candidates[idx]
                chunk.score = round(float(score), 6)
                ranked.append(chunk)
        if not ranked:
            return await self._fallback.rerank(query, chunks, top_k)
        return ranked[:top_k]

    async def _post_rerank(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        endpoint = f"{self._base_url}/rerank"
        if _HAS_AIOHTTP:
            return await self._post_aiohttp(endpoint, payload)
        return await self._post_urllib(endpoint, payload)

    async def _post_aiohttp(self, endpoint: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if self._session is None or self._session.closed:
            if self._session_lock is None:
                self._session_lock = asyncio.Lock()
            async with self._session_lock:
                # 内层再判：拿到锁时另一个协程可能已建好
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self._timeout),
                    )
        async with self._session.post(
            endpoint, json=payload, headers=self._headers(),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"rerank HTTP {resp.status}: {text[:200]}")
            data = await resp.json()
            return data.get("results", [])

    async def _post_urllib(self, endpoint: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        import urllib.request

        def _call() -> list[dict[str, Any]]:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body.get("results", [])

        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _call),
            timeout=self._timeout + 5,
        )
        return result

    async def aclose(self) -> None:
        """关闭 aiohttp session，避免 Kernel shutdown 时 "Unclosed client session".。"""
        if self._session is not None and not self._session.closed:
            try:
                await self._session.close()
                # ★ Windows Proactor 下需 sleep 排干 connector，避免 transport.__del__
                #   对已关 loop 调 call_soon 报 Event loop is closed（同 LLM adapter）。
                await asyncio.sleep(0.25)
            except Exception:  # noqa: BLE001
                pass
        self._session = None

