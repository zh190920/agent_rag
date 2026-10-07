"""示例 02：多轮对话记忆 + 多租户隔离。

演示：
- 同一 ``session_id`` 下多轮追问，短期记忆自动带入上下文；
- 不同租户的知识库彼此隔离，越权检索拿不到对方数据；
- PII 入库前自动脱敏。

运行::

    python examples/02_multi_turn_and_isolation.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_rag.core.config import load_config  # noqa: E402
from fusion_rag.core.kernel import Kernel  # noqa: E402


async def main() -> None:
    config = load_config(None, {
        "app": {"data_dir": tempfile.mkdtemp(prefix="fusion-mt-"), "log_level": "WARNING"},
        # 声明两个租户，各自可见不同知识库
        "security": {"tenants": {
            "default": {"kbs": ["general"], "rate_limit_rps": 50, "burst": 100},
            "acme": {"kbs": ["hr"], "rate_limit_rps": 50, "burst": 100},
        }},
    })

    kernel = Kernel(config)
    async with kernel:
        # 公共知识库
        await kernel.indexer.add_text(
            "AgentScope 是多智能体框架，支持分布式编排与高并发问答，并提供可视化会话管理。",
            tenant_id="default", kb_id="general", title="agentscope",
        )
        # acme 私有 HR 知识库（含手机号，入库前会被脱敏）
        await kernel.indexer.add_text(
            "ACME 年假政策：入职满一年享 10 天年假，咨询 HR 热线 13800001111。",
            tenant_id="acme", kb_id="hr", title="acme-hr",
        )

        print("== 多轮对话（session=demo）==")
        turns = ["AgentScope 支持高并发吗？", "它还有别的特性吗？"]
        for q in turns:
            r = await kernel.orchestrator.ask(q, tenant_id="default", session_id="demo")
            print(f"\n问：{q}\n答：{r.answer[:120]}")
            print(f"（引用 {len(r.citations)} 条，置信度 {r.confidence:.2f}）")

        store = kernel.service("store")
        msgs = await store.get_messages("demo")
        print(f"\n会话累计消息数：{len(msgs)}（多轮上下文已持久化）")

        print("\n== 租户隔离 ==")
        # acme 能查到自己的 HR 内容
        r_acme = await kernel.orchestrator.ask(
            "年假有多少天？", tenant_id="acme", session_id="acme-1",
        )
        print(f"[acme] 引用来源：{[c.title for c in r_acme.citations]}")

        # default 租户查同样问题，拿不到 acme 的私有 HR 数据
        r_default = await kernel.orchestrator.ask(
            "年假有多少天？", tenant_id="default", session_id="def-1",
        )
        leaked = any(c.title == "acme-hr" for c in r_default.citations)
        print(f"[default] 引用来源：{[c.title for c in r_default.citations]}")
        print(f"跨租户泄露：{'是（异常！）' if leaked else '否（隔离生效）'}")

        print("\n== 脱敏验证 ==")
        hits = await kernel.service("retriever").retrieve(
            "HR 热线", top_k=3, tenant_id="acme", kb_ids=["hr"],
        )
        for h in hits:
            assert "13800001111" not in h.content, "手机号未脱敏！"
            if "热线" in h.content or "HR" in h.content:
                print(f"检索内容片段：{h.content[:80]}")
        print("手机号已在入库前脱敏，原文不留存。")


if __name__ == "__main__":
    asyncio.run(main())
