"""示例 03：高并发压测（大批量并发问答）。

演示 AgentScope 式高并发处理能力：一次性发起 N 路并发问答，统计吞吐、
延迟分位、成功率，并展示沙箱并发闸门与令牌桶限流的护栏效果。

运行::

    python examples/03_concurrent_load.py [并发数，默认100]
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_rag.core.config import load_config  # noqa: E402
from fusion_rag.core.kernel import Kernel  # noqa: E402

QUESTIONS = [
    "AgentScope 有哪些能力？",
    "Harness 的微内核有什么特点？",
    "DeepAgents 如何处理长任务？",
    "OpenClaw 的隐私特性是什么？",
    "Claude Code 的校验能力如何？",
]


async def main(n: int) -> None:
    config = load_config(None, {
        "app": {"data_dir": tempfile.mkdtemp(prefix="fusion-load-"), "log_level": "WARNING"},
        "kernel": {"max_concurrent_requests": 32, "request_timeout": 60},
        "security": {"tenants": {
            "default": {"kbs": ["general"], "rate_limit_rps": 1000, "burst": 2000},
        }},
    })

    kernel = Kernel(config)
    async with kernel:
        docs = [
            "AgentScope 是多智能体分布式编排框架，支持高并发问答与可视化会话管理。",
            "DeepSeek Harness 是全插件化微内核架构，模型无绑定，支持轨迹回放与沙箱。",
            "DeepAgents 擅长长任务规划与复杂问题分层拆解，支持上下文卸载与人工介入。",
            "OpenClaw 强调本地化隐私可控、持久化会话与私有知识库安全隔离。",
            "Claude Code 具备长文本精读、结果校验与逻辑纠错能力。",
        ]
        for i, text in enumerate(docs):
            await kernel.indexer.add_text(
                text, tenant_id="default", kb_id="general", title=f"doc{i}",
            )
        print(f"已建库 {len(docs)} 篇，开始压测 {n} 路并发问答…\n")

        async def one(i: int):
            q = QUESTIONS[i % len(QUESTIONS)]
            t0 = time.monotonic()
            try:
                res = await kernel.orchestrator.ask(
                    q, tenant_id="default", session_id=f"load-{i}",
                )
                return (time.monotonic() - t0) * 1000, True, res.degraded
            except Exception:  # noqa: BLE001
                return (time.monotonic() - t0) * 1000, False, False

        started = time.monotonic()
        results = await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.monotonic() - started

        latencies = sorted(r[0] for r in results)
        ok = sum(1 for r in results if r[1])
        degraded = sum(1 for r in results if r[2])

        def pct(p: float) -> float:
            idx = min(len(latencies) - 1, int(len(latencies) * p))
            return latencies[idx]

        print("== 压测结果 ==")
        print(f"总请求    ：{n}")
        print(f"成功      ：{ok}（{ok / n * 100:.1f}%）")
        print(f"降级(兜底) ：{degraded}")
        print(f"墙钟耗时  ：{wall:.2f}s")
        print(f"吞吐      ：{n / wall:.1f} req/s")
        print(f"延迟 p50  ：{pct(0.50):.1f}ms")
        print(f"延迟 p95  ：{pct(0.95):.1f}ms")
        print(f"延迟 p99  ：{pct(0.99):.1f}ms")
        print(f"在途归零  ：{kernel.sandbox.inflight == 0}")

        snap = kernel.service("metrics").snapshot()
        print(f"\n指标快照 uptime={snap['uptime_seconds']}s，"
              f"counters={len(snap['counters'])} 项，histograms={len(snap['histograms'])} 项")


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    asyncio.run(main(total))
