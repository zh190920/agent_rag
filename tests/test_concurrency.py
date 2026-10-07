"""并发与过载保护测试：批量并发问答、会话隔离、沙箱闸门。"""

from __future__ import annotations

import asyncio
import time

from conftest import make_kernel, run, seed


def test_ask_many_batch_concurrent():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await seed(kernel)
            requests = [
                {"question": q, "tenant_id": "default", "session_id": f"b{i}"}
                for i, q in enumerate([
                    "AgentScope 是什么？", "Harness 有哪些特性？",
                    "DeepAgents 擅长什么？", "OpenClaw 强调什么？",
                ] * 8)  # 32 条
            ]
            results = await kernel.orchestrator.ask_many(requests)
            assert len(results) == 32
            assert all(r.answer.strip() for r in results)
            # 每条都应有独立 trace
            assert len({r.trace_id for r in results}) == 32

    run(scenario)


def test_high_concurrency_gather():
    """50 路并发提问全部成功，且完成后无在途请求。"""
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await seed(kernel)
            started = time.monotonic()
            results = await asyncio.gather(*(
                kernel.orchestrator.ask(
                    "多智能体框架有哪些能力？", tenant_id="default",
                    session_id=f"g{i}",
                )
                for i in range(50)
            ))
            elapsed = time.monotonic() - started
            assert len(results) == 50
            assert all(r.answer.strip() for r in results)
            assert kernel.sandbox.inflight == 0, "全部完成后在途请求应归零"
            return elapsed

    run(scenario)


def test_sandbox_bounds_concurrency():
    """并发闸门应把在途请求确定性地限制在 max_concurrent 以内。"""
    from fusion_rag.core.sandbox import Sandbox

    async def scenario():
        sb = Sandbox(max_concurrent=3, request_timeout=5.0)
        state = {"cur": 0, "peak": 0}

        async def work():
            state["cur"] += 1
            state["peak"] = max(state["peak"], state["cur"])
            await asyncio.sleep(0.02)
            state["cur"] -= 1
            return True

        results = await asyncio.gather(*(sb.run_guarded(work) for _ in range(20)))
        assert len(results) == 20 and all(results)
        assert state["peak"] <= 3, f"在途峰值 {state['peak']} 超过闸门上限 3"
        assert sb.inflight == 0

    run(scenario)


def test_token_bucket_rate_limit():
    """令牌桶耗尽后等待超时应抛 RateLimitError。"""
    from fusion_rag.core.exceptions import RateLimitError
    from fusion_rag.core.sandbox import TokenBucket

    async def scenario():
        bucket = TokenBucket(rate=1.0, burst=2)
        await bucket.acquire(timeout=0.5)
        await bucket.acquire(timeout=0.5)
        raised = False
        try:
            # 令牌已耗尽，rate=1/s，0.1s 内无法补充 → 超时
            await bucket.acquire(timeout=0.1)
        except RateLimitError:
            raised = True
        assert raised, "令牌耗尽应触发限流"

    run(scenario)


def test_session_isolation_under_concurrency():
    """不同会话的记忆互不串扰。"""
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await seed(kernel)
            await asyncio.gather(
                kernel.orchestrator.ask("AgentScope 是什么？", tenant_id="default", session_id="A"),
                kernel.orchestrator.ask("OpenClaw 是什么？", tenant_id="default", session_id="B"),
            )
            store = kernel.service("store")
            a = await store.get_messages("A")
            b = await store.get_messages("B")
            a_text = " ".join(m.get("content", "") for m in a)
            b_text = " ".join(m.get("content", "") for m in b)
            assert "AgentScope" in a_text
            assert "OpenClaw" in b_text
            # A 会话不应记录 B 的提问
            assert "OpenClaw 是什么" not in a_text

    run(scenario)
