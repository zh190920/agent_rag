"""ConcurrencyQuota：按租户/请求的两层并发子配额测试。

覆盖：
- 子配额关闭（None / 0 / >= 全局）→ 行为等价旧单全局闸门；
- 单 key 并发被稳定限制在 per_key 上限内（fanout 背压）；
- 不同 key 各自受限、共享全局闸门（跨租户公平，防饿死）；
- max_keys 超限回落"仅全局闸门"（无界增长防护 + 不死锁）；
- quota_key ContextVar 读写与空串归一。
"""

from __future__ import annotations

import asyncio

from fusion_rag.core.quota import (
    ConcurrencyQuota,
    get_quota_key,
    quota_key_var,
    set_quota_key,
)


class _Tracker:
    """记录整体与按 key 的并发峰值。"""

    def __init__(self) -> None:
        self.cur = 0
        self.peak = 0
        self.cur_key: dict[str, int] = {}
        self.peak_key: dict[str, int] = {}

    async def task(self, key: str, quota: ConcurrencyQuota, hold: float = 0.02) -> None:
        async with quota.acquire(key):
            self.cur += 1
            self.peak = max(self.peak, self.cur)
            self.cur_key[key] = self.cur_key.get(key, 0) + 1
            self.peak_key[key] = max(self.peak_key.get(key, 0), self.cur_key[key])
            await asyncio.sleep(hold)
            self.cur_key[key] -= 1
            self.cur -= 1


def test_quota_disabled_when_per_key_none():
    async def scenario():
        q = ConcurrencyQuota(4, None)
        assert not q.enabled
        t = _Tracker()
        await asyncio.gather(*(t.task("A", q) for _ in range(12)))
        assert t.peak <= 4, f"仅全局闸门应<=4，实测 {t.peak}"
        # 关闭态不应登记任何 per-key 信号量
        assert not q._per_key

    asyncio.run(scenario())


def test_quota_per_key_bounds_single_tenant():
    """同一 key 的并发被钉死在 per_key 上限内（fanout 风暴背压）。"""
    async def scenario():
        q = ConcurrencyQuota(16, 2)
        assert q.enabled
        t = _Tracker()
        await asyncio.gather(*(t.task("A", q) for _ in range(10)))
        assert t.peak_key["A"] <= 2, f"租户 A 应<=2，实测 {t.peak_key['A']}"

    asyncio.run(scenario())


def test_quota_two_keys_each_capped_and_share_global():
    """两个 key 各自受 per-key 上限约束，合计受全局约束（跨租户公平）。"""
    async def scenario():
        # 全局 8 足够宽；per-key=2 → A、B 各最多 2，合计上限 4
        q = ConcurrencyQuota(8, 2)
        t = _Tracker()
        await asyncio.gather(
            *(t.task("A", q) for _ in range(6)),
            *(t.task("B", q) for _ in range(6)),
        )
        assert t.peak_key["A"] <= 2
        assert t.peak_key["B"] <= 2
        assert t.peak <= 4, f"两 key 合计应<=4，实测 {t.peak}"

    asyncio.run(scenario())


def test_quota_per_key_ge_global_is_disabled():
    """per_key >= global 时子配额无意义 → 关闭，仅全局闸门。

    Semaphore 在构造时绑定事件循环（与旧单闸门同行为），故需在建 loop 内。
    """
    async def scenario():
        assert not ConcurrencyQuota(4, 4).enabled
        assert not ConcurrencyQuota(4, 8).enabled
        assert ConcurrencyQuota(4, 2).enabled

    asyncio.run(scenario())


def test_quota_global_is_binding():
    """per_key(4) > global(2) 时，实际并发由更紧的全局闸门决定。"""
    async def scenario():
        q = ConcurrencyQuota(2, 4)
        assert not q.enabled  # per_key >= global → 关闭子配额
        t = _Tracker()
        await asyncio.gather(*(t.task("A", q) for _ in range(8)))
        assert t.peak <= 2

    asyncio.run(scenario())


def test_quota_max_keys_fallback_no_deadlock():
    """key 数超过 max_keys 后新 key 回落仅全局闸门，仍可完成（不死锁）。"""
    async def scenario():
        q = ConcurrencyQuota(2, 1, max_keys=1)
        t = _Tracker()
        # 两个不同 key，但 max_keys=1 → 只会有一个 key 建子信号量，
        # 另一个回落仅全局闸门；靠 global=2 推进，不死锁。
        await asyncio.gather(
            *(t.task("A", q) for _ in range(3)),
            *(t.task("B", q) for _ in range(3)),
        )
        assert len(q._per_key) <= 1, "登记的 key 数应受 max_keys 约束"
        assert t.peak <= 2
        assert t.cur == 0  # 全部完成、无泄漏

    asyncio.run(scenario())


def test_quota_key_contextvar_roundtrip():
    tok = quota_key_var.set(None)
    try:
        assert get_quota_key() is None
        set_quota_key("tenant-x")
        assert get_quota_key() == "tenant-x"
        set_quota_key("")  # 空串归一为 None（关闭子配额）
        assert get_quota_key() is None
        set_quota_key(None)
        assert get_quota_key() is None
    finally:
        quota_key_var.reset(tok)
