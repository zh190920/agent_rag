"""Tracer 采样行为回归测试。

采样机制的目的：高 QPS 生产环境避免 JSONL 无限膨胀；同时保证异常/降级
trace 不被采样吞掉——否则采样反而拉低可观测度。
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import pytest

from fusion_rag.observability.metrics import MetricsRegistry
from fusion_rag.observability.tracer import Tracer


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_sample_rate_one_persists_everything(tmp_path: Path):
    """默认 sample_rate=1.0 → 与旧行为一致,所有 trace 全量落盘。"""
    p = tmp_path / "traces.jsonl"
    tracer = Tracer(p, store=None, sample_rate=1.0)
    for i in range(5):
        t = tracer.begin(trace_id=f"t{i}", session_id="s", tenant_id="d")
        with tracer.span("work", i=i):
            pass
        asyncio.run(tracer.end(t))
    rows = _read_jsonl(p)
    assert len(rows) == 5


def test_sample_rate_zero_drops_clean_traces(tmp_path: Path):
    """sample_rate=0 → 无异常/无降级 tags 的 trace 一条不落。"""
    p = tmp_path / "traces.jsonl"
    tracer = Tracer(p, store=None, sample_rate=0.0)
    for i in range(5):
        t = tracer.begin(trace_id=f"t{i}", session_id="s", tenant_id="d")
        with tracer.span("work"):
            pass
        asyncio.run(tracer.end(t))
    assert _read_jsonl(p) == []


def test_error_trace_kept_regardless_of_sample_rate(tmp_path: Path):
    """sample_rate=0 + span 抛异常 → 强制保留(error trace 不能被采样吞)。"""
    p = tmp_path / "traces.jsonl"
    tracer = Tracer(p, store=None, sample_rate=0.0)

    t = tracer.begin(trace_id="err1", session_id="s", tenant_id="d")
    with pytest.raises(RuntimeError):
        with tracer.span("boom"):
            raise RuntimeError("intentional")
    asyncio.run(tracer.end(t))

    rows = _read_jsonl(p)
    assert len(rows) == 1
    assert rows[0]["trace_id"] == "err1"
    assert any(s["status"] == "error" for s in rows[0]["spans"])


def test_degraded_tag_trace_kept(tmp_path: Path):
    """tags.degraded=True → 强制保留(agentic 循环超时/LLM 失败这类降级)。"""
    p = tmp_path / "traces.jsonl"
    tracer = Tracer(p, store=None, sample_rate=0.0)

    t = tracer.begin(trace_id="deg1", session_id="s", tenant_id="d")
    with tracer.span("work"):
        pass
    tracer.tag(degraded=True, confidence=0.4)
    asyncio.run(tracer.end(t))

    rows = _read_jsonl(p)
    assert len(rows) == 1
    assert rows[0]["trace_id"] == "deg1"
    assert rows[0]["tags"]["degraded"] is True


def test_validated_false_trace_kept(tmp_path: Path):
    """tags.validated=False → 强制保留(validator 未通过的答案)。"""
    p = tmp_path / "traces.jsonl"
    tracer = Tracer(p, store=None, sample_rate=0.0)

    t = tracer.begin(trace_id="inv1", session_id="s", tenant_id="d")
    with tracer.span("work"):
        pass
    tracer.tag(validated=False)
    asyncio.run(tracer.end(t))

    rows = _read_jsonl(p)
    assert len(rows) == 1
    assert rows[0]["trace_id"] == "inv1"


def test_keep_error_traces_disabled_drops_everything(tmp_path: Path):
    """keep_error_traces=False → 采样纯粹按 rate,连错误 trace 也不留。"""
    p = tmp_path / "traces.jsonl"
    tracer = Tracer(p, store=None, sample_rate=0.0, keep_error_traces=False)

    t = tracer.begin(trace_id="err2", session_id="s", tenant_id="d")
    with pytest.raises(RuntimeError):
        with tracer.span("boom"):
            raise RuntimeError("intentional")
    asyncio.run(tracer.end(t))

    assert _read_jsonl(p) == []


def test_dropped_count_bumped_to_metrics(tmp_path: Path):
    """被采样丢弃的 trace 计数递增,运维能监控采样命中率。"""
    p = tmp_path / "traces.jsonl"
    metrics = MetricsRegistry()
    tracer = Tracer(p, store=None, sample_rate=0.0, metrics=metrics)

    for i in range(3):
        t = tracer.begin(trace_id=f"t{i}", session_id="s", tenant_id="d")
        with tracer.span("work"):
            pass
        asyncio.run(tracer.end(t))

    assert metrics.counter("trace_dropped").value() == 3.0


def test_partial_sampling_keeps_reasonable_portion(tmp_path: Path):
    """sample_rate=0.5 → 大量样本下命中比例大致一半(±10%)。

    固定 random seed 保证测试可重放,不因随机性偶发失败。
    """
    p = tmp_path / "traces.jsonl"
    tracer = Tracer(p, store=None, sample_rate=0.5)

    random.seed(42)
    N = 200
    for i in range(N):
        t = tracer.begin(trace_id=f"t{i}", session_id="s", tenant_id="d")
        with tracer.span("work"):
            pass
        asyncio.run(tracer.end(t))

    kept = len(_read_jsonl(p))
    assert 0.4 * N <= kept <= 0.6 * N, f"kept={kept} out of expected range"


def test_begin_sets_sampled_flag(tmp_path: Path):
    """begin() 时就把 sampled 定下,而不是 end() 时才决定 —— 便于中途
    其他组件按 trace.sampled 做二次优化(如 span 属性精简)。"""
    tracer = Tracer(tmp_path / "t.jsonl", store=None, sample_rate=0.0)
    t = tracer.begin(trace_id="x")
    assert t.sampled is False

    tracer2 = Tracer(tmp_path / "t.jsonl", store=None, sample_rate=1.0)
    t2 = tracer2.begin(trace_id="y")
    assert t2.sampled is True
