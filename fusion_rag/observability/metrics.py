"""轻量指标注册表（无 Prometheus 依赖，可导出其文本格式）。

提供 Counter / Histogram / Gauge 三类指标，线程安全（锁保护），支持：
- ``snapshot()`` 输出 JSON 快照（供 OpsAgent 落盘/告警）
- ``render_prometheus()`` 输出 Prometheus 文本暴露格式（供 /metrics）
- 直方图内置分位数（P50/P90/P95/P99）计算
"""

from __future__ import annotations

import threading
import time
from typing import Any


class Counter:
    """单调递增计数器，可带 label 维度。"""

    def __init__(self, name: str, help_text: str = "") -> None:
        self.name = name
        self.help = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"name": self.name, "labels": dict(k), "value": v}
                for k, v in self._values.items()
            ]


class Gauge:
    """可增可减的瞬时值。"""

    def __init__(self, name: str, help_text: str = "") -> None:
        self.name = name
        self.help = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        self.inc(-amount, **labels)

    def value(self, **labels: str) -> float:
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"name": self.name, "labels": dict(k), "value": v}
                for k, v in self._values.items()
            ]


_DEFAULT_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class Histogram:
    """直方图：计数 + 求和 + 分桶，支持分位数估算。"""

    def __init__(
        self, name: str, help_text: str = "",
        buckets: tuple[float, ...] = _DEFAULT_BUCKETS,
    ) -> None:
        self.name = name
        self.help = help_text
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _slot(self, key: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        slot = self._counts.get(key)
        if slot is None:
            slot = {
                "count": 0, "sum": 0.0,
                "buckets": [0] * len(self.buckets),
                "samples": [],  # 保留有限样本用于精确分位数
            }
            self._counts[key] = slot
        return slot

    def observe(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            slot = self._slot(key)
            slot["count"] += 1
            slot["sum"] += value
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    slot["buckets"][i] += 1
            # 有界样本（最近 1000 个）用于分位数
            samples = slot["samples"]
            samples.append(value)
            if len(samples) > 1000:
                del samples[0:len(samples) - 1000]

    def percentile(self, q: float, **labels: str) -> float:
        """基于样本的分位数（q ∈ [0,100]）。"""
        key = tuple(sorted(labels.items()))
        with self._lock:
            slot = self._counts.get(key)
            if not slot or not slot["samples"]:
                return 0.0
            data = sorted(slot["samples"])
            if len(data) == 1:
                return data[0]
            rank = (q / 100.0) * (len(data) - 1)
            lo = int(rank)
            hi = min(lo + 1, len(data) - 1)
            frac = rank - lo
            return data[lo] * (1 - frac) + data[hi] * frac

    def count(self, **labels: str) -> int:
        key = tuple(sorted(labels.items()))
        with self._lock:
            return int(self._counts.get(key, {}).get("count", 0))

    def collect(self) -> list[dict[str, Any]]:
        with self._lock:
            out = []
            for key, slot in self._counts.items():
                labels = dict(key)
                out.append({
                    "name": self.name, "labels": labels,
                    "count": slot["count"], "sum": round(slot["sum"], 6),
                })
            return out


class MetricsRegistry:
    """指标注册表 + 便捷工厂。"""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()
        self._created = time.time()

    def counter(self, name: str, help_text: str = "") -> Counter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, help_text)
            return self._counters[name]

    def gauge(self, name: str, help_text: str = "") -> Gauge:
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, help_text)
            return self._gauges[name]

    def histogram(
        self, name: str, help_text: str = "", buckets: tuple[float, ...] = _DEFAULT_BUCKETS,
    ) -> Histogram:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, help_text, buckets)
            return self._histograms[name]

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """输出全量指标 JSON 快照。"""
        with self._lock:
            counters = [c.collect() for c in self._counters.values()]
            gauges = [g.collect() for g in self._gauges.values()]
            histograms = {
                name: {
                    "series": h.collect(),
                    "p50": h.percentile(50),
                    "p90": h.percentile(90),
                    "p95": h.percentile(95),
                    "p99": h.percentile(99),
                }
                for name, h in self._histograms.items()
            }
        return {
            "uptime_seconds": round(time.time() - self._created, 1),
            "counters": counters,
            "gauges": gauges,
            "histograms": histograms,
        }

    def render_prometheus(self) -> str:
        """导出 Prometheus 文本暴露格式。"""
        lines: list[str] = []
        with self._lock:
            for counter in self._counters.values():
                if counter.help:
                    lines.append(f"# HELP {counter.name} {counter.help}")
                lines.append(f"# TYPE {counter.name} counter")
                for item in counter.collect():
                    lines.append(f"{counter.name}{_fmt_labels(item['labels'])} {item['value']}")
            for gauge in self._gauges.values():
                if gauge.help:
                    lines.append(f"# HELP {gauge.name} {gauge.help}")
                lines.append(f"# TYPE {gauge.name} gauge")
                for item in gauge.collect():
                    lines.append(f"{gauge.name}{_fmt_labels(item['labels'])} {item['value']}")
            for hist in self._histograms.values():
                if hist.help:
                    lines.append(f"# HELP {hist.name} {hist.help}")
                lines.append(f"# TYPE {hist.name} histogram")
                for item in hist.collect():
                    labels = item["labels"]
                    lines.append(
                        f"{hist.name}_count{_fmt_labels(labels)} {item['count']}",
                    )
                    lines.append(
                        f"{hist.name}_sum{_fmt_labels(labels)} {item['sum']}",
                    )
                    for q, pname in ((50, "0.5"), (95, "0.95"), (99, "0.99")):
                        val = hist.percentile(q, **labels)
                        qlabels = {**labels, "quantile": pname}
                        lines.append(f"{hist.name}{_fmt_labels(qlabels)} {round(val, 6)}")
        return "\n".join(lines) + "\n"


def _fmt_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"
