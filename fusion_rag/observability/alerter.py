"""Alerter：基于指标阈值的异常告警（借鉴 DeepSeek-Harness 高可观测性）。

规则以「指标计算函数 + 阈值比较」表达，OpsAgent 周期性调用
:meth:`Alerter.evaluate`；命中即：日志告警 + 事件总线广播（可挂
钉钉/企业微信/webhook 订阅者）+ 冷却去重避免风暴。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.events import Event, EventBus
from ..core.logging import get_logger
from .metrics import MetricsRegistry

logger = get_logger(__name__)


@dataclass
class Alert:
    """一条告警。"""

    name: str
    severity: str          # info / warning / critical
    value: float
    threshold: float
    message: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "severity": self.severity,
            "value": round(self.value, 4), "threshold": self.threshold,
            "message": self.message, "ts": round(self.ts, 3),
        }


@dataclass
class AlertRule:
    """告警规则。"""

    name: str
    compute: Callable[[MetricsRegistry], float]
    threshold: float
    severity: str = "warning"
    comparison: str = ">"       # ">" 或 "<"
    cooldown: float = 60.0
    message: str = ""

    def breached(self, value: float) -> bool:
        if self.comparison == "<":
            return value < self.threshold
        return value > self.threshold


def _error_rate(reg: MetricsRegistry) -> float:
    total = reg.counter("requests_total").value()
    failed = reg.counter("requests_failed").value()
    if total <= 0:
        return 0.0
    return failed / total


def _p95_latency(reg: MetricsRegistry) -> float:
    return reg.histogram("request_latency_seconds").percentile(95)


def _llm_error_rate(reg: MetricsRegistry) -> float:
    calls = reg.counter("llm_calls").value()
    errors = reg.counter("llm_errors").value()
    if calls <= 0:
        return 0.0
    return errors / calls


class Alerter:
    """告警器。"""

    def __init__(
        self,
        registry: MetricsRegistry,
        events: EventBus | None = None,
        rules: list[AlertRule] | None = None,
    ) -> None:
        self._registry = registry
        self._events = events
        self._rules = rules if rules is not None else self.default_rules()
        self._last_fired: dict[str, float] = {}
        self.history: list[Alert] = []

    @staticmethod
    def default_rules(
        error_rate_threshold: float = 0.2,
        p95_latency_threshold: float = 30.0,
    ) -> list[AlertRule]:
        return [
            AlertRule(
                name="high_error_rate", compute=_error_rate,
                threshold=error_rate_threshold, severity="critical",
                message="请求错误率超过阈值",
            ),
            AlertRule(
                name="high_p95_latency", compute=_p95_latency,
                threshold=p95_latency_threshold, severity="warning",
                message="P95 响应延迟超过阈值(秒)",
            ),
            AlertRule(
                name="high_llm_error_rate", compute=_llm_error_rate,
                threshold=0.5, severity="warning",
                message="模型调用失败率偏高（可能触发降级）",
            ),
        ]

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def evaluate(self) -> list[Alert]:
        """评估所有规则，返回本次新触发的告警（含冷却去重）。"""
        fired: list[Alert] = []
        now = time.time()
        for rule in self._rules:
            try:
                value = rule.compute(self._registry)
            except Exception:  # noqa: BLE001
                logger.exception("规则 %s 计算失败", rule.name)
                continue
            if not rule.breached(value):
                continue
            last = self._last_fired.get(rule.name, 0.0)
            if now - last < rule.cooldown:
                continue  # 冷却期内不重复告警
            self._last_fired[rule.name] = now
            alert = Alert(
                name=rule.name, severity=rule.severity, value=value,
                threshold=rule.threshold,
                message=f"{rule.message}: value={value:.3f} {rule.comparison} "
                        f"threshold={rule.threshold}",
            )
            fired.append(alert)
            self.history.append(alert)
            self._emit(alert)
        # 限制历史长度
        if len(self.history) > 500:
            del self.history[: len(self.history) - 500]
        return fired

    def _emit(self, alert: Alert) -> None:
        log_fn = logger.error if alert.severity == "critical" else logger.warning
        log_fn("🚨 告警[%s] %s", alert.severity, alert.message)
        if self._events is not None:
            self._events.publish(Event("alert", alert.to_dict()))

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.history[-limit:]]
