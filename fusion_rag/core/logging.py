"""结构化日志（JSON lines）+ trace_id 自动注入。

借鉴 AgentScope Java 的 traceId 跨异步传播：trace_id 存放于
``contextvars``，日志 Filter 自动附加，无需业务代码显式传参。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any

# 全链路 trace_id / session_id 上下文变量（observability.tracer 负责设置）
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "fusion_trace_id", default="-",
)
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "fusion_session_id", default="-",
)

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class _ContextFilter(logging.Filter):
    """把 contextvars 中的 trace/session 注入每条日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        record.session_id = session_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """JSON lines 格式化器，extra 字段一并输出。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """人类可读格式（开发调试用）。"""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s [%(trace_id)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


def setup_logging(level: str = "INFO", as_json: bool = True) -> None:
    """初始化根日志。幂等：重复调用只更新 level/formatter。"""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        root.addHandler(handler)
    formatter: logging.Formatter = JsonFormatter() if as_json else TextFormatter()
    for handler in root.handlers:
        handler.setFormatter(formatter)
        # 避免重复 addFilter
        if not any(isinstance(f, _ContextFilter) for f in handler.filters):
            handler.addFilter(_ContextFilter())
    # 降噪第三方
    for noisy in ("asyncio", "urllib3", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def utc_now_ms() -> int:
    return int(time.time() * 1000)
