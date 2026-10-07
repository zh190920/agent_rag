"""分层配置：内置默认 → YAML 文件（可选依赖）→ 环境变量覆盖。

- 点号路径读取：``config.get("retrieval.top_k", 6)``
- 环境变量覆盖：``FUSION_RAG__RETRIEVAL__TOP_K=10``
- ``${data_dir}`` 占位符在字符串值中自动展开
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from .exceptions import ConfigError
from .logging import get_logger

logger = get_logger(__name__)

_DEFAULTS: dict[str, Any] = {
    "app": {
        "name": "fusion-rag-agent",
        "data_dir": "~/.fusion_rag",
        "log_level": "INFO",
        "log_json": True,
    },
    "kernel": {
        "max_concurrent_requests": 64,
        "request_timeout": 120,
        "cpu_workers": 8,
        "shutdown_grace": 10,
        # ★ 下游 LLM/embedding 并发子配额作域：tenant=按租户公平（默认）
        #   / session=按请求（会话）/ none=关闭子配额（仅全局闸门）。
        "quota_scope": "tenant",
    },
    "llm": {
        "router": {"default": "echo", "retries": 2, "fallback_chain": ["echo"]},
        "providers": {"echo": {"type": "echo"}},
        # ★ 单租户/请求最多占用的模型并发槽位（子配额）；需 < llm 全局
        #   max_concurrency(默认16) 才生效；6 允许单请求 fanout(≤5) 全并发又不
        #   跨租户吞满窗口。0/缺失=关闭（行为同旧单闸门）。
        "tenant_max_concurrency": 6,
    },
    "embedding": {
        "provider": "hash",
        "dimensions": 256,
        # ★ 嵌入子配额（同 llm.tenant_max_concurrency，对 embedding 全局 8）。
        "tenant_max_concurrency": 4,
    },
    "retrieval": {
        "top_k": 6,
        "vector_recall": 20,
        "bm25_recall": 20,
        "rrf_k": 60,
        "score_threshold": 0.05,
        "reranker": None,
        "chunk": {"max_tokens": 400, "overlap_tokens": 60},
        "agentic_tools": False,
        # ★ 策略A：seed 已含 file:line 行级证据时首轮直答（跳过多余工具轮）。
        "agentic_seed_direct": True,
        # ★ 满足才停阈值（①+③）：策略A 直答自报置信低于此值（或呈“资料未
        #   提及”对冲）则不收敛，回退工具循环继续深挖。
        "agentic_seed_direct_min_conf": 0.7,
        # ★ seed 子查询是否走大模型拆解（②b）：会在预检索前额外发一次 chat、
        #   可能顶破 40s；拆解完全由大模型驱动（无确定性正则拆分器）。默认开。
        "seed_llm_decompose": True,
        # ★ 多手册并发子 Agent 检索（fanout）：seed 命中≥2本手册时分派并发 Worker，
        #   任一 Worker self_confidence 达阈即广播停止其他 Worker。默认关，不改变现有行为。
        "fanout": {
            "enabled": False,
            "max_manuals": 5,
            "confidence_threshold": 0.7,
            "worker_max_iters": 4,
            "merge_enabled": True,
        },
    },
    "tools": {
        "enabled": True,
        "max_iters": 6,
        "call_timeout": 15,
        "allow": ["grep", "glob", "find", "list_dir", "read_file"],
        "roots": {},
        "kb_roots": {},
    },
    "skills": {
        "enabled": True,
        # 分层 skill 根目录，顺序即优先级（后层同名覆盖前层，如 base→user→project）。
        # 每个根目录下「含 SKILL.md 的子目录」即一个技能；随 tool_agent 渐进披露。
        "dirs": [],
    },
    "planning": {
        "enabled": True,
        "max_sub_questions": 5,
        "decompose_threshold": 30,
        "context_token_budget": 6000,
        "short_term_rounds": 6,
    },
    "validation": {"enabled": True, "max_retry": 2, "min_confidence": 0.5},
    "security": {
        "redaction_enabled": True,
        "hitl_enabled": False,
        "default_tenant": "default",
        "tenants": {"default": {"kbs": ["general"], "rate_limit_rps": 10, "burst": 20}},
    },
    "storage": {
        "sqlite": {
            "path": "${data_dir}/state/fusion_rag.db",
            "wal": True,
            "busy_timeout_ms": 5000,
        },
        "vector_store": {"type": "local", "path": "${data_dir}/vectors"},
    },
    "observability": {
        "trace_path": "${data_dir}/traces/traces.jsonl",
        "trace_sample_rate": 1.0,
        "trace_keep_error_traces": True,
        "metrics_path": "${data_dir}/metrics/snapshot.json",
        "alert": {
            "error_rate_threshold": 0.2,
            "p95_latency_threshold": 30,
            "window_seconds": 60,
        },
    },
    "api": {"host": "127.0.0.1", "port": 8300},
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce(raw: str) -> Any:
    """环境变量字符串 → python 值。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


class Config:
    """只读点号访问配置对象。"""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return self._expand(node)

    def section(self, path: str) -> dict[str, Any]:
        value = self.get(path, {})
        if not isinstance(value, dict):
            raise ConfigError(f"配置节 {path} 不是对象")
        return value

    def path(self, path: str) -> Path:
        """读取并展开为绝对 Path（自动建父目录由调用方决定）。"""
        raw = self.get(path)
        if raw is None:
            raise ConfigError(f"缺少路径配置: {path}")
        return Path(str(raw)).expanduser().resolve()

    @property
    def data_dir(self) -> Path:
        return Path(str(self.get("app.data_dir"))).expanduser().resolve()

    def _expand(self, value: Any) -> Any:
        if isinstance(value, str) and "${data_dir}" in value:
            return value.replace("${data_dir}", str(self.data_dir))
        if isinstance(value, dict):
            return {k: self._expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._expand(v) for v in value]
        return value

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({json.dumps(self._data, ensure_ascii=False)[:200]}...)"


def _apply_env_overrides(data: dict[str, Any]) -> None:
    prefix = "FUSION_RAG__"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = [p.lower() for p in env_key[len(prefix):].split("__") if p]
        if not parts:
            continue
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                logger.warning("环境变量 %s 与现有配置类型冲突，忽略", env_key)
                break
        else:
            node[parts[-1]] = _coerce(env_val)


def load_config(
    file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """按 默认 → 文件 → overrides → 环境变量 顺序合成配置。"""
    data = copy.deepcopy(_DEFAULTS)
    if file is not None:
        path = Path(file).expanduser()
        if path.exists():
            data = _deep_merge(data, _read_file(path))
        else:
            logger.warning("配置文件不存在，使用默认配置: %s", path)
    if overrides:
        data = _deep_merge(data, overrides)
    _apply_env_overrides(data)
    return Config(data)


def _read_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ConfigError(
                "读取 YAML 配置需要 PyYAML：pip install pyyaml，或改用 .json 配置",
            ) from exc
        loaded = yaml.safe_load(text) or {}
    elif path.suffix == ".json":
        loaded = json.loads(text)
    else:
        raise ConfigError(f"不支持的配置文件格式: {path.suffix}")
    if not isinstance(loaded, dict):
        raise ConfigError("配置文件根节点必须是对象")
    return loaded
