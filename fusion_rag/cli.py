"""命令行入口（零第三方依赖，纯 argparse）。

子命令：
- ``serve``         启动 HTTP 服务（需 fastapi/uvicorn）
- ``ask``           单轮/多轮问答（in-process Kernel，离线可跑）
- ``chat``          交互式多轮会话
- ``index-file``    索引单个本地文件
- ``index-dir``     批量索引本地目录
- ``stats``         查看知识库索引统计
- ``replay``        按 trace_id 回放一次问答全链路
- ``demo``          自包含端到端演示（建库 → 提问 → 展示引用与轨迹）

所有 in-process 命令共享 ``--config`` 与 ``--data-dir``；数据默认落本地
``~/.fusion_rag``，无任何云端上传（OpenClaw 隐私留存）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .core.config import load_config
from .core.kernel import Kernel
from .core.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# 辅助
# ======================================================================
def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _build_kernel(args: argparse.Namespace) -> Kernel:
    overrides: dict[str, Any] = {}
    if getattr(args, "data_dir", None):
        overrides["app"] = {"data_dir": args.data_dir}
    config = load_config(getattr(args, "config", None), overrides or None)
    return Kernel(config)


def _render_answer(result: Any) -> None:
    print("\n" + "=" * 60)
    print(result.answer)
    print("=" * 60)
    if result.citations:
        print("引用来源：")
        for c in result.citations:
            title = c.title or c.document_id[:8]
            print(f"  [{c.index}] {title} (score={c.score:.4f})")
    print(
        f"\n置信度={result.confidence:.2f} 校验={'通过' if result.validated else '未过'} "
        f"降级={'是' if result.degraded else '否'} 模型={result.model} "
        f"耗时={result.latency_ms}ms trace={result.trace_id}",
    )


# ======================================================================
# 命令实现
# ======================================================================
async def _cmd_ask(args: argparse.Namespace) -> int:
    kernel = _build_kernel(args)
    async with kernel:
        result = await kernel.orchestrator.ask(
            args.question, tenant_id=args.tenant, session_id=args.session,
            kb_ids=args.kb.split(",") if args.kb else None,
        )
        if args.json:
            _print_json(result.to_dict())
        else:
            _render_answer(result)
    return 0


async def _cmd_chat(args: argparse.Namespace) -> int:
    kernel = _build_kernel(args)
    session = args.session or "cli-chat"
    print(f"进入多轮会话（session={session}），输入 :q 退出，:reset 换会话。")
    async with kernel:
        while True:
            try:
                line = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in (":q", ":quit", ":exit"):
                break
            if line == ":reset":
                session = f"cli-chat-{id(object())}"
                print(f"已切换到新会话 session={session}")
                continue
            result = await kernel.orchestrator.ask(
                line, tenant_id=args.tenant, session_id=session,
            )
            print(f"\n助手> {result.answer}")
            if result.citations:
                srcs = ", ".join(
                    f"[{c.index}]{c.title or c.document_id[:6]}" for c in result.citations
                )
                print(f"来源：{srcs}")
    return 0


async def _cmd_index_file(args: argparse.Namespace) -> int:
    from .knowledge.loader import load_file

    loaded = load_file(args.path)
    if loaded is None:
        print(f"无法加载文件：{args.path}", file=sys.stderr)
        return 2
    kernel = _build_kernel(args)
    async with kernel:
        result = await kernel.indexer.add_text(
            loaded.content, tenant_id=args.tenant, kb_id=args.kb,
            title=loaded.title, source=loaded.metadata.get("source", ""),
            metadata=loaded.metadata, force=args.force,
        )
        _print_json({
            "document_id": result.document_id, "chunks": result.chunks,
            "skipped": result.skipped, "reason": result.reason,
        })
    return 0


async def _cmd_index_dir(args: argparse.Namespace) -> int:
    kernel = _build_kernel(args)
    async with kernel:
        results = await kernel.indexer.ingest_directory(
            args.path, tenant_id=args.tenant, kb_id=args.kb,
            globs=args.globs.split(",") if args.globs else None, force=args.force,
        )
        _print_json({
            "total": len(results),
            "indexed": sum(1 for r in results if not r.skipped),
            "skipped": sum(1 for r in results if r.skipped),
        })
    return 0


async def _cmd_stats(args: argparse.Namespace) -> int:
    kernel = _build_kernel(args)
    async with kernel:
        _print_json(await kernel.indexer.stats())
    return 0


async def _cmd_replay(args: argparse.Namespace) -> int:
    kernel = _build_kernel(args)
    async with kernel:
        trace = await kernel.service("tracer").replay(args.trace_id)
        if trace is None:
            print(f"轨迹不存在：{args.trace_id}", file=sys.stderr)
            return 2
        _print_json(trace)
    return 0


async def _cmd_demo(args: argparse.Namespace) -> int:
    """自包含演示：临时数据目录内建库、提问、展示引用与轨迹回放。"""
    import tempfile

    if not getattr(args, "data_dir", None):
        args.data_dir = tempfile.mkdtemp(prefix="fusion-rag-demo-")
    docs = [
        ("AgentScope 是阿里巴巴开源的多智能体框架，支持分布式编排、可视化会话管理"
         "与高并发问答处理，适配多角色协作场景。", "agentscope"),
        ("DeepSeek Harness 采用全插件化微内核架构，模型无绑定，支持任务轨迹回放、"
         "可插拔工具链与轻量化沙箱，可观测性强。", "harness"),
        ("DeepAgents 擅长长任务智能规划与复杂问题分层拆解，提供上下文卸载、超长对话"
         "记忆与人工介入校验机制。", "deepagents"),
        ("OpenClaw 强调本地化隐私可控、持久化智能体会话与私有知识库安全隔离，"
         "适配企业私密知识问答场景。", "openclaw"),
    ]
    kernel = _build_kernel(args)
    async with kernel:
        for text, title in docs:
            await kernel.indexer.add_text(
                text, tenant_id=args.tenant, kb_id=args.kb, title=title,
            )
        print(f"\n已索引 {len(docs)} 篇文档。统计：")
        _print_json(await kernel.indexer.stats())

        question = args.question or "DeepAgents 擅长什么？"
        print(f"\n提问：{question}")
        result = await kernel.orchestrator.ask(
            question, tenant_id=args.tenant, session_id="demo",
        )
        _render_answer(result)

        trace = await kernel.service("tracer").replay(result.trace_id)
        if trace:
            spans = trace.get("spans") or []
            print(f"\n轨迹回放 trace={result.trace_id} 共 {len(spans)} 个 span：")
            for s in spans[:12]:
                print(f"  - {s.get('name')} ({s.get('duration_ms', '?')}ms)")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # type: ignore
    except ImportError:
        print("启动 HTTP 服务需安装：pip install fastapi uvicorn", file=sys.stderr)
        return 2
    from .api.app import create_app

    overrides: dict[str, Any] = {}
    if args.data_dir:
        overrides["app"] = {"data_dir": args.data_dir}
    app = create_app(args.config, **overrides)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


# ======================================================================
# 参数解析
# ======================================================================
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="配置文件路径 (.yaml/.json)")
    p.add_argument("--data-dir", default=None, help="覆盖数据目录（本地留存）")
    p.add_argument("--tenant", default="default", help="租户 id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fusion-rag",
        description="融合多智能体优势的企业级 RAG 知识问答框架 CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ask", help="单轮问答")
    p.add_argument("question")
    p.add_argument("--kb", default=None, help="限定知识库，逗号分隔")
    p.add_argument("--session", default=None)
    p.add_argument("--json", action="store_true", help="输出完整 JSON")
    _add_common(p)
    p.set_defaults(func=_cmd_ask, coro=True)

    p = sub.add_parser("chat", help="交互式多轮会话")
    p.add_argument("--session", default=None)
    _add_common(p)
    p.set_defaults(func=_cmd_chat, coro=True)

    p = sub.add_parser("index-file", help="索引单个文件")
    p.add_argument("path")
    p.add_argument("--kb", default="general")
    p.add_argument("--force", action="store_true")
    _add_common(p)
    p.set_defaults(func=_cmd_index_file, coro=True)

    p = sub.add_parser("index-dir", help="批量索引目录")
    p.add_argument("path")
    p.add_argument("--kb", default="general")
    p.add_argument("--globs", default=None, help="逗号分隔的 glob，如 **/*.md,**/*.txt")
    p.add_argument("--force", action="store_true")
    _add_common(p)
    p.set_defaults(func=_cmd_index_dir, coro=True)

    p = sub.add_parser("stats", help="知识库索引统计")
    _add_common(p)
    p.set_defaults(func=_cmd_stats, coro=True)

    p = sub.add_parser("replay", help="按 trace_id 回放问答链路")
    p.add_argument("trace_id")
    _add_common(p)
    p.set_defaults(func=_cmd_replay, coro=True)

    p = sub.add_parser("demo", help="自包含端到端演示")
    p.add_argument("--question", default=None)
    p.add_argument("--kb", default="general")
    _add_common(p)
    p.set_defaults(func=_cmd_demo, coro=True)

    p = sub.add_parser("serve", help="启动 HTTP 服务")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--log-level", default="info")
    _add_common(p)
    p.set_defaults(func=_cmd_serve, coro=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        cfg = load_config(args.config)
        args.host = args.host or str(cfg.get("api.host", "127.0.0.1"))
        args.port = args.port or int(cfg.get("api.port", 8300))
        return args.func(args)

    if getattr(args, "coro", False):
        return asyncio.run(args.func(args))
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
