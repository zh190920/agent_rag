"""示例 01：最小可用 RAG 问答（离线、零 API Key）。

演示：建库 → 提问 → 带引用作答 → 轨迹回放。默认使用 EchoLLM + HashEmbedding，
无需任何网络或密钥即可跑通全链路；配置真实模型后即得到生成式答案。

运行::

    python examples/01_quickstart.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

# 允许直接运行脚本（未安装为包时）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_rag.core.config import load_config  # noqa: E402
from fusion_rag.core.kernel import Kernel  # noqa: E402

DOCS = [
    ("AgentScope 是阿里巴巴开源的多智能体框架，支持分布式编排、可视化会话管理"
     "与高并发问答处理，适配多角色协作问答场景。", "agentscope"),
    ("DeepSeek Harness 采用全插件化微内核架构，模型无绑定，支持任务轨迹回放、"
     "可插拔工具链与轻量化沙箱，具备高可观测性。", "harness"),
    ("DeepAgents 擅长长任务智能规划与复杂问题分层拆解，提供上下文卸载、超长对话"
     "记忆与人工介入校验机制。", "deepagents"),
    ("OpenClaw 强调本地化隐私可控、持久化智能体会话与私有知识库安全隔离，"
     "适配企业私密知识问答场景。", "openclaw"),
    ("Claude Code 具备深度语义理解、长文本精读推理、精细化结果校验与逻辑纠错能力。",
     "claudecode"),
]


async def main() -> None:
    # 用独立临时数据目录，避免污染 ~/.fusion_rag
    config = load_config(None, {"app": {
        "data_dir": tempfile.mkdtemp(prefix="fusion-quickstart-"),
        "log_level": "WARNING",
    }})

    kernel = Kernel(config)
    async with kernel:
        print("== 1) 建库 ==")
        for text, title in DOCS:
            await kernel.indexer.add_text(
                text, tenant_id="default", kb_id="general", title=title,
            )
        stats = await kernel.indexer.stats()
        print(f"已索引 {stats['documents']} 篇文档，向量维度 {stats['dimensions']}")

        print("\n== 2) 提问 ==")
        question = "哪个框架强调本地化隐私与私有知识库隔离？"
        result = await kernel.orchestrator.ask(
            question, tenant_id="default", session_id="quickstart",
        )
        print(f"问：{question}")
        print(f"答：{result.answer}")
        print("\n引用来源：")
        for c in result.citations:
            print(f"  [{c.index}] {c.title or c.document_id[:8]} (score={c.score:.4f})")
        print(
            f"\n置信度={result.confidence:.2f} 校验={'通过' if result.validated else '未过'} "
            f"降级={'是' if result.degraded else '否'} 耗时={result.latency_ms}ms",
        )

        print("\n== 3) 轨迹回放 ==")
        trace = await kernel.service("tracer").replay(result.trace_id)
        if trace:
            for span in trace.get("spans", []):
                print(f"  - {span['name']} ({span.get('duration_ms', '?')}ms)")


if __name__ == "__main__":
    asyncio.run(main())
