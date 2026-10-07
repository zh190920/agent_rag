"""知识增量索引测试：去重、更新、删除、目录批量、脱敏。"""

from __future__ import annotations

from conftest import make_kernel, run


def test_index_and_dedup_by_hash():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            r1 = await kernel.indexer.add_text(
                "AgentScope 支持多智能体分布式编排与高并发问答。",
                tenant_id="default", kb_id="general", title="as",
            )
            assert not r1.skipped and r1.chunks >= 1
            # 相同内容再次索引 → 按 content_hash 跳过（幂等）
            r2 = await kernel.indexer.add_text(
                "AgentScope 支持多智能体分布式编排与高并发问答。",
                tenant_id="default", kb_id="general", title="as",
            )
            assert r2.skipped and r2.reason == "duplicate-hash"
            stats = await kernel.indexer.stats()
            assert stats["documents"] == 1
            return stats

    run(scenario)


def test_force_reindex_updates_document():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            r1 = await kernel.indexer.add_text(
                "旧内容：框架 A 的能力说明。", tenant_id="default",
                kb_id="general", title="doc", document_id="fixed-id",
            )
            r2 = await kernel.indexer.add_text(
                "新内容：框架 B 的更新能力说明，更加详细。", tenant_id="default",
                kb_id="general", title="doc", document_id="fixed-id", force=True,
            )
            assert not r2.skipped
            stats = await kernel.indexer.stats()
            # 同 document_id 覆盖，不产生重复文档
            assert stats["documents"] == 1

    run(scenario)


def test_delete_document_removes_indexes():
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            r = await kernel.indexer.add_text(
                "待删除的临时知识内容。", tenant_id="default", kb_id="general",
            )
            assert (await kernel.indexer.stats())["documents"] == 1
            await kernel.indexer.delete_document(r.document_id)
            assert (await kernel.indexer.stats())["documents"] == 0

    run(scenario)


def test_ingest_directory(tmp_path):
    async def scenario():
        d = tmp_path / "kb"
        d.mkdir()
        (d / "a.md").write_text("# A\nAgentScope 多智能体编排框架。", encoding="utf-8")
        (d / "b.txt").write_text("OpenClaw 本地化隐私留存。", encoding="utf-8")
        (d / "skip.bin").write_bytes(b"\x00\x01binary")

        kernel = make_kernel()
        async with kernel:
            results = await kernel.indexer.ingest_directory(
                str(d), tenant_id="default", kb_id="general",
            )
            indexed = [r for r in results if not r.skipped]
            assert len(indexed) == 2, "只应索引 .md/.txt 两个文本文件"
            stats = await kernel.indexer.stats()
            assert stats["documents"] == 2

    run(scenario)


def test_redaction_before_index():
    """入库前应脱敏，敏感串不得进入可检索内容。"""
    async def scenario():
        kernel = make_kernel()
        async with kernel:
            await kernel.indexer.add_text(
                "运维联系人手机号 13812345678 负责知识库维护。",
                tenant_id="default", kb_id="general", title="contact",
            )
            hits = await kernel.service("retriever").retrieve(
                "手机号", top_k=5, tenant_id="default", kb_ids=["general"],
            )
            assert hits, "应能检索到该文档"
            for h in hits:
                assert "13812345678" not in h.content, "原文手机号不应留存"

    run(scenario)


def test_cross_tenant_write_isolation():
    async def scenario():
        kernel = make_kernel(**{"security__tenants__acme__kbs": ["hr"]})
        async with kernel:
            await kernel.indexer.add_text(
                "ACME 公司的 HR 政策内容。", tenant_id="acme", kb_id="hr", title="hr",
            )
            # default 租户检索不到 acme 的内容
            hits = await kernel.service("retriever").retrieve(
                "HR 政策", top_k=5, tenant_id="default", kb_ids=["general"],
            )
            assert all(h.tenant_id != "acme" for h in hits)

    run(scenario)
