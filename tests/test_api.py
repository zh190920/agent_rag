"""HTTP API 测试（需 fastapi + httpx，缺失则跳过）。"""

from __future__ import annotations

import tempfile

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fusion_rag.api.app import create_app  # noqa: E402


@pytest.fixture()
def client():
    tmp = tempfile.mkdtemp(prefix="fusion-api-")
    app = create_app(None, app={"data_dir": tmp, "log_json": False, "log_level": "WARNING"})
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_and_ask_flow(client):
    r = client.post("/v1/documents", json={
        "text": "AgentScope 是多智能体分布式编排框架，支持高并发问答处理。",
        "tenant_id": "default", "kb_id": "general", "title": "as",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] >= 1 and not body["skipped"]

    r = client.post("/v1/ask", json={"question": "AgentScope 支持什么？", "tenant_id": "default"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"].strip()
    assert data["citations"]
    assert data["trace_id"]


def test_idempotent_index(client):
    payload = {
        "text": "OpenClaw 强调本地化隐私与私有知识库隔离。",
        "tenant_id": "default", "kb_id": "general", "title": "oc",
    }
    first = client.post("/v1/documents", json=payload).json()
    second = client.post("/v1/documents", json=payload).json()
    assert not first["skipped"]
    assert second["skipped"] and second["reason"] == "duplicate-hash"


def test_stats_and_metrics(client):
    client.post("/v1/documents", json={
        "text": "DeepAgents 擅长长任务规划与问题拆解。",
        "tenant_id": "default", "kb_id": "general",
    })
    stats = client.get("/v1/stats").json()
    assert stats["documents"] >= 1
    m = client.get("/metrics")
    assert m.status_code == 200
    assert "requests_total" in m.text or "fusion" in m.text.lower() or m.text


def test_trace_replay(client):
    r = client.post("/v1/ask", json={"question": "任意问题", "tenant_id": "default"})
    tid = r.json()["trace_id"]
    tr = client.get(f"/v1/traces/{tid}")
    assert tr.status_code == 200
    assert "spans" in tr.json()


def test_trace_not_found(client):
    assert client.get("/v1/traces/nonexistent").status_code == 404


def test_batch_ask(client):
    r = client.post("/v1/ask/batch", json={"requests": [
        {"question": "AgentScope?"}, {"question": "OpenClaw?"},
    ]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_ask_validation_error(client):
    # question 为空 → pydantic 校验失败 422
    r = client.post("/v1/ask", json={"question": ""})
    assert r.status_code == 422


def test_delete_document(client):
    doc = client.post("/v1/documents", json={
        "text": "临时文档内容，用于删除测试。",
        "tenant_id": "default", "kb_id": "general",
    }).json()
    did = doc["document_id"]
    r = client.delete(f"/v1/documents/{did}")
    assert r.status_code == 200
    assert r.json()["deleted"] == did
