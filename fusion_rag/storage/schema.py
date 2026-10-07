"""SQLite 表结构 DDL（集中定义，便于版本演进）。

所有表围绕「本地全留存」设计：会话、消息、长期记忆、文档登记、问答
反馈、轨迹、指标快照、通用 KV 均落本地库。
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    user_id      TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    summary      TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    meta         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    tokens       INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    meta         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS long_term_memory (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    namespace    TEXT NOT NULL DEFAULT 'default',
    mem_key      TEXT NOT NULL,
    mem_value    TEXT NOT NULL,
    hits         INTEGER NOT NULL DEFAULT 1,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    UNIQUE(tenant_id, namespace, mem_key)
);
CREATE INDEX IF NOT EXISTS idx_ltm_ns ON long_term_memory(tenant_id, namespace, hits DESC);

CREATE TABLE IF NOT EXISTS documents (
    document_id  TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    kb_id        TEXT NOT NULL DEFAULT 'general',
    title        TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL DEFAULT '',
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    meta         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(tenant_id, kb_id);

CREATE TABLE IF NOT EXISTS qa_feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    session_id   TEXT NOT NULL DEFAULT '',
    trace_id     TEXT NOT NULL DEFAULT '',
    question     TEXT NOT NULL,
    answer       TEXT NOT NULL DEFAULT '',
    confidence   REAL NOT NULL DEFAULT 0,
    validated    INTEGER NOT NULL DEFAULT 0,
    latency_ms   INTEGER NOT NULL DEFAULT 0,
    good         INTEGER NOT NULL DEFAULT 1,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qa_tenant ON qa_feedback(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_conf ON qa_feedback(confidence);

CREATE TABLE IF NOT EXISTS traces (
    trace_id     TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL DEFAULT '',
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    question     TEXT NOT NULL DEFAULT '',
    payload      TEXT NOT NULL DEFAULT '{}',
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at DESC);

CREATE TABLE IF NOT EXISTS metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    snapshot     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS kv (
    kv_key       TEXT PRIMARY KEY,
    kv_value     TEXT NOT NULL,
    updated_at   REAL NOT NULL
);
"""
