"""SQLiteStore：WAL 模式生产级落盘存储（完整借鉴 hermes hermes_state 实践）。

并发模型（关键）：
- **WAL 模式** —— 允许「多读 + 单写」并发，读不阻塞写、写不阻塞读。
- **单写线程** —— 所有写操作经 ``max_workers=1`` 的写线程池串行化，
  彻底规避 SQLite 写锁竞争与 ``database is locked``。
- **读连接池** —— 读操作走多线程池，每线程一条 thread-local 连接。
- 对上层暴露 ``async`` 接口，全部经 ``run_in_executor`` 卸载到线程，
  绝不阻塞事件循环。

后续迁移 Redis：实现同名的 async 方法即可（本类是事实上的 SessionStore
接口），业务层无感切换。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict

from ..core.exceptions import StorageError
from ..core.logging import get_logger
from .schema import SCHEMA_SQL

logger = get_logger(__name__)

Row = Dict[str, Any]


def _row_to_dict(row: sqlite3.Row) -> Row:
    return {key: row[key] for key in row.keys()}


class SQLiteStore:
    """WAL 落盘存储。"""

    def __init__(
        self,
        path: str | Path,
        *,
        wal: bool = True,
        busy_timeout_ms: int = 5000,
        read_workers: int = 4,
    ) -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._wal = wal
        self._busy_timeout = busy_timeout_ms
        self._loop: asyncio.AbstractEventLoop | None = None
        self._write_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fusion-sql-w")
        self._read_pool = ThreadPoolExecutor(
            max_workers=max(1, read_workers), thread_name_prefix="fusion-sql-r",
        )
        self._write_conn: sqlite3.Connection | None = None
        self._read_local = threading.local()
        self._closed = False

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    def _make_conn(self, readonly: bool) -> sqlite3.Connection:
        uri = f"file:{self._path.as_posix()}"
        if readonly:
            uri += "?mode=rw"  # 读连接也需可写权限以启用 WAL 读；不加 immutable
        conn = sqlite3.connect(
            uri, uri=True, timeout=self._busy_timeout / 1000.0,
            check_same_thread=False, isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout}")
        conn.execute("PRAGMA foreign_keys=ON")
        if self._wal:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    def _get_write_conn(self) -> sqlite3.Connection:
        if self._write_conn is None:
            self._write_conn = self._make_conn(readonly=False)
        return self._write_conn

    def _get_read_conn(self) -> sqlite3.Connection:
        conn = getattr(self._read_local, "conn", None)
        if conn is None:
            conn = self._make_conn(readonly=True)
            self._read_local.conn = conn
        return conn

    def _run_write(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        conn = self._get_write_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            result = fn(conn)
            # fn 内的 executescript 等可能已隐式提交，仅在事务仍活跃时显式提交
            if conn.in_transaction:
                conn.execute("COMMIT")
            return result
        except Exception:
            if conn.in_transaction:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise

    def _run_read(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        conn = self._get_read_conn()
        return fn(conn)

    async def _write(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        self._check_open()
        assert self._loop is not None
        try:
            return await self._loop.run_in_executor(self._write_pool, self._run_write, fn)
        except sqlite3.Error as exc:
            raise StorageError(f"写入失败: {exc}") from exc

    async def _read(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        self._check_open()
        assert self._loop is not None
        try:
            return await self._loop.run_in_executor(self._read_pool, self._run_read, fn)
        except sqlite3.Error as exc:
            raise StorageError(f"读取失败: {exc}") from exc

    def _check_open(self) -> None:
        if self._closed:
            raise StorageError("SQLiteStore 已关闭")
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()

        def _init(conn: sqlite3.Connection) -> None:
            conn.executescript(SCHEMA_SQL)

        await self._write(_init)
        logger.info("SQLiteStore 就绪: %s (WAL=%s)", self._path, self._wal)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        # 关闭写连接
        if self._write_conn is not None:
            try:
                self._write_conn.close()
            except sqlite3.Error:
                pass
            self._write_conn = None
        self._write_pool.shutdown(wait=True)
        self._read_pool.shutdown(wait=True)
        logger.info("SQLiteStore 已关闭")

    # ==================================================================
    # 会话与消息
    # ==================================================================
    async def upsert_session(
        self,
        session_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "",
        title: str = "",
        summary: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO sessions
                   (session_id, tenant_id, user_id, title, summary, created_at, updated_at, meta)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     updated_at=excluded.updated_at,
                     title=CASE WHEN excluded.title='' THEN sessions.title ELSE excluded.title END,
                     summary=CASE WHEN excluded.summary='' THEN sessions.summary ELSE excluded.summary END,
                     meta=excluded.meta""",
                (session_id, tenant_id, user_id, title, summary, now, now,
                 json.dumps(meta or {}, ensure_ascii=False)),
            )

        await self._write(_op)

    async def touch_session(self, session_id: str, summary: str | None = None) -> None:
        now = time.time()

        def _op(conn: sqlite3.Connection) -> None:
            if summary is None:
                conn.execute(
                    "UPDATE sessions SET updated_at=? WHERE session_id=?", (now, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at=?, summary=? WHERE session_id=?",
                    (now, summary, session_id),
                )

        await self._write(_op)

    async def get_session(self, session_id: str) -> Row | None:
        def _op(conn: sqlite3.Connection) -> Row | None:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,),
            ).fetchone()
            return _row_to_dict(row) if row else None

        return await self._read(_op)

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        tokens: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> int:
        now = time.time()

        def _op(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                """INSERT INTO messages (session_id, role, content, tokens, created_at, meta)
                   VALUES (?,?,?,?,?,?)""",
                (session_id, role, content, tokens, now,
                 json.dumps(meta or {}, ensure_ascii=False)),
            )
            return int(cur.lastrowid or 0)

        return await self._write(_op)

    async def get_messages(
        self, session_id: str, *, limit: int = 50, offset: int = 0,
    ) -> list[Row]:
        def _op(conn: sqlite3.Connection) -> list[Row]:
            rows = conn.execute(
                """SELECT * FROM messages WHERE session_id=?
                   ORDER BY id DESC LIMIT ? OFFSET ?""",
                (session_id, limit, offset),
            ).fetchall()
            return [_row_to_dict(r) for r in reversed(rows)]

        return await self._read(_op)

    async def count_messages(self, session_id: str) -> int:
        def _op(conn: sqlite3.Connection) -> int:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id=?", (session_id,),
            ).fetchone()
            return int(row["c"]) if row else 0

        return await self._read(_op)

    # ==================================================================
    # 长期记忆
    # ==================================================================
    async def remember(
        self,
        key: str,
        value: str,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
    ) -> None:
        now = time.time()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO long_term_memory
                   (tenant_id, namespace, mem_key, mem_value, hits, created_at, updated_at)
                   VALUES (?,?,?,?,1,?,?)
                   ON CONFLICT(tenant_id, namespace, mem_key) DO UPDATE SET
                     mem_value=excluded.mem_value,
                     hits=long_term_memory.hits+1,
                     updated_at=excluded.updated_at""",
                (tenant_id, namespace, key, value, now, now),
            )

        await self._write(_op)

    async def recall(
        self,
        *,
        tenant_id: str = "default",
        namespace: str = "default",
        limit: int = 20,
    ) -> list[Row]:
        def _op(conn: sqlite3.Connection) -> list[Row]:
            rows = conn.execute(
                """SELECT mem_key, mem_value, hits FROM long_term_memory
                   WHERE tenant_id=? AND namespace=? ORDER BY hits DESC, updated_at DESC
                   LIMIT ?""",
                (tenant_id, namespace, limit),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

        return await self._read(_op)

    # ==================================================================
    # 文档登记（增量索引判重）
    # ==================================================================
    async def record_document(
        self,
        document_id: str,
        *,
        tenant_id: str,
        kb_id: str,
        title: str,
        source: str,
        content_hash: str,
        chunk_count: int,
        meta: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO documents
                   (document_id, tenant_id, kb_id, title, source, content_hash,
                    chunk_count, created_at, updated_at, meta)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(document_id) DO UPDATE SET
                     title=excluded.title, source=excluded.source,
                     content_hash=excluded.content_hash, chunk_count=excluded.chunk_count,
                     updated_at=excluded.updated_at, meta=excluded.meta""",
                (document_id, tenant_id, kb_id, title, source, content_hash,
                 chunk_count, now, now, json.dumps(meta or {}, ensure_ascii=False)),
            )

        await self._write(_op)

    async def get_document(self, document_id: str) -> Row | None:
        def _op(conn: sqlite3.Connection) -> Row | None:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id=?", (document_id,),
            ).fetchone()
            return _row_to_dict(row) if row else None

        return await self._read(_op)

    async def find_document_by_hash(
        self, tenant_id: str, kb_id: str, content_hash: str,
    ) -> Row | None:
        def _op(conn: sqlite3.Connection) -> Row | None:
            row = conn.execute(
                """SELECT * FROM documents
                   WHERE tenant_id=? AND kb_id=? AND content_hash=? LIMIT 1""",
                (tenant_id, kb_id, content_hash),
            ).fetchone()
            return _row_to_dict(row) if row else None

        return await self._read(_op)

    async def list_documents(
        self, tenant_id: str, kb_id: str | None = None,
    ) -> list[Row]:
        def _op(conn: sqlite3.Connection) -> list[Row]:
            if kb_id is None:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE tenant_id=? ORDER BY updated_at DESC",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM documents WHERE tenant_id=? AND kb_id=?
                       ORDER BY updated_at DESC""",
                    (tenant_id, kb_id),
                ).fetchall()
            return [_row_to_dict(r) for r in rows]

        return await self._read(_op)

    async def delete_document(self, document_id: str) -> None:
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM documents WHERE document_id=?", (document_id,))

        await self._write(_op)

    # ==================================================================
    # 问答反馈（问题沉淀）
    # ==================================================================
    async def log_qa(
        self,
        *,
        question: str,
        answer: str,
        tenant_id: str = "default",
        session_id: str = "",
        trace_id: str = "",
        confidence: float = 0.0,
        validated: bool = False,
        latency_ms: int = 0,
        good: bool = True,
    ) -> None:
        now = time.time()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO qa_feedback
                   (tenant_id, session_id, trace_id, question, answer, confidence,
                    validated, latency_ms, good, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (tenant_id, session_id, trace_id, question, answer, confidence,
                 int(validated), latency_ms, int(good), now),
            )

        await self._write(_op)

    async def frequent_questions(
        self, tenant_id: str = "default", limit: int = 20,
    ) -> list[Row]:
        def _op(conn: sqlite3.Connection) -> list[Row]:
            rows = conn.execute(
                """SELECT question, COUNT(*) AS c, AVG(confidence) AS avg_conf
                   FROM qa_feedback WHERE tenant_id=?
                   GROUP BY question ORDER BY c DESC LIMIT ?""",
                (tenant_id, limit),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

        return await self._read(_op)

    async def low_quality_qa(
        self, tenant_id: str = "default", threshold: float = 0.5, limit: int = 50,
    ) -> list[Row]:
        def _op(conn: sqlite3.Connection) -> list[Row]:
            rows = conn.execute(
                """SELECT * FROM qa_feedback
                   WHERE tenant_id=? AND (confidence < ? OR good=0)
                   ORDER BY created_at DESC LIMIT ?""",
                (tenant_id, threshold, limit),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

        return await self._read(_op)

    # ==================================================================
    # 轨迹与指标
    # ==================================================================
    async def save_trace(
        self,
        trace_id: str,
        payload: dict[str, Any],
        *,
        session_id: str = "",
        tenant_id: str = "default",
        question: str = "",
    ) -> None:
        now = time.time()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO traces (trace_id, session_id, tenant_id, question, payload, created_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(trace_id) DO UPDATE SET payload=excluded.payload""",
                (trace_id, session_id, tenant_id, question,
                 json.dumps(payload, ensure_ascii=False), now),
            )

        await self._write(_op)

    async def get_trace(self, trace_id: str) -> Row | None:
        def _op(conn: sqlite3.Connection) -> Row | None:
            row = conn.execute(
                "SELECT * FROM traces WHERE trace_id=?", (trace_id,),
            ).fetchone()
            return _row_to_dict(row) if row else None

        return await self._read(_op)

    async def recent_traces(self, limit: int = 50) -> list[Row]:
        def _op(conn: sqlite3.Connection) -> list[Row]:
            rows = conn.execute(
                "SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

        return await self._read(_op)

    async def save_metrics(self, snapshot: dict[str, Any]) -> None:
        now = time.time()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO metrics (ts, snapshot) VALUES (?,?)",
                (now, json.dumps(snapshot, ensure_ascii=False)),
            )
            # 只保留最近 2000 条快照
            conn.execute(
                """DELETE FROM metrics WHERE id NOT IN
                   (SELECT id FROM metrics ORDER BY id DESC LIMIT 2000)""",
            )

        await self._write(_op)

    # ==================================================================
    # 通用 KV
    # ==================================================================
    async def kv_set(self, key: str, value: Any) -> None:
        now = time.time()

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO kv (kv_key, kv_value, updated_at) VALUES (?,?,?)
                   ON CONFLICT(kv_key) DO UPDATE SET
                     kv_value=excluded.kv_value, updated_at=excluded.updated_at""",
                (key, json.dumps(value, ensure_ascii=False), now),
            )

        await self._write(_op)

    async def kv_get(self, key: str, default: Any = None) -> Any:
        def _op(conn: sqlite3.Connection) -> Any:
            row = conn.execute(
                "SELECT kv_value FROM kv WHERE kv_key=?", (key,),
            ).fetchone()
            return json.loads(row["kv_value"]) if row else default

        return await self._read(_op)
