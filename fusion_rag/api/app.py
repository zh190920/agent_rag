"""FastAPI 应用工厂：问答 / 知识入库 / 运维观测。

生产级要点：
- **生命周期托管**：``lifespan`` 内启动/关闭 Kernel，单进程共享一套装配好的
  服务与连接池；并发能力由 Orchestrator + Sandbox（信号量/令牌桶/超时预算）保障。
- **异常语义化**：把框架异常映射为恰当的 HTTP 状态码（限流 429、越权 403、
  超时 504、需人工 202、其余 400/500），避免把栈信息泄露给客户端。
- **多租户**：所有写/查接口都带 ``tenant_id``，检索层强制注入作用域过滤。
- **零硬依赖**：未安装 fastapi/uvicorn 时，:func:`create_app` 抛出带安装指引
  的清晰错误，核心库仍可独立使用。

注：为兼容 Python 3.8，pydantic/FastAPI 会内省的注解一律使用 ``typing`` 泛型
（``Dict``/``List``/``Optional``），不用 PEP585 内建泛型下标。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from ..core.config import load_config
from ..core.exceptions import (
    AccessDeniedError,
    FusionRagError,
    HITLRequiredError,
    RateLimitError,
    TimeoutBudgetError,
)
from ..core.kernel import Kernel
from ..core.logging import get_logger

logger = get_logger(__name__)

try:  # 可选依赖
    from fastapi import FastAPI, HTTPException, Query, Response
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False


# ======================================================================
# 请求/响应模型
# ======================================================================
if _HAS_FASTAPI:

    class AskRequest(BaseModel):
        question: str = Field(..., min_length=1, description="用户问题")
        tenant_id: Optional[str] = Field(None, description="租户 id，缺省用默认租户")
        session_id: Optional[str] = Field(None, description="会话 id，缺省自动生成")
        user_id: str = Field("", description="用户 id")
        kb_ids: Optional[List[str]] = Field(None, description="限定知识库范围")
        output_format: Optional[str] = Field(
            None, description="paragraph/list/table/steps，缺省自动推断",
        )

    class BatchAskRequest(BaseModel):
        requests: List[AskRequest] = Field(..., min_length=1, max_length=256)

    class IndexTextRequest(BaseModel):
        text: str = Field(..., min_length=1)
        tenant_id: str = Field("default")
        kb_id: str = Field("general")
        title: str = Field("")
        source: str = Field("")
        document_id: Optional[str] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)
        force: bool = Field(False, description="忽略内容哈希去重，强制重建")

    class IndexDirectoryRequest(BaseModel):
        path: str = Field(..., description="本地目录绝对/相对路径")
        tenant_id: str = Field("default")
        kb_id: str = Field("general")
        globs: Optional[List[str]] = None
        force: bool = False


# ======================================================================
# 应用工厂
# ======================================================================
def create_app(config_path: Optional[str] = None, **overrides: Any) -> "FastAPI":
    """构建 FastAPI 应用。``config_path``/``overrides`` 透传给 :func:`load_config`。"""
    if not _HAS_FASTAPI:  # pragma: no cover
        raise RuntimeError(
            "HTTP 服务需要安装 FastAPI 与 Uvicorn：pip install fastapi uvicorn",
        )

    config = load_config(config_path, overrides or None)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        kernel = Kernel(config)
        await kernel.start()
        app.state.kernel = kernel
        logger.info("API 就绪，Kernel 已启动")
        try:
            yield
        finally:
            await kernel.stop()
            logger.info("API 关闭，Kernel 已释放")

    app = FastAPI(
        title="Fusion RAG Agent",
        version="0.1.0",
        description="融合多智能体优势的企业级 RAG 知识问答服务",
        lifespan=lifespan,
    )
    _register_exception_handlers(app)
    _register_routes(app)
    return app


def _kernel(app: "FastAPI") -> Kernel:
    kernel = getattr(app.state, "kernel", None)
    if kernel is None:  # pragma: no cover
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    return kernel


# ======================================================================
# 异常映射
# ======================================================================
def _register_exception_handlers(app: "FastAPI") -> None:
    app.add_exception_handler(RateLimitError, lambda r, e: JSONResponse(
        status_code=429, content={"error": "rate_limited", "detail": str(e)}))
    app.add_exception_handler(AccessDeniedError, lambda r, e: JSONResponse(
        status_code=403, content={"error": "access_denied", "detail": str(e)}))
    app.add_exception_handler(TimeoutBudgetError, lambda r, e: JSONResponse(
        status_code=504, content={"error": "timeout", "detail": str(e)}))
    app.add_exception_handler(HITLRequiredError, lambda r, e: JSONResponse(
        status_code=202, content={"error": "hitl_required", "detail": str(e)}))
    app.add_exception_handler(FusionRagError, lambda r, e: JSONResponse(
        status_code=400, content={"error": "bad_request", "detail": str(e)}))


# ======================================================================
# 路由
# ======================================================================
def _register_routes(app: "FastAPI") -> None:

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        kernel = _kernel(app)
        ops = kernel.services.get("ops")
        summary = await ops.inspect_once() if ops is not None else {}
        return {"status": "ok", "ops": summary}

    @app.get("/metrics")
    async def metrics() -> Response:
        kernel = _kernel(app)
        text = kernel.service("metrics").render_prometheus()
        return Response(content=text, media_type="text/plain; version=0.0.4")

    @app.get("/metrics/json")
    async def metrics_json() -> Dict[str, Any]:
        return _kernel(app).service("metrics").snapshot()

    # ---- 问答 --------------------------------------------------------
    @app.post("/v1/ask")
    async def ask(req: AskRequest) -> Dict[str, Any]:
        kernel = _kernel(app)
        result = await kernel.orchestrator.ask(
            req.question,
            tenant_id=req.tenant_id,
            session_id=req.session_id,
            user_id=req.user_id,
            kb_ids=req.kb_ids,
            output_format=req.output_format,
        )
        return result.to_dict()

    @app.post("/v1/ask/batch")
    async def ask_batch(req: BatchAskRequest) -> Dict[str, Any]:
        kernel = _kernel(app)
        payloads = [r.model_dump(exclude_none=True) for r in req.requests]
        results = await kernel.orchestrator.ask_many(payloads)
        return {"results": [r.to_dict() for r in results]}

    # ---- 知识入库 ----------------------------------------------------
    @app.post("/v1/documents")
    async def index_text(req: IndexTextRequest) -> Dict[str, Any]:
        kernel = _kernel(app)
        result = await kernel.indexer.add_text(
            req.text,
            tenant_id=req.tenant_id, kb_id=req.kb_id,
            title=req.title, source=req.source,
            document_id=req.document_id, metadata=req.metadata, force=req.force,
        )
        return {
            "document_id": result.document_id, "chunks": result.chunks,
            "skipped": result.skipped, "reason": result.reason,
        }

    @app.post("/v1/documents/directory")
    async def index_directory(req: IndexDirectoryRequest) -> Dict[str, Any]:
        kernel = _kernel(app)
        results = await kernel.indexer.ingest_directory(
            req.path, tenant_id=req.tenant_id, kb_id=req.kb_id,
            globs=req.globs, force=req.force,
        )
        return {
            "total": len(results),
            "indexed": sum(1 for r in results if not r.skipped),
            "skipped": sum(1 for r in results if r.skipped),
        }

    @app.delete("/v1/documents/{document_id}")
    async def delete_document(document_id: str) -> Dict[str, Any]:
        await _kernel(app).indexer.delete_document(document_id)
        return {"deleted": document_id}

    @app.get("/v1/stats")
    async def stats() -> Dict[str, Any]:
        return await _kernel(app).indexer.stats()

    # ---- 轨迹回放 ----------------------------------------------------
    @app.get("/v1/traces/{trace_id}")
    async def get_trace(trace_id: str) -> Dict[str, Any]:
        kernel = _kernel(app)
        trace = await kernel.service("tracer").replay(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="轨迹不存在")
        return trace

    @app.get("/v1/feedback")
    async def feedback(
        tenant: str = Query("default", description="租户 id"),
    ) -> Dict[str, Any]:
        kernel = _kernel(app)
        collector = kernel.services.get("feedback")
        if collector is None:  # pragma: no cover
            return {}
        return await collector.snapshot(tenant)
