"""HTTP 接入层（FastAPI）。

对外暴露问答、知识入库、运维观测三类接口，是「用户交互层」的服务端实现。
Kernel 生命周期由 FastAPI ``lifespan`` 托管：进程启动即装配、关闭即优雅释放。
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
