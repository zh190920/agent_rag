"""Windows Proactor 事件循环的 socket transport 优雅关停补丁。

Python 3.8+ 在 Windows 上跑 asyncio（Proactor 策略）时，aiohttp 会命中一个
已知解释器 bug（CPython issue #88050）：`_ProactorBasePipeTransport.__del__`
在 GC 阶段被隐式触发时，会对已经关闭的 event loop 调 `call_soon`，抛
`RuntimeError: Event loop is closed`。异常本身被 Python 吞掉（"Exception
ignored in ..."），但每次 run_kb 结束都会往 stderr 刷一片 Traceback，运维
误以为真的出了问题。

本模块把该 `__del__` 替换为在 loop 已关时静默返回的实现。只在 Windows +
Proactor 策略下生效，其他平台/策略无副作用。

调用点在 :mod:`fusion_rag` 包初始化里；显式幂等，重复 import 只 patch 一次。
"""

from __future__ import annotations

import sys

_PATCHED = False


def apply_windows_proactor_patch() -> None:
    """安装补丁。非 Windows 或补丁已装则无操作。"""
    global _PATCHED
    if _PATCHED or sys.platform != "win32":
        return
    try:
        import asyncio.proactor_events as _proactor
    except ImportError:  # pragma: no cover
        return

    orig_del = _proactor._ProactorBasePipeTransport.__del__

    def _silenced_del(self, _orig=orig_del):  # type: ignore[no-untyped-def]
        # loop 已关 → 直接返回；未关 → 走原逻辑（保持正常关闭时序）
        loop = getattr(self, "_loop", None)
        if loop is not None and loop.is_closed():
            return
        try:
            _orig(self)
        except RuntimeError:
            # 兜底：万一 is_closed() 判定后 loop 又关了，异常继续被吞掉
            pass

    _proactor._ProactorBasePipeTransport.__del__ = _silenced_del
    _PATCHED = True
