"""PathJail：把工具的所有路径操作限制在授权目录内（隐私安全纵深防御）。

吸收 hermes ``file_tools._is_blocked_device_path`` 与 claw-code ``path_scope``
的思路：任何进入文件工具的路径都先经 :meth:`PathJail.resolve` 归一化并校验，
确保最终真实路径（含软链解析后）仍落在某个授权 root 之内，否则抛
:class:`~fusion_rag.core.exceptions.AccessDeniedError`。

拦截项：
- ``..`` 目录穿越 / 绝对路径越界
- 软链接逃逸（``os.path.realpath`` 解析后复核）
- 设备与特殊文件（``/dev`` ``/proc`` ``/sys``、Windows ``\\\\.\\`` 设备命名空间）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from ..core.exceptions import AccessDeniedError
from ..constants import PROBE_SUFFIXES

# 类 Unix 设备/内核伪文件系统前缀
_BLOCKED_PREFIXES = ("/dev/", "/proc/", "/sys/")
# Windows 设备命名空间前缀
_BLOCKED_WIN_PREFIXES = ("\\\\.\\", "\\\\?\\")
# Windows 保留设备名（无扩展名主干）
_BLOCKED_WIN_NAMES = {"con", "prn", "aux", "nul"}


class PathJail:
    """授权目录 jail。一个 jail 对应一个租户在本次请求中可访问的根集合。"""

    def __init__(
        self, roots: Iterable[os.PathLike | str], *, tenant_id: str = "default",
    ) -> None:
        self.tenant_id = tenant_id
        self._roots: list[Path] = []
        seen: set[str] = set()
        for raw in roots:
            try:
                resolved = Path(os.path.realpath(str(Path(raw).expanduser())))
            except OSError:  # pragma: no cover - 极端非法路径
                continue
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                self._roots.append(resolved)

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)

    def __bool__(self) -> bool:
        return bool(self._roots)

    # ------------------------------------------------------------------
    def resolve(
        self, raw_path: os.PathLike | str, *, base_hint: os.PathLike | str | None = None,
    ) -> Path:
        """把用户/模型给出的路径解析为 jail 内的真实绝对路径。

        Args:
            raw_path: 相对或绝对路径。相对路径以 ``base_hint`` 或首个 root 为基准。
            base_hint: 相对路径的基准目录（通常是某个具体知识库根）。

        Raises:
            AccessDeniedError: 无授权根、越界、软链逃逸或命中设备文件。
        """
        if not self._roots:
            raise AccessDeniedError(self.tenant_id, str(raw_path))

        candidate = Path(str(raw_path)).expanduser()
        if not candidate.is_absolute():
            base = self._resolve_base(base_hint)
            candidate = base / candidate

        real = Path(os.path.realpath(str(candidate)))
        if _is_blocked_device(str(real)):
            raise AccessDeniedError(self.tenant_id, str(raw_path))
        for root in self._roots:
            if _is_within(real, root):
                return real
        raise AccessDeniedError(self.tenant_id, str(raw_path))

    def search_bases(
        self, raw_path: os.PathLike | str | None,
    ) -> list[Path]:
        """为搜索类工具（grep/glob/list_dir/find）计算遍历基准目录集合。

        - ``raw_path`` 为空或 ``"."`` 时返回全部授权根；
        - 传入**相对子目录**时，跨**所有** root 探测该子目录，合并存在且仍在 jail
          内的目录（实现多知识库目录的统一检索）；若均不存在则回落到
          :meth:`resolve` 以保留越权拦截语义；
        - **绝对路径**统一交给 :meth:`resolve`（越界即拒绝），只返回单一目录。
        """
        text = str(raw_path or "").strip()
        if text in ("", ".", "./"):
            return list(self._roots)

        candidate = Path(text)
        if candidate.is_absolute():
            return [self.resolve(text)]

        # 智能映射：模型常把 root 自身的相对路径（如 "datas/md"）当参数传入。
        # 若 text 正好是某个 root 相对当前工作目录的后缀，则直接指向该 root，
        # 避免默认拼接 root+text 产生不存在的目录导致"扫描 0 文件"隐性失败。
        norm = text.replace("\\", "/").strip("/")
        if norm:
            cwd_parts = Path(os.getcwd()).resolve().parts
            for root in self._roots:
                root_posix = str(root).replace("\\", "/").strip("/")
                if root_posix.endswith("/" + norm) or root_posix == norm:
                    return [root]
                # 也兼容从项目根开始的相对路径（例如 root=…/fusion-rag-agent/datas/md，
                # 模型传 "fusion-rag-agent/datas/md"）
                rel_to_cwd = _relative_to_cwd(root, cwd_parts)
                if rel_to_cwd and (rel_to_cwd == norm or rel_to_cwd.endswith("/" + norm)):
                    return [root]

        merged: list[Path] = []
        seen: set[str] = set()
        for root in self._roots:
            probe = root / candidate
            try:
                real = Path(os.path.realpath(str(probe)))
            except OSError:
                continue
            key = str(real)
            if key in seen:
                continue
            if not real.is_dir():
                continue
            if not any(_is_within(real, r) for r in self._roots):
                continue  # 越界（如 ../..）跳过，交由下方回落报拒绝
            seen.add(key)
            merged.append(real)
        if merged:
            return merged
        # 无任何 root 命中该子目录：回落 resolve，触发越权/不存在等正确异常
        return [self.resolve(text)]

    def _strip_redundant_root_prefix(self, candidate: Path) -> Path | None:
        """若 candidate 尾部路径与 root 目录名重叠，剥离冗余前缀。

        e.g. root="/data/datas/md", candidate="datas/md/file.md"
        → 返回 Path("file.md")。
        """
        parts = candidate.parts
        for root in self._roots:
            root_parts = root.parts
            # root 的最后 N 段与 candidate 的前 N 段相同？
            for n in range(min(len(root_parts), len(parts)), 0, -1):
                if parts[:n] == root_parts[-n:]:
                    remainder = Path(*parts[n:]) if n < len(parts) else None
                    return remainder
        return None

    def resolve_first_existing(self, raw_path: os.PathLike | str) -> Path:
        """相对路径**跨所有授权根**探测，返回首个存在的真实路径。

        与 :meth:`search_bases` 的多根合并语义对齐，供**单文件**读取（read_file）
        使用：模型从渐进披露的技能清单（或多知识库根）拿到的相对路径，未必落在
        ``roots[0]`` 下。逐一在各 root 内解析，命中第一个存在者即返回；都不存在
        时回落到 :meth:`resolve`（保留越权拒绝与「文件不存在」的正常报错语义）。
        **绝对路径**统一交给 :meth:`resolve`（越界即拒绝）。

        ★ 若模型忘了写扩展名（LLM 常见），自动尝试补全 .md / .txt 等已知后缀。
        """
        candidate = Path(str(raw_path))
        if not candidate.is_absolute():
            for root in self._roots:
                probe = root / candidate
                try:
                    real = Path(os.path.realpath(str(probe)))
                except OSError:
                    continue
                if _is_blocked_device(str(real)):
                    continue
                if not any(_is_within(real, r) for r in self._roots):
                    continue
                if real.exists():
                    return real
            # ★ 容错：模型可能把授权根目录名当作路径前缀重复拼接
            #   （e.g. root="datas/md" + candidate="datas/md/file.md"）
            #   尝试剥离冗余前缀后重新解析。
            stripped = self._strip_redundant_root_prefix(candidate)
            if stripped is not None:
                for root in self._roots:
                    probe = root / stripped
                    try:
                        real = Path(os.path.realpath(str(probe)))
                    except OSError:
                        continue
                    if _is_blocked_device(str(real)):
                        continue
                    if not any(_is_within(real, r) for r in self._roots):
                        continue
                    if real.exists():
                        return real
            # ★ 无扩展名时自动补全常见后缀（仅当 candidate.suffix == ""）
            if candidate.suffix == "":
                for ext in PROBE_SUFFIXES:
                    for root in self._roots:
                        probe = root / (str(candidate) + ext)
                        try:
                            real = Path(os.path.realpath(str(probe)))
                        except OSError:
                            continue
                        if _is_blocked_device(str(real)):
                            continue
                        if not any(_is_within(real, r) for r in self._roots):
                            continue
                        if real.is_file():
                            return real
        return self.resolve(raw_path)

    def relative(self, path: os.PathLike | str) -> str:
        """把绝对路径转为相对最近授权根的展示路径（失败则返回原字符串）。

        统一以 POSIX 分隔符（``/``）输出，保证跨平台一致的展示与引用回链。
        """
        p = Path(str(path))
        best: str | None = None
        for root in self._roots:
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                continue
            if best is None or len(rel) < len(best):
                best = rel
        return best if best is not None else str(path)

    # ------------------------------------------------------------------
    def _resolve_base(self, base_hint: os.PathLike | str | None) -> Path:
        if base_hint is not None:
            hint = Path(str(base_hint)).expanduser()
            if hint.is_absolute():
                real = Path(os.path.realpath(str(hint)))
                for root in self._roots:
                    if _is_within(real, root):
                        return real
            else:
                return self._roots[0] / hint
        return self._roots[0]


def _is_within(child: Path, parent: Path) -> bool:
    """child 是否位于 parent 之内（含相等）。兼容 Python 3.8。"""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _relative_to_cwd(root: Path, cwd_parts: tuple[str, ...]) -> str | None:
    """将 root 相对于当前工作目录展开成 POSIX 风格相对路径。root 不在 cwd 下时返回 None。"""
    parts = root.parts
    if len(parts) < len(cwd_parts) or parts[: len(cwd_parts)] != cwd_parts:
        return None
    rel = "/".join(parts[len(cwd_parts):])
    return rel or None


def _is_blocked_device(path: str) -> bool:
    """拦截设备/特殊文件（吸收 hermes _is_blocked_device_path）。"""
    low = path.lower().replace("\\", "/")
    if any(low.startswith(pre.replace("\\", "/")) for pre in _BLOCKED_PREFIXES):
        return True
    raw_low = path.lower()
    if any(raw_low.startswith(pre) for pre in _BLOCKED_WIN_PREFIXES):
        return True
    stem = os.path.basename(raw_low).split(".")[0]
    if stem in _BLOCKED_WIN_NAMES:
        return True
    # COM1..COM9 / LPT1..LPT9
    if len(stem) == 4 and stem[:3] in ("com", "lpt") and stem[3].isdigit():
        return True
    return False
