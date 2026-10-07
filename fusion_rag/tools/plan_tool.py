"""问题分解式规划：会话内的“要解决哪些子问题”清单 + 模型可调的 ``plan_update`` 工具。

对标成熟智能体框架的 todo 机制（hermes ``todo_list`` / Claude Code ``TodoWrite``
/ agentscope ``TaskCreate``），但定位不同：这不是“必须逐条打勾才能交差”的任务表，
而是一份**帮模型想清楚“要回答本问题需要厘清哪些子问题”的导航**。模型据此：

  1. 用 ``plan_update`` 把复合问题拆成若干【待厘清前提 / 子问题】；
  2. 每轮主循环会把这份分解作为参考回显，驱动模型自主决定下一步查什么；
  3. 随理解加深，可随时用 merge=true 增/删/改子问题——它是活文档，不是定稿。

**不设完成门**：证据够了模型随时可直接作答，不必把所有项都勾成 completed。
status 只是可选的进度标记（供模型自己记账），不影响能否收工。设计：RAG 的子
问题扁平、不做 parent 嵌套；每次调用返回全量清单 + 进度摘要。状态是纯内存、无副作用，
工具本身无状态（清单实例经 :class:`ToolContext.plan_store` 注入，一次问答一份）。
"""

from __future__ import annotations

import json
from typing import Any

from .base import Tool, ToolContext, ToolResult

#: 合法状态机（对齐 Claude Code / hermes）
VALID_STATUSES = ("pending", "in_progress", "completed", "cancelled")
#: 仍"未完成"的状态——完成门据此判断能否收工
ACTIVE_STATUSES = ("pending", "in_progress")
#: 清单条数与单条正文的硬上限，防模型灌爆上下文
MAX_ITEMS = 32
MAX_CONTENT_CHARS = 500
_TRUNC = "…"


class PlanStore:
    """会话内规划清单。``items`` 为 ``{id, content, status}``，列表顺序即优先级。"""

    def __init__(self) -> None:
        self._items: list[dict[str, str]] = []

    # -- 写 --------------------------------------------------------------
    def write(
        self, todos: list[dict[str, Any]], *, merge: bool = False,
    ) -> list[dict[str, str]]:
        """merge=True 按 id 字段级增量更新；否则整表替换。返回写后的全量清单。"""
        if merge:
            by_id = {it["id"]: it for it in self._items}
            for t in self._dedupe(todos):
                if not isinstance(t, dict):
                    continue
                item_id = str(t.get("id", "")).strip()
                if not item_id:
                    continue
                cur = by_id.get(item_id)
                if cur is None:
                    validated = self._validate(t)
                    self._items.append(validated)
                    by_id[item_id] = validated
                    continue
                # 只更新显式提供的字段，未给的 content 保持原值（不清空）。
                content = str(t.get("content", "")).strip()
                if content:
                    cur["content"] = self._cap(content)
                status = str(t.get("status", "")).strip().lower()
                if status in VALID_STATUSES:
                    cur["status"] = status
            del self._items[MAX_ITEMS:]
        else:
            self._items = [self._validate(t) for t in self._dedupe(todos)][:MAX_ITEMS]
        return self.read()

    # -- 读 --------------------------------------------------------------
    def read(self) -> list[dict[str, str]]:
        return [dict(it) for it in self._items]

    def is_empty(self) -> bool:
        return not self._items

    def active_items(self) -> list[dict[str, str]]:
        return [it for it in self._items if it["status"] in ACTIVE_STATUSES]

    def all_resolved(self) -> bool:
        """清单非空且没有任何 pending/in_progress 项 → 可收工。"""
        return bool(self._items) and not self.active_items()

    def counts(self) -> dict[str, int]:
        c = {"total": len(self._items)}
        for s in VALID_STATUSES:
            c[s] = sum(1 for it in self._items if it["status"] == s)
        return c

    # -- 渲染 ------------------------------------------------------------
    def render(self, *, only_active: bool = False) -> str:
        """把清单渲染成带状态标记的多行文本，供注入/展示。

        only_active=True 只保留未完成项（用于每轮提醒，避免重复已完成项）。
        """
        marks = {
            "completed": "[x]", "in_progress": "[>]",
            "pending": "[ ]", "cancelled": "[~]",
        }
        lines: list[str] = []
        for it in self._items:
            if only_active and it["status"] not in ACTIVE_STATUSES:
                continue
            lines.append(
                f"{marks.get(it['status'], '[?]')} #{it['id']} "
                f"{it['content']}（{it['status']}）"
            )
        return "\n".join(lines)

    # -- 内部 ------------------------------------------------------------
    @staticmethod
    def _cap(content: str) -> str:
        """截断到字符上限，保留头部（可执行部分）+ 标记。"""
        if len(content) > MAX_CONTENT_CHARS:
            return content[: MAX_CONTENT_CHARS - 1] + _TRUNC
        return content

    @staticmethod
    def _validate(item: Any) -> dict[str, str]:
        if not isinstance(item, dict):
            return {"id": "?", "content": "(无效项)", "status": "pending"}
        item_id = str(item.get("id", "")).strip() or "?"
        content = str(item.get("content", "")).strip() or "(无描述)"
        content = PlanStore._cap(content)
        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"
        return {"id": item_id, "content": content, "status": status}

    @staticmethod
    def _dedupe(todos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """同 id 只保留最后一次出现（保持其出现顺序）。"""
        if not isinstance(todos, list):
            return []
        last: dict[str, int] = {}
        for i, item in enumerate(todos):
            key = str(item.get("id", "")).strip() if isinstance(item, dict) else f"__x{i}"
            last[key or "?"] = i
        return [todos[i] for i in sorted(last.values())]


class PlanUpdateTool(Tool):
    """模型在 agentic 循环里维护规划清单的唯一入口：传 ``todos`` 写、省略则读。"""

    name = "plan_update"
    description = (
        "维护本问题的子问题清单（活文档）。首轮传 todos=[{id,content,status:'pending'},...] 写入；"
        "修改时 merge=true + 目标 id；省略 todos 则读取当前清单。证据够即可作答，不要求逐条完成。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "要写入的清单项（省略表示只读）",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "稳定标识"},
                        "content": {"type": "string", "description": "子问题内容"},
                        "status": {
                            "type": "string",
                            "enum": list(VALID_STATUSES),
                        },
                    },
                    "required": ["id", "content", "status"],
                },
            },
            "merge": {
                "type": "boolean",
                "description": "true=按 id 增量；false=整表替换",
                "default": False,
            },
        },
        "required": [],
    }
    best_for = ("复合/多前提问题的拆解与进度跟踪",)
    avoid_for: tuple[str, ...] = ()
    applies_to_domains: tuple[str, ...] = ()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        store: PlanStore | None = getattr(ctx, "plan_store", None)
        if store is None:
            return ToolResult(
                content="plan_update 不可用：当前运行未启用规划清单。",
                is_error=True,
            )
        todos = args.get("todos")
        try:
            if todos is None:
                items = store.read()
            elif not isinstance(todos, list):
                return ToolResult(
                    content=f"todos 必须是数组，收到 {type(todos).__name__}。",
                    is_error=True,
                )
            else:
                items = store.write(todos, merge=bool(args.get("merge", False)))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"plan_update 失败：{exc}", is_error=True)
        summary = store.counts()
        return ToolResult(
            content=json.dumps(
                {
                    "todos": items,
                    "summary": summary,
                    "all_resolved": store.all_resolved(),
                    "checklist": store.render(),
                },
                ensure_ascii=False,
            ),
            meta={"plan_count": summary.get("total", 0)},
        )
