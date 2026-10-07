#!/usr/bin/env python3
"""一键跑通 PDF 知识库：转换 → 入库 → 问答（带完整流程轨迹可视化）。

用法::

    python scripts/run_kb.py "easy320是否支持fins"
    python scripts/run_kb.py --reindex "H5U 定时器怎么用"
    python scripts/run_kb.py --quiet "问题"          # 只看答案，不看流程
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Windows 下确保 stdout/stderr 使用 UTF-8，避免 GBK 报错
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---- 路径常量 ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PYTHON = sys.executable
DATAS_DIR = PROJECT_DIR / "datas"
MD_DIR = DATAS_DIR / "md"
RAG_DATA = DATAS_DIR / ".rag-data"
KB_NAME = "general"


# ==================================================================
# 通用工具
# ==================================================================
def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(cmd: list[str], *, label: str = "", capture: bool = False) -> tuple[int, str]:
    """执行子进程；capture=True 时返回 stdout 内容。"""
    display = " ".join(cmd[:5]) + (" ..." if len(cmd) > 5 else "")
    if label:
        print(f"\n[{label}] $ {display}")
    else:
        print(f"$ {display}")
    if capture:
        result = subprocess.run(
            cmd, env=_env(), cwd=str(PROJECT_DIR),
            capture_output=True, text=True, encoding="utf-8",
        )
        return result.returncode, result.stdout or ""
    rc = subprocess.call(cmd, env=_env(), cwd=str(PROJECT_DIR))
    return rc, ""


def _looks_like_json_log(line: str) -> bool:
    """识别 JSON 日志行，以便在默认模式下过滤降噪。"""
    s = line.strip()
    return s.startswith('{"ts":') or s.startswith('{"level":')


def _truncate(text: str, n: int = 200) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + f"...(共{len(text)}字)"


def _box(title: str, content: str = "", color: str = "━") -> None:
    """打印带标题的方框。"""
    print(f"\n┌─ {title} " + "─" * max(0, 60 - len(title)))
    if content:
        for line in content.splitlines():
            print(f"│ {line}")
    print("└" + "─" * 62)


# ==================================================================
# 步骤 1、2（转换、索引）
# ==================================================================
def step1_convert(force: bool) -> bool:
    if not force and MD_DIR.is_dir() and any(MD_DIR.glob("*.md")):
        print("[Step 1/3] PDF → MD: 已存在，跳过。")
        return True
    pdfs = list(DATAS_DIR.glob("*.pdf"))
    if not pdfs:
        print("[Step 1/3] 无 PDF，跳过。")
        return True
    print(f"[Step 1/3] 转换 {len(pdfs)} 个 PDF → Markdown ...")
    rc, _ = _run([PYTHON, str(SCRIPT_DIR / "pdf2md.py"), str(DATAS_DIR)], label="pdf2md")
    return rc == 0


def step2_index(config: str, force: bool) -> bool:
    db_file = RAG_DATA / "state" / "fusion_rag.db"
    if not force and db_file.exists():
        print("[Step 2/3] 索引已存在，跳过（加 --reindex 强制重建）。")
        return True
    if not MD_DIR.is_dir():
        print("[Step 2/3] MD 目录不存在。", file=sys.stderr)
        return False
    print(f"[Step 2/3] 索引 {len(list(MD_DIR.glob('*.md')))} 个 MD ...")
    cmd = [
        PYTHON, "-m", "fusion_rag.cli", "index-dir", str(MD_DIR),
        "--kb", KB_NAME, "--data-dir", str(RAG_DATA), "--globs", "**/*.md",
    ]
    if config and Path(config).exists():
        cmd += ["--config", config]
    if force:
        cmd.append("--force")
    rc, _ = _run(cmd, label="index-dir")
    return rc == 0


# ==================================================================
# 步骤 3：问答 + 流程可视化
# ==================================================================
def _render_trace(trace: dict[str, Any]) -> None:
    """把 replay 输出的 trace 渲染成流程树。"""
    spans = trace.get("spans", [])
    tags = trace.get("tags", {}) or {}

    # 建父子索引
    children: dict[str, list[dict]] = {}
    for s in spans:
        children.setdefault(s.get("parent_id", ""), []).append(s)

    # 阶段中文名
    stage_label = {
        "intent.classify": "🧠 意图识别",
        "plan.decompose": "📋 任务规划",
        "memory.build_context": "💾 记忆装配",
        "retrieval.retrieve": "🔍 混合检索 (BM25 + Vector + Rerank)",
        "retrieval.rerank": "   ↳ Reranker 精排",
        "reasoning.reason": "✍️  LLM 推理生成",
        "validator.validate": "✅ 校验评估",
        "tool_agent.run": "🤖 Agentic 工具循环",
        "tool_agent.iteration": "   ↳ 迭代",
        "llm.chat": "      · LLM 调用",
    }

    def walk(span: dict, depth: int = 0) -> None:
        name = span.get("name", "")
        attrs = span.get("attributes", {}) or {}
        if name.startswith("tool_agent.iteration"):
            iter_n = attrs.get("n", "?")
            print(f"\n  ▶ Agentic 迭代 #{iter_n}")
        elif name.startswith("tool."):
            tool = name.replace("tool.", "", 1)
            err = "❌" if attrs.get("is_error") else "✅"
            size = attrs.get("size", 0)
            extra = ""
            if attrs.get("timeout"):
                extra = "  ⏱超时"
            elif attrs.get("error"):
                extra = f"  ⚠ {attrs['error']}"
            print(
                f"    {err} {tool:<12} [{span.get('duration_ms', 0)}ms] "
                f"→ {size} 字节{extra}"
            )
            # 展示工具入参摘要（从 ToolAgent 记录的 transcript 中取）
            args = attrs.get("args_summary") or attrs.get("args") or {}
            if args:
                arg_line = ", ".join(
                    f"{k}={_truncate(str(v), 40)}" for k, v in args.items()
                )
                print(f"       ↳ {arg_line}")
            preview = attrs.get("preview")
            if preview:
                print(f"       ↳ 回包：{_truncate(str(preview), 160)}")
        else:
            label = stage_label.get(name, name)
            dur = span.get("duration_ms", 0)
            print(f"\n{'  ' * depth}{label}  [{dur}ms]")
            for k in ("domain", "risk", "sub_questions", "evidence", "confidence",
                      "validated", "issues", "iterations", "tool_calls",
                      "citations", "degraded", "attempts", "provider",
                      "intent_domain", "tools_active", "mode"):
                if k in attrs:
                    v = attrs[k]
                    if isinstance(v, (list, dict)):
                        v = _truncate(json.dumps(v, ensure_ascii=False), 120)
                    print(f"{'  ' * depth}   {k} = {v}")
        for c in children.get(span.get("span_id", ""), []):
            walk(c, depth + 1)

    for s in children.get("", []):
        walk(s, 0)

    # 最终汇总
    print("\n" + "=" * 62)
    print("📊 汇总")
    print("=" * 62)
    for k, v in tags.items():
        if isinstance(v, (list, dict)):
            v = _truncate(json.dumps(v, ensure_ascii=False), 100)
        print(f"  {k}: {v}")


def step3_ask(
    question: str, config: str,
    verbose: bool = True, debug: bool = False,
    replay: bool = False,
) -> bool:
    print(f'\n[Step 3/3] 提问："{question}"')
    print("-" * 60)
    sys.stdout.flush()

    # stdout 写文件（最终 JSON）；stderr 实时流式到终端。
    tmp_out = tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False, encoding="utf-8",
    )
    tmp_out.close()
    tmp_path = tmp_out.name

    try:
        cmd = [
            PYTHON, "-m", "fusion_rag.cli", "ask", question,
            "--kb", KB_NAME, "--data-dir", str(RAG_DATA), "--json",
        ]
        if config and Path(config).exists():
            cmd += ["--config", config]
        global _t_request_submit, _wall_ttft_printed
        _t_request_submit = time.monotonic()
        _wall_ttft_printed = False
        with open(tmp_path, "wb") as fout:
            proc = subprocess.Popen(
                cmd, env=_env(), cwd=str(PROJECT_DIR),
                stdout=fout, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
            assert proc.stderr is not None
            try:
                for line in proc.stderr:
                    _handle_stderr_line(line, debug=debug)
            finally:
                rc = proc.wait()
                try:
                    proc.stderr.close()
                except OSError:
                    pass

        # 失败不直接退出：尝试 replay（partial trace 也有参考价值）
        raw = Path(tmp_path).read_text(encoding="utf-8")
        d: dict[str, Any] | None = None
        if raw.strip():
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                d = None

        if d is None:
            print(f"\n❌ ask 失败 rc={rc}，无 JSON 输出。可手动运行：\n"
                  f"   python -m fusion_rag.cli replay <trace_id> --data-dir {RAG_DATA}")
            return False

        if verbose:
            _render_answer(d, full=True)
        trace_id = d.get("trace_id", "")
        # ★ 默认不再重放整棵 span 树（与流式输出重叠）；只有 --replay 才展开。
        #   失败时自动展开 trace（帮助定位问题）。
        need_replay = replay or (rc != 0 and verbose)
        if need_replay and trace_id:
            _do_replay(trace_id, config)
        elif verbose and trace_id:
            # 仅保留一行提示，告知用户可手动 replay 查看完整轨迹
            print(
                f"\n🧾 trace 已落盘（需完整重放请手动执行：\n"
                f"   python -m fusion_rag.cli replay {trace_id} --data-dir {RAG_DATA}\n"
                f"   或直接加 --replay 参数重跑）"
            )
        return rc == 0
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ------------------------------------------------------------------
# 子进程 stderr 流式处理：结构化事件 → 人可读，其他行默认丢弃
# ------------------------------------------------------------------
_PHASE_LABEL = {
    "intent": "🧠 意图识别",
    "plan": "📋 任务规划",
    "memory": "💾 记忆装配",
    "retrieval": "🔍 检索",
    "reason": "✍️  推理生成",
    "validate": "✅ 校验",
}

# 追踪当前 agentic 迭代是否已经打印过思考行，避免重复与提供兆底
_thinking_seen_for_iter: dict[int, bool] = {}
# 缓存 seed 信息供 iter#1 前展示
_last_seed_info: dict[str, Any] = {}
# ★ fanout 并发场景：缓存 dispatch 事件里的 worker_id -> 手册名映射，
#   用于后续子 Agent 工具事件前缀展示（如 [手册A]）；非 fanout 时为空 dict。
_fanout_worker_manuals: dict[str, str] = {}

# ★ 真实“提交请求→答案首 token”墙钟基准：run_ask_subprocess 在 Popen 前
#   置 _t_request_submit；子进程首个带 answer_e2e_ttft_s 的 llm_chat_done
#   事件到达时打印真实墙钟。注：流式 chat 在完整回答生成后才回报该事件，
#   故该墙钟是“提交→答案首 token”的上界（含进程启动/索引加载/意图/规划/
#   seed 等 ask() 起点之前无法计入的耗时），用于对照 40s 硬约束。
_t_request_submit: float = 0.0
_wall_ttft_printed: bool = False


def _worker_prefix(rec: dict[str, Any]) -> str:
    """fanout 子 Agent 事件前缀：有 worker_id 时返回 ``[手册X] ``，否则空串。"""
    wid = str(rec.get("worker_id", "") or "")
    if not wid:
        return ""
    manual = _fanout_worker_manuals.get(wid) or wid
    return f"[{manual}] "


def _infer_intent(tool: str, args: dict[str, Any]) -> str:
    """当模型未返回文字思考时，从工具名 + 关键入参合成一行“推断意图”。"""
    args = args or {}
    key = (
        args.get("query") or args.get("pattern") or args.get("glob")
        or args.get("path") or args.get("dir") or ""
    )
    key = str(key).strip()
    hint_map = {
        "search_kb": "先混合检索定位相关片段",
        "grep": "在原文里精确匹配关键词定位行号",
        "glob": "按名字模式列文件",
        "find": "按条件找文件",
        "list_dir": "先看看目录里有什么",
        "read_file": "精读文件内容确认上下文",
    }
    purpose = hint_map.get(tool, "推进取证与定位")
    if key:
        return f"🤔 推断意图（模型未回文字）：{purpose}，关键字：{_truncate(key, 40)}"
    return f"🤔 推断意图（模型未回文字）：{purpose}"


def _handle_stderr_line(line: str, *, debug: bool) -> None:
    line = line.rstrip("\n")
    if not line.strip():
        return
    s = line.strip()
    if not (s.startswith("{") and s.endswith("}")):
        # 非 JSON 行（traceback / 手写 print）直接透传
        print(f"   | {line}")
        sys.stdout.flush()
        return
    try:
        rec = json.loads(s)
    except json.JSONDecodeError:
        print(f"   | {line}")
        sys.stdout.flush()
        return
    event = rec.get("event")
    if not event:
        level = str(rec.get("level", "")).upper()
        if level in ("WARNING", "ERROR", "CRITICAL"):
            # 非事件但重要的日志（警告/错误）一率透传，方便定位失败原因
            print(f"    ⚠ [{level}] {_truncate(rec.get('msg', ''), 220)}")
            sys.stdout.flush()
        elif debug:
            print(f"   · [{level}] {_truncate(rec.get('msg', ''), 140)}")
            sys.stdout.flush()
        return
    _render_event(rec, event)


def _render_event(rec: dict[str, Any], event: str) -> None:
    """把一个结构化事件映射成一行人可读输出。"""
    global _wall_ttft_printed
    if event == "phase":
        phase = rec.get("phase", "")
        label = _PHASE_LABEL.get(phase, phase)
        attempt = rec.get("attempt")
        tail = f"  (attempt #{attempt})" if attempt else ""
        print(f"\n▶ {label}{tail}")
    elif event == "phase_done":
        phase = rec.get("phase", "")
        if phase == "intent":
            print(
                f"  结果：intent={rec.get('intent', '-')} "
                f"domain={rec.get('domain', '-')} risk={rec.get('risk', '-')}"
            )
        elif phase == "validate":
            ok = rec.get("passed")
            issues = rec.get("issues") or []
            conf = rec.get("confidence", 0.0)
            mark = "✅ 通过" if ok else "❌ 未通过"
            print(
                f"  {mark}  conf={conf:.2f} issues={len(issues)}"
            )
            for it in issues[:3]:
                print(f"     • {_truncate(str(it), 140)}")
            if len(issues) > 3:
                print(f"     … 还有 {len(issues) - 3} 条")
        else:
            print(f"  ✓ {phase} done")
    elif event == "agentic_attempt":
        att = rec.get("attempt", 1)
        mx = rec.get("max_attempts", 1)
        if att == 1:
            # 多 attempt 场景下的首轮，不称“重试”而标“主链路”
            print(
                f"\n🤖 Agentic 主链路启动 (max_attempts={mx}, "
                f"followup={rec.get('followup_count', 0)})"
            )
        else:
            print(
                f"\n🔁 重试尝试 #{att}/{mx} "
                f"(followup={rec.get('followup_count', 0)})"
            )
        _thinking_seen_for_iter.clear()
    elif event == "seed_evidence_built":
        _last_seed_info["seed"] = rec.get("seed", "")
        _last_seed_info["sub_queries"] = rec.get("sub_queries", [])
    elif event == "agentic_iter_start":
        # ★ 第一轮前显示 seed 线索，让用户理解模型的起点
        _wp = _worker_prefix(rec)
        if int(rec.get("iter", 0)) == 1 and not _wp:
            seed = _last_seed_info.get("seed", "")
            subs = _last_seed_info.get("sub_queries", [])
            if subs:
                print(f"  \033[36m🔍 子查询拆分: {', '.join(f'「{s}」' for s in subs)}\033[0m")
            if seed:
                for sl in seed.splitlines()[:8]:
                    print(f"  \033[90m  {sl}\033[0m")
        print(
            f"\n  {_wp and _wp + ' ' or ''}⚡ Agentic 迭代 #{rec.get('iter')}/{rec.get('max_iters')}"
        )
        _thinking_seen_for_iter.pop(int(rec.get("iter", 0)), None)
    elif event == "llm_chat_done":
        _wp = _worker_prefix(rec)
        lat = rec.get("latency_s", 0)
        pt = rec.get("prompt_tokens", 0)
        ct = rec.get("completion_tokens", 0)
        e2e = rec.get("answer_e2e_ttft_s")
        if e2e is not None:
            print(f"    {_wp}⏱ 首 token 时延: {lat:.1f}s  (in={pt} tok, out={ct} tok)")
            print(f"    {_wp}🚀 端到端 TTFT: {e2e:.1f}s  (从问题提交→答案首token)")
            # ★ 真实全链路墙钟（仅首个答案事件打印一次）：含子进程启动/索引
            #   加载等 ask() 起点之前无法计入的耗时，是 40s 硬约束的对照基准。
            if not _wall_ttft_printed and _t_request_submit > 0:
                _wall_ttft_printed = True
                _wall = time.monotonic() - _t_request_submit
                _ok = "✓ 达标" if _wall <= 40.0 else "✗ 超 40s"
                print(
                    f"  🕐 提交→答案首 token（真实全链路墙钟上界）: "
                    f"{_wall:.1f}s  [{_ok}]"
                )
        else:
            print(f"    {_wp}⏱ 首token时延: {lat:.1f}s  (in={pt} tok, out={ct} tok)")
    elif event == "agentic_thinking":
        _wp = _worker_prefix(rec)
        text = str(rec.get("thinking", "")).strip()
        if text:
            _thinking_seen_for_iter[int(rec.get("iter", 0))] = True
            print(f"    {_wp}💭 思考：{_truncate(text, 300)}")
    elif event == "tool_call":
        _wp = _worker_prefix(rec)
        args = rec.get("tool_args") or rec.get("args") or {}
        if isinstance(args, dict):
            arg_line = ", ".join(
                f"{k}={_truncate(str(v), 40)}" for k, v in args.items()
            )
        else:
            arg_line = str(args)
        # 兵到当前迭代模型已先写过思考？没有则补一行兵底（内隐式推断）
        it = int(rec.get("iter", 0))
        if not _thinking_seen_for_iter.get(it):
            inline = str(rec.get("thinking", "")).strip()
            if inline:
                print(f"    {_wp}💭 思考：{_truncate(inline, 300)}")
            else:
                print(f"    {_wp}{_infer_intent(str(rec.get('tool', '')), args if isinstance(args, dict) else {})}")
            _thinking_seen_for_iter[it] = True
        print(
            f"    {_wp}🔧 调用 {rec.get('tool', '?')}({arg_line})"
        )
    elif event == "tool_batch":
        print(
            f"    ⚡ 并行派发 {rec.get('count')} 个工具，总耗时 {rec.get('batch_ms')}ms"
        )
    elif event == "agentic_seed_direct_answer":
        _wp = _worker_prefix(rec)
        print(
            f"  {_wp}⚡ 策略A：seed 已含行级证据，首轮直接作答（跳过工具往返）"
        )
    elif event == "agentic_budget_exceeded":
        print(
            f"    ⏱ 预算耗尽 ({rec.get('budget_ms')}ms)，强制从当前证据收敛"
        )
    elif event == "agentic_empty_shell":
        print(
            f"    ⚠ 模型仅回标签、没实质内容 → 提醒后回退重试"
            f" (iter #{rec.get('iter')})"
        )
    elif event == "tool_result":
        _wp = _worker_prefix(rec)
        is_err = rec.get("is_error")
        mark = "⚠️ 错误" if is_err else "✅"
        print(
            f"    {_wp}{mark} 回包 {rec.get('tool', '?')} "
            f"[{rec.get('duration_ms', 0)}ms, {rec.get('size', 0)}B]"
        )
        preview = str(rec.get("preview", "")).strip()
        if preview:
            first = preview.splitlines()[0] if preview else ""
            print(f"       ↳ {_truncate(first, 160)}")
        if is_err:
            print(f"       → 模型将在下一轮调整策略")
    elif event == "agentic_final":
        _wp = _worker_prefix(rec)
        preview = str(rec.get("answer_preview", "")).strip()
        print(f"    {_wp}🎯 收敛答案 (iter #{rec.get('iter')})")
        if preview:
            print(f"       {_truncate(preview, 200)}")
    elif event == "agentic_plan":
        plan = str(rec.get("plan") or rec.get("plan_preview", "")).strip()
        summary = rec.get("summary") or {}
        head = "  🧭 问题分解"
        if summary:
            head += f"（已厘清 {summary.get('completed', 0)}/{summary.get('total', 0)}）"
        if rec.get("iter") is not None:
            head += f"  (iter #{rec.get('iter')})"
        if plan:
            print("\n" + head + "：")
            for _ln in plan.splitlines():
                if _ln.strip():
                    print(f"     │ {_truncate(_ln.strip(), 300)}")
    elif event == "agentic_plan_reinject":
        print(
            f"    ↻ 第 {rec.get('iter')} 轮：已将问题分解"
            f"（{rec.get('items', 0)} 项）注入本轮 prompt"
        )
    elif event == "agentic_evidence_reinject":
        print(
            f"    ≡ 第 {rec.get('iter')} 轮：已把 {rec.get('entries', 0)} 条证据"
            f"（{rec.get('chars', 0)} 字）注入本轮 prompt"
        )
    elif event == "agentic_early_stop_armed":
        print(
            f"    ⚑ 本轮新增 {rec.get('delta', 0)} 条行级引用（累计 "
            f"{rec.get('cites', 0)}）→ 下一轮提前禁工具强制收敛"
        )
    elif event == "agentic_plan_only_refund":
        print(
            f"    ↺ 第 {rec.get('iter')} 轮：只改拆解未干活，回收该轮额度"
            f"（已回收 {rec.get('rewinds', 0)}/2）"
        )
    # ------------------------------------------------------------------
    # ★ fanout 多手册并发子 Agent 事件（见 fusion_rag/agents/fanout.py）
    # ------------------------------------------------------------------
    elif event == "fanout_candidates":
        manuals = rec.get("manuals") or []
        print(
            f"\n🔎 候选手册分组：命中 {len(manuals)} 本 -> "
            f"{', '.join(f'《{m}》' for m in manuals)}"
        )
    elif event == "fanout_dispatched":
        _fanout_worker_manuals.clear()
        manuals = rec.get("manuals") or []
        for m in manuals:
            _fanout_worker_manuals[f"manual:{m}"] = m
        print(
            f"\n🔀 并发扇出: {rec.get('worker_count', len(manuals))} 本子Agent -> "
            f"{', '.join(f'《{m}》' for m in manuals)}"
        )
    elif event == "fanout_worker_marked_solved":
        _wp = _worker_prefix(rec)
        print(
            f"  🏁 {_wp}自报置信度 {rec.get('confidence')} 达标"
            f"(阈值 {rec.get('threshold')})，广播停止其他子Agent"
        )
    elif event == "fanout_stopped_early":
        _wp = _worker_prefix(rec) or f"[{rec.get('worker_id', '?')}] "
        print(
            f"  ⏹ 子Agent{_wp}第 {rec.get('iter')} 轮提前结束"
            f"（原因: {rec.get('reason', 'peer_solved')}）"
        )
    elif event == "fanout_solved":
        print(
            f"\n🏆 子Agent[{rec.get('worker_id', '?')}] 已解决 "
            f"(confidence={rec.get('confidence')})"
        )
    elif event == "fanout_merged":
        print(
            f"\n🧩 无人达标，合并 {rec.get('contributing_sections', rec.get('worker_count', 0))} 份"
            f"子Agent证据做一次合成调用 (confidence={rec.get('confidence')})"
        )
    else:
        print(f"    · {event}: {_truncate(str(rec.get('msg', '')), 140)}")
    sys.stdout.flush()


def _do_replay(trace_id: str, config: str) -> None:
    cmd = [
        PYTHON, "-m", "fusion_rag.cli", "replay", trace_id,
        "--data-dir", str(RAG_DATA),
    ]
    if config and Path(config).exists():
        cmd += ["--config", config]
    rc, out = _run(cmd, label="replay", capture=True)
    if rc == 0 and out:
        try:
            _render_trace(json.loads(out))
        except json.JSONDecodeError:
            pass


def _render_answer(d: dict[str, Any], full: bool = False) -> None:
    """格式化答案展示。"""
    print("\n" + "=" * 62)
    print("💡 最终答案")
    print("=" * 62)
    ans = d.get("answer", "")
    print(ans if full else _truncate(ans, 300))

    print("\n" + "-" * 62)
    print("📌 关键指标")
    print("-" * 62)
    print(f"  模型:       {d.get('model', '')}")
    print(f"  置信度:     {d.get('confidence', 0)}")
    print(f"  已校验:     {'✓' if d.get('validated') else '✗'}")
    print(f"  trace_id:   {d.get('trace_id', '')}")

    meta = d.get("meta", {}) or {}
    print(f"  尝试次数:   {meta.get('attempts', '-')}")
    print(f"  意图:       {meta.get('domain', '-')} / risk={meta.get('risk', '-')}")
    print(f"  子问题:     {meta.get('decomposed') and '已拆解' or '未拆解'}")
    if meta.get("issues"):
        print(f"  校验问题:   {len(meta['issues'])} 条")
        for i, iss in enumerate(meta["issues"], 1):
            print(f"    [{i}] {_truncate(iss, 150)}")

    agentic = meta.get("agentic")
    if agentic:
        print(f"\n  🤖 Agentic: {agentic}")

    cits = d.get("citations", [])
    if cits:
        print(f"\n  📚 引用 ({len(cits)} 条):")
        for c in cits[:8]:
            # Citation dataclass 的字段：index / document_id / title / source /
            # chunk_index / snippet / score（之前误读 doc_title/doc_id/locator 导致列都空）
            title = c.get("title") or c.get("source") or c.get("document_id") or ""
            score = c.get("score", 0)
            snippet = c.get("snippet") or ""
            # snippet 里已含行号列表（path:123,456），直接拼后面
            tail = f"  【{snippet}】" if snippet and snippet != title else ""
            print(f"    • {title}  score={score:.4f}{tail}")
        if len(cits) > 8:
            print(f"    ... 还有 {len(cits) - 8} 条")


# ==================================================================
# 主入口
# ==================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="一键跑通 PDF 知识库：转换 → 入库 → 问答（带流程可视化）",
    )
    parser.add_argument("question", nargs="?", default=None, help="要提问的问题")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--reindex", default=False, help="强制重新转换+重建索引")
    parser.add_argument("--quiet", action="store_true", help="不显示流程轨迹")
    parser.add_argument(
        "--debug", action="store_true",
        help="打印子进程完整日志（包括 JSON 行）",
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="问结束后额外重放完整 trace 树（默认只保留流式输出，避免重复）",
    )
    args = parser.parse_args()

    # q = "程序备份指令支持多少种备份方法，分别是什么"
    # q = "大点数机器设备报错短路故障，怎么恢复错误？"
    # q = "h5u有哪些型号plc"
    q = "easy320是否支持fins"
    # q = "H5U可以带多少个扩展模块"
    q = "Ethercat主站怎么配置"

    question = args.question or q
    if not args.question:
        print(f'[INFO] 使用默认问题："{question}"')

    config = args.config or str(PROJECT_DIR / "config.yaml")

    print("=" * 62)
    print(" FusionRAG 知识库一键流程")
    print(f"   项目：  {PROJECT_DIR}")
    print(f"   配置：  {config}")
    print(f"   数据：  {RAG_DATA}")
    print(f"   模式：  {'全量重建' if args.reindex else '增量'}")
    print("=" * 62)

    if not step1_convert(force=args.reindex):
        return 1
    if not step2_index(config, force=args.reindex):
        return 1
    if not step3_ask(
        question, config,
        verbose=not args.quiet, debug=args.debug, replay=args.replay,
    ):
        return 1

    print("\n✅ 完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
