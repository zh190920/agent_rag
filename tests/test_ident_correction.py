"""代码标识符「截断订正」回归测试。

覆盖 _is_truncation 纯函数与 ToolAgent._correct_truncated_identifiers 兜底：
用户把长标识符（如系统变量 `_RTU_ECT_INFO[0].DisableSlotControl`）截断成
`_RTU_ECT_INF[0].DisableSlotContr` 提问时，即便模型照抄了截断串，最终回答也
要被确定性订正回手册原文的确切拼写。
"""

from __future__ import annotations

from fusion_rag.agents.tool_agent import (
    ToolAgent,
    ToolRunResult,
    _is_truncation,
)


def _agent() -> ToolAgent:
    # 该订正方法不依赖实例状态，绕过 __init__ 直接造宿主即可测
    return ToolAgent.__new__(ToolAgent)


# ----------------------------------------------------------------------
# _is_truncation 纯函数
# ----------------------------------------------------------------------
def test_truncation_detected_per_segment():
    assert _is_truncation(
        "_RTU_ECT_INF[0].DisableSlotContr",
        "_RTU_ECT_INFO[0].DisableSlotControl",
    )


def test_truncation_false_when_equal():
    assert not _is_truncation("_A[0].B", "_A[0].B")


def test_truncation_false_when_not_prefix():
    assert not _is_truncation("_A[0].B", "_C[1].D")


def test_truncation_false_when_short_is_longer():
    # short 反而更长 → 不是截断
    assert not _is_truncation(
        "_RTU_ECT_INFO[0].DisableSlotControl",
        "_RTU_ECT_INF[0].DisableSlotContr",
    )


def test_truncation_false_for_unsegmented_token():
    # 无 `.`/`[]` 分段（<2 段）不参与截断判定，避免误伤普通词
    assert not _is_truncation("_AB", "_ABC")


def test_truncation_false_for_dropped_segment():
    # 整段被丢（段数不等）不处理，避免跨段重建误伤
    assert not _is_truncation(
        "_RTU_ECT_INF[0]", "_RTU_ECT_INFO[0].DisableSlotControl"
    )


# ----------------------------------------------------------------------
# _correct_truncated_identifiers 兜底
# ----------------------------------------------------------------------
def test_answer_truncated_identifier_gets_corrected():
    agent = _agent()
    result = ToolRunResult(answer="把 _RTU_ECT_INF[0].DisableSlotContr 设为 6 即可")
    result.evidence = [{
        "tool": "grep",
        "args": {},
        "content": "需在第一个扫描周期写系统变量 _RTU_ECT_INFO[0].DisableSlotControl 的值为6",
    }]
    agent._correct_truncated_identifiers(result)
    body = result.answer.split("📖")[0]  # 截断串合理地保留在订正说明里
    assert "_RTU_ECT_INFO[0].DisableSlotControl" in body
    assert "_RTU_ECT_INF[0].DisableSlotContr" not in body
    assert "术语订正" in result.answer


def test_correct_answer_left_untouched():
    agent = _agent()
    result = ToolRunResult(answer="使用 _RTU_ECT_INFO[0].DisableSlotControl 禁用从站")
    result.evidence = [{
        "content": "系统变量 _RTU_ECT_INFO[0].DisableSlotControl 每一位对应一个模块",
    }]
    agent._correct_truncated_identifiers(result)
    assert "术语订正" not in result.answer
    assert result.answer == "使用 _RTU_ECT_INFO[0].DisableSlotControl 禁用从站"


def test_no_correction_without_evidence():
    agent = _agent()
    result = ToolRunResult(answer="_RTU_ECT_INF[0].DisableSlotContr 是什么")
    result.evidence = []
    agent._correct_truncated_identifiers(result)
    # 无证据可核对 → 不擅自改动
    assert "_RTU_ECT_INF[0].DisableSlotContr" in result.answer
    assert "术语订正" not in result.answer


def test_ambiguous_candidates_not_corrected():
    agent = _agent()
    # DisableSlot 同时是 DisableSlotControl / DisableSlotCounter 的前缀
    # → 不唯一 → 保守不订正
    result = ToolRunResult(answer="读 _RTU_ECT_INF[0].DisableSlot 的值")
    result.evidence = [{
        "content": (
            "_RTU_ECT_INFO[0].DisableSlotControl 与 "
            "_RTU_ECT_INFO[0].DisableSlotCounter 都存在"
        ),
    }]
    agent._correct_truncated_identifiers(result)
    assert "术语订正" not in result.answer
