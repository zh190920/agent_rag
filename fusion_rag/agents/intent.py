"""IntentAgent：意图识别 + 知识域判定 + 风险分级 + 无效提问过滤。

LLM 优先（结构化 JSON，带 __INTENT__ 协议标记）；失败降级为规则分类器，
保证任何情况下都能给出可用意图。
"""

from __future__ import annotations

import re
from typing import Any

from ..constants import INOVANCE_PERSONA
from ..core.logging import get_logger
from ..llm.base import Message
from ..types import IntentResult
from .base import AgentContext, BaseAgent

logger = get_logger(__name__)

_GREETING = re.compile(
    r"^(你好|您好|hi|hello|在吗|在么|嗨|hey|早上好|晚上好)\W*$", re.IGNORECASE,
)
_CHITCHAT_WORDS = ("谢谢", "感谢", "再见", "哈哈", "聊聊", "无聊", "你是谁", "讲个笑话")
_REJECT_WORDS = ("违法", "暴力", "自杀", "制毒", "枪支", "赌博网站")
_HIGH_RISK_WORDS = ("诊断", "开药", "用药剂量", "起诉", "判决", "投资建议", "手术方案")
_CLARIFY_LEN = 2


class IntentAgent(BaseAgent):
    """意图识别 Agent。"""

    name = "intent"
    role = "识别用户问答意图、问题类型、知识域，过滤无效/模糊/高风险提问"

    def __init__(self, *args: Any, use_llm: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_llm = use_llm

    async def classify(self, question: str, ctx: AgentContext) -> IntentResult:
        text = (question or "").strip()
        if not text:
            return IntentResult(intent="reject", valid=False, reason="空提问")

        result: IntentResult | None = None
        if self.use_llm and self.llm is not None:
            with self.span("classify_llm", length=len(text)):
                result = await self._llm_classify(text)
        if result is None:
            result = self._rule_classify(text)

        result = self._post_adjust(text, result)
        self.count("intent_classified", intent=result.intent, risk=result.risk)
        with self.span("result", intent=result.intent, risk=result.risk, valid=result.valid):
            pass
        return result

    # ------------------------------------------------------------------
    async def _llm_classify(self, text: str) -> IntentResult | None:
        system = (
            INOVANCE_PERSONA + "\n"
            "你是意图识别器。判断用户输入的类型与风险，只输出 JSON。__INTENT__\n"
            "字段：intent(qa/chitchat/reject/clarify), domain(知识域,如 general/tech/legal/medical/hr), "
            "risk(low/medium/high), valid(是否为有效知识问答), reason(简述)。\n"
            '示例：{"intent":"qa","domain":"tech","risk":"low","valid":true,"reason":"技术问答"}'
        )
        try:
            data = await self.llm.chat_json(  # type: ignore[union-attr]
                [Message.system(system), Message.user(text)],
                provider=self.provider, temperature=0.0, max_tokens=200,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("意图识别 LLM 失败，降级规则：%s", exc)
            return None
        try:
            return IntentResult(
                intent=str(data.get("intent", "qa")),
                domain=str(data.get("domain", "general")),
                risk=str(data.get("risk", "low")),
                valid=bool(data.get("valid", True)),
                reason=str(data.get("reason", "")),
            )
        except (TypeError, ValueError):
            return None

    def _rule_classify(self, text: str) -> IntentResult:
        if any(w in text for w in _REJECT_WORDS):
            return IntentResult(intent="reject", risk="high", valid=False, reason="命中拒答词")
        if _GREETING.match(text) or any(w in text for w in _CHITCHAT_WORDS):
            return IntentResult(intent="chitchat", risk="low", valid=False, reason="闲聊/寒暄")
        if len(text) <= _CLARIFY_LEN or text.count("?") + text.count("？") == len(text):
            return IntentResult(intent="clarify", risk="low", valid=False, reason="提问过短/模糊")
        domain = "general"
        if re.search(r"代码|报错|接口|数据库|api|函数|部署", text, re.IGNORECASE):
            domain = "tech"
        elif any(w in text for w in ("合同", "条款", "法律", "诉讼")):
            domain = "legal"
        elif any(w in text for w in ("症状", "药", "病", "治疗")):
            domain = "medical"
        elif any(w in text for w in ("报销", "请假", "入职", "薪资", "考勤")):
            domain = "hr"
        return IntentResult(intent="qa", domain=domain, risk="low", valid=True, reason="规则判定为知识问答")

    def _post_adjust(self, text: str, result: IntentResult) -> IntentResult:
        """规则兜底修正：高风险词提升 risk；拒答词强制 reject。"""
        if any(w in text for w in _REJECT_WORDS):
            result.intent = "reject"
            result.risk = "high"
            result.valid = False
        if any(w in text for w in _HIGH_RISK_WORDS) and result.risk == "low":
            result.risk = "medium"
        return result
