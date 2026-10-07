"""Fixed Egyptian Arabic response templates. Questions never contain invented values.

Templates use masculine second person as the default address form; the wording is listed in
docs/task_contract.md for native-speaker review before the dataset is frozen.
"""

from __future__ import annotations

from datetime import datetime

from athar.extract import Extraction
from athar.policy import Decision
from athar.schema import Response, ToolCall
from athar.timeexpr import Clock

AR_WEEKDAYS = ["الاتنين", "التلات", "الأربع", "الخميس", "الجمعة", "السبت", "الحد"]
AR_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر",
             "أكتوبر", "نوفمبر", "ديسمبر"]

ABSTAIN_MESSAGE = "معلش، أنا أقدر أعمل تذكير أو منبه لمرة واحدة ليك بس، ومقدرش أنفذ الطلب ده."
QUESTIONS = {
    "date": "تقصد أنهي يوم بالظبط؟",
    "time": {"create_reminder": "عايزني أفكرك الساعة كام؟", "create_alarm": "عايز المنبه يرن الساعة كام؟"},
    "am_pm": "الساعة {h} الصبح ولا بالليل؟",
    "past": "الوقت ده عدّى خلاص، تحب أخليه إمتى؟",
    "reminder_text": "أفكرك بإيه؟",
}


def _period_word(hour: int) -> str:
    if 5 <= hour <= 11:
        return "الصبح"
    if 12 <= hour <= 14:
        return "الضهر"
    if 15 <= hour <= 17:
        return "العصر"
    return "بالليل"


def describe_datetime(dt: datetime) -> str:
    h12 = dt.hour % 12 or 12
    return (f"يوم {AR_WEEKDAYS[dt.weekday()]} {dt.day} {AR_MONTHS[dt.month - 1]} "
            f"الساعة {h12}:{dt.minute:02d} {_period_word(dt.hour)}")


def _said_hour(ext: Extraction) -> str | None:
    clocks = [m.value for m in ext.temporal.mentions if m.kind == "clock"]
    if len(clocks) == 1 and isinstance(clocks[0], Clock):
        c = clocks[0]
        total = c.hour * 60 + c.minute
        h, m = (total // 60) % 12 or 12, total % 60
        return f"{h}:{m:02d}" if m else str(h)
    return None


def question(decision: Decision, ext: Extraction) -> str:
    parts = []
    for gap in decision.targets:
        q = QUESTIONS[gap]
        if isinstance(q, dict):
            q = q[decision.intent]
        if gap == "am_pm":
            h = _said_hour(ext)
            q = q.format(h=h) if h else "تقصد الصبح ولا بالليل؟"
        parts.append(q)
    return " ".join(parts)


def build_response(system: str, decision: Decision, ext: Extraction) -> Response:
    tool_call = None
    if decision.route == "ACT":
        assert ext.datetime is not None
        text = ext.reminder_text if decision.intent == "create_reminder" else None
        tool_call = ToolCall(name=decision.intent, datetime=ext.datetime, reminder_text=text)  # type: ignore[arg-type]
        when = describe_datetime(ext.datetime)
        message = (f"تمام، هفكرك بـ«{text}» {when}." if text else f"تمام، المنبه هيرن {when}.")
    elif decision.route == "CLARIFY":
        message = question(decision, ext)
    else:
        message = ABSTAIN_MESSAGE
    return Response(
        system=system,
        route=decision.route,  # type: ignore[arg-type]
        intent=decision.intent,  # type: ignore[arg-type]
        tool_call=tool_call,
        clarify_targets=list(decision.targets),  # type: ignore[arg-type]
        message=message,
        extracted_datetime=ext.datetime,
        extracted_reminder_text=ext.reminder_text,
    )
