"""Closed word lists for the supported time expressions and request framing.

Entries may be written in natural spelling; they are normalized at import time with the same
normalizer the input goes through, so ``بكرة``/``بكره`` or ``الأربعاء``/``الاربعاء`` need to be
listed once. Every list here is part of the task contract (docs/task_contract.md) and is frozen
before the test split is opened.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from athar.normalize import norm


def _n(words: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(norm(w) for w in words))


def _nmap(mapping: Mapping[str, object]) -> dict[str, object]:
    return {norm(k): v for k, v in mapping.items()}


def alt(words: Iterable[str]) -> str:
    """Regex alternation, longest first so that ``بعد بكره`` wins over ``بكره``."""
    ws = sorted(set(words), key=len, reverse=True)
    return "(?:" + "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in ws) + ")"


# --------------------------------------------------------------------------- dates
#: relative day words -> offset in days from the reference date
DAY_WORDS: dict[str, int] = _nmap({
    "النهارده": 0, "النهاردة": 0, "النهاردا": 0, "انهارده": 0, "النهار ده": 0, "اليوم": 0,
    "today": 0,
    "بكرة": 1, "بكرا": 1, "tomorrow": 1, "tmrw": 1, "tmw": 1,
    "بعد بكرة": 2, "بعد بكرا": 2, "day after tomorrow": 2, "the day after tomorrow": 2,
    "after tomorrow": 2,
})  # type: ignore[assignment]

#: "tonight" = today's date + night period
TONIGHT_WORDS = _n(["الليلة", "الليلادي", "tonight"])

#: weekday names -> Python weekday (Mon=0 ... Sun=6)
WEEKDAYS: dict[str, int] = _nmap({
    "السبت": 5,
    "الحد": 6, "الأحد": 6,
    "الاتنين": 0, "الإتنين": 0, "الاثنين": 0, "الإثنين": 0,
    "التلات": 1, "التلاتاء": 1, "التلاتا": 1, "الثلاثاء": 1, "الثلاثا": 1,
    "الأربع": 2, "الأربعاء": 2, "الأربعا": 2, "الاربعة": 2,
    "الخميس": 3,
    "الجمعة": 4,
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5,
    "sunday": 6,
})  # type: ignore[assignment]

#: weekday modifiers. "next": the occurrence 1-7 days ahead (7 when the weekday is today).
#: "this": the occurrence 0-6 days ahead (0 = today).
WEEKDAY_NEXT = _n(["الجاي", "الجاية", "اللي جاي", "اللي جاية", "اللى جاى"])
WEEKDAY_NEXT_EN = _n(["next", "this coming", "coming"])
WEEKDAY_THIS = _n(["ده", "دي", "دة"])
WEEKDAY_THIS_EN = _n(["this"])

MONTHS: dict[str, int] = _nmap({
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "ابريل": 4, "إبريل": 4, "مايو": 5,
    "يونيو": 6, "يونيه": 6, "يوليو": 7, "يوليه": 7, "أغسطس": 8, "سبتمبر": 9, "أكتوبر": 10,
    "نوفمبر": 11, "ديسمبر": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
})  # type: ignore[assignment]

#: date references the contract treats as ambiguous (-> CLARIFY with target ``date``)
VAGUE_DATE = _n([
    "الأسبوع الجاي", "الاسبوع اللي جاي", "الأسبوع الجاى", "next week", "الشهر الجاي",
    "الشهر اللي جاي", "next month", "الويك إند", "الويكند", "الويك اند", "weekend",
    "the weekend", "آخر الأسبوع", "أول الأسبوع", "آخر الشهر", "أول الشهر", "some day",
    "someday",
])

# --------------------------------------------------------------------------- times
#: hour number words (1-12). Only accepted after ``الساعة`` or before a period word.
HOUR_WORDS: dict[str, int] = _nmap({
    "واحدة": 1, "واحد": 1, "اتنين": 2, "اثنين": 2, "تلاتة": 3, "ثلاثة": 3, "تلاته": 3,
    "أربعة": 4, "اربع": 4, "خمسة": 5, "خمس": 5, "ستة": 6, "ست": 6, "سبعة": 7, "سبع": 7,
    "تمانية": 8, "ثمانية": 8, "تمنية": 8, "تمانيه": 8, "تسعة": 9, "تسع": 9, "عشرة": 10,
    "عشر": 10, "حداشر": 11, "احداشر": 11, "حداشار": 11, "اتناشر": 12, "اثناشر": 12,
    "اطناشر": 12,
})  # type: ignore[assignment]

#: minute words used after ``و`` / ``إلا``
MINUTE_WORDS: dict[str, int] = _nmap({
    "خمسة": 5, "خمس": 5, "عشرة": 10, "عشر": 10, "ربع": 15, "تلت": 20, "ثلث": 20,
    "نص": 30, "نصف": 30, "عشرين": 20, "خمسة وعشرين": 25, "تلاتين": 30, "ثلاثين": 30,
    "خمستاشر": 15, "خمستاشر دقيقة": 15,
})  # type: ignore[assignment]

#: amounts for relative offsets ("بعد ساعتين", "بعد عشر دقايق", "in 20 minutes")
AMOUNT_WORDS: dict[str, int] = _nmap({
    **{k: v for k, v in MINUTE_WORDS.items()},
    "واحدة": 1, "اتنين": 2, "تلاتة": 3, "أربعة": 4, "ستة": 6, "سبعة": 7, "تمانية": 8,
    "تسعة": 9, "أربعين": 40, "خمسين": 50, "an": 1, "a": 1, "one": 1, "two": 2, "three": 3,
    "five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "half an": 0.5, "half": 0.5,
})  # type: ignore[assignment]
MINUTE_UNITS = _n(["دقيقة", "دقايق", "دقائق", "دقيقه", "minute", "minutes", "min", "mins"])
MINUTE_UNITS_DUAL = _n(["دقيقتين"])
HOUR_UNITS = _n(["ساعة", "ساعات", "hour", "hours", "hr", "hrs"])
HOUR_UNITS_DUAL = _n(["ساعتين"])

#: day periods. Each maps to a period key that the resolver turns into a 24 h hour.
PERIODS: dict[str, str] = _nmap({
    "الصبح": "morning", "الصباح": "morning", "صباحا": "morning", "صباحاً": "morning",
    "الصبحية": "morning", "in the morning": "morning", "morning": "morning",
    "الفجر": "dawn",
    "الضهر": "noon", "الظهر": "noon", "الضهرية": "noon",
    "بعد الضهر": "afternoon", "بعد الظهر": "afternoon", "العصر": "afternoon",
    "in the afternoon": "afternoon", "afternoon": "afternoon",
    "المغرب": "maghrib",
    "مساء": "pm", "مساءا": "pm", "مساءً": "pm", "بالمسا": "pm", "المسا": "pm",
    "in the evening": "pm", "evening": "pm",
    "بالليل": "night", "بليل": "night", "الليل": "night", "at night": "night",
    "night": "night", "العشا": "night", "بعد العشا": "night",
})  # type: ignore[assignment]

#: clock markers that may follow a time directly ("7 م", "7 pm")
AM_MARKERS = _n(["ص", "am", "a.m.", "a.m"])
PM_MARKERS = _n(["م", "pm", "p.m.", "p.m"])

#: exact clock words
NOON_WORDS = _n(["noon", "midday"])
MIDNIGHT_WORDS = _n(["midnight", "نص الليل", "نصف الليل"])

#: vague times: carry no clock value (-> CLARIFY ``time``) but are removed from reminder text
VAGUE_TIME = _n([
    "بدري", "بعدين", "بعد شوية", "كمان شوية", "later", "soon",
    "آخر النهار", "أول النهار", "قبل الضهر", "بعد الشغل", "after work", "in a bit",
    "في أي وقت", "any time", "anytime",
])

#: hedges dropped before a clock time ("حوالي الساعة 7")
HEDGES = _n(["حوالي", "تقريبا", "تقريباً", "around", "approximately", "approx"])

# --------------------------------------------------------------------------- request framing
#: request verbs/phrases removed from the reminder text wherever they occur
TRIGGERS = _n([
    "عايزك تفكرني", "عاوزك تفكرني", "عايزاك تفكرني", "ممكن تفكرني", "ياريت تفكرني",
    "فكرني", "فكريني", "ذكرني", "ذكريني", "تفكرني", "تفكريني", "تذكرني", "تذكريني",
    "متنسانيش", "ما تنسانيش", "نبهني", "نبهيني", "صحيني", "صحّيني", "قومني",
    "remind me", "wake me up", "wake me",
])
#: verbs that only count as framing when followed by a tool noun ("اعمل تذكير")
SETUP_VERBS = _n([
    "حطلي", "حط لي", "حط", "اعملي", "اعمل لي", "اعمل", "ظبطلي", "ضبطلي", "ظبط", "ضبط",
    "ظبط لي", "سيتلي", "سيت", "set", "create", "add", "make", "put", "schedule",
])
TOOL_NOUNS = _n([
    "تذكير", "ريمايندر", "ريماندر", "reminder", "a reminder", "منبه", "المنبه", "alarm",
    "an alarm", "the alarm", "الارم", "ألارم",
])
FILLERS = _n([
    "لو سمحت", "لو سمحتي", "من فضلك", "ياريت", "يا ريت", "معلش", "بقولك", "بليز", "please",
    "pls", "plz", "kindly", "can you", "could you", "would you", "يا أثر", "ya athar", "athar",
])
#: words stripped only at the edges of the reminder text
EDGE_WORDS = _n([
    "و", "إن", "إني", "انك", "إننا", "عشان", "علشان", "ب", "بإن", "بإني", "ل", "لي", "ليا",
    "على", "ع", "في", "يوم", "الساعة", "to", "about", "that", "of", "for", "me", "at", "on",
    "in", "by", "and", "i", "need",
])
#: reminder contents too vague to act on (-> CLARIFY ``reminder_text``)
VAGUE_CONTENT = _n(["حاجة", "حاجات", "حاجة مهمة", "الحاجة", "حاجه", "something", "stuff", "thing"])
