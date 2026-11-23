"""Core v1.2 (post hoc): the four extractor fixes, on invented sentences (not dataset items)."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from athar.v2 import core as V11
from athar.v2 import core12 as V12

CAIRO = ZoneInfo("Africa/Cairo")
REF = dt.datetime(2026, 10, 27, 9, 15, tzinfo=CAIRO)   # a Tuesday


def text_of(message: str) -> str | None:
    r = V12.respond_with_intent("create_reminder", message, REF)
    return r.tool_call.reminder_text if r.tool_call else None


def test_emphasis_word_is_removed_anywhere() -> None:
    assert text_of("فكرني اشتري عيش ضروري بكره الساعه 5 العصر") == "اشتري عيش"
    assert text_of("فكرني ضروري اكلم احمد بكره الساعه 5 العصر") == "اكلم احمد"


@pytest.mark.parametrize("framing", ["عايز تفكرني", "محتاج تفكرني", "وعايزك تفكرني"])
def test_want_verb_before_trigger_is_framing(framing: str) -> None:
    assert text_of(f"{framing} اكلم احمد بكره الساعه 5 العصر") == "اكلم احمد"


def test_leading_want_verb_is_framing() -> None:
    assert text_of("فكرني بكره الساعه 5 العصر محتاج اكلم احمد") == "اكلم احمد"


@pytest.mark.parametrize("message", ["فكرني اراجع علي حاجه مهمه بكره الساعه 5 العصر",
                                     "فكرني بحاجه مهمه جدا بكره الساعه 5 العصر",
                                     "فكرني ابعت حاجه ضروري بكره الساعه 5 العصر"])
def test_vague_content_asks_for_the_text(message: str) -> None:
    r = V12.respond_with_intent("create_reminder", message, REF)
    assert r.route == "CLARIFY" and r.clarify_targets == ["reminder_text"]


def test_specific_content_with_a_vague_noun_is_kept() -> None:
    assert text_of("فكرني اجيب حاجات للبيت بكره الساعه 5 العصر") == "اجيب حاجات للبيت"


@pytest.mark.parametrize(("weekday", "day"), [("الخميس", 5), ("التلات", 10)])
def test_weekday_after_next_is_one_week_after_next(weekday: str, day: int) -> None:
    # sent Tuesday 27 Oct: «الخميس الجاي» = 29 Oct, «التلات الجاي» = 3 Nov (today's weekday -> +7)
    r = V12.respond_with_intent("create_reminder", f"فكرني {weekday} بعد الجاي الساعه 8 الصبح اخد الدوا", REF)
    assert r.route == "ACT" and r.tool_call is not None
    assert r.tool_call.datetime.date() == dt.date(2026, 11, day) and r.tool_call.reminder_text == "اخد الدوا"


def test_core_v11_is_unchanged() -> None:
    # the frozen core still keeps the emphasis word: v1.2 is a separate module
    r = V11.respond_with_intent("create_reminder", "فكرني اشتري عيش ضروري بكره الساعه 5 العصر", REF)
    assert r.tool_call is not None and r.tool_call.reminder_text == "اشتري عيش ضروري"
