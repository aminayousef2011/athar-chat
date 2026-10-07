"""Contract cases for date/time resolution. Reference: Wed 2026-10-07 10:00 Africa/Cairo."""

from datetime import UTC, timezone

import pytest

from athar.normalize import norm
from athar.timeexpr import cairo, is_after, parse_temporal, to_24h


def run(text, ref):
    t = parse_temporal(norm(text), ref)
    return (t.datetime.isoformat(timespec="minutes") if t.datetime else None), sorted(t.gaps)


RESOLVED = [
    # day words + periods
    ("فكرني بكرة الساعة ٧ الصبح", "2026-10-08T07:00+03:00"),
    ("بكرة الصبح الساعة 7", "2026-10-08T07:00+03:00"),
    ("بعد بكرة الساعة 9 بالليل", "2026-10-09T21:00+03:00"),
    ("النهارده الساعة 5 العصر", "2026-10-07T17:00+03:00"),
    ("النهاردة الساعة 2 الضهر", "2026-10-07T14:00+03:00"),
    ("الليلة الساعة 10", "2026-10-07T22:00+03:00"),
    ("بكرة المغرب الساعة 6", "2026-10-08T18:00+03:00"),
    ("بكرة الفجر الساعة 4", "2026-10-08T04:00+03:00"),
    # markers
    ("بكرة الساعة 7 م", "2026-10-08T19:00+03:00"),
    ("بكرة الساعة 7 ص", "2026-10-08T07:00+03:00"),
    ("tomorrow at 7pm", "2026-10-08T19:00+03:00"),
    ("tomorrow at 12 am", "2026-10-08T00:00+03:00"),
    ("tomorrow at 12pm", "2026-10-08T12:00+03:00"),
    ("tomorrow at noon", "2026-10-08T12:00+03:00"),
    # minutes, Egyptian style
    ("بكرة الساعة 7 ونص الصبح", "2026-10-08T07:30+03:00"),
    ("بكرة الساعة 7 وربع الصبح", "2026-10-08T07:15+03:00"),
    ("بكرة الساعة 7 وتلت الصبح", "2026-10-08T07:20+03:00"),
    ("بكرة الساعة 7 إلا ربع الصبح", "2026-10-08T06:45+03:00"),
    ("بكرة الساعة 7 إلا تلت الصبح", "2026-10-08T06:40+03:00"),
    ("بكرة الساعة 7 ونص إلا خمسة الصبح", "2026-10-08T07:25+03:00"),
    ("بكرة الساعة 7 ونص وخمسة الصبح", "2026-10-08T07:35+03:00"),
    ("بكرة الساعة 7 و10 دقايق الصبح", "2026-10-08T07:10+03:00"),
    ("بكرة الساعة سبعة ونص الصبح", "2026-10-08T07:30+03:00"),
    ("بكرة الساعة 12 إلا ربع الضهر", "2026-10-08T11:45+03:00"),
    ("بكرة 7:45 الصبح", "2026-10-08T07:45+03:00"),
    # 24-hour forms need no period
    ("بكرة الساعة 15:30", "2026-10-08T15:30+03:00"),
    ("بكرة الساعة 07:00", "2026-10-08T07:00+03:00"),
    ("بكرة الساعة 19", "2026-10-08T19:00+03:00"),
    # weekdays (ref is Wednesday)
    ("يوم الخميس الساعة 5 العصر", "2026-10-08T17:00+03:00"),
    ("الخميس الجاي الساعة 5 العصر", "2026-10-08T17:00+03:00"),
    ("الحد الساعة 9 الصبح", "2026-10-11T09:00+03:00"),
    ("الأربع الجاي الساعة 9 الصبح", "2026-10-14T09:00+03:00"),
    ("الأربع ده الساعة 9 بالليل", "2026-10-07T21:00+03:00"),
    ("next friday at 8 am", "2026-10-09T08:00+03:00"),
    # explicit dates (D/M order), DST change on 30 Oct
    ("15/10 الساعة 3 العصر", "2026-10-15T15:00+03:00"),
    ("15-10-2026 الساعة 3 العصر", "2026-10-15T15:00+03:00"),
    ("يوم 30 أكتوبر الساعة 9 الصبح", "2026-10-30T09:00+02:00"),
    ("5 يناير الساعة 9 الصبح", "2027-01-05T09:00+02:00"),
    ("october 20 at 6 pm", "2026-10-20T18:00+03:00"),
    ("2026-11-01 الساعة 8 الصبح", "2026-11-01T08:00+02:00"),
    # time only -> next occurrence
    ("الساعة 11 الصبح", "2026-10-07T11:00+03:00"),
    ("الساعة 9 الصبح", "2026-10-08T09:00+03:00"),
    ("الساعة 2 بالليل", "2026-10-08T02:00+03:00"),
    ("7 الصبح", "2026-10-08T07:00+03:00"),
    # relative offsets
    ("بعد ساعة", "2026-10-07T11:00+03:00"),
    ("بعد ساعتين", "2026-10-07T12:00+03:00"),
    ("بعد ساعة ونص", "2026-10-07T11:30+03:00"),
    ("بعد نص ساعة", "2026-10-07T10:30+03:00"),
    ("بعد ربع ساعة", "2026-10-07T10:15+03:00"),
    ("بعد 10 دقايق", "2026-10-07T10:10+03:00"),
    ("بعد عشر دقايق", "2026-10-07T10:10+03:00"),
    ("in 20 minutes", "2026-10-07T10:20+03:00"),
    ("in an hour", "2026-10-07T11:00+03:00"),
    # a date mentioned twice consistently
    ("بكرة الخميس الساعة 5 العصر", "2026-10-08T17:00+03:00"),
]


@pytest.mark.parametrize("text, expected", RESOLVED)
def test_resolved(text, expected, ref):
    assert run(text, ref) == (expected, [])


GAPS = [
    # missing / vague time
    ("فكرني اكلم ماما", ["time"]),
    ("بكرة الصبح", ["time"]),
    ("بكرة بدري", ["time"]),
    ("بعدين", ["time"]),
    ("الساعة 7 أو 8 الصبح", ["time"]),
    # missing AM/PM
    ("بكرة الساعة 7", ["am_pm"]),
    ("الساعة 6:30", ["am_pm"]),
    ("tomorrow at 7", ["am_pm"]),
    ("بكرة الصبح الساعة 7 بالليل", ["am_pm"]),
    # period inconsistent with hour
    ("بكرة الساعة 8 الضهر", ["time"]),
    # ambiguous dates
    ("الأربع الساعة 9 بالليل", ["date"]),  # bare weekday == today
    ("يوم 15 الساعة 3 العصر", ["date"]),  # day of month only
    ("الأسبوع الجاي", ["date", "time"]),
    ("بكرة الجمعة الساعة 5 العصر", ["date"]),  # tomorrow is Thursday: conflict
    ("31/2 الساعة 5 العصر", ["date"]),  # invalid date
    ("بكرة الساعة 2 بالليل", ["date"]),  # which night?
    ("بكرة نص الليل", ["date"]),
    # past
    ("النهارده الساعة 8 الصبح", ["past"]),
    ("1/1/2026 الساعة 8 الصبح", ["past"]),
    # combined
    ("بكرة الساعة 7 بعد ساعة", ["time"]),
]


@pytest.mark.parametrize("text, expected", GAPS)
def test_gaps(text, expected, ref):
    assert run(text, ref) == (None, expected)


@pytest.mark.parametrize(
    "text",
    ["اشتري 2 لتر لبن", "اشتري خمسة كيلو", "اكلم 3 عملاء", "ادفع 500 جنيه"],
)
def test_numbers_that_are_not_times(text, ref):
    assert run(text, ref) == (None, ["time"])


def test_dst_relative_offset_crosses_change():
    ref = cairo("2026-10-29T23:30")  # last +03:00 evening
    t = parse_temporal(norm("بعد ساعة"), ref)
    # one elapsed hour later; the wall clock falls back at midnight, so it reads 23:30 again
    utc = UTC
    assert (t.datetime.astimezone(utc) - ref.astimezone(utc)).total_seconds() == 3600
    assert t.datetime.isoformat(timespec="minutes") == "2026-10-29T23:30+02:00"


def test_next_occurrence_inside_repeated_hour():
    # 23:40 (+03:00) -> "الساعة 23:35" is next met in the repeated hour at 23:35 (+02:00)
    ref = cairo("2026-10-29T23:40")
    t = parse_temporal(norm("الساعة 23:35"), ref)
    assert t.datetime.isoformat(timespec="minutes") == "2026-10-29T23:35+02:00"
    assert is_after(t.datetime, ref)


def test_reference_with_fixed_offset_still_uses_cairo_rules():
    from datetime import datetime, timedelta

    ref = datetime(2026, 10, 29, 10, 0, tzinfo=timezone(timedelta(hours=3)))
    t = parse_temporal(norm("بكرة الساعة 9 الصبح"), ref)
    assert t.datetime.isoformat() == "2026-10-30T09:00:00+02:00"


@pytest.mark.parametrize(
    "hour, period, expected",
    [(12, "am", (0, False)), (12, "pm", (12, False)), (12, "morning", None), (1, "noon", (13, False)),
     (12, "night", (0, True)), (11, "night", (23, False)), (4, "maghrib", None), (6, "afternoon", (18, False))],
)
def test_period_table(hour, period, expected):
    assert to_24h(hour, period) == expected
