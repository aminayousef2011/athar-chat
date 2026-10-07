"""Property-based tests (Hypothesis): invariants of the shared core over generated inputs."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from hypothesis import given, settings
from hypothesis import strategies as st

from athar.extract import canonical_text, extract
from athar.normalize import norm, normalize
from athar.timeexpr import is_after

CAIRO = ZoneInfo("Africa/Cairo")
ARABIC = "ابتثجحخدذرسشصضطظعغفقكلمنهويءآأإؤئىة"
EXTRA = "ًٌٍَُِّْـ٠١٢٣٤٥٦٧٨٩0123456789 ،؟.!abcXYZ"
text = st.text(alphabet=ARABIC + EXTRA, max_size=60)

# reference instants across a whole year, including both DST changes
refs = st.datetimes(min_value=datetime(2026, 1, 1), max_value=datetime(2026, 12, 31, 23, 0)).map(
    lambda d: d.replace(second=0, microsecond=0, tzinfo=CAIRO))


@given(text)
def test_normalization_is_idempotent(t):
    assert norm(norm(t)) == norm(t)


@given(text)
def test_offsets_are_monotone_and_in_range(t):
    n = normalize(t)
    assert len(n.offsets) == len(n.text)
    assert all(0 <= o < len(t) for o in n.offsets)
    assert list(n.offsets) == sorted(n.offsets)


@given(text)
def test_normal_form_has_no_diacritics_tatweel_or_double_spaces(t):
    s = norm(t)
    assert not any(c in "ًٌٍَُِّْـ" for c in s)
    assert "  " not in s and s == s.strip()
    assert not any(c in "٠١٢٣٤٥٦٧٨٩" for c in s)


@given(text)
def test_canonical_text_is_idempotent(t):
    c = canonical_text(t)
    if c is not None:
        assert canonical_text(c) == c


@settings(max_examples=300)
@given(refs, st.integers(1, 11))
def test_tomorrow_morning_resolves_to_that_wall_clock(ref, h):
    ext = extract(f"فكرني بكرة الساعة {h} الصبح أشتري عيش", ref)
    dt = ext.datetime
    assert dt is not None and ext.gaps == frozenset()
    local = dt.astimezone(CAIRO)
    assert (local.date(), local.hour, local.minute) == ((ref + timedelta(days=1)).date(), h, 0)
    assert local.utcoffset() == local.replace(tzinfo=CAIRO).utcoffset()  # offset of that instant


@settings(max_examples=300)
@given(refs, st.integers(1, 11), st.sampled_from(["الصبح", "بالليل"]))
def test_resolved_times_are_always_in_the_future(ref, h, period):
    if period == "بالليل" and h < 6:
        return  # after-midnight readings are covered by the contract tests
    ext = extract(f"فكرني الساعة {h} {period} أشتري عيش", ref)
    assert ext.datetime is not None
    assert is_after(ext.datetime, ref)
    # next occurrence, never later; compare instants (same-zone subtraction is wall-clock)
    assert ext.datetime.astimezone(UTC) - ref.astimezone(UTC) <= timedelta(days=1, hours=1)


@settings(max_examples=200)
@given(refs, st.integers(1, 12))
def test_bare_hour_always_asks_am_pm(ref, h):
    assert "am_pm" in extract(f"فكرني بكرة الساعة {h} أشتري عيش", ref).gaps


@settings(max_examples=200)
@given(st.one_of(refs, st.just(datetime(2026, 10, 29, 23, 30, tzinfo=CAIRO))), st.integers(1, 180))
def test_relative_minutes_are_exact(ref, m):
    ext = extract(f"فكرني بعد {m} دقيقة أشتري عيش", ref)
    assert ext.datetime is not None
    assert ext.datetime.astimezone(UTC) - ref.astimezone(UTC) == timedelta(minutes=m)
