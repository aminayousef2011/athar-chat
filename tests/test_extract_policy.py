"""Reminder-text extraction, policy routing, response schema."""

import pytest
from pydantic import ValidationError

from athar.extract import canonical_text, extract
from athar.pipeline import FixedIntent, Pipeline, respond_with_intent
from athar.schema import GoldItem, Response


@pytest.mark.parametrize(
    "text, surface",
    [
        ("فكرني بكرة الساعة ٧ الصبح أشتري عيش", "أشتري عيش"),
        ("فكرني أشتري عيش بكرة الساعة ٧ الصبح", "أشتري عيش"),
        ("فكرني إني أكلم ماما بكرة الساعة 5 العصر", "أكلم ماما"),
        ("فكرني بالاجتماع بكرة الساعة 10 الصبح", "الاجتماع"),
        ("لو سمحت فكرني أدفع الفاتورة يوم الخميس الساعة 9 الصبح", "أدفع الفاتورة"),
        ("ممكن تفكرني أكلم أحمد بكرة الساعة 5 العصر عشان عيد ميلاده", "أكلم أحمد عشان عيد ميلاده"),
        ("اعملي تذكير بكرة الساعة 8 بالليل أخد الدوا", "أخد الدوا"),
        ("remind me tomorrow at 7pm to call mom", "call mom"),
        ("فكرني بكرة الساعة 3 العصر meeting مع الـ team", "meeting مع الـ team"),
        ("فكرني أعمل الواجب بكرة الساعة 6 المغرب", "أعمل الواجب"),
        ("فكرني أشتري لبن، عيش بكرة الساعة 7 الصبح", "أشتري لبن، عيش"),
    ],
)
def test_reminder_text(text, surface, ref):
    ext = extract(text, ref)
    assert ext.reminder_text == surface
    assert ext.reminder_text_norm == canonical_text(surface)


@pytest.mark.parametrize("text", ["فكرني بكرة الساعة 7 الصبح", "فكرني بحاجة بكرة الساعة 7 الصبح",
                                  "remind me tomorrow at 7 am"])
def test_missing_reminder_text(text, ref):
    assert "reminder_text" in extract(text, ref).gaps


def test_route_act(ref):
    r = respond_with_intent("create_reminder", "فكرني بكرة الساعة ٧ الصبح أشتري عيش", ref)
    assert r.route == "ACT"
    assert r.tool_call.name == "create_reminder"
    assert r.tool_call.datetime.isoformat() == "2026-10-08T07:00:00+03:00"
    assert r.tool_call.reminder_text == "أشتري عيش"
    assert r.clarify_targets == []


def test_route_clarify_am_pm_question(ref):
    r = respond_with_intent("create_reminder", "فكرني بكرة الساعة ٧ أشتري عيش", ref)
    assert (r.route, r.clarify_targets) == ("CLARIFY", ["am_pm"])
    assert r.message == "الساعة 7 الصبح ولا بالليل؟"


def test_clarify_collects_every_gap_in_fixed_order(ref):
    r = respond_with_intent("create_reminder", "فكرني الأسبوع الجاي", ref)
    assert r.clarify_targets == ["date", "time", "reminder_text"]


def test_alarm_ignores_reminder_text(ref):
    r = respond_with_intent("create_alarm", "صحيني بكرة الساعة 6 الصبح", ref)
    assert r.route == "ACT"
    assert r.tool_call.reminder_text is None
    r = respond_with_intent("create_alarm", "صحيني بكرة", ref)
    assert r.clarify_targets == ["time"]


def test_unsupported_abstains_even_when_arguments_are_complete(ref):
    r = respond_with_intent("unsupported", "فكرني كل يوم الساعة 7 الصبح أشرب ميه", ref)
    assert (r.route, r.tool_call, r.clarify_targets) == ("ABSTAIN", None, [])


def test_pipeline_with_fixed_intent(ref):
    p = Pipeline(FixedIntent(["create_alarm", "unsupported"]))
    out = p.run_batch(["صحيني الساعة 6:30 الصبح", "امسح المنبه"], [ref, ref])
    assert [o.route for o in out] == ["ACT", "ABSTAIN"]
    assert out[0].system == "oracle"


def test_response_schema_rejects_inconsistent_output():
    with pytest.raises(ValidationError):
        Response(system="x", route="ACT", intent="create_alarm", message="")  # no tool_call
    with pytest.raises(ValidationError):
        Response(system="x", route="ABSTAIN", intent="create_alarm", message="")
    with pytest.raises(ValidationError):
        Response(system="x", route="CLARIFY", intent="create_alarm", clarify_targets=[], message="")


GOLD = dict(id="t001", family_id="f001", text="فكرني بكرة الساعة ٧ الصبح أشتري عيش",
            ref_datetime="2026-10-07T10:00:00+03:00", intent="create_reminder", route="ACT",
            args={"datetime": "2026-10-08T07:00:00+03:00", "reminder_text": "أشتري عيش"})


def test_gold_item_valid():
    item = GoldItem(**GOLD)
    assert item.route == "ACT"


@pytest.mark.parametrize(
    "patch",
    [
        {"route": "CLARIFY"},  # CLARIFY without targets
        {"route": "ABSTAIN"},  # ABSTAIN with supported intent
        {"args": {"datetime": "2026-10-06T07:00:00+03:00", "reminder_text": "x"}},  # past
        {"args": {"datetime": "2026-10-08T07:00:00"}},  # naive datetime
        {"intent": "unsupported", "route": "ABSTAIN", "args": {}},  # missing reason
        {"clarify_targets": ["time"]},  # ACT with targets
        {"intent": "create_alarm", "route": "CLARIFY", "clarify_targets": ["reminder_text"], "args": {}},
    ],
)
def test_gold_item_rejects_contract_violations(patch):
    with pytest.raises(ValidationError):
        GoldItem(**{**GOLD, **patch})


@pytest.mark.parametrize("hour, day", [("7", "الصبح"), ("11", "الصبح"), ("12", "الضهر"), ("2", "الضهر"),
                                        ("4", "العصر"), ("5", "العصر")])
def test_am_pm_question_pairs_the_hour_with_its_daytime_period(hour, day, ref):
    r = respond_with_intent("create_reminder", f"فكرني بكرة الساعة {hour} أشتري عيش", ref)
    assert r.message == f"الساعة {hour} {day} ولا بالليل؟"
