"""The demo's logic on the frozen systems and the committed locked-test results, and the
Streamlit app run headless (skipped without the ``ui`` extra)."""

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from athar import demo

ROOT = Path(__file__).resolve().parents[1]
REF = dt.datetime(2026, 10, 19, 10, 0, tzinfo=ZoneInfo("Africa/Cairo"))


@pytest.fixture(scope="module")
def frozen():
    return demo.load_systems(ROOT / "models")


def by_system(text, frozen):
    return {a.system: a for a in demo.answer_all(text, REF, frozen)}


def test_frozen_core_systems_load_verified(frozen):
    assert [s.name for s in frozen] == ["S1", "S2", "S3"]
    assert frozen[0].tau is None and all(s.tau is not None for s in frozen[1:])


def test_noun_form_reminder_shows_the_trigger_shift(frozen):
    a = by_system("حطلي تذكير يوم الخميس الساعة 7 الصبح عشان اراجع خطة السبرينت", frozen)
    assert a["S1"].response.tool_call.name == "create_reminder"
    assert {a["S2"].intent, a["S3"].intent} == {"create_alarm"}
    # every system resolves the same instant: only the tool differs
    assert len({x.response.tool_call.datetime for x in a.values()}) == 1


def test_extractor_text_bug_is_shared_by_every_system(frozen):
    a = by_system("عايز تذكير بكرة الساعة 5 العصر علشان اراجع الميزانية", frozen)
    assert {x.response.tool_call.reminder_text for x in a.values()} == {"عايز علشان اراجع الميزانية"}


def test_routes_for_incomplete_and_out_of_scope_requests(frozen):
    assert {x.response.route for x in by_system("فكرني اكلم العميل", frozen).values()} == {"CLARIFY"}
    assert {x.response.route for x in by_system("امسح التذكير بتاع بكرة", frozen).values()} == {"ABSTAIN"}
    recurring = by_system("فكرني كل يوم الساعة 9 اشرب مية", frozen)
    assert recurring["S1"].response.route == "ABSTAIN"


def test_every_example_gets_one_valid_answer_per_system(frozen):
    for ex in demo.EXAMPLES:
        answers = demo.answer_all(ex.text, REF, frozen)
        assert [a.system for a in answers] == ["S1", "S2", "S3"]
        assert all(a.response.message for a in answers)


def test_naive_reference_time_is_read_in_cairo(frozen):
    naive = demo.answer_all("فكرني بكرة الساعة 9 الصبح ابعت التقرير", REF.replace(tzinfo=None), frozen)
    aware = demo.answer_all("فكرني بكرة الساعة 9 الصبح ابعت التقرير", REF, frozen)
    assert [a.response.tool_call.datetime for a in naive] == [a.response.tool_call.datetime for a in aware]


def test_gate_note(frozen):
    act = by_system("فكرني بكرة الساعة 9 الصبح ابعت التقرير لمديري", frozen)["S2"]
    assert demo.gate_note(act, 0.0) == "no effect"
    assert demo.gate_note(act, 0.99) == "would ask to confirm"
    assert demo.gate_note(act, 0.01) == "passes"
    ask = by_system("فكرني اكلم العميل", frozen)["S2"]
    assert demo.gate_note(ask, 0.99) == "no effect"   # the gate never touches CLARIFY / ABSTAIN


def test_locked_results_are_read_from_the_committed_files():
    summary = {r["system"]: r for r in demo.test_summary(ROOT / "results")}
    assert summary["oracle (gold intent)"]["STS"] == 0.808
    assert summary["S1 keyword rules"]["STS"] == 0.767
    h = {r["comparison"]: r for r in demo.test_comparisons(ROOT / "results")}
    assert h["S2 vs S1"]["significant"] and h["S3 vs S1"]["significant"] and not h["S3 vs S2"]["significant"]
    rows = demo.failures(ROOT / "results")
    assert demo.cause_counts(rows, "oracle") == {"wrong reminder text only": 13, "asked for the wrong slots": 4,
                                                 "executed an incomplete request": 3, "asked although complete": 3}


def test_streamlit_app_runs_headless():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("disagree" in w.value for w in at.warning)          # default example: the trigger shift
    at.selectbox[0].select("فكرني اكلم العميل").run()
    assert not at.exception and not at.warning                      # all three ask for the time
    assert len(at.dataframe) >= 3                                   # results tab: summary, H1-H3, failures
