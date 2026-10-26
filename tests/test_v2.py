"""v2 systems pre-registered in docs/protocol_natural.md: core v1.1, S1 v1.1, grouped-CV S2/S3, the
S9 hybrid, the evaluation-only manifest, multi-writer import and the natural-phrasing evaluation."""

import json
from pathlib import Path

import pytest

from athar.data import contrib as C
from athar.data.cli import main as data_main
from athar.data.split import LockedSplitError, load_split
from athar.extract import extract as extract_v1
from athar.normalize import norm
from athar.schema import GoldItem
from athar.timeexpr import cairo
from athar.v2 import core, gcv, hybrid, natural
from athar.v2.rules import classify

REPO = Path(__file__).resolve().parents[1]
REF = cairo("2026-10-18T07:40")          # a Sunday
FAST = {"seed": 1, "level": 0.95, "resamples": 200, "permutations": 200, "alpha": 0.05,
        "calibration_bins": 10, "oracle": False, "perturbations": [], "comparisons": []}


# --------------------------------------------------------------------------- core v1.1
@pytest.mark.parametrize("text, content", [
    ("عايز تذكير بكرة الساعة 5 العصر علشان اراجع الميزانية", "اراجع الميزانية"),
    ("اعمللي تذكير بعد بكرة الساعة 9 بالليل عشان اسلم الشيت", "اسلم الشيت"),
    ("محتاج تذكير الخميس الساعة 3 العصر عشان اكلم المورد", "اكلم المورد"),
])
def test_core_v11_strips_want_verbs_and_the_doubled_spelling(text, content):
    assert core.extract(text, REF).reminder_text == content
    assert extract_v1(text, REF).reminder_text != content          # v1 is untouched: it keeps the verb


def test_core_v11_fillers_leave_no_fake_content():
    for text in ("محتاج تذكير كده", "اعمللي تذكير لو ينفع"):
        assert core.extract(text, REF).gaps >= {"time", "reminder_text"}


def test_core_v11_ignores_period_words_inside_a_purpose_clause_after_a_clock_time():
    ext = core.extract("عايز منبه بكرة الساعة 9 الصبح علشان اصحى لصلاة الفجر", REF)
    assert ext.datetime is not None and (ext.datetime.hour, ext.datetime.minute) == (9, 0) and not ext.gaps - {"reminder_text"}
    assert extract_v1("عايز منبه بكرة الساعة 9 الصبح علشان اصحى لصلاة الفجر", REF).datetime is None
    # without a clock time before the clause, a period word still counts
    assert core.extract("فكرني بكرة عشان اكلم ماما الصبح", REF).datetime is None


def test_core_v11_keeps_v1_behaviour_elsewhere():
    for text in ("حطلي تذكير الخميس الجاي الساعة 7 الصبح عشان احضر الستاند اب", "فكرني بحاجة", "صحيني الساعة 7"):
        a, b = core.extract(text, REF), extract_v1(text, REF)
        assert (a.datetime, a.gaps, a.reminder_text) == (b.datetime, b.gaps, b.reminder_text)


# --------------------------------------------------------------------------- S1 v1.1
@pytest.mark.parametrize("text, intent", [
    ("قدم منبه الاجتماع ساعة", "unsupported"),
    ("اخر تذكير مكالمة العميل", "unsupported"),
    ("خلي تذكير الدوا بالليل", "unsupported"),
    ("الغيلي تذكير الستاند اب", "unsupported"),
    ("فكرني اخر النهار اكلم ماما", "create_reminder"),         # «آخر النهار»: a vague time, not a blocker
    ("اخر النهار فكرني اكلم ماما", "create_reminder"),
    ("فكرني اقدم التايم شيت بكرة", "create_reminder"),
    ("صحيني بكرة الساعة 7", "create_alarm"),
])
def test_s1_v11_blockers(text, intent):
    assert classify(norm(text)) == intent


# --------------------------------------------------------------------------- S9
def _item(iid, text, intent, route, targets=(), dt_=None, rt=None, reason=None):
    args = {k: v for k, v in (("datetime", dt_), ("reminder_text", rt)) if v}
    return GoldItem.model_validate({
        "id": iid, "family_id": f"fam-{iid[:4]}", "text": text, "ref_datetime": "2026-10-18T07:40:00+03:00",
        "intent": intent, "route": route, "args": args, "clarify_targets": list(targets),
        "unsupported_reason": reason, "source": "contributor"})


def _call(name, **arguments):
    return {"calls": [{"name": name, "arguments": arguments}]}


def test_s9_lets_the_policy_decide_and_the_llm_extract():
    complete = _item("c001-a-w01", "فكرني بكرة الساعة 9 الصبح اراجع الباكلوج", "create_reminder", "ACT",
                     dt_="2026-10-19T09:00:00+03:00", rt="اراجع الباكلوج")
    incomplete = _item("c001-b-w01", "فكرني اراجع الباكلوج", "create_reminder", "CLARIFY", ["time"])
    # S8's own arguments are used when the request is complete
    r = hybrid.s9_response(complete, _call("create_reminder", datetime="2026-10-19T09:00", reminder_text="أراجع الـ backlog"))
    assert r.route == "ACT" and r.tool_call.reminder_text == "أراجع الـ backlog"
    # S8 executes an incomplete request with a guessed time: S9 asks
    r = hybrid.s9_response(incomplete, _call("create_reminder", datetime="2026-10-19T09:00", reminder_text="x"))
    assert r.route == "CLARIFY" and r.clarify_targets == ["time"]
    # S8 asks although the request is complete: S9 acts with core v1.1's arguments
    r = hybrid.s9_response(complete, _call("ask_clarification", tool="create_reminder", missing=["time"]))
    assert r.route == "ACT" and r.tool_call.reminder_text == "اراجع الباكلوج"
    # S8 declines; S8 answers in plain text (the intent is S1 v1.1's)
    assert hybrid.s9_response(complete, _call("decline", reason="other_tool")).route == "ABSTAIN"
    assert hybrid.s9_response(complete, {"calls": []}).route == "ACT"
    assert hybrid.s8_response({"calls": []}, "Africa/Cairo") is None


# --------------------------------------------------------------------------- grouped-CV freeze
def test_gcv_freeze_verify_and_refuse(split_dir):
    out = split_dir / "models_v2"
    fz = gcv.freeze(split_dir / "manifest.json", split_dir / "train.yaml", out)
    assert set(fz["systems"]) == {"S2-gcv", "S3-gcv"} and gcv.verify(out) == []
    assert all(len(g["cv"]["folds"]) == gcv.FOLDS for s in fz["systems"].values() for g in s["grid"])
    assert gcv.load_recognizer("S2-gcv", out).predict([norm("فكرني أشتري حاجات")]) == ["create_reminder"]
    with pytest.raises(gcv.FreezeError):
        gcv.freeze(split_dir / "manifest.json", split_dir / "train.yaml", out)       # never overwritten
    (out / "S2-gcv.joblib").write_bytes(b"tampered")
    assert any("S2-gcv" in p for p in gcv.verify(out))
    with pytest.raises(gcv.FreezeError):
        gcv.load_recognizer("S2-gcv", out)


# --------------------------------------------------------------------------- data: multi-writer import, lock
def test_multi_writer_import_keeps_every_writer(tmp_path):
    C.write_kit(tmp_path)
    rows = [{"card_id": "c001", "version": v, "contributor": w, "text": f"فكرني {w} {v} اراجع", "comment": ""}
            for w in ("W01", "h18") for v in ("A", "B")]
    C.write_collection(rows, tmp_path / "kept.csv")
    C.import_contributions(tmp_path / "kept.csv", tmp_path / "scenario_cards.csv", tmp_path / "sheet.csv",
                           multi_writer=True)
    import csv
    got = list(csv.DictReader(open(tmp_path / "sheet.csv", encoding="utf-8-sig")))
    assert [r["id"] for r in got] == ["c001-a-w01", "c001-b-w01", "c001-a-h18", "c001-b-h18"]
    assert {r["pair_id"] for r in got} == {"pair-c001-w01", "pair-c001-h18"}
    assert {r["family_id"] for r in got} == {"fam-c001"}


def _natural(tmp_path):
    items = [
        _item("c001-a-w01", "فكرني بكرة الساعة 9 الصبح اراجع الباكلوج", "create_reminder", "ACT",
              dt_="2026-10-19T09:00:00+03:00", rt="اراجع الباكلوج"),
        _item("c001-b-w01", "فكرني اراجع الباكلوج", "create_reminder", "CLARIFY", ["time"]),
        _item("c001-a-w02", "عايز تذكير بكرة الساعة 5 العصر علشان اكلم العميل", "create_reminder", "ACT",
              dt_="2026-10-19T17:00:00+03:00", rt="اكلم العميل"),
        _item("c126-a-w02", "امسح المنبه بتاع بكرة", "unsupported", "ABSTAIN", reason="deletion"),
    ]
    data = tmp_path / "natural.jsonl"
    data.write_text("".join(json.dumps(it.model_dump(mode="json", exclude_defaults=True), ensure_ascii=False) + "\n"
                            for it in items), encoding="utf-8")
    assert data_main(["lock", str(data), "--out", str(tmp_path / "manifest.json"), "--note", "test"]) == 0
    assert data_main(["lock", str(data), "--out", str(tmp_path / "manifest.json"), "--note", "again"]) == 1
    calls = {"c001-a-w01": _call("create_reminder", datetime="2026-10-19T09:00", reminder_text="اراجع الباكلوج"),
             "c001-b-w01": _call("create_reminder", datetime="2026-10-19T09:00", reminder_text="اراجع الباكلوج"),
             "c001-a-w02": _call("ask_clarification", tool="create_reminder", missing=["time"]),
             "c126-a-w02": _call("decline", reason="deletion")}
    traces = []
    for seed in (0, 1):
        t = tmp_path / f"trace{seed}.jsonl"
        t.write_text("".join(json.dumps({"id": i, **c}, ensure_ascii=False) + "\n" for i, c in calls.items()),
                     encoding="utf-8")
        traces.append(t)
    return tmp_path / "manifest.json", traces


def test_locked_manifest_needs_unlock(tmp_path):
    manifest, _ = _natural(tmp_path)
    with pytest.raises(LockedSplitError):
        load_split(manifest, "test")
    assert len(load_split(manifest, "test", unlock_test=True)) == 4
    assert load_split(manifest, "train") == []


def test_natural_evaluation_end_to_end(tmp_path):
    manifest, traces = _natural(tmp_path)
    gcv.freeze(REPO / "data" / "splits" / "manifest.json", tmp_path_train_yaml(tmp_path), tmp_path / "models_v2")
    r = natural.evaluate(manifest, REPO / "data" / "splits" / "manifest.json", REPO / "models",
                         tmp_path / "models_v2", traces, [], cfg=FAST)
    assert r["confirmatory_items"] == 4 and r["writers"] == ["w01", "w02"]
    assert r["hypothesis_tests"].startswith("descriptive only") and "p_holm" not in r["tests"]["N1 S1 vs S2"]
    s = {n: v["sts"] for n, v in r["summary"].items()}
    assert s["S1-v1.1"] == 1.0 and s["S1"] == 0.75          # the want-verb item: v1 keeps «عايز علشان»
    assert s["S9 (seed mean)"] == 1.0 and s["S8 (seed mean)"] == 0.5
    assert r["author_vs_contributor"]["slots"] == 3           # c001-a, c001-b, c126-a: also in the author's test
    assert set(r["sensitivity"]["leave_one_writer_out"]) == {"w01", "w02"}


def tmp_path_train_yaml(tmp_path):
    import yaml

    small = {"seed": 0, "common": {"class_weight": "balanced", "max_iter": 500},
             "S2": {"C": [1], "word_ngram": [[1, 1]]}, "S3": {"C": [1], "word_ngram": [[1, 1]], "char_ngram": [[2, 4]]}}
    p = tmp_path / "train_small.yaml"
    p.write_text(yaml.safe_dump(small), encoding="utf-8")
    return p
