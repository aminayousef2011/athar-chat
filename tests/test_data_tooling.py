"""Dataset tooling: validator, family splitter + manifest, agreement, stats, CLI."""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
from sklearn.metrics import cohen_kappa_score

from athar.data import agreement as ag
from athar.data.cli import main
from athar.data.crosscheck import crosscheck
from athar.data.io import DatasetError, load_items, read_records
from athar.data.split import (
    LockedSplitError,
    SplitConfig,
    load_split,
    make_split,
    verify_manifest,
    write_manifest,
)
from athar.data.stats import END, START, stats_markdown, write_into_card
from athar.data.validate import validate_records
from athar.timeexpr import cairo

PILOT = Path(__file__).resolve().parents[1] / "data" / "pilot" / "pilot.jsonl"
REF = "2026-10-07T10:00:00+03:00"


def act(i, fam, pair=None, **kw):
    return {"id": f"i{i:03d}", "family_id": fam, "text": f"فكرني بكرة الساعة 7 الصبح أشتري عيش رقم {i}",
            "ref_datetime": REF, "intent": "create_reminder", "route": "ACT", "pair_id": pair,
            "args": {"datetime": "2026-10-08T07:00:00+03:00", "reminder_text": f"أشتري عيش رقم {i}"}, **kw}


def clarify(i, fam, pair=None, **kw):
    return {"id": f"i{i:03d}", "family_id": fam, "text": f"فكرني بكرة أشتري عيش رقم {i}",
            "ref_datetime": REF, "intent": "create_reminder", "route": "CLARIFY", "pair_id": pair,
            "clarify_targets": ["time"], **kw}


def abstain(i, fam, **kw):
    return {"id": f"i{i:03d}", "family_id": fam, "text": f"امسح التذكير رقم {i}", "ref_datetime": REF,
            "intent": "unsupported", "route": "ABSTAIN", "unsupported_reason": "deletion", **kw}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def issues_for(tmp_path, rows):
    return validate_records(read_records(write_jsonl(tmp_path / "d.jsonl", rows)))


def errors(issues):
    return [i.message for i in issues if i.level == "error"]


# --------------------------------------------------------------------------- validator
def test_pilot_is_valid_and_matches_the_shared_core():
    """The pilot doubles as a contract fixture: hand-written gold == oracle-intent core."""
    assert validate_records(read_records(PILOT)) == []
    assert crosscheck(load_items(PILOT)) == []


def test_valid_rows_have_no_issues(tmp_path):
    assert issues_for(tmp_path, [act(1, "f1", "p1"), clarify(2, "f1", "p1"), abstain(3, "f2")]) == []


def test_bad_json_and_schema_errors_are_reported_per_line(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"id": "x"\n' + json.dumps({**act(1, "f1"), "route": "CLARIFY"}) + "\n", encoding="utf-8")
    msgs = [str(i) for i in validate_records(read_records(p))]
    assert any("line 1" in m and "invalid JSON" in m for m in msgs)
    assert any("line 2 (i001)" in m and "clarify_target" in m for m in msgs)


def test_duplicate_id(tmp_path):
    rows = [act(1, "f1"), {**abstain(2, "f2"), "id": "i001"}]
    assert any("duplicate id" in m for m in errors(issues_for(tmp_path, rows)))


def test_offset_must_match_zone_at_that_instant(tmp_path):
    # 30 Oct 2026 is after Cairo's DST ended: +03:00 names a different instant than intended
    row = act(1, "f1", args={"datetime": "2026-10-30T08:00:00+03:00", "reminder_text": "x"})
    row["text"] = "فكرني يوم الجمعة 30 أكتوبر الساعة 8 الصبح"
    msgs = errors(issues_for(tmp_path, [row]))
    assert any("args.datetime offset +03:00" in m and "(+02:00)" in m for m in msgs)


def test_unknown_timezone(tmp_path):
    assert any("unknown IANA timezone" in m
               for m in errors(issues_for(tmp_path, [act(1, "f1", timezone="Africa/Kairo")])))


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([act(1, "f1", "p1")], "exactly 2"),
        ([act(1, "f1", "p1"), clarify(2, "f2", "p1")], "different families"),
        ([act(1, "f1", "p1"), act(2, "f1", "p1")], "one complete (ACT)"),
        ([act(1, "f1", "p1"), {**clarify(2, "f1", "p1"), "intent": "create_alarm"}], "different intents"),
    ],
)
def test_pair_rules(tmp_path, rows, expected):
    assert any(expected in m for m in errors(issues_for(tmp_path, rows)))


def test_same_text_in_two_families_is_an_error(tmp_path):
    a, b = act(1, "f1"), act(2, "f2")
    b["text"] = a["text"].replace("بكرة", "بكره")  # same after normalization
    assert any("same normalized text" in m for m in errors(issues_for(tmp_path, [a, b])))


def test_alts_need_a_reminder_text(tmp_path):
    row = clarify(1, "f1", args={"reminder_text_alts": ["عيش"]})
    assert any("reminder_text_alts" in m for m in errors(issues_for(tmp_path, [row])))


def test_warnings_code_switching_and_foreign_reminder_words(tmp_path):
    row = act(1, "f1", args={"datetime": "2026-10-08T07:00:00+03:00", "reminder_text": "أشتري لبن"})
    row["text"] += " please"
    warns = [i.message for i in issues_for(tmp_path, [row]) if i.level == "warning"]
    assert any("code_switched is False" in w for w in warns)
    assert any("لبن" in w for w in warns)


# --------------------------------------------------------------------------- split
def synthetic(n_pairs=60, n_abstain=25, n_draft_pairs=5):
    rows, i = [], 0
    for f in range(n_pairs + n_draft_pairs):
        src = {"source": "assistant_draft"} if f >= n_pairs else {}
        rows += [act(i, f"f{f:03d}", f"p{f:03d}", **src), clarify(i + 1, f"f{f:03d}", f"p{f:03d}", **src)]
        i += 2
    for f in range(n_abstain):
        rows.append(abstain(i, f"a{f:03d}"))
        i += 1
    return rows


def split_of(tmp_path, rows, cfg=None):
    data = write_jsonl(tmp_path / "data.jsonl", rows)
    records = read_records(data)
    cfg = cfg or SplitConfig(seed=7, test={"ACT": 20, "CLARIFY": 20, "ABSTAIN": 8}, dev=20)
    mpath = tmp_path / "splits" / "manifest.json"
    manifest, short, notes = make_split([r.item for r in records],
                                        {r.raw["id"]: r.raw for r in records}, cfg, data, mpath)
    write_manifest(manifest, mpath)
    return data, mpath, manifest, short, notes


def test_split_meets_targets_keeps_families_whole_and_drafts_out_of_test(tmp_path):
    rows = synthetic()
    _, mpath, m, short, notes = split_of(tmp_path, rows)
    assert short == []
    assert m["splits"]["test"]["counts"] == {"ACT": 20, "CLARIFY": 20, "ABSTAIN": 8}
    # 9/8/3 is proportional, but pairs add ACT and CLARIFY together: dev is close, and noted
    assert m["splits"]["dev"]["counts"] == {"ACT": 8, "CLARIFY": 8, "ABSTAIN": 3}
    assert notes == ["dev: got 8 ACT / 8 CLARIFY / 3 ABSTAIN, proportional target 9 ACT / 8 CLARIFY / 3 ABSTAIN"]
    fam = {r["id"]: r["family_id"] for r in rows}
    where = {}
    for name, s in m["splits"].items():
        for iid in s["items"]:
            assert where.setdefault(fam[iid], name) == name
    drafts = {r["id"] for r in rows if r.get("source") == "assistant_draft"}
    assert not drafts & set(m["splits"]["test"]["items"])
    assert sum(len(s["items"]) for s in m["splits"].values()) == len(rows)
    assert verify_manifest(mpath) == []


def test_split_is_deterministic(tmp_path):
    rows = synthetic()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    m1 = split_of(tmp_path / "a", rows)[2]
    m2 = split_of(tmp_path / "b", rows)[2]
    assert m1 == m2


def test_shortfall_is_reported(tmp_path):
    cfg = SplitConfig(seed=1, test={"ACT": 100, "CLARIFY": 0, "ABSTAIN": 0}, dev=0)
    short = split_of(tmp_path, synthetic(), cfg)[3]
    assert short and short[0].startswith("test")
    assert any("assistant_draft" in s for s in short)


def test_test_split_is_locked_and_hash_checked(tmp_path):
    data, mpath, m, *_ = split_of(tmp_path, synthetic())
    with pytest.raises(LockedSplitError):
        load_split(mpath, "test")
    assert len(load_split(mpath, "test", unlock_test=True)) == 48
    assert [it.id for it in load_split(mpath, "dev")] == list(m["splits"]["dev"]["items"])

    # editing a dev item after the split breaks both the loader and verify
    dev_id = next(iter(m["splits"]["dev"]["items"]))
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r["id"] == dev_id:
            r["notes"] = "edited"
    write_jsonl(data, rows)
    with pytest.raises(DatasetError):
        load_split(mpath, "dev")
    msgs = [i.message for i in verify_manifest(mpath)]
    assert any("changed since the split" in x for x in msgs)
    assert any(dev_id in x and "changed after the split" in x for x in msgs)


def test_hand_edited_manifest_is_rejected(tmp_path):
    _, mpath, m, *_ = split_of(tmp_path, synthetic())
    m["splits"]["train"]["items"].popitem()
    mpath.write_text(json.dumps(m), encoding="utf-8")
    assert any("manifest hash mismatch" in i.message for i in verify_manifest(mpath))


# --------------------------------------------------------------------------- agreement
@pytest.mark.parametrize(
    "a, b",
    [
        (["x", "y", "x", "z", "y", "x"], ["x", "y", "y", "z", "y", "z"]),
        (["ACT"] * 5 + ["CLARIFY"] * 5, ["ACT"] * 4 + ["CLARIFY"] * 6),
    ],
)
def test_kappa_matches_sklearn(a, b):
    assert ag.cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))


def test_kappa_undefined_when_both_raters_use_one_label():
    assert ag.cohen_kappa(["a"] * 4, ["a"] * 4) is None
    assert ag.interpret(None) == "undefined"


def test_sample_is_stratified_family_unique_and_excludes_drafts(tmp_path):
    items = load_items(write_jsonl(tmp_path / "d.jsonl", synthetic()))
    s = ag.sample_items(items, 30, seed=3)
    assert len(s) == 30
    assert all(it.source != "assistant_draft" for it in s)
    # 145 non-draft items (60/60/25) -> 12.4/12.4/5.2 -> largest remainder 13/12/5
    assert Counter(it.route for it in s) == {"ACT": 13, "CLARIFY": 12, "ABSTAIN": 5}
    assert len({it.family_id for it in s}) == 30
    assert [it.id for it in ag.sample_items(items, 30, seed=3)] == [it.id for it in s]


def test_sheet_roundtrip_and_report(tmp_path):
    items = load_items(write_jsonl(tmp_path / "d.jsonl", synthetic()))
    sample = ag.sample_items(items, 12, seed=5)
    sheet = tmp_path / "sheet.csv"
    ag.write_sheet(sample, sheet)
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["intent"] == "" and rows[0]["ref_local"].startswith("يوم الأربع")

    gold = {it.id: it for it in items}
    flipped = None
    for r in rows:  # annotator copies the gold, except one CLARIFY item read as am_pm
        it = gold[r["id"]]
        r.update(intent=it.intent, route=it.route, clarify_targets=" ".join(it.clarify_targets),
                 unsupported_reason=it.unsupported_reason or "",
                 datetime=it.args.datetime.strftime("%Y-%m-%d %H:%M") if it.args.datetime else "",
                 reminder_text=it.args.reminder_text or "")
        if it.route == "CLARIFY" and flipped is None:
            r["clarify_targets"], flipped = "am_pm", it.id
    rows.append({"id": "nope", "intent": "", "route": ""})
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ag.SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    labels, errs = ag.read_labels(sheet)
    assert errs == [f"row {len(rows) + 1} (nope): not annotated"]
    rep = ag.agreement(items, labels)
    assert rep["n_compared"] == 12
    assert rep["intent"]["kappa"] == pytest.approx(1.0)
    assert rep["route"]["agreement"] == 1.0
    assert rep["targets_exact_set"]["agreement"] < 1.0
    assert rep["datetime_same_instant"]["agreement"] == 1.0
    assert rep["reminder_text_same"]["agreement"] == 1.0
    assert [d["id"] for d in rep["disagreements"]] == [flipped]
    md = ag.report_markdown(rep, "self-review", "2026-10-07")
    assert "| intent | 12 | 100.0% | 1.000 | almost perfect |" in md


def test_read_labels_reads_day_month_dates_and_ignores_dates_off_act(tmp_path):
    sheet = tmp_path / "s.csv"
    rows = [{"id": "a", "intent": "create_alarm", "route": "ACT", "datetime": "3/11/2026 17:00"},
            {"id": "b", "intent": "create_alarm", "route": "ACT", "datetime": "24/10/2026 8:05"},
            {"id": "c", "intent": "create_alarm", "route": "CLARIFY", "clarify_targets": "am_pm",
             "datetime": "22/10/2026"},                 # a date on a CLARIFY row: ignored, row kept
            {"id": "d", "intent": "create_alarm", "route": "ACT", "datetime": "2026-11-03 17:00"}]
    with sheet.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ag.SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    labels, errs = ag.read_labels(sheet)
    assert not errs
    assert labels["a"].datetime == labels["d"].datetime == cairo("2026-11-03T17:00")   # 3/11 = 3 November
    assert labels["b"].datetime == cairo("2026-10-24T08:05")
    assert labels["c"].datetime is None and labels["c"].targets == {"am_pm"}


def test_read_labels_rejects_values_outside_the_contract(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [{"id": "a", "intent": "create_timer", "route": "ACT"},
                    {"id": "b", "intent": "create_alarm", "route": "CLARIFY", "clarify_targets": ["when"]}])
    labels, errs = ag.read_labels(p)
    assert labels == {}
    assert len(errs) == 2


# --------------------------------------------------------------------------- stats + CLI
def test_stats_block_is_written_into_the_card(tmp_path):
    card = tmp_path / "card.md"
    card.write_text(f"# Card\n\n{START}\nold\n{END}\n\nrest\n", encoding="utf-8")
    md = stats_markdown(load_items(PILOT))
    write_into_card(card, md)
    text = card.read_text(encoding="utf-8")
    assert "old" not in text and text.endswith("rest\n")
    assert "20 items in 12 families" in text
    assert "| **items** | 20 |" in text


def test_cli_exit_codes(tmp_path, capsys):
    assert main(["validate", str(PILOT)]) == 0
    bad = write_jsonl(tmp_path / "bad.jsonl", [{**act(1, "f1"), "route": "CLARIFY"}])
    assert main(["validate", str(bad)]) == 1
    data, mpath, *_ = split_of(tmp_path, synthetic())
    assert main(["verify", str(mpath)]) == 0
    assert main(["crosscheck", "--pilot", str(data)]) == 2  # annotated data never via --pilot
    capsys.readouterr()
