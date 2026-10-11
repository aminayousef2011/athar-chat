"""Contributor kit (athar.data.contrib): cards -> collection -> blind sheet -> labelled dataset."""

import csv

from athar.data import contrib as C
from athar.data.cli import main
from athar.data.io import load_items


def _rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_cards_meet_the_annotation_guide_targets():
    cards = C.make_cards()
    pairs = [c for c in cards if c["route_plan"] == "ACT+CLARIFY"]
    abstain = [c for c in cards if c["route_plan"] == "ABSTAIN"]
    assert (len(pairs), len(abstain)) == (125, 50)                       # 125 ACT + 125 CLARIFY + 50 ABSTAIN
    pm_items = sum(2 if c["version_b"] else 1 for c in cards if c["domain"] == "project_management")
    assert 0.65 <= pm_items / 300 <= 0.75
    for gap in ("date", "time", "am_pm", "past", "reminder_text"):
        assert sum(gap in c["intended_gaps"].split() for c in pairs) >= 10
    for reason in C.REASON_COUNTS:
        assert sum(c["intended_reason"] == reason for c in abstain) >= 5
    assert {c["ref_datetime"][-6:] for c in cards} == {"+03:00", "+02:00"}  # both sides of the DST change
    assert not any(any("؀" <= ch <= "ۿ" for ch in c["situation"] + c["version_a"] + c["version_b"])
                   for c in cards)                                       # no Arabic wording to copy
    assert C.make_cards() == cards                                       # deterministic


def test_round_trip_from_collection_to_dataset(tmp_path, capsys):
    assert main(["cards", "--out", str(tmp_path)]) == 0
    sheet = _rows(tmp_path / "collection_sheet.csv")
    assert len(sheet) == 300
    # c001 is a reminder pair card (B: no time); the last card is out of scope
    filled = {("c001", "A"): "فكرني بكرة الساعة 9 الصبح أجهز الـ update بتاع الـ standup",
              ("c001", "B"): "فكرني بكرة أجهز الـ update بتاع الـ standup",
              ("c175", "A"): "اعمل ticket على Jira للبج ده"}
    for r in sheet:
        if (r["card_id"], r["version"]) in filled:
            r["text"], r["contributor"] = filled[(r["card_id"], r["version"])], "P01"
    _write(tmp_path / "filled.csv", sheet)
    out = tmp_path / "label.csv"
    assert main(["import-contrib", str(tmp_path / "filled.csv"), "--cards", str(tmp_path / "scenario_cards.csv"),
                 "--out", str(out)]) == 0
    rows = _rows(out)
    assert [r["id"] for r in rows] == ["c001-a", "c001-b", "c175-a"]
    assert "intended_intent" not in rows[0] and rows[0]["intent"] == ""   # blind: no plan, no labels
    assert rows[0]["pair_id"] == rows[1]["pair_id"] == "pair-c001" and rows[2]["pair_id"] == ""
    assert rows[0]["code_switched"] == "yes" and rows[0]["source"] == "contributor"
    # the annotator labels; c175 is deliberately labelled against its card to see the check
    rows[0].update(intent="create_reminder", route="ACT", datetime="2026-10-19 09:00",
                   reminder_text="أجهز الـ update بتاع الـ standup")
    rows[1].update(intent="create_reminder", route="CLARIFY", clarify_targets="time")
    rows[2].update(intent="unsupported", route="ABSTAIN", unsupported_reason="other_tool")
    _write(out, rows)
    data = tmp_path / "athar.jsonl"
    assert main(["build", str(out), "--out", str(data), "--cards", str(tmp_path / "scenario_cards.csv")]) == 0
    items = load_items(data)
    assert [(it.id, it.route) for it in items] == [("c001-a", "ACT"), ("c001-b", "CLARIFY"), ("c175-a", "ABSTAIN")]
    assert all(it.source == "contributor" for it in items) and items[0].domain == "project_management"
    assert "0 labels differ" in capsys.readouterr().out


def test_build_reports_contract_violations_and_skips_them(tmp_path):
    cols = (*C.SHEET_COLUMNS, *C.LABEL_EXTRA)
    row = dict.fromkeys(cols, "")
    row.update(id="c001-a", text="فكرني بكرة أشتري عيش", ref_datetime="2026-10-18T07:40:00+03:00",
               timezone="Africa/Cairo", family_id="fam-c001", intent="create_reminder", route="ACT")
    _write(tmp_path / "s.csv", [row])                   # ACT without a datetime breaks the contract
    r = C.build_dataset(tmp_path / "s.csv", tmp_path / "out.jsonl")
    assert r["items"] == 0 and "ACT items need args.datetime" in r["errors"][0]


def test_wave_one_covers_the_locked_test_split_with_margin():
    w1 = [c for c in C.make_cards() if c["wave"] == "1"]
    pairs = [c for c in w1 if c["version_b"]]
    abstain = [c for c in w1 if c["route_plan"] == "ABSTAIN"]
    assert (len(pairs), len(abstain)) == (60, 30)          # >= test targets 50 / 50 / 20
    req = 2 * len(pairs) + len(abstain)
    pm = sum(2 if c["version_b"] else 1 for c in w1 if c["domain"] == "project_management")
    assert req == 150 and 0.65 <= pm / req <= 0.75
    for gap in ("date", "time", "am_pm", "past", "reminder_text"):
        assert sum(gap in c["intended_gaps"].split() for c in pairs) >= 8
    assert {c["intended_reason"] for c in abstain} == set(C.REASON_COUNTS)
    assert {c["intended_intent"] for c in pairs} == {"create_reminder", "create_alarm"}


def test_form_script_and_form_export_import(tmp_path):
    import json
    import re

    C.write_kit(tmp_path)
    cards = tmp_path / "scenario_cards.csv"
    n = C.form_script(cards, tmp_path / "make_forms.gs", wave="1", people=5)
    assert n == {"forms": 5, "requests": 150}
    js = (tmp_path / "make_forms.gs").read_text(encoding="utf-8")
    kit = json.loads(re.search(r"const KIT = (.*?);\n\nfunction", js, re.S).group(1))
    titles = [q["title"] for f in kit["forms"] for p in f["pages"] for q in p["questions"]]
    assert len(titles) == len(set(titles)) == 150                 # every request asked exactly once
    for page in (p for f in kit["forms"] for p in f["pages"]):      # A and B of a card on one page
        assert len({q["title"][1:5] for q in page["questions"]}) == 1
    # a Google Forms export: one row per contributor, one column per question
    q = [t for t in titles if t.startswith("[c")][:3]
    export = [{"Timestamp": "2026/10/10", "Consent": "yes", "Contributor code (for example P03)": "P01",
               q[0]: "فكرني بكرة الساعة 9 الصبح أبعت الـ report", q[1]: "", q[2]: "فكرني أبعت الـ report"}]
    _write(tmp_path / "export.csv", export)
    rows = C.read_collection(tmp_path / "export.csv")
    assert [(r["card_id"], r["version"], r["contributor"]) for r in rows] == [
        (re.match(r"\[(c\d{3})-([AB])\]", t).group(1), re.match(r"\[(c\d{3})-([AB])\]", t).group(2), "P01") for t in q]
    r = C.import_contributions(tmp_path / "export.csv", cards, tmp_path / "label.csv")
    assert r["rows"] == 2 and any("no text" in s for s in r["skipped"])


def test_author_workbook_round_trip(tmp_path):
    openpyxl = __import__("pytest").importorskip("openpyxl")

    C.write_kit(tmp_path)
    cards = tmp_path / "scenario_cards.csv"
    book = tmp_path / "author.xlsx"
    assert C.author_sheet(cards, book) == {"requests": 150, "core": 120}
    wb = openpyxl.load_workbook(book)
    ws = wb["requests"]
    header = [c.value for c in ws[1]]
    rows = [dict(zip(header, (c.value for c in r), strict=True)) for r in ws.iter_rows(min_row=2)]
    core = [r for r in rows if r["core"] == "1"]
    assert len(core) == 120 and all(r["core"] == "1" for r in rows[:120])      # core first
    plan = {c["card_id"]: c for c in C.make_cards()}
    routes = [("ABSTAIN" if plan[r["card_id"]]["route_plan"] == "ABSTAIN" else "ACT" if r["version"] == "A"
               else "CLARIFY") for r in core]
    assert (routes.count("ACT"), routes.count("CLARIFY"), routes.count("ABSTAIN")) == (50, 50, 20)
    pm = sum(plan[r["card_id"]]["domain"] == "project_management" for r in core) / 120
    assert 0.65 <= pm <= 0.75
    # the author writes two requests; the workbook is imported as source=author
    first = rows[0]
    ws.cell(row=2, column=header.index("text") + 1, value="فكرني بكرة الساعة 9 الصبح أراجع نتايج الـ UAT")
    wb.save(book)
    r = C.import_contributions(book, cards, tmp_path / "label.csv", source="author")
    assert r["rows"] == 1
    sheet = _rows(tmp_path / "label.csv")
    assert sheet[0]["id"] == f"{first['card_id']}-{first['version'].lower()}" and sheet[0]["source"] == "author"
    assert sheet[0]["code_switched"] == "yes"
