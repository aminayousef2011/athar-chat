"""Contributor kit (athar.data.contrib): cards -> collection -> blind sheet -> labelled dataset."""

import csv
from pathlib import Path

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


def test_check_author_flags_the_typing_moment_copies_and_dates_on_out_of_scope(tmp_path):
    openpyxl = __import__("pytest").importorskip("openpyxl")

    C.write_kit(tmp_path)
    cards = tmp_path / "scenario_cards.csv"
    book = tmp_path / "author.xlsx"
    C.author_sheet(cards, book)
    wb = openpyxl.load_workbook(book)
    ws = wb["requests"]
    header = [c.value for c in ws[1]]
    assert "you are typing at (NOT the reminder time)" in header and "imagine_it_is_ar" not in header
    plan = {c["card_id"]: c for c in C.make_cards()}
    col = header.index("text") + 1
    rows = list(ws.iter_rows(min_row=2))
    by = {(r[header.index("card_id")].value, r[header.index("version")].value): r for r in rows}
    pair = next(cid for (cid, v) in by if v == "B")
    oos = next(cid for (cid, v) in by if plan[cid]["route_plan"] == "ABSTAIN")
    from datetime import datetime
    when = datetime.fromisoformat(plan[pair]["ref_datetime"])
    copied = f"فكرني أبعت الريبورت يوم {when.day} الساعة {when.hour % 12 or 12}:{when.minute:02d}"
    by[(pair, "A")][col - 1].value = copied
    by[(pair, "B")][col - 1].value = copied + " "
    o_when = datetime.fromisoformat(plan[oos]["ref_datetime"])
    by[(oos, "A")][col - 1].value = f"شغل مزيكا يوم {o_when.day} الساعة {o_when.hour % 12 or 12}:{o_when.minute:02d}"
    wb.save(book)
    r = C.check_author(book, cards)
    text = "\n".join(r["problems"])
    assert f"{pair}-A: asks for the typing moment" in text
    assert f"{pair}-B: almost the same as version A" in text
    assert f"{oos}-A: out-of-scope request carries the typing moment" in text
    assert "core row is empty" in text                               # the rest is not written yet
    # a natural pair (a later time in A, B written from scratch) raises nothing for its card
    other = next(cid for (cid, v) in by if v == "B" and cid != pair)
    by[(other, "A")][col - 1].value = "فكرني بكرة الساعة 9 الصبح أكلم العميل"
    by[(other, "B")][col - 1].value = "ابقى فكرني اكلم العميل بكرة"
    wb.save(book)
    assert not [p for p in C.check_author(book, cards)["problems"] if p.startswith(other)]


def test_labelling_workbook_keeps_datetimes_as_text_and_builds(tmp_path):
    openpyxl = __import__("pytest").importorskip("openpyxl")

    C.write_kit(tmp_path)
    cards = tmp_path / "scenario_cards.csv"
    book = tmp_path / "author.xlsx"
    C.author_sheet(cards, book)
    wb = openpyxl.load_workbook(book)
    ws = wb["requests"]
    header = [c.value for c in ws[1]]
    ws.cell(row=2, column=header.index("text") + 1, value="فكرني بكرة الساعة 9 الصبح اراجع الـ backlog")
    wb.save(book)
    lab = tmp_path / "label.xlsx"
    assert C.import_contributions(book, cards, lab, source="author")["rows"] == 1
    wb = openpyxl.load_workbook(lab)
    ws = wb["label"]
    h = [c.value for c in ws[1]]
    assert ws.cell(row=2, column=h.index("datetime") + 1).number_format == "@"
    assert len(ws.data_validations.dataValidation) == 3                    # intent, route, reason
    ref = ws.cell(row=2, column=h.index("ref_datetime") + 1).value
    from datetime import datetime, timedelta
    when = (datetime.fromisoformat(ref) + timedelta(days=1)).replace(hour=9, minute=0)
    for k, v in {"intent": "create_reminder", "route": "ACT", "datetime": when.strftime("%Y-%m-%d %H:%M"),
                 "reminder_text": "اراجع الـ backlog"}.items():
        ws.cell(row=2, column=h.index(k) + 1, value=v)
    wb.save(lab)
    r = C.build_dataset(lab, tmp_path / "athar.jsonl")
    assert r["items"] == 1 and not r["errors"]
    item = load_items(tmp_path / "athar.jsonl")[0]
    assert item.source == "author" and item.route == "ACT" and item.args.datetime.hour == 9


def test_agreement_sheet_as_xlsx_round_trips_into_kappa(tmp_path):
    openpyxl = __import__("pytest").importorskip("openpyxl")
    from athar.data import agreement as ag

    items = load_items(Path(__file__).resolve().parents[1] / "data" / "pilot" / "pilot.jsonl")[:3]
    sheet = tmp_path / "nested" / "sheet.xlsx"                 # the folder does not exist yet
    ag.write_sheet(items, sheet)
    wb = openpyxl.load_workbook(sheet)
    ws = wb["label"]
    h = [c.value for c in ws[1]]
    assert "family_id" not in h and ws.cell(row=2, column=h.index("datetime") + 1).number_format == "@"
    for i, it in enumerate(items, start=2):                    # the second annotator agrees with gold
        ws.cell(row=i, column=h.index("intent") + 1, value=it.intent)
        ws.cell(row=i, column=h.index("route") + 1, value=it.route)
        ws.cell(row=i, column=h.index("clarify_targets") + 1, value=" ".join(it.clarify_targets))
        ws.cell(row=i, column=h.index("unsupported_reason") + 1, value=it.unsupported_reason or "")
        if it.args.datetime:
            ws.cell(row=i, column=h.index("datetime") + 1,
                    value=it.args.datetime.astimezone(ag.ZoneInfo(it.timezone)).strftime("%Y-%m-%d %H:%M"))
        ws.cell(row=i, column=h.index("reminder_text") + 1, value=it.args.reminder_text or "")
    wb.save(sheet)
    labels, errors = ag.read_labels(sheet)
    assert not errors and set(labels) == {it.id for it in items}
    assert all(labels[it.id] == ag.gold_label(it) for it in items)

    # Excel turned a typed "3/11/2026 17:00" into 11 March: refused, never read as a wrong date
    import datetime as dt

    act = next(i for i, it in enumerate(items, start=2) if it.route == "ACT")
    ws.cell(row=act, column=h.index("datetime") + 1, value=dt.datetime(2026, 3, 11, 17, 0))
    wb.save(sheet)
    labels, errors = ag.read_labels(sheet)
    assert len(errors) == 1 and "day and month may be swapped" in errors[0]
    assert items[act - 2].id not in labels


def test_build_unpairs_versions_that_are_not_a_minimal_pair(tmp_path):
    cols = (*C.SHEET_COLUMNS, *C.LABEL_EXTRA)
    base = dict.fromkeys(cols, "")
    base.update(ref_datetime="2026-10-18T07:40:00+03:00", timezone="Africa/Cairo", family_id="fam-c001",
                pair_id="pair-c001", source="author", domain="general")
    a = {**base, "id": "c001-a", "text": "فكرني بكرة الساعة 9 الصبح اشتري عيش", "intent": "create_reminder",
         "route": "ACT", "datetime": "2026-10-19 09:00", "reminder_text": "اشتري عيش"}
    b = {**base, "id": "c001-b", "text": "صحيني بكرة", "intent": "create_alarm", "route": "CLARIFY",
         "clarify_targets": "time"}
    _write(tmp_path / "s.csv", [a, b])
    r = C.build_dataset(tmp_path / "s.csv", tmp_path / "out.jsonl")
    assert r["items"] == 2 and r["unpaired"] == ["pair-c001: different tools (create_reminder / create_alarm)"]
    items = load_items(tmp_path / "out.jsonl")
    assert all(it.pair_id is None and it.family_id == "fam-c001" for it in items)   # still one family
    assert [it.text for it in items] == [a["text"], b["text"]]                       # texts untouched
    assert "pair dropped" in items[1].notes


def test_form_script_selects_cards_and_uses_the_natural_intro(tmp_path):
    import json
    import re

    C.write_kit(tmp_path)
    cards = tmp_path / "scenario_cards.csv"
    chosen = {"c001", "c002", "c003", "c126", "c142"}
    n = C.form_script(cards, tmp_path / "f.gs", wave="2", people=2, card_ids=chosen, intro=C.FORM_INTRO_NATURAL)
    kit = json.loads(re.search(r"const KIT = (.*?);\n\nfunction", (tmp_path / "f.gs").read_text(encoding="utf-8"),
                               re.S).group(1))
    asked = {q["title"][1:5] for f in kit["forms"] for p in f["pages"] for q in p["questions"]}
    assert asked == chosen and n["forms"] == 2 and kit["intro"] == C.FORM_INTRO_NATURAL
    assert all("(wave 2, block" in f["title"] for f in kit["forms"])


def _screen_fixture(tmp_path):
    C.write_kit(tmp_path)
    q = {("c001", "A"): "[c001-A] Write the full request", ("c001", "B"): "[c001-B] Write it again, as the card says",
         ("c002", "A"): "[c002-A] Write the full request", ("c002", "B"): "[c002-B] Write it again, as the card says",
         ("c126", "A"): "[c126-A] Write the full request"}
    natural = {("c001", "A"): "فكرني بكرة الساعة 9 الصبح احضر الستاند اب", ("c001", "B"): "فكرني احضر الستاند اب",
               ("c002", "A"): "حطلي تذكير السبت الجاي 6 بالليل ابعت التقرير", ("c002", "B"): "فكرني ابعت التقرير 8",
               ("c126", "A"): "امسح المنبه بتاع بكرة"}
    other = {("c001", "A"): "عايزك تفكرني الحد الصبح بدري عندي ميتنج", ("c001", "B"): "ابقى فكرني بالميتنج",
             ("c002", "A"): "يوم السبت على 6 المغرب فكرني بالريبورت", ("c002", "B"): "متنسانيش الريبورت الساعة 8",
             ("c126", "A"): "شيل المنبه اللي ظبطته امبارح"}
    copy = {k: v + " ولو ينفع" for k, v in other.items()}                          # near-duplicate of "other"
    years = {("c001", "A"): "Set a reminder (Standup) يوم ٢٤ أكتوبر ٢٠٢٦ الساعة ٩:٠٠",   # machine formatting
             ("c001", "B"): "Reminder (Daily Standup) يوم ٢٤ أكتوبر ٢٠٢٦",
             ("c002", "A"): "تذكير بالتقرير الأسبوعي يوم السبت ٢٤ أكتوبر ٢٠٢٦ الساعة ٦:٠٠ مساءً",
             ("c002", "B"): "ذكّرني بإرسال التقرير الأسبوعي للمدير", ("c126", "A"): "أرجو حذف المنبه الخاص بالغد"}
    b_is_a = {("c001", "A"): "صحيني بدري قبل الاجتماع بنص ساعة", ("c001", "B"): "صحيني بدري قبل الاجتماع بنص ساعة",
              ("c002", "A"): "خليك فاكر تقولي على الشيت بتاع الحسابات", ("c002", "B"): "خليك فاكر تقولي على الشيت بتاع الحسابات",
              ("c126", "A"): "لغي منبه الفجر"}
    author = {("c001", "A"): "نبهني على الساعة عشرة كده عشان الكول", ("c001", "B"): "نبهني على الكول",
              ("c002", "A"): "عاوز ريمايندر للتاسك بتاعة الديزاين", ("c002", "B"): "ريمايندر للديزاين لو سمحت",
              ("c126", "A"): "مش عايز منبه الجيم خلاص"}
    subs = [("P01", natural), ("P02", other), ("P03", copy), ("P04", years), ("P05", b_is_a), ("P01", author),
            ("AUTHOR", author)]
    export = [{"Timestamp": f"2026/10/2{i}", "Consent": "yes", "Contributor code (for example P03)": code,
               **{q[k]: t for k, t in texts.items()}} for i, (code, texts) in enumerate(subs)]
    _write(tmp_path / "export.csv", export)
    return tmp_path / "export.csv", tmp_path / "scenario_cards.csv", natural


def test_check_contrib_applies_every_screening_rule(tmp_path):
    export, cards, natural = _screen_fixture(tmp_path)
    r = C.check_contrib(export, cards, reference_texts=[natural[("c126", "A")]], exclude_codes=["author"],
                        training_cards={"c002"})
    why = {s["submission"]: " ".join(s["reasons"]) for s in r["submissions"]}
    assert why["P01"] == "" and why["P02"].startswith("R1") and why["P03"].startswith("R1")   # R1 both copies
    assert "R2" in why["P04"] and "R4" in why["P05"]
    assert "R3 repeated" in why["P01#2"] and "R3 excluded code" in why["AUTHOR"]
    kept = {(k["card_id"], k["version"]): k for k in r["kept"]}
    assert set(kept) == {("c001", "A"), ("c001", "B"), ("c002", "A"), ("c002", "B")}       # c126-A copies a reference
    assert kept[("c002", "A")]["comment"] and not kept[("c001", "A")]["comment"]          # training scenario marked
    assert r["writers"] == ["P01"]


def test_check_contrib_cli_writes_an_importable_collection(tmp_path, capsys):
    export, cards, _ = _screen_fixture(tmp_path)
    ref = tmp_path / "ref.jsonl"
    ref.write_text('{"text": "امسح المنبه بتاع بكرة"}\n', encoding="utf-8")
    assert main(["check-contrib", str(export), "--cards", str(cards), "--out", str(tmp_path / "kept.csv"),
                 "--reference", str(ref), "--exclude-codes", "AUTHOR", "--report", str(tmp_path / "r.json")]) == 0
    assert "4 messages from 1 writers" in capsys.readouterr().out
    r = C.import_contributions(tmp_path / "kept.csv", cards, tmp_path / "label.csv")
    assert r["rows"] == 4 and not r["skipped"]
