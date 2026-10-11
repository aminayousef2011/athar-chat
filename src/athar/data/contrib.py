"""Collecting the 300 items from native Egyptian Arabic speakers (annotation guide section 2).

    athar-data cards --out contrib                      # scenario cards + empty collection sheet
    athar-data import-contrib contrib/collection_sheet.csv --cards contrib/scenario_cards.csv \\
        --out runs/labelling/sheet.csv                  # blind labelling sheet
    athar-data build runs/labelling/sheet.csv --out data/athar.jsonl
    athar-data form-script --cards contrib/scenario_cards.csv --wave 1 --people 5 \
        --out contrib/make_forms.gs                     # Google Forms, one per contributor

Wave 1 (90 cards, 150 requests) covers the 120-item locked test split with margin: only test
must be native-written (drafts may train), so it is collected first. Wave 2 is the rest.

Contributors never see Arabic example wording: a card describes a situation in English and
says what to include or leave out, and the contributor writes the request in their own words.
The cards cover the guide's targets (125 ACT/CLARIFY pairs + 50 ABSTAIN = 300 items, ~70%
project management, every clarification target and unsupported reason, reference times across
weekdays, times of day and Cairo's DST change). A card's intended label is kept out of the
labelling sheet: the annotator labels what was written, and ``build`` only reports where the
two differ, as a check on the card or the label.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from athar.data.agreement import SHEET_COLUMNS, _cell, _label
from athar.respond import describe_datetime
from athar.schema import GoldItem

TZ = "Africa/Cairo"
CARD_COLUMNS = ("card_id", "wave", "domain", "route_plan", "intended_intent", "intended_gaps", "intended_reason",
                "now_local_en", "now_local_ar", "ref_datetime", "situation", "version_a", "version_b")
COLLECTION_COLUMNS = ("card_id", "version", "contributor", "text", "comment")
LABEL_EXTRA = ("family_id", "pair_id", "domain", "source", "code_switched", "contributor")

# (tool, situation) -- situations, never wording. Reminders say what the reminder is about.
PM_TASKS = [
    ("create_reminder", "you want a reminder to prepare your update before the daily standup"),
    ("create_reminder", "you want a reminder to send the weekly status report to your manager"),
    ("create_reminder", "you want a reminder to follow up with the client about their sign-off on the design"),
    ("create_reminder", "you want a reminder to review the sprint backlog before sprint planning"),
    ("create_reminder", "you want a reminder to chase the vendor about the overdue invoice"),
    ("create_reminder", "you want a reminder to update the Jira tickets before the sprint review"),
    ("create_reminder", "you want a reminder to submit your timesheet"),
    ("create_reminder", "you want a reminder to check the UAT results with the QA team lead"),
    ("create_reminder", "you want a reminder to prepare the slides for the steering committee"),
    ("create_reminder", "you want a reminder to send the meeting minutes from the kickoff"),
    ("create_reminder", "you want a reminder to review the budget variance before finance asks"),
    ("create_reminder", "you want a reminder to confirm the go-live checklist"),
    ("create_reminder", "you want a reminder to call the client about the change request"),
    ("create_reminder", "you want a reminder to update the risk register"),
    ("create_reminder", "you want a reminder to prepare the handover document for operations"),
    ("create_reminder", "you want a reminder to book time for a one-on-one with a team member"),
    ("create_reminder", "you want a reminder to send the release notes"),
    ("create_reminder", "you want a reminder to review the resource plan for next month"),
    ("create_reminder", "you want a reminder to approve the purchase order"),
    ("create_reminder", "you want a reminder to check the milestone dates with the PMO"),
    ("create_alarm", "you need to wake up early for a site visit"),
    ("create_alarm", "you need an alarm to wake up for an early flight to meet a client"),
    ("create_alarm", "you need an alarm a few minutes before an important client call"),
    ("create_alarm", "you need to wake up early to finish the report before the deadline"),
    ("create_alarm", "you need an alarm before the go-live window starts at night"),
    ("create_alarm", "you need an alarm to leave the office in time for a vendor meeting"),
]
GENERAL_TASKS = [
    ("create_reminder", "you want a reminder to buy bread on the way home"),
    ("create_reminder", "you want a reminder to take your medicine"),
    ("create_reminder", "you want a reminder to call your mother"),
    ("create_reminder", "you want a reminder to pay the electricity bill"),
    ("create_reminder", "you want a reminder to pick up the laundry"),
    ("create_reminder", "you want a reminder to renew your car licence"),
    ("create_reminder", "you want a reminder about a friend's birthday"),
    ("create_reminder", "you want a reminder to recharge your phone credit"),
    ("create_alarm", "you need to wake up for the gym"),
    ("create_alarm", "you need to wake up for the dawn prayer"),
    ("create_alarm", "you need an alarm for a nap of a few hours"),
    ("create_alarm", "you need an alarm to wake up for university or a course"),
]
# what to leave out in version B, by clarification target(s)
GAP_INSTRUCTIONS = {
    ("time",): "leave out the clock time, or give only a vague time of day",
    ("am_pm",): "say the hour (1-12) but NOT whether it is morning, afternoon or evening",
    ("date",): "mention the day in a way that could mean more than one day (only a day-of-month number, "
               "'next week', 'the weekend', or a weekday that is today)",
    ("past",): "ask for a time earlier today that has already passed (as if you got the time wrong)",
    ("reminder_text",): "do NOT say what the reminder is about (or say it only vaguely, like 'something')",
    ("time", "reminder_text"): "leave out BOTH the time and what the reminder is about",
    ("date", "am_pm"): "make the day ambiguous AND leave out morning/evening",
}
REMINDER_GAPS = [("time",), ("am_pm",), ("date",), ("past",), ("reminder_text",), ("time",), ("am_pm",),
                 ("time", "reminder_text"), ("date",), ("am_pm",), ("past",), ("date", "am_pm")]
ALARM_GAPS = [("time",), ("am_pm",), ("date",), ("past",), ("am_pm",), ("time",)]
# out-of-scope situations by unsupported reason (annotation guide step 1 order)
UNSUPPORTED = {
    "deletion": {"project_management": ["you want to cancel a reminder you already set about the standup",
                                        "you want to delete tomorrow's alarm for the site visit"],
                 "general": ["you want to delete the alarm you set for the morning"]},
    "modification": {"project_management": ["you want to move the reminder about the client call to a later time",
                                            "you want to change the alarm for the go-live to an hour earlier"],
                     "general": ["you want to change the medicine reminder to the evening"]},
    "recurring": {"project_management": ["you want a reminder before EVERY daily standup",
                                         "you want a reminder every Thursday to send the weekly report"],
                  "general": ["you want an alarm every day for the gym"]},
    "other_person": {"project_management": ["you want the assistant to remind a team member to send their update",
                                            "you want the whole team reminded about the sprint review"],
                     "general": ["you want your brother reminded to pay the bill"]},
    "multi_action": {"project_management": ["in one message you want two reminders: one for the report and one for "
                                            "the client call, at different times"],
                     "general": ["in one message you want an alarm to wake up AND a reminder to buy bread"]},
    "other_tool": {"project_management": ["you want the assistant to create a Jira ticket",
                                          "you want the assistant to book a meeting room",
                                          "you want the assistant to email the client the status report",
                                          "you want the assistant to schedule a meeting in the team calendar"],
                   "general": ["you want the assistant to play some music", "you want the weather for tomorrow"]},
}
REASON_COUNTS = {"deletion": 8, "modification": 8, "recurring": 9, "other_person": 8, "multi_action": 7,
                 "other_tool": 10}


def _ref_times(n: int) -> list[datetime]:
    """Varied reference times: every weekday, morning to late night, before and after Cairo's
    DST change (29 Oct 2026)."""
    tz = ZoneInfo(TZ)
    start = datetime(2026, 10, 18, tzinfo=tz)
    hours = [(7, 40), (9, 15), (11, 5), (13, 30), (16, 45), (19, 20), (21, 50), (23, 35), (8, 0), (18, 10)]
    out = []
    for i in range(n):
        day = start + timedelta(days=(i * 3) % 21)
        h, m = hours[i % len(hours)]
        out.append(day.replace(hour=h, minute=m))
    return out


def make_cards() -> list[dict[str, str]]:
    """125 pair cards (ACT version A + CLARIFY version B) and 50 ABSTAIN cards, deterministic."""
    cards: list[dict[str, str]] = []
    pair_specs: list[tuple[str, str, str]] = []
    for i in range(88):
        tool, sit = PM_TASKS[i % len(PM_TASKS)]
        pair_specs.append(("project_management", tool, sit))
    for i in range(37):
        tool, sit = GENERAL_TASKS[i % len(GENERAL_TASKS)]
        pair_specs.append(("general", tool, sit))
    abstain_specs: list[tuple[str, str, str]] = []
    for reason, k in REASON_COUNTS.items():
        pm, gen = UNSUPPORTED[reason]["project_management"], UNSUPPORTED[reason]["general"]
        n_pm = round(k * 0.7)
        abstain_specs += [("project_management", reason, pm[j % len(pm)]) for j in range(n_pm)]
        abstain_specs += [("general", reason, gen[j % len(gen)]) for j in range(k - n_pm)]
    refs = _ref_times(len(pair_specs) + len(abstain_specs))
    r_i = a_i = 0
    for n, (domain, tool, sit) in enumerate(pair_specs):
        if tool == "create_reminder":
            gaps, r_i = REMINDER_GAPS[r_i % len(REMINDER_GAPS)], r_i + 1
            complete = "the day, the exact time with morning/evening, and what to be reminded of"
        else:
            gaps, a_i = ALARM_GAPS[a_i % len(ALARM_GAPS)], a_i + 1
            complete = "the day and the exact time with morning/evening"
        ref = refs[n]
        cards.append({
            "card_id": f"c{n + 1:03d}", "domain": domain, "route_plan": "ACT+CLARIFY",
            "intended_intent": tool, "intended_gaps": " ".join(gaps), "intended_reason": "",
            "now_local_en": ref.strftime("%A %d %B %Y, %H:%M"), "now_local_ar": describe_datetime(ref),
            "ref_datetime": ref.isoformat(), "situation": sit,
            "version_a": f"Ask the assistant for this with everything it needs: {complete}. The time must be "
                         f"in the future.",
            "version_b": f"Ask for the same thing again, but {GAP_INSTRUCTIONS[gaps]}."})
    base = len(pair_specs)
    for j, (domain, reason, sit) in enumerate(abstain_specs):
        ref = refs[base + j]
        cards.append({
            "card_id": f"c{base + j + 1:03d}", "domain": domain, "route_plan": "ABSTAIN",
            "intended_intent": "unsupported", "intended_gaps": "", "intended_reason": reason,
            "now_local_en": ref.strftime("%A %d %B %Y, %H:%M"), "now_local_ar": describe_datetime(ref),
            "ref_datetime": ref.isoformat(), "situation": sit,
            "version_a": "Ask the assistant for this, the way you naturally would.", "version_b": ""})
    assign_waves(cards)
    return cards


WAVE1_PAIRS = {"project_management": 42, "general": 18}
WAVE1_ABSTAIN_PER_REASON = 5


def assign_waves(cards: list[dict[str, str]]) -> None:
    """Wave 1: 60 pair cards (42 PM, 18 general), taken round-robin over (tool, planned gaps) so
    every gap and both tools are covered, plus 5 out-of-scope cards per reason (~70% PM)."""
    for c in cards:
        c["wave"] = "2"
    for domain, n in WAVE1_PAIRS.items():
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for c in cards:
            if c["route_plan"] == "ACT+CLARIFY" and c["domain"] == domain:
                groups.setdefault((c["intended_intent"], c["intended_gaps"]), []).append(c)
        queues = [groups[k] for k in sorted(groups)]
        picked = 0
        while picked < n and any(queues):
            for q in queues:
                if q and picked < n:
                    q.pop(0)["wave"] = "1"
                    picked += 1
    for reason in REASON_COUNTS:
        pool = [c for c in cards if c["intended_reason"] == reason]
        pm = [c for c in pool if c["domain"] == "project_management"]
        gen = [c for c in pool if c["domain"] == "general"]
        n_pm = min(len(pm), round(WAVE1_ABSTAIN_PER_REASON * 0.7))
        for c in pm[:n_pm] + gen[:WAVE1_ABSTAIN_PER_REASON - n_pm]:
            c["wave"] = "1"


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel shows Arabic
        w = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    if Path(path).suffix.lower() == ".xlsx":
        return _read_xlsx(path)
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def _openpyxl() -> Any:
    try:
        import openpyxl  # type: ignore[import-untyped,unused-ignore]
    except ImportError as e:  # optional: only the author workbook needs it
        raise ValueError("reading or writing .xlsx needs openpyxl: run with `uv run --with openpyxl ...`") from e
    return openpyxl


def _read_xlsx(path: str | Path, sheet: str | None = None) -> list[dict[str, str]]:
    """Rows of the first sheet (or ``sheet``) as strings, keyed by the header row."""
    wb = _openpyxl().load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    out = []
    for r in rows[1:]:
        if any(v not in (None, "") for v in r):
            out.append({h: ("" if v is None else str(v)).strip() for h, v in zip(header, r, strict=False) if h})
    return out


def write_kit(out: str | Path) -> dict[str, int]:
    """``scenario_cards.csv`` (what contributors read) and ``collection_sheet.csv`` (one row per
    request to write: version A for every card, version B for pair cards)."""
    out = Path(out)
    cards = make_cards()
    _write_csv(out / "scenario_cards.csv", CARD_COLUMNS, cards)
    rows = [{"card_id": c["card_id"], "version": v, "contributor": "", "text": "", "comment": ""}
            for c in cards for v in (("A", "B") if c["version_b"] else ("A",))]
    _write_csv(out / "collection_sheet.csv", COLLECTION_COLUMNS, rows)
    return {"cards": len(cards), "requests": len(rows)}


_LATIN = re.compile(r"[A-Za-z]{2,}")


_QUESTION = re.compile(r"^\[(c\d{3})-([AB])\]")


def read_collection(path: str | Path) -> list[dict[str, str]]:
    """Collection rows from the collection sheet, or from a Google Forms response export made
    by :func:`form_script` (one row per contributor, one column per ``[cNNN-A]`` question)."""
    rows = _read_csv(path)
    if not rows or "card_id" in rows[0]:
        return rows
    code_col = next((k for k in rows[0] if k.lower().startswith("contributor code")), None)
    if code_col is None:
        raise ValueError(f"{path}: neither a collection sheet nor a form export (no 'Contributor code' column)")
    out = []
    for r in rows:
        for col, text in r.items():
            if m := _QUESTION.match(col):
                out.append({"card_id": m.group(1), "version": m.group(2), "contributor": r[code_col],
                            "text": text, "comment": ""})
    return out


def import_contributions(collection: str | Path, cards_path: str | Path, out: str | Path,
                         source: str = "contributor") -> dict[str, Any]:
    """Blind labelling sheet: the agreement-sheet columns (labels empty) plus provenance. Each
    card is a family; versions A and B of a pair card are a pair. Intended labels stay out.
    ``collection``: the filled collection sheet, a Google Forms export, or the author workbook
    (``source="author"``: requests written by the project author, annotation guide 2.2)."""
    if source not in ("author", "contributor"):
        raise ValueError(f"source must be author or contributor, not {source!r}")
    cards = {c["card_id"]: c for c in _read_csv(cards_path)}
    rows, skipped = [], []
    seen: set[str] = set()
    for r in read_collection(collection):
        key = f"{r['card_id']}-{r['version'].lower()}"
        if r["text"] and key in seen:
            skipped.append(f"{r['card_id']}{r['version']}: written twice, kept the first")
            continue
        card = cards.get(r["card_id"])
        if card is None or not r["text"]:
            skipped.append(f"{r['card_id']}{r.get('version', '')}: " + ("unknown card" if card is None else "no text"))
            continue
        if not r["contributor"]:
            skipped.append(f"{r['card_id']}{r['version']}: no contributor code")
            continue
        seen.add(key)
        ref = datetime.fromisoformat(card["ref_datetime"])
        rows.append({"id": f"{card['card_id']}-{r['version'].lower()}", "text": r["text"],
                     "ref_datetime": card["ref_datetime"], "ref_local": describe_datetime(ref), "timezone": TZ,
                     "family_id": f"fam-{card['card_id']}",
                     "pair_id": f"pair-{card['card_id']}" if card["version_b"] else "",
                     "domain": card["domain"], "source": source,
                     "code_switched": "yes" if _LATIN.search(r["text"]) else "no",
                     "contributor": r["contributor"]})
    # a pair needs both halves; a lone half is kept, without a pair id
    halves: dict[str, int] = {}
    for row in rows:
        if row["pair_id"]:
            halves[row["pair_id"]] = halves.get(row["pair_id"], 0) + 1
    for row in rows:
        if row["pair_id"] and halves[row["pair_id"]] != 2:
            row["pair_id"] = ""
    _write_csv(Path(out), (*SHEET_COLUMNS, *LABEL_EXTRA), rows)
    return {"rows": len(rows), "skipped": skipped}


def build_dataset(sheet: str | Path, out: str | Path, cards_path: str | Path | None = None) -> dict[str, Any]:
    """``data/athar.jsonl`` from the labelled sheet. Returns errors (rows that fail the contract,
    not written) and, with the cards, the items whose label differs from the card's plan."""
    plan = {c["card_id"]: c for c in _read_csv(cards_path)} if cards_path else {}
    items, errors, differs = [], [], []
    for i, row in enumerate(_read_csv(sheet), start=2):
        if not row.get("intent") and not row.get("route"):
            errors.append(f"row {i} ({row.get('id')}): not labelled")
            continue
        try:
            lab = _label(row, row.get("timezone") or TZ)
            item = GoldItem.model_validate({
                "id": row["id"], "family_id": row["family_id"], "text": row["text"],
                "ref_datetime": row["ref_datetime"], "timezone": row.get("timezone") or TZ,
                "intent": lab.intent, "route": lab.route,
                "args": {"datetime": lab.datetime, "reminder_text": lab.reminder_text,
                         "reminder_text_alts": [t for t in _cell(row, "reminder_text_alts").split("|") if t]},
                "clarify_targets": sorted(lab.targets), "unsupported_reason": lab.reason,
                "pair_id": row.get("pair_id") or None,
                "code_switched": row.get("code_switched", "").lower() in ("yes", "true", "1"),
                "source": row.get("source") or "contributor", "domain": row.get("domain") or "general",
                "notes": " ".join(x for x in (row.get("notes", ""),
                                              f"contributor {row['contributor']}" if row.get("contributor") else "") if x)})
        except ValidationError as e:
            errors.append(f"row {i} ({row.get('id')}): " + "; ".join(x["msg"] for x in e.errors()))
            continue
        except ValueError as e:
            errors.append(f"row {i} ({row.get('id')}): {e}")
            continue
        items.append(item)
        card = plan.get(row["id"].rsplit("-", 1)[0])
        if card:
            planned = (card["intended_intent"], "ABSTAIN" if card["route_plan"] == "ABSTAIN"
                       else ("ACT" if row["id"].endswith("-a") else "CLARIFY"))
            if (item.intent, item.route) != planned:
                differs.append(f"{item.id}: labelled {item.intent}/{item.route}, card planned {planned[0]}/{planned[1]}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(json.dumps(it.model_dump(mode="json", exclude_defaults=True), ensure_ascii=False)
                                 + "\n" for it in items), encoding="utf-8", newline="\n")
    return {"items": len(items), "errors": errors, "differs_from_card": differs}


# --------------------------------------------------------------------------- Google Forms
FORM_INTRO = (
    "ATHAR Chat is a university NLP project: an assistant that sets one reminder or one alarm from "
    "a message in Egyptian Arabic. For each situation below, type the message YOU would really send, "
    "in your own words, the way you normally type (English work words are fine). "
    "Do not use ChatGPT or any AI or translation tool, and do not copy anyone's wording. "
    "Use invented names only. About 25 minutes.\n\n"
    "مشروع جامعي في معالجة اللغة: مساعد بيعمل تذكير أو منبه من رسالة بالمصري. "
    "لكل موقف، اكتب الرسالة اللي إنت فعلاً هتبعتها، بكلامك وبطريقتك. "
    "ممنوع ChatGPT أو أي أداة ذكاء اصطناعي أو ترجمة. أسماء من خيالك بس.")
CONSENT = ("I agree that my messages are used for this course project and its report, stored with my "
           "contributor code only (not my name). I can ask for them to be removed before the dataset is frozen.")


def blocks(cards: Sequence[dict[str, str]], wave: str, people: int) -> list[list[dict[str, str]]]:
    """Deal the wave's cards to ``people`` contributors round-robin over (route plan, domain), so
    everyone writes PM and general, complete, incomplete and out-of-scope requests."""
    chosen = [c for c in cards if c["wave"] == wave]
    order = sorted(chosen, key=lambda c: (c["route_plan"], c["domain"], c["card_id"]))
    out: list[list[dict[str, str]]] = [[] for _ in range(people)]
    for i, c in enumerate(order):
        out[i % people].append(c)
    return [sorted(b, key=lambda c: c["card_id"]) for b in out]


def form_script(cards_path: str | Path, out: str | Path, wave: str = "1", people: int = 5) -> dict[str, int]:
    """Google Apps Script that creates one Google Form per contributor block. Paste it into
    script.google.com and run ``createForms``; the form links appear in the execution log."""
    cards = _read_csv(cards_path)
    forms: list[dict[str, Any]] = []
    n_requests = 0
    for k, block in enumerate(blocks(cards, wave, people), start=1):
        pages: list[dict[str, Any]] = []
        for c in block:
            qs = [{"title": f"[{c['card_id']}-A] Write the full request", "help": c["version_a"]}]
            if c["version_b"]:
                qs.append({"title": f"[{c['card_id']}-B] Write it again, as the card says", "help": c["version_b"]})
            n_requests += len(qs)
            pages.append({"title": f"Situation: {c['situation']}",
                          "help": f"Imagine you are typing on {c['now_local_en']} ({c['now_local_ar']}).",
                          "questions": qs})
        forms.append({"title": f"ATHAR Chat: Egyptian Arabic requests (wave {wave}, block {k})", "pages": pages})
    data = json.dumps({"intro": FORM_INTRO, "consent": CONSENT, "forms": forms}, ensure_ascii=False, indent=1)
    js = f"""// Generated by `athar-data form-script`. Paste into https://script.google.com (New project),
// then Run -> createForms and allow access. Each form's links are printed in the execution log.
// Give each contributor ONE form link and a contributor code (P01, P02, ...).
// Responses: in each form, Responses -> Download responses (.csv), then
//   uv run athar-data import-contrib <export.csv> --cards contrib/scenario_cards.csv --out runs/labelling/<name>.csv
const KIT = {data};

function createForms() {{
  KIT.forms.forEach(function (spec) {{
    const form = FormApp.create(spec.title);
    form.setDescription(KIT.intro);
    form.setCollectEmail(false);
    form.setProgressBar(true);
    form.setConfirmationMessage("Thank you! / شكراً جداً!");
    form.addCheckboxItem().setTitle("Consent").setChoiceValues([KIT.consent]).setRequired(true);
    form.addTextItem().setTitle("Contributor code (for example P03)").setRequired(true);
    spec.pages.forEach(function (page) {{
      form.addPageBreakItem().setTitle(page.title).setHelpText(page.help);
      page.questions.forEach(function (q) {{
        form.addParagraphTextItem().setTitle(q.title).setHelpText(q.help);
      }});
    }});
    Logger.log(spec.title + "\\n  share: " + form.getPublishedUrl() + "\\n  edit:  " + form.getEditUrl());
  }});
}}
"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(js, encoding="utf-8", newline="\n")
    return {"forms": len(forms), "requests": n_requests}


# --------------------------------------------------------------------------- author workbook
AUTHOR_COLUMNS = ("order", "core", "card_id", "version", "situation", "imagine_it_is", "imagine_it_is_ar",
                  "what_to_write", "text", "contributor", "comment")
CORE_PAIRS, CORE_ABSTAIN = 50, 20      # the locked test targets: 50 ACT + 50 CLARIFY + 20 ABSTAIN
AUTHOR_GUIDE = [
    ("How to write (read once, then don't think about it)", ""),
    ("1", "Write each request the way YOU would really type it to an assistant on your phone: short, Egyptian, quick."),
    ("2", "Do not look at the draft dataset, the pilot items or any example wording while you write."),
    ("3", "Do not think about what the system can understand. Write naturally, not to make it easy or hard."),
    ("4", "Vary naturally, as real people do: English work words, typos, Arabic or Western digits, لو سمحت, word order."),
    ("5", "Version B: the same request again, leaving out exactly what the card says, the way people really forget it."),
    ("6", "Out-of-scope cards (one version only): ask for that thing naturally; the assistant is expected to decline."),
    ("7", "Invented names only. No real phone numbers, clients or private details."),
    ("8", "Write in 2-3 sittings, in the order given. Rows marked core = 1 come first: 120 requests are the minimum."),
    ("9", "Leave 'contributor' as author. Use 'comment' if a card is unclear; skip a row rather than force it."),
    ("", ""),
    ("بالعربي", "اكتب كل طلب زي ما إنت فعلاً بتكتبه للمساعد على الموبايل: قصير وبالمصري. متبصش على أي أمثلة أو داتا قبل كده، "
                "ومتفكرش السيستم هيفهم ولا لأ. في version B شيل الحاجة اللي الكارت بيقول عليها بس. أول 120 سطر (core = 1) هما الأساس."),
]


def core_cards(cards: Sequence[dict[str, str]]) -> set[str]:
    """Wave-1 cards that make the locked test targets: 50 pair cards (35 PM, 15 general) round-robin
    over (tool, planned gaps), and 20 out-of-scope cards (14 PM, 6 general) round-robin over
    unsupported reasons, so the core keeps the ~70% project-management share."""
    w1 = [c for c in cards if c["wave"] == "1"]
    chosen: set[str] = set()
    quotas = (("intended_gaps", {"project_management": 35, "general": 15}),
              ("intended_reason", {"project_management": 14, "general": 6}))
    for key, per_domain in quotas:
        for domain, n in per_domain.items():
            pool = [c for c in w1 if c["domain"] == domain
                    and (c["route_plan"] == "ABSTAIN") == (key == "intended_reason")]
            groups: dict[tuple[str, str], list[dict[str, str]]] = {}
            for c in sorted(pool, key=lambda c: c["card_id"]):
                groups.setdefault((c["intended_intent"], c[key]), []).append(c)
            queues = [groups[k] for k in sorted(groups)]
            picked = 0
            while picked < n and any(queues):
                for q in queues:
                    if q and picked < n:
                        chosen.add(q.pop(0)["card_id"])
                        picked += 1
    return chosen


def author_rows(cards: Sequence[dict[str, str]], seed: int = 20261010) -> list[dict[str, str]]:
    """Wave-1 requests for the author: core cards first, each group in a shuffled order (so route
    types are interleaved), versions A and B of a card next to each other."""
    import random

    core = core_cards(cards)
    w1 = [c for c in cards if c["wave"] == "1"]
    rng = random.Random(seed)
    first = sorted((c for c in w1 if c["card_id"] in core), key=lambda c: c["card_id"])
    rest = sorted((c for c in w1 if c["card_id"] not in core), key=lambda c: c["card_id"])
    rng.shuffle(first)
    rng.shuffle(rest)
    rows: list[dict[str, str]] = []
    for c in first + rest:
        for v in ("A", "B") if c["version_b"] else ("A",):
            rows.append({"order": str(len(rows) + 1), "core": "1" if c["card_id"] in core else "0",
                         "card_id": c["card_id"], "version": v, "situation": c["situation"],
                         "imagine_it_is": c["now_local_en"], "imagine_it_is_ar": c["now_local_ar"],
                         "what_to_write": c["version_a"] if v == "A" else c["version_b"],
                         "text": "", "contributor": "author", "comment": ""})
    return rows


def author_sheet(cards_path: str | Path, out: str | Path) -> dict[str, int]:
    """An Excel workbook to write the wave-1 requests in: a 'requests' sheet (the ``text`` column
    is right-to-left and wrapped) and a 'how to write' sheet."""
    openpyxl = _openpyxl()
    from openpyxl.styles import (  # type: ignore[import-untyped,unused-ignore]
        Alignment,
        Font,
        PatternFill,
    )

    rows = author_rows(_read_csv(cards_path))
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "requests"
    ws.append(list(AUTHOR_COLUMNS))
    for r in rows:
        ws.append([r[c] for c in AUTHOR_COLUMNS])
    widths = {"order": 7, "core": 6, "card_id": 8, "version": 8, "situation": 48, "imagine_it_is": 26,
              "imagine_it_is_ar": 30, "what_to_write": 60, "text": 60, "contributor": 12, "comment": 30}
    fill = PatternFill("solid", fgColor="FFF4CC")
    for j, col in enumerate(AUTHOR_COLUMNS, start=1):
        letter = ws.cell(row=1, column=j).column_letter
        ws.column_dimensions[letter].width = widths[col]
        ws.cell(row=1, column=j).font = Font(bold=True)
        for i in range(2, len(rows) + 2):
            cell = ws.cell(row=i, column=j)
            cell.alignment = Alignment(wrap_text=True, vertical="top",
                                       horizontal="right" if col in ("text", "imagine_it_is_ar") else "left",
                                       readingOrder=2 if col in ("text", "imagine_it_is_ar") else 1)
            if col == "text":
                cell.fill = fill
    ws.freeze_panes = "E2"
    guide = wb.create_sheet("how to write")
    for a, b in AUTHOR_GUIDE:
        guide.append([a, b])
    guide.column_dimensions["A"].width = 12
    guide.column_dimensions["B"].width = 120
    for row in guide.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return {"requests": len(rows), "core": sum(r["core"] == "1" for r in rows)}
