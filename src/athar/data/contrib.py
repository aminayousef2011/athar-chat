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
                         source: str = "contributor", *, multi_writer: bool = False) -> dict[str, Any]:
    """Blind labelling sheet: the agreement-sheet columns (labels empty) plus provenance. Each
    card is a family; versions A and B of a pair card are a pair. Intended labels stay out.
    ``collection``: the filled collection sheet, a Google Forms export, or the author workbook
    (``source="author"``: requests written by the project author, annotation guide 2.2).
    ``multi_writer``: several contributors may write the same card (the natural-phrasing set);
    ids become ``<card>-<version>-<code>`` and each writer's A and B form their own pair, while
    the family stays the card (the scenario)."""
    if source not in ("author", "contributor"):
        raise ValueError(f"source must be author or contributor, not {source!r}")
    cards = {c["card_id"]: c for c in _read_csv(cards_path)}
    rows, skipped = [], []
    seen: set[str] = set()
    for r in read_collection(collection):
        writer = re.sub(r"[^a-z0-9]+", "", _code(r["contributor"]).lower()) if multi_writer else ""
        key = f"{r['card_id']}-{r['version'].lower()}" + (f"-{writer}" if writer else "")
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
        rows.append({"id": key, "text": r["text"],
                     "ref_datetime": card["ref_datetime"], "ref_local": describe_datetime(ref), "timezone": TZ,
                     "family_id": f"fam-{card['card_id']}",
                     "pair_id": f"pair-{card['card_id']}" + (f"-{writer}" if writer else "") if card["version_b"] else "",
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
    if Path(out).suffix.lower() == ".xlsx":
        labelling_workbook(rows, out)
    else:
        _write_csv(Path(out), (*SHEET_COLUMNS, *LABEL_EXTRA), rows)
    return {"rows": len(rows), "skipped": skipped}


LABEL_HELP = {
    "intent": "create_reminder / create_alarm / unsupported",
    "route": "ACT / CLARIFY / ABSTAIN",
    "clarify_targets": "CLARIFY only, space-separated, from: date time am_pm past reminder_text",
    "unsupported_reason": "ABSTAIN only: deletion modification recurring other_person multi_action other_tool other",
    "datetime": "ACT only: the resolved time, as 2026-10-19 08:00 (Cairo local time)",
    "reminder_text": "ACT reminders: the content, in the user's own spelling",
}


def labelling_workbook(rows: Sequence[dict[str, str]], out: str | Path,
                       columns: Sequence[str] | None = None) -> None:
    """The blind labelling sheet as an Excel workbook: every cell is text (Excel cannot rewrite
    the datetimes), intent / route / reason are drop-down lists, and a help row explains each
    label column. :func:`build_dataset` reads it directly."""
    openpyxl = _openpyxl()
    from openpyxl.styles import (  # type: ignore[import-untyped,unused-ignore]
        Alignment,
        Font,
        PatternFill,
    )
    from openpyxl.worksheet.datavalidation import (  # type: ignore[import-untyped,unused-ignore]
        DataValidation,  # type: ignore[import-untyped,unused-ignore]
    )

    cols = tuple(columns) if columns is not None else (*SHEET_COLUMNS, *LABEL_EXTRA)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "label"
    ws.append(list(cols))
    for r in rows:
        ws.append([r.get(c, "") for c in cols])
    n = len(rows) + 1
    fill = PatternFill("solid", fgColor="E8F1FF")
    lists = {"intent": "create_reminder,create_alarm,unsupported", "route": "ACT,CLARIFY,ABSTAIN",
             "unsupported_reason": "deletion,modification,recurring,other_person,multi_action,other_tool,other"}
    for j, col in enumerate(cols, start=1):
        letter = ws.cell(row=1, column=j).column_letter
        ws.cell(row=1, column=j).font = Font(bold=True)
        ws.column_dimensions[letter].width = {"text": 55, "ref_local": 30, "notes": 30, "reminder_text": 30,
                                              "clarify_targets": 18, "datetime": 18}.get(col, 14)
        for i in range(2, n + 1):
            cell = ws.cell(row=i, column=j)
            cell.number_format = "@"
            if col in LABEL_HELP:
                cell.fill = fill
            if col in ("text", "ref_local", "reminder_text"):
                cell.alignment = Alignment(horizontal="right", readingOrder=2, wrap_text=True, vertical="top")
        if col in lists:
            dv = DataValidation(type="list", formula1=f'"{lists[col]}"', allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"{letter}2:{letter}{n}")
    ws.freeze_panes = "C2"
    help_ws = wb.create_sheet("how to label")
    help_ws.append(["column", "what to enter"])
    for col, text in LABEL_HELP.items():
        help_ws.append([col, text])
    help_ws.append(["rules", "docs/annotation_guide.md section 3: intent, then arguments, then every gap, then route. "
                             "Use only the request and its reference time (ref_local). Never run a system first."])
    help_ws.column_dimensions["A"].width = 20
    help_ws.column_dimensions["B"].width = 110
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


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
    items, unpaired = _keep_minimal_pairs(items)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(json.dumps(it.model_dump(mode="json", exclude_defaults=True), ensure_ascii=False)
                                 + "\n" for it in items), encoding="utf-8", newline="\n")
    return {"items": len(items), "errors": errors, "differs_from_card": differs, "unpaired": unpaired}


def _keep_minimal_pairs(items: Sequence[GoldItem]) -> tuple[list[GoldItem], list[str]]:
    """A card's versions A and B are a pair only if, as written and labelled, they are a minimal
    pair: the same intent, one ACT and one CLARIFY (annotation guide 2.1). Otherwise the pair link
    is dropped (the items stay in their family, so they still split together) and the reason is
    noted. The request texts are never changed to make a pair fit."""
    by_pair: dict[str, list[GoldItem]] = {}
    for it in items:
        if it.pair_id:
            by_pair.setdefault(it.pair_id, []).append(it)
    drop: dict[str, str] = {}
    for pid, members in by_pair.items():
        if len(members) != 2:
            reason = "only one version labelled"
        elif members[0].intent != members[1].intent:
            reason = f"different tools ({members[0].intent} / {members[1].intent})"
        elif sorted(m.route for m in members) != ["ACT", "CLARIFY"]:
            reason = f"not one complete and one incomplete ({' / '.join(sorted(m.route for m in members))})"
        else:
            continue
        drop[pid] = reason
    out = []
    for it in items:
        if it.pair_id in drop:
            note = f"pair dropped: {drop[it.pair_id]}"
            it = it.model_copy(update={"pair_id": None, "notes": f"{it.notes} | {note}" if it.notes else note})
        out.append(it)
    return out, [f"{pid}: {why}" for pid, why in sorted(drop.items())]


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


#: wave 2 (natural-phrasing evaluation, docs/protocol_natural.md): shorter forms, and the lessons of
#: wave 1 (calendar dates with the year, copied wording, B identical to A) made explicit
FORM_INTRO_NATURAL = (
    "ATHAR Chat is a university NLP project: an assistant that sets one reminder or one alarm from "
    "a message in Egyptian Arabic. For each situation, type the message YOU would really send, the way "
    "you text on your phone (spelling mistakes and English work words are fine). Say days the way you "
    "speak (بكرة، الخميس الجاي، النهارده), not as a calendar date with the year. Version B is the same "
    "wish written again from scratch, missing what the card says. Do not use ChatGPT or any AI or "
    "translation tool, do not look at anyone else's answers, and use invented names only. Each page says "
    "when you are typing: that is NOT the reminder time, never copy it into your message. About 10 minutes.\n\n"
    "مشروع جامعي: مساعد بيعمل تذكير أو منبه من رسالة بالمصري. لكل موقف اكتب الرسالة اللي إنت فعلاً "
    "هتبعتها، زي ما بتكتب على الموبايل. قول الأيام زي ما بتقولها (بكرة، الخميس الجاي) من غير تاريخ وسنة. "
    "كل صفحة بتقولك إنت بتكتب إمتى: ده مش وقت التذكير، متنقلوش في الرسالة. "
    "ممنوع ChatGPT أو أي أداة ذكاء اصطناعي، وممنوع تشوف إجابات حد تاني. أسماء من خيالك بس.")


def relative_moment(card: dict[str, str]) -> str:
    """When the contributor is typing, as weekday and clock time only: no day number, month or
    year to copy (wave 2b; wave-2 contributors copied the full date into their requests)."""
    from athar.respond import AR_WEEKDAYS

    when = datetime.fromisoformat(card["ref_datetime"])
    h12, mm = when.hour % 12 or 12, f"{when.minute:02d}"
    ar_period = ("الصبح" if 5 <= when.hour <= 11 else "الضهر" if 12 <= when.hour <= 14
                 else "العصر" if 15 <= when.hour <= 17 else "بالليل")
    return (f"It is now {when.strftime('%A')}, {h12}:{mm} {'am' if when.hour < 12 else 'pm'}: the moment you SEND "
            "the message, NOT the reminder time. Ask for a later time the way you would say it, and do not "
            "write a date or a year.\n"
            f"دلوقتي {AR_WEEKDAYS[when.weekday()]} الساعة {h12}:{mm} {ar_period}: ده وقت ما بتبعت الرسالة، "
            "مش وقت التذكير. اطلب وقت بعد كده بطريقتك، ومتكتبش تاريخ أو سنة.")


def blocks(cards: Sequence[dict[str, str]], wave: str, people: int,
           card_ids: set[str] | None = None) -> list[list[dict[str, str]]]:
    """Deal the wave's cards (or the cards in ``card_ids``) to ``people`` contributors round-robin
    over (route plan, domain), so everyone writes PM and general, complete, incomplete and
    out-of-scope requests."""
    chosen = [c for c in cards if (c["card_id"] in card_ids if card_ids is not None else c["wave"] == wave)]
    order = sorted(chosen, key=lambda c: (c["route_plan"], c["domain"], c["card_id"]))
    out: list[list[dict[str, str]]] = [[] for _ in range(people)]
    for i, c in enumerate(order):
        out[i % people].append(c)
    return [sorted(b, key=lambda c: c["card_id"]) for b in out]


def form_script(cards_path: str | Path, out: str | Path, wave: str = "1", people: int = 5, *,
                card_ids: set[str] | None = None, intro: str = FORM_INTRO,
                relative: bool = False) -> dict[str, int]:
    """Google Apps Script that creates one Google Form per contributor block. Paste it into
    script.google.com and run ``createForms``; the form links appear in the execution log.
    ``card_ids`` selects the cards instead of ``wave`` (wave 2: the test-family cards);
    ``relative`` shows the typing moment as weekday and time only (:func:`relative_moment`)."""
    cards = _read_csv(cards_path)
    forms: list[dict[str, Any]] = []
    n_requests = 0
    for k, block in enumerate(blocks(cards, wave, people, card_ids), start=1):
        pages: list[dict[str, Any]] = []
        for c in block:
            qs = [{"title": f"[{c['card_id']}-A] Write the full request", "help": c["version_a"]}]
            if c["version_b"]:
                qs.append({"title": f"[{c['card_id']}-B] Write it again, as the card says", "help": c["version_b"]})
            n_requests += len(qs)
            pages.append({"title": f"Situation: {c['situation']}",
                          "help": relative_moment(c) if relative else
                          f"Imagine you are typing on {c['now_local_en']} ({c['now_local_ar']}).",
                          "questions": qs})
        forms.append({"title": f"ATHAR Chat: Egyptian Arabic requests (wave {wave}, block {k})", "pages": pages})
    data = json.dumps({"intro": intro, "consent": CONSENT, "forms": forms}, ensure_ascii=False, indent=1)
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
AUTHOR_COLUMNS = ("order", "core", "card_id", "version", "situation", "typing_at", "what_to_write", "text",
                  "contributor", "comment")
#: header shown in the workbook for each column (the typing moment is NOT the reminder time)
AUTHOR_HEADERS = {"typing_at": "you are typing at (NOT the reminder time)"}
CORE_PAIRS, CORE_ABSTAIN = 50, 20      # the locked test targets: 50 ACT + 50 CLARIFY + 20 ABSTAIN
AUTHOR_GUIDE = [
    ("How to write (read once)", ""),
    ("1", "'you are typing at' is the moment you send the message. It is NOT the reminder time. Do not copy it into the request."),
    ("2", "Version A asks for a time LATER than that moment: later today, tonight, tomorrow, Thursday, the 3rd... "
          "Say it the way you would (\"بكرة 9 الصبح\", \"الخميس الجاي الساعة 3 العصر\"), not as a full calendar date."),
    ("3", "Version B is the same wish, typed again from scratch, leaving out what the card says (e.g. no clock time). "
          "Do not copy A and delete words; write it as you would if you really forgot that detail."),
    ("4", "Out-of-scope cards: ask for that thing naturally. No date or time unless you would really say one."),
    ("5", "Type it the way you really type on your phone: no need for shadda, full stops or perfect spelling. "
          "Best: type each message on your phone (e.g. to yourself on WhatsApp) and paste it here."),
    ("6", "Your own words only: no ChatGPT or other AI tool, no translation tool, no templates or formulas."),
    ("7", "Do not look at the draft dataset, the pilot items or any examples while you write."),
    ("8", "Invented names only. Rows with core = 1 come first: those 120 are the minimum."),
    ("9", "Before sending: uv run --with openpyxl athar-data check-author <file> --cards contrib/scenario_cards.csv"),
    ("", ""),
    ("بالعربي", "'you are typing at' ده وقت ما بتكتب الرسالة، مش وقت التذكير: متنقلوش في الطلب. في A اطلب وقت بعد كده "
                "(بكرة، الخميس، بالليل...) بالطريقة اللي بتقولها عادي. B نفس الطلب مكتوب من الأول من غير الحاجة اللي الكارت "
                "بيقول عليها. اكتب زي ما بتكتب على الموبايل، بكلامك إنت، من غير أي أداة ذكاء اصطناعي."),
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


def _what_to_write(card: dict[str, str], version: str) -> str:
    if card["route_plan"] == "ABSTAIN":
        return ("Ask the ASSISTANT to do this itself, the way you naturally would (not a reminder for you to do "
                "it). No date or time unless you would really say one.")
    if version == "A":
        what = "" if card["intended_intent"] == "create_alarm" else ", and what to be reminded of"
        return (f"Full request for a time LATER than the moment you are typing: the day, the clock time with "
                f"morning/evening{what}. Say the day the way you would (tomorrow, Thursday...), not as a calendar date.")
    gaps = tuple(card["intended_gaps"].split())
    return (f"The same wish typed again from scratch, but {GAP_INSTRUCTIONS[gaps]}. Not a copy of A with words deleted.")


def author_rows(cards: Sequence[dict[str, str]], seed: int = 20261010,
                reasons: Sequence[str] = ()) -> list[dict[str, str]]:
    """Wave-1 requests for the author: core cards first, each group in a shuffled order (so route
    types are interleaved), versions A and B of a card next to each other. With ``reasons``:
    instead, every wave-2 out-of-scope card with those unsupported reasons (all core), to fill a
    coverage gap."""
    import random

    if reasons:
        w1 = [c for c in cards if c["wave"] == "2" and c["intended_reason"] in set(reasons)]
        core = {c["card_id"] for c in w1}
    else:
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
                         "typing_at": c["now_local_en"], "what_to_write": _what_to_write(c, v),
                         "text": "", "contributor": "author", "comment": ""})
    return rows


def author_sheet(cards_path: str | Path, out: str | Path, reasons: Sequence[str] = ()) -> dict[str, int]:
    """An Excel workbook to write the wave-1 requests in: a 'requests' sheet (the ``text`` column
    is right-to-left and wrapped) and a 'how to write' sheet."""
    openpyxl = _openpyxl()
    from openpyxl.styles import (  # type: ignore[import-untyped,unused-ignore]
        Alignment,
        Font,
        PatternFill,
    )

    rows = author_rows(_read_csv(cards_path), reasons=reasons)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "requests"
    ws.append([AUTHOR_HEADERS.get(c, c) for c in AUTHOR_COLUMNS])
    for r in rows:
        ws.append([r[c] for c in AUTHOR_COLUMNS])
    widths = {"order": 7, "core": 6, "card_id": 8, "version": 8, "situation": 48, "typing_at": 26,
              "what_to_write": 60, "text": 60, "contributor": 12, "comment": 30}
    fill = PatternFill("solid", fgColor="FFF4CC")
    for j, col in enumerate(AUTHOR_COLUMNS, start=1):
        letter = ws.cell(row=1, column=j).column_letter
        ws.column_dimensions[letter].width = widths[col]
        ws.cell(row=1, column=j).font = Font(bold=True)
        for i in range(2, len(rows) + 2):
            cell = ws.cell(row=i, column=j)
            cell.alignment = Alignment(wrap_text=True, vertical="top",
                                       horizontal="right" if col == "text" else "left",
                                       readingOrder=2 if col == "text" else 1)
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


# --------------------------------------------------------------------------- author checks
def _typing_moment_in(text: str, card: dict[str, str]) -> bool:
    """The card's typing moment (day number and clock time) copied into the request."""
    when = datetime.fromisoformat(card["ref_datetime"])
    clock = f"{when.hour % 12 or 12}:{when.minute:02d}"
    return clock in text and re.search(rf"(?<!\d){when.day}(?!\d)", text) is not None


def check_author(path: str | Path, cards_path: str | Path) -> dict[str, Any]:
    """Writing checks on the author workbook before import: per-row problems (the typing moment
    used as the reminder time, B copied from A, a date bolted onto an out-of-scope request,
    duplicates, empty core rows) and style warnings (uniform shadda, full stops, template)."""
    import difflib

    cards = {c["card_id"]: c for c in _read_csv(cards_path)}
    rows = [r for r in _read_csv(path) if r.get("card_id")]
    texts = {(r["card_id"], r["version"]): r.get("text", "").strip() for r in rows}
    problems: list[str] = []
    seen: dict[str, str] = {}
    for r in rows:
        t, key, card = r.get("text", "").strip(), f"{r['card_id']}-{r['version']}", cards.get(r["card_id"])
        if card is None:
            problems.append(f"{key}: unknown card")
            continue
        if not t:
            if r.get("core") == "1":
                problems.append(f"{key}: core row is empty")
            continue
        if t in seen:
            problems.append(f"{key}: same text as {seen[t]}")
        seen.setdefault(t, key)
        if card["route_plan"] == "ABSTAIN":
            if _typing_moment_in(t, card):
                problems.append(f"{key}: out-of-scope request carries the typing moment as a date/time")
        elif r["version"] == "A" and _typing_moment_in(t, card):
            problems.append(f"{key}: asks for the typing moment itself (not a later time): it would not be ACT")
        elif r["version"] == "B":
            a = texts.get((r["card_id"], "A"), "")
            if a and difflib.SequenceMatcher(None, a, t).ratio() >= 0.9:
                problems.append(f"{key}: almost the same as version A (write B from scratch)")
    filled = [t for t in texts.values() if t]
    n = max(len(filled), 1)
    style = {"shadda": sum("\u0651" in t for t in filled) / n, "full_stop": sum(t.endswith(".") for t in filled) / n}
    warnings = []
    if style["shadda"] > 0.5:
        warnings.append(f"{style['shadda']:.0%} of requests contain a typed shadda; people rarely type it on a phone")
    if style["full_stop"] > 0.8:
        warnings.append(f"{style['full_stop']:.0%} end with a full stop; real messages vary")
    return {"rows": len(rows), "filled": len(filled), "problems": problems, "warnings": warnings, "style": style}


# --------------------------------------------------------------------------- contributor screening
#: Screening rules for contributor responses (docs/protocol_natural.md, section 3). They were written
#: after inspecting the 2026-10-26 form export and apply unchanged to every later response. A
#: *submission* is one filled form; a contributor code used twice is two submissions.
SCREEN_RULES = {
    "message_duplicate": 0.90,         # M1: a message this similar to another submission's answer to the same question
    "reference_copy": 0.90,            # M4: a message this similar to an existing dataset text
    "submission_duplicate": 0.85,      # R1: two submissions whose answers are this similar on average
    "submission_flagged_share": 1 / 3,  # R2: a submission with more than this share of M1/M2/M4 messages
    "b_copies_a_share": 1 / 3,         # R4: a submission whose B equals its A on more than this share of pairs
}
_YEAR = re.compile(r"(?<![\d٠-٩])(?:2026|٢٠٢٦)(?![\d٠-٩])")
_GLOSS = re.compile(r"\([A-Za-z][^)]*\)")


def _code(raw: str) -> str:
    """Contributor code as typed: spreadsheet exports turn ``1`` into ``1.0``."""
    c = raw.strip().upper()
    return c[:-2] if c.endswith(".0") and c[:-2].isdigit() else c


def _similarity(a: str, b: str) -> float:
    """Similarity of two normalized texts (difflib ratio, 0-1)."""
    import difflib

    from athar.normalize import norm

    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def _at_least(a: str, b: str, threshold: float) -> bool:
    """``_similarity(a, b) >= threshold``, skipping the full comparison when its upper bounds fail."""
    import difflib

    from athar.normalize import norm

    sm = difflib.SequenceMatcher(None, norm(a), norm(b))
    return sm.real_quick_ratio() >= threshold and sm.quick_ratio() >= threshold and sm.ratio() >= threshold


def check_contrib(collection: str | Path, cards_path: str | Path, *, reference_texts: Sequence[str] = (),
                  exclude_codes: Sequence[str] = (), training_cards: set[str] | None = None) -> dict[str, Any]:
    """Screen a Google Forms export (.csv or .xlsx) before labelling.

    Message rules: M1 near-duplicate of another submission's answer to the same question; M2
    machine formatting (a written year, or an English gloss in parentheses); M3 version B
    identical to version A (B dropped, A kept); M4 near-copy of an existing dataset text.
    Submission rules (the whole submission is excluded): R1 near-duplicate of another submission;
    R2 more than a third of its messages flagged by M1, M2 or M4; R3 a code in ``exclude_codes``
    (the project author) or a repeated submission of the same code (the first is kept); R4 version
    B identical to version A in more than a third of its pairs (the task was not done).

    Returns the per-submission decisions and ``kept``: collection-sheet rows for
    :func:`import_contributions`. Messages for cards in ``training_cards`` (scenarios whose family
    is in train or dev) are kept but marked, so they can be reported apart."""
    from athar.normalize import norm

    rows = _read_csv(collection)
    if not rows:
        raise ValueError(f"{collection}: empty export")
    code_col = next((k for k in rows[0] if k.lower().startswith("contributor code")), None)
    if code_col is None:
        raise ValueError(f"{collection}: not a form export (no 'Contributor code' column)")
    cards = {c["card_id"]: c for c in _read_csv(cards_path)}
    excluded_codes = {_code(c) for c in exclude_codes}

    subs: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for r in rows:
        code = _code(r[code_col])
        seen[code] = seen.get(code, 0) + 1
        answers = {}
        for col, text in r.items():
            if (m := _QUESTION.match(col)) and text.strip() and m.group(1) in cards:
                answers[(m.group(1), m.group(2))] = text.strip()
        subs.append({"submission": code if seen[code] == 1 else f"{code}#{seen[code]}", "code": code,
                     "timestamp": r.get("Timestamp", ""), "answers": answers, "reasons": [],
                     "flags": {k: [] for k in answers}})

    refs = list(dict.fromkeys(t for t in reference_texts if t))
    for s in subs:
        for (cid, v), t in s["answers"].items():
            if _YEAR.search(t) or _GLOSS.search(t):
                s["flags"][(cid, v)].append("M2 machine formatting")
            if v == "B" and (cid, "A") in s["answers"] and norm(t) == norm(s["answers"][(cid, "A")]):
                s["flags"][(cid, v)].append("M3 B identical to A")
            if any(_at_least(t, ref, SCREEN_RULES["reference_copy"]) for ref in refs):
                s["flags"][(cid, v)].append("M4 copies an existing dataset text")
    for i, a in enumerate(subs):
        for b in subs[i + 1:]:
            common = [k for k in a["answers"] if k in b["answers"]]
            sims = {k: _similarity(a["answers"][k], b["answers"][k]) for k in common}
            for k, sim in sims.items():
                if sim >= SCREEN_RULES["message_duplicate"]:
                    a["flags"][k].append(f"M1 duplicates {b['submission']}")
                    b["flags"][k].append(f"M1 duplicates {a['submission']}")
            if sims and sum(sims.values()) / len(sims) >= SCREEN_RULES["submission_duplicate"]:
                a["reasons"].append(f"R1 duplicates submission {b['submission']}")
                b["reasons"].append(f"R1 duplicates submission {a['submission']}")
    for s in subs:
        heavy = [k for k, f in s["flags"].items() if any(x[:2] in ("M1", "M2", "M4") for x in f)]
        if s["answers"] and len(heavy) / len(s["answers"]) > SCREEN_RULES["submission_flagged_share"]:
            s["reasons"].append(f"R2 {len(heavy)}/{len(s['answers'])} messages flagged")
        pairs = [k for k in s["answers"] if k[1] == "B"]
        copied = [k for k in pairs if "M3 B identical to A" in s["flags"][k]]
        if pairs and len(copied) / len(pairs) > SCREEN_RULES["b_copies_a_share"]:
            s["reasons"].append(f"R4 version B copies A in {len(copied)}/{len(pairs)} pairs")
        if s["code"] in excluded_codes:
            s["reasons"].append("R3 excluded code")
        if "#" in s["submission"]:
            s["reasons"].append("R3 repeated submission of this code")

    kept: list[dict[str, str]] = []
    report = []
    for s in subs:
        keep_msgs = [] if s["reasons"] else [k for k, f in s["flags"].items() if not f]
        for cid, v in sorted(keep_msgs):
            seen_in_training = training_cards is not None and cid in training_cards
            kept.append({"card_id": cid, "version": v, "contributor": s["code"], "text": s["answers"][(cid, v)],
                         "comment": "scenario family in train/dev" if seen_in_training else ""})
        report.append({"submission": s["submission"], "timestamp": s["timestamp"],
                       "status": "excluded" if s["reasons"] else "kept",
                       "reasons": sorted(set(s["reasons"])), "messages": len(s["answers"]), "kept": len(keep_msgs),
                       "flagged": {f"{c}-{v}": f for (c, v), f in sorted(s["flags"].items()) if f}})
    return {"rules": SCREEN_RULES, "submissions": report, "kept": kept,
            "writers": sorted({r["contributor"] for r in kept})}


def write_collection(rows: Sequence[dict[str, str]], out: str | Path) -> None:
    """A collection sheet (input of :func:`import_contributions`)."""
    _write_csv(Path(out), COLLECTION_COLUMNS, rows)
