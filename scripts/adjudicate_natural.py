"""Adjudicate the natural-phrasing labels (docs/protocol_natural.md, log 2026-11-01).

    uv run --with openpyxl python scripts/adjudicate_natural.py sheet
    uv run --with openpyxl python scripts/adjudicate_natural.py apply

``sheet`` compares the author's and Ibrahim's v2 sheets and writes one row per item where they
differ (intent, route, clarification targets, unsupported_reason, requested wall-clock time,
reminder text after ``canonical_text``), labels side by side, to
data/raw/private/labelling/adjudication.xlsx. The two annotators fill ``decision`` together:
``A`` (author's label), ``I`` (Ibrahim's label) or ``other`` (type the label in the ``final_``
columns), and a short ``why``. A row whose ``final_`` columns are filled is decided by them, and
``apply`` sets its decision to A, I or other by comparing that label with the two sheets.

``apply`` checks that every row is decided and writes the gold sheet
data/raw/private/labelling/natural_gold.xlsx: the author's sheet with the adjudicated labels put
in. Ibrahim's labels are taken with his format slips fixed (no reminder text or reminder_text
target on alarms; the time read as Cairo wall-clock time). It also writes
results/natural/adjudication.json (ids and decisions, no message texts).
Then: athar-data build natural_gold.xlsx -> validate --strict -> lock.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from athar.data.agreement import parse_when
from athar.data.contrib import _openpyxl
from athar.extract import canonical_text
from athar.timeexpr import GAP_ORDER

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "raw" / "private" / "labelling"
AUTHOR, OTHER = LAB / "natural_label_Amina_v2.xlsx", LAB / "natural_label_Ibrahim_v2.xlsx"
SHEET, GOLD = LAB / "adjudication.xlsx", LAB / "natural_gold.xlsx"
RECORD = ROOT / "results" / "natural" / "adjudication.json"
TZ = "Africa/Cairo"
FIELDS = ("intent", "route", "clarify_targets", "unsupported_reason", "datetime", "reminder_text")
DECISIONS = ("A", "I", "other")


def cell(v: Any) -> str:
    if v is None:
        return ""
    return v.isoformat() if isinstance(v, datetime) else str(v).strip()


def read(path: Path) -> dict[str, dict[str, str]]:
    ws = _openpyxl().load_workbook(path, read_only=True, data_only=True)["label"]
    rows = list(ws.iter_rows(values_only=True))
    head = [cell(h) for h in rows[0]]
    return {cell(r[0]): {h: cell(v) for h, v in zip(head, r, strict=False) if h} for r in rows[1:] if r[0]}


def targets(row: dict[str, str]) -> tuple[str, ...]:
    got = {t for t in re.split(r"[\s,;|]+", row["clarify_targets"]) if t}
    if row["intent"] == "create_alarm":
        got.discard("reminder_text")
    return tuple(t for t in GAP_ORDER if t in got)


def wall(row: dict[str, str]) -> str:
    """Requested time as Cairo wall-clock text; the written offset is not used."""
    return parse_when(row["datetime"], TZ).strftime("%Y-%m-%d %H:%M") if row["datetime"] else ""


def clean(row: dict[str, str]) -> dict[str, str]:
    """A label as it goes into the gold sheet (format slips fixed, nothing else changed)."""
    out = {f: row[f] for f in FIELDS}
    out["route"] = route = out["route"].upper()
    out["clarify_targets"] = " ".join(targets(row)) if route == "CLARIFY" else ""
    if row["intent"] != "unsupported":
        out["unsupported_reason"] = ""
    if row["intent"] == "create_alarm":
        out["reminder_text"] = ""
    if route != "ACT":
        out["datetime"] = ""
    if out["datetime"]:
        local = datetime.strptime(wall(row), "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(TZ))
        out["datetime"] = local.isoformat()
    return out


def same(x: dict[str, str], y: dict[str, str]) -> bool:
    """Two cleaned labels are the same label (reminder text compared as written)."""
    return all(x[f].strip() == y[f].strip() for f in FIELDS)


def differences(a: dict[str, str], b: dict[str, str]) -> list[str]:
    diff = [f for f in ("intent", "route") if a[f].upper() != b[f].upper()]
    if targets(a) != targets(b):
        diff.append("clarify_targets")
    if a["intent"] == b["intent"] == "unsupported" and a["unsupported_reason"] != b["unsupported_reason"]:
        diff.append("unsupported_reason")
    if a["route"].upper() == b["route"].upper() == "ACT":
        if wall(a) != wall(b):
            diff.append("datetime")
        if (a["intent"] == b["intent"] == "create_reminder"
                and canonical_text(a["reminder_text"]) != canonical_text(b["reminder_text"])):
            diff.append("reminder_text")
    return diff


def show(row: dict[str, str]) -> list[str]:
    return [row["intent"], row["route"].upper(), " ".join(targets(row)), row["unsupported_reason"],
            wall(row), "" if row["intent"] == "create_alarm" else row["reminder_text"]]


HOW = [
    "How to adjudicate (both annotators together)",
    "",
    "Each row is a message where the two sheets differ; 'differs' says in what.",
    "Read the message and docs/annotation_guide.md, agree on the right label, then fill:",
    "  decision = A      -> Amina's label is right",
    "  decision = I      -> Ibrahim's label is right",
    "  decision = other  -> neither: type the full label in the final_ columns",
    "                       (datetime as 2026-11-03 17:00, Cairo time)",
    "  why              -> one short reason (which guide rule decides it)",
    "Do not change any other column, row or file.",
    "",
    "طريقة الحسم (انتي وإبراهيم مع بعض)",
    "كل سطر رسالة اختلفتوا فيها، وعمود differs بيقول في إيه.",
    "اقروا الرسالة والـ annotation guide واتفقوا على التصنيف الصح، وبعدين:",
    "  A = تصنيف أمينة صح، I = تصنيف إبراهيم صح، other = الاتنين غلط واكتبوا التصنيف في أعمدة final_",
    "  why = سبب قصير (أنهي قاعدة في الـ guide)",
]


def cmd_sheet(_: argparse.Namespace) -> int:
    if SHEET.exists():
        raise SystemExit(f"{SHEET.relative_to(ROOT)} exists: it may hold decisions; move it away first")
    a, b = read(AUTHOR), read(OTHER)
    if set(a) != set(b):
        raise SystemExit("the two sheets do not hold the same ids")
    rows = [(i, d) for i in a if (d := differences(a[i], b[i]))]  # author's sheet order (shuffled)
    opx = _openpyxl()
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = opx.Workbook()
    ws = wb.active
    ws.title = "adjudicate"
    label = ["intent", "route", "clarify_targets", "unsupported_reason", "datetime", "reminder_text"]
    head = (["id", "message", "sent at", "differs"] + [f"amina_{x}" for x in label]
            + [f"ibrahim_{x}" for x in label] + ["decision"] + [f"final_{x}" for x in label] + ["why"])
    ws.append(head)
    for i, d in rows:
        ws.append([i, a[i]["text"], a[i]["ref_local"], ", ".join(d)] + show(a[i]) + show(b[i])
                  + [""] * (len(label) + 2))
    fills = {"amina_": "DDEBF7", "ibrahim_": "FCE4D6", "decision": "FFF2CC", "final_": "E2EFDA", "why": "FFF2CC"}
    for c in ws[1]:
        c.font = Font(bold=True)
        for prefix, colour in fills.items():
            if str(c.value).startswith(prefix):
                c.fill = PatternFill("solid", fgColor=colour)
    widths = {"id": 14, "message": 60, "sent at": 26, "differs": 22, "decision": 10, "why": 40}
    for col, name in enumerate(head, 1):
        letter = opx.utils.get_column_letter(col)
        ws.column_dimensions[letter].width = widths.get(name, 17)
        for c in ws[letter][1:]:
            c.alignment = Alignment(wrap_text=True, vertical="top",
                                    horizontal="right" if name in ("message", "sent at") else None)
    dv = DataValidation(type="list", formula1='"' + ",".join(DECISIONS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{opx.utils.get_column_letter(head.index('decision') + 1)}2:"
           f"{opx.utils.get_column_letter(head.index('decision') + 1)}{len(rows) + 1}")
    ws.freeze_panes = "C2"
    how = wb.create_sheet("how to")
    for line in HOW:
        how.append([line])
    how.column_dimensions["A"].width = 110
    wb.save(SHEET)
    count: dict[str, int] = {}
    for _, d in rows:
        for f in d:
            count[f] = count.get(f, 0) + 1
    print(f"{len(rows)} of {len(a)} items to adjudicate -> {SHEET.relative_to(ROOT)}")
    print("differ in: " + ", ".join(f"{f} {n}" for f, n in count.items()))
    return 0


def cmd_apply(_: argparse.Namespace) -> int:
    a, b = read(AUTHOR), read(OTHER)
    ws = _openpyxl().load_workbook(SHEET, read_only=True, data_only=True)["adjudicate"]
    rows = list(ws.iter_rows(values_only=True))
    head = [cell(h) for h in rows[0]]
    decided = [{h: cell(v) for h, v in zip(head, r, strict=False)} for r in rows[1:] if r[0]]
    for d in decided:  # a typed final label decides the row; A/I/other then only says whose it is
        if d["final_intent"]:
            typed = {f: d[f"final_{f}"] for f in FIELDS}
            fixed = clean(typed)
            if typed["intent"] == "create_alarm" and ("reminder_text" in typed["clarify_targets"]
                                                       or typed["reminder_text"]):
                print(f"note {d['id']}: alarms carry no reminder text; it is dropped from the final label")
            d["decision"] = ("A" if same(fixed, clean(a[d["id"]])) else
                             "I" if same(fixed, clean(b[d["id"]])) else "other")
    want = {i for i in a if differences(a[i], b[i])}
    problems = [f"{d['id']}: decision {d['decision']!r} is not one of {DECISIONS}"
                for d in decided if d["decision"] not in DECISIONS]
    problems += [f"{d['id']}: no reason in 'why'" for d in decided if not d["why"]]
    if missing := want - {d["id"] for d in decided}:
        problems.append(f"items missing from the sheet: {sorted(missing)}")
    if problems:
        print("\n".join(problems))
        return 1
    final: dict[str, dict[str, str]] = {}
    for d in decided:
        if d["decision"] == "I":
            final[d["id"]] = clean(b[d["id"]])
        elif d["decision"] == "other":
            final[d["id"]] = clean({f: d[f"final_{f}"] for f in FIELDS})
    bad = [f"{i}: CLARIFY with no target left (use 'other' and type the targets)"
           for i, x in final.items() if x["route"] == "CLARIFY" and not x["clarify_targets"]]
    bad += [f"{i}: ACT with no datetime (use 'other' and type it)"
            for i, x in final.items() if x["route"] == "ACT" and not x["datetime"]]
    if bad:
        print("\n".join(bad))
        return 1
    wb = _openpyxl().load_workbook(AUTHOR)
    gold = wb["label"]
    col = {cell(c.value): c.column for c in gold[1]}
    why = {d["id"]: d["why"] for d in decided}
    for r in gold.iter_rows(min_row=2):
        i = cell(r[0].value)
        if i in final:
            for f in FIELDS:  # set .value directly: ws.cell(value=None) leaves the old value
                gold.cell(row=r[0].row, column=col[f]).value = final[i][f] or None
        if i in why:  # guide 7.3: a sentence in notes for every adjudicated item
            old = cell(gold.cell(row=r[0].row, column=col["notes"]).value)
            gold.cell(row=r[0].row, column=col["notes"]).value = " | ".join(
                x for x in (old, f"adjudicated: {why[i]}") if x)
    wb.save(GOLD)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    record = {"protocol": "docs/protocol_natural.md (log 2026-11-01)", "items": len(a),
              "adjudicated": len(decided),
              "decisions": {k: sum(d["decision"] == k for d in decided) for k in DECISIONS},
              "rows": [{"id": d["id"], "differs": d["differs"].split(", "), "decision": d["decision"],
                        "final": {f: (final.get(d["id"]) or clean(a[d["id"]]))[f]
                                  for f in ("intent", "route", "clarify_targets", "unsupported_reason")}}
                       for d in decided]}
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(decided)} decisions ({record['decisions']}) -> {GOLD}; record -> {RECORD}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(required=True)
    sub.add_parser("sheet", help="write the adjudication sheet").set_defaults(func=cmd_sheet)
    sub.add_parser("apply", help="write the gold sheet from the decisions").set_defaults(func=cmd_apply)
    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
