"""Inter-annotator agreement on a 60-item sample (or a dated self-review when no second
annotator is available).

Workflow: :func:`sample_items` draws a route-stratified sample (one item per family where
possible, never ``assistant_draft``), :func:`write_sheet` writes a blind CSV that opens in a
spreadsheet (labels empty, reference time spelled out in Arabic), the second annotator fills it
in, and :func:`agreement` compares it with the gold labels.

Reported: observed agreement and Cohen's kappa for intent and route; kappa per clarification
target (present/absent over all compared items) and exact target-set agreement on items both
annotators routed CLARIFY; unsupported_reason on items both called unsupported; datetime and
reminder text on items both routed ACT.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, get_args
from zoneinfo import ZoneInfo

from athar.extract import canonical_text
from athar.respond import describe_datetime
from athar.schema import INTENTS, ROUTES, GoldItem, UnsupportedReason
from athar.timeexpr import GAP_ORDER, cairo, is_after

SHEET_COLUMNS = ("id", "text", "ref_datetime", "ref_local", "timezone", "intent", "route",
                 "clarify_targets", "unsupported_reason", "datetime", "reminder_text", "notes")
REASONS: tuple[str, ...] = get_args(UnsupportedReason)


# --------------------------------------------------------------------------- kappa
def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa for two raters; ``None`` when chance agreement is 1 (kappa undefined)."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def interpret(k: float | None) -> str:
    """Landis & Koch (1977) bands, the usual reading in NLP annotation reports."""
    if k is None:
        return "undefined"
    for bound, label in ((0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"),
                         (0.8, "substantial")):
        if k <= bound:
            return label
    return "almost perfect"


# --------------------------------------------------------------------------- sampling
def _largest_remainder(total: int, counts: dict[str, int]) -> dict[str, int]:
    s = sum(counts.values())
    exact = {k: total * v / s for k, v in counts.items()}
    out = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def sample_items(items: Sequence[GoldItem], n: int, seed: int) -> list[GoldItem]:
    pool = [it for it in items if it.source != "assistant_draft"]
    if n > len(pool):
        raise ValueError(f"asked for {n} items, only {len(pool)} non-draft items exist")
    rng = random.Random(seed)
    by_route: dict[str, list[GoldItem]] = defaultdict(list)
    for it in sorted(pool, key=lambda x: x.id):
        by_route[it.route].append(it)
    quota = _largest_remainder(n, {r: len(v) for r, v in by_route.items()})
    chosen: list[GoldItem] = []
    used_families: set[str] = set()
    for route in ROUTES:
        cands, want = by_route.get(route, []), quota.get(route, 0)
        rng.shuffle(cands)
        # first pass: one item per family; second pass fills the quota if families run out
        picked: list[GoldItem] = []
        for c in cands:
            if len(picked) < want and c.family_id not in used_families:
                picked.append(c)
                used_families.add(c.family_id)
        picked += [c for c in cands if c not in picked][: want - len(picked)]
        chosen += picked
    rng.shuffle(chosen)  # annotators should not see the items grouped by route
    return chosen


def write_sheet(items: Sequence[GoldItem], path: str | Path) -> None:
    """Blind annotation sheet: request and reference time only, label columns empty. A ``.xlsx``
    path writes an Excel workbook with text cells and drop-down lists (Excel cannot rewrite the
    datetimes the second annotator types); any other path writes a CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if Path(path).suffix.lower() == ".xlsx":
        from athar.data.contrib import labelling_workbook

        labelling_workbook([{"id": it.id, "text": it.text, "ref_datetime": it.ref_datetime.isoformat(),
                             "ref_local": describe_datetime(it.ref_datetime.astimezone(ZoneInfo(it.timezone))),
                             "timezone": it.timezone} for it in items], path, columns=SHEET_COLUMNS)
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel shows Arabic
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for it in items:
            local = it.ref_datetime.astimezone(ZoneInfo(it.timezone))
            w.writerow({"id": it.id, "text": it.text, "ref_datetime": it.ref_datetime.isoformat(),
                        "ref_local": describe_datetime(local), "timezone": it.timezone})


# --------------------------------------------------------------------------- reading labels
@dataclass(frozen=True)
class Label:
    intent: str
    route: str
    targets: frozenset[str]
    reason: str | None
    datetime: datetime | None
    reminder_text: str | None


def _cell(row: dict[str, Any], key: str) -> str:
    v = row.get(key)
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(map(str, v))
    return str(v).strip()


_DAY_MONTH = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})(?:[ T](\d{1,2}):(\d{2}))?")
EXCEL_DATE = ("Excel stored the datetime as a date, so day and month may be swapped (3/11 read as "
              "11 March); type it as text: 2026-11-03 17:00 or 3/11/2026 17:00")


def parse_when(value: str, tz: str) -> datetime:
    """ISO (``2026-11-03 17:00``) or day/month text (``3/11/2026 17:00``, contract 4.1)."""
    v = value.strip()
    if m := _DAY_MONTH.fullmatch(v):
        d, mo, y, hh, mm = m.groups()
        return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), tzinfo=ZoneInfo(tz))
    return cairo(v.replace(" ", "T"), tz)


def _xlsx_rows(path: Path) -> list[dict[str, Any]]:
    """Sheet rows as strings, keeping one fact the strings would lose: whether Excel turned the
    ``datetime`` cell into a date (then day and month may have been swapped)."""
    from athar.data.contrib import _openpyxl

    ws = _openpyxl().load_workbook(path, read_only=True, data_only=True).worksheets[0]
    raw = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in raw[0]]
    rows = []
    for r in raw[1:]:
        if any(v not in (None, "") for v in r):
            cells = dict(zip(header, r, strict=False))
            row: dict[str, Any] = {h: ("" if v is None else str(v)).strip() for h, v in cells.items() if h}
            row["_excel_date"] = isinstance(cells.get("datetime"), datetime)
            rows.append(row)
    return rows


def _label(row: dict[str, Any], tz: str) -> Label:
    intent, route = _cell(row, "intent"), _cell(row, "route").upper()
    if intent not in INTENTS:
        raise ValueError(f"intent {intent!r} not in {INTENTS}")
    if route not in ROUTES:
        raise ValueError(f"route {route!r} not in {ROUTES}")
    targets = frozenset(t for t in re.split(r"[\s,;|]+", _cell(row, "clarify_targets")) if t)
    if bad := targets - set(GAP_ORDER):
        raise ValueError(f"unknown clarify_targets {sorted(bad)}")
    reason = _cell(row, "unsupported_reason") or None
    if reason is not None and reason not in REASONS:
        raise ValueError(f"unsupported_reason {reason!r} not in {REASONS}")
    # the moment is compared on ACT items only; a date written on another route is ignored,
    # so it can never make the row unreadable and drop it from the comparison
    when = _cell(row, "datetime") if route == "ACT" else ""
    if when and row.get("_excel_date"):
        raise ValueError(EXCEL_DATE)
    return Label(intent, route, targets, reason, parse_when(when, tz) if when else None,
                 _cell(row, "reminder_text") or None)


def read_labels(path: str | Path) -> tuple[dict[str, Label], list[str]]:
    """Second-annotator labels from the sheet (.xlsx or .csv) or a JSONL file; returns (labels, errors)."""
    p = Path(path)
    rows: list[dict[str, Any]]
    if p.suffix.lower() == ".xlsx":
        rows = _xlsx_rows(p)
    elif p.suffix.lower() == ".csv":
        with p.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    labels: dict[str, Label] = {}
    errors = []
    for i, row in enumerate(rows, start=2 if p.suffix.lower() in (".csv", ".xlsx") else 1):
        iid = _cell(row, "id")
        if not iid:
            errors.append(f"row {i}: no id")
            continue
        if not _cell(row, "intent") and not _cell(row, "route"):
            errors.append(f"row {i} ({iid}): not annotated")
            continue
        try:
            labels[iid] = _label({**row, **row.get("args", {})} if "args" in row else row,
                                 _cell(row, "timezone") or "Africa/Cairo")
        except ValueError as e:
            errors.append(f"row {i} ({iid}): {e}")
    return labels, errors


def gold_label(it: GoldItem) -> Label:
    return Label(it.intent, it.route, frozenset(it.clarify_targets), it.unsupported_reason,
                 it.args.datetime, it.args.reminder_text)


# --------------------------------------------------------------------------- report
def _confusion(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    c = Counter(zip(a, b, strict=True))
    return [[c[(x, y)] for y in labels] for x in labels]


def _categorical(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    n = len(a)
    k = cohen_kappa(a, b)
    return {"n": n, "agreement": (sum(x == y for x, y in zip(a, b, strict=True)) / n) if n else None,
            "kappa": k, "reading": interpret(k), "labels": list(labels),
            "confusion": _confusion(a, b, labels)}


def _rate(flags: Sequence[bool]) -> dict[str, Any]:
    return {"n": len(flags), "agreement": (sum(flags) / len(flags)) if flags else None}


def agreement(gold: Sequence[GoldItem], other: dict[str, Label]) -> dict[str, Any]:
    by_id = {it.id: it for it in gold}
    ids = sorted(set(by_id) & set(other))
    g = [gold_label(by_id[i]) for i in ids]
    o = [other[i] for i in ids]
    both_clarify = [(x, y) for x, y in zip(g, o, strict=True) if x.route == y.route == "CLARIFY"]
    both_unsup = [(x, y) for x, y in zip(g, o, strict=True) if x.intent == y.intent == "unsupported"]
    both_act = [(x, y) for x, y in zip(g, o, strict=True) if x.route == y.route == "ACT"]

    def same_text(x: Label, y: Label, it: GoldItem) -> bool:
        golds = {canonical_text(t) for t in [x.reminder_text, *it.args.reminder_text_alts] if t}
        return (canonical_text(y.reminder_text) if y.reminder_text else None) in (golds or {None})

    def same_instant(x: datetime | None, y: datetime | None) -> bool:
        return (x is None and y is None) or (x is not None and y is not None
                                             and not is_after(x, y) and not is_after(y, x))

    return {
        "n_compared": len(ids),
        "unknown_ids": sorted(set(other) - set(by_id)),
        "intent": _categorical([x.intent for x in g], [y.intent for y in o], INTENTS),
        "route": _categorical([x.route for x in g], [y.route for y in o], ROUTES),
        "targets_exact_set": _rate([x.targets == y.targets for x, y in both_clarify]),
        "targets_kappa": {t: _categorical([str(t in x.targets) for x in g],
                                          [str(t in y.targets) for y in o], ("True", "False"))
                          for t in GAP_ORDER},
        "unsupported_reason": _categorical([x.reason or "-" for x, _ in both_unsup],
                                           [y.reason or "-" for _, y in both_unsup], REASONS),
        "datetime_same_instant": _rate([same_instant(x.datetime, y.datetime) for x, y in both_act]),
        "reminder_text_same": _rate([same_text(x, y, by_id[i])
                                     for i, x, y in zip(ids, g, o, strict=True)
                                     if x.route == y.route == "ACT" and x.intent == y.intent == "create_reminder"]),
        "disagreements": [
            {"id": i, "gold": f"{x.intent}/{x.route}/{','.join(sorted(x.targets, key=GAP_ORDER.index))}",
             "annotator": f"{y.intent}/{y.route}/{','.join(sorted(y.targets, key=GAP_ORDER.index))}"}
            for i, x, y in zip(ids, g, o, strict=True)
            if (x.intent, x.route, x.targets) != (y.intent, y.route, y.targets)
        ],
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _k(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def report_markdown(rep: dict[str, Any], annotator: str, date: str) -> str:
    lines = [f"# Annotation agreement ({annotator}, {date})", "",
             f"Items compared: {rep['n_compared']}", "",
             "| Field | n | Agreement | Cohen's κ | Reading |", "|---|---|---|---|---|"]
    for name in ("intent", "route"):
        r = rep[name]
        lines.append(f"| {name} | {r['n']} | {_pct(r['agreement'])} | {_k(r['kappa'])} | {r['reading']} |")
    for t, r in rep["targets_kappa"].items():
        lines.append(f"| target `{t}` (present/absent) | {r['n']} | {_pct(r['agreement'])} | "
                     f"{_k(r['kappa'])} | {r['reading']} |")
    r = rep["unsupported_reason"]
    lines.append(f"| unsupported_reason (both unsupported) | {r['n']} | {_pct(r['agreement'])} | "
                 f"{_k(r['kappa'])} | {r['reading']} |")
    for name, label in (("targets_exact_set", "target set (both CLARIFY)"),
                        ("datetime_same_instant", "datetime (both ACT)"),
                        ("reminder_text_same", "reminder text (both ACT reminders)")):
        r = rep[name]
        lines.append(f"| {label} | {r['n']} | {_pct(r['agreement'])} | | |")
    for name in ("intent", "route"):
        r = rep[name]
        lines += ["", f"**{name} confusion** (rows: gold, columns: annotator)", "",
                  "| | " + " | ".join(r["labels"]) + " |", "|---" * (len(r["labels"]) + 1) + "|"]
        lines += [f"| {lab} | " + " | ".join(map(str, row)) + " |"
                  for lab, row in zip(r["labels"], r["confusion"], strict=True)]
    if rep["disagreements"]:
        lines += ["", "**Label disagreements** (intent/route/targets)", "",
                  "| id | gold | annotator |", "|---|---|---|"]
        lines += [f"| {d['id']} | {d['gold']} | {d['annotator']} |" for d in rep["disagreements"]]
    if rep["unknown_ids"]:
        lines += ["", f"Ids not in the gold dataset (ignored): {rep['unknown_ids']}"]
    return "\n".join(lines) + "\n"
