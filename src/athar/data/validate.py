"""Dataset-level validation on top of the per-item schema (athar.schema.GoldItem).

Errors break the contract and block splitting. Warnings are likely annotation slips that a
human should look at; ``--strict`` turns them into errors.

Per item (schema): field types, closed label sets, route/intent/target consistency.
Across items:
- unique ``id``;
- ``timezone`` is an IANA zone and every datetime carries that zone's offset *at that instant*
  (``+03:00`` after Cairo's DST ends is an error, not a different time);
- ``pair_id`` joins exactly two items of one family and one intent: one ACT, one CLARIFY;
- the same normalized text never appears in two families (it would leak across splits);
- ``reminder_text_alts`` only next to a ``reminder_text``.
Warnings: ``code_switched`` disagrees with the presence of Latin letters; a gold reminder text
uses words that are not in the request.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from athar.data.io import Record
from athar.extract import canonical_text
from athar.normalize import norm
from athar.schema import GoldItem

RX_LATIN = re.compile(r"[A-Za-z]")
RX_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7} {self.where}: {self.message}"


def _fmt_offset(d: datetime) -> str:
    off = d.utcoffset()
    assert off is not None
    mins = int(off.total_seconds() // 60)
    sign = "+" if mins >= 0 else "-"
    return f"{sign}{abs(mins) // 60:02d}:{abs(mins) % 60:02d}"


def _offset_issue(name: str, d: datetime, tz: ZoneInfo) -> str | None:
    expected = d.astimezone(tz)
    if d.utcoffset() != expected.utcoffset():
        return (f"{name} offset {_fmt_offset(d)} is not {tz.key} at that instant "
                f"({_fmt_offset(expected)}); write {expected.isoformat()} if that instant is meant")
    return None


def _item_issues(r: Record) -> list[Issue]:
    it = r.item
    assert it is not None
    out: list[Issue] = []
    try:
        tz = ZoneInfo(it.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return [Issue("error", r.where, f"unknown IANA timezone {it.timezone!r}")]
    for name, d in (("ref_datetime", it.ref_datetime), ("args.datetime", it.args.datetime)):
        if d is not None and (msg := _offset_issue(name, d, tz)):
            out.append(Issue("error", r.where, msg))
    if it.args.reminder_text_alts and not it.args.reminder_text:
        out.append(Issue("error", r.where, "reminder_text_alts given without reminder_text"))

    has_latin = bool(RX_LATIN.search(it.text))
    if has_latin != it.code_switched:
        out.append(Issue("warning", r.where,
                         f"code_switched is {it.code_switched} but the text "
                         f"{'contains' if has_latin else 'has no'} Latin letters"))
    request_words = set(RX_WORD.findall(norm(it.text)))  # same tokenization as canonical_text
    request_words |= {w[1:] for w in request_words if w.startswith("بال")}  # "بالايجار" -> "الايجار"
    for gold in [it.args.reminder_text, *it.args.reminder_text_alts]:
        canon = canonical_text(gold) if gold else None
        missing = [w for w in (canon or "").split() if w not in request_words]
        if missing:
            out.append(Issue("warning", r.where,
                             f"reminder text {gold!r} uses words not in the request: {missing}"))
    return out


def _pair_issues(items: Sequence[GoldItem]) -> list[Issue]:
    pairs: dict[str, list[GoldItem]] = defaultdict(list)
    for it in items:
        if it.pair_id:
            pairs[it.pair_id].append(it)
    out = []
    for pid, members in sorted(pairs.items()):
        where = f"pair {pid}"
        ids = [m.id for m in members]
        if len(members) != 2:
            out.append(Issue("error", where, f"needs exactly 2 items, has {len(members)}: {ids}"))
            continue
        a, b = members
        if a.family_id != b.family_id:
            out.append(Issue("error", where, f"items {ids} are in different families"))
        if a.intent != b.intent:
            out.append(Issue("error", where, f"items {ids} have different intents"))
        if {a.route, b.route} != {"ACT", "CLARIFY"}:
            out.append(Issue("error", where,
                             f"needs one complete (ACT) and one incomplete (CLARIFY) item, "
                             f"has {sorted([a.route, b.route])}"))
    return out


def _duplicate_issues(records: Sequence[Record]) -> list[Issue]:
    by_text: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        assert r.item is not None
        by_text[norm(r.item.text)].append(r)
    out = []
    for group in by_text.values():
        if len(group) < 2:
            continue
        families = {r.item.family_id for r in group if r.item}
        where = ", ".join(r.where for r in group)
        if len(families) > 1:
            out.append(Issue("error", where,
                             f"same normalized text in families {sorted(families)}; "
                             f"put them in one family so they land in one split"))
        elif len({r.item.ref_datetime for r in group if r.item}) < len(group):
            out.append(Issue("error", where, "duplicate item (same text and ref_datetime)"))
    return out


def validate_records(records: Sequence[Record]) -> list[Issue]:
    issues = [Issue("error", r.where, e) for r in records for e in r.errors]
    valid = [r for r in records if r.item is not None]

    seen: dict[str, Record] = {}
    for r in valid:
        assert r.item is not None
        if r.item.id in seen:
            issues.append(Issue("error", r.where, f"duplicate id (first at {seen[r.item.id].where})"))
        else:
            seen[r.item.id] = r
        issues.extend(_item_issues(r))
    issues.extend(_pair_issues([r.item for r in valid if r.item]))
    issues.extend(_duplicate_issues(valid))
    return issues


def has_errors(issues: Sequence[Issue], strict: bool = False) -> bool:
    return any(i.level == "error" or strict for i in issues)
