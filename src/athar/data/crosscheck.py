"""Gold vs. shared core with the gold intent (the oracle-intent path), item by item.

A disagreement means either an annotation slip or an extractor/policy gap. Both are useful on
train and dev. On the test split they are not: fixing the extractor (or the gold) because of a
test item leaks the test set into the system, and the contract freezes the extraction rules
before the test split is opened. The CLI therefore only runs this on train/dev through a split
manifest, or on a file made only of ``assistant_draft`` items (the pilot).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from athar.extract import canonical_text
from athar.pipeline import respond_with_intent
from athar.schema import GoldItem
from athar.timeexpr import is_after


@dataclass(frozen=True)
class Disagreement:
    id: str
    field: str
    gold: str
    system: str


def _same_instant(a: datetime | None, b: datetime | None) -> bool:
    if a is None or b is None:
        return a is b
    return not is_after(a, b) and not is_after(b, a)


def _text_ok(gold: GoldItem, predicted: str | None) -> bool:
    golds = [g for g in [gold.args.reminder_text, *gold.args.reminder_text_alts] if g]
    if not golds:
        return predicted is None
    return predicted is not None and canonical_text(predicted) in {canonical_text(g) for g in golds}


def crosscheck(items: Sequence[GoldItem]) -> list[Disagreement]:
    out: list[Disagreement] = []
    for it in items:
        r = respond_with_intent(it.intent, it.text, it.ref_datetime, it.timezone)
        if r.route != it.route:
            out.append(Disagreement(it.id, "route", it.route, r.route))
        if r.clarify_targets != it.clarify_targets:
            out.append(Disagreement(it.id, "clarify_targets", ",".join(it.clarify_targets) or "-",
                                    ",".join(r.clarify_targets) or "-"))
        if it.route == "ACT" and r.route == "ACT":
            assert r.tool_call is not None
            if not _same_instant(it.args.datetime, r.tool_call.datetime):
                out.append(Disagreement(it.id, "datetime", str(it.args.datetime.isoformat()
                                                               if it.args.datetime else None),
                                        r.tool_call.datetime.isoformat()))
            if it.intent == "create_reminder" and not _text_ok(it, r.tool_call.reminder_text):
                out.append(Disagreement(it.id, "reminder_text", str(it.args.reminder_text),
                                        str(r.tool_call.reminder_text)))
    return out
