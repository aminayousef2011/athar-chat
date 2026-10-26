"""S1 v1.1: S1's keyword rules plus the out-of-scope verbs missed on the v1 test
(docs/protocol_natural.md, section 5; clarification of 2026-10-26 in its log).

- deletion: the suffixed «الغيلي / لغيلي» (v1 lists «الغي» but not its suffixed form);
- modification: «قدم», «اخر / أخر / أخّر» and «خلي», each only when directly followed by a tool noun
  («قدم منبه الـ go-live ساعة», «اخر تذكير مكالمة العميل», «خلي تذكير الدوا بالليل»). The condition
  keeps «آخر النهار» (a vague time) and «فكرني اقدم ...» (reminder content) in scope.

The decision procedure is v1's (athar.intent.rules), with these cues added to its blockers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from athar import lexicon as L
from athar.intent import rules as R

DELETION_EXTRA = L._n(["الغيلي", "الغيلى", "لغيلي"])
MODIFY_BEFORE_TOOL = L._n(["قدم", "اخر", "أخر", "أخّر", "خلي"])
RX_BLOCKERS = re.compile(
    rf"{R.B}{R.CONJ}(?:{L.alt(R.DELETION + DELETION_EXTRA + R.MODIFICATION + R.OTHER_TOOL)}"
    rf"|{L.alt(MODIFY_BEFORE_TOOL)}\s+{L.alt(L.TOOL_NOUNS)}){R.E}"
)


def classify(nt: str) -> str:
    """Intent of one normalized text: :func:`athar.intent.rules.classify` with the v1.1 blockers."""
    if R.RX_RECURRING.search(nt) or R.RX_OTHER_PERSON.search(nt) or R.RX_REMIND_OTHER.search(nt):
        return "unsupported"
    cues = [(m.start(), "create_alarm") for m in R.RX_ALARM.finditer(nt)]
    cues += [(m.start(), "create_reminder") for m in R.RX_REMINDER.finditer(nt)]
    first = min(cues)[0] if cues else len(nt) + 1
    blocker = RX_BLOCKERS.search(nt)
    if blocker and blocker.start() < first:
        return "unsupported"
    if len({m.group().removeprefix("و").strip() for m in R.RX_VERB.finditer(nt)}) > 1:
        return "unsupported"
    return min(cues)[1] if cues else "unsupported"


class KeywordIntentV11:
    """S1 v1.1 intent recognizer (stateless). Pair it with core v1.1 (``CoreV11System``)."""

    name = "S1-v1.1"

    def predict(self, normalized_texts: Sequence[str]) -> list[str]:
        return [classify(t) for t in normalized_texts]
