"""Core v1.2 (post hoc, exploratory): core v1.1 plus four fixes for the extractor errors that the
oracle (gold intents) made on the natural set.

The fixes were written from the natural set's oracle failures only (``scripts/posthoc_core_v12.py``
lists them); the v3 set was not inspected and is scored once, as a held-out check. Core v1.1 stays
frozen: every registered result keeps using it, and nothing here changes a frozen system.

1. **Emphasis words** («ضروري») are removed from the reminder text wherever they occur, as the
   adjudicated natural-set gold does («اعدل الrisk register ضروري» -> «اعدل الrisk register»).
2. **Want-verb + trigger framing**: a want-verb directly before a reminder trigger is framing
   («عايز تفكرني», «محتاج تفكرني», «وعايزك تفكرني»), and so is a want-verb left at the start of the
   reminder text («فكرني … محتاج اجهز الابديت» -> «اجهز الابديت»).
3. **Vague content**: a reminder text that is only a vague noun («حاجه»), optionally with
   intensifiers («مهمه», «جدا»), a preposition and one verb before it («اراجع علي حاجه مهمه»), is
   vague (contract: ``reminder_text`` gap), not executable content.
4. **Weekday after next** («الخميس بعد الجاي»): one week after the weekday that «الجاي» names
   (contract 4.1: «X الجاي» = 1-7 days ahead). The frozen contract left this form open
   (docs/protocol_natural.md, adjudication log); the annotators resolved it this way.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from athar import extract as X
from athar import lexicon as L
from athar import timeexpr as T
from athar.normalize import normalize
from athar.policy import decide
from athar.respond import build_response
from athar.schema import INTENTS, GoldItem, Response
from athar.v2 import core as V11

B, E = V11.B, V11.E
VERSION = "1.2"

EMPHASIS = L._n(["ضروري", "ضروريه"])
WANT = L._n(["عايز", "عاوز", "عايزه", "عاوزه", "محتاج", "محتاجه", "عايزك", "عاوزك", "عايزاك", "محتاجك"])
VAGUE_NOUNS = frozenset(L._n(["حاجة", "حاجات"]))
INTENSIFIERS = frozenset(L._n(["مهمة", "جدا", "اوي", "خالص"]))
PREPOSITIONS = frozenset(L._n(["على", "علي", "في", "ع"]))

RX_EMPHASIS = re.compile(rf"{B}{L.alt(EMPHASIS)}{E}")
RX_WANT_TRIGGER = re.compile(rf"{B}(?:و\s?)?{L.alt(WANT)}(?=\s+{L.alt(L.TRIGGERS)}{E})")
RX_AFTER_NEXT = re.compile(
    rf"{B}{T.CONJ}(?:يوم\s+)?(?P<wd>{L.alt(T._AR(L.WEEKDAYS))})\s+بعد\s+{L.alt(L.WEEKDAY_NEXT)}{E}")


def parse_temporal(normalized_text: str, ref: datetime, tz_name: str = "Africa/Cairo") -> T.Temporal:
    """Core v1.1 detection (with its purpose-clause rule), with «X بعد الجاي» read as one week
    after «X الجاي» (fix 4)."""
    local = ref.astimezone(ZoneInfo(tz_name))
    mentions = T.detect(normalized_text, local)
    purpose = V11.RX_PURPOSE.search(normalized_text)
    if purpose and any(m.kind == "clock" and m.end <= purpose.start() for m in mentions):
        mentions = [m for m in mentions if not (m.kind == "period" and m.start >= purpose.end())]
    for m in RX_AFTER_NEXT.finditer(normalized_text):
        wd = int(L.WEEKDAYS[m["wd"]])
        today = local.date()
        delta = (wd - today.weekday()) % 7
        day = today + timedelta(days=(delta or 7) + 7)
        mentions = [x for x in mentions if x.end <= m.start() or x.start >= m.end()]
        mentions.append(T.Mention("date", m.start(), m.end(), T.DateRef(frozenset({day}), "weekday after next")))
    mentions.sort(key=lambda x: x.start)
    return T.resolve(mentions, local, tz_name)


def is_vague(words: Sequence[str]) -> bool:
    """[verb] [preposition] (ب)حاجه [intensifiers…]: content with nothing specific in it (fix 3)."""
    w = list(words)
    while w and w[-1] in INTENSIFIERS:
        w.pop()
    if not w:
        return False
    last = w.pop()
    if last not in VAGUE_NOUNS and not (last[:1] == "ب" and last[1:] in VAGUE_NOUNS):
        return False
    if w and w[-1] in PREPOSITIONS:
        w.pop()
    return len(w) <= 1


def extract(text: str, ref: datetime, tz_name: str = "Africa/Cairo") -> X.Extraction:
    """Core v1.1's extraction with fixes 1-4."""
    nz = normalize(text)
    temporal = parse_temporal(nz.text, ref, tz_name)
    removed = (temporal.spans + [m.span() for m in V11.RX_FRAMING.finditer(nz.text)]
               + [m.span() for m in RX_EMPHASIS.finditer(nz.text)]
               + [m.span() for m in RX_WANT_TRIGGER.finditer(nz.text)])
    toks = X._content_tokens(nz.text, removed)
    while toks and toks[0][2] in WANT:   # a want-verb left at the start of the text (fix 2)
        removed.append((toks[0][0], toks[0][1]))
        toks = X._content_tokens(nz.text, removed)
    words = [w for _, _, w in toks]
    norm_text = " ".join(words) or None
    if norm_text in X._VAGUE_CONTENT or (norm_text and norm_text[0] == "ب" and norm_text[1:] in X._VAGUE_CONTENT):
        norm_text = None
    if norm_text and is_vague(words):
        norm_text = None
    surface = X._surface(nz, toks, removed) if norm_text else None
    return X.Extraction(temporal, surface, norm_text, nz.text, tuple(sorted(removed)))


def respond_with_intent(intent: str, text: str, ref: datetime, tz_name: str = "Africa/Cairo",
                        system: str = "oracle-v1.2") -> Response:
    ext = extract(text, ref, tz_name)
    return build_response(system, decide(intent, ext), ext)


class CoreV12System:
    """An intent recognizer (or the gold intent) followed by core v1.2."""

    def __init__(self, recognizer: object | None, name: str) -> None:
        self.recognizer = recognizer
        self.name = name

    def respond_items(self, items: Sequence[GoldItem]) -> list[Response | None]:
        from athar.normalize import norm

        if self.recognizer is None:
            intents = [it.intent for it in items]
        else:
            intents = list(self.recognizer.predict([norm(it.text) for it in items]))  # type: ignore[attr-defined]
        return [respond_with_intent(i, it.text, it.ref_datetime, it.timezone, self.name) if i in INTENTS else None
                for i, it in zip(intents, items, strict=True)]
