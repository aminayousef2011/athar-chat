"""Core v1.1: the shared extractor with the three fixes of docs/protocol_natural.md, section 5.

1. Request framing also covers the want-verbs («عايز», «محتاج», ...) and the spellings «اعمللي»,
   each only when directly followed by a tool noun («عايز تذكير», «اعمللي منبه»).
2. More fillers are removed from the reminder text («كده», «لو ينفع», «لو عندك وقت», ...).
3. A day-period word inside a purpose clause (after the first «عشان/علشان») is not a time
   expression when a clock time occurs before that clause («الساعة 9 الصبح عشان اصحى لصلاة الفجر»).

Everything else is v1: the normalizer, the time detector and resolver, the policy, the templates
and the scoring (gold and predicted reminder texts are still compared with v1's ``canonical_text``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from athar import extract as X
from athar import lexicon as L
from athar.normalize import normalize
from athar.policy import decide
from athar.respond import build_response
from athar.schema import INTENTS, GoldItem, Response
from athar.timeexpr import Temporal, detect, resolve

B, E = r"(?<!\w)", r"(?!\w)"
WANT_VERBS = L._n(["عايز", "عاوز", "عايزه", "عايزة", "محتاج", "محتاجه", "محتاجة"])
SPELLINGS = L._n(["اعمللي", "اعملّي"])
EXTRA_FILLERS = L._n(["كده", "لو ينفع", "لو تقدر", "لو عندك وقت", "بعد اذنك"])
RX_FRAMING = re.compile(
    rf"{B}(?:(?:و\s?)?{L.alt(L.TRIGGERS)}"
    rf"|{L.alt(L.SETUP_VERBS + SPELLINGS + WANT_VERBS)}(?:\s+(?:لي|ليا|for\s+me))?\s+{L.alt(L.TOOL_NOUNS)}"
    rf"|{L.alt(L.TOOL_NOUNS)}|{L.alt(L.FILLERS + EXTRA_FILLERS)}|{L.alt(L.HEDGES)}){E}"
)
RX_PURPOSE = re.compile(rf"{B}(?:عشان|علشان){E}")
VERSION = "1.1"


def parse_temporal(normalized_text: str, ref: datetime, tz_name: str = "Africa/Cairo") -> Temporal:
    """v1 detection and resolution, without period words inside a purpose clause that follows a
    clock time (fix 3)."""
    ref = ref.astimezone(ZoneInfo(tz_name))
    mentions = detect(normalized_text, ref)
    purpose = RX_PURPOSE.search(normalized_text)
    if purpose and any(m.kind == "clock" and m.end <= purpose.start() for m in mentions):
        mentions = [m for m in mentions if not (m.kind == "period" and m.start >= purpose.end())]
    return resolve(mentions, ref, tz_name)


def extract(text: str, ref: datetime, tz_name: str = "Africa/Cairo") -> X.Extraction:
    """:func:`athar.extract.extract` with the v1.1 framing, fillers and purpose-clause rule."""
    nz = normalize(text)
    temporal = parse_temporal(nz.text, ref, tz_name)
    removed = temporal.spans + [m.span() for m in RX_FRAMING.finditer(nz.text)]
    toks = X._content_tokens(nz.text, removed)
    norm_text = " ".join(w for _, _, w in toks) or None
    if norm_text in X._VAGUE_CONTENT or (norm_text and norm_text[0] == "ب" and norm_text[1:] in X._VAGUE_CONTENT):
        norm_text = None
    surface = X._surface(nz, toks, removed) if norm_text else None
    return X.Extraction(temporal, surface, norm_text, nz.text, tuple(sorted(removed)))


def respond_with_intent(intent: str, text: str, ref: datetime, tz_name: str = "Africa/Cairo",
                        system: str = "oracle-v1.1") -> Response:
    ext = extract(text, ref, tz_name)
    return build_response(system, decide(intent, ext), ext)


class CoreV11System:
    """An intent recognizer (or the gold intent) followed by core v1.1. It answers whole items, so
    the v1 harness scores it as it scores every other system."""

    def __init__(self, recognizer: object | None, name: str) -> None:
        self.recognizer = recognizer   # None: the gold intent (oracle v1.1)
        self.name = name

    def respond_items(self, items: Sequence[GoldItem]) -> list[Response | None]:
        from athar.normalize import norm

        if self.recognizer is None:
            intents = [it.intent for it in items]
        else:
            intents = list(self.recognizer.predict([norm(it.text) for it in items]))  # type: ignore[attr-defined]
        return [respond_with_intent(i, it.text, it.ref_datetime, it.timezone, self.name) if i in INTENTS else None
                for i, it in zip(intents, items, strict=True)]
