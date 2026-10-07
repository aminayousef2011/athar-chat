"""Shared argument extractor: datetime + reminder text + clarification gaps.

Reminder text rule (task contract section 4, annotation guide): the reminder text is the request
minus (a) request framing (``فكرني``, ``remind me``, ``اعمل تذكير`` ...), (b) politeness fillers,
(c) every temporal expression, and (d) function words left dangling at either edge. The same
rule is applied to gold strings by :func:`canonical_text`, so prediction and gold are compared
on equal terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from athar import lexicon as L
from athar.normalize import Normalized, normalize
from athar.timeexpr import REMINDER_TEXT, Temporal, parse_temporal

B, E = r"(?<!\w)", r"(?!\w)"
RX_FRAMING = re.compile(
    rf"{B}(?:(?:و\s?)?{L.alt(L.TRIGGERS)}"
    rf"|{L.alt(L.SETUP_VERBS)}(?:\s+(?:لي|ليا|for\s+me))?\s+{L.alt(L.TOOL_NOUNS)}"
    rf"|{L.alt(L.TOOL_NOUNS)}|{L.alt(L.FILLERS)}|{L.alt(L.HEDGES)}){E}"
)
RX_TOKEN = re.compile(r"\w+")
_EDGE = set(L.EDGE_WORDS)
_VAGUE_CONTENT = set(L.VAGUE_CONTENT)


@dataclass(frozen=True)
class Extraction:
    temporal: Temporal
    reminder_text: str | None  # surface form, user's own spelling
    reminder_text_norm: str | None  # canonical form used for comparison
    normalized: str
    removed_spans: tuple[tuple[int, int], ...] = field(default=())

    @property
    def datetime(self) -> datetime | None:
        return self.temporal.datetime

    @property
    def gaps(self) -> frozenset[str]:
        """All gaps; the policy keeps only those required by the predicted tool."""
        g = set(self.temporal.gaps)
        if self.reminder_text_norm is None:
            g.add(REMINDER_TEXT)
        return frozenset(g)


def _strip_edges(tokens: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    toks = list(tokens)
    changed = True
    while toks and changed:
        changed = False
        if toks and toks[0][2] in _EDGE:
            toks.pop(0)
            changed = True
        if toks and toks[-1][2] in _EDGE:
            toks.pop()
            changed = True
        if toks and toks[0][2].startswith("بال") and len(toks[0][2]) > 3:
            s, e, w = toks[0]  # "بالاجتماع" -> "الاجتماع"
            toks[0] = (s + 1, e, w[1:])
            changed = True
    return toks


def _content_tokens(nt: str, removed: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    def kept(s: int, e: int) -> bool:
        return all(e <= a or s >= b for a, b in removed)

    return _strip_edges([(m.start(), m.end(), m.group()) for m in RX_TOKEN.finditer(nt) if kept(*m.span())])


def canonical_text(text: str) -> str | None:
    """Canonical comparison form of a reminder text (gold or predicted)."""
    nt = normalize(text).text
    removed = [m.span() for m in RX_FRAMING.finditer(nt)]
    toks = _content_tokens(nt, removed)
    out = " ".join(w for _, _, w in toks)
    return out or None


def _surface(nz: Normalized, toks: list[tuple[int, int, str]], removed: list[tuple[int, int]]) -> str:
    """Rebuild the user's own spelling; pieces separated by a removed span are joined by one space."""
    pieces: list[list[tuple[int, int, str]]] = []
    for t in toks:
        gap_s = pieces[-1][-1][1] if pieces else 0
        if pieces and all(e <= gap_s or s >= t[0] for s, e in removed):
            pieces[-1].append(t)
        else:
            pieces.append([t])
    return " ".join(nz.source_text(p[0][0], p[-1][1]).strip() for p in pieces)


def extract(text: str, ref: datetime, tz_name: str = "Africa/Cairo") -> Extraction:
    nz = normalize(text)
    temporal = parse_temporal(nz.text, ref, tz_name)
    removed = temporal.spans + [m.span() for m in RX_FRAMING.finditer(nz.text)]
    toks = _content_tokens(nz.text, removed)
    norm_text = " ".join(w for _, _, w in toks) or None
    if norm_text in _VAGUE_CONTENT or (norm_text and norm_text[0] == "ب" and norm_text[1:] in _VAGUE_CONTENT):
        norm_text = None  # "فكرني بحاجة" names no task
    surface = _surface(nz, toks, removed) if norm_text else None
    return Extraction(temporal, surface, norm_text, nz.text, tuple(sorted(removed)))
