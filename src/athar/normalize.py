"""Orthographic normalizer shared by every system (S1-S3) and by the argument extractor.

The rules follow the common Arabic NLP conventions (as in CAMeL Tools' ``normalize_alef``,
``normalize_alef_maksura`` and ``normalize_teh_marbuta``) plus a few choices for informal
Egyptian writing:

* NFKC, which folds Arabic presentation forms and the lam-alef ligature into base letters
* diacritics (tashkeel, U+064B-U+065F, superscript alef U+0670) and tatweel (U+0640) removed
* alef variants (أ إ آ ٱ) -> ا, alef maksura ى -> ي, taa marbuta ة -> ه
* Arabic-Indic and Extended Arabic-Indic digits -> ASCII digits
* Arabic punctuation (، ؛ ؟) -> ASCII equivalents
* Latin text lower-cased (code-switching)
* letter elongation (3+ repeats of the same letter) collapsed to one letter
* a space inserted between a digit and an adjacent letter (``الساعة7م`` -> ``الساعه 7 م``)
* whitespace collapsed

Every output character keeps the index of the input character it came from, so a span found
in normalized text maps back to the user's original wording (see :meth:`Normalized.source_span`).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

_TATWEEL = "ـ"
_ALEF_VARIANTS = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
_CHAR_MAP = {
    **_ALEF_VARIANTS,
    "ى": "ي",  # alef maksura -> yaa
    "ة": "ه",  # taa marbuta -> haa
    "،": ",",  # Arabic comma
    "؛": ";",  # Arabic semicolon
    "؟": "?",  # Arabic question mark
    "٫": ".",  # Arabic decimal separator
    "٬": ",",  # Arabic thousands separator
}
for _i in range(10):
    _CHAR_MAP[chr(0x0660 + _i)] = str(_i)  # Arabic-Indic digits
    _CHAR_MAP[chr(0x06F0 + _i)] = str(_i)  # Extended Arabic-Indic (Persian/Urdu) digits


def _is_diacritic(ch: str) -> bool:
    cp = ord(ch)
    return 0x064B <= cp <= 0x065F or cp == 0x0670 or ch == _TATWEEL


def _is_arabic_letter(ch: str) -> bool:
    return "ء" <= ch <= "ي"


@dataclass(frozen=True)
class Normalized:
    """Normalized text plus, for each output character, the index of its source character."""

    original: str
    text: str
    offsets: tuple[int, ...]

    def source_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a half-open span of ``text`` to the covering half-open span of ``original``."""
        if start >= end:
            raise ValueError("empty span")
        return self.offsets[start], self.offsets[end - 1] + 1

    def source_text(self, start: int, end: int) -> str:
        s, e = self.source_span(start, end)
        return self.original[s:e]


def normalize(text: str) -> Normalized:
    chars: list[tuple[str, int]] = []
    for idx, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch):
            if _is_diacritic(c):
                continue
            c = _CHAR_MAP.get(c, c)
            if c.isascii():
                c = c.lower()
            if c.isspace():
                c = " "
            chars.append((c, idx))

    # collapse elongation: 3+ identical Arabic letters -> 1 (Arabic never triples a letter)
    collapsed: list[tuple[str, int]] = []
    i = 0
    while i < len(chars):
        j = i
        while j < len(chars) and chars[j][0] == chars[i][0]:
            j += 1
        run = j - i
        keep = 1 if (run >= 3 and _is_arabic_letter(chars[i][0])) else run
        collapsed.extend(chars[i : i + keep])
        i = j

    # split digits from adjacent letters, then collapse whitespace
    out: list[tuple[str, int]] = []
    for c, idx in collapsed:
        if out:
            prev = out[-1][0]
            boundary = (prev.isdigit() and c.isalpha()) or (prev.isalpha() and c.isdigit())
            if boundary:
                out.append((" ", idx))
        if c == " " and (not out or out[-1][0] == " "):
            continue
        out.append((c, idx))
    while out and out[-1][0] == " ":
        out.pop()

    return Normalized(original=text, text="".join(c for c, _ in out), offsets=tuple(i for _, i in out))


def norm(text: str) -> str:
    """Shortcut returning only the normalized string."""
    return normalize(text).text
