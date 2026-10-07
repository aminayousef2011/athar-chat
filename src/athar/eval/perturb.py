"""Label-preserving perturbations for invariance testing (CheckList INV tests, Ribeiro et al. 2020).

Each perturbation rewrites a request the way real users vary their typing, and must leave the
gold label unchanged: same intent, route, targets, datetime and (canonical) reminder text. A
system is scored on the perturbed copies exactly as on the originals, and the report gives
Safe Task Success per perturbation plus the *invariance rate* (share of items whose output is
unchanged). Orthographic perturbations probe the shared normalizer; the lexical ones
(politeness, English trigger) probe the intent recognizers.

The complete/incomplete pairs in the dataset are the matching *directional* tests (contrast
sets, Gardner et al. 2020): removing one detail must turn ACT into CLARIFY.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from athar.schema import GoldItem

_TO_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ARABIC_LETTER = re.compile(r"[ء-ي]")
_HAMZA = str.maketrans("أإآ", "ااا")
_TRIGGER = re.compile(r"^(?:ممكن\s+)?(?:فكرني|فكريني)(?=\s)")


def _taa_marbuta(t: str) -> str | None:
    out = re.sub(r"ة(?=\s|$|[?؟،,.])", "ه", t)
    return out if out != t else None


def _hamza(t: str) -> str | None:
    out = t.translate(_HAMZA)
    return out if out != t else None


def _yaa(t: str) -> str | None:
    out = re.sub(r"ي(?=\s|$)", "ى", t)
    return out if out != t else None


def _digits(t: str) -> str | None:
    if re.search(r"[0-9]", t):
        return t.translate(_TO_INDIC)
    if re.search(r"[٠-٩]", t):
        return t.translate(_TO_WESTERN)
    return None


def _diacritics(t: str) -> str | None:
    """Add a fatha after every third Arabic letter (people sometimes vocalize words)."""
    count = 0

    def add(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return m.group() + ("َ" if count % 3 == 0 else "")

    out = _ARABIC_LETTER.sub(add, t)
    return out if out != t else None


def _elongation(t: str) -> str | None:
    words = t.split(" ")
    for i, w in enumerate(words):
        if len(w) > 2 and _ARABIC_LETTER.fullmatch(w[-1]):
            words[i] = w + w[-1] * 3
            return " ".join(words)
    return None


def _punctuation(t: str) -> str | None:
    stripped = re.sub(r"[?؟!.,،]", "", t).strip()
    return stripped + "؟" if stripped == t else stripped


def _politeness(t: str) -> str | None:
    return None if t.startswith("لو سمحت") else "لو سمحت " + t


def _english_trigger(t: str) -> str | None:
    out = _TRIGGER.sub("remind me", t, count=1)
    return out if out != t else None


PERTURBATIONS: dict[str, Callable[[str], str | None]] = {
    "taa_marbuta": _taa_marbuta,  # ة -> ه at word end
    "hamza": _hamza,  # أ إ آ -> ا
    "alef_maqsura": _yaa,  # final ي -> ى
    "digits": _digits,  # Western <-> Arabic-Indic digits
    "diacritics": _diacritics,
    "elongation": _elongation,  # فكرنييي
    "punctuation": _punctuation,
    "politeness": _politeness,  # لو سمحت prefix
    "english_trigger": _english_trigger,  # فكرني -> remind me
}


def perturbed_copies(items: Sequence[GoldItem], kinds: Sequence[str] | None = None) -> dict[str, list[GoldItem]]:
    """For each perturbation, copies of the items it applies to (ids suffixed ``--kind``)."""
    out: dict[str, list[GoldItem]] = {}
    for kind in kinds or PERTURBATIONS:
        fn = PERTURBATIONS[kind]
        copies = []
        for it in items:
            text = fn(it.text)
            if text is not None and text != it.text:
                update: dict[str, object] = {"text": text, "id": f"{it.id}--{kind}"}
                if kind == "english_trigger":
                    update["code_switched"] = True
                copies.append(it.model_copy(update=update))
        out[kind] = copies
    return out
