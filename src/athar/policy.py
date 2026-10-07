"""Deterministic execution policy shared by every system.

    unsupported intent                         -> ABSTAIN
    supported intent, no gap required by tool  -> ACT
    supported intent, otherwise                -> CLARIFY (targets = required gaps)

Intent recognizers only supply the intent label; the route always comes from here.
"""

from __future__ import annotations

from dataclasses import dataclass

from athar.extract import Extraction
from athar.schema import REQUIRED_GAPS
from athar.timeexpr import GAP_ORDER


@dataclass(frozen=True)
class Decision:
    route: str
    intent: str
    targets: tuple[str, ...]


def decide(intent: str, ext: Extraction) -> Decision:
    if intent == "unsupported":
        return Decision("ABSTAIN", intent, ())
    if intent not in REQUIRED_GAPS:
        raise ValueError(f"unknown intent {intent!r}")
    blocking = ext.gaps & REQUIRED_GAPS[intent]
    if not blocking:
        return Decision("ACT", intent, ())
    return Decision("CLARIFY", intent, tuple(g for g in GAP_ORDER if g in blocking))
