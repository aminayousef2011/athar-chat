"""End-to-end runtime path: normalize -> (intent || extract) -> policy -> response -> gate.

Only the intent recognizer differs between systems; everything else is shared. The optional
confidence gate (athar.gate) turns a low-confidence execution into a confirmation question.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from athar.extract import extract
from athar.gate import apply_gate, confidences
from athar.normalize import norm
from athar.policy import decide
from athar.respond import build_response
from athar.schema import Response


class IntentRecognizer(Protocol):
    name: str

    def predict(self, normalized_texts: Sequence[str]) -> list[str]:
        """Return one of create_reminder / create_alarm / unsupported per text."""
        ...


class FixedIntent:
    """Returns given intents; used for oracle-intent upper bounds and for tests."""

    def __init__(self, intents: Sequence[str], name: str = "oracle") -> None:
        self.name = name
        self._intents = list(intents)

    def predict(self, normalized_texts: Sequence[str]) -> list[str]:
        if len(normalized_texts) != len(self._intents):
            raise ValueError("FixedIntent got a different number of texts than intents")
        return list(self._intents)


def respond_with_intent(intent: str, text: str, ref: datetime, tz_name: str = "Africa/Cairo",
                        system: str = "oracle") -> Response:
    ext = extract(text, ref, tz_name)
    return build_response(system, decide(intent, ext), ext)


class Pipeline:
    def __init__(self, recognizer: IntentRecognizer, gate_threshold: float | None = None) -> None:
        self.recognizer = recognizer
        self.gate_threshold = gate_threshold

    def run_batch(self, texts: Sequence[str], refs: Sequence[datetime],
                  tz_names: Sequence[str] | None = None) -> list[Response]:
        tz_names = tz_names or ["Africa/Cairo"] * len(texts)
        normalized = [norm(t) for t in texts]
        intents = self.recognizer.predict(normalized)
        conf = confidences(self.recognizer, normalized, intents)
        out = []
        for i, c, t, r, z in zip(intents, conf, texts, refs, tz_names, strict=True):
            gated = apply_gate(respond_with_intent(i, t, r, z, self.recognizer.name), c, self.gate_threshold)
            assert gated is not None
            out.append(gated)
        return out

    def run(self, text: str, ref: datetime, tz_name: str = "Africa/Cairo") -> Response:
        return self.run_batch([text], [ref], [tz_name])[0]
