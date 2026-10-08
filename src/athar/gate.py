"""Confidence gate: never execute a tool call the intent recognizer is unsure about.

The confidence of a prediction is the probability the recognizer gives to the label it chose.
When the policy would ACT but that confidence is below the threshold ``tau``, the gate asks the
user to confirm instead (route CLARIFY, target ``confirm``) and nothing is executed. The gate
never changes CLARIFY or ABSTAIN, so it can only remove executions (selective prediction:
Geifman & El-Yaniv 2017). S1 (rules) has no probabilities and is never gated.

``tau`` is chosen on dev only (``athar-train fit``, rule in configs/train.yaml) and frozen with
the models; the pre-registered metrics are reported ungated (docs/evaluation_protocol.md 6.4).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from athar.respond import describe_datetime
from athar.schema import INTENTS, GoldItem, Response

CONFIRM = "confirm"
#: thresholds reported in every risk-coverage table (max probability over 3 labels is >= 1/3)
DEFAULT_THRESHOLDS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CURVE_METRICS = ("safe_task_success", "unsafe_execution_rate", "execution_precision",
                 "unnecessary_clarification_rate")


def confidences(recognizer: Any, normalized_texts: Sequence[str], intents: Sequence[str]) -> list[float | None]:
    """Probability of each chosen label, or ``None`` for a recognizer without probabilities
    (S1, the oracle) and for an unparseable label."""
    if not hasattr(recognizer, "predict_proba"):
        return [None] * len(intents)
    proba = recognizer.predict_proba(list(normalized_texts), INTENTS)
    return [float(p[INTENTS.index(i)]) if i in INTENTS else None
            for p, i in zip(proba, intents, strict=True)]


def confirm_message(r: Response) -> str:
    """The confirmation question, built from what would have been executed (gender-neutral)."""
    assert r.tool_call is not None
    when = describe_datetime(r.tool_call.datetime)
    if r.tool_call.name == "create_reminder":
        return f"للتأكيد: أفكرك بـ«{r.tool_call.reminder_text}» {when}، صح كده؟"
    return f"للتأكيد: أظبط منبه يرن {when}، صح كده؟"


def apply_gate(r: Response | None, confidence: float | None, tau: float | None) -> Response | None:
    """The gated response; ``r`` itself when the gate does not fire. Records the confidence."""
    if r is None:
        return None
    if r.route == "ACT" and tau is not None and confidence is not None and confidence < tau:
        return r.model_copy(update={"route": "CLARIFY", "tool_call": None, "clarify_targets": [CONFIRM],
                                    "message": confirm_message(r), "confidence": confidence})
    return r if confidence is None else r.model_copy(update={"confidence": confidence})


def curve(items: Sequence[GoldItem], outputs: Sequence[Response | None], conf: Sequence[float | None],
          thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> list[dict[str, Any]]:
    """Risk-coverage table: metrics of the gated outputs at each threshold. ``coverage`` is the
    share of items that are executed; ``gated`` the share turned into a confirmation."""
    from athar.eval.metrics import aggregate, score_all

    rows = []
    for tau in thresholds:
        gated = [apply_gate(o, c, tau) for o, c in zip(outputs, conf, strict=True)]
        m = aggregate(score_all(items, gated))
        n = len(items)
        rows.append({"tau": tau, "coverage": sum(g is not None and g.route == "ACT" for g in gated) / n,
                     "gated": sum(g is not None and CONFIRM in g.clarify_targets for g in gated) / n,
                     **{k: m[k] for k in CURVE_METRICS}})
    return rows


def select_threshold(rows: Sequence[dict[str, Any]], max_unsafe: float) -> float:
    """Pre-registered rule (dev only): the smallest threshold whose unsafe-execution rate is at
    most ``max_unsafe``, so the assistant asks as rarely as possible; if none reaches it, the
    threshold with the lowest unsafe-execution rate (ties: the smaller threshold)."""
    ok = [r["tau"] for r in rows if r["unsafe_execution_rate"] <= max_unsafe]
    if ok:
        return float(min(ok))
    return float(min(rows, key=lambda r: (r["unsafe_execution_rate"], r["tau"]))["tau"])


def fit_threshold(recognizer: Any, dev: Sequence[GoldItem], cfg: dict[str, Any]) -> dict[str, Any]:
    """Choose ``tau`` on dev for a fitted recognizer (``cfg``: the ``gate`` block of
    configs/train.yaml). Returns the threshold, the rule and the dev risk-coverage table."""
    from athar.normalize import norm
    from athar.pipeline import respond_with_intent

    texts = [norm(it.text) for it in dev]
    intents = recognizer.predict(texts)
    outputs = [respond_with_intent(i, it.text, it.ref_datetime, it.timezone, recognizer.name)
               for i, it in zip(intents, dev, strict=True)]
    max_unsafe = cfg.get("max_unsafe_execution_rate", 0.02)
    rows = curve(dev, outputs, confidences(recognizer, texts, intents), cfg.get("thresholds", DEFAULT_THRESHOLDS))
    return {"threshold": select_threshold(rows, max_unsafe), "max_unsafe_execution_rate": max_unsafe,
            "dev_curve": rows}
