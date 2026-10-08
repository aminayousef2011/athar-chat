"""Confidence gate (athar.gate): it only ever removes executions, and its threshold is frozen on dev."""

import json

import pytest
import yaml
from pydantic import ValidationError

from athar import gate as G
from athar.eval import harness as H
from athar.eval.metrics import score_item
from athar.pipeline import Pipeline, respond_with_intent
from athar.schema import INTENTS, GoldItem, Response

FAST = {"seed": 1, "level": 0.95, "resamples": 200, "permutations": 200, "alpha": 0.05,
        "calibration_bins": 10, "oracle": True, "comparisons": [], "perturbations": []}
TEXT = "فكرني بكرة الساعة 7 الصبح أشتري عيش"
REF = "2026-10-07T10:00:00+03:00"


def gold(**kw):
    base = dict(id="g1", family_id="f1", text=TEXT, ref_datetime=REF, intent="create_reminder", route="ACT",
                args={"datetime": "2026-10-08T07:00:00+03:00", "reminder_text": "أشتري عيش"})
    return GoldItem(**{**base, **kw})


class Fixed:
    """A probabilistic recognizer with chosen labels and confidences."""

    def __init__(self, labels, conf, name="P"):
        self.name, self.labels, self.conf = name, labels, conf

    def predict(self, texts):
        return list(self.labels[:len(texts)])

    def predict_proba(self, texts, labels):
        rows = []
        for lab, c in zip(self.labels, self.conf, strict=True):
            rest = (1 - c) / (len(INTENTS) - 1)
            p = {i: (c if i == lab else rest) for i in INTENTS}
            rows.append([p[x] for x in labels])
        return rows


def act():
    return respond_with_intent("create_reminder", TEXT, gold().ref_datetime, system="P")


def test_gate_turns_a_low_confidence_execution_into_a_confirmation():
    r = G.apply_gate(act(), 0.55, 0.7)
    assert (r.route, r.tool_call, r.clarify_targets, r.confidence) == ("CLARIFY", None, ["confirm"], 0.55)
    assert r.message.startswith("للتأكيد: أفكرك بـ«أشتري عيش»") and r.message.endswith("صح كده؟")
    assert r.extracted_datetime is not None  # what would have run is still visible
    assert not score_item(gold(), r).executed


def test_gate_keeps_confident_and_non_executing_answers():
    assert G.apply_gate(act(), 0.9, 0.7).route == "ACT"
    assert G.apply_gate(act(), 0.9, 0.7).confidence == 0.9
    assert G.apply_gate(act(), 0.1, None).route == "ACT"       # no threshold: gate off
    assert G.apply_gate(act(), None, 0.9).route == "ACT"       # no probabilities (S1): never gated
    abstain = respond_with_intent("unsupported", "امسح المنبه", gold().ref_datetime)
    assert G.apply_gate(abstain, 0.1, 0.9).route == "ABSTAIN"
    clarify = respond_with_intent("create_alarm", "صحيني بكرة", gold().ref_datetime)
    assert G.apply_gate(clarify, 0.1, 0.9).clarify_targets == clarify.clarify_targets
    assert G.apply_gate(None, 0.1, 0.9) is None


def test_alarm_confirmation_and_schema_rules():
    alarm = respond_with_intent("create_alarm", "صحيني بكرة الساعة 6 الصبح", gold().ref_datetime)
    assert G.apply_gate(alarm, 0.4, 0.5).message.startswith("للتأكيد: أظبط منبه يرن")
    with pytest.raises(ValidationError, match="on its own"):
        Response(system="x", route="CLARIFY", intent="create_alarm", clarify_targets=["time", "confirm"], message="?")
    with pytest.raises(ValidationError):  # annotators never write `confirm`
        gold(route="CLARIFY", args={}, clarify_targets=["confirm"])


def test_curve_only_removes_executions_and_rule_picks_smallest_safe_threshold():
    items = [gold(id="a", family_id="a"),
             gold(id="b", family_id="b", intent="unsupported", route="ABSTAIN", args={},
                  unsupported_reason="other_tool")]
    outputs = [act(), act()]                       # the second is an unsafe execution
    rows = G.curve(items, outputs, [0.95, 0.6], [0.0, 0.5, 0.7, 0.99])
    cov = [r["coverage"] for r in rows]
    unsafe = [r["unsafe_execution_rate"] for r in rows]
    assert cov == sorted(cov, reverse=True) and unsafe == sorted(unsafe, reverse=True)
    assert [r["unsafe_execution_rate"] for r in rows] == [0.5, 0.5, 0.0, 0.0]
    assert G.select_threshold(rows, 0.0) == 0.7     # smallest threshold reaching the target
    assert G.select_threshold(rows, 0.5) == 0.0     # target met with the gate off
    floor = [{"tau": t, "unsafe_execution_rate": u} for t, u in ((0.0, 0.3), (0.5, 0.2), (0.9, 0.2))]
    assert G.select_threshold(floor, 0.0) == 0.5    # unreachable: lowest rate, then smaller


def test_pipeline_applies_the_gate_and_reports_confidence():
    ref = gold().ref_datetime
    low = Pipeline(Fixed(["create_reminder"], [0.5]), gate_threshold=0.8).run(TEXT, ref)
    assert (low.route, low.confidence) == ("CLARIFY", 0.5)
    high = Pipeline(Fixed(["create_reminder"], [0.9]), gate_threshold=0.8).run(TEXT, ref)
    assert (high.route, high.confidence) == ("ACT", 0.9)


def test_harness_reports_gate_beside_the_ungated_metrics():
    items = [gold(id="a", family_id="a"),
             gold(id="b", family_id="b", intent="unsupported", route="ABSTAIN", args={},
                  unsupported_reason="other_tool")]
    rec = Fixed(["create_reminder", "create_reminder"], [0.95, 0.6])
    rep = H.evaluate({"P": rec}, items, FAST, {"P": 0.7})
    p = rep["systems"]["P"]
    assert p["metrics"]["unsafe_execution_rate"] == 0.5          # pre-registered: ungated
    g = p["gate"]
    assert g["frozen_threshold"] == 0.7 and g["rule_threshold_on_this_split"] == 0.7
    assert g["at_frozen_threshold"]["metrics"]["unsafe_execution_rate"] == 0.0
    assert g["confidence"] == [0.95, 0.6]
    assert "gate" not in rep["systems"]["oracle"]
    json.dumps(rep, default=str)


def test_fit_freezes_a_threshold_and_test_run_applies_it(split_dir_with_test, monkeypatch):
    from athar.intent import train as T

    d = split_dir_with_test
    models = d / "models"
    f = T.fit_and_freeze(d / "manifest.json", d / "train.yaml", models, log=lambda _: None)
    for name in ("S2", "S3"):
        gate = f["systems"][name]["gate"]
        assert gate["threshold"] in G.DEFAULT_THRESHOLDS and len(gate["dev_curve"]) == len(G.DEFAULT_THRESHOLDS)
    assert "gate" not in f["systems"]["S1"]
    assert "src/athar/gate.py" in f["frozen_sources"]
    cfg = d / "eval.yaml"
    cfg.write_text(yaml.safe_dump({**FAST, "perturbations": ["digits"]}), encoding="utf-8")
    monkeypatch.chdir(d)
    assert H.main(["--manifest", str(d / "manifest.json"), "--models", str(models), "--config", str(cfg),
                   "--split", "dev"]) == 0
    result = json.loads((d / "runs" / "eval-dev.json").read_text(encoding="utf-8"))
    for name in ("S2", "S3"):
        assert result["systems"][name]["gate"]["frozen_threshold"] == f["systems"][name]["gate"]["threshold"]
