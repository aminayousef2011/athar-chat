"""Evaluation layer: item scoring, aggregate metrics, statistics, robustness, harness."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from athar.data.io import load_items
from athar.eval import harness as H
from athar.eval.metrics import aggregate, score_item
from athar.eval.perturb import PERTURBATIONS, perturbed_copies
from athar.eval.stats import (
    aurc,
    brier_score,
    cluster_bootstrap,
    expected_calibration_error,
    holm,
    mcnemar_exact,
    mean_ci,
    paired_permutation_test,
)
from athar.intent.rules import KeywordIntent
from athar.pipeline import respond_with_intent
from athar.schema import GoldItem

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "pilot" / "pilot.jsonl"
DRAFT = ROOT / "data" / "draft" / "athar_draft.jsonl"
FAST = {"seed": 1, "level": 0.95, "resamples": 200, "permutations": 200, "alpha": 0.05,
        "calibration_bins": 10, "oracle": True, "comparisons": [["S1", "oracle"]]}


def gold(**kw):
    base = dict(id="g1", family_id="f1", text="فكرني بكرة الساعة 7 الصبح أشتري عيش",
                ref_datetime="2026-10-07T10:00:00+03:00", intent="create_reminder", route="ACT",
                args={"datetime": "2026-10-08T07:00:00+03:00", "reminder_text": "أشتري عيش"})
    return GoldItem(**{**base, **kw})


def out(text, intent="create_reminder"):
    g = gold()
    return respond_with_intent(intent, text, g.ref_datetime)


# --------------------------------------------------------------------------- item scoring
def test_act_success_and_failure_modes():
    g = gold()
    assert score_item(g, out(g.text)).success
    wrong_time = score_item(g, out("فكرني بكرة الساعة 8 الصبح أشتري عيش"))
    assert (wrong_time.success, wrong_time.executed, wrong_time.unsafe_execution) == (False, True, True)
    assert not score_item(g, out("فكرني بكرة الساعة 7 الصبح أشتري لبن")).success
    assert not score_item(g, out(g.text, "create_alarm")).success  # wrong tool


def test_act_accepts_annotated_alternative_text():
    g = gold(args={"datetime": "2026-10-08T07:00:00+03:00", "reminder_text": "عيش",
                   "reminder_text_alts": ["أشتري عيش"]})
    assert score_item(g, out("فكرني بكرة الساعة 7 الصبح أشتري عيش")).success


def test_same_instant_in_another_offset_is_success():
    g = gold(args={"datetime": "2026-10-08T04:00:00+00:00", "reminder_text": "أشتري عيش"})
    assert score_item(g, out("فكرني بكرة الساعة 7 الصبح أشتري عيش")).success


def test_clarify_needs_exact_target_set_and_intent():
    g = gold(text="فكرني بكرة أشتري عيش", route="CLARIFY", clarify_targets=["time"], args={})
    assert score_item(g, out(g.text)).success
    assert not score_item(g, out("فكرني الأسبوع الجاي أشتري عيش")).success  # date+time != time
    assert not score_item(g, out(g.text, "create_alarm")).success  # same targets, wrong intent


def test_abstain_and_invalid_output():
    g = gold(text="امسح المنبه", intent="unsupported", route="ABSTAIN", args={},
             unsupported_reason="deletion")
    assert score_item(g, out(g.text, "unsupported")).success
    s = score_item(gold(), {"system": "x", "route": "ACT", "intent": "create_alarm", "message": ""})
    assert (s.valid, s.success, s.executed, s.pred_route) == (False, False, False, "invalid")
    assert not score_item(gold(), None).valid


def test_aggregate_safety_and_helpfulness_rates():
    items = [gold(id="a", pair_id="p1"),
             gold(id="b", text="فكرني بكرة أشتري عيش", route="CLARIFY", clarify_targets=["time"],
                  args={}, pair_id="p1"),
             gold(id="c", family_id="f2", text="امسح المنبه", intent="unsupported", route="ABSTAIN",
                  args={}, unsupported_reason="deletion")]
    outputs = [out("فكرني بكرة الساعة 7 أشتري عيش"),  # CLARIFY am_pm on a gold ACT
               out("فكرني بكرة أشتري عيش"),  # correct CLARIFY
               out("فكرني بكرة الساعة 7 الصبح أشتري عيش")]  # ACT on a gold ABSTAIN: unsafe
    m = aggregate([score_item(g, o) for g, o in zip(items, outputs, strict=True)])
    assert m["safe_task_success"] == pytest.approx(1 / 3)
    assert m["unnecessary_clarification_rate"] == 1.0
    assert m["unsafe_execution_rate"] == pytest.approx(1 / 3)
    assert m["unsafe_execution_on_non_act"] == 0.5
    assert m["missed_abstention_rate"] == 1.0
    assert m["pair_both_success"] == 0.0
    assert m["success_by_gold_route"] == {"ACT": 0.0, "CLARIFY": 1.0, "ABSTAIN": 0.0}


# --------------------------------------------------------------------------- statistics
def test_bootstrap_of_a_constant_is_degenerate():
    ci = mean_ci([1.0] * 10, [f"f{i // 2}" for i in range(10)], resamples=200)
    assert (ci.estimate, ci.low, ci.high) == (1.0, 1.0, 1.0)


def test_bootstrap_resamples_whole_families():
    # two families with opposite outcomes: every resample is 0, 0.5 or 1 (never in between)
    values = np.array([1.0] * 5 + [0.0] * 5)
    seen = set()
    cluster_bootstrap(lambda idx: seen.add(round(float(values[idx].mean()), 3)) or 0.0,
                      ["a"] * 5 + ["b"] * 5, resamples=300)
    assert seen <= {0.0, 0.5, 1.0}


def test_permutation_test_basic_properties():
    clusters = [f"f{i}" for i in range(40)]
    same = paired_permutation_test([1, 0] * 20, [1, 0] * 20, clusters, permutations=500, resamples=100)
    assert (same.difference, same.p_value) == (0.0, 1.0)
    strong = paired_permutation_test([1.0] * 40, [0.0] * 40, clusters, permutations=2000, resamples=100)
    assert strong.difference == 1.0 and strong.p_value < 0.001
    assert strong.ci.low == strong.ci.high == 1.0


def test_permutation_test_respects_families():
    # the whole difference sits in a single family: item-level tests would call it significant,
    # the family-level test cannot (only 2 sign assignments)
    a, b = [1.0] * 10 + [0.0] * 30, [0.0] * 40
    fam = ["big"] * 10 + [f"f{i}" for i in range(30)]
    t = paired_permutation_test(a, b, fam, permutations=2000, resamples=100)
    assert t.p_value > 0.3
    assert mcnemar_exact([bool(x) for x in a], [bool(x) for x in b])["p_value"] < 0.01


def test_mcnemar_exact_known_value():
    r = mcnemar_exact([True] * 8 + [False] * 2, [False] * 8 + [False] * 2)
    assert (r["only_a"], r["only_b"]) == (8, 0)
    assert r["p_value"] == pytest.approx(2 * 0.5**8)


def test_holm_known_values():
    adj = holm({"h1": 0.01, "h2": 0.04, "h3": 0.03})
    assert adj == pytest.approx({"h1": 0.03, "h3": 0.06, "h2": 0.06})


def test_calibration_metrics():
    assert expected_calibration_error([1.0, 1.0, 0.0], [True, True, False]) == 0.0
    # all predictions at 0.8 confidence, 50% correct -> ECE 0.3
    assert expected_calibration_error([0.8] * 4, [True, False, True, False]) == pytest.approx(0.3)
    assert brier_score(np.eye(3), [0, 1, 2]) == 0.0
    assert brier_score(np.array([[0.5, 0.5, 0.0]]), [0]) == pytest.approx(0.5)
    assert aurc([0.9, 0.8], [True, True]) == 0.0
    assert aurc([0.9, 0.1], [False, True]) == pytest.approx((1 + 0.5) / 2)


# --------------------------------------------------------------------------- robustness
def test_perturbations_preserve_labels_and_change_text():
    items = load_items(DRAFT)
    copies = perturbed_copies(items)
    assert set(copies) == set(PERTURBATIONS)
    for kind, cs in copies.items():
        assert cs, kind
        for c in cs[:20]:
            base = next(it for it in items if it.id == c.id.rsplit("--", 1)[0])
            assert c.text != base.text
            assert (c.intent, c.route, c.clarify_targets, c.args) == (base.intent, base.route,
                                                                     base.clarify_targets, base.args)


@pytest.mark.parametrize("kind", ["taa_marbuta", "hamza", "alef_maqsura", "digits", "diacritics",
                                  "elongation", "punctuation", "politeness"])
def test_shared_core_is_invariant_to_orthographic_and_filler_perturbations(kind):
    """With the gold intent, the shared core must give the same answer on perturbed text."""
    items = load_items(DRAFT)
    for c in perturbed_copies(items, [kind])[kind]:
        r = respond_with_intent(c.intent, c.text, c.ref_datetime)
        assert score_item(c, r).success, (kind, c.id, c.text)


# --------------------------------------------------------------------------- harness
def test_harness_on_the_pilot():
    items = load_items(PILOT)
    rep = H.evaluate({"S1": KeywordIntent()}, items, {**FAST, "perturbations": ["digits", "politeness"]})
    assert rep["systems"]["oracle"]["metrics"]["safe_task_success"] == 1.0
    s1 = rep["systems"]["S1"]
    assert s1["metrics"]["safe_task_success"] == 1.0
    assert s1["ci"]["safe_task_success"]["low"] == 1.0
    assert s1["calibration"] is None  # rules have no probabilities
    assert s1["robustness"]["digits"]["invariance_rate"] == 1.0
    assert set(rep["comparisons"]) == {"S1 vs oracle"}
    assert rep["comparisons"]["S1 vs oracle"]["p_holm"] == 1.0


def test_eval_config_preregisters_three_primary_comparisons():
    cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    assert cfg["comparisons"] == [["S2", "S1"], ["S3", "S1"], ["S3", "S2"]]
    assert cfg["resamples"] >= 10_000 and cfg["permutations"] >= 10_000
    assert set(cfg["perturbations"]) == set(PERTURBATIONS)


def test_test_split_needs_unlock_flag(tmp_path, capsys):
    assert H.main(["--manifest", str(tmp_path / "m.json"), "--split", "test"]) == 1
    assert "opened once" in capsys.readouterr().out


def test_results_json_is_serialisable():
    items = load_items(PILOT)[:4]
    rep = H.evaluate({"S1": KeywordIntent()}, items, {**FAST, "perturbations": ["digits"]})
    json.dumps(rep, default=str)


def test_power_simulation_sanity():
    from athar.eval.stats import simulate_power

    assert simulate_power(30, 2, 1.0, 0.0, sims=20, permutations=200) == 1.0  # huge effect
    assert simulate_power(30, 2, 0.7, 0.7, sims=60, permutations=200) <= 0.15  # no effect ~ alpha


def test_locked_test_run_is_one_shot(split_dir_with_test, monkeypatch, capsys):
    from athar.intent import train as T

    d = split_dir_with_test
    models = d / "models"
    T.fit_and_freeze(d / "manifest.json", d / "train.yaml", models, log=lambda _: None)
    cfg = d / "eval.yaml"
    cfg.write_text(yaml.safe_dump({**FAST, "comparisons": [["S2", "S1"], ["S3", "S1"], ["S3", "S2"]],
                                   "perturbations": ["digits"]}), encoding="utf-8")
    monkeypatch.chdir(d)
    base = ["--manifest", str(d / "manifest.json"), "--models", str(models), "--config", str(cfg)]

    assert H.main([*base, "--split", "dev"]) == 0
    assert (d / "runs" / "eval-dev.json").exists()
    assert H.main([*base, "--split", "test", "--unlock-test"]) == 0
    result = json.loads((d / "results" / "test_results.json").read_text(encoding="utf-8"))
    assert result["split"] == "test" and result["n_items"] == 12
    assert result["freeze_sha256"] == T.read_freeze(models)["freeze_sha256"]
    assert set(result["comparisons"]) == {"S2 vs S1", "S3 vs S1", "S3 vs S2"}
    assert all("p_holm" in v for v in result["comparisons"].values())
    # a second opening of the test split is refused
    assert H.main([*base, "--split", "test", "--unlock-test"]) == 1
    assert "already been evaluated" in capsys.readouterr().out


def test_test_run_refuses_a_changed_model(split_dir_with_test, monkeypatch, capsys):
    from athar.intent import train as T

    d = split_dir_with_test
    models = d / "models"
    T.fit_and_freeze(d / "manifest.json", d / "train.yaml", models, log=lambda _: None)
    (models / "S3.joblib").write_bytes((models / "S3.joblib").read_bytes() + b"\0")
    monkeypatch.chdir(d)
    rc = H.main(["--manifest", str(d / "manifest.json"), "--models", str(models), "--config",
                 str(ROOT / "configs" / "eval.yaml"), "--split", "test", "--unlock-test"])
    assert rc == 1 and "refusing to open the test split" in capsys.readouterr().out
    assert not (d / "results").exists()


def test_vectorised_metrics_equal_the_reference_definitions():
    from athar.eval.metrics import ScoreArrays, aggregate, score_all

    items = load_items(DRAFT)
    rng = np.random.default_rng(0)
    intents = ["create_reminder", "create_alarm", "unsupported"]
    noisy = [it.intent if rng.random() < 0.8 else intents[rng.integers(3)] for it in items]
    outputs = [respond_with_intent(i, it.text, it.ref_datetime) for i, it in zip(noisy, items, strict=True)]
    outputs[3] = None  # one invalid output
    scores = score_all(items, outputs)
    arrays = ScoreArrays(scores)
    for _ in range(5):
        idx = rng.integers(0, len(items), size=len(items))
        ref = aggregate([scores[i] for i in idx])
        for name in ScoreArrays.METRICS:
            assert arrays.metric(name)(idx) == pytest.approx(ref[name], abs=1e-12), name


def test_oracle_is_invariant_under_every_orthographic_perturbation():
    """Regression: invariance compares the canonical reminder text, not the user's spelling."""
    items = load_items(DRAFT)[:60]
    kinds = ["taa_marbuta", "hamza", "alef_maqsura", "digits", "diacritics", "elongation", "punctuation"]
    rep = H.evaluate({}, items, {**FAST, "perturbations": kinds, "comparisons": []})
    rob = rep["systems"]["oracle"]["robustness"]
    assert {k: v["invariance_rate"] for k, v in rob.items()} == dict.fromkeys(rob, 1.0)


def test_subgroups_cover_domain_code_switching_and_tool():
    items = load_items(DRAFT)[:80]
    rep = H.evaluate({}, items, {**FAST, "perturbations": ["digits"], "comparisons": []})
    sub = rep["systems"]["oracle"]["subgroups"]
    assert set(sub) == {"domain", "code_switching", "tool"}
    assert sum(v["n"] for v in sub["domain"].values()) == len(items)
    assert all(v["safe_task_success"]["estimate"] == 1.0 for v in sub["domain"].values())


def test_macro_f1_leaves_out_labels_absent_from_the_sample():
    from athar.eval.metrics import _codes, macro_f1, macro_f1_codes
    from athar.schema import INTENTS

    gold, pred = ["create_reminder", "create_alarm"], ["create_reminder", "create_alarm"]
    assert macro_f1(gold, pred, INTENTS) == 1.0  # 'unsupported' absent: not a zero
    assert macro_f1_codes(_codes(gold, INTENTS), _codes(pred, INTENTS), 3) == 1.0
    pred2 = ["create_reminder", "unsupported"]  # predicted but absent from gold: counts
    assert macro_f1(gold, pred2, INTENTS) == pytest.approx((1 + 0 + 0) / 3)
    assert macro_f1_codes(_codes(gold, INTENTS), _codes(pred2, INTENTS), 3) == pytest.approx(1 / 3)
