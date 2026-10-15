"""Evaluation harness: replay items through the runtime path, score, add uncertainty.

    athar-eval --manifest data/splits/manifest.json --models models --split dev
    athar-eval --manifest data/splits/manifest.json --models models --split test --unlock-test

Dev runs are unrestricted. The test run is the pre-registered one-shot evaluation
(docs/evaluation_protocol.md): it requires a clean freeze (models, shared core, this harness,
the protocol, the eval config and the task contract unchanged since ``athar-train fit``),
writes ``results/test_results.json`` and refuses to run again once that file exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC
from pathlib import Path
from typing import Any, Protocol, cast

import yaml

from athar.data.io import canonical_json, sha256_text
from athar.data.split import load_split
from athar.eval.metrics import ScoreArrays, aggregate, score_all
from athar.eval.perturb import perturbed_copies
from athar.eval.stats import (
    aurc,
    brier_score,
    cluster_bootstrap,
    expected_calibration_error,
    holm,
    mcnemar_exact,
    paired_permutation_test,
)
from athar.extract import canonical_text
from athar.gate import DEFAULT_THRESHOLDS, apply_gate, confidences, curve, select_threshold
from athar.normalize import norm
from athar.pipeline import FixedIntent, IntentRecognizer, respond_with_intent
from athar.schema import INTENTS, GoldItem, Response

#: metrics reported with a family-bootstrap CI
CI_METRICS = ScoreArrays.METRICS


class ProbabilisticRecognizer(Protocol):
    def predict_proba(self, normalized_texts: Sequence[str], labels: Sequence[str]) -> list[list[float]]: ...


class Oracle:
    """Gold intent for whatever items it is run on: the extraction/policy ceiling."""

    name = "oracle"


class ResponseSystem(Protocol):
    """A system that answers whole items itself instead of predicting an intent for the shared
    pipeline (S8, ``athar.agent``). None = no valid response, which scores as a failure."""

    name: str

    def respond_items(self, items: Sequence[GoldItem]) -> list[Response | None]: ...


System = IntentRecognizer | Oracle | ResponseSystem


def run(recognizer: System, items: Sequence[GoldItem]) -> list[Response | None]:
    """Runtime path for every item. A recognizer answer outside the three intent labels (an
    LLM that produced no parseable label) yields no response, which scores as a failure."""
    respond = getattr(recognizer, "respond_items", None)
    if respond is not None:
        outputs: list[Response | None] = list(respond(items))
        if len(outputs) != len(items):
            raise ValueError(f"{recognizer.name} returned {len(outputs)} responses for {len(items)} items")
        return outputs
    if isinstance(recognizer, Oracle):
        recognizer = FixedIntent([it.intent for it in items])
    assert not isinstance(recognizer, Oracle)
    intents = cast(IntentRecognizer, recognizer).predict([norm(it.text) for it in items])
    if len(intents) != len(items):
        raise ValueError(f"{recognizer.name} returned {len(intents)} labels for {len(items)} items")
    return [respond_with_intent(i, it.text, it.ref_datetime, it.timezone, recognizer.name)
            if i in INTENTS else None for i, it in zip(intents, items, strict=True)]


def _calibration(recognizer: Any, items: Sequence[GoldItem], bins: int) -> dict[str, float] | None:
    if not hasattr(recognizer, "predict_proba"):
        return None
    import numpy as np

    proba = np.asarray(recognizer.predict_proba([norm(it.text) for it in items], INTENTS))
    gold = [INTENTS.index(it.intent) for it in items]
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == np.asarray(gold)
    return {"ece": expected_calibration_error(conf, correct, bins), "brier": brier_score(proba, gold),
            "aurc": aurc(conf, correct)}


SUBGROUPS: dict[str, Any] = {
    "domain": lambda it: it.domain,  # project_management (main target users) vs general
    "code_switching": lambda it: "code-switched" if it.code_switched else "arabic-only",
    "tool": lambda it: it.intent,
}


def _subgroups(items: Sequence[GoldItem], scores: Sequence[Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """STS and intent macro-F1 with family-bootstrap CIs per subgroup (protocol section 6)."""
    out: dict[str, Any] = {}
    for group, key in SUBGROUPS.items():
        members: dict[str, list[int]] = {}
        for i, it in enumerate(items):
            members.setdefault(str(key(it)), []).append(i)
        out[group] = {}
        for value, idx in sorted(members.items()):
            sub = [scores[i] for i in idx]
            arrays, clusters = ScoreArrays(sub), [s.family_id for s in sub]
            out[group][value] = {"n": len(sub), **{
                m: cluster_bootstrap(arrays.metric(m), clusters, resamples=cfg["resamples"],
                                     level=cfg["level"], seed=cfg["seed"]).as_dict()
                for m in ("safe_task_success", "intent_macro_f1")}}
    return out


def _with_ci(scores: Sequence[Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    clusters = [s.family_id for s in scores]
    arrays = ScoreArrays(scores)
    return aggregate(scores), {m: cluster_bootstrap(arrays.metric(m), clusters, resamples=cfg["resamples"],
                                                    level=cfg["level"], seed=cfg["seed"]).as_dict()
                               for m in CI_METRICS}


def _gate(recognizer: System, items: Sequence[GoldItem], outputs: Sequence[Response | None],
          cfg: dict[str, Any], tau: float | None) -> dict[str, Any] | None:
    """Confidence-gate analysis (protocol 6.4), secondary to the ungated pre-registered metrics:
    the risk-coverage table, the threshold the dev rule would pick on *this* split (meaningful on
    dev only), and full metrics at the threshold frozen on dev, when one is given."""
    if not hasattr(recognizer, "predict_proba"):
        return None
    g = cfg.get("gate", {})
    intents = [o.intent if o is not None else "invalid" for o in outputs]
    conf = confidences(recognizer, [norm(it.text) for it in items], intents)
    rows = curve(items, outputs, conf, g.get("thresholds", DEFAULT_THRESHOLDS))
    out: dict[str, Any] = {"curve": rows, "max_unsafe_execution_rate": g.get("max_unsafe_execution_rate", 0.02),
                           "rule_threshold_on_this_split": select_threshold(rows, g.get("max_unsafe_execution_rate", 0.02)),
                           "confidence": conf}
    if tau is not None:
        gated = [apply_gate(o, c, tau) for o, c in zip(outputs, conf, strict=True)]
        metrics, ci = _with_ci(score_all(items, gated), cfg)
        out["frozen_threshold"] = tau
        out["at_frozen_threshold"] = {"metrics": metrics, "ci": ci}
    return out


def evaluate_system(recognizer: System, items: Sequence[GoldItem], cfg: dict[str, Any],
                    gate_threshold: float | None = None) -> dict[str, Any]:
    outputs = run(recognizer, items)
    scores = score_all(items, outputs)
    metrics, ci = _with_ci(scores, cfg)
    report: dict[str, Any] = {"metrics": metrics, "ci": ci}
    report["calibration"] = _calibration(recognizer, items, cfg.get("calibration_bins", 10))
    report["subgroups"] = _subgroups(items, scores, cfg)

    robustness = {}
    original = {it.id: o for it, o in zip(items, outputs, strict=True)}
    for kind, copies in perturbed_copies(items, cfg.get("perturbations")).items():
        if not copies:
            continue
        p_out = run(recognizer, copies)
        p_scores = score_all(copies, p_out)
        base_ids = [c.id.rsplit("--", 1)[0] for c in copies]
        same = [_core(o) == _core(original[b]) for o, b in zip(p_out, base_ids, strict=True)]
        robustness[kind] = {"n": len(copies), "safe_task_success": aggregate(p_scores)["safe_task_success"],
                            "invariance_rate": sum(same) / len(same)}
    report["robustness"] = robustness
    if (gate := _gate(recognizer, items, outputs, cfg, gate_threshold)) is not None:
        report["gate"] = gate
    report["items"] = [s.__dict__ for s in scores]
    return report


def _core(r: Response | None) -> tuple[Any, ...]:
    """What must not change under a label-preserving perturbation. The reminder text is
    compared in canonical form: echoing the user's own spelling (``أشتري`` vs ``اشتري``) is
    correct behaviour, not a change of decision."""
    if r is None:
        return ("invalid",)
    call = r.tool_call
    return (r.route, r.intent, tuple(r.clarify_targets),
            None if call is None else (call.name, call.datetime.astimezone(UTC).isoformat(),
                                       canonical_text(call.reminder_text) if call.reminder_text else None))


def compare(reports: dict[str, dict[str, Any]], comparisons: Sequence[Sequence[str]],
            cfg: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered paired comparisons on Safe Task Success, Holm-adjusted."""
    tests: dict[str, Any] = {}
    for a, b in comparisons:
        sa = [it["success"] for it in reports[a]["items"]]
        sb = [it["success"] for it in reports[b]["items"]]
        clusters = [it["family_id"] for it in reports[a]["items"]]
        t = paired_permutation_test([float(x) for x in sa], [float(x) for x in sb], clusters,
                                    permutations=cfg["permutations"], resamples=cfg["resamples"],
                                    level=cfg["level"], seed=cfg["seed"])
        tests[f"{a} vs {b}"] = {**t.as_dict(), "mcnemar": mcnemar_exact(sa, sb)}
    adjusted = holm({k: v["p_value"] for k, v in tests.items()})
    for k, v in tests.items():
        v["p_holm"] = adjusted[k]
        v["significant"] = adjusted[k] < cfg["alpha"]
    return tests


SEED_SUMMARY_METRICS = ("safe_task_success", "intent_macro_f1", "route_macro_f1",
                        "unsafe_execution_rate", "unnecessary_clarification_rate")


def seed_summaries(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Mean and SD over seeds for multi-seed systems named ``<system>-s<seed>`` (e.g. S6)."""
    import re

    import numpy as np

    groups: dict[str, list[str]] = {}
    for name in reports:
        m = re.fullmatch(r"(.+)-s\d+", name)
        if m:
            groups.setdefault(m.group(1), []).append(name)
    out: dict[str, Any] = {}
    for base, names in groups.items():
        summary: dict[str, Any] = {"seeds": sorted(names)}
        for metric in SEED_SUMMARY_METRICS:
            vals = [float(reports[n]["metrics"][metric] or 0.0) for n in names]
            summary[metric] = {"mean": float(np.mean(vals)),
                               "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                               "min": min(vals), "max": max(vals)}
        out[base] = summary
    return out


def evaluate(recognizers: Mapping[str, IntentRecognizer | ResponseSystem], items: Sequence[GoldItem],
             cfg: dict[str, Any], gate_thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """``gate_thresholds``: per-system confidence-gate thresholds frozen on dev (athar.gate)."""
    systems: dict[str, System] = dict(recognizers)
    if cfg.get("oracle", True):
        systems["oracle"] = Oracle()
    taus = gate_thresholds or {}
    reports = {name: evaluate_system(r, items, cfg, taus.get(name)) for name, r in systems.items()}
    comparisons = [c for c in cfg.get("comparisons", []) if c[0] in reports and c[1] in reports]
    return {"config": cfg, "config_sha256": sha256_text(canonical_json(cfg)), "n_items": len(items),
            "systems": reports, "seed_summaries": seed_summaries(reports),
            "comparisons": compare(reports, comparisons, cfg)}


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    from athar.intent.train import (
        FreezeError,
        frozen_systems,
        load_recognizer,
        read_freeze,
        verify_freeze,
    )

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="athar-eval", description="ATHAR Chat evaluation harness")
    p.add_argument("--manifest", required=True)
    p.add_argument("--models", default="models")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--systems", nargs="+", default=["S1", "S2", "S3"],
                   help="system names, or 'all' for every frozen system (S5, S6-s<seed> need the neural extra)")
    p.add_argument("--unlock-test", action="store_true", help="the one-shot locked test run")
    p.add_argument("--out")
    a = p.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    try:
        out = Path(a.out) if a.out else Path("runs") / f"eval-{a.split}.json"
        if a.split == "test":
            if not a.unlock_test:
                print("the test split is opened once, with --unlock-test (docs/evaluation_protocol.md)")
                return 1
            problems = verify_freeze(a.models)
            if problems:
                print("refusing to open the test split: " + "; ".join(problems))
                return 1
            out = Path("results/test_results.json")
            if out.exists():
                print(f"{out} exists: the locked test set has already been evaluated")
                return 1
        items = load_split(a.manifest, a.split, unlock_test=a.split == "test")
        names = frozen_systems(a.models) if a.systems == ["all"] else a.systems
        strict = a.split == "test"
        if not strict and (changed := [p for p in verify_freeze(a.models) if p.startswith("source ")]):
            print(f"note: {len(changed)} frozen source(s) changed since the freeze; fine for a dev run, "
                  f"the test run requires a clean freeze")
        recognizers: dict[str, IntentRecognizer] = {s: load_recognizer(s, a.models, strict=strict)
                                                    for s in names}
        frozen = read_freeze(a.models)["systems"]
        taus = {s: frozen[s]["gate"]["threshold"] for s in names if "gate" in frozen.get(s, {})}
        result = evaluate(recognizers, items, cfg, taus)
        result["freeze_sha256"] = read_freeze(a.models)["freeze_sha256"]
        result["split"] = a.split
    except (FreezeError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
                   encoding="utf-8", newline="\n")
    for name, rep in result["systems"].items():
        ci = rep["ci"]["safe_task_success"]
        print(f"{name:7} STS {ci['estimate']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]  "
              f"intent F1 {rep['metrics']['intent_macro_f1']:.3f}  "
              f"unsafe {rep['metrics']['unsafe_execution_rate']:.3f}")
        if "at_frozen_threshold" in rep.get("gate", {}):
            g = rep["gate"]["at_frozen_threshold"]["metrics"]
            print(f"{'':7} gated (tau={rep['gate']['frozen_threshold']}): STS {g['safe_task_success']:.3f}  "
                  f"unsafe {g['unsafe_execution_rate']:.3f}  unnecessary clarification "
                  f"{g['unnecessary_clarification_rate'] or 0:.3f}")
    for base, summ in result["seed_summaries"].items():
        sts = summ["safe_task_success"]
        print(f"{base:7} STS {sts['mean']:.3f} +/- {sts['sd']:.3f} over {len(summ['seeds'])} seeds")
    for k, v in result["comparisons"].items():
        print(f"{k:12} diff {v['difference']:+.3f}  p={v['p_value']:.4f}  p_holm={v['p_holm']:.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
