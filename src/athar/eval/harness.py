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
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

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
from athar.normalize import norm
from athar.pipeline import FixedIntent, IntentRecognizer, Pipeline
from athar.schema import INTENTS, GoldItem, Response

#: metrics reported with a family-bootstrap CI
CI_METRICS = ScoreArrays.METRICS


class ProbabilisticRecognizer(Protocol):
    def predict_proba(self, normalized_texts: Sequence[str], labels: Sequence[str]) -> list[list[float]]: ...


class Oracle:
    """Gold intent for whatever items it is run on: the extraction/policy ceiling."""

    name = "oracle"


def run(recognizer: IntentRecognizer | Oracle, items: Sequence[GoldItem]) -> list[Response]:
    if isinstance(recognizer, Oracle):
        recognizer = FixedIntent([it.intent for it in items])
    return Pipeline(recognizer).run_batch([it.text for it in items], [it.ref_datetime for it in items],
                                          [it.timezone for it in items])


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


def evaluate_system(recognizer: IntentRecognizer | Oracle, items: Sequence[GoldItem], cfg: dict[str, Any]) -> dict[str, Any]:
    outputs = run(recognizer, items)
    scores = score_all(items, outputs)
    clusters = [s.family_id for s in scores]
    seed, resamples, level = cfg["seed"], cfg["resamples"], cfg["level"]
    report: dict[str, Any] = {"metrics": aggregate(scores)}
    arrays = ScoreArrays(scores)
    report["ci"] = {m: cluster_bootstrap(arrays.metric(m), clusters, resamples=resamples,
                                         level=level, seed=seed).as_dict() for m in CI_METRICS}
    report["calibration"] = _calibration(recognizer, items, cfg.get("calibration_bins", 10))

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
    report["items"] = [s.__dict__ for s in scores]
    return report


def _core(r: Response) -> tuple[Any, ...]:
    call = r.tool_call
    return (r.route, r.intent, tuple(r.clarify_targets),
            None if call is None else (call.name, call.datetime.isoformat(), call.reminder_text))


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


def evaluate(recognizers: dict[str, IntentRecognizer], items: Sequence[GoldItem], cfg: dict[str, Any]) -> dict[str, Any]:
    systems: dict[str, IntentRecognizer | Oracle] = dict(recognizers)
    if cfg.get("oracle", True):
        systems["oracle"] = Oracle()
    reports = {name: evaluate_system(r, items, cfg) for name, r in systems.items()}
    comparisons = [c for c in cfg.get("comparisons", []) if c[0] in reports and c[1] in reports]
    return {"config": cfg, "config_sha256": sha256_text(canonical_json(cfg)), "n_items": len(items),
            "systems": reports, "comparisons": compare(reports, comparisons, cfg)}


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    from athar.intent.train import FreezeError, load_recognizer, read_freeze, verify_freeze

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="athar-eval", description="ATHAR Chat evaluation harness")
    p.add_argument("--manifest", required=True)
    p.add_argument("--models", default="models")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--systems", nargs="+", default=["S1", "S2", "S3"])
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
        recognizers: dict[str, IntentRecognizer] = {s: load_recognizer(s, a.models) for s in a.systems}
        result = evaluate(recognizers, items, cfg)
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
    for k, v in result["comparisons"].items():
        print(f"{k:12} diff {v['difference']:+.3f}  p={v['p_value']:.4f}  p_holm={v['p_holm']:.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
