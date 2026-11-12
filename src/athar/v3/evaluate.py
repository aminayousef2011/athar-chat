"""Gate G5 of protocol v3: the one scoring run (docs/protocol_v3.md, sections 3 and 5).

    python -m athar.v3.evaluate --manifest data/v3/manifest.json

Runs only when every v3 freeze verifies, the locked gold set holds exactly the frozen v3 items
(same ids, texts and sending moments), and ``results/v3/evaluation.json`` does not exist yet. It
scores the committed predictions through core v1.1 and applies the registered analysis without
any choice left open:

- **Primary**, Holm over H-a and H-b: S3-human vs S3-drafted, and S6-human vs S6-drafted on
  per-item means over the five seeds. Paired approximate randomization swapping whole scenario
  families. Confirmatory only with at least 150 confirmatory items from at least 8 writers.
- **Secondary**, unadjusted: H-c, ABSTAIN success of S6-human vs S6-drafted on gold-ABSTAIN items.
- **Estimation**, no test: S3-human and S6-human minus S1 v1.1, with the fixed reading rule
  ("within 0.10 of the rules" only when the interval's lower bound is above -0.10).
- **Descriptive**: the size-matched control S3-contributors-only (amendment A4), every system's
  success by gold route, writer-clustered tests, exact McNemar for the single-seed comparison and
  leave-one-writer-out.

The GPU systems of protocol section 6 (amendment A5) are scored only if their predictions were
frozen before G4; a system missing at G3 is not scored (section 3).

This module is new code (amendment A6); none of the frozen v3 sources changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from athar.data.io import sha256_file
from athar.data.split import load_split, read_manifest
from athar.eval.metrics import ItemScore, aggregate, score_all
from athar.eval.stats import cluster_bootstrap, holm, mcnemar_exact, paired_permutation_test
from athar.schema import INTENTS, GoldItem
from athar.v2.core import respond_with_intent
from athar.v2.natural import writer_of
from athar.v2.rules import KeywordIntentV11
from athar.v3 import amend
from athar.v3 import data as D
from athar.v3 import systems as S

REPO = D.REPO
OUT = REPO / "results" / "v3" / "evaluation.json"
MAIN = REPO / "data" / "splits" / "manifest.json"
MIN_ITEMS, MIN_WRITERS = 150, 8
SEEDS = (0, 1, 2, 3, 4)
MARGIN = 0.10
HYPOTHESES = {"H-a S3-human vs S3-drafted": ("S3-human", "S3-drafted"),
              "H-b S6-human vs S6-drafted (seed means)": ("S6-human (seed mean)", "S6-drafted (seed mean)")}


class EvaluationError(RuntimeError):
    """The scoring run's preconditions do not hold; nothing is written."""


def _cfg() -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load((REPO / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    return cfg


# --------------------------------------------------------------------------- scoring (pure)
def respond(intents: dict[str, str], items: Sequence[GoldItem], name: str) -> list[ItemScore]:
    """Score frozen intent predictions through core v1.1; an id with no valid intent fails."""
    outputs = [respond_with_intent(intents[it.id], it.text, it.ref_datetime, it.timezone, name)
               if intents.get(it.id) in INTENTS else None for it in items]
    return score_all(items, outputs)


def _success(scores: Sequence[ItemScore]) -> list[float]:
    return [float(s.success) for s in scores]


def _seed_mean(runs: Sequence[Sequence[ItemScore]]) -> list[float]:
    return [float(x) for x in np.mean([_success(r) for r in runs], axis=0)]


def _paired(a: Sequence[float], b: Sequence[float], clusters: Sequence[str], cfg: dict[str, Any]) -> dict[str, Any]:
    return paired_permutation_test(list(a), list(b), clusters, permutations=cfg["permutations"],
                                   resamples=cfg["resamples"], level=cfg["level"], seed=cfg["seed"]).as_dict()


def _ci(values: Sequence[float], clusters: Sequence[str], cfg: dict[str, Any]) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    return cluster_bootstrap(lambda idx: float(v[idx].mean()), clusters, resamples=cfg["resamples"],
                             level=cfg["level"], seed=cfg["seed"]).as_dict()


def _by_route(values: Sequence[float], items: Sequence[GoldItem]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for route in ("ACT", "CLARIFY", "ABSTAIN"):
        v = [x for x, it in zip(values, items, strict=True) if it.route == route]
        out[route] = round(float(np.mean(v)), 3) if v else None
    return out


def analyse(items: Sequence[GoldItem], predictions: dict[str, dict[str, str]], training_families: set[str],
            cfg: dict[str, Any]) -> dict[str, Any]:
    """The registered analysis on ``items`` given every system's frozen intents by item id.

    ``predictions`` holds S3-drafted, S3-human, S3-contrib-only and S6-{drafted,human}-s{0..4}; S1
    v1.1 and the oracle are computed here."""
    conf = [it for it in items if it.family_id not in training_families]
    fams = [it.family_id for it in conf]
    writers = sorted({writer_of(it.id) for it in conf})
    runs: dict[str, list[ItemScore]] = {name: respond(p, conf, name) for name, p in predictions.items()}
    rules = KeywordIntentV11()
    from athar.normalize import norm
    runs["S1-v1.1"] = respond(dict(zip([it.id for it in conf], rules.predict([norm(it.text) for it in conf]),
                                       strict=True)), conf, "S1-v1.1")
    runs["oracle-v1.1"] = respond({it.id: it.intent for it in conf}, conf, "oracle-v1.1")
    scores: dict[str, list[float]] = {n: _success(r) for n, r in runs.items()}
    for kind in ("drafted", "human"):
        scores[f"S6-{kind} (seed mean)"] = _seed_mean([runs[f"S6-{kind}-s{s}"] for s in SEEDS])

    confirmatory = len(conf) >= MIN_ITEMS and len(writers) >= MIN_WRITERS
    primary = {h: _paired(scores[a], scores[b], fams, cfg) for h, (a, b) in HYPOTHESES.items()}
    primary["H-a S3-human vs S3-drafted"]["mcnemar"] = mcnemar_exact(
        [bool(x) for x in scores["S3-human"]], [bool(x) for x in scores["S3-drafted"]])
    if confirmatory:
        adjusted = holm({h: t["p_value"] for h, t in primary.items()})
        for h, t in primary.items():
            t["p_holm"], t["significant"] = adjusted[h], adjusted[h] < cfg["alpha"]

    abstain = [k for k, it in enumerate(conf) if it.route == "ABSTAIN"]
    secondary = {"H-c ABSTAIN success, S6-human vs S6-drafted (seed means)": {
        "n": len(abstain),
        **(_paired([scores["S6-human (seed mean)"][k] for k in abstain],
                   [scores["S6-drafted (seed mean)"][k] for k in abstain], [fams[k] for k in abstain], cfg)
           if abstain else {"note": "no gold-ABSTAIN item"}),
        "p_adjustment": "none (secondary)"}}

    estimation = {}
    for name in ("S3-human", "S6-human (seed mean)"):
        diff = [a - b for a, b in zip(scores[name], scores["S1-v1.1"], strict=True)]
        ci = _ci(diff, fams, cfg)
        estimation[f"{name} minus S1-v1.1"] = {
            "difference": round(float(np.mean(diff)), 4), "ci": ci,
            "within_0.10_of_the_rules": bool(ci["low"] > -MARGIN),
            "reading": (f"within {MARGIN:.2f} of the rules" if ci["low"] > -MARGIN
                        else f"not shown to be within {MARGIN:.2f} of the rules"),
            "never_claimed": "not worse than the rules (protocol section 5)"}

    descriptive = {"size-matched control (A4): S3-contrib-only vs S3-drafted":
                   _paired(scores["S3-contrib-only"], scores["S3-drafted"], fams, cfg)}
    sensitivity: dict[str, Any] = {}
    if len(writers) > 1:
        wcl = [writer_of(it.id) for it in conf]
        sensitivity["writer_clustered"] = {h: _paired(scores[a], scores[b], wcl, cfg) for h, (a, b) in HYPOTHESES.items()}
        sensitivity["leave_one_writer_out"] = {
            w: {n: round(float(np.mean([v for v, it in zip(s, conf, strict=True) if writer_of(it.id) != w])), 3)
                for n, s in scores.items()} for w in writers}

    summary = {n: {"sts": round(float(np.mean(s)), 3), "ci": _ci(s, fams, cfg), "by_route": _by_route(s, conf)}
               for n, s in scores.items()}
    return {
        "n_items": len(items), "confirmatory_items": len(conf), "writers": writers,
        "families": len(set(fams)),
        "hypothesis_tests": ("confirmatory (Holm over H-a and H-b)" if confirmatory else
                             f"descriptive only: fewer than {MIN_ITEMS} confirmatory items or {MIN_WRITERS} writers"),
        "summary": summary, "primary": primary, "secondary": secondary, "estimation": estimation,
        "descriptive": descriptive, "sensitivity": sensitivity,
        "metrics": {n: {k: v for k, v in aggregate(r).items() if not isinstance(v, (dict, list))}
                    for n, r in runs.items()},
        "items": {n: [{"id": s.id, "success": bool(s.success), "pred_intent": s.pred_intent,
                       "pred_route": s.pred_route} for s in r] for n, r in runs.items()},
    }


# --------------------------------------------------------------------------- the run (I/O)
def _freeze(name: str) -> dict[str, Any]:
    path = S.MODELS_V3 / name
    if not path.exists():
        raise EvaluationError(f"{path.relative_to(REPO)} is missing")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def frozen_predictions() -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Every frozen prediction file, after checking each freeze (models, predictions, sources,
    protocol, amendments) is unchanged."""
    problems = S.verify()
    controls = _freeze("freeze_controls.json")
    problem = amend.check(controls)
    if problem:
        problems.append(f"freeze_controls.json: {problem}")
    if problems:
        raise EvaluationError("v3 freezes do not verify:\n  " + "\n  ".join(problems))
    preds: dict[str, dict[str, str]] = {}
    hashes: dict[str, Any] = {}
    for fname in ("freeze_s3.json", "freeze_s6.json", "freeze_controls.json"):
        fz = _freeze(fname)
        hashes[fname] = fz["freeze_sha256"]
        for name, rec in fz["predictions"].items():
            preds[name] = json.loads((REPO / rec["file"]).read_text(encoding="utf-8"))
    gpu = S.MODELS_V3 / "freeze_gpu.json"
    hashes["freeze_gpu.json"] = (json.loads(gpu.read_text(encoding="utf-8"))["freeze_sha256"] if gpu.exists()
                                 else "absent: the GPU systems of section 6 were not frozen before G4 and are not scored")
    expected = {"S3-drafted", "S3-human", "S3-contrib-only",
                *(f"S6-{k}-s{s}" for k in ("drafted", "human") for s in SEEDS)}
    if set(preds) != expected:
        raise EvaluationError(f"frozen predictions {sorted(preds)} are not the registered systems {sorted(expected)}")
    return preds, hashes


def check_gold(items: Sequence[GoldItem], frozen_set_sha256: str, predictions: dict[str, dict[str, str]]) -> None:
    """The gold set must be the frozen v3 set: same ids, texts and sending moments, and every
    system must have predicted every item."""
    as_v3 = [D.V3Item(id=it.id, text=it.text, ref_datetime=it.ref_datetime, timezone=it.timezone,
                      family_id=it.family_id, block="", writer=writer_of(it.id)) for it in items]
    if D.set_hash(as_v3) != frozen_set_sha256:
        raise EvaluationError("the locked gold set is not the frozen v3 set (ids, texts or sending moments differ)")
    ids = {it.id for it in items}
    for name, p in predictions.items():
        if set(p) != ids:
            raise EvaluationError(f"{name} does not hold exactly one prediction per gold item")


def evaluate(manifest: Path, out: Path | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    out = out or OUT   # read at call time, so the output path is never fixed at import
    if out.exists():
        raise EvaluationError(f"{out} exists: the v3 scoring run happens once (gate G5)")
    cfg = cfg or _cfg()
    preds, hashes = frozen_predictions()
    items = list(load_split(manifest, "test", unlock_test=True))
    check_gold(items, _freeze("freeze_s3.json")["v3_set_sha256"], preds)
    m = read_manifest(MAIN)
    training = set(m["splits"]["train"]["families"]) | set(m["splits"]["dev"]["families"])
    result = {"protocol": "docs/protocol_v3.md", "protocol_sha256": sha256_file(S.PROTOCOL),
              "amendments": "docs/protocol_v3_amendments.md",
              "v3_manifest": read_manifest(manifest)["manifest_sha256"], "freezes": hashes,
              "eval_config_sha256": sha256_file(REPO / "configs" / "eval.yaml"),
              **analyse(items, preds, training, cfg)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m athar.v3.evaluate", description=(__doc__ or "").splitlines()[0])
    p.add_argument("--manifest", type=Path, required=True, help="locked v3 gold manifest (athar-data lock)")
    p.add_argument("--check", action="store_true",
                   help="check the preconditions (freezes, gold set) and write nothing")
    a = p.parse_args(argv)
    try:
        if a.check:
            preds, _ = frozen_predictions()
            check_gold(list(load_split(a.manifest, "test", unlock_test=True)),
                       _freeze("freeze_s3.json")["v3_set_sha256"], preds)
            print("ready: freezes verify and the gold set is the frozen v3 set")
            return 0
        r = evaluate(a.manifest, out=OUT)
    except EvaluationError as e:
        print(f"refused: {e}")
        return 1
    print(f"wrote {OUT}: {r['confirmatory_items']} items, {len(r['writers'])} writers; "
          f"{r['hypothesis_tests']}")
    for h, t in r["primary"].items():
        print(f"  {h}: {t['difference']:+.3f}  p {t['p_value']:.4f}  Holm p {t.get('p_holm', 'n/a')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
