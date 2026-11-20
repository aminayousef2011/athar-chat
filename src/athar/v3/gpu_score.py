"""Descriptive scoring of protocol v3's GPU systems (section 6, amendments A5 and A7).

``athar.v3.evaluate`` (gate G5) records the hash of ``models_v3/freeze_gpu.json`` but scores only the
registered S3/S6 systems. This module scores the GPU systems frozen at G3, descriptively, with the
rules every other system uses, and writes ``results/v3/evaluation_gpu.json`` once:

- ``S4-jais3-7b-inline-norm`` / ``-raw``: the frozen intents through core v1.1 (as in G5);
- ``S8-fair``: the frozen greedy tool-calling trace, replayed (``athar.v2.hybrid.S8Replay``);
- ``S9-fair``: the hybrid computed from the same trace (``athar.v2.hybrid.S9``), as for the natural set;
- ``S1-v1.1``: the rules, for reference.

Comparisons (descriptive, unadjusted p, family-level as G5): S4 raw vs normalized text, S9-fair vs
S8-fair, and each GPU system minus S1 v1.1. Nothing here is a registered hypothesis.

    python -m athar.v3.gpu_score --manifest data/v3/manifest.json [--check]

It runs only after G5 (``results/v3/evaluation.json`` exists), when the GPU freeze, its output
hashes and the amendments file verify, and the locked gold set is the frozen v3 set.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from athar.data.io import sha256_file
from athar.data.split import load_split, read_manifest
from athar.eval.metrics import ItemScore, aggregate, score_all
from athar.normalize import norm
from athar.schema import GoldItem
from athar.v2.hybrid import S9, S8Replay
from athar.v2.rules import KeywordIntentV11
from athar.v3 import amend
from athar.v3 import evaluate as E

OUT = E.REPO / "results" / "v3" / "evaluation_gpu.json"
S4_NAMES = ("S4-jais3-7b-inline-norm", "S4-jais3-7b-inline-raw")
S8_NAME = "S8-fair"
COMPARISONS = {
    "S4 raw vs normalized text": ("S4-jais3-7b-inline-raw", "S4-jais3-7b-inline-norm"),
    "S9-fair vs S8-fair": ("S9-fair", "S8-fair"),
    "S4 normalized minus S1-v1.1": ("S4-jais3-7b-inline-norm", "S1-v1.1"),
    "S4 raw minus S1-v1.1": ("S4-jais3-7b-inline-raw", "S1-v1.1"),
    "S8-fair minus S1-v1.1": ("S8-fair", "S1-v1.1"),
    "S9-fair minus S1-v1.1": ("S9-fair", "S1-v1.1"),
}


def analyse_gpu(items: Sequence[GoldItem], s4: Mapping[str, Mapping[str, str]],
                trace: Mapping[str, Mapping[str, Any]], training_families: set[str],
                cfg: dict[str, Any]) -> dict[str, Any]:
    """Descriptive scores of the GPU systems on the confirmatory ``items``.

    ``s4`` maps each S4 system to its intents by item id; ``trace`` is S8-fair's trace by item id."""
    conf = [it for it in items if it.family_id not in training_families]
    fams = [it.family_id for it in conf]
    runs: dict[str, list[ItemScore]] = {name: E.respond(dict(s4[name]), conf, name) for name in S4_NAMES}
    s8_out = S8Replay(trace, S8_NAME).respond_items(conf)
    runs[S8_NAME] = score_all(conf, s8_out)
    runs["S9-fair"] = score_all(conf, S9(trace, "S9-fair").respond_items(conf))
    rules = KeywordIntentV11()
    runs["S1-v1.1"] = E.respond(dict(zip([it.id for it in conf], rules.predict([norm(it.text) for it in conf]),
                                         strict=True)), conf, "S1-v1.1")
    scores = {n: [float(s.success) for s in r] for n, r in runs.items()}
    comparisons = {k: {**E._paired(scores[a], scores[b], fams, cfg), "p_adjustment": "none (descriptive)"}
                   for k, (a, b) in COMPARISONS.items()}
    return {
        "status": "descriptive (protocol v3 section 6; amendments A5, A7)",
        "confirmatory_items": len(conf), "families": len(set(fams)),
        "s8_fair_no_valid_call": sum(o is None for o in s8_out),
        "s4_norm_raw_same_intent": sum(s4[S4_NAMES[0]].get(it.id) == s4[S4_NAMES[1]].get(it.id) for it in conf),
        "summary": {n: {"sts": round(float(np.mean(s)), 3), "ci": E._ci(s, fams, cfg),
                        "by_route": E._by_route(s, conf)} for n, s in scores.items()},
        "comparisons": comparisons,
        "metrics": {n: {k: v for k, v in aggregate(r).items() if not isinstance(v, (dict, list))}
                    for n, r in runs.items()},
        "items": {n: [{"id": s.id, "success": bool(s.success), "pred_intent": s.pred_intent,
                       "pred_route": s.pred_route} for s in r] for n, r in runs.items()},
    }


def frozen_gpu() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]], str]:
    """The GPU systems' frozen outputs, after checking the freeze, its output hashes and the
    amendments file (append-only)."""
    fz = E._freeze("freeze_gpu.json")
    problems = []
    if (p := amend.check(fz)):
        problems.append(f"freeze_gpu.json: {p}")
    for name, rec in fz["outputs"].items():
        path = E.REPO / rec["file"]
        if not path.exists() or sha256_file(path) != rec["sha256"]:
            problems.append(f"{name}: {rec['file']} is missing or changed")
    if set(fz["outputs"]) != {*S4_NAMES, S8_NAME}:
        problems.append(f"freeze_gpu.json holds {sorted(fz['outputs'])}, not the A5 systems")
    if problems:
        raise E.EvaluationError("the GPU freeze does not verify:\n  " + "\n  ".join(problems))
    s4 = {}
    for name in S4_NAMES:
        body = json.loads((E.REPO / fz["outputs"][name]["file"]).read_text(encoding="utf-8"))
        s4[name] = {i: rec["intent"] for i, rec in body.items()}
    lines = (E.REPO / fz["outputs"][S8_NAME]["file"]).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return s4, {r["id"]: r for r in rows}, str(fz["freeze_sha256"])


def run(manifest: Path, out: Path | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    out = out or OUT
    if out.exists():
        raise E.EvaluationError(f"{out} exists: the GPU systems are scored once")
    if not E.OUT.exists():
        raise E.EvaluationError("results/v3/evaluation.json is missing: run gate G5 first")
    cfg = cfg or E._cfg()
    s4, trace, fz_sha = frozen_gpu()
    items = list(load_split(manifest, "test", unlock_test=True))
    E.check_gold(items, E._freeze("freeze_s3.json")["v3_set_sha256"],
                 {**s4, S8_NAME: {i: "" for i in trace}})
    m = read_manifest(E.MAIN)
    training = set(m["splits"]["train"]["families"]) | set(m["splits"]["dev"]["families"])
    result = {"freeze_gpu_sha256": fz_sha, "v3_manifest": read_manifest(manifest)["manifest_sha256"],
              "eval_config_sha256": sha256_file(E.REPO / "configs" / "eval.yaml"),
              **analyse_gpu(items, s4, trace, training, cfg)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m athar.v3.gpu_score", description=(__doc__ or "").splitlines()[0])
    p.add_argument("--manifest", type=Path, required=True, help="locked v3 gold manifest (athar-data lock)")
    p.add_argument("--check", action="store_true", help="verify the GPU freeze only and write nothing")
    a = p.parse_args(argv)
    try:
        if a.check:
            frozen_gpu()
            print("ready: the GPU freeze, its outputs and the amendments file verify")
            return 0
        r = run(a.manifest)
    except E.EvaluationError as e:
        print(f"refused: {e}")
        return 1
    print(f"wrote {OUT}: {r['confirmatory_items']} items (descriptive)")
    for k, t in r["comparisons"].items():
        print(f"  {k}: {t['difference']:+.3f}  p {t['p_value']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
