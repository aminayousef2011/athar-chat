"""``athar-v2``: freeze the v2 systems and run the pre-registered natural-phrasing evaluation.

    athar-v2 freeze   --manifest data/splits/manifest.json --config configs/train.yaml --out models_v2
    athar-v2 verify   models_v2
    athar-v2 evaluate --natural data/natural/manifest.json --manifest data/splits/manifest.json \\
                      --s8-traces T0.jsonl T1.jsonl T2.jsonl [--llm-results REPORT.json ...] \\
                      --out results/natural/evaluation.json

``evaluate`` follows docs/protocol_natural.md: the confirmatory set is the items whose scenario
family is not in train/dev; N1 (S1 vs S2), N2 (S1 v1.1 vs S1) and N3 (S9 vs S8, seed means) are
tested with family-level paired randomization and Holm, only when the set has at least 60 items
from at least 3 writers; everything else is descriptive. It refuses to overwrite its output.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from athar.data.split import load_split, read_manifest
from athar.eval import harness as H
from athar.eval.metrics import safe_task_success, score_all
from athar.eval.stats import cluster_bootstrap, holm, mcnemar_exact, paired_permutation_test
from athar.intent.rules import KeywordIntent
from athar.intent.train import load_recognizer
from athar.pipeline import respond_with_intent
from athar.schema import INTENTS, GoldItem
from athar.v2 import core as core_v11
from athar.v2 import gcv, hybrid
from athar.v2.core import CoreV11System
from athar.v2.rules import KeywordIntentV11

REPO = Path(__file__).resolve().parents[3]
MIN_ITEMS, MIN_WRITERS, WRITER_CLUSTER_MIN = 60, 3, 5


def writer_of(item_id: str) -> str:
    """Contributor code from a multi-writer id ``<card>-<version>-<code>``."""
    parts = item_id.split("-", 2)
    return parts[2] if len(parts) == 3 else "unknown"


def _cfg() -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load((REPO / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    cfg["comparisons"] = []
    return cfg


def _paired(a: Sequence[float], b: Sequence[float], clusters: Sequence[str], cfg: dict[str, Any]) -> dict[str, Any]:
    t = paired_permutation_test(list(a), list(b), clusters, permutations=cfg["permutations"],
                                resamples=cfg["resamples"], level=cfg["level"], seed=cfg["seed"])
    return t.as_dict()


def _success(report: dict[str, Any]) -> list[float]:
    return [float(it["success"]) for it in report["items"]]


def _mean_over_seeds(reports: Sequence[dict[str, Any]]) -> list[float]:
    return [float(x) for x in np.mean([_success(r) for r in reports], axis=0)]


def systems(models: Path, models_v2: Path, traces: Sequence[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {"S1": KeywordIntent(), "S2": load_recognizer("S2", models),
                           "S3": load_recognizer("S3", models), "oracle": H.Oracle(),
                           "S1-v1.1": CoreV11System(KeywordIntentV11(), "S1-v1.1"),
                           "S2-gcv": CoreV11System(gcv.load_recognizer("S2-gcv", models_v2), "S2-gcv"),
                           "S3-gcv": CoreV11System(gcv.load_recognizer("S3-gcv", models_v2), "S3-gcv"),
                           "oracle-v1.1": CoreV11System(None, "oracle-v1.1")}
    for seed, path in enumerate(traces):
        trace = hybrid.read_trace(path)
        out[f"S8-s{seed}"] = hybrid.S8Replay(trace, f"S8-s{seed}")
        out[f"S9-s{seed}"] = hybrid.S9(trace, f"S9-s{seed}")
    return out


def _sts_ci(values: Sequence[float], clusters: Sequence[str], cfg: dict[str, Any]) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    return cluster_bootstrap(lambda idx: float(v[idx].mean()), clusters, resamples=cfg["resamples"],
                             level=cfg["level"], seed=cfg["seed"]).as_dict()


def author_vs_contributor(items: Sequence[GoldItem], reports: dict[str, dict[str, Any]],
                          systems_: dict[str, Any], main_manifest: Path) -> dict[str, Any]:
    """On the (card, version) slots both wrote: each system's STS on the author's locked test items
    and on the contributors' items (descriptive)."""
    test = {it.id: it for it in load_split(main_manifest, "test", unlock_test=True)}
    slots: dict[str, list[int]] = defaultdict(list)
    for k, it in enumerate(items):
        slot = it.id.rsplit("-", 1)[0] if writer_of(it.id) != "unknown" else it.id
        if slot in test:
            slots[slot].append(k)
    if not slots:
        return {"slots": 0}
    author_items = [test[s] for s in sorted(slots)]
    out: dict[str, Any] = {"slots": len(slots), "contributor_items": sum(len(v) for v in slots.values()), "systems": {}}
    for name, system in systems_.items():
        if name.startswith(("S8", "S9")):
            continue   # no S8 trace exists for these author items in this run
        author_sts = safe_task_success(score_all(author_items, H.run(system, author_items)))
        contrib = [np.mean([reports[name]["items"][k]["success"] for k in slots[s]]) for s in sorted(slots)]
        out["systems"][name] = {"author": round(author_sts, 3), "contributors": round(float(np.mean(contrib)), 3)}
    return out


def llm_v11(items: Sequence[GoldItem], llm_reports: Sequence[Path]) -> dict[str, Any]:
    """S4/S7 rebuilt with core v1.1 from the intents they predicted in their GPU run (descriptive)."""
    by_id = {it.id: it for it in items}
    out: dict[str, Any] = {}
    for path in llm_reports:
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        for name, sysrep in rep["systems"].items():
            stored = [x for x in sysrep["items"] if x["id"] in by_id]
            if not stored:
                continue
            its = [by_id[x["id"]] for x in stored]
            v1 = [respond_with_intent(x["pred_intent"], it.text, it.ref_datetime, it.timezone, name)
                  if x["pred_intent"] in INTENTS else None for x, it in zip(stored, its, strict=True)]
            v2 = [core_v11.respond_with_intent(x["pred_intent"], it.text, it.ref_datetime, it.timezone, name)
                  if x["pred_intent"] in INTENTS else None for x, it in zip(stored, its, strict=True)]
            out[name] = {"n": len(its), "stored": round(float(np.mean([x["success"] for x in stored])), 3),
                         "rebuilt_v1": round(safe_task_success(score_all(its, v1)), 3),
                         "with_core_v1.1": round(safe_task_success(score_all(its, v2)), 3)}
    return out


def evaluate(natural: Path, main_manifest: Path, models: Path, models_v2: Path, traces: Sequence[Path],
             llm_reports: Sequence[Path], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """The whole evaluation. ``cfg`` overrides configs/eval.yaml (tests only)."""
    cfg = cfg or _cfg()
    items = list(load_split(natural, "test", unlock_test=True))
    m = read_manifest(main_manifest)
    training = set(m["splits"]["train"]["families"]) | set(m["splits"]["dev"]["families"])
    conf = [it for it in items if it.family_id not in training]
    apart = [it for it in items if it.family_id in training]
    writers = sorted({writer_of(it.id) for it in conf})
    sys_ = systems(models, models_v2, traces)

    def run_all(subset: list[GoldItem]) -> dict[str, dict[str, Any]]:
        reps = {}
        for name, system in sys_.items():
            c = cfg if not name.startswith(("S8", "S9")) else {**cfg, "perturbations": []}
            reps[name] = H.evaluate_system(system, subset, c)
        return reps

    reports = run_all(conf)
    fams = [it.family_id for it in conf]
    s8 = _mean_over_seeds([reports[f"S8-s{s}"] for s in range(len(traces))])
    s9 = _mean_over_seeds([reports[f"S9-s{s}"] for s in range(len(traces))])
    scores = {**{n: _success(r) for n, r in reports.items()}, "S8 (seed mean)": s8, "S9 (seed mean)": s9}
    hypotheses = {"N1 S1 vs S2": ("S1", "S2"), "N2 S1-v1.1 vs S1": ("S1-v1.1", "S1"),
                  "N3 S9 vs S8": ("S9 (seed mean)", "S8 (seed mean)")}
    confirmatory = len(conf) >= MIN_ITEMS and len(writers) >= MIN_WRITERS
    tests: dict[str, Any] = {}
    for h, (a, b) in hypotheses.items():
        tests[h] = _paired(scores[a], scores[b], fams, cfg)
        if not h.startswith("N3"):
            tests[h]["mcnemar"] = mcnemar_exact([bool(x) for x in scores[a]], [bool(x) for x in scores[b]])
    if confirmatory:
        adjusted = holm({h: t["p_value"] for h, t in tests.items()})
        for h, t in tests.items():
            t["p_holm"], t["significant"] = adjusted[h], adjusted[h] < cfg["alpha"]
    sensitivity: dict[str, Any] = {}
    if len(writers) >= WRITER_CLUSTER_MIN:
        wcl = [writer_of(it.id) for it in conf]
        sensitivity["writer_clustered"] = {h: _paired(scores[a], scores[b], wcl, cfg) for h, (a, b) in hypotheses.items()}
    sensitivity["leave_one_writer_out"] = {
        w: {n: round(float(np.mean([v for v, it in zip(s, conf, strict=True) if writer_of(it.id) != w])), 3)
            for n, s in scores.items()} for w in writers if len(writers) > 1}
    summary = {n: {"sts": round(float(np.mean(s)), 3), "ci": _sts_ci(s, fams, cfg)} for n, s in scores.items()}
    return {
        "protocol": "docs/protocol_natural.md",
        "natural_manifest": read_manifest(natural)["manifest_sha256"],
        "v1_freeze": json.loads((models / "freeze.json").read_text(encoding="utf-8"))["freeze_sha256"],
        "v2_freeze": gcv.read_freeze(models_v2)["freeze_sha256"],
        "n_items": len(items), "confirmatory_items": len(conf), "writers": writers,
        "hypothesis_tests": "confirmatory (Holm over N1-N3)" if confirmatory
        else f"descriptive only: fewer than {MIN_ITEMS} items or {MIN_WRITERS} writers",
        "summary": summary, "tests": tests, "sensitivity": sensitivity,
        "training_scenarios": ({n: round(safe_task_success(score_all(apart, H.run(s, apart))), 3)
                                for n, s in sys_.items()} if apart else {}),
        "author_vs_contributor": author_vs_contributor(conf, reports, sys_, main_manifest),
        "llm_with_core_v1.1": llm_v11(conf, llm_reports),
        "systems": {n: {k: v for k, v in r.items() if k in ("metrics", "ci", "items")} for n, r in reports.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="athar-v2", description=(__doc__ or "").splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--manifest", default="data/splits/manifest.json")
    f.add_argument("--config", default="configs/train.yaml")
    f.add_argument("--out", default="models_v2")
    v = sub.add_parser("verify")
    v.add_argument("models_v2", nargs="?", default="models_v2")
    e = sub.add_parser("evaluate")
    e.add_argument("--natural", required=True)
    e.add_argument("--manifest", default="data/splits/manifest.json")
    e.add_argument("--models", default="models")
    e.add_argument("--models-v2", default="models_v2")
    e.add_argument("--s8-traces", nargs="+", required=True)
    e.add_argument("--llm-results", nargs="*", default=[])
    e.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "freeze":
        fz = gcv.freeze(a.manifest, a.config, a.out)
        for n, s in fz["systems"].items():
            best = next(g for g in s["grid"] if g["params"] == s["params"])
            print(f"{n}: C={s['params']['C']} word={s['params']['word_ngram']} "
                  f"char={s['params'].get('char_ngram')} cv F1 {best['cv']['intent_macro_f1']}")
        print(f"wrote {a.out}/freeze.json  freeze_sha256={fz['freeze_sha256']}")
        return 0
    if a.cmd == "verify":
        problems = gcv.verify(a.models_v2)
        print("\n".join(problems) if problems else f"{a.models_v2}: OK  "
              f"freeze_sha256={gcv.read_freeze(a.models_v2)['freeze_sha256']}")
        return 1 if problems else 0
    if Path(a.out).exists():
        print(f"{a.out} exists: the natural-phrasing evaluation runs once; nothing written")
        return 1
    r = evaluate(Path(a.natural), Path(a.manifest), Path(a.models), Path(a.models_v2),
                 [Path(t) for t in a.s8_traces], [Path(x) for x in a.llm_results])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"{r['confirmatory_items']} confirmatory items from {len(r['writers'])} writers: {r['hypothesis_tests']}")
    for n, s in r["summary"].items():
        print(f"  {n:16} STS {s['sts']:.3f} [{s['ci']['low']:.3f}, {s['ci']['high']:.3f}]")
    for h, t in r["tests"].items():
        holm_p = f" Holm {t['p_holm']:.4f}" if "p_holm" in t else ""
        print(f"  {h:20} diff {t['difference']:+.3f} [{t['ci']['low']:+.3f}, {t['ci']['high']:+.3f}] "
              f"p {t['p_value']:.4f}{holm_p}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
