"""Post hoc: do the LLM intent recognizers (S4, S7) differ from the keyword rules (S1)?

Written on 2026-11-18, after every system had been scored on both locked sets. Nothing is rerun:
the script reads the stored per-item outputs of the GPU runs and of S1, keeps the items both
scored, and applies the pre-registered machinery (configs/eval.yaml: family-level paired
randomization test, family bootstrap CI, same seed) with Holm over the six comparisons of each
set. It answers a reviewer's question, so it is exploratory and labelled so in the report.

    uv run python scripts/posthoc_llm_vs_rules.py      # -> results/posthoc/llm_vs_rules.json

All systems share the core v1 extractor and policy here (S1 as frozen, the LLMs as run), so a
difference comes from intent recognition only. S7 is the per-item mean over its three seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from athar.eval.stats import holm, mcnemar_exact, paired_permutation_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/posthoc/llm_vs_rules.json"

S4_MODELS = {"jais2-8b": "Jais 2 8B", "jais3-7b": "Jais-3-7B", "jais3-3.7b": "Jais-3-3.7B",
             "nile-4b": "Nile-Chat 4B", "qwen3-8b": "Qwen3 8B"}
TEST_DIRS = {"jais2-8b": "7b-others", "nile-4b": "7b-others", "qwen3-8b": "7b-others",
             "jais3-7b": "7a-jais3", "jais3-3.7b": "7a-jais3"}
NAT_DIRS = {"jais2-8b": "3-s4-jais2", "jais3-7b": "2-s4-jais3", "jais3-3.7b": "2-s4-jais3",
            "nile-4b": "4-s4-nile-qwen", "qwen3-8b": "4-s4-nile-qwen"}


def load(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def items_of(rel: str, system: str) -> dict[str, dict[str, Any]]:
    return {x["id"]: x for x in load(rel)["systems"][system]["items"]}


def seed_mean(rels: list[str], systems: list[str]) -> dict[str, dict[str, Any]]:
    """Per-item S7 outcome averaged over seeds: success and unsafe (executed, not a success)."""
    runs = [items_of(r, s) for r, s in zip(rels, systems, strict=True)]
    ids = set.intersection(*(set(r) for r in runs))
    return {k: {**runs[0][k], "success": float(np.mean([r[k]["success"] for r in runs])),
                "unsafe": float(np.mean([bool(r[k]["executed"]) and not r[k]["success"] for r in runs]))}
            for k in ids}


def unsafe(item: dict[str, Any]) -> float:
    return float(item["unsafe"]) if "unsafe" in item else float(bool(item["executed"]) and not item["success"])


def compare(ref: dict[str, dict[str, Any]], others: dict[str, dict[str, dict[str, Any]]],
            cfg: dict[str, Any]) -> dict[str, Any]:
    ids = sorted(ref)
    fams = [ref[k]["family_id"] for k in ids]
    s1 = [float(ref[k]["success"]) for k in ids]
    out: dict[str, Any] = {"n": len(ids), "families": len(set(fams)),
                           "S1": {"sts": float(np.mean(s1)),
                                  "unsafe_execution_rate": float(np.mean([unsafe(ref[k]) for k in ids]))},
                           "comparisons": {}}
    pvals = {}
    for name, its in others.items():
        if set(its) & set(ids) != set(ids):
            sys.exit(f"{name}: stored outputs do not cover the {len(ids)} items")
        for k in ids:  # same gold on both sides, or the comparison is meaningless
            if (its[k]["gold_route"], its[k]["gold_intent"]) != (ref[k]["gold_route"], ref[k]["gold_intent"]):
                sys.exit(f"{name}: gold of {k} differs from S1's")
        a = [float(its[k]["success"]) for k in ids]
        t = paired_permutation_test(a, s1, fams, permutations=cfg["permutations"],
                                    resamples=cfg["resamples"], level=cfg["level"], seed=cfg["seed"])
        rec = {"sts": float(np.mean(a)), "unsafe_execution_rate": float(np.mean([unsafe(its[k]) for k in ids])),
               **t.as_dict()}
        if all(x in (0.0, 1.0) for x in a):
            rec["mcnemar"] = mcnemar_exact([bool(x) for x in a], [bool(x) for x in s1])
        out["comparisons"][f"{name} vs S1"] = rec
        pvals[f"{name} vs S1"] = rec["p_value"]
    for key, p in holm(pvals).items():
        out["comparisons"][key]["p_holm"] = p
    return out


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs/eval.yaml").read_text(encoding="utf-8"))
    # locked test (author, 120 items)
    ref_t = {x["id"]: x for x in load("results/test_results.json")["systems"]["S1"]["items"]}
    test = {f"S4 {label}": items_of(f"results/s4/{TEST_DIRS[m]}/s4_results/{m}.json", f"S4-{m}-6shot-prior")
            for m, label in S4_MODELS.items()}
    test["S7 Jais 2 + QLoRA"] = seed_mean(
        [f"results/s4/7cd-s7-s8/s4_results/s7_jais2_qlora_s{i}_test.json" for i in range(3)],
        [f"S7-jais2-qlora-s{i}-prior" for i in range(3)])
    # natural set (17 other writers, 238 confirmatory items)
    ref_n = {x["id"]: x for x in load("results/natural/evaluation.json")["systems"]["S1"]["items"]}
    nat = {f"S4 {label}": items_of(f"results/natural/kaggle/{NAT_DIRS[m]}/s4_results/{m}.json",
                                   f"S4-{m}-6shot-prior") for m, label in S4_MODELS.items()}
    nat["S7 Jais 2 + QLoRA"] = seed_mean(
        [f"results/natural/kaggle/5-s7/s4_results/s7_jais2_qlora_s{i}_test.json" for i in range(3)],
        [f"S7-jais2-qlora-s{i}-prior" for i in range(3)])
    nat = {k: {i: v[i] for i in ref_n if i in v} for k, v in nat.items()}
    result = {"note": "post hoc, exploratory; Holm over the six comparisons within each set",
              "config": {k: cfg[k] for k in ("seed", "level", "resamples", "permutations")},
              "locked_test": compare(ref_t, test, cfg), "natural": compare(ref_n, nat, cfg)}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for split in ("locked_test", "natural"):
        r = result[split]
        print(f"{split}: n={r['n']} families={r['families']} S1 STS={r['S1']['sts']:.3f} "
              f"unsafe={r['S1']['unsafe_execution_rate']:.3f}")
        for k, v in r["comparisons"].items():
            print(f"  {k:28s} STS {v['sts']:.3f} unsafe {v['unsafe_execution_rate']:.3f} "
                  f"diff {v['difference']:+.3f} [{v['ci']['low']:+.3f}, {v['ci']['high']:+.3f}] "
                  f"p {v['p_value']:.4f} Holm {v['p_holm']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
