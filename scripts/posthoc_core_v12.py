"""Post hoc, exploratory: does core v1.2 (``athar.v2.core12``) raise Safe Task Success?

Core v1.2 adds four extractor fixes written from the oracle's natural-set failures (development
set). v3 was not inspected while the fixes were written and is the held-out check; the author's
locked test is reported too. Every system keeps its frozen intents (S1 rules, the frozen v3
classifier predictions, or the gold intent); only the extractor changes, so each comparison is
paired item by item, with the family-level bootstrap and permutation test of the registered
analyses (unadjusted p, descriptive). No registered result changes: they all use core v1.1.

    uv run python scripts/posthoc_core_v12.py            # writes results/posthoc/core_v12.json

Only aggregates and item ids are written (the v3 texts are private).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athar.data.split import load_split, read_manifest  # noqa: E402
from athar.eval.metrics import score_all  # noqa: E402
from athar.eval.stats import cluster_bootstrap, paired_permutation_test  # noqa: E402
from athar.normalize import norm  # noqa: E402
from athar.schema import INTENTS, GoldItem, Response  # noqa: E402
from athar.v2 import core as V11  # noqa: E402
from athar.v2 import core12 as V12  # noqa: E402
from athar.v2.rules import KeywordIntentV11  # noqa: E402

OUT = ROOT / "results/posthoc/core_v12.json"
SEEDS = range(5)


def run(respond: Callable[..., Response], intents: dict[str, str], items: Sequence[GoldItem], name: str) -> list[float]:
    outs = [respond(intents[it.id], it.text, it.ref_datetime, it.timezone, name)
            if intents.get(it.id) in INTENTS else None for it in items]
    return [float(s.success) for s in score_all(items, outs)]


def compare(name: str, intent_sets: list[dict[str, str]], items: Sequence[GoldItem], cfg: dict[str, Any]) -> dict[str, Any]:
    """v1.1 vs v1.2 for one system (seed mean when several intent sets are given)."""
    a = np.mean([run(V11.respond_with_intent, i, items, name) for i in intent_sets], axis=0)
    b = np.mean([run(V12.respond_with_intent, i, items, name) for i in intent_sets], axis=0)
    fams = [it.family_id for it in items]
    test = paired_permutation_test(list(b), list(a), fams, permutations=cfg["permutations"],
                                   resamples=cfg["resamples"], level=cfg["level"], seed=cfg["seed"]).as_dict()
    ci = cluster_bootstrap(lambda idx: float(b[idx].mean()), fams, resamples=cfg["resamples"],
                           level=cfg["level"], seed=cfg["seed"]).as_dict()
    return {"n": len(items), "sts_v1.1": round(float(a.mean()), 3), "sts_v1.2": round(float(b.mean()), 3),
            "sts_v1.2_ci": ci, "difference": round(float((b - a).mean()), 3), "ci": test["ci"],
            "p_value": test["p_value"],
            "items_fixed": [it.id for it, x, y in zip(items, a, b, strict=True) if y > x],
            "items_broken": [it.id for it, x, y in zip(items, a, b, strict=True) if y < x]}


def gold(items: Sequence[GoldItem]) -> dict[str, str]:
    return {it.id: it.intent for it in items}


def rules(items: Sequence[GoldItem]) -> dict[str, str]:
    return dict(zip([it.id for it in items], KeywordIntentV11().predict([norm(it.text) for it in items]), strict=True))


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs/eval.yaml").read_text(encoding="utf-8"))
    nat_ids = {i["id"] for i in json.loads((ROOT / "results/natural/evaluation.json").read_text(encoding="utf-8"))
               ["systems"]["S1"]["items"]}
    natural = [it for it in load_split(ROOT / "data/natural/manifest.json", "test", unlock_test=True) if it.id in nat_ids]
    test = list(load_split(ROOT / "data/splits/manifest.json", "test", unlock_test=True))
    m = read_manifest(ROOT / "data/splits/manifest.json")
    training = set(m["splits"]["train"]["families"]) | set(m["splits"]["dev"]["families"])
    v3 = [it for it in load_split(ROOT / "data/v3/manifest.json", "test", unlock_test=True)
          if it.family_id not in training]

    from athar.v3.evaluate import frozen_predictions  # verifies every v3 freeze first
    preds, _ = frozen_predictions()

    out: dict[str, Any] = {
        "analysis": "post hoc, exploratory; fixes written from the natural-set oracle failures; v3 held out",
        "core": V12.VERSION,
        "natural (development)": {"oracle": compare("oracle", [gold(natural)], natural, cfg),
                                  "S1 rules": compare("S1", [rules(natural)], natural, cfg)},
        "v3 (held-out)": {
            "oracle": compare("oracle", [gold(v3)], v3, cfg),
            "S1 rules": compare("S1", [rules(v3)], v3, cfg),
            "S3 + human data": compare("S3-human", [preds["S3-human"]], v3, cfg),
            "S6 MARBERTv2 + human data (5 seeds)": compare("S6-human", [preds[f"S6-human-s{s}"] for s in SEEDS], v3, cfg),
            "S3 drafted": compare("S3-drafted", [preds["S3-drafted"]], v3, cfg),
            "S6 MARBERTv2 drafted (5 seeds)": compare("S6-drafted", [preds[f"S6-drafted-s{s}"] for s in SEEDS], v3, cfg),
        },
        "locked test": {"oracle": compare("oracle", [gold(test)], test, cfg),
                        "S1 rules": compare("S1", [rules(test)], test, cfg)},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    for part in ("natural (development)", "v3 (held-out)", "locked test"):
        for name, r in out[part].items():
            print(f"{part:22s} {name:38s} {r['sts_v1.1']:.3f} -> {r['sts_v1.2']:.3f} "
                  f"({r['difference']:+.3f}, p={r['p_value']:.4f}; fixed {len(r['items_fixed'])}, broken {len(r['items_broken'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
