"""Post hoc (2026-11-05): the encoder baselines S5/S6 and training on human-written data.

Two questions raised in review, answered without changing any frozen system, split or label:

1. ``encoders``: S5 (frozen multilingual-e5-small + logistic regression) and S6 (fine-tuned
   MARBERTv2, 5 seeds) were implemented and their recipes fixed in configs/train.yaml before the
   test, but never fit on the final split. Here they are fit exactly as configured (S5: C on dev;
   S6: fixed settings, no search) on the frozen train split, and scored through the shared core on
   the locked test (core v1, as S2/S3) and on the 238 confirmatory contributor items (core v1 and
   core v1.1).

2. ``human``: does human-written training data close the gap between the classifiers and the rules?
   Leave-one-writer-out over the 17 contributors of the natural set. For each held-out writer, the
   training pool is train+dev plus the other writers' messages, minus every message on a scenario
   family the held-out writer wrote (the other writer of a block wrote the same cards). Three
   pools are compared on exactly the same predictions slots: drafted+author only (train+dev),
   drafted+author+contributors, and contributors only. Hyper-parameters are fixed in advance (the
   frozen S2-gcv / S3-gcv settings; S5 C = 1), so the pool is the only thing that changes. Every
   confirmatory item is predicted exactly once, by the fold that held out its writer.

Everything here is post hoc and exploratory: it is reported in the report's section 9 and
docs/posthoc_analyses.md and does not touch H1-H3 or N1-N3.

    uv run --extra neural python scripts/posthoc_encoders_human.py encoders
    uv run --extra neural python scripts/posthoc_encoders_human.py human
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from athar.data.split import load_split, read_manifest  # noqa: E402
from athar.eval import harness as H  # noqa: E402
from athar.eval.metrics import ItemScore, aggregate, score_all  # noqa: E402
from athar.eval.stats import (  # noqa: E402
    cluster_bootstrap,
    holm,
    mcnemar_exact,
    paired_permutation_test,
)
from athar.intent import neural as N  # noqa: E402
from athar.intent.models import SklearnIntent, build  # noqa: E402
from athar.intent.rules import KeywordIntent  # noqa: E402
from athar.intent.train import load_recognizer, scores  # noqa: E402
from athar.normalize import norm  # noqa: E402
from athar.schema import GoldItem  # noqa: E402
from athar.v2 import gcv  # noqa: E402
from athar.v2.core import CoreV11System  # noqa: E402
from athar.v2.natural import writer_of  # noqa: E402
from athar.v2.rules import KeywordIntentV11  # noqa: E402

OUT = REPO / "results" / "posthoc"
CFG = yaml.safe_load((REPO / "configs" / "train.yaml").read_text(encoding="utf-8"))
EVAL = yaml.safe_load((REPO / "configs" / "eval.yaml").read_text(encoding="utf-8"))
MAIN = REPO / "data" / "splits" / "manifest.json"
NATURAL = REPO / "data" / "natural" / "manifest.json"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Fixed:
    """Intent predictions computed elsewhere, served in item order (for the v1/v1.1 cores)."""

    def __init__(self, by_text: dict[str, str], name: str) -> None:
        self.by_text, self.name = by_text, name

    def predict(self, normalized_texts: Sequence[str]) -> list[str]:
        return [self.by_text[t] for t in normalized_texts]


def splits() -> tuple[list[GoldItem], list[GoldItem], list[GoldItem], list[GoldItem], list[GoldItem]]:
    train, dev = list(load_split(MAIN, "train")), list(load_split(MAIN, "dev"))
    test = list(load_split(MAIN, "test", unlock_test=True))
    natural = list(load_split(NATURAL, "test", unlock_test=True))
    m = read_manifest(MAIN)
    seen = set(m["splits"]["train"]["families"]) | set(m["splits"]["dev"]["families"])
    conf = [it for it in natural if it.family_id not in seen]
    return train, dev, test, natural, conf


def success(system: Any, items: Sequence[GoldItem]) -> list[ItemScore]:
    return score_all(items, H.run(system, items))


def summarize(sc: Sequence[Any]) -> dict[str, Any]:
    v = np.array([float(s.success) for s in sc])
    agg = aggregate(sc)
    ci = cluster_bootstrap(lambda idx: float(v[idx].mean()), [s.family_id for s in sc],
                           resamples=EVAL["resamples"], level=EVAL["level"], seed=EVAL["seed"]).as_dict()
    by_route = {r: round(float(np.mean([s.success for s in sc if s.gold_route == r])), 3)
                for r in ("ACT", "CLARIFY", "ABSTAIN")}
    return {"sts": round(float(v.mean()), 3), "ci": ci, "intent_accuracy": agg.get("intent_accuracy"),
            "intent_macro_f1": agg.get("intent_macro_f1"), "unsafe_execution": agg.get("unsafe_execution_rate"),
            "by_route": by_route}


def paired(a: Sequence[Any], b: Sequence[Any]) -> dict[str, Any]:
    xa, xb = [float(s.success) for s in a], [float(s.success) for s in b]
    t = paired_permutation_test(xa, xb, [s.family_id for s in a], permutations=EVAL["permutations"],
                                resamples=EVAL["resamples"], level=EVAL["level"], seed=EVAL["seed"]).as_dict()
    t["mcnemar"] = mcnemar_exact([bool(x) for x in xa], [bool(x) for x in xb])
    return t


def mean_success(runs: Sequence[Sequence[Any]]) -> list[float]:
    return [float(x) for x in np.mean([[float(s.success) for s in r] for r in runs], axis=0)]


# --------------------------------------------------------------------------- encoders
def encoders(seeds: Sequence[int]) -> dict[str, Any]:
    train, dev, test, _, conf = splits()
    xtr, ytr = [norm(it.text) for it in train], [it.intent for it in train]
    xdv, ydv = [norm(it.text) for it in dev], [it.intent for it in dev]
    texts = sorted({norm(it.text) for it in (*test, *conf)})
    out: dict[str, Any] = {"note": "post hoc 2026-11-05; recipes from configs/train.yaml, fit on the frozen train split",
                           "systems": {}}

    def record(name: str, preds: dict[str, str], extra: dict[str, Any]) -> dict[str, Any]:
        rec = Fixed(preds, name)
        res = {"locked_test_core_v1": success(rec, test), "natural_core_v1": success(rec, conf),
               "natural_core_v1.1": success(CoreV11System(rec, name), conf)}
        out["systems"][name] = {**extra, **{k: summarize(v) for k, v in res.items()}}
        log(f"{name}: test {out['systems'][name]['locked_test_core_v1']['sts']}  "
            f"natural {out['systems'][name]['natural_core_v1']['sts']} / v1.1 {out['systems'][name]['natural_core_v1.1']['sts']}")
        return res

    log("S5: multilingual-e5-small embeddings")
    enc = N.load_encoder(CFG["S5"]["model"], CFG["S5"]["revision"])
    s5, params, table = N.fit_s5(enc, xtr, ytr, xdv, ydv, CFG["S5"], scores)
    s5_res = record("S5", dict(zip(texts, s5.predict(texts), strict=True)),
                    {"params": params, "dev_grid": table})

    # Each seed's predictions are cached as soon as it finishes, so an interrupted run resumes at
    # the next seed instead of starting over. Training is seeded but not bit-exact on CPU: a rerun
    # of a seed can differ by about one item (seed 0: test 0.800 vs 0.792), so never mix cached and
    # retrained seeds silently; the report cites results/posthoc/encoders_run1.json.
    cache = OUT / "encoders_cache"
    cache.mkdir(parents=True, exist_ok=True)
    s6_res = []
    for seed in seeds:
        path = cache / f"S6-s{seed}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            log(f"S6 seed {seed}: cached predictions from {path.relative_to(REPO)}")
            s6_res.append(record(f"S6-s{seed}", saved["predictions"], saved["extra"]))
            continue
        log(f"S6 seed {seed}: fine-tuning MARBERTv2 on {len(xtr)} items")
        net, tok = N.load_pretrained(CFG["S6"]["model"], CFG["S6"]["revision"])
        unk = N.unk_rate(tok, xtr)
        t0 = time.time()
        model = N.fine_tune(net, tok, xtr, ytr, CFG["S6"], seed, log=log)
        dev_f1 = scores(ydv, model.predict(xdv))
        preds = dict(zip(texts, model.predict(texts), strict=True))
        extra = {"seed": seed, "unk_rate": round(unk, 4), "train_seconds": round(time.time() - t0), "dev": dev_f1}
        path.write_text(json.dumps({"predictions": preds, "extra": extra}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        s6_res.append(record(f"S6-s{seed}", preds, extra))

    # seed means and paired tests against the systems the report compares with
    refs = {"S1": KeywordIntent(), "S2": load_recognizer("S2", REPO / "models"),
            "S3": load_recognizer("S3", REPO / "models")}
    ref_res = {n: {"locked_test_core_v1": success(r, test), "natural_core_v1": success(r, conf)} for n, r in refs.items()}
    ref_res["S1-v1.1"] = {"natural_core_v1.1": success(CoreV11System(KeywordIntentV11(), "S1-v1.1"), conf)}
    out["s6_seed_mean"] = {k: round(float(np.mean(mean_success([r[k] for r in s6_res]))), 3) for k in s6_res[0]}
    out["s6_seed_sd"] = {k: round(float(np.std([np.mean([s.success for s in r[k]]) for r in s6_res], ddof=1)), 3)
                         for k in s6_res[0]} if len(s6_res) > 1 else {}
    out["references"] = {n: {k: summarize(v) for k, v in r.items()} for n, r in ref_res.items()}
    tests: dict[str, Any] = {}
    for setting in ("locked_test_core_v1", "natural_core_v1"):
        for ref in ("S1", "S2"):
            tests[f"S5 vs {ref} ({setting})"] = paired(s5_res[setting], ref_res[ref][setting])
            fams = [s.family_id for s in s6_res[0][setting]]
            t = paired_permutation_test(mean_success([r[setting] for r in s6_res]),
                                        [float(s.success) for s in ref_res[ref][setting]], fams,
                                        permutations=EVAL["permutations"], resamples=EVAL["resamples"],
                                        level=EVAL["level"], seed=EVAL["seed"]).as_dict()
            tests[f"S6 seed mean vs {ref} ({setting})"] = t
    out["tests"] = tests
    return out


# --------------------------------------------------------------------------- human data
def human(include_s6: bool) -> dict[str, Any]:
    train, dev, _, natural, conf = splits()
    base = train + dev
    writers = sorted({writer_of(it.id) for it in conf})
    fz = gcv.read_freeze(REPO / "models_v2")
    params = {"S2": fz["systems"]["S2-gcv"]["params"], "S3": fz["systems"]["S3-gcv"]["params"]}
    log(f"{len(conf)} confirmatory items, {len(writers)} writers; fixed params {params}")
    enc = N.load_encoder(CFG["S5"]["model"], CFG["S5"]["revision"])
    emb_cache: dict[str, np.ndarray] = {}

    def emb(texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in emb_cache]
        if missing:
            for t, v in zip(missing, N.embed(enc, missing, CFG["S5"]["prefix"]), strict=True):
                emb_cache[t] = v
        return np.stack([emb_cache[t] for t in texts])

    def fit_predict(kind: str, pool: Sequence[GoldItem], xte: list[str], seed: int = 0) -> list[str]:
        x, y = [norm(it.text) for it in pool], [it.intent for it in pool]
        if kind in ("S2", "S3"):
            return SklearnIntent(kind, build(kind, params[kind], CFG["seed"]).fit(x, y)).predict(xte)
        if kind == "S5":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=0).fit(emb(x), y)
            return [str(p) for p in clf.predict(emb(xte))]
        net, tok = N.load_pretrained(CFG["S6"]["model"], CFG["S6"]["revision"])
        return N.fine_tune(net, tok, x, y, CFG["S6"], seed, log=lambda m: None).predict(xte)

    kinds = ["S2", "S3", "S5"] + (["S6"] if include_s6 else [])
    pools = ("drafted+author", "drafted+author+contributors", "contributors only")
    preds: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)   # (kind, pool) -> item id -> intent
    fold_sizes = {}
    for w in writers:
        held = [it for it in conf if writer_of(it.id) == w]
        held_fams = {it.family_id for it in held}
        others = [it for it in natural if writer_of(it.id) != w and it.family_id not in held_fams]
        fold_sizes[w] = {"held_out": len(held), "contributor_training": len(others)}
        xte = [norm(it.text) for it in held]
        pool_items = {"drafted+author": base, "drafted+author+contributors": base + others, "contributors only": others}
        for kind in kinds:
            for p in pools:
                if kind == "S6" and p == "contributors only":
                    continue
                if kind != "S6" and p == "drafted+author" and (kind, p) in preds and all(
                        it.id in preds[(kind, p)] for it in held):
                    continue
                for it, y in zip(held, fit_predict(kind, pool_items[p], xte), strict=True):
                    preds[(kind, p)][it.id] = y
        log(f"writer {w}: {len(held)} held out, {len(others)} contributor messages in training")

    out: dict[str, Any] = {"note": "post hoc 2026-11-05; leave-one-writer-out, held-out writer's families removed "
                                   "from training; hyper-parameters fixed (S2-gcv, S3-gcv, S5 C=1, S6 config)",
                           "params": params, "folds": fold_sizes, "systems": {}}
    res: dict[str, list[Any]] = {}
    for (kind, p), by_id in preds.items():
        name = f"{kind} | {p}"
        assert set(by_id) == {it.id for it in conf}, f"{name}: not every item predicted exactly once"
        rec = _ById(by_id, name)
        res[name] = score_all(conf, [_respond(rec, it) for it in conf])
        out["systems"][name] = summarize(res[name])
        log(f"{name:40} STS {out['systems'][name]['sts']:.3f}  intent acc {out['systems'][name]['intent_accuracy']}")
    res["S1-v1.1"] = success(CoreV11System(KeywordIntentV11(), "S1-v1.1"), conf)
    res["oracle-v1.1"] = success(CoreV11System(None, "oracle-v1.1"), conf)
    for n in ("S1-v1.1", "oracle-v1.1"):
        out["systems"][n] = summarize(res[n])
    tests = {}
    for kind in kinds:
        tests[f"{kind}: +contributors vs drafted+author"] = paired(res[f"{kind} | drafted+author+contributors"],
                                                                   res[f"{kind} | drafted+author"])
        tests[f"{kind} +contributors vs S1-v1.1"] = paired(res[f"{kind} | drafted+author+contributors"], res["S1-v1.1"])
    adj = holm({k: v["p_value"] for k, v in tests.items()})
    for k, v in tests.items():
        v["p_holm_within_this_analysis"] = adj[k]
    out["tests"] = tests
    # writer-level view: how often does each pool beat S1 v1.1
    out["per_writer"] = {}
    for name in out["systems"]:
        per = defaultdict(list)
        for s, it in zip(res[name], conf, strict=True):
            per[writer_of(it.id)].append(float(s.success))
        out["per_writer"][name] = {w: round(float(np.mean(v)), 3) for w, v in sorted(per.items())}
    return out


class _ById:
    def __init__(self, by_id: dict[str, str], name: str) -> None:
        self.by_id, self.name = by_id, name


def _respond(rec: _ById, it: GoldItem) -> Any:
    from athar.v2.core import respond_with_intent
    return respond_with_intent(rec.by_id[it.id], it.text, it.ref_datetime, it.timezone, rec.name)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("part", choices=["encoders", "human"])
    p.add_argument("--seeds", type=int, nargs="+", default=CFG["S6"]["seeds"])
    p.add_argument("--with-s6", action="store_true", help="human: also fine-tune S6 per fold (slow on CPU)")
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    N.use_system_certificates()
    t0 = time.time()
    result = encoders(a.seeds) if a.part == "encoders" else human(a.with_s6)
    result["runtime_seconds"] = round(time.time() - t0)
    path = OUT / ("encoders.json" if a.part == "encoders" else "human_training.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
