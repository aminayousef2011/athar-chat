"""Post hoc analyses of the locked test run, written after every system was scored (2026-10-20).

None of this changes a frozen system, the locked outputs or the pre-registered results. The script
reads the finished outputs (results/test_results.json, results/s4/...), the frozen models and the
data, and answers five questions raised in review of the results report:

1. ``template_sensitivity.csv``: are H1-H3 robust when test items that share a sentence frame are
   treated as dependent, not only items that share a scenario family?
2. ``frame_overlap.csv``: how many test items reuse a frame seen in train, and does that explain
   any system's score?
3. ``regularization_sweep.csv``: how S2/S3 depend on the regularization strength C, and which C a
   test-blind, family-grouped cross-validation on train+dev would have chosen.
4. ``extractor_fix.csv`` / ``extractor_fix_items.csv``: what the shared-core framing fix found in
   the error analysis (want-verbs and the «اعمللي» spelling) would change. The fix is applied in
   memory only; ``athar/`` is not edited.
5. ``lexicon_coverage.csv``: the share of in-scope requests that contain S1's gold-intent cue.

    uv run python scripts/posthoc_analysis.py --out results/posthoc

Every number here is post hoc and exploratory. The report labels it so.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from athar import extract as X
from athar import lexicon as L
from athar.data.split import load_split
from athar.eval.metrics import macro_f1, safe_task_success, score_all
from athar.eval.stats import cluster_bootstrap, holm, paired_permutation_test
from athar.intent.models import build
from athar.intent.rules import RX_ALARM, RX_REMINDER
from athar.intent.train import read_freeze
from athar.normalize import norm
from athar.pipeline import respond_with_intent
from athar.schema import INTENTS, GoldItem

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "splits" / "manifest.json"
S4 = {"jais2-8b": "7b-others", "jais3-3.7b": "7a-jais3", "jais3-7b": "7a-jais3", "nile-4b": "7b-others",
      "qwen3-8b": "7b-others"}
HYPOTHESES = (("H1", "S2", "S1"), ("H2", "S3", "S1"), ("H3", "S3", "S2"))
RX_PURPOSE = re.compile(r"\s(?:عشان|علشان)\s")


# --------------------------------------------------------------------------- stored outputs
def stored_items() -> dict[str, list[dict[str, Any]]]:
    """System name -> item scores of its locked test run (core, S4 primary, S7 prior, S8)."""
    out: dict[str, list[dict[str, Any]]] = {}
    core = json.loads((ROOT / "results" / "test_results.json").read_text(encoding="utf-8"))
    for n in ("S1", "S2", "S3", "oracle"):
        out[n] = core["systems"][n]["items"]
    for key, d in S4.items():
        rep = json.loads((ROOT / "results" / "s4" / d / "s4_results" / f"{key}.json").read_text(encoding="utf-8"))
        name = f"S4-{key}-6shot-prior"
        out[name] = rep["systems"][name]["items"]
    s4dir = ROOT / "results" / "s4" / "7cd-s7-s8" / "s4_results"
    for seed in (0, 1, 2):
        name = f"S7-jais2-qlora-s{seed}-prior"
        rep = json.loads((s4dir / f"s7_jais2_qlora_s{seed}_test.json").read_text(encoding="utf-8"))
        out[name] = rep["systems"][name]["items"]
    s8 = json.loads((s4dir / "s8_jais3-7b.json").read_text(encoding="utf-8"))
    for name, rep in s8["systems"].items():
        out[name] = rep["items"]
    return out


def core_config() -> dict[str, Any]:
    return dict(json.loads((ROOT / "results" / "test_results.json").read_text(encoding="utf-8"))["config"])


# --------------------------------------------------------------------------- sentence frames
def frame_purpose(text: str) -> str:
    """Frame A: the normalized request before its purpose clause («عشان/علشان ...»); a request
    without one keeps its first four tokens (request verb and time words come first)."""
    t = norm(text)
    head = RX_PURPOSE.split(t, maxsplit=1)[0]
    return head if head != t else " ".join(t.split()[:4])


def frame_masked(item: GoldItem) -> str:
    """Frame B: the normalized request with every run of content words (the words the shared
    extractor keeps as reminder content) replaced by ``X``. Only framing and time words remain."""
    ext = X.extract(item.text, item.ref_datetime, item.timezone)
    nt = ext.normalized
    out: list[str] = []
    for m in re.finditer(r"\w+", nt):
        removed = any(not (m.end() <= a or m.start() >= b) for a, b in ext.removed_spans)
        word = m.group() if removed else "X"
        if not (word == "X" and out and out[-1] == "X"):
            out.append(word)
    return " ".join(out)


def connected(ids: Sequence[str], keys: Sequence[Callable[[str], str]]) -> list[str]:
    """Union-find: items are joined when they share any of the ``keys``."""
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key in keys:
        groups: dict[str, list[str]] = defaultdict(list)
        for i in ids:
            groups[key(i)].append(i)
        for g in groups.values():
            for x in g[1:]:
                parent[find(x)] = find(g[0])
    return [find(i) for i in ids]


# --------------------------------------------------------------------------- 1. template sensitivity
def template_sensitivity(test: list[GoldItem], stored: dict[str, list[dict[str, Any]]],
                         cfg: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {g.id: g for g in test}
    ids = [it["id"] for it in stored["S1"]]
    fa = {i: frame_purpose(by_id[i].text) for i in ids}
    fb = {i: frame_masked(by_id[i]) for i in ids}
    clusterings = {
        "family (pre-registered)": [by_id[i].family_id for i in ids],
        "frame A (purpose-clause cut)": [fa[i] for i in ids],
        "frame B (content masked)": [fb[i] for i in ids],
        "family + frame A (connected)": connected(ids, [lambda i: by_id[i].family_id, fa.__getitem__]),
    }
    success = {s: [float(it["success"]) for it in stored[s]] for s in ("S1", "S2", "S3")}
    arrays = {s: np.asarray(v) for s, v in success.items()}
    rows = []
    for name, clusters in clusterings.items():
        sizes = sorted(np.unique(clusters, return_counts=True)[1].tolist(), reverse=True)
        tests = {h: paired_permutation_test(success[a], success[b], clusters, permutations=cfg["permutations"],
                                            resamples=cfg["resamples"], level=cfg["level"], seed=cfg["seed"])
                 for h, a, b in HYPOTHESES}
        adj = holm({h: t.p_value for h, t in tests.items()})
        cis = {s: cluster_bootstrap(lambda idx, v=arrays[s]: float(v[idx].mean()), clusters,
                                    resamples=cfg["resamples"], level=cfg["level"], seed=cfg["seed"])
               for s in ("S1", "S2", "S3")}
        for h, a, b in HYPOTHESES:
            t = tests[h]
            rows.append({"clustering": name, "clusters": len(sizes), "largest_clusters": " ".join(map(str, sizes[:3])),
                         "hypothesis": h, "comparison": f"{a} vs {b}", "difference": round(t.difference, 4),
                         "ci_low": round(t.ci.low, 4), "ci_high": round(t.ci.high, 4),
                         "p": round(t.p_value, 4), "p_holm": round(adj[h], 4),
                         **{f"{s}_ci": f"[{cis[s].low:.3f}, {cis[s].high:.3f}]" for s in ("S1", "S2", "S3")}})
    return rows


# --------------------------------------------------------------------------- 2. frame overlap
def frame_overlap(test: list[GoldItem], train: list[GoldItem], dev: list[GoldItem],
                  stored: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_frames = {frame_purpose(g.text) for g in train}
    author_frames = {frame_purpose(g.text) for g in train if g.source == "author"}
    dev_frames = {frame_purpose(g.text) for g in dev}
    seen = {g.id for g in test if frame_purpose(g.text) in train_frames}
    summary = {"test_items": len(test), "distinct_frames_A": len({frame_purpose(g.text) for g in test}),
               "distinct_frames_B": len({frame_masked(g) for g in test}),
               "frame_seen_in_train": len(seen),
               "frame_seen_in_author_train": sum(frame_purpose(g.text) in author_frames for g in test),
               "frame_seen_in_dev": sum(frame_purpose(g.text) in dev_frames for g in test)}
    rows = []
    for name, items in stored.items():
        row: dict[str, Any] = {"system": name}
        for part, keep in (("seen", True), ("unseen", False)):
            sub = [it for it in items if (it["id"] in seen) == keep]
            row[f"n_{part}"] = len(sub)
            row[f"sts_{part}"] = round(sum(it["success"] for it in sub) / len(sub), 3)
            row[f"intent_acc_{part}"] = round(sum(it["pred_intent"] == it["gold_intent"] for it in sub) / len(sub), 3)
        rows.append(row)
    return summary, rows


# --------------------------------------------------------------------------- 3. regularization
def _sts(items: Sequence[GoldItem], intents: Sequence[str], name: str) -> float:
    outs = [respond_with_intent(i, g.text, g.ref_datetime, g.timezone, name) for i, g in zip(intents, items, strict=True)]
    return safe_task_success(score_all(items, outs))


def regularization_sweep(train: list[GoldItem], dev: list[GoldItem], test: list[GoldItem]) -> list[dict[str, Any]]:
    """Vary only C around each frozen configuration. Grouped CV (5 folds by family, train+dev,
    selection metric intent macro-F1 as in configs/train.yaml) never sees test."""
    freeze = read_freeze(ROOT / "models")
    grid_c = (0.1, 0.3, 1, 3, 10, 30)
    xtr, ytr = [norm(g.text) for g in train], [g.intent for g in train]
    xdv, ydv = [norm(g.text) for g in dev], [g.intent for g in dev]
    xte, yte = [norm(g.text) for g in test], [g.intent for g in test]
    pool = train + dev
    xpool, ypool = np.array(xtr + xdv, dtype=object), np.array(ytr + ydv, dtype=object)
    groups = np.array([g.family_id for g in pool])
    folds = list(GroupKFold(n_splits=5).split(xpool, ypool, groups))
    rows = []
    for system in ("S2", "S3"):
        frozen = freeze["systems"][system]["params"]
        for c in grid_c:
            params = {**frozen, "C": c}
            model = build(system, params).fit(xtr, ytr)
            p_dv, p_te = list(model.predict(xdv)), list(model.predict(xte))
            cv = []
            for tr_idx, va_idx in folds:
                m = build(system, params).fit(list(xpool[tr_idx]), list(ypool[tr_idx]))
                cv.append(macro_f1(list(ypool[va_idx]), list(m.predict(list(xpool[va_idx]))), INTENTS))
            rows.append({"system": system, "C": c, "frozen_choice": c == frozen["C"],
                         "train_acc": round(float(np.mean(model.predict(xtr) == np.array(ytr))), 3),
                         "dev_intent_f1": round(macro_f1(ydv, p_dv, INTENTS), 3),
                         "dev_acc": round(float(np.mean(np.array(p_dv) == np.array(ydv))), 3),
                         "groupcv_intent_f1": round(float(np.mean(cv)), 3),
                         "test_intent_acc": round(float(np.mean(np.array(p_te) == np.array(yte))), 3),
                         "test_sts": round(_sts(test, p_te, system), 3)})
        sys_rows = [r for r in rows if r["system"] == system]
        best = max(sys_rows, key=lambda r: (r["groupcv_intent_f1"], -r["C"]))  # ties: smaller C
        for r in sys_rows:
            r["groupcv_choice"] = r is best
    return rows


# --------------------------------------------------------------------------- 4. extractor fix
#: request phrasings the frozen framing lexicon misses (found in the post hoc error analysis)
WANT_VERBS = ("عايز", "عاوز", "عايزه", "عايزة", "محتاج", "محتاجه", "محتاجة")
SPELLINGS = ("اعمللي", "اعملّي")


@contextmanager
def framing_fix() -> Iterator[None]:
    """Count «عايز/محتاج + تذكير/منبه» and «اعمللي + تذكير/منبه» as request framing, in memory."""
    old = X.RX_FRAMING
    setup = L.SETUP_VERBS + L._n([*SPELLINGS, *WANT_VERBS])
    b, e = r"(?<!\w)", r"(?!\w)"
    X.RX_FRAMING = re.compile(
        rf"{b}(?:(?:و\s?)?{L.alt(L.TRIGGERS)}"
        rf"|{L.alt(setup)}(?:\s+(?:لي|ليا|for\s+me))?\s+{L.alt(L.TOOL_NOUNS)}"
        rf"|{L.alt(L.TOOL_NOUNS)}|{L.alt(L.FILLERS)}|{L.alt(L.HEDGES)}){e}")
    try:
        yield
    finally:
        X.RX_FRAMING = old


def extractor_fix(splits: dict[str, list[GoldItem]], stored: dict[str, list[dict[str, Any]]]
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test = splits["test"]
    pipeline = [n for n in stored if not n.startswith("S8")]

    def run_all() -> dict[str, list[bool]]:
        out: dict[str, list[bool]] = {}
        for name in pipeline:
            intents = [g.intent for g in test] if name == "oracle" else [it["pred_intent"] for it in stored[name]]
            outs = [respond_with_intent(i, g.text, g.ref_datetime, g.timezone, name) if i in INTENTS else None
                    for i, g in zip(intents, test, strict=True)]
            out[name] = [s.success for s in score_all(test, outs)]
        for split in ("train", "dev"):  # no regression on the data the core was developed on
            items = splits[split]
            outs = [respond_with_intent(g.intent, g.text, g.ref_datetime, g.timezone) for g in items]
            out[f"oracle ({split})"] = [s.success for s in score_all(items, outs)]
        return out

    before = run_all()
    for name in pipeline:  # the rebuild must reproduce the locked run before anything is changed
        if before[name] != [it["success"] for it in stored[name]]:
            raise SystemExit(f"rebuild of {name} does not reproduce the locked run")
    with framing_fix():
        after = run_all()
    rows = [{"system": n, "n": len(before[n]), "sts_locked": round(float(np.mean(before[n])), 3),
             "sts_with_fix": round(float(np.mean(after[n])), 3),
             "fixed": sum(a and not b for a, b in zip(after[n], before[n], strict=True)),
             "broken": sum(b and not a for a, b in zip(after[n], before[n], strict=True))} for n in before]
    items = [{"id": g.id, "text": g.text, "oracle_locked": b, "oracle_with_fix": a}
             for g, b, a in zip(test, before["oracle"], after["oracle"], strict=True) if a != b]
    return rows, items


# --------------------------------------------------------------------------- 5. lexicon coverage
def lexicon_coverage(splits: dict[str, list[GoldItem]]) -> list[dict[str, Any]]:
    rows = []
    for name, items in splits.items():
        scope = [g for g in items if g.intent != "unsupported"]
        hit = sum(bool((RX_ALARM if g.intent == "create_alarm" else RX_REMINDER).search(norm(g.text))) for g in scope)
        rows.append({"split": name, "in_scope": len(scope), "contains_gold_intent_cue": hit,
                     "share": round(hit / len(scope), 3)})
    return rows


# --------------------------------------------------------------------------- main
def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "results" / "posthoc"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {s: list(load_split(MANIFEST, s, unlock_test=(s == "test"))) for s in ("train", "dev", "test")}
    stored = stored_items()
    order = {it["id"]: k for k, it in enumerate(stored["S1"])}
    splits["test"].sort(key=lambda g: order[g.id])

    sens = template_sensitivity(splits["test"], stored, core_config())
    write_csv(out / "template_sensitivity.csv", sens)
    overlap_summary, overlap = frame_overlap(splits["test"], splits["train"], splits["dev"], stored)
    write_csv(out / "frame_overlap.csv", overlap)
    sweep = regularization_sweep(splits["train"], splits["dev"], splits["test"])
    write_csv(out / "regularization_sweep.csv", sweep)
    fix, fix_items = extractor_fix(splits, stored)
    write_csv(out / "extractor_fix.csv", fix)
    write_csv(out / "extractor_fix_items.csv", fix_items)
    coverage = lexicon_coverage(splits)
    write_csv(out / "lexicon_coverage.csv", coverage)
    (out / "frame_overlap_summary.json").write_text(json.dumps(overlap_summary, indent=2), encoding="utf-8")

    for r in sens:
        print(f"{r['clustering']:30} ({r['clusters']:3} clusters) {r['hypothesis']} {r['difference']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p {r['p']:.4f} Holm {r['p_holm']:.4f}")
    print("frames:", overlap_summary)
    for r in sweep:
        print(f"{r['system']} C={r['C']:<4} train {r['train_acc']:.3f} dev F1 {r['dev_intent_f1']:.3f} "
              f"groupCV F1 {r['groupcv_intent_f1']:.3f} test acc {r['test_intent_acc']:.3f} STS {r['test_sts']:.3f}"
              f"{'  <- frozen' if r['frozen_choice'] else ''}{'  <- grouped CV' if r['groupcv_choice'] else ''}")
    for r in fix:
        print(f"{r['system']:30} STS {r['sts_locked']:.3f} -> {r['sts_with_fix']:.3f} "
              f"(+{r['fixed']} / -{r['broken']})")
    print("coverage:", coverage)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
