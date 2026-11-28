"""Post hoc, exploratory: a learned reminder-text extractor in place of the hand-written lists.

The shared extractor finds the reminder text by deleting request framing, fillers and time
expressions with hand-written word lists (``athar.v2.core``). Here a token tagger learns that
choice from labelled messages: every word of the normalized message is labelled *content* when it
belongs to the gold reminder text, and a logistic-regression classifier over window features
(the word, its neighbours, prefixes/suffixes, position, whether core v1.1's time parser covers it)
predicts the label. Dates and times are still resolved by core v1.1 (that part is computation, not
classification), the policy and the scoring are unchanged, and gold and predicted texts are
compared with the same ``canonical_text`` as every registered result.

Two variants: **learned** (no framing or filler list at all: the tagged words are the reminder
text, and an empty tagging means the text is missing) and **learned + rules** (core v1.1's framing
match is one more feature, and the output is cleaned with core v1.1's framing, filler and
vague-content lists, as the rule extractor's output is). Scoring canonicalizes gold and predicted
texts alike for every system. Fixed settings (C = 1, threshold 0.5), chosen
before any evaluation set was scored; nothing is tuned on an evaluation set.

Training data -> evaluation sets (all with the gold intent, i.e. the oracle):

* drafted + author (train + dev, 338 items) -> natural set (238) and v3 (236): the registered
  training data;
* natural set (247 human messages) -> v3: in-distribution human data (they share v3's scenario
  cards, as in Finding 2 of the report).

Each comparison is paired item by item against oracle v1.1 (the rule extractor), with the
family-level bootstrap and permutation test of the registered analyses (unadjusted p,
descriptive). No registered result changes.

    uv run python scripts/posthoc_learned_text.py   # writes results/posthoc/learned_text.json

Only aggregates are written (the v3 texts are private).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athar import extract as X  # noqa: E402
from athar.data.split import load_split, read_manifest  # noqa: E402
from athar.eval.metrics import score_all  # noqa: E402
from athar.eval.stats import paired_permutation_test  # noqa: E402
from athar.normalize import normalize  # noqa: E402
from athar.policy import decide  # noqa: E402
from athar.respond import build_response  # noqa: E402
from athar.schema import GoldItem, Response  # noqa: E402
from athar.v2 import core as V11  # noqa: E402

OUT = ROOT / "results/posthoc/learned_text.json"
RX_TOKEN = re.compile(r"\w+")
C, THRESHOLD = 1.0, 0.5


# ------------------------------------------------------------------ tokens, labels, features ----

def tokens(item: GoldItem) -> tuple[list[tuple[int, int, str]], set[int], set[int], str]:
    """Word tokens of the normalized message, the indices core v1.1's time parser and framing
    rule cover, and the normalized text."""
    nt = normalize(item.text).text
    toks = [(m.start(), m.end(), m.group()) for m in RX_TOKEN.finditer(nt)]
    spans = V11.parse_temporal(nt, item.ref_datetime, item.timezone).spans
    framing = [m.span() for m in V11.RX_FRAMING.finditer(nt)]

    def covered(s: int, e: int, sp: Sequence[tuple[int, int]]) -> bool:
        return any(not (e <= a or s >= b) for a, b in sp)

    return (toks, {i for i, (s, e, _) in enumerate(toks) if covered(s, e, spans)},
            {i for i, (s, e, _) in enumerate(toks) if covered(s, e, framing)}, nt)


def _matches(tok: str, gold: str) -> bool:
    return tok == gold or (tok.startswith("ب") and tok[1:] == gold)  # canonical_text drops a leading «ب»


def labels(item: GoldItem, toks: list[tuple[int, int, str]]) -> list[int] | None:
    """1 for words of the gold reminder text, else 0; None when the item says nothing about it."""
    if item.intent != "create_reminder":
        return None
    gold = X.canonical_text(item.args.reminder_text) if item.args.reminder_text else None
    if gold is None:
        return [0] * len(toks) if "reminder_text" in item.clarify_targets else None
    want, out, j = gold.split(), [0] * len(toks), 0
    for i, (_, _, w) in enumerate(toks):  # gold words in order, as a subsequence of the message
        if j < len(want) and _matches(w, want[j]):
            out[i], j = 1, j + 1
    return out if j == len(want) else None


def features(toks: list[tuple[int, int, str]], timed: set[int], framed: set[int], rule: bool) -> list[dict[str, Any]]:
    words = [w for _, _, w in toks]
    n = len(words)
    rows = []
    for i, w in enumerate(words):
        f: dict[str, Any] = {"w=" + w: 1, "p2=" + w[:2]: 1, "p3=" + w[:3]: 1, "s2=" + w[-2:]: 1,
                             "s3=" + w[-3:]: 1, "latin": int(w.isascii()), "digit": int(w.isdigit()),
                             "first": int(i == 0), "last": int(i == n - 1), "pos": i / max(n - 1, 1),
                             "time": int(i in timed), "bias": 1}
        for d in (-2, -1, 1, 2):
            k = i + d
            f[f"w{d:+d}=" + (words[k] if 0 <= k < n else "<pad>")] = 1
            f[f"time{d:+d}"] = int(k in timed)
        f["after_time"] = int(any(k < i for k in timed))
        if rule:
            f["framing"] = int(i in framed)
            f["framing-1"] = int(i - 1 in framed)
        rows.append(f)
    return rows


class TextTagger:
    def __init__(self, rule: bool) -> None:
        self.rule = rule
        self.vec = DictVectorizer()
        self.clf = LogisticRegression(C=C, max_iter=5000)

    def fit(self, items: Sequence[GoldItem]) -> dict[str, int]:
        X_, y, used = [], [], 0
        for it in items:
            toks, timed, framed, _ = tokens(it)
            lab = labels(it, toks)
            if lab is None:
                continue
            used += 1
            X_ += features(toks, timed, framed, self.rule)
            y += lab
        self.clf.fit(self.vec.fit_transform(X_), y)
        return {"messages": used, "words": len(y), "content_words": int(sum(y))}

    def text(self, item: GoldItem) -> str | None:
        toks, timed, framed, _ = tokens(item)
        if not toks:
            return None
        p = self.clf.predict_proba(self.vec.transform(features(toks, timed, framed, self.rule)))[:, 1]
        kept = [w for (_, _, w), q in zip(toks, p, strict=True) if q >= THRESHOLD]
        if not self.rule:   # purely learned: the tagged words, no word list (scoring canonicalizes both sides)
            return " ".join(kept) or None
        out = X.canonical_text(" ".join(kept)) if kept else None   # hybrid: core v1.1's lists clean the output
        if out in X._VAGUE_CONTENT:
            return None
        return out


# ------------------------------------------------------------------ pipeline and scoring --------

def respond(item: GoldItem, tagger: TextTagger | None) -> Response:
    """Oracle (gold intent) through core v1.1, with the reminder text from ``tagger`` if given."""
    ext = V11.extract(item.text, item.ref_datetime, item.timezone)
    if tagger is not None:
        t = tagger.text(item)
        ext = X.Extraction(ext.temporal, t, t, ext.normalized, ext.removed_spans)
    return build_response("oracle", decide(item.intent, ext), ext)


def text_accuracy(items: Sequence[GoldItem], tagger: TextTagger | None) -> dict[str, Any]:
    """Share of gold-ACT reminders whose canonical reminder text is right (time not required)."""
    act = [it for it in items if it.route == "ACT" and it.intent == "create_reminder"]
    ok = 0
    for it in act:
        pred = (tagger.text(it) if tagger is not None
                else V11.extract(it.text, it.ref_datetime, it.timezone).reminder_text)
        ok += X.canonical_text(pred) in {X.canonical_text(t) for t in
                                        [it.args.reminder_text, *it.args.reminder_text_alts] if t} if pred else 0
    return {"n": len(act), "correct": ok, "accuracy": round(ok / len(act), 3)}


def compare(items: Sequence[GoldItem], tagger: TextTagger, cfg: dict[str, Any]) -> dict[str, Any]:
    a = np.array([float(s.success) for s in score_all(items, [respond(it, None) for it in items])])
    b = np.array([float(s.success) for s in score_all(items, [respond(it, tagger) for it in items])])
    fams = [it.family_id for it in items]
    test = paired_permutation_test(list(b), list(a), fams, permutations=cfg["permutations"],
                                   resamples=cfg["resamples"], level=cfg["level"], seed=cfg["seed"]).as_dict()
    return {"n": len(items), "oracle_rules_sts": round(float(a.mean()), 3),
            "oracle_learned_sts": round(float(b.mean()), 3), "difference": round(float((b - a).mean()), 3),
            "ci": test["ci"], "p_value": test["p_value"],
            "items_gained": int(((b - a) > 0).sum()), "items_lost": int(((b - a) < 0).sum()),
            "text_rules": text_accuracy(items, None), "text_learned": text_accuracy(items, tagger)}


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs/eval.yaml").read_text(encoding="utf-8"))
    m = read_manifest(ROOT / "data/splits/manifest.json")
    fitting = (list(load_split(ROOT / "data/splits/manifest.json", "train"))
               + list(load_split(ROOT / "data/splits/manifest.json", "dev")))
    training_fams = set(m["splits"]["train"]["families"]) | set(m["splits"]["dev"]["families"])
    nat_ids = {i["id"] for i in json.loads((ROOT / "results/natural/evaluation.json").read_text(encoding="utf-8"))
               ["systems"]["S1"]["items"]}
    natural_all = list(load_split(ROOT / "data/natural/manifest.json", "test", unlock_test=True))
    natural = [it for it in natural_all if it.id in nat_ids]
    v3 = [it for it in load_split(ROOT / "data/v3/manifest.json", "test", unlock_test=True)
          if it.family_id not in training_fams]
    nat_fams = {it.family_id for it in natural_all}

    out: dict[str, Any] = {
        "analysis": "post hoc, exploratory; learned reminder-text tagger vs the rule extractor (core v1.1), "
                    "gold intents; fixed settings, nothing tuned on an evaluation set",
        "settings": {"classifier": "logistic regression over window features", "C": C, "threshold": THRESHOLD},
        "v3_families_shared_with_natural": len({it.family_id for it in v3} & nat_fams),
        "runs": {},
    }
    for rule in (False, True):
        name = "learned + rules" if rule else "learned"
        drafted = TextTagger(rule)
        fit_d = drafted.fit(fitting)
        human = TextTagger(rule)
        fit_h = human.fit(natural_all)
        out["runs"][name] = {
            "drafted + author -> natural": {"training": fit_d, **compare(natural, drafted, cfg)},
            "drafted + author -> v3": {"training": fit_d, **compare(v3, drafted, cfg)},
            "natural (human) -> v3": {"training": fit_h, **compare(v3, human, cfg)},
        }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    for name, runs in out["runs"].items():
        for k, r in runs.items():
            print(f"{name:24s} {k:30s} text {r['text_rules']['accuracy']:.3f} -> {r['text_learned']['accuracy']:.3f} "
                  f"(n={r['text_rules']['n']}); oracle STS {r['oracle_rules_sts']:.3f} -> {r['oracle_learned_sts']:.3f} "
                  f"({r['difference']:+.3f}, p={r['p_value']:.4f}; +{r['items_gained']}/-{r['items_lost']}); "
                  f"train msgs {r['training']['messages']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
