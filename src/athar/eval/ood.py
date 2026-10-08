"""Out-of-distribution intent check on native-written Egyptian Arabic (protocol section 6).

    athar-ood --models models --systems S1 S2 S3

Data: the 200-item test split of ``Alhasan/egyptian-nlu`` (CC BY 4.0, derived from Amazon
MASSIVE): voice-assistant commands written and annotated by hand by a native Egyptian speaker.
It is pinned to one revision and checked by SHA-256. Only 4 items set an alarm and none is a
plain one-off reminder, so the check mainly measures whether a system **abstains on
out-of-scope Egyptian requests** (unsupported recall, n = 192) rather than its F1.

Mapping: ``alarm_set`` -> create_alarm; ``calendar_set`` excluded (it mixes reminders with
calendar events, as for MASSIVE); every other intent -> unsupported. Intent only: the dataset
has no reference time, so no datetime gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from athar.normalize import norm

REPO_ID = "Alhasan/egyptian-nlu"
REVISION = "5094a703965e72686fe3ca43543d3bae342fd2c0"
FILE = "egy_test.jsonl"
SHA256 = "01a12d0b566f0cfb02aae2b2ffc9c3fe1f87f5e8df0522cb8817733855cef2a0"
URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{FILE}"
CACHE = Path("data/external/egyptian-nlu") / FILE
ATTRIBUTION = ("Egyptian NLU test set by Alhasan Abdellatif (huggingface.co/datasets/Alhasan/egyptian-nlu), "
               "CC BY 4.0, derived from Amazon MASSIVE (CC BY 4.0)")
EXCLUDED = frozenset({"calendar_set"})
LABELS = ("create_alarm", "unsupported")


def map_intent(massive_intent: str) -> str | None:
    if massive_intent in EXCLUDED:
        return None
    return "create_alarm" if massive_intent == "alarm_set" else "unsupported"


def load(path: str | Path = CACHE, download: bool = True) -> list[dict[str, str]]:
    """``[{id, text, source_intent, gold}]`` for the human-written items, calendar_set excluded."""
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:  # noqa: S310 (fixed https URL)
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path}: SHA-256 {digest[:12]} does not match the pinned file {SHA256[:12]}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("source") != "human":
            continue
        gold = map_intent(r["intent"])
        if gold is not None:
            rows.append({"id": r["id"], "text": r["utt"], "source_intent": r["intent"], "gold": gold})
    return rows


def score(gold: Sequence[str], pred: Sequence[str]) -> dict[str, Any]:
    """Recall per gold label (a supported prediction other than alarm still counts as a miss),
    and macro-F1 over the two gold labels."""
    from sklearn.metrics import f1_score

    out: dict[str, Any] = {"n": len(gold)}
    for lab in LABELS:
        idx = [i for i, g in enumerate(gold) if g == lab]
        out[f"recall_{lab}"] = sum(pred[i] == lab for i in idx) / len(idx) if idx else None
        out[f"n_{lab}"] = len(idx)
    out["macro_f1"] = float(f1_score(gold, pred, labels=list(LABELS), average="macro", zero_division=0))
    out["false_execution_intents"] = sum(g == "unsupported" and p != "unsupported" for g, p in zip(gold, pred, strict=True))
    return out


def evaluate(recognizer: Any, rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    pred = recognizer.predict([norm(r["text"]) for r in rows])
    return {**score([r["gold"] for r in rows], pred),
            "errors": [{"id": r["id"], "text": r["text"], "source_intent": r["source_intent"], "pred": p}
                       for r, p in zip(rows, pred, strict=True) if p != r["gold"]]}


def main(argv: Sequence[str] | None = None) -> int:
    from athar.intent.train import FreezeError, load_recognizer

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="athar-ood", description="Egyptian out-of-distribution intent check")
    p.add_argument("--models", default="models")
    p.add_argument("--systems", nargs="+", default=["S1", "S2", "S3"])
    p.add_argument("--data", default=str(CACHE))
    p.add_argument("--out", default="runs/ood-egyptian-nlu.json")
    a = p.parse_args(argv)
    try:
        rows = load(a.data)
        results = {s: evaluate(load_recognizer(s, a.models, strict=False), rows) for s in a.systems}
    except (FreezeError, FileNotFoundError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"data": {"repo": REPO_ID, "revision": REVISION, "file": FILE, "sha256": SHA256,
                                        "attribution": ATTRIBUTION}, "systems": results},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rows)} native-written Egyptian items ({ATTRIBUTION})")
    for s, r in results.items():
        print(f"{s:6} unsupported recall {r['recall_unsupported']:.3f} (n={r['n_unsupported']})  "
              f"alarm recall {r['recall_create_alarm']:.3f} (n={r['n_create_alarm']})  macro-F1 {r['macro_f1']:.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
