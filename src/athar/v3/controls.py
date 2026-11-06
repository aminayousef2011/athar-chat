"""Descriptive control S3-contributors-only (docs/protocol_v3_amendments.md, A4).

S3 with the frozen S3-gcv settings, trained per block on only the natural-set messages whose card
is outside the block's cards: no drafted and no author item. It separates "human data" from "more
data" in H-a (S3-human = drafted+author + contributors). Descriptive; frozen with its predictions
before any v3 label exists.

    python -m athar.v3.controls freeze
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib

from athar.data.io import sha256_file
from athar.intent.train import source_hash
from athar.normalize import norm
from athar.schema import GoldItem
from athar.v3 import amend
from athar.v3 import data as D
from athar.v3 import systems as S

SOURCES = ("src/athar/v3/controls.py", "src/athar/v3/amend.py", *S.V3_SOURCES, *S.V2_SOURCES_USED)
AMENDMENTS = D.REPO / "docs" / "protocol_v3_amendments.md"


def contributors_pool(block: str, cards: dict[str, set[str]], natural: Sequence[GoldItem]) -> list[GoldItem]:
    """The natural-set messages on cards outside ``block`` (nothing else)."""
    return [it for it in natural if it.family_id not in cards[block]]


def freeze(out: Path = S.MODELS_V3) -> dict[str, Any]:
    if (out / "freeze_controls.json").exists():
        raise S.FreezeError(f"{out}/freeze_controls.json exists")
    items = D.load()
    cards = D.block_cards()
    natural = S.natural_items()
    models: dict[str, Any] = {}
    preds: dict[str, str] = {}
    for block in sorted({it.block for it in items}):
        pool = contributors_pool(block, cards, natural)
        model = S.fit_s3(pool)
        path = out / f"S3-contrib-only-{block}.joblib"
        joblib.dump(model.model, path)
        models[f"S3-contrib-only-{block}"] = {"file": path.name, "sha256": sha256_file(path), "n_train": len(pool)}
        block_items = [it for it in items if it.block == block]
        preds.update(zip([it.id for it in block_items], model.predict([norm(it.text) for it in block_items]),
                         strict=True))
    assert set(preds) == {it.id for it in items}
    rel, h = S._write_predictions("S3-contrib-only", preds)
    common = S._common(items)
    common["sources"] = {rel_: source_hash(rel_) for rel_ in SOURCES}
    return S._write_once(out / "freeze_controls.json", {
        **common, "amendment": "A4", **amend.snapshot(AMENDMENTS), "gate": "G2+G3 (control)",
        "params": S.s3_params(), "models": models,
        "predictions": {"S3-contrib-only": {"file": rel, "sha256": h}}})


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args != ["freeze"]:
        print("usage: python -m athar.v3.controls freeze")
        return 2
    fz = freeze()
    print(f"models_v3/freeze_controls.json  freeze_sha256={fz['freeze_sha256']}")
    print(json.dumps({k: m["n_train"] for k, m in fz["models"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
