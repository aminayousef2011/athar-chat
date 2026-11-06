"""The v3 set and the card-level cross-fitting folds (docs/protocol_v3.md, sections 2 and 4.1).

The v3 set is every screened ("kept") contributor message whose writer was *not* drawn in the
natural-set sample (``data/natural/sample.json``). Items are built with the same function that built
the natural set's blind labelling sheet (``athar.data.contrib.import_contributions`` with
``multi_writer``), so ids are ``<card>-<version>-<code>``, the family is the card and the sending
moment comes from the card. The texts are private (consent: course project, stored by code only):
they stay under ``data/raw/private/v3/``; only ids and hashes are committed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from athar.data.contrib import import_contributions

REPO = Path(__file__).resolve().parents[3]
PRIVATE = REPO / "data" / "raw" / "private"
CONTRIB = PRIVATE / "contrib"
V3_DIR = PRIVATE / "v3"
CARDS = REPO / "contrib" / "scenario_cards.csv"
SAMPLE = REPO / "data" / "natural" / "sample.json"
NATURAL = REPO / "data" / "natural" / "natural.jsonl"
COLLECTION = V3_DIR / "v3_collection.csv"
SHEET = V3_DIR / "v3_unlabelled.csv"


@dataclass(frozen=True)
class V3Item:
    """An unlabelled v3 message: what a system sees, plus provenance."""

    id: str
    text: str
    ref_datetime: datetime
    timezone: str
    family_id: str
    block: str
    writer: str


def _block_of(path: Path) -> str:
    """``wave2b_block3/block3_kept.csv`` -> ``block3`` (waves 2 and 2b of block 8 show the same cards)."""
    return path.name.split("_kept")[0]


def sampled_writers() -> set[str]:
    drawn = json.loads(SAMPLE.read_text(encoding="utf-8"))["drawn"]
    return {w for ws in drawn.values() for w in ws}


def collect(contrib_dir: Path = CONTRIB) -> list[dict[str, str]]:
    """Kept messages of the writers not drawn, with their block."""
    sampled = sampled_writers()
    rows = []
    for path in sorted(contrib_dir.glob("*/*_kept.csv")):
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r["contributor"] not in sampled:
                    rows.append({**r, "block": _block_of(path)})
    return rows


def prepare(contrib_dir: Path = CONTRIB, out_dir: Path = V3_DIR) -> dict[str, object]:
    """Write the v3 collection and its blind sheet (private) and return counts and hashes."""
    rows = collect(contrib_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collection = out_dir / COLLECTION.name
    with collection.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["card_id", "version", "contributor", "text", "comment", "block"])
        w.writeheader()
        w.writerows(rows)
    sheet = out_dir / SHEET.name
    report = import_contributions(collection, CARDS, sheet, multi_writer=True)
    items = load(sheet, collection)
    return {"messages": len(items), "writers": len({it.writer for it in items}),
            "families": len({it.family_id for it in items}), "skipped": report["skipped"],
            "set_sha256": set_hash(items)}


def load(sheet: Path = SHEET, collection: Path = COLLECTION) -> list[V3Item]:
    """The v3 items, in sheet order, with each item's block from the collection."""
    block = {}
    with collection.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            block[(r["card_id"], r["version"].lower(), r["contributor"])] = r["block"]
    items = []
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            card, version, _ = r["id"].split("-", 2)
            items.append(V3Item(id=r["id"], text=r["text"], ref_datetime=datetime.fromisoformat(r["ref_datetime"]),
                                timezone=r["timezone"], family_id=r["family_id"],
                                block=block[(card, version, r["contributor"])], writer=r["contributor"]))
    return items


def set_hash(items: list[V3Item]) -> str:
    """SHA-256 over (id, text, sending moment) in id order: freezes exactly what systems saw."""
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x.id):
        h.update(json.dumps([it.id, it.text, it.ref_datetime.isoformat()], ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def block_cards(contrib_dir: Path = CONTRIB) -> dict[str, set[str]]:
    """Block -> every scenario card (family id) its form showed, from all kept messages of that
    block (drawn writers included), so the exclusion covers the block's cards, not only the ones
    the v3 writers answered (protocol_v3.md, section 4.1). ``screened_kept_*`` (wave1-block5, H18)
    is not a v3 block: its writer is in the natural set."""
    out: dict[str, set[str]] = defaultdict(set)
    for path in sorted(contrib_dir.glob("*/*_kept.csv")):
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                out[_block_of(path)].add(f"fam-{r['card_id']}")
    return dict(out)
