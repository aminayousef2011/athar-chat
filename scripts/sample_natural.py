"""Draw the labelled sample of the natural-phrasing set (docs/protocol_natural.md, log 2026-10-29).

Plan B: before any message is labelled, keep whole writers, not single messages, so each writer's
A/B pairs stay intact. For every scenario block (the set of cards one form covered) two writers
are drawn at random with a fixed seed; a block with fewer writers keeps them all. Every kept
message of a drawn writer is in the sample, including messages on train/dev scenarios (reported
apart). Messages of writers not drawn are not labelled and not used. The rows are shuffled (same
seed) so the labelling order does not follow writers, cards or A/B.

    uv run python scripts/sample_natural.py

Reads the screened collections in data/raw/private/contrib/ (private) and writes
data/raw/private/contrib/natural_sample_collection.csv (texts, private) and
data/natural/sample.json (which writers were drawn; no texts).
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "raw" / "private" / "contrib"
SEED = 20261029
WRITERS_PER_BLOCK = 2
#: scenario block -> screened collections of that card set (wave-2 and wave-2b block 8 share cards)
BLOCKS = {
    "wave1-block5": ["screened_kept_2026-10-26.csv"],
    **{f"block{b}": [f"wave2b_block{b}/block{b}_kept.csv"] for b in range(1, 8)},
    "block8": ["wave2_block8/block8_kept.csv", "wave2b_block8/block8_kept.csv"],
}
COLUMNS = ("card_id", "version", "contributor", "text", "comment")


def main() -> int:
    rng = random.Random(SEED)
    sample, drawn, pool = [], {}, {}
    for block, files in BLOCKS.items():
        rows = [r for f in files for r in csv.DictReader(open(PRIVATE / f, encoding="utf-8-sig"))]
        writers = sorted({r["contributor"] for r in rows})
        chosen = sorted(rng.sample(writers, min(WRITERS_PER_BLOCK, len(writers))))
        drawn[block] = chosen
        pool[block] = {w: sum(r["contributor"] == w for r in rows) for w in writers}
        sample += [r for r in rows if r["contributor"] in chosen]
    if len({r["contributor"] for r in sample}) != sum(len(v) for v in drawn.values()):
        raise SystemExit("a writer code occurs in two blocks: resolve it before sampling")
    rng.shuffle(sample)
    out = PRIVATE / "natural_sample_collection.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows({k: r[k] for k in COLUMNS} for r in sample)
    record = {"protocol": "docs/protocol_natural.md (log 2026-10-29, plan B)", "seed": SEED,
              "writers_per_block": WRITERS_PER_BLOCK, "drawn": drawn, "pool_messages_per_writer": pool,
              "sample_messages": len(sample),
              "sample_confirmatory": sum(not r["comment"] for r in sample),
              "sample_writers": len({r["contributor"] for r in sample})}
    (ROOT / "data" / "natural").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "natural" / "sample.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for block, chosen in drawn.items():
        print(f"{block:13} drew {', '.join(chosen):12} from {len(pool[block])} writers")
    print(f"sample: {record['sample_messages']} messages ({record['sample_confirmatory']} confirmatory) "
          f"from {record['sample_writers']} writers -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
