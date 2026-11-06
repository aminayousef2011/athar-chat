"""Protocol v3 data handling: which writers are in the v3 set, and the card-level folds."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from athar.v3 import data as D
from athar.v3 import systems as S


def _kept(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card_id", "version", "contributor", "text", "comment"])
        for r in rows:
            w.writerow([*r, ""])


@pytest.fixture
def contrib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"drawn": {"block1": ["E1"], "block2": ["Q1"]}}), encoding="utf-8")
    monkeypatch.setattr(D, "SAMPLE", sample)
    root = tmp_path / "contrib"
    _kept(root / "wave2b_block1" / "block1_kept.csv",
          [("c001", "A", "E1", "drawn writer"), ("c001", "A", "AD1", "v3 writer"), ("c002", "B", "AD1", "v3 b")])
    _kept(root / "wave2b_block2" / "block2_kept.csv", [("c003", "A", "Q1", "drawn"), ("c004", "A", "Q1", "drawn")])
    return root


def test_collect_keeps_only_writers_not_drawn(contrib: Path) -> None:
    rows = D.collect(contrib)
    assert {r["contributor"] for r in rows} == {"AD1"}
    assert {r["block"] for r in rows} == {"block1"}


def test_block_cards_include_cards_answered_only_by_drawn_writers(contrib: Path) -> None:
    cards = D.block_cards(contrib)
    assert cards["block1"] == {"fam-c001", "fam-c002"}
    assert cards["block2"] == {"fam-c003", "fam-c004"}   # no v3 writer answered these


def test_human_pool_excludes_natural_messages_on_the_blocks_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    class It:
        def __init__(self, fam: str) -> None:
            self.family_id = fam

    monkeypatch.setattr(S, "drafted_pool", lambda: [])
    natural = [It("fam-c001"), It("fam-c002"), It("fam-c009"), It("fam-c012")]
    cards = {"block1": {"fam-c001", "fam-c002"}, "block5": {"fam-c012"}}
    assert [i.family_id for i in S.human_pool("block1", cards, natural)] == ["fam-c009", "fam-c012"]
    # a card shared with another wave (c012: wave1-block5 and block5) is excluded by card, not label
    assert "fam-c012" not in [i.family_id for i in S.human_pool("block5", cards, natural)]


def test_set_hash_depends_on_text_and_not_on_order() -> None:
    from datetime import datetime

    a = D.V3Item("c1-a-x", "t1", datetime(2026, 11, 1), "Africa/Cairo", "fam-c1", "block1", "X")
    b = D.V3Item("c2-a-x", "t2", datetime(2026, 11, 1), "Africa/Cairo", "fam-c2", "block1", "X")
    assert D.set_hash([a, b]) == D.set_hash([b, a])
    changed = D.V3Item("c1-a-x", "t1 edited", datetime(2026, 11, 1), "Africa/Cairo", "fam-c1", "block1", "X")
    assert D.set_hash([changed, b]) != D.set_hash([a, b])


def test_freeze_files_are_written_once(tmp_path: Path) -> None:
    out = tmp_path / "freeze_x.json"
    S._write_once(out, {"a": 1})
    with pytest.raises(S.FreezeError):
        S._write_once(out, {"a": 2})
