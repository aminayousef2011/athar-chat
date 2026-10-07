import json

import pytest
import yaml

from athar.data.io import read_records
from athar.data.split import SplitConfig, make_split, write_manifest
from athar.timeexpr import cairo


@pytest.fixture
def ref():
    """Wednesday 7 Oct 2026, 10:00 Cairo (UTC+03:00, DST until 29 Oct)."""
    return cairo("2026-10-07T10:00")


def _row(i, fam, text, intent):
    route = "ABSTAIN" if intent == "unsupported" else "CLARIFY"
    row = {"id": f"x{i:03d}", "family_id": fam, "text": text, "ref_datetime": "2026-10-07T10:00:00+03:00",
           "intent": intent, "route": route}
    if intent == "unsupported":
        row["unsupported_reason"] = "deletion"
    else:
        row["clarify_targets"] = ["time"]
    return row


def make_split_dir(tmp_path, test=None, dev=15):
    """60 synthetic items (3 intents, families of one), split and with a small training config."""
    rows = []
    for i in range(20):
        rows.append(_row(3 * i, f"r{i}", f"فكرني أشتري حاجات البيت رقم {i}", "create_reminder"))
        rows.append(_row(3 * i + 1, f"a{i}", f"صحيني بدري عشان الشغل رقم {i}", "create_alarm"))
        rows.append(_row(3 * i + 2, f"u{i}", f"امسح المنبه بتاع الشغل رقم {i}", "unsupported"))
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    records = read_records(data)
    cfg = SplitConfig(seed=1, test=test or {"ACT": 0, "CLARIFY": 0, "ABSTAIN": 0}, dev=dev)
    mpath = tmp_path / "manifest.json"
    manifest, short, _ = make_split([r.item for r in records], {r.raw["id"]: r.raw for r in records},
                                    cfg, data, mpath)
    assert short == []
    write_manifest(manifest, mpath)
    small = {"seed": 0, "common": {"class_weight": "balanced", "max_iter": 500},
             "S2": {"C": [10, 1], "word_ngram": [[1, 2], [1, 1]]},
             "S3": {"C": [1], "word_ngram": [[1, 1]], "char_ngram": [[2, 4]]}}
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(small), encoding="utf-8")
    return tmp_path


@pytest.fixture
def split_dir(tmp_path):
    """Train 45 / dev 15, no test."""
    return make_split_dir(tmp_path)


@pytest.fixture
def split_dir_with_test(tmp_path):
    """Train 36 / dev 12 / test 12 (8 CLARIFY, 4 ABSTAIN)."""
    return make_split_dir(tmp_path, test={"ACT": 0, "CLARIFY": 8, "ABSTAIN": 4}, dev=12)
