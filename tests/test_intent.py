"""Intent recognizers (S1 rules, S2/S3 classifiers) and the Phase 3 search + freeze."""

import json
from pathlib import Path

import pytest
import yaml

from athar.data.io import load_items, read_records
from athar.data.split import SplitConfig, make_split, write_manifest
from athar.intent import train as T
from athar.intent.models import SklearnIntent, build
from athar.intent.rules import KeywordIntent, classify
from athar.normalize import norm
from athar.pipeline import Pipeline
from athar.timeexpr import cairo

PILOT = Path(__file__).resolve().parents[1] / "data" / "pilot" / "pilot.jsonl"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"


# --------------------------------------------------------------------------- S1
def test_s1_on_the_pilot():
    items = load_items(PILOT)
    assert KeywordIntent().predict([norm(it.text) for it in items]) == [it.intent for it in items]


@pytest.mark.parametrize(
    "text, intent",
    [
        ("فكرني اتصل بماما بكرة الساعة 5 العصر", "create_reminder"),  # "call" is content here
        ("اتصل بماما", "unsupported"),  # ... and a different tool here
        ("فكرني أمسح الصور بكرة", "create_reminder"),
        ("امسح المنبه بتاع بكرة", "unsupported"),
        ("غير المنبه لـ 7", "unsupported"),
        ("فكرني كل يوم الساعة 9", "unsupported"),
        ("فكّر أحمد يجيب الشاحن", "unsupported"),
        ("remind ahmed tomorrow", "unsupported"),
        ("فكرني بكرة 9 الصبح وصحيني الساعة 7", "unsupported"),  # two requests
        ("remind me tomorrow at 7 pm to call mom", "create_reminder"),
        ("اعملي تذكير بكرة", "create_reminder"),
        ("ظبطلي منبه 6 الصبح", "create_alarm"),
        ("set an alarm for 6 am", "create_alarm"),
        ("الجو عامل ايه بكرة", "unsupported"),
        ("ازيك", "unsupported"),
    ],
)
def test_s1_rules(text, intent):
    assert classify(norm(text)) == intent


# --------------------------------------------------------------------------- S2 / S3
TEXTS = (["فكرني بكرة أشتري عيش", "فكرني أكلم ماما", "remind me to pay the bill", "فكرني بالاجتماع"]
         + ["صحيني الساعة 6", "ظبطلي منبه بكرة", "wake me up at 7", "صحيني بدري"]
         + ["امسح المنبه", "الغي التذكير", "شغلي تايمر", "فكرني كل يوم"])
LABELS = ["create_reminder"] * 4 + ["create_alarm"] * 4 + ["unsupported"] * 4


@pytest.mark.parametrize("system, extra", [("S2", {}), ("S3", {"char_ngram": [2, 4]})])
def test_learned_systems_fit_and_predict_intents_only(system, extra):
    model = build(system, {"C": 10, "word_ngram": [1, 2], **extra}).fit([norm(t) for t in TEXTS], LABELS)
    pred = SklearnIntent(system, model).predict([norm(t) for t in TEXTS])
    assert set(pred) <= {"create_reminder", "create_alarm", "unsupported"}
    assert sum(p == g for p, g in zip(pred, LABELS, strict=True)) >= 10


def test_grid_covers_the_config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert len(T.grid("S2", cfg)) == 12
    assert len(T.grid("S3", cfg)) == 36
    assert all(p["class_weight"] == "balanced" for p in T.grid("S3", cfg))


# --------------------------------------------------------------------------- search + freeze
def _row(i, fam, text, intent):
    route = "ABSTAIN" if intent == "unsupported" else "CLARIFY"
    row = {"id": f"x{i:03d}", "family_id": fam, "text": text, "ref_datetime": "2026-10-07T10:00:00+03:00",
           "intent": intent, "route": route}
    if intent == "unsupported":
        row["unsupported_reason"] = "deletion"
    else:
        row["clarify_targets"] = ["time"]
    return row


@pytest.fixture
def split_dir(tmp_path):
    """60 items, 3 intents, families of one; dev 15, no test."""
    rows = []
    for i in range(20):
        rows.append(_row(3 * i, f"r{i}", f"فكرني أشتري حاجات البيت رقم {i}", "create_reminder"))
        rows.append(_row(3 * i + 1, f"a{i}", f"صحيني بدري عشان الشغل رقم {i}", "create_alarm"))
        rows.append(_row(3 * i + 2, f"u{i}", f"امسح المنبه بتاع الشغل رقم {i}", "unsupported"))
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    records = read_records(data)
    cfg = SplitConfig(seed=1, test={"ACT": 0, "CLARIFY": 0, "ABSTAIN": 0}, dev=15)
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


def test_search_breaks_ties_toward_the_simplest_model(split_dir):
    from athar.data.split import load_split

    cfg = yaml.safe_load((split_dir / "train.yaml").read_text(encoding="utf-8"))
    m = split_dir / "manifest.json"
    best, results = T.search("S2", cfg, load_split(m, "train"), load_split(m, "dev"))
    assert len(results) == 4
    assert all(r["dev"]["intent_macro_f1"] == 1.0 for r in results)
    assert (best["C"], best["word_ngram"]) == (1, [1, 1])


def test_fit_freeze_verify_load(split_dir, monkeypatch):
    opened = []
    real = T.load_split
    monkeypatch.setattr(T, "load_split", lambda m, s, **kw: opened.append(s) or real(m, s, **kw))
    out = split_dir / "models"
    f = T.fit_and_freeze(split_dir / "manifest.json", split_dir / "train.yaml", out, log=lambda _: None)

    assert sorted(opened) == ["dev", "train"]  # the test split is never read
    assert f["manifest"]["n_train"] == 45 and f["manifest"]["n_dev"] == 15
    assert set(f["frozen_sources"]) == set(T.FROZEN_SOURCES)
    for name in ("S2", "S3"):
        s = f["systems"][name]
        assert (out / s["file"]).exists() and s["dev"]["intent_macro_f1"] == 1.0
        assert {"final_fit", "search", "predict_ms_per_item"} <= set(s["resources"])
        assert s["resources"]["final_fit"]["cpu_seconds"] >= 0
    assert T.verify_freeze(out) == []

    r = Pipeline(T.load_recognizer("S3", out)).run("صحيني بكرة الساعة 6 الصبح", cairo("2026-10-07T10:00"))
    assert (r.system, r.intent, r.route) == ("S3", "create_alarm", "ACT")

    # a changed model file is refused before it is unpickled
    (out / "S2.joblib").write_bytes((out / "S2.joblib").read_bytes() + b"\0")
    assert T.verify_freeze(out) == ["S2: S2.joblib changed after freezing"]
    with pytest.raises(T.FreezeError):
        T.load_recognizer("S1", out)


def test_hand_edited_freeze_is_rejected(split_dir):
    out = split_dir / "models"
    T.fit_and_freeze(split_dir / "manifest.json", split_dir / "train.yaml", out, log=lambda _: None)
    f = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    f["systems"]["S2"]["dev"]["intent_macro_f1"] = 0.99
    (out / "freeze.json").write_text(json.dumps(f), encoding="utf-8")
    assert "edited after freezing" in T.verify_freeze(out)[0]


def test_changed_shared_source_is_detected(split_dir, monkeypatch):
    out = split_dir / "models"
    T.fit_and_freeze(split_dir / "manifest.json", split_dir / "train.yaml", out, log=lambda _: None)
    real = T.source_hash
    monkeypatch.setattr(T, "source_hash", lambda rel: "0" * 64 if rel == "lexicon.py" else real(rel))
    assert T.verify_freeze(out) == ["source lexicon.py changed after freezing"]


def test_cli_refuses_to_refreeze(split_dir, capsys):
    args = ["fit", "--manifest", str(split_dir / "manifest.json"), "--config",
            str(split_dir / "train.yaml"), "--out", str(split_dir / "models")]
    assert T.main(args) == 0
    assert T.main(args) == 1
    assert T.main(["verify", str(split_dir / "models")]) == 0
    capsys.readouterr()
