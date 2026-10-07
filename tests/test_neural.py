"""Extended baselines S5/S6, tested offline: a deterministic fake encoder for S5 and a tiny
randomly initialised BERT (built locally, nothing downloaded) for S6."""

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from athar.data.split import load_split  # noqa: E402
from athar.eval import harness as H  # noqa: E402
from athar.intent import neural  # noqa: E402
from athar.intent import train as T  # noqa: E402
from athar.normalize import norm  # noqa: E402
from athar.schema import INTENTS  # noqa: E402

S6_CFG = {"model": "tiny", "revision": "local", "seeds": [0, 1], "epochs": 12, "batch_size": 8,
          "learning_rate": 5e-3, "weight_decay": 0.0, "warmup_ratio": 0.1, "max_length": 16}


class FakeEncoder:
    """Hashed word features, L2-normalised: deterministic and download-free."""

    def encode(self, sentences, **kw):
        out = np.zeros((len(sentences), 64))
        for i, s in enumerate(sentences):
            for w in s.split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def tiny_bert(texts, tmp_path):
    words = sorted({w for t in texts for w in t.split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    # transformers 5 takes the vocabulary itself; a ``vocab_file=`` argument is silently ignored
    tok = transformers.BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=False)
    assert neural.unk_rate(tok, texts) == 0.0
    torch.manual_seed(123)
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                                  num_attention_heads=2, intermediate_size=64, num_labels=len(INTENTS),
                                  id2label=dict(enumerate(INTENTS)),
                                  label2id={lab: i for i, lab in enumerate(INTENTS)})
    return transformers.BertForSequenceClassification(cfg), tok


def _texts(split_dir, split):
    items = load_split(split_dir / "manifest.json", split)
    return [norm(it.text) for it in items], [it.intent for it in items]


def test_s5_learns_from_frozen_embeddings(split_dir):
    xtr, ytr = _texts(split_dir, "train")
    xdv, ydv = _texts(split_dir, "dev")
    model, best, results = neural.fit_s5(FakeEncoder(), xtr, ytr, xdv, ydv,
                                         {"C": [0.1, 1, 10], "prefix": "query: "}, T.scores)
    assert len(results) == 3 and best["C"] in (0.1, 1, 10)
    assert T.scores(ydv, model.predict(xdv))["intent_macro_f1"] == 1.0
    proba = np.asarray(model.predict_proba(xdv, INTENTS))
    assert proba.shape == (len(xdv), 3) and np.allclose(proba.sum(axis=1), 1.0)


def test_s6_fine_tunes_deterministically_and_round_trips(split_dir, tmp_path):
    xtr, ytr = _texts(split_dir, "train")
    xdv, ydv = _texts(split_dir, "dev")
    runs = []
    for _ in range(2):
        net, tok = tiny_bert(xtr + xdv, tmp_path)
        runs.append(neural.fine_tune(net, tok, xtr, ytr, S6_CFG, seed=7, log=lambda _: None))
    a, b = runs
    assert a.predict(xdv) == b.predict(xdv)  # same seed, same model
    assert np.allclose(a.predict_proba(xdv, INTENTS), b.predict_proba(xdv, INTENTS))
    assert T.scores(ydv, a.predict(xdv))["intent_macro_f1"] >= 0.8
    neural.save_s6(a, tmp_path / "s6")
    again = neural.load_s6(tmp_path / "s6", S6_CFG["max_length"], "S6-s7")
    assert again.predict(xdv) == a.predict(xdv)


def test_extended_freeze_verify_load_and_seed_summary(split_dir):
    import yaml

    cfg = yaml.safe_load((split_dir / "train.yaml").read_text(encoding="utf-8"))
    cfg["S5"] = {"model": "fake", "revision": "local", "C": [1, 10], "prefix": ""}
    cfg["S6"] = S6_CFG
    (split_dir / "train.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    xtr, _ = _texts(split_dir, "train")
    xdv, _ = _texts(split_dir, "dev")
    out = split_dir / "models"
    f = T.fit_and_freeze(split_dir / "manifest.json", split_dir / "train.yaml", out, log=lambda _: None,
                         extended=True, encoder_factory=lambda m, r: FakeEncoder(),
                         pretrained_factory=lambda m, r: tiny_bert(xtr + xdv, split_dir))
    assert {"S5", "S6-s0", "S6-s1", "S6"} <= set(f["systems"])
    assert f["systems"]["S6"]["kind"] == "seed summary"
    assert any(k.endswith("model.safetensors") for k in f["systems"]["S6-s0"]["files"])
    assert "src/athar/intent/neural.py" in f["frozen_sources"]
    assert T.verify_freeze(out) == []
    assert T.frozen_systems(out) == ["S1", "S2", "S3", "S5", "S6-s0", "S6-s1"]

    s6 = T.load_recognizer("S6-s1", out)
    items = load_split(split_dir / "manifest.json", "dev")
    rep = H.evaluate({"S6-s0": T.load_recognizer("S6-s0", out), "S6-s1": s6}, items,
                     {"seed": 0, "level": 0.95, "resamples": 50, "permutations": 50, "alpha": 0.05,
                      "oracle": False, "perturbations": ["digits"], "comparisons": []})
    summ = rep["seed_summaries"]["S6"]
    assert summ["seeds"] == ["S6-s0", "S6-s1"]
    assert 0.0 <= summ["safe_task_success"]["mean"] <= 1.0 and summ["safe_task_success"]["sd"] >= 0
    assert rep["systems"]["S6-s0"]["calibration"] is not None  # probabilistic system

    weights = next(out.glob("S6/seed0/model.safetensors"))
    weights.write_bytes(weights.read_bytes()[:-1] + b"\x00")
    assert any("S6-s0" in p and "changed after freezing" in p for p in T.verify_freeze(out))
