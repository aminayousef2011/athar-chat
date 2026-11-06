"""v3 systems, their training pools, freeze and predictions (docs/protocol_v3.md, sections 3-4).

- ``S3-drafted``: S3 with the frozen S3-gcv settings, trained on train + dev.
- ``S3-human``:   same settings, trained per block *b* on train + dev + every natural-set message
                  whose card is not one of block *b*'s cards (8 fits).
- ``S6-drafted`` / ``S6-human``: MARBERTv2 with the ``configs/train.yaml`` recipe, seeds 0-4, the
                  same pools. Trained with :func:`fine_tune`, the recipe of
                  ``athar.intent.neural.fine_tune`` (frozen v1 code, CPU only) with one change: the
                  model and batches are moved to ``device``, so it can run on a GPU.

Predictions are intents by item id (no text); routes and arguments come from core v1.1 at
scoring time. Freezes are written once and never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import joblib
import yaml

from athar.data.io import canonical_json, sha256_file, sha256_text
from athar.data.split import load_split
from athar.intent.models import SklearnIntent, build
from athar.intent.train import source_hash
from athar.normalize import norm
from athar.schema import INTENTS, GoldItem
from athar.v2 import gcv
from athar.v3 import data as D

REPO = D.REPO
MAIN = REPO / "data" / "splits" / "manifest.json"
NATURAL_MANIFEST = REPO / "data" / "natural" / "manifest.json"
PROTOCOL = REPO / "docs" / "protocol_v3.md"
CONFIG = REPO / "configs" / "train.yaml"
MODELS_V3 = REPO / "models_v3"
PRED_DIR = REPO / "results" / "v3" / "predictions"
V3_SOURCES = tuple(f"src/athar/v3/{m}.py" for m in ("__init__", "data", "systems", "__main__"))
V2_SOURCES_USED = ("src/athar/v2/core.py", "src/athar/v2/rules.py", "src/athar/v2/gcv.py")


class FreezeError(RuntimeError):
    """A v3 freeze is missing, would be overwritten, or something in it changed."""


# --------------------------------------------------------------------------- pools
def drafted_pool() -> list[GoldItem]:
    return list(load_split(MAIN, "train")) + list(load_split(MAIN, "dev"))


def natural_items() -> list[GoldItem]:
    return list(load_split(NATURAL_MANIFEST, "test", unlock_test=True))


def human_pool(block: str, cards: dict[str, set[str]], natural: Sequence[GoldItem]) -> list[GoldItem]:
    """train + dev + the natural-set messages on cards outside ``block``."""
    excluded = cards[block]
    return drafted_pool() + [it for it in natural if it.family_id not in excluded]


def _xy(pool: Sequence[GoldItem]) -> tuple[list[str], list[str]]:
    return [norm(it.text) for it in pool], [it.intent for it in pool]


# --------------------------------------------------------------------------- S3
def s3_params() -> dict[str, Any]:
    params: dict[str, Any] = gcv.read_freeze(REPO / "models_v2")["systems"]["S3-gcv"]["params"]
    return params


def fit_s3(pool: Sequence[GoldItem], seed: int = 0) -> SklearnIntent:
    x, y = _xy(pool)
    return SklearnIntent("S3", build("S3", s3_params(), seed).fit(x, y))


# --------------------------------------------------------------------------- S6
def s6_config() -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["S6"]
    return cfg


def fine_tune(net: Any, tokenizer: Any, xtr: Sequence[str], ytr: Sequence[str], cfg: dict[str, Any],
              seed: int, device: str = "cpu", log: Callable[[str], None] = print) -> Any:
    """The S6 recipe of ``athar.intent.neural.fine_tune`` (AdamW, linear warm-up and decay,
    class-balanced cross-entropy, gradient clipping 1.0, fixed epochs, no dev peeking), on
    ``device``. Returns the trained network in eval mode."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    label_id = {lab: i for i, lab in enumerate(INTENTS)}
    y = torch.tensor([label_id[v] for v in ytr])
    counts = torch.bincount(y, minlength=len(INTENTS)).double()
    weights = torch.where(counts > 0, len(y) / (len(INTENTS) * counts.clamp(min=1)), torch.zeros_like(counts))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.float().to(device))
    enc = tokenizer(list(xtr), padding=True, truncation=True, max_length=cfg["max_length"], return_tensors="pt")
    net = net.to(device)
    bs, epochs = cfg["batch_size"], cfg["epochs"]
    steps = epochs * math.ceil(len(y) / bs)
    warmup = max(1, int(cfg.get("warmup_ratio", 0.1) * steps))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.01))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup)))
    gen = torch.Generator().manual_seed(seed)
    net.train()
    for epoch in range(epochs):
        order = torch.randperm(len(y), generator=gen)
        total = 0.0
        for i in range(0, len(y), bs):
            idx = order[i:i + bs]
            batch = {k: v[idx].to(device) for k, v in enc.items()}
            out = net(**batch)
            loss = loss_fn(out.logits, y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        log(f"    epoch {epoch + 1}/{epochs} loss {total / len(y):.4f}")
    return net.eval()


def predict_s6(net: Any, tokenizer: Any, texts: Sequence[str], max_length: int, device: str = "cpu") -> list[str]:
    import torch

    labels = [net.config.id2label[i] for i in range(net.config.num_labels)]
    out: list[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = tokenizer(list(texts[i:i + 32]), padding=True, truncation=True, max_length=max_length,
                              return_tensors="pt")
            logits = net(**{k: v.to(device) for k, v in batch.items()}).logits
            out += [labels[int(j)] for j in logits.argmax(dim=1).tolist()]
    return out


def state_hash(net: Any) -> str:
    """SHA-256 over the trained weights (names, shapes, bytes): identifies a model without storing it."""
    h = hashlib.sha256()
    for name, t in sorted(net.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- freeze helpers
def _sources() -> dict[str, str]:
    return {rel: source_hash(rel) for rel in (*V3_SOURCES, *V2_SOURCES_USED)}


def _write_once(path: Path, obj: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FreezeError(f"{path} exists: a v3 freeze is written once and never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in obj.items() if k != "freeze_sha256"}
    out = {**body, "freeze_sha256": sha256_text(canonical_json(body))}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


def _write_predictions(name: str, preds: dict[str, str]) -> tuple[str, str]:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{name}.json"
    if path.exists():
        raise FreezeError(f"{path} exists: predictions are written once")
    path.write_text(json.dumps(dict(sorted(preds.items())), ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8", newline="\n")
    return str(path.relative_to(REPO)).replace("\\", "/"), sha256_file(path)


def _common(items: Sequence[D.V3Item]) -> dict[str, Any]:
    return {"protocol": "docs/protocol_v3.md", "protocol_sha256": sha256_file(PROTOCOL),
            "v3_set_sha256": D.set_hash(list(items)), "v3_items": len(items),
            "natural_manifest": json.loads(NATURAL_MANIFEST.read_text(encoding="utf-8"))["manifest_sha256"],
            "config_sha256": sha256_file(CONFIG), "sources": _sources()}


# --------------------------------------------------------------------------- S3 freeze
def freeze_s3(out: Path = MODELS_V3) -> dict[str, Any]:
    """Fit S3-drafted (1 model) and S3-human (one per block), save them, predict the v3 set, and
    write ``models_v3/freeze_s3.json`` and the predictions. Refuses to overwrite."""
    items = D.load()
    cards = D.block_cards()
    natural = natural_items()
    if (out / "freeze_s3.json").exists():
        raise FreezeError(f"{out}/freeze_s3.json exists")
    out.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    drafted = fit_s3(drafted_pool())
    path = out / "S3-drafted.joblib"
    joblib.dump(drafted.model, path)
    models["S3-drafted"] = {"file": path.name, "sha256": sha256_file(path), "n_train": len(drafted_pool())}
    pred_drafted = dict(zip([it.id for it in items], drafted.predict([norm(it.text) for it in items]), strict=True))
    pred_human: dict[str, str] = {}
    for block in sorted({it.block for it in items}):
        pool = human_pool(block, cards, natural)
        model = fit_s3(pool)
        path = out / f"S3-human-{block}.joblib"
        joblib.dump(model.model, path)
        models[f"S3-human-{block}"] = {"file": path.name, "sha256": sha256_file(path), "n_train": len(pool),
                                       "natural_in_pool": len(pool) - len(drafted_pool()),
                                       "excluded_cards": sorted(cards[block])}
        block_items = [it for it in items if it.block == block]
        for it, y in zip(block_items, model.predict([norm(it.text) for it in block_items]), strict=True):
            pred_human[it.id] = y
    assert set(pred_human) == {it.id for it in items}
    predictions = {}
    for name, preds in (("S3-drafted", pred_drafted), ("S3-human", pred_human)):
        rel, h = _write_predictions(name, preds)
        predictions[name] = {"file": rel, "sha256": h}
    return _write_once(out / "freeze_s3.json", {**_common(items), "gate": "G2+G3 (S3)", "params": s3_params(),
                                                "models": models, "predictions": predictions})


# --------------------------------------------------------------------------- S6 freeze
def freeze_s6(device: str = "cpu", seeds: Sequence[int] | None = None, out: Path = MODELS_V3,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """Train S6-drafted (per seed) and S6-human (per seed and block) and write
    ``models_v3/freeze_s6.json`` with weight hashes and the predictions. Weights are not stored
    (13 x 0.65 GB per seed); the committed predictions are what the protocol scores."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from athar.intent.neural import use_system_certificates

    use_system_certificates()
    if (out / "freeze_s6.json").exists():
        raise FreezeError(f"{out}/freeze_s6.json exists")
    cfg = s6_config()
    seeds = list(seeds if seeds is not None else cfg["seeds"])
    items = D.load()
    cards = D.block_cards()
    natural = natural_items()
    texts = {it.id: norm(it.text) for it in items}
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])

    def new_net() -> Any:
        return AutoModelForSequenceClassification.from_pretrained(
            cfg["model"], revision=cfg["revision"], num_labels=len(INTENTS),
            id2label=dict(enumerate(INTENTS)), label2id={lab: i for i, lab in enumerate(INTENTS)})

    models: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    for seed in seeds:
        log(f"S6-drafted seed {seed}")
        x, y = _xy(drafted_pool())
        net = fine_tune(new_net(), tok, x, y, cfg, seed, device, log)
        models[f"S6-drafted-s{seed}"] = {"weights_sha256": state_hash(net), "n_train": len(x)}
        ids = [it.id for it in items]
        preds = dict(zip(ids, predict_s6(net, tok, [texts[i] for i in ids], cfg["max_length"], device), strict=True))
        rel, h = _write_predictions(f"S6-drafted-s{seed}", preds)
        predictions[f"S6-drafted-s{seed}"] = {"file": rel, "sha256": h}
        human: dict[str, str] = {}
        for block in sorted({it.block for it in items}):
            log(f"S6-human seed {seed} block {block}")
            pool = human_pool(block, cards, natural)
            x, y = _xy(pool)
            net = fine_tune(new_net(), tok, x, y, cfg, seed, device, log)
            models[f"S6-human-s{seed}-{block}"] = {"weights_sha256": state_hash(net), "n_train": len(x)}
            bids = [it.id for it in items if it.block == block]
            human.update(zip(bids, predict_s6(net, tok, [texts[i] for i in bids], cfg["max_length"], device),
                             strict=True))
        assert set(human) == set(texts)
        rel, h = _write_predictions(f"S6-human-s{seed}", human)
        predictions[f"S6-human-s{seed}"] = {"file": rel, "sha256": h}
    return _write_once(out / "freeze_s6.json", {**_common(items), "gate": "G2+G3 (S6)", "recipe": cfg,
                                                "device": device, "seeds": seeds, "models": models,
                                                "predictions": predictions})


# --------------------------------------------------------------------------- verify
def verify(out: Path = MODELS_V3) -> list[str]:
    """Problems with the v3 freezes present (empty when nothing changed)."""
    problems: list[str] = []
    found = sorted(out.glob("freeze_*.json"))
    if not found:
        return [f"no v3 freeze in {out}"]
    for f in found:
        data = json.loads(f.read_text(encoding="utf-8"))
        body = {k: v for k, v in data.items() if k != "freeze_sha256"}
        if sha256_text(canonical_json(body)) != data.get("freeze_sha256"):
            problems.append(f"{f.name} was edited")
        for name, m in data.get("models", {}).items():
            if "file" in m and (not (out / m["file"]).exists() or sha256_file(out / m["file"]) != m["sha256"]):
                problems.append(f"{f.name}: model {name} missing or changed")
        for name, p in data.get("predictions", {}).items():
            path = REPO / p["file"]
            if not path.exists() or sha256_file(path) != p["sha256"]:
                problems.append(f"{f.name}: predictions {name} missing or changed")
        for rel, h in data["sources"].items():
            if not (REPO / rel).exists() or source_hash(rel) != h:
                problems.append(f"{f.name}: source {rel} changed since this freeze")
        if sha256_file(PROTOCOL) != data["protocol_sha256"]:
            problems.append(f"{f.name}: docs/protocol_v3.md changed since this freeze (log it in section 9)")
    return problems
