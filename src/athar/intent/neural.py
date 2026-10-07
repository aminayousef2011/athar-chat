"""Extended baselines S5 and S6 (optional ``neural`` extra). Reported separately from H1-H3.

S5: a frozen pretrained sentence encoder + logistic regression. Only C is searched on dev. The
    encoder is fixed in advance and pinned to a revision.
S6: a fine-tuned dialectal Arabic encoder (MARBERTv2) with a classification head. Its
    hyper-parameters are fixed in configs/train.yaml before training (no dev search), and it is
    trained with several seeds; the report gives mean and SD over seeds (Reimers & Gurevych
    2017: single-seed results of fine-tuned encoders are not reliable on small data).

Both predict the intent label only, from the same normalized text every system receives.
Models come from the Hugging Face Hub at pinned revisions. Behind a TLS-intercepting proxy the
system certificate store is used (``truststore``).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from athar.schema import INTENTS


def use_system_certificates() -> None:
    """Trust the OS certificate store (needed behind TLS-intercepting proxies)."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:  # pragma: no cover - optional
        pass


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- S5
class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


def load_encoder(model: str, revision: str) -> Any:
    use_system_certificates()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model, revision=revision, device="cpu")


def embed(encoder: Encoder, texts: Sequence[str], prefix: str) -> NDArray[np.float64]:
    vectors = encoder.encode([prefix + t for t in texts], batch_size=32, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float64)


class EmbeddingIntent:
    """S5: frozen sentence embeddings + logistic regression."""

    def __init__(self, encoder: Encoder, clf: LogisticRegression, prefix: str, name: str = "S5") -> None:
        self.name = name
        self.encoder = encoder
        self.clf = clf
        self.prefix = prefix

    def predict(self, normalized_texts: Sequence[str]) -> list[str]:
        return [str(p) for p in self.clf.predict(embed(self.encoder, normalized_texts, self.prefix))]

    def predict_proba(self, normalized_texts: Sequence[str], labels: Sequence[str]) -> list[list[float]]:
        proba = self.clf.predict_proba(embed(self.encoder, normalized_texts, self.prefix))
        col = {str(c): i for i, c in enumerate(self.clf.classes_)}
        return [[float(row[col[lab]]) if lab in col else 0.0 for lab in labels] for row in proba]


def fit_s5(encoder: Encoder, xtr: Sequence[str], ytr: Sequence[str], xdv: Sequence[str],
           ydv: Sequence[str], cfg: dict[str, Any],
           score: Callable[[Sequence[str], Sequence[str]], dict[str, float]]) -> tuple[EmbeddingIntent, dict[str, Any], list[dict[str, Any]]]:
    """Search C on dev (ties: smaller C), refit on train. Embeddings are computed once."""
    prefix = cfg.get("prefix", "")
    etr, edv = embed(encoder, xtr, prefix), embed(encoder, xdv, prefix)
    results = []
    for c in cfg["C"]:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=cfg.get("max_iter", 2000),
                                 random_state=cfg.get("seed", 0)).fit(etr, list(ytr))
        results.append({"params": {"C": c}, "dev": score(ydv, [str(p) for p in clf.predict(edv)])})
    best = min(results, key=lambda r: (-r["dev"]["intent_macro_f1"], r["params"]["C"]))
    clf = LogisticRegression(C=best["params"]["C"], class_weight="balanced",
                             max_iter=cfg.get("max_iter", 2000), random_state=cfg.get("seed", 0)).fit(etr, list(ytr))
    return EmbeddingIntent(encoder, clf, prefix), best["params"], results


def save_s5(model: EmbeddingIntent, path: Path) -> None:
    joblib.dump({"clf": model.clf, "prefix": model.prefix}, path)


def load_s5(path: Path, cfg: dict[str, Any]) -> EmbeddingIntent:
    blob = joblib.load(path)
    return EmbeddingIntent(load_encoder(cfg["model"], cfg["revision"]), blob["clf"], blob["prefix"])


# --------------------------------------------------------------------------- S6
class FineTunedIntent:
    """S6: a fine-tuned encoder with a 3-way intent head."""

    def __init__(self, model: Any, tokenizer: Any, max_length: int, name: str = "S6") -> None:
        self.name = name
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    def _logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = self.tokenizer(list(texts[i:i + 32]), padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors="pt")
                out.append(self.model(**batch).logits.double().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def predict(self, normalized_texts: Sequence[str]) -> list[str]:
        return [self.labels[int(i)] for i in self._logits(normalized_texts).argmax(axis=1)]

    def predict_proba(self, normalized_texts: Sequence[str], labels: Sequence[str]) -> list[list[float]]:
        z = self._logits(normalized_texts)
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        col = {lab: i for i, lab in enumerate(self.labels)}
        return [[float(row[col[lab]]) if lab in col else 0.0 for lab in labels] for row in p]


def load_pretrained(model: str, revision: str) -> tuple[Any, Any]:
    use_system_certificates()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    net = AutoModelForSequenceClassification.from_pretrained(
        model, revision=revision, num_labels=len(INTENTS),
        id2label=dict(enumerate(INTENTS)), label2id={lab: i for i, lab in enumerate(INTENTS)})
    return net, tok


def unk_rate(tokenizer: Any, texts: Sequence[str]) -> float:
    """Share of word pieces that are the unknown token: a tokenizer/vocabulary sanity check
    (a broken or mismatched tokenizer silently turns every word into [UNK])."""
    pieces = [p for t in texts for p in tokenizer.tokenize(t)]
    unk = tokenizer.unk_token
    return sum(p == unk for p in pieces) / len(pieces) if pieces else 0.0


def fine_tune(net: Any, tokenizer: Any, xtr: Sequence[str], ytr: Sequence[str], cfg: dict[str, Any],
              seed: int, log: Callable[[str], None] = print) -> FineTunedIntent:
    """Plain PyTorch loop: AdamW, linear warm-up and decay, class-balanced cross-entropy
    (the counterpart of ``class_weight="balanced"`` in S2/S3), fixed epochs, no dev peeking."""
    import torch

    _seed_everything(seed)
    label_id = {lab: i for i, lab in enumerate(INTENTS)}
    y = torch.tensor([label_id[v] for v in ytr])
    counts = torch.bincount(y, minlength=len(INTENTS)).double()
    weights = torch.where(counts > 0, len(y) / (len(INTENTS) * counts.clamp(min=1)), torch.zeros_like(counts))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.float())
    enc = tokenizer(list(xtr), padding=True, truncation=True, max_length=cfg["max_length"], return_tensors="pt")

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
            out = net(input_ids=enc["input_ids"][idx], attention_mask=enc["attention_mask"][idx],
                      **({"token_type_ids": enc["token_type_ids"][idx]} if "token_type_ids" in enc else {}))
            loss = loss_fn(out.logits, y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        log(f"  S6 seed {seed} epoch {epoch + 1}/{epochs} loss {total / len(y):.4f}")
    return FineTunedIntent(net, tokenizer, cfg["max_length"], name=f"S6-s{seed}")


def save_s6(model: FineTunedIntent, directory: Path) -> None:
    model.model.save_pretrained(directory)
    model.tokenizer.save_pretrained(directory)


def load_s6(directory: Path, max_length: int, name: str) -> FineTunedIntent:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return FineTunedIntent(AutoModelForSequenceClassification.from_pretrained(directory),
                           AutoTokenizer.from_pretrained(directory), max_length, name=name)
