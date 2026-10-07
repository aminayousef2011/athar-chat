"""S2 (word TF-IDF + logistic regression) and S3 (S2 + character n-grams).

Both learn the three intent labels only, from normalized text; the route always comes from the
shared policy. Vectorizers are fitted on the train split only (they live inside the sklearn
pipeline, so ``fit`` on train is the only place they see data).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

#: keep one-character tokens: "ب", "م" (pm marker) and single digits carry signal here
TOKEN_PATTERN = r"(?u)\b\w+\b"


def build(system: str, params: dict[str, Any], seed: int = 0) -> Pipeline:
    """``params``: C, word_ngram [lo, hi], (S3) char_ngram [lo, hi], plus shared options."""
    word = TfidfVectorizer(analyzer="word", token_pattern=TOKEN_PATTERN,
                           ngram_range=tuple(params["word_ngram"]),
                           sublinear_tf=params.get("sublinear_tf", True),
                           min_df=params.get("min_df", 1))
    if system == "S2":
        features: Any = word
    elif system == "S3":
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(params["char_ngram"]),
                               sublinear_tf=params.get("sublinear_tf", True),
                               min_df=params.get("min_df", 1))
        features = FeatureUnion([("word", word), ("char", char)])
    else:
        raise ValueError(f"unknown learned system {system!r}")
    clf = LogisticRegression(C=params["C"], class_weight=params.get("class_weight", "balanced"),
                             max_iter=params.get("max_iter", 2000), random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


class SklearnIntent:
    """A fitted S2/S3 pipeline behind the IntentRecognizer protocol."""

    def __init__(self, name: str, model: Pipeline) -> None:
        self.name = name
        self.model = model

    def predict(self, normalized_texts: Sequence[str]) -> list[str]:
        return [str(p) for p in self.model.predict(list(normalized_texts))]

    def predict_proba(self, normalized_texts: Sequence[str], labels: Sequence[str]) -> list[list[float]]:
        """Class probabilities in the order of ``labels`` (0 for a label unseen in training)."""
        proba = self.model.predict_proba(list(normalized_texts))
        col = {str(c): i for i, c in enumerate(self.model.classes_)}
        return [[float(row[col[lab]]) if lab in col else 0.0 for lab in labels] for row in proba]
