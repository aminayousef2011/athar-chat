"""Uncertainty and significance, pre-registered in docs/evaluation_protocol.md section 5.

Test items are not independent: paraphrases and complete/incomplete pairs share a scenario
family. Every resampling procedure here therefore works on *families* (clusters), never on
single items:

- :func:`cluster_bootstrap` - percentile CI for any metric, resampling families with
  replacement (Efron & Tibshirani 1993; cluster bootstrap, Field & Welsh 2007);
- :func:`paired_permutation_test` - approximate randomization (Noreen 1989), flipping the two
  systems' outputs per family; the recommended test for non-normal NLP metrics on small test
  sets (Dror et al. 2018);
- :func:`mcnemar_exact` - item-level sensitivity analysis (exact binomial McNemar);
- :func:`holm` - Holm-Bonferroni adjustment over the pre-registered family of comparisons;
- :func:`expected_calibration_error`, :func:`brier_score`, :func:`aurc` - calibration and
  selective-prediction diagnostics for the probabilistic recognizers (Guo et al. 2017;
  Geifman & El-Yaniv 2017);
- :func:`simulate_power` - power of the paired permutation test for a planned test set.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest

Index = NDArray[np.int64]


def _cluster_index(clusters: Sequence[str]) -> list[Index]:
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    return [np.asarray(v, dtype=np.int64) for _, v in sorted(groups.items())]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    level: float
    resamples: int

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high, "level": self.level,
                "resamples": self.resamples}


def cluster_bootstrap(metric: Callable[[Index], float], clusters: Sequence[str], *,
                      resamples: int = 10_000, level: float = 0.95, seed: int = 0) -> Interval:
    """Percentile interval of ``metric(indices)`` under family resampling.

    ``metric`` receives an array of item indices (with repeats) and returns the statistic, so
    any metric (a mean, a macro-F1, a difference between two systems) can be bootstrapped.
    """
    groups = _cluster_index(clusters)
    rng = np.random.default_rng(seed)
    point = float(metric(np.arange(len(clusters), dtype=np.int64)))
    stats = np.empty(resamples)
    for b in range(resamples):
        pick = rng.integers(0, len(groups), size=len(groups))
        stats[b] = metric(np.concatenate([groups[g] for g in pick]))
    a = (1 - level) / 2
    low, high = np.quantile(stats, [a, 1 - a])
    return Interval(point, float(low), float(high), level, resamples)


def mean_ci(values: Sequence[float], clusters: Sequence[str], **kw: float) -> Interval:
    v = np.asarray(values, dtype=float)
    return cluster_bootstrap(lambda idx: float(v[idx].mean()), clusters, **kw)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PairedTest:
    difference: float  # mean(a) - mean(b)
    p_value: float
    permutations: int
    ci: Interval

    def as_dict(self) -> dict[str, object]:
        return {"difference": self.difference, "p_value": self.p_value,
                "permutations": self.permutations, "ci": self.ci.as_dict()}


def paired_permutation_test(a: Sequence[float], b: Sequence[float], clusters: Sequence[str], *,
                            permutations: int = 10_000, resamples: int = 10_000,
                            level: float = 0.95, seed: int = 0) -> PairedTest:
    """Two-sided approximate randomization test of mean(a) = mean(b), swapping per family.

    Under H0 the two systems are exchangeable, so swapping their outputs on every item of a
    family leaves the distribution unchanged. With per-family difference sums ``d_c`` the
    statistic is ``sum(s_c * d_c) / n`` for random signs ``s_c``. The p-value includes the
    observed assignment: ``(1 + #{|T*| >= |T|}) / (R + 1)``.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape or len(x) != len(clusters):
        raise ValueError("a, b and clusters must have the same length")
    groups = _cluster_index(clusters)
    d = np.array([float((x[g] - y[g]).sum()) for g in groups])
    n = len(x)
    observed = d.sum() / n
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, len(d)))
    null = signs @ d / n
    p = (1 + int(np.sum(np.abs(null) >= abs(observed) - 1e-12))) / (permutations + 1)
    diff = x - y
    ci = cluster_bootstrap(lambda idx: float(diff[idx].mean()), clusters,
                           resamples=resamples, level=level, seed=seed + 1)
    return PairedTest(float(observed), float(p), permutations, ci)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float]:
    """Exact McNemar on item-level success (ignores families: sensitivity analysis only)."""
    only_a = sum(bool(x) and not bool(y) for x, y in zip(a, b, strict=True))
    only_b = sum(bool(y) and not bool(x) for x, y in zip(a, b, strict=True))
    p = 1.0 if only_a + only_b == 0 else float(binomtest(only_a, only_a + only_b, 0.5).pvalue)
    return {"only_a": only_a, "only_b": only_b, "p_value": p}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    order = sorted(p_values, key=p_values.__getitem__)
    m = len(order)
    out: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[key]))
        out[key] = running
    return out


# --------------------------------------------------------------------------- calibration
def expected_calibration_error(confidence: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """ECE with equal-width confidence bins (Guo et al. 2017)."""
    c = np.asarray(confidence, dtype=float)
    k = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(c, edges[1:-1], right=True), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            ece += m.mean() * abs(k[m].mean() - c[m].mean())
    return float(ece)


def brier_score(probabilities: NDArray[np.float64], gold_index: Sequence[int]) -> float:
    """Multi-class Brier score: mean squared distance to the one-hot gold vector."""
    p = np.asarray(probabilities, dtype=float)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(p)), np.asarray(gold_index)] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def aurc(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve: mean selective risk when answering the k most
    confident items, k = 1..n (lower is better)."""
    order = np.argsort(-np.asarray(confidence, dtype=float), kind="stable")
    errors = 1.0 - np.asarray(correct, dtype=float)[order]
    return float((np.cumsum(errors) / np.arange(1, len(errors) + 1)).mean())


# --------------------------------------------------------------------------- power
def simulate_power(n_families: int, family_size: int, p_a: float, p_b: float, *,
                   discordance: float = 1.0, alpha: float = 0.05, sims: int = 500,
                   permutations: int = 2000, seed: int = 0) -> float:
    """Power of :func:`paired_permutation_test` when system A succeeds with ``p_a`` and B with
    ``p_b`` on independent families of equal size.

    ``discordance`` in (0, 1] scales how often the systems disagree: 1 means independent
    outcomes, smaller values couple them (two systems that share most of the pipeline mostly
    succeed and fail on the same items, which raises power).
    """
    rng = np.random.default_rng(seed)
    clusters = [f"f{i}" for i in range(n_families) for _ in range(family_size)]
    n = n_families * family_size
    hits = 0
    for s in range(sims):
        shared = rng.random(n)
        a = shared < p_a
        b = shared < p_b
        redraw = rng.random(n) < discordance
        b = np.where(redraw, rng.random(n) < p_b, b)
        a = np.where(redraw, rng.random(n) < p_a, a)
        test = paired_permutation_test(a.astype(float).tolist(), b.astype(float).tolist(), clusters,
                                       permutations=permutations, resamples=1, seed=seed + s)
        hits += test.p_value < alpha
    return hits / sims
