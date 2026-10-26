"""S2-gcv / S3-gcv and the v2 freeze (docs/protocol_natural.md, section 5).

S2/S3 as in v1 (same builders, grid, selection metric and tie rule from configs/train.yaml), with
one change: the configuration is selected by 5-fold ``GroupKFold`` over the train+dev scenario
families (mean intent macro-F1 over folds) instead of on the 40-item dev split, then refitted on
train+dev. They are paired with core v1.1 at evaluation time and are not gated.

``freeze`` writes ``models_v2/``: the two models and ``freeze.json`` with their hashes, the
selection table, and the hashes of the v2 sources and of every v1 source v2 builds on. ``verify``
and ``load_recognizer`` refuse anything that changed since.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.model_selection import GroupKFold

from athar.data.io import canonical_json, sha256_file, sha256_text
from athar.data.split import load_split, read_manifest
from athar.intent.models import SklearnIntent, build
from athar.intent.train import FROZEN_SOURCES, REPO, _simplicity, grid, scores, source_hash
from athar.normalize import norm
from athar.schema import GoldItem

SYSTEMS = ("S2", "S3")
FOLDS = 5
V2_SOURCES = tuple(f"src/athar/v2/{m}.py" for m in ("__init__", "core", "rules", "gcv", "hybrid", "natural"))


class FreezeError(RuntimeError):
    """The v2 freeze is missing or something in it changed."""


def select(system: str, cfg: dict[str, Any], pool: Sequence[GoldItem]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Best configuration by grouped cross-validation over ``pool`` (train+dev)."""
    x = np.array([norm(it.text) for it in pool], dtype=object)
    y = np.array([it.intent for it in pool], dtype=object)
    folds = list(GroupKFold(n_splits=FOLDS).split(x, y, [it.family_id for it in pool]))
    metric = cfg.get("selection_metric", "intent_macro_f1")
    results = []
    for params in grid(system, cfg):
        per_fold = []
        for tr, va in folds:
            model = build(system, params, cfg.get("seed", 0)).fit(list(x[tr]), list(y[tr]))
            per_fold.append(scores(list(y[va]), list(model.predict(list(x[va]))))[metric])
        results.append({"params": params, "cv": {metric: round(float(np.mean(per_fold)), 4),
                                                 "folds": [round(f, 4) for f in per_fold]}})
    best = min(results, key=lambda r: (-r["cv"][metric], *_simplicity(r["params"])))
    return best["params"], results


def _hashes() -> dict[str, str]:
    return {rel: source_hash(rel) for rel in (*V2_SOURCES, *FROZEN_SOURCES)}


def freeze(manifest: str | Path, config: str | Path, out: str | Path) -> dict[str, Any]:
    out = Path(out)
    if (out / "freeze.json").exists():
        raise FreezeError(f"{out}/freeze.json exists: the v2 systems are frozen; never overwrite a freeze")
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    pool = list(load_split(manifest, "train")) + list(load_split(manifest, "dev"))
    out.mkdir(parents=True, exist_ok=True)
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        params, table = select(system, cfg, pool)
        model = build(system, params, cfg.get("seed", 0)).fit([norm(it.text) for it in pool], [it.intent for it in pool])
        path = out / f"{system}-gcv.joblib"
        joblib.dump(model, path)
        systems[f"{system}-gcv"] = {"params": params, "selection": f"GroupKFold({FOLDS}) over train+dev families",
                                    "refit_on": "train+dev", "n_items": len(pool), "file": path.name,
                                    "sha256": sha256_file(path), "grid": table}
    body = {"protocol": "docs/protocol_natural.md", "core_version": "1.1",
            "manifest": read_manifest(manifest)["manifest_sha256"], "config_sha256": sha256_file(config),
            "sources": _hashes(), "systems": systems}
    freeze_json = {**body, "freeze_sha256": sha256_text(canonical_json(body))}
    (out / "freeze.json").write_text(json.dumps(freeze_json, ensure_ascii=False, indent=1) + "\n",
                                     encoding="utf-8", newline="\n")
    return freeze_json


def read_freeze(models_dir: str | Path) -> dict[str, Any]:
    f = Path(models_dir) / "freeze.json"
    if not f.exists():
        raise FreezeError(f"{f} not found: run `athar-v2 freeze` first")
    data: dict[str, Any] = json.loads(f.read_text(encoding="utf-8"))
    return data


def verify(models_dir: str | Path) -> list[str]:
    """Problems with the v2 freeze (empty when every model and source is unchanged)."""
    f = read_freeze(models_dir)
    body = {k: v for k, v in f.items() if k != "freeze_sha256"}
    problems = [] if sha256_text(canonical_json(body)) == f.get("freeze_sha256") else ["freeze.json was edited"]
    for name, s in f["systems"].items():
        path = Path(models_dir) / s["file"]
        if not path.exists() or sha256_file(path) != s["sha256"]:
            problems.append(f"model {name} missing or changed")
    for rel, h in f["sources"].items():
        if not (REPO / rel).exists() or source_hash(rel) != h:
            problems.append(f"source {rel} changed since the v2 freeze")
    return problems


def load_recognizer(name: str, models_dir: str | Path = "models_v2") -> SklearnIntent:
    """A frozen S2-gcv / S3-gcv recognizer (hash-checked before the pickle is loaded)."""
    if problems := verify(models_dir):
        raise FreezeError("; ".join(problems))
    s = read_freeze(models_dir)["systems"][name]
    return SklearnIntent(name, joblib.load(Path(models_dir) / s["file"]))
