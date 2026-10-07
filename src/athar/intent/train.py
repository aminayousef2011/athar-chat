"""Phase 3: dev-only search, training on train, freezing, CPU/RAM log.

    athar-train fit    --manifest data/splits/manifest.json --config configs/train.yaml --out models
    athar-train verify models

``fit`` reads only the train and dev splits (the test split is locked by the loader). For S2
and S3 every grid point is fitted on train and scored on dev; the best one by dev intent
macro-F1 (ties: smaller C, then shorter n-grams) is refitted on train under measurement and
saved. ``models/freeze.json`` records the manifest hash, the config and its hash, the SHA-256
of every model file and of the shared-core and S1 source files, library versions, dev scores
and resource use. ``verify`` (and :func:`load_recognizer`) refuse anything that changed since.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy
import psutil
import sklearn
import yaml
from sklearn.metrics import accuracy_score, f1_score

import athar
from athar import CONTRACT_VERSION
from athar.data.io import DatasetError, canonical_json, sha256_file, sha256_text
from athar.data.split import load_split, read_manifest, verify_manifest
from athar.intent.models import SklearnIntent, build
from athar.intent.rules import KeywordIntent
from athar.normalize import norm
from athar.schema import GoldItem

LEARNED = ("S2", "S3")
SYSTEMS = ("S1", *LEARNED)
PKG = Path(athar.__file__).resolve().parent
REPO = PKG.parents[1]
#: repo-relative files frozen before the test split is opened: everything shared by all
#: systems, S1's rules and the S2/S3 builders, the scoring/statistics code, and the documents
#: and settings that pre-register the evaluation
FROZEN_SOURCES = tuple(
    [f"src/athar/{m}.py" for m in ("normalize", "lexicon", "timeexpr", "extract", "policy",
                                   "respond", "schema", "pipeline")]
    + [f"src/athar/intent/{m}.py" for m in ("rules", "models", "neural", "llm")]
    + [f"src/athar/eval/{m}.py" for m in ("metrics", "stats", "perturb", "harness")]
    + ["docs/task_contract.md", "docs/evaluation_protocol.md", "configs/eval.yaml"]
)


class FreezeError(RuntimeError):
    """Frozen artefacts are missing or changed since they were frozen."""


# --------------------------------------------------------------------------- hashing
def source_hash(rel: str) -> str:
    """Line-ending independent, so a CRLF checkout does not look like a code change."""
    return sha256_text((REPO / rel).read_text(encoding="utf-8").replace("\r\n", "\n"))


def source_hashes() -> dict[str, str]:
    return {rel: source_hash(rel) for rel in FROZEN_SOURCES}


# --------------------------------------------------------------------------- resources
def _peak_rss_mb() -> float:
    if sys.platform == "win32":
        peak = float(psutil.Process().memory_info().peak_wset)
    else:
        import resource  # POSIX only

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = float(ru if sys.platform == "darwin" else ru * 1024)
    return round(peak / 2**20, 1)


@contextmanager
def measure() -> Iterator[dict[str, float]]:
    """CPU time, wall time and peak traced allocation (Python + numpy) of the block."""
    out: dict[str, float] = {}
    tracemalloc.start()
    c0, w0 = time.process_time(), time.perf_counter()
    try:
        yield out
    finally:
        out["cpu_seconds"] = round(time.process_time() - c0, 4)
        out["wall_seconds"] = round(time.perf_counter() - w0, 4)
        out["peak_traced_mb"] = round(tracemalloc.get_traced_memory()[1] / 2**20, 2)
        tracemalloc.stop()


def environment() -> dict[str, Any]:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "scikit-learn": sklearn.__version__, "numpy": numpy.__version__,
            "joblib": joblib.__version__}


# --------------------------------------------------------------------------- search
def scores(gold: Sequence[str], pred: Sequence[str]) -> dict[str, float]:
    # macro-F1 over the labels present in gold or predictions (sklearn's default)
    return {"intent_macro_f1": round(float(f1_score(gold, pred, average="macro", zero_division=0)), 4),
            "intent_accuracy": round(float(accuracy_score(gold, pred)), 4)}


def grid(system: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    space = cfg[system]
    keys = sorted(space)
    return [{**cfg.get("common", {}), **dict(zip(keys, combo, strict=True))}
            for combo in itertools.product(*(space[k] for k in keys))]


def _simplicity(p: dict[str, Any]) -> tuple[float, int, int]:
    return (p["C"], p["word_ngram"][1], p.get("char_ngram", [0, 0])[1])


def search(system: str, cfg: dict[str, Any], train: Sequence[GoldItem],
           dev: Sequence[GoldItem]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    xtr, ytr = [norm(it.text) for it in train], [it.intent for it in train]
    xdv, ydv = [norm(it.text) for it in dev], [it.intent for it in dev]
    metric = cfg.get("selection_metric", "intent_macro_f1")
    results = []
    for params in grid(system, cfg):
        model = build(system, params, cfg.get("seed", 0)).fit(xtr, ytr)
        results.append({"params": params, "dev": scores(ydv, list(model.predict(xdv)))})
    best = min(results, key=lambda r: (-r["dev"][metric], *_simplicity(r["params"])))
    return best["params"], results


# --------------------------------------------------------------------------- fit + freeze
def _timed_predict(predict: Callable[[list[str]], list[str]], texts: list[str]) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    pred = predict(texts)
    return pred, round(1000 * (time.perf_counter() - t0) / max(len(texts), 1), 4)


def _dir_hashes(directory: Path, root: Path) -> dict[str, str]:
    return {f.relative_to(root).as_posix(): sha256_file(f) for f in sorted(directory.rglob("*")) if f.is_file()}


def _train_extended(cfg: dict[str, Any], xtr: list[str], ytr: Sequence[str], xdv: list[str], ydv: Sequence[str],
                    out: Path, log: Callable[[str], None], encoder_factory: Callable[[str, str], Any] | None,
                    pretrained_factory: Callable[[str, str], tuple[Any, Any]] | None) -> dict[str, Any]:
    """S5 and S6 (extended baselines, reported separately). Needs the ``neural`` extra."""
    from athar.intent import neural

    systems: dict[str, Any] = {}
    if "S5" in cfg:
        c5 = {"seed": cfg.get("seed", 0), **cfg["S5"]}
        with measure() as enc_res:
            encoder = (encoder_factory or neural.load_encoder)(c5["model"], c5["revision"])
        with measure() as fit_res:
            model, best, results = neural.fit_s5(encoder, xtr, ytr, xdv, ydv, c5, scores)
        pred, ms = _timed_predict(model.predict, xdv)
        path = out / "S5.joblib"
        neural.save_s5(model, path)
        systems["S5"] = {"kind": "frozen sentence embeddings + logistic regression",
                         "model": c5["model"], "revision": c5["revision"], "params": best,
                         "dev": scores(ydv, pred), "files": {path.name: sha256_file(path)},
                         "resources": {"load_encoder": enc_res,
                                       "search_and_fit": {**fit_res, "n_fits": len(results)},
                                       "predict_ms_per_item": ms}, "grid": results}
        log(f"S5  dev {systems['S5']['dev']}  C={best['C']}")
    if "S6" in cfg:
        c6 = cfg["S6"]
        for seed in c6["seeds"]:
            name = f"S6-s{seed}"
            net, tok = (pretrained_factory or neural.load_pretrained)(c6["model"], c6["revision"])
            unk = neural.unk_rate(tok, xtr)
            if unk > c6.get("max_unk_rate", 0.05):
                raise ValueError(f"S6 tokenizer maps {unk:.1%} of training word pieces to its unknown "
                                 f"token; the tokenizer or vocabulary is wrong")
            with measure() as fit_res:
                model6 = neural.fine_tune(net, tok, xtr, ytr, c6, seed, log)
            pred, ms = _timed_predict(model6.predict, xdv)
            directory = out / "S6" / f"seed{seed}"
            neural.save_s6(model6, directory)
            systems[name] = {"kind": "fine-tuned encoder", "model": c6["model"], "revision": c6["revision"],
                             "seed": seed, "dir": directory.relative_to(out).as_posix(),
                             "dev": scores(ydv, pred), "files": _dir_hashes(directory, out),
                             "train_unk_rate": round(unk, 4),
                             "resources": {"fine_tune": {**fit_res, "process_peak_rss_mb": _peak_rss_mb()},
                                           "predict_ms_per_item": ms}}
            log(f"{name}  dev {systems[name]['dev']}  fine-tune {fit_res['wall_seconds']}s wall")
        f1s = [systems[f"S6-s{s}"]["dev"]["intent_macro_f1"] for s in c6["seeds"]]
        systems["S6"] = {"kind": "seed summary", "seeds": list(c6["seeds"]),
                         "dev_intent_macro_f1_mean": round(float(numpy.mean(f1s)), 4),
                         "dev_intent_macro_f1_sd": round(float(numpy.std(f1s, ddof=1)), 4) if len(f1s) > 1 else 0.0}
    return systems


def fit_and_freeze(manifest_path: str | Path, config_path: str | Path, out: str | Path,
                   log: Callable[[str], None] = print, *, extended: bool = False,
                   encoder_factory: Callable[[str, str], Any] | None = None,
                   pretrained_factory: Callable[[str, str], tuple[Any, Any]] | None = None) -> dict[str, Any]:
    problems = verify_manifest(manifest_path)
    if problems:
        raise DatasetError(f"manifest does not verify: {problems[0]}")
    manifest = read_manifest(manifest_path)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    train, dev = load_split(manifest_path, "train"), load_split(manifest_path, "dev")
    if not train or not dev:
        raise DatasetError("train and dev must both be non-empty")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    xdv, ydv = [norm(it.text) for it in dev], [it.intent for it in dev]
    xtr, ytr = [norm(it.text) for it in train], [it.intent for it in train]

    systems: dict[str, Any] = {}
    s1 = KeywordIntent()
    pred, ms = _timed_predict(s1.predict, xdv)
    systems["S1"] = {"kind": "keyword rules", "dev": scores(ydv, pred), "predict_ms_per_item": ms}
    log(f"S1  dev {systems['S1']['dev']}")

    for name in LEARNED:
        with measure() as search_res:
            best, results = search(name, cfg, train, dev)
        with measure() as fit_res:
            model = build(name, best, cfg.get("seed", 0)).fit(xtr, ytr)
        recognizer = SklearnIntent(name, model)
        pred, ms = _timed_predict(recognizer.predict, xdv)
        path = out / f"{name}.joblib"
        joblib.dump(model, path)
        systems[name] = {
            "kind": "tf-idf + logistic regression", "params": best, "dev": scores(ydv, pred),
            "file": path.name, "sha256": sha256_file(path), "files": {path.name: sha256_file(path)},
            "size_kb": round(path.stat().st_size / 1024, 1),
            "n_features": int(len(model.named_steps["clf"].coef_[0])),
            "resources": {"final_fit": fit_res, "search": {**search_res, "n_fits": len(results)},
                          "predict_ms_per_item": ms},
            "grid": results,
        }
        log(f"{name}  dev {systems[name]['dev']}  C={best['C']} word={best['word_ngram']}"
            + (f" char={best['char_ngram']}" if "char_ngram" in best else "")
            + f"  fit {fit_res['cpu_seconds']}s CPU, {fit_res['peak_traced_mb']} MB peak")

    if extended:
        systems.update(_train_extended(cfg, xtr, ytr, xdv, ydv, out, log, encoder_factory, pretrained_factory))

    body = {
        "contract_version": CONTRACT_VERSION,
        "manifest": {"path": Path(manifest_path).as_posix(), "sha256": manifest["manifest_sha256"],
                     "n_train": len(train), "n_dev": len(dev)},
        "config": cfg, "config_sha256": sha256_text(canonical_json(cfg)),
        "frozen_sources": source_hashes(),
        "environment": environment(),
        "process_peak_rss_mb": _peak_rss_mb(),
        "systems": systems,
    }
    freeze = {**body, "freeze_sha256": sha256_text(canonical_json(body))}
    (out / "freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")
    return freeze


# --------------------------------------------------------------------------- verify + load
def read_freeze(models_dir: str | Path) -> dict[str, Any]:
    p = Path(models_dir) / "freeze.json"
    if not p.exists():
        raise FreezeError(f"{p} not found; run `athar-train fit` first")
    f: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    body = {k: v for k, v in f.items() if k != "freeze_sha256"}
    if sha256_text(canonical_json(body)) != f.get("freeze_sha256"):
        raise FreezeError(f"{p} was edited after freezing")
    return f


def verify_freeze(models_dir: str | Path) -> list[str]:
    try:
        f = read_freeze(models_dir)
    except FreezeError as e:
        return [str(e)]
    problems = []
    for name in LEARNED:
        if name not in f["systems"]:
            problems.append(f"{name}: model file missing")
    for name, s in f["systems"].items():
        for rel, h in s.get("files", {}).items():
            path = Path(models_dir) / rel
            if not path.exists():
                problems.append(f"{name}: model file missing ({rel})")
            elif sha256_file(path) != h:
                problems.append(f"{name}: {rel} changed after freezing")
    for rel, h in f["frozen_sources"].items():
        if not (REPO / rel).exists():
            problems.append(f"source {rel} is missing")
        elif source_hash(rel) != h:
            problems.append(f"source {rel} changed after freezing")
    return problems


def frozen_systems(models_dir: str | Path) -> list[str]:
    """Loadable system names in a freeze (S6 is loaded per seed: S6-s0, S6-s1, ...)."""
    return [n for n, s in read_freeze(models_dir)["systems"].items() if s.get("kind") != "seed summary"]


def load_recognizer(name: str, models_dir: str | Path = "models", *, strict: bool = True) -> Any:
    """A frozen system. Model files are pickles: they are always hash-checked before loading.

    ``strict`` (the default, and what the locked test run uses) also requires every frozen
    source (shared core, scoring code, protocol, contract) to be unchanged. Dev evaluations may
    pass ``strict=False`` to re-score frozen models after a fix to scoring code."""
    problems = verify_freeze(models_dir)
    if not strict:
        problems = [p for p in problems if not p.startswith("source ")]
    if problems:
        raise FreezeError("; ".join(problems))
    f = read_freeze(models_dir)
    if name not in SYSTEMS and name not in f["systems"]:
        raise ValueError(f"unknown system {name!r}; choose from {frozen_systems(models_dir)}")
    if name == "S1":
        return KeywordIntent()
    if name in LEARNED:
        return SklearnIntent(name, joblib.load(Path(models_dir) / f["systems"][name]["file"]))
    from athar.intent import neural

    s = f["systems"][name]
    if name == "S5":
        return neural.load_s5(Path(models_dir) / "S5.joblib", f["config"]["S5"])
    if name.startswith("S6-s"):
        return neural.load_s6(Path(models_dir) / s["dir"], f["config"]["S6"]["max_length"], name)
    raise ValueError(f"{name!r} is not a loadable system")


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="athar-train", description="ATHAR Chat Phase 3: search + freeze")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fit", help="dev-only search, fit on train, freeze")
    s.add_argument("--manifest", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--out", default="models")
    s.add_argument("--force", action="store_true", help="overwrite an existing freeze")
    s.add_argument("--extended", action="store_true",
                   help="also train S5/S6 (needs the neural extra: uv sync --extra neural)")
    s = sub.add_parser("verify", help="check frozen models and sources")
    s.add_argument("models_dir", nargs="?", default="models")
    a = p.parse_args(argv)
    try:
        if a.cmd == "fit":
            if (Path(a.out) / "freeze.json").exists() and not a.force:
                print(f"{a.out}/freeze.json exists; models are frozen once (use --force to refit)")
                return 1
            f = fit_and_freeze(a.manifest, a.config, a.out, extended=a.extended)
            print(f"froze {', '.join(frozen_systems(a.out))} in {a.out}  freeze_sha256={f['freeze_sha256']}")
            return 0
        problems = verify_freeze(a.models_dir)
        for msg in problems:
            print(f"ERROR {msg}")
        if not problems:
            print(f"{a.models_dir}: OK  freeze_sha256={read_freeze(a.models_dir)['freeze_sha256']}")
        return 1 if problems else 0
    except (DatasetError, FreezeError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
