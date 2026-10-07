"""Family-level train/dev/test split with a hashed manifest, and the locked split loader.

Rules (task contract, HLD invariants):
- a scenario family (paraphrases, complete/incomplete pairs) is never divided between splits;
- the test split hits per-route targets (proposal: 50 ACT / 50 CLARIFY / 20 ABSTAIN);
- dev takes ``dev`` items with the route mix of what remains; train is the rest;
- a family with any ``assistant_draft`` item never enters test;
- the same dataset + config always gives the same manifest (seeded, no timestamps).

The manifest stores a SHA-256 per item, the dataset file hash, the config and the contract
version, and is itself hashed. :func:`load_split` refuses the test split unless the caller
unlocks it explicitly, and refuses any item that changed after the split was made.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from athar import CONTRACT_VERSION
from athar.data.io import (
    DatasetError,
    canonical_json,
    item_hash,
    read_records,
    sha256_file,
    sha256_text,
)
from athar.data.validate import Issue
from athar.schema import ROUTES, GoldItem

SPLITS = ("train", "dev", "test")
MANIFEST_VERSION = 1


class LockedSplitError(PermissionError):
    """The test split is opened once, by the locked evaluation run (Phase 4)."""


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    test: dict[str, int]  # per-route item targets
    dev: int  # item target; route mix follows the remaining pool
    trials: int = 2000

    @classmethod
    def load(cls, path: str | Path) -> SplitConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        test = {r: int(raw.get("test", {}).get(r, 0)) for r in ROUTES}
        unknown = set(raw.get("test", {})) - set(ROUTES)
        if unknown:
            raise ValueError(f"unknown routes in test targets: {sorted(unknown)}")
        return cls(seed=int(raw["seed"]), test=test, dev=int(raw.get("dev", 0)),
                   trials=int(raw.get("trials", 2000)))


Vec = tuple[int, int, int]  # items per route, in ROUTES order


def _vec(items: Sequence[GoldItem]) -> Vec:
    c = Counter(it.route for it in items)
    return tuple(c[r] for r in ROUTES)  # type: ignore[return-value]


def _largest_remainder(total: int, weights: Vec) -> Vec:
    s = sum(weights)
    if s == 0 or total == 0:
        return (0, 0, 0)
    exact = [total * w / s for w in weights]
    out = [int(x) for x in exact]
    for i in sorted(range(3), key=lambda i: exact[i] - out[i], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return tuple(out)  # type: ignore[return-value]


def _pick(families: Sequence[str], vecs: dict[str, Vec], target: Vec, rng: random.Random,
          trials: int) -> list[str]:
    """Randomised greedy packing: never overshoot a route target, minimise the shortfall."""
    best: list[str] = []
    best_short: int | None = None
    if sum(target) == 0:
        return best
    for _ in range(trials):
        order = list(families)
        rng.shuffle(order)
        have = [0, 0, 0]
        chosen = []
        for f in order:
            v = vecs[f]
            if all(h + x <= t for h, x, t in zip(have, v, target, strict=True)):
                chosen.append(f)
                have = [h + x for h, x in zip(have, v, strict=True)]
                if tuple(have) == target:
                    break
        short = sum(target) - sum(have)
        if best_short is None or short < best_short:
            best, best_short = sorted(chosen), short
        if short == 0:
            break
    return best


def _fmt_vec(v: Vec) -> str:
    return " / ".join(f"{n} {r}" for r, n in zip(ROUTES, v, strict=True))


def make_split(items: Sequence[GoldItem], raws: dict[str, dict[str, Any]], config: SplitConfig,
               dataset_path: Path, manifest_path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    """Return (manifest, shortfall, notes). Items must already pass validation.

    Missing a test target is a shortfall (blocking). Dev's route mix is a soft goal: families
    come in fixed shapes (an ACT+CLARIFY pair adds one of each), so an exact proportional mix
    is often unreachable, and a deviation is only noted.
    """
    fams: dict[str, list[GoldItem]] = defaultdict(list)
    for it in items:
        fams[it.family_id].append(it)
    vecs = {f: _vec(m) for f, m in fams.items()}
    names = sorted(fams)
    rng = random.Random(config.seed)

    eligible = [f for f in names if all(it.source != "assistant_draft" for it in fams[f])]
    test_target: Vec = tuple(config.test[r] for r in ROUTES)  # type: ignore[assignment]
    test = _pick(eligible, vecs, test_target, rng, config.trials)
    rest = [f for f in names if f not in set(test)]
    pool: Vec = tuple(sum(vecs[f][i] for f in rest) for i in range(3))  # type: ignore[assignment]
    dev_target = _largest_remainder(min(config.dev, sum(pool)), pool)
    dev = _pick(rest, vecs, dev_target, rng, config.trials)
    assigned = {**{f: "test" for f in test}, **{f: "dev" for f in dev}}
    train = [f for f in rest if f not in assigned]

    def got(chosen: list[str]) -> Vec:
        return tuple(sum(vecs[f][i] for f in chosen) for i in range(3))  # type: ignore[return-value]

    shortfall, notes = [], []
    if got(test) != test_target:
        shortfall.append(f"test: got {_fmt_vec(got(test))}, target {_fmt_vec(test_target)}")
        if n_draft := len(names) - len(eligible):
            shortfall.append(f"{n_draft} famil{'y' if n_draft == 1 else 'ies'} with "
                             f"assistant_draft items cannot enter test")
    if got(dev) != dev_target:
        notes.append(f"dev: got {_fmt_vec(got(dev))}, proportional target {_fmt_vec(dev_target)}")
    if config.dev > sum(pool):
        notes.append(f"dev: only {sum(pool)} items left after test, target {config.dev}")

    splits: dict[str, Any] = {}
    for name, fam_list in (("train", train), ("dev", dev), ("test", test)):
        members = [it for f in sorted(fam_list) for it in sorted(fams[f], key=lambda x: x.id)]
        splits[name] = {
            "families": sorted(fam_list),
            "counts": dict(zip(ROUTES, _vec(members), strict=True)),
            "items": {it.id: item_hash(raws[it.id]) for it in members},
        }
    body = {
        "manifest_version": MANIFEST_VERSION,
        "contract_version": CONTRACT_VERSION,
        "dataset": {
            "path": Path(os.path.relpath(dataset_path.resolve(), manifest_path.resolve().parent)).as_posix(),
            "sha256": sha256_file(dataset_path),
            "n_items": len(items),
            "n_families": len(names),
        },
        "config": asdict(config),
        "splits": splits,
    }
    return {**body, "manifest_sha256": sha256_text(canonical_json(body))}, shortfall, notes


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # LF on every OS: .gitattributes keeps data files LF, so hashes match a fresh checkout
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")


def read_manifest(path: str | Path) -> dict[str, Any]:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    if sha256_text(canonical_json(body)) != m.get("manifest_sha256"):
        raise DatasetError(f"{path}: manifest hash mismatch; the manifest was edited by hand")
    return m


def dataset_path_of(manifest: dict[str, Any], manifest_path: str | Path) -> Path:
    return Path(manifest_path).resolve().parent / manifest["dataset"]["path"]


def _raw_by_id(dataset: Path) -> dict[str, dict[str, Any]]:
    return {r.raw["id"]: r.raw for r in read_records(dataset) if isinstance(r.raw.get("id"), str)}


def verify_manifest(manifest_path: str | Path) -> list[Issue]:
    """Check a manifest against its dataset. Hash or family violations are errors."""
    where = str(manifest_path)
    try:
        m = read_manifest(manifest_path)
    except DatasetError as e:
        return [Issue("error", where, str(e))]
    dataset = dataset_path_of(m, manifest_path)
    if not dataset.exists():
        return [Issue("error", where, f"dataset {dataset} not found")]
    issues: list[Issue] = []
    if sha256_file(dataset) != m["dataset"]["sha256"]:
        issues.append(Issue("error", where, "dataset file changed since the split was made"))
    raws = _raw_by_id(dataset)
    split_of: dict[str, str] = {}
    fam_split: dict[str, str] = {}
    for name in SPLITS:
        for iid, h in m["splits"][name]["items"].items():
            if iid in split_of:
                issues.append(Issue("error", where, f"{iid} is in {split_of[iid]} and {name}"))
            split_of[iid] = name
            raw = raws.get(iid)
            if raw is None:
                issues.append(Issue("error", where, f"{name} item {iid} is missing from the dataset"))
                continue
            if item_hash(raw) != h:
                issues.append(Issue("error", where, f"{name} item {iid} changed after the split"))
            fam = raw.get("family_id")
            if fam_split.setdefault(fam, name) != name:
                issues.append(Issue("error", where, f"family {fam} is in {fam_split[fam]} and {name}"))
            if name == "test" and raw.get("source") == "assistant_draft":
                issues.append(Issue("error", where, f"assistant_draft item {iid} is in test"))
    for iid in sorted(set(raws) - set(split_of)):
        issues.append(Issue("error", where, f"dataset item {iid} is in no split"))
    return issues


def load_split(manifest_path: str | Path, split: str, *, unlock_test: bool = False) -> list[GoldItem]:
    """Items of one split, in manifest order, after checking each against its recorded hash."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "test" and not unlock_test:
        raise LockedSplitError("the test split is locked until the Phase 4 evaluation run; "
                              "pass unlock_test=True only there")
    m = read_manifest(manifest_path)
    raws = _raw_by_id(dataset_path_of(m, manifest_path))
    out = []
    for iid, h in m["splits"][split]["items"].items():
        raw = raws.get(iid)
        if raw is None or item_hash(raw) != h:
            raise DatasetError(f"{split} item {iid} is missing or changed since the split; "
                               f"run `athar-data verify {manifest_path}`")
        out.append(GoldItem.model_validate(raw))
    return out
