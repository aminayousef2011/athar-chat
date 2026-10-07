"""``athar-data``: command line for the dataset tooling.

    athar-data validate  DATA.jsonl [--strict]
    athar-data split     DATA.jsonl --config CFG.yaml --out MANIFEST.json [--allow-shortfall]
    athar-data verify    MANIFEST.json
    athar-data crosscheck (--manifest MANIFEST.json --split train|dev | --pilot PILOT.jsonl)
    athar-data agreement-sheet DATA.jsonl --out SHEET.csv [--n 60] [--seed N]
    athar-data kappa     DATA.jsonl SHEET.csv --annotator NAME [--date YYYY-MM-DD] [--out PREFIX]
    athar-data stats     DATA.jsonl [--manifest MANIFEST.json] [--card docs/data_card.md]

Exit status is 1 when a check fails, so the commands can gate a commit or a CI job.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from athar.data import agreement as ag
from athar.data.crosscheck import crosscheck
from athar.data.io import DatasetError, load_items, read_records
from athar.data.split import (
    SplitConfig,
    load_split,
    make_split,
    read_manifest,
    verify_manifest,
    write_manifest,
)
from athar.data.stats import stats_markdown, write_into_card
from athar.data.validate import Issue, has_errors, validate_records


def _print_issues(issues: Sequence[Issue]) -> None:
    for i in issues:
        print(i)


def cmd_validate(a: argparse.Namespace) -> int:
    records = read_records(a.data)
    issues = validate_records(records)
    _print_issues(issues)
    n_err = sum(i.level == "error" for i in issues)
    print(f"{a.data}: {len(records)} items, {n_err} errors, {len(issues) - n_err} warnings")
    return 1 if has_errors(issues, a.strict) else 0


def cmd_split(a: argparse.Namespace) -> int:
    records = read_records(a.data)
    issues = validate_records(records)
    if has_errors(issues):
        _print_issues([i for i in issues if i.level == "error"])
        print("fix the errors above before splitting")
        return 1
    items = [r.item for r in records if r.item is not None]
    raws = {r.raw["id"]: r.raw for r in records}
    manifest, shortfall, notes = make_split(items, raws, SplitConfig.load(a.config), Path(a.data),
                                            Path(a.out))
    for msg in shortfall:
        print(f"SHORTFALL {msg}")
    for msg in notes:
        print(f"NOTE      {msg}")
    if shortfall and not a.allow_shortfall:
        print("test targets not met; nothing written (use --allow-shortfall to write anyway)")
        return 1
    write_manifest(manifest, a.out)
    for name, s in manifest["splits"].items():
        counts = " / ".join(f"{n} {r}" for r, n in s["counts"].items())
        print(f"{name:5}: {len(s['items']):3} items, {len(s['families']):3} families ({counts})")
    print(f"wrote {a.out}  manifest_sha256={manifest['manifest_sha256']}")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    issues = verify_manifest(a.manifest)
    _print_issues(issues)
    if not issues:
        m = read_manifest(a.manifest)
        print(f"{a.manifest}: OK  manifest_sha256={m['manifest_sha256']}")
    return 1 if issues else 0


def cmd_crosscheck(a: argparse.Namespace) -> int:
    if a.pilot:
        items = load_items(a.pilot)
        if any(it.source != "assistant_draft" for it in items):
            print("--pilot only takes files made of assistant_draft items. For annotated data, "
                  "split first and crosscheck train or dev through --manifest.")
            return 2
    else:
        items = load_split(a.manifest, a.split)  # test raises LockedSplitError
    diffs = crosscheck(items)
    for d in diffs:
        print(f"{d.id:14} {d.field:16} gold={d.gold}  core={d.system}")
    n_items = len({d.id for d in diffs})
    print(f"{len(items)} items, {n_items} with a disagreement between gold and the shared core")
    return 0


def cmd_sheet(a: argparse.Namespace) -> int:
    items = ag.sample_items(load_items(a.data), a.n, a.seed)
    ag.write_sheet(items, a.out)
    print(f"wrote {len(items)} items to {a.out} (seed {a.seed}); routes: "
          + ", ".join(f"{r} {sum(it.route == r for it in items)}" for r in ("ACT", "CLARIFY", "ABSTAIN")))
    return 0


def cmd_kappa(a: argparse.Namespace) -> int:
    gold = load_items(a.data)
    labels, errors = ag.read_labels(a.sheet)
    for e in errors:
        print(f"WARNING {e}")
    rep = ag.agreement(gold, labels)
    md = ag.report_markdown(rep, a.annotator, a.date)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(f"{a.out}.md").write_text(md, encoding="utf-8", newline="\n")
        payload = {"annotator": a.annotator, "date": a.date, "sheet": str(a.sheet), **rep}
        Path(f"{a.out}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8", newline="\n")
        print(f"wrote {a.out}.md and {a.out}.json")
    return 0


def cmd_stats(a: argparse.Namespace) -> int:
    items = load_items(a.data)
    splits = None
    if a.manifest:
        m = read_manifest(a.manifest)
        by_id = {it.id: it for it in items}
        # counts only: test items are not opened, just looked up by id
        splits = {name: [by_id[i] for i in s["items"] if i in by_id] for name, s in m["splits"].items()}
    md = stats_markdown(items, splits, title=f"`{Path(a.data).as_posix()}`")
    if a.card:
        write_into_card(a.card, md)
        print(f"updated {a.card}")
    else:
        print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="athar-data", description="ATHAR Chat dataset tooling")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate", help="schema + dataset-level checks")
    s.add_argument("data")
    s.add_argument("--strict", action="store_true", help="treat warnings as errors")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("split", help="family-level split, writes a hashed manifest")
    s.add_argument("data")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--allow-shortfall", action="store_true")
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("verify", help="check a manifest against its dataset")
    s.add_argument("manifest")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("crosscheck", help="gold vs. shared core with gold intent (never on test)")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest")
    g.add_argument("--pilot")
    s.add_argument("--split", choices=["train", "dev"], default="dev")
    s.set_defaults(func=cmd_crosscheck)

    s = sub.add_parser("agreement-sheet", help="blind CSV sample for a second annotator")
    s.add_argument("data")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=20261007)
    s.set_defaults(func=cmd_sheet)

    s = sub.add_parser("kappa", help="agreement + Cohen's kappa against the gold labels")
    s.add_argument("data")
    s.add_argument("sheet")
    s.add_argument("--annotator", required=True, help="who filled the sheet (or 'self-review')")
    s.add_argument("--date", default=dt.date.today().isoformat())
    s.add_argument("--out", help="write OUT.md and OUT.json")
    s.set_defaults(func=cmd_kappa)

    s = sub.add_parser("stats", help="dataset statistics (Markdown)")
    s.add_argument("data")
    s.add_argument("--manifest")
    s.add_argument("--card", help="write into this data card's stats block")
    s.set_defaults(func=cmd_stats)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Arabic output on a Windows console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (DatasetError, PermissionError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
