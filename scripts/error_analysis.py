"""Error analysis of the locked test run (protocol section 7), after every system was scored.

The shared pipeline is deterministic given an intent, so each pipeline system's full output
(tool, instant, reminder text, clarification targets) is rebuilt from the intent it predicted
in the locked run. The rebuild must reproduce every stored item-level success exactly; the
script stops otherwise. Failures are then classified by cause.

    uv run python scripts/error_analysis.py --out results/error_analysis

Reads only finished locked-run outputs (results/test_results.json, results/s4/...), never
re-scores a system for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from athar.data.split import load_split
from athar.eval.metrics import _same_instant, _text_matches, score_item
from athar.extract import canonical_text
from athar.pipeline import respond_with_intent
from athar.schema import GoldItem

ROOT = Path(__file__).resolve().parents[1]
S4 = {"jais2-8b": "7b-others", "jais3-3.7b": "7a-jais3", "jais3-7b": "7a-jais3", "nile-4b": "7b-others",
      "qwen3-8b": "7b-others"}


def stored_items() -> dict[str, list[dict]]:
    """System name -> stored item scores of its locked test run (core, S4 primary + inline, S7 prior)."""
    out: dict[str, list[dict]] = {}
    core = json.loads((ROOT / "results" / "test_results.json").read_text(encoding="utf-8"))
    for n in ("S1", "S2", "S3", "oracle"):
        out[n] = core["systems"][n]["items"]
    for key, d in S4.items():
        rep = json.loads((ROOT / "results" / "s4" / d / "s4_results" / f"{key}.json").read_text(encoding="utf-8"))
        for name in (f"S4-{key}-6shot-prior", f"S4-{key}-6shot-inline-prior"):
            out[name] = rep["systems"][name]["items"]
    for seed in (0, 1, 2):
        p = ROOT / "results" / "s4" / "7cd-s7-s8" / "s4_results" / f"s7_jais2_qlora_s{seed}_test.json"
        name = f"S7-jais2-qlora-s{seed}-prior"
        out[name] = json.loads(p.read_text(encoding="utf-8"))["systems"][name]["items"]
    return out


def classify(g: GoldItem, resp) -> str:  # type: ignore[no-untyped-def]
    """Cause of a failure (the item is known to have failed)."""
    if resp is None:
        return "no valid output"
    if g.route == "ABSTAIN":
        return "executed out-of-scope request" if resp.route == "ACT" else "asked instead of declining"
    if g.route == "CLARIFY":
        if resp.route == "ACT":
            return "executed an incomplete request"
        if resp.route == "ABSTAIN":
            return "declined an in-scope request"
        if resp.intent != g.intent:
            return "clarified with the wrong intent"
        return "asked for the wrong slots"
    # gold ACT
    if resp.route == "CLARIFY":
        return "asked although complete"
    if resp.route == "ABSTAIN":
        return "declined an in-scope request"
    call = resp.tool_call
    if call.name != g.intent:
        return "wrong tool"
    instant_ok = _same_instant(call.datetime, g.args.datetime)
    text_ok = g.intent != "create_reminder" or _text_matches(g, call.reminder_text)
    if not instant_ok and not text_ok:
        return "wrong instant and wrong text"
    return "wrong instant" if not instant_ok else "wrong reminder text only"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "results" / "error_analysis"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    test = {g.id: g for g in load_split(ROOT / "data" / "splits" / "manifest.json", "test", unlock_test=True)}
    systems = stored_items()
    rows, summary = [], []
    for name, items in systems.items():
        causes: Counter[str] = Counter()
        for it in items:
            g = test[it["id"]]
            intent = g.intent if name == "oracle" else it["pred_intent"]
            resp = None if intent not in ("create_reminder", "create_alarm", "unsupported") else \
                respond_with_intent(intent, g.text, g.ref_datetime, g.timezone, name)
            rebuilt = score_item(g, resp)
            if rebuilt.success != it["success"] or rebuilt.pred_route != it["pred_route"]:
                print(f"REBUILD MISMATCH {name} {g.id}: stored {it['success']}/{it['pred_route']}, "
                      f"rebuilt {rebuilt.success}/{rebuilt.pred_route}")
                return 1
            if it["success"]:
                continue
            cause = classify(g, resp)
            if cause == "wrong reminder text only" and resp is not None:   # gate-free pipeline: no gating here
                pass
            causes[cause] += 1
            call = resp.tool_call if resp is not None else None
            rows.append({
                "system": name, "id": g.id, "family_id": g.family_id, "text": g.text, "gold_route": g.route,
                "gold_intent": g.intent, "pred_intent": intent, "pred_route": resp.route if resp else "invalid",
                "cause": cause,
                "gold_datetime": g.args.datetime.isoformat() if g.args.datetime else "",
                "pred_datetime": call.datetime.isoformat() if call is not None and call.datetime else "",
                "gold_text": g.args.reminder_text or "", "pred_text": (call.reminder_text or "") if call else "",
                "gold_canonical": canonical_text(g.args.reminder_text) if g.args.reminder_text else "",
                "pred_canonical": canonical_text(call.reminder_text) if call is not None and call.reminder_text else "",
                "gold_targets": "|".join(g.clarify_targets), "pred_targets": "|".join(resp.clarify_targets) if resp else "",
            })
        n_exec_wrong = causes["wrong instant"] + causes["wrong instant and wrong text"] + causes["wrong tool"]
        summary.append({"system": name, "n": len(items), "failures": sum(causes.values()),
                        "harmful_executions": n_exec_wrong + causes["executed an incomplete request"]
                        + causes["executed out-of-scope request"],
                        **dict(sorted(causes.items()))})
        print(f"{name:32} rebuilt {len(items)} items exactly; failures {sum(causes.values()):3d}: {dict(causes)}")

    fields = sorted({k for r in summary for k in r} - {"system", "n", "failures", "harmful_executions"})
    with open(out / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "n", "failures", "harmful_executions", *fields], restval=0)
        w.writeheader()
        w.writerows(summary)
    with open(out / "failures.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out / "summary.csv", "and", out / "failures.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
