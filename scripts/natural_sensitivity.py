"""Post hoc sensitivity of the natural-phrasing confirmatory results (report section 8.6).

    uv run --with openpyxl python scripts/natural_sensitivity.py

Rescores the frozen systems (CPU; S8/S9 from the committed traces) on the 238 confirmatory items
under alternative labels and scoring rules, and reruns the pre-registered paired tests (family-
level randomization, 10,000 permutations) for N1 (S1 vs S2) and N3 (S9 vs S8, seed means):

- locked gold (reproduces results/natural/evaluation.json);
- the author's labels before adjudication;
- the second annotator's labels, with only his format slips fixed as in adjudication
  (``scripts/adjudicate_natural.clean``); items whose label is then invalid are dropped;
- the locked gold with a lenient text rule (an ACT reminder with the right tool and instant counts
  as a success whatever its text);
- the locked gold without the items that contain expressions the contract does not resolve.

It also splits every ACT failure into wrong tool, wrong instant, text only, and not executed.
Nothing locked is changed. Output: results/natural/sensitivity.md and .json. Holm is not applied
across these analyses; they are sensitivity checks, not new hypotheses.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adjudicate_natural as adj  # noqa: E402

from athar.data.contrib import _openpyxl, build_dataset  # noqa: E402
from athar.data.io import read_records  # noqa: E402
from athar.eval import harness as H  # noqa: E402
from athar.eval.metrics import _same_instant, _text_matches  # noqa: E402
from athar.v2.natural import _cfg, _paired, systems  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAT = ROOT / "results" / "natural"
TRACES = [NAT / "kaggle/1-s8/s4_results" / f"s8_trace_S8-jais3-7b-s{k}.jsonl" for k in range(3)]
UNCOVERED = re.compile(r"بعد\s*(اسبوع|أسبوع|اسبوعين|أسبوعين|شهر|شهرين)|بعد\s+(ال)?جا?ي|بعد\s+اللي\s+جي|بدقايق")
NAMES = ("oracle", "S1", "S1-v1.1", "S2", "S3")


def outcome(gold, resp, lenient: bool = False) -> str:
    """success / wrong_tool / wrong_instant / text_only / not_executed (ACT); success / failure else."""
    if resp is None:
        return "not_executed" if gold.route == "ACT" else "failure"
    if gold.route == "ABSTAIN":
        return "success" if resp.route == "ABSTAIN" else "failure"
    if gold.route == "CLARIFY":
        ok = resp.route == "CLARIFY" and resp.intent == gold.intent and resp.clarify_targets == gold.clarify_targets
        return "success" if ok else "failure"
    call = resp.tool_call
    if resp.route != "ACT" or call is None:
        return "not_executed"
    if call.name != gold.intent:
        return "wrong_tool"
    if not _same_instant(call.datetime, gold.args.datetime):
        return "wrong_instant"
    if gold.intent == "create_reminder" and not _text_matches(gold, call.reminder_text):
        return "success" if lenient else "text_only"
    return "success"


def second_annotator_items(conf: set[str]) -> tuple[list, int]:
    """The second annotator's labels as items (format slips fixed as in adjudication)."""
    a, b = adj.read(adj.AUTHOR), adj.read(adj.OTHER)
    wb = _openpyxl().load_workbook(adj.AUTHOR)
    ws = wb["label"]
    col = {adj.cell(c.value): c.column for c in ws[1]}
    dropped = 0
    for r in ws.iter_rows(min_row=2):
        i = adj.cell(r[0].value)
        if i not in b:
            continue
        lab = adj.clean(b[i])
        bad = (lab["route"] == "CLARIFY" and not lab["clarify_targets"]) or (lab["route"] == "ACT" and not lab["datetime"])
        if bad:
            dropped += i in conf
            for f in adj.FIELDS:
                ws.cell(row=r[0].row, column=col[f]).value = None   # leaves the row invalid: build drops it
            continue
        for f in adj.FIELDS:
            ws.cell(row=r[0].row, column=col[f]).value = lab[f] or None
    assert set(a) == set(b)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        sheet, out = Path(tmp) / "second.xlsx", Path(tmp) / "second.jsonl"
        wb.save(sheet)
        build_dataset(sheet, out)
        items = [r.item for r in read_records(out) if r.item and r.item.id in conf]
    return items, dropped


def evaluate(items, sys_, cfg, lenient=False) -> dict:
    fams = [it.family_id for it in items]
    res: dict = {"n": len(items), "systems": {}, "act_failures": {}}
    scores = {}
    for n in [*NAMES, *(f"S8-s{k}" for k in range(3)), *(f"S9-s{k}" for k in range(3))]:
        outs = H.run(sys_[n], items)
        oc = [outcome(g, o, lenient) for g, o in zip(items, outs, strict=True)]
        scores[n] = [float(x == "success") for x in oc]
        act = [x for g, x in zip(items, oc, strict=True) if g.route == "ACT"]
        res["act_failures"][n] = {k: act.count(k) for k in ("wrong_tool", "wrong_instant", "text_only", "not_executed")}
    for base in ("S8", "S9"):
        scores[base] = [float(x) for x in np.mean([scores[f"{base}-s{k}"] for k in range(3)], axis=0)]
        res["act_failures"][base] = {k: float(np.mean([res["act_failures"][f"{base}-s{s}"][k] for s in range(3)]))
                                     for k in ("wrong_tool", "wrong_instant", "text_only", "not_executed")}
    for n in (*NAMES, "S8", "S9"):
        res["systems"][n] = round(float(np.mean(scores[n])), 3)
    for h, (x, y) in {"N1 S1 vs S2": ("S1", "S2"), "N3 S9 vs S8": ("S9", "S8")}.items():
        t = _paired(scores[x], scores[y], fams, cfg)
        res[h] = {"difference": round(t["difference"], 3), "ci": [round(t["ci"]["low"], 3), round(t["ci"]["high"], 3)],
                  "p_value": round(t["p_value"], 4)}
    return res


def main() -> int:
    cfg = _cfg()
    ev = json.loads((NAT / "evaluation.json").read_text(encoding="utf-8"))
    conf = {x["id"] for x in ev["systems"]["S1"]["items"]}
    locked = [r.item for r in read_records(ROOT / "data/natural/natural.jsonl") if r.item and r.item.id in conf]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out = Path(tmp) / "author.jsonl"
        build_dataset(adj.AUTHOR, out)
        author = [r.item for r in read_records(out) if r.item and r.item.id in conf]
    second, dropped = second_annotator_items(conf)
    covered = [it for it in locked if not UNCOVERED.search(it.text)]
    sys_ = systems(ROOT / "models", ROOT / "models_v2", TRACES)
    variants = {
        "locked gold (pre-registered)": evaluate(locked, sys_, cfg),
        "author's labels before adjudication": evaluate(author, sys_, cfg),
        f"second annotator's labels ({dropped} invalid dropped)": evaluate(second, sys_, cfg),
        "locked gold, lenient reminder text": evaluate(locked, sys_, cfg, lenient=True),
        f"locked gold without uncovered expressions ({len(locked) - len(covered)} dropped)": evaluate(covered, sys_, cfg),
    }
    ref = variants["locked gold (pre-registered)"]
    for n in ("S1", "S2", "oracle"):
        assert abs(ref["systems"][n] - ev["summary"][n]["sts"]) < 1e-9, (n, ref["systems"][n])
    assert abs(ref["systems"]["S9"] - ev["summary"]["S9 (seed mean)"]["sts"]) < 1e-9
    NAT.joinpath("sensitivity.json").write_text(json.dumps(variants, ensure_ascii=False, indent=1) + "\n",
                                                encoding="utf-8", newline="\n")
    md = ["# Natural-phrasing set: sensitivity of N1 and N3 (post hoc)", "",
          "Generated by `scripts/natural_sensitivity.py`; do not edit by hand. Not Holm-adjusted: sensitivity "
          "checks, not new hypotheses.", "",
          "| Labels / rule | n | S1 | S2 | N1 diff [95% CI], p | S8 | S9 | N3 diff [95% CI], p | oracle |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, v in variants.items():
        n1, n3 = v["N1 S1 vs S2"], v["N3 S9 vs S8"]
        md.append(f"| {name} | {v['n']} | {v['systems']['S1']:.3f} | {v['systems']['S2']:.3f} | "
                  f"{n1['difference']:+.3f} [{n1['ci'][0]:+.3f}, {n1['ci'][1]:+.3f}], {n1['p_value']:.4f} | "
                  f"{v['systems']['S8']:.3f} | {v['systems']['S9']:.3f} | "
                  f"{n3['difference']:+.3f} [{n3['ci'][0]:+.3f}, {n3['ci'][1]:+.3f}], {n3['p_value']:.4f} | "
                  f"{v['systems']['oracle']:.3f} |")
    md += ["", "## ACT failures on the locked gold (S8, S9: mean over seeds)", "",
           f"{sum(it.route == 'ACT' for it in locked)} gold-ACT items.", "",
           "| System | wrong tool | wrong instant | text only | not executed |", "|---|---|---|---|---|"]
    for n in ("oracle", "S1", "S2", "S8", "S9"):
        f = ref["act_failures"][n]
        md.append(f"| {n} | {f['wrong_tool']:g} | {f['wrong_instant']:g} | {f['text_only']:g} | {f['not_executed']:g} |")
    md += ["", "## Items with expressions the contract does not resolve", "",
           *[f"- {it.id} ({it.route}{', ' + ' '.join(it.clarify_targets) if it.clarify_targets else ''})"
             for it in locked if UNCOVERED.search(it.text)], ""]
    NAT.joinpath("sensitivity.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
