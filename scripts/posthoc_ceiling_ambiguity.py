"""Post hoc: how much of the natural-set oracle ceiling sits on items the two annotators labelled
differently (contract ambiguity) rather than on items they agreed on (pipeline errors).

    uv run --with openpyxl python scripts/posthoc_ceiling_ambiguity.py

"Differently" uses the rule that chose the items for adjudication (``adjudicate_natural.differences``
on the two independent sheets: intent, route, slot set, unsupported reason, wall-clock datetime,
canonical reminder text). Before anything is written the script checks that this rule reproduces the
committed adjudication exactly (the 66 item ids of results/natural/adjudication.json and the field
counts of the protocol log, 2026-11-01). For the split, only differences that can change Safe Task
Success count (a different unsupported reason cannot: any decline succeeds). It also reports how many
both-ACT datetime differences of results/natural/agreement.json (computed on the sheet as submitted)
are the same wall-clock time written with a different UTC offset. The frozen systems are scored
against the locked gold, unchanged. Descriptive only: no test, no new hypothesis.
Output: results/posthoc/ceiling_ambiguity.json and .md.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adjudicate_natural as adj  # noqa: E402
from natural_sensitivity import TRACES, outcome  # noqa: E402

from athar.data import agreement as ag  # noqa: E402
from athar.data.io import read_records  # noqa: E402
from athar.eval import harness as H  # noqa: E402
from athar.v2.natural import systems  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "posthoc"
# field counts of the adjudicated items, as logged in docs/protocol_natural.md (2026-11-01)
LOGGED = {"clarify_targets": 42, "route": 14, "intent": 10, "datetime": 8, "unsupported_reason": 6,
          "reminder_text": 5}
STS_FIELDS = {"intent", "route", "clarify_targets", "datetime", "reminder_text"}


def main() -> int:
    a, b = adj.read(adj.AUTHOR), adj.read(adj.OTHER)
    ids = sorted(set(a) & set(b))
    diff = {i: adj.differences(a[i], b[i]) for i in ids}

    # 1. the rule must reproduce the committed adjudication exactly
    adjud = json.loads((ROOT / "results/natural/adjudication.json").read_text(encoding="utf-8"))
    differing = {i for i, d in diff.items() if d}
    fields = Counter(f for d in diff.values() for f in d)
    checks = {
        "differing ids equal results/natural/adjudication.json": differing == {r["id"] for r in adjud["rows"]},
        "66 differing items": len(differing) == adjud["adjudicated"] == 66,
        "field counts equal the protocol log": dict(fields) == LOGGED,
    }
    for k, v in checks.items():
        print(("ok   " if v else "FAIL ") + k)
    if not all(checks.values()):
        print("validation failed: nothing written", dict(fields))
        return 1

    # 2. both-ACT datetime differences as submitted vs at wall-clock level (agreement.json)
    ref = json.loads((ROOT / "results/natural/agreement.json").read_text(encoding="utf-8"))
    la, _ = ag.read_labels(adj.AUTHOR)
    lb, _ = ag.read_labels(adj.OTHER)
    both_act = [i for i in sorted(set(la) & set(lb)) if la[i].route == lb[i].route == "ACT"]
    as_written = [i for i in both_act if la[i].datetime != lb[i].datetime]
    wall = [i for i in both_act if adj.wall(a[i]) != adj.wall(b[i])]
    agree_written = 1 - len(as_written) / len(both_act)
    if not (len(both_act) == ref["datetime_same_instant"]["n"]
            and abs(agree_written - ref["datetime_same_instant"]["agreement"]) < 1e-9):
        print("datetime agreement does not reproduce results/natural/agreement.json: nothing written")
        return 1
    print(f"ok   both-ACT datetime agreement reproduced ({len(both_act)} items, {agree_written:.3f})")

    # 3. oracle and systems on agreed vs disagreed items of the locked gold
    ev = json.loads((ROOT / "results/natural/evaluation.json").read_text(encoding="utf-8"))
    conf = {x["id"] for x in ev["systems"]["S1"]["items"]}
    locked = [r.item for r in read_records(ROOT / "data/natural/natural.jsonl") if r.item and r.item.id in conf]
    disagreed = {i for i, d in diff.items() if set(d) & STS_FIELDS}
    groups = {"agreed": [it.id for it in locked if it.id not in disagreed],
              "disagreed": [it.id for it in locked if it.id in disagreed]}
    res: dict = {
        "n": len(locked),
        "rule": "adjudicate_natural.differences on the two independent sheets; STS-relevant fields only",
        "validation": checks,
        "both_act_datetime": {"n": len(both_act), "differ_as_written": len(as_written),
                              "differ_wall_clock": len(wall), "offset_only": len(as_written) - len(wall),
                              "agreement_as_written": agree_written,
                              "agreement_wall_clock": 1 - len(wall) / len(both_act)},
        "items": {k: len(v) for k, v in groups.items()},
        "systems": {},
    }
    sys_ = systems(ROOT / "models", ROOT / "models_v2", TRACES)
    for name in ("oracle", "S1", "S1-v1.1", "S2", "S3"):
        outs = H.run(sys_[name], locked)
        ok = {g.id: outcome(g, o) == "success" for g, o in zip(locked, outs, strict=True)}
        assert round(sum(ok.values()) / len(locked), 3) == ev["summary"][name]["sts"], name  # stored rounded
        row = {"sts": sum(ok.values()) / len(locked), "failures": sum(not v for v in ok.values())}
        for k, g in groups.items():
            s = sum(ok[i] for i in g)
            row[k] = {"n": len(g), "sts": s / len(g), "failures": len(g) - s}
        res["systems"][name] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ceiling_ambiguity.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                                                encoding="utf-8", newline="\n")
    d = res["both_act_datetime"]
    md = ["# Natural set: ceiling and annotator agreement (post hoc, descriptive)", "",
          "Generated by `scripts/posthoc_ceiling_ambiguity.py`; do not edit by hand.", "",
          f"Rule: {res['rule']}. Validation: " + "; ".join(k for k in checks) + " (all passed).", "",
          f"Both-ACT datetimes: {d['differ_as_written']} of {d['n']} differ as written "
          f"(agreement {d['agreement_as_written']:.3f}), {d['offset_only']} of them only in the written UTC "
          f"offset; {d['differ_wall_clock']} differ in wall-clock time (agreement {d['agreement_wall_clock']:.3f}).", "",
          f"Confirmatory items: {res['items']}", "",
          "| System | STS | failures | agreed: STS (failures / n) | disagreed: STS (failures / n) |",
          "|---|---|---|---|---|"]
    for n, r in res["systems"].items():
        x, y = r["agreed"], r["disagreed"]
        md.append(f"| {n} | {r['sts']:.3f} | {r['failures']} | {x['sts']:.3f} ({x['failures']} / {x['n']}) | "
                  f"{y['sts']:.3f} ({y['failures']} / {y['n']}) |")
    (OUT / "ceiling_ambiguity.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
