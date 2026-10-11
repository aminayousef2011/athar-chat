"""Tables and tests for the S4 diagnostics run (notebooks/kaggle_s4_llm.ipynb, diagnostics cell).

    uv run --with pandas --with tabulate python scripts/analyse_s4_diagnostics.py runs/s4/diagnostics

Reads ``s4_diag_items.csv`` (one prediction per model x shot seed x method x item) and
``s4_diag_content_free.csv``. Reports, per model and scoring method:

1. dev Safe Task Success and intent macro-F1, mean +/- SD over the three exemplar sets;
2. native Egyptian out-of-scope recall (n = 192), mean +/- SD over exemplar sets, and the
   95% Wilson interval for each set;
3. paired tests against the primary method (raw), per exemplar set: exact McNemar on the
   Egyptian out-of-scope items, and the family-level paired permutation test on dev STS,
   Holm-adjusted over all tests in the table;
4. the paired dialect gap: out-of-scope recall on the same requests in their Saudi/MSA original
   and in Egyptian, with an exact McNemar test (Holm over all gap tests);
5. the label distribution each model gives the content-free inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from athar.eval.stats import holm, mcnemar_exact, paired_permutation_test
from athar.schema import INTENTS

PRIMARY = "raw"


def wilson(k: int, n: int) -> tuple[float, float]:
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return float(ci.low), float(ci.high)


def summary(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, method, seed), g in items.groupby(["model", "method", "shot_seed"]):
        dev = g[g.set == "dev"]
        egy = g[(g.set == "egyptian") & (g.gold == "unsupported")]
        k = int((egy.pred == "unsupported").sum())
        low, high = wilson(k, len(egy))
        rows.append({"model": model, "method": method, "shot_seed": seed,
                     "dev_STS": dev.success.mean(),
                     "dev_intent_F1": f1_score(dev.gold, dev.pred, labels=list(INTENTS), average="macro",
                                               zero_division=0),
                     "egy_oos_recall": k / len(egy), "egy_ci_low": low, "egy_ci_high": high, "egy_n": len(egy)})
    return pd.DataFrame(rows)


def paired(items: pd.DataFrame, permutations: int = 10000, seed: int = 20261007) -> pd.DataFrame:
    rows = []
    for (model, shot_seed), g in items.groupby(["model", "shot_seed"]):
        base = g[g.method == PRIMARY]
        for method in sorted(set(g.method) - {PRIMARY}):
            other = g[g.method == method]
            egy = base[(base.set == "egyptian") & (base.gold == "unsupported")].merge(
                other[(other.set == "egyptian") & (other.gold == "unsupported")], on="id", suffixes=("_p", "_o"))
            mc = mcnemar_exact(list(egy.pred_p == "unsupported"), list(egy.pred_o == "unsupported"))
            dev = base[base.set == "dev"].merge(other[other.set == "dev"], on=["id", "family_id"], suffixes=("_p", "_o"))
            pt = paired_permutation_test([float(x) for x in dev.success_p], [float(x) for x in dev.success_o],
                                         list(dev.family_id), permutations=permutations, resamples=1000,
                                         level=0.95, seed=seed)
            rows.append({"model": model, "shot_seed": shot_seed, "comparison": f"{PRIMARY} vs {method}",
                         "egy_only_primary": mc["only_a"], "egy_only_other": mc["only_b"], "egy_p": mc["p_value"],
                         "dev_STS_diff": pt.difference, "dev_p": pt.p_value})
    df = pd.DataFrame(rows)
    keys = [f"{i}:{c}" for i in df.index for c in ("egy_p", "dev_p")]
    adj = holm({k: float(df.loc[int(k.split(":")[0]), k.split(":")[1]]) for k in keys})
    df["egy_p_holm"] = [adj[f"{i}:egy_p"] for i in df.index]
    df["dev_p_holm"] = [adj[f"{i}:dev_p"] for i in df.index]
    return df


def dialect_gap(items: pd.DataFrame) -> pd.DataFrame:
    """Out-of-scope recall on the MSA original vs the Egyptian rewrite of the same items."""
    oos = items[items.set.isin(["egyptian", "msa"]) & (items.gold == "unsupported")]
    if oos.empty or "msa" not in set(oos.set):
        return pd.DataFrame()
    rows = []
    for (model, method, seed), g in oos.groupby(["model", "method", "shot_seed"]):
        wide = g.pivot(index="id", columns="set", values="pred")
        if not {"msa", "egyptian"} <= set(wide.columns):
            continue                     # this model has not been scored on both versions (yet)
        wide = wide.dropna(subset=["msa", "egyptian"])
        msa_ok, egy_ok = list(wide.msa == "unsupported"), list(wide.egyptian == "unsupported")
        mc = mcnemar_exact(msa_ok, egy_ok)
        rows.append({"model": model, "method": method, "shot_seed": seed, "n": len(wide),
                     "msa_recall": sum(msa_ok) / len(wide), "egy_recall": sum(egy_ok) / len(wide),
                     "gap_msa_minus_egy": (sum(msa_ok) - sum(egy_ok)) / len(wide),
                     "only_msa": mc["only_a"], "only_egy": mc["only_b"], "p": mc["p_value"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    adj = holm({str(i): float(p) for i, p in df.p.items()})
    df["p_holm"] = [adj[str(i)] for i in df.index]
    return df


def _load(d: Path, part_glob: str, combined: str, keys: list[str]) -> pd.DataFrame:
    """All part files (several Kaggle versions, any subfolder) plus any combined files, merged;
    a row seen twice (same keys) is kept once."""
    files = sorted(d.rglob(part_glob)) + sorted(d.rglob(combined))
    if not files:
        raise FileNotFoundError(f"no {part_glob} or {combined} under {d}")
    return pd.concat(map(pd.read_csv, files)).drop_duplicates(subset=keys).reset_index(drop=True)


def report(directory: str | Path, permutations: int = 10000) -> str:
    d = Path(directory)
    items = _load(d, "s4_diag_part_*.csv", "s4_diag_items.csv", ["model", "shot_seed", "method", "set", "id"])
    items["success"] = items.success.astype(str).str.lower().isin(["true", "1"])
    per_seed = summary(items)
    agg = per_seed.groupby(["model", "method"]).agg(
        dev_STS=("dev_STS", "mean"), dev_STS_sd=("dev_STS", "std"),
        dev_F1=("dev_intent_F1", "mean"), dev_F1_sd=("dev_intent_F1", "std"),
        egy_recall=("egy_oos_recall", "mean"), egy_recall_sd=("egy_oos_recall", "std")).round(3).reset_index()
    tests = paired(items, permutations)
    out = ["# S4 diagnostics", "",
           f"{items.shot_seed.nunique()} exemplar sets x {items.method.nunique()} scoring methods x "
           f"{items.model.nunique()} models. Dev = {items[items.set == 'dev'].id.nunique()} items; Egyptian "
           f"out-of-scope = {int(per_seed.egy_n.iloc[0])} native-written items. Mean +/- SD over exemplar sets.", "",
           "## Summary", "", agg.to_markdown(index=False), "",
           "## Per exemplar set (Egyptian recall with 95% Wilson interval)", "",
           per_seed.round(3).to_markdown(index=False), "",
           f"## Paired tests against `{PRIMARY}` (Holm over all {2 * len(tests)} tests)", "",
           "McNemar on the Egyptian out-of-scope items (`egy_only_*`: items only that method abstains on); "
           "family-level paired permutation test on dev STS.", "",
           tests.round(4).to_markdown(index=False), ""]
    gap = dialect_gap(items)
    if not gap.empty:
        out += ["## Paired dialect gap (same requests: Saudi/MSA original vs Egyptian rewrite)", "",
                "Positive gap: the model abstains on out-of-scope requests less often when they are written in "
                f"Egyptian. Exact McNemar per row, Holm over all {len(gap)} rows.", "",
                gap.round(4).to_markdown(index=False), ""]
    try:
        cf = _load(d, "s4_diag_cf_*.csv", "s4_diag_content_free.csv", ["model", "shot_seed"]).sort_values(["model", "shot_seed"])
    except FileNotFoundError:
        cf = None
    if cf is not None:
        out += ["## Label distribution on content-free inputs", "",
                "Contextual calibration divides by this. A large `unsupported` share means calibration "
                "pushes real out-of-scope requests towards the in-scope labels.", "",
                cf.round(3).to_markdown(index=False), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("directory")
    p.add_argument("--out")
    p.add_argument("--permutations", type=int, default=10000)
    a = p.parse_args(argv)
    md = report(a.directory, a.permutations)
    out = Path(a.out) if a.out else Path(a.directory) / "report.md"
    out.write_text(md + "\n", encoding="utf-8", newline="\n")
    print(md)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
