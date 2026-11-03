"""Figures for docs/report.md, drawn from the stored results (nothing is re-scored except the
deterministic core systems on dev, with the hash-verified frozen models).

    uv run --with matplotlib python scripts/posthoc_analysis.py          # first: results/posthoc/
    uv run --with matplotlib python scripts/make_figures.py --out docs/figures

1. ``dev_vs_test.png``: Safe Task Success on the drafted dev split vs the locked test split.
2. ``regularization.png``: S2/S3 by regularization strength C (post hoc, results/posthoc).
3. ``failure_profiles.png``: success by gold route, shared pipeline vs end-to-end Jais 3 (S8).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from athar.data.split import load_split  # noqa: E402
from athar.eval.harness import Oracle, run  # noqa: E402
from athar.eval.metrics import safe_task_success, score_all  # noqa: E402
from athar.intent.train import load_recognizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
BLUE, RED, AMBER, GREY = "#1f5aa6", "#c8414b", "#d08c1a", "#9aa5b1"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def spread(values: list[float], gap: float) -> list[float]:
    """Label positions for sorted ``values``: at least ``gap`` apart, as close to the values as
    possible (overlapping labels are grouped and centred on their points)."""
    groups = [[v] for v in values]
    merged = True
    while merged:
        merged = False
        out: list[list[float]] = []
        for g in groups:
            if out:
                prev = out[-1]
                prev_top = sum(prev) / len(prev) + gap * (len(prev) - 1) / 2
                g_bottom = sum(g) / len(g) - gap * (len(g) - 1) / 2
                if g_bottom - prev_top < gap:
                    out[-1] = prev + g
                    merged = True
                    continue
            out.append(g)
        groups = out
    pos = []
    for g in groups:
        centre = sum(g) / len(g)
        pos += [centre + gap * (k - (len(g) - 1) / 2) for k in range(len(g))]
    return pos


def dev_test_pairs() -> list[tuple[str, float, float, str]]:
    """(label, dev STS, test STS, colour) for every system in report section 9."""
    dev = load_split(ROOT / "data" / "splits" / "manifest.json", "dev")
    core = _json(RES / "test_results.json")["systems"]
    rows = []
    for name, colour in (("S1", BLUE), ("S2", RED), ("S3", RED), ("oracle", GREY)):
        system = Oracle() if name == "oracle" else load_recognizer(name, ROOT / "models")
        dev_sts = safe_task_success(score_all(dev, run(system, dev)))
        rows.append((name, dev_sts, core[name]["metrics"]["safe_task_success"], colour))
    dev_s4 = {r["system"]: float(r["STS"]) for d in ("2026-10-16-run2-jais3", "2026-10-16-run3-others")
              for r in _csv(RES / "dev" / d / "summary.csv")}
    test_s4 = {r["system"]: float(r["STS"]) for d in ("7a-jais3", "7b-others")
               for r in _csv(RES / "s4" / d / "s4_results" / "summary.csv")}
    labels = {"jais2-8b": "S4 Jais 2", "jais3-3.7b": "S4 Jais-3-3.7B", "jais3-7b": "S4 Jais-3-7B",
              "nile-4b": "S4 Nile-Chat", "qwen3-8b": "S4 Qwen3"}
    for key, label in labels.items():
        name = f"S4-{key}-6shot-prior"
        rows.append((label, dev_s4[name], test_s4[name], AMBER))
    s7_dev = [float(r["STS"]) for r in _csv(RES / "dev" / "2026-10-18-run6b-s7" / "s7_seeds.csv")
              if r["system"].endswith("-prior")]
    s7_test = [_json(RES / "s4" / "7cd-s7-s8" / "s4_results" / f"s7_jais2_qlora_s{s}_test.json")["systems"]
               [f"S7-jais2-qlora-s{s}-prior"]["metrics"]["safe_task_success"] for s in (0, 1, 2)]
    rows.append(("S7 (3 seeds)", _mean(s7_dev), _mean(s7_test), AMBER))
    s8_dev = [float(r["STS"]) for r in _csv(RES / "dev" / "2026-10-16-run5-s8" / "s8_summary.csv")]
    s8_test = [float(r["STS"]) for r in _csv(RES / "s4" / "7cd-s7-s8" / "s4_results" / "s8_summary.csv")]
    rows.append(("S8 (3 seeds)", _mean(s8_dev), _mean(s8_test), AMBER))
    return rows


def fig_dev_vs_test(out: Path) -> None:
    rows = dev_test_pairs()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for _label, d, t, colour in rows:
        ax.plot([0, 1], [d, t], color=colour, lw=2 if colour in (BLUE, RED) else 1.3, marker="o", ms=4,
                alpha=0.95 if colour != AMBER else 0.75)
    # right-hand labels spread just enough to stay readable, each joined to its point
    ordered = sorted(rows, key=lambda r: r[2])
    ys = spread([r[2] for r in ordered], gap=0.017)
    for (label, _, t, colour), y in zip(ordered, ys, strict=True):
        ax.plot([1.0, 1.03], [t, y], color=GREY, lw=0.6)
        ax.text(1.04, y, f"{label}  {t:.3f}", va="center", fontsize=8.5, color=colour if colour != GREY else MUTED)
    ax.set_xticks([0, 1], ["dev (40 items, 39 AI-drafted)", "locked test (120, author-written)"])
    ax.set_xlim(-0.1, 1.75)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Safe Task Success")
    ax.grid(axis="x", visible=False)
    ax.set_title("Every system dropped from dev to test; the classifiers dropped most",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "dev_vs_test.png", dpi=200)
    plt.close(fig)


def fig_regularization(out: Path) -> None:
    rows = _csv(RES / "posthoc" / "regularization_sweep.csv")
    s1 = _json(RES / "test_results.json")["systems"]["S1"]["metrics"]["safe_task_success"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for system, colour in (("S2", RED), ("S3", BLUE)):
        r = [x for x in rows if x["system"] == system]
        cs = [float(x["C"]) for x in r]
        left.plot(cs, [float(x["dev_intent_f1"]) for x in r], color=colour, marker="o", ms=4, label=f"{system} dev")
        left.plot(cs, [float(x["groupcv_intent_f1"]) for x in r], color=colour, ls="--", marker="s", ms=3.5,
                  label=f"{system} grouped CV")
        right.plot(cs, [float(x["test_sts"]) for x in r], color=colour, marker="o", ms=4, label=system)
        for x in r:
            c = float(x["C"])
            if x["frozen_choice"] == "True":
                right.scatter([c], [float(x["test_sts"])], s=110, facecolors="none", edgecolors=colour, lw=1.6)
            if x["groupcv_choice"] == "True":
                right.scatter([c], [float(x["test_sts"])], s=110, marker="s", facecolors="none",
                              edgecolors=colour, lw=1.2)
    right.axhline(s1, color=INK, lw=1, ls=":")
    right.text(0.1, s1 - 0.013, f"S1 keyword rules {s1:.3f}", fontsize=8.5, color=INK)
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xlabel("C (inverse regularization strength)")
    left.set_ylabel("intent macro-F1 (selection metric)")
    left.set_title("Selection data: dev prefers small C", loc="left", fontsize=10)
    left.legend(fontsize=8, frameon=False)
    right.set_ylabel("locked test Safe Task Success")
    right.set_title("Test: larger C transfers better (post hoc)", loc="left", fontsize=10)
    right.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    right.text(0.98, 0.02, "○ frozen choice   □ grouped-CV choice", transform=right.transAxes, ha="right",
               va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "regularization.png", dpi=200)
    plt.close(fig)


def fig_failure_profiles(out: Path) -> None:
    core = _json(RES / "test_results.json")["systems"]
    s4dir = RES / "s4" / "7cd-s7-s8" / "s4_results"
    s7 = [_json(s4dir / f"s7_jais2_qlora_s{s}_test.json")["systems"][f"S7-jais2-qlora-s{s}-prior"]["metrics"]
          ["success_by_gold_route"] for s in (0, 1, 2)]
    s8 = [m["metrics"]["success_by_gold_route"] for m in _json(s4dir / "s8_jais3-7b.json")["systems"].values()]
    routes = ("ACT", "CLARIFY", "ABSTAIN")
    series = [("oracle (gold intent, shared core)", [core["oracle"]["metrics"]["success_by_gold_route"][r]
                                                     for r in routes], GREY),
              ("S1 keyword rules", [core["S1"]["metrics"]["success_by_gold_route"][r] for r in routes], BLUE),
              ("S7 Jais 2 + QLoRA (mean of 3 seeds)", [_mean([m[r] for m in s7]) for r in routes], AMBER),
              ("S8 Jais-3-7B end to end (mean of 3 seeds)", [_mean([m[r] for m in s8]) for r in routes], RED)]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    width = 0.2
    for k, (label, vals, colour) in enumerate(series):
        xs = [i + (k - 1.5) * width for i in range(len(routes))]
        ax.bar(xs, vals, width=width * 0.92, color=colour, label=label)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(routes)), [f"{r}\n(gold, n = {n})" for r, n in zip(routes, (50, 50, 20), strict=True)])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("success rate on the locked test")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("The shared core fails complete requests; end-to-end Jais 3 fails incomplete ones",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "failure_profiles.png", dpi=200)
    plt.close(fig)


def fig_natural_routes(out: Path) -> None:
    """Success by gold route on the locked test (author) and the natural set (17 writers)."""
    routes = ("ACT", "CLARIFY", "ABSTAIN")
    core = _json(RES / "test_results.json")["systems"]
    s8_v1 = [m["metrics"]["success_by_gold_route"]
             for m in _json(RES / "s4" / "7cd-s7-s8" / "s4_results" / "s8_jais3-7b.json")["systems"].values()]
    nat = _json(RES / "natural" / "evaluation.json")["systems"]

    def by_route(names: list[str]) -> dict[str, float]:
        out_: dict[str, float] = {}
        for r in routes:
            vals = [float(x["success"]) for n in names for x in nat[n]["items"] if x["gold_route"] == r]
            out_[r] = _mean(vals)
        return out_

    systems = [("oracle", core["oracle"]["metrics"]["success_by_gold_route"], by_route(["oracle"])),
               ("S1 rules", core["S1"]["metrics"]["success_by_gold_route"], by_route(["S1"])),
               ("S2 TF-IDF", core["S2"]["metrics"]["success_by_gold_route"], by_route(["S2"])),
               ("S8 tool calling", {r: _mean([m[r] for m in s8_v1]) for r in routes},
                by_route([f"S8-s{k}" for k in range(3)])),
               ("S9 hybrid", None, by_route([f"S9-s{k}" for k in range(3)]))]
    n_v1 = {r: sum(x["gold_route"] == r for x in core["S1"]["items"]) for r in routes}
    n_nat = {r: sum(x["gold_route"] == r for x in nat["S1"]["items"]) for r in routes}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9), sharey=True)
    h = 0.38
    for ax, r in zip(axes, routes, strict=True):
        for k, (_label, v1, v2) in enumerate(systems):
            y = len(systems) - 1 - k
            if v1 is not None:
                ax.barh(y + h / 2, v1[r], height=h * 0.9, color=AMBER, label="locked test (author)" if k == 0 else None)
                ax.text(v1[r] + 0.02, y + h / 2, f"{v1[r]:.2f}", va="center", fontsize=7.5, color=INK)
            ax.barh(y - h / 2, v2[r], height=h * 0.9, color=BLUE, label="natural set (17 writers)" if k == 0 else None)
            ax.text(v2[r] + 0.02, y - h / 2, f"{v2[r]:.2f}", va="center", fontsize=7.5, color=INK)
        ax.set_title(f"{r} (gold n = {n_v1[r]} / {n_nat[r]})", loc="left", fontsize=9.5, color=INK)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1.0])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(systems)), [s[0] for s in reversed(systems)])
    axes[1].set_xlabel("success rate (Safe Task Success on items of that gold route)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncols=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.005, 0.93))
    fig.suptitle("Success by gold route: S8's lead on complete requests disappears on new writers",
                 x=0.01, ha="left", y=0.985, fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out / "natural_routes.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    out = Path(ap.parse_args(argv).out)
    out.mkdir(parents=True, exist_ok=True)
    fig_dev_vs_test(out)
    fig_regularization(out)
    fig_failure_profiles(out)
    fig_natural_routes(out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
