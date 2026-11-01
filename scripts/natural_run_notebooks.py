"""Write one ready-to-commit Kaggle notebook per natural-phrasing GPU run (docs/protocol_natural.md,
log 2026-11-01, "GPU run settings").

    uv run python scripts/natural_run_notebooks.py

Copies notebooks/kaggle_s4_llm.ipynb and changes only the settings lines of its configuration
cell, so each run commits exactly the logged settings. Every replaced line must occur exactly
once, or nothing is written. Output: dist/natural_runs/run<k>_<name>.ipynb (not committed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "kaggle_s4_llm.ipynb"
OUT = ROOT / "dist" / "natural_runs"

COMMON = {
    "EVAL_MANIFEST = MANIFEST": 'EVAL_MANIFEST = f"{KIT}/data/natural/manifest.json"   # natural-phrasing set',
    'SPLIT = "dev"': 'SPLIT = "test"',
    "UNLOCK_TEST = False": "UNLOCK_TEST = True",
    "GATE_TAU = {}": ('GATE_TAU = {"S4-jais3-7b-6shot-prior": 0.0, "S4-jais3-3.7b-6shot-prior": 0.0,   # v1 test run\n'
                      '            "S4-jais2-8b-6shot-prior": 0.7, "S4-nile-4b-6shot-prior": 0.0,\n'
                      '            "S4-qwen3-8b-6shot-prior": 0.0}'),
    "RUN_PROMPT_CHECK = True": "RUN_PROMPT_CHECK = False",
    "S7_EGYPTIAN = True": "S7_EGYPTIAN = False",
}
RUNS = {
    "run1_s8": {"RUN_S8 = False": "RUN_S8 = True"},
    "run2_s4_jais3": {"RUN_S4 = False": "RUN_S4 = True",
                      "RUN_MODELS = []": 'RUN_MODELS = ["jais3-7b", "jais3-3.7b"]'},
    "run3_s4_jais2": {"RUN_S4 = False": "RUN_S4 = True", "RUN_MODELS = []": 'RUN_MODELS = ["jais2-8b"]'},
    "run4_s4_nile_qwen": {"RUN_S4 = False": "RUN_S4 = True",
                          "RUN_MODELS = []": 'RUN_MODELS = ["nile-4b", "qwen3-8b"]'},
    "run5_s7": {"RUN_S7_QLORA = False": "RUN_S7_QLORA = True"},
}


def configure(nb: dict, changes: dict[str, str], name: str) -> dict:
    nb = json.loads(json.dumps(nb))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code" and "NOTEBOOK_VERSION =" in "".join(c["source"])]
    if len(cells) != 1:
        raise SystemExit("configuration cell not found exactly once")
    lines = "".join(cells[0]["source"]).split("\n")
    for old, new in changes.items():
        hits = [i for i, line in enumerate(lines) if line.split("#")[0].strip() == old]
        if len(hits) != 1:
            raise SystemExit(f"{name}: {old!r} found {len(hits)} times; nothing written")
        lines[hits[0]] = new
    src = "\n".join(lines).replace('print("ATHAR notebook version", NOTEBOOK_VERSION)',
                                   f'print("ATHAR notebook version", NOTEBOOK_VERSION, "| natural-phrasing {name}")')
    cells[0]["source"] = src.splitlines(keepends=True)
    return nb


def main() -> int:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    built = {name: configure(nb, {**COMMON, **extra}, name) for name, extra in RUNS.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, out in built.items():
        path = OUT / f"{name}.ipynb"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print("wrote", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
