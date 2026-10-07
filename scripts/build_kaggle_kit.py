"""Package what the Kaggle notebooks need into dist/athar-chat-kit.zip.

    uv run python scripts/build_kaggle_kit.py

Upload the zip to Kaggle as a Dataset named ``athar-chat-kit`` (Datasets -> New Dataset), then
attach it to notebooks/kaggle_s4_llm.ipynb. Contents: the ``athar`` wheel (the same scoring,
pipeline and statistics code as this repo), the data with its split manifests, and configs.
The real dataset (data/athar.jsonl + data/splits/manifest.json) is included when it exists.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "dist" / "athar-chat-kit"
FILES = [
    "configs/eval.yaml", "configs/train.yaml",
    "data/pilot/pilot.jsonl", "data/pilot/manifest.json",
    "data/draft/athar_draft.jsonl", "data/draft/manifest.json",
    "data/athar.jsonl", "data/splits/manifest.json",  # real data, once it exists
    "docs/evaluation_protocol.md", "docs/task_contract.md",
]


def main() -> int:
    if KIT.exists():
        shutil.rmtree(KIT)
    KIT.mkdir(parents=True)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(KIT)], cwd=ROOT, check=True)
    copied = []
    for rel in FILES:
        src = ROOT / rel
        if src.exists():
            dst = KIT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    archive = ROOT / "dist" / "athar-chat-kit.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(KIT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(KIT).as_posix())
    wheel = next(KIT.glob("*.whl")).name
    print(f"wrote {archive.relative_to(ROOT)} ({archive.stat().st_size / 1024:.0f} KB): {wheel}, "
          + ", ".join(copied))
    return 0


if __name__ == "__main__":
    sys.exit(main())
