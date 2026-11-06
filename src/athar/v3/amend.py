"""Append-only check for docs/protocol_v3_amendments.md (amendment A1).

A freeze written after an amendment records the amendments file's length and SHA-256 at that moment
(``amendments_bytes``, ``amendments_sha256``). The file may grow afterwards, but its first
``amendments_bytes`` bytes must still hash to the recorded value. That is the append-only rule,
checked, not just stated. Line endings are normalized to LF first, as in source hashes.

    python -m athar.v3.amend    # check every v3 freeze that records the amendments file
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
AMENDMENTS = REPO / "docs" / "protocol_v3_amendments.md"
MODELS_V3 = REPO / "models_v3"


def _bytes(path: Path = AMENDMENTS) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def snapshot(path: Path = AMENDMENTS) -> dict[str, Any]:
    """What a freeze records about the amendments file now."""
    data = _bytes(path)
    return {"amendments_bytes": len(data), "amendments_sha256": hashlib.sha256(data).hexdigest()}


def check(recorded: dict[str, Any], path: Path = AMENDMENTS) -> str | None:
    """None when the file still starts with exactly the recorded bytes; else the problem."""
    data = _bytes(path)
    n = int(recorded["amendments_bytes"])
    if len(data) < n:
        return "the amendments file is shorter than when it was recorded (an entry was removed)"
    if hashlib.sha256(data[:n]).hexdigest() != recorded["amendments_sha256"]:
        return "an earlier amendment was edited (the file is append-only)"
    return None


def check_all(models_dir: Path = MODELS_V3) -> list[str]:
    problems = []
    for f in sorted(models_dir.glob("freeze_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if "amendments_bytes" in data and (p := check(data)):
            problems.append(f"{f.name}: {p}")
    return problems


def main() -> int:
    problems = check_all()
    print("\n".join(problems) if problems else "amendments: append-only OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
