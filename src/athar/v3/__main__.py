"""``python -m athar.v3``: protocol v3 steps (docs/protocol_v3.md).

    python -m athar.v3 prepare                 # build the private v3 items from the kept messages
    python -m athar.v3 freeze-s3               # gates G2+G3 for S3-drafted / S3-human (CPU)
    python -m athar.v3 freeze-s6 --device cuda # gates G2+G3 for S6-drafted / S6-human (GPU advised)
    python -m athar.v3 verify                  # every v3 freeze, model, prediction and source unchanged
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m athar.v3", description=(__doc__ or "").splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("freeze-s3")
    s6 = sub.add_parser("freeze-s6")
    s6.add_argument("--device", default="cpu")
    s6.add_argument("--seeds", type=int, nargs="*", default=None)
    sub.add_parser("verify")
    a = p.parse_args(argv)

    from athar.v3 import data as D
    from athar.v3 import systems as S

    if a.cmd == "prepare":
        print(D.prepare())
    elif a.cmd == "freeze-s3":
        fz = S.freeze_s3()
        print(f"models_v3/freeze_s3.json  freeze_sha256={fz['freeze_sha256']}  v3_set={fz['v3_set_sha256'][:12]}")
    elif a.cmd == "freeze-s6":
        fz = S.freeze_s6(device=a.device, seeds=a.seeds)
        print(f"models_v3/freeze_s6.json  freeze_sha256={fz['freeze_sha256']}")
    else:
        problems = S.verify()
        print("\n".join(problems) if problems else "models_v3: OK")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
