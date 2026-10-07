"""Fail if a data file has CRLF line endings: split and freeze hashes depend on exact bytes."""

import sys

bad = [f for f in sys.argv[1:] if b"\r\n" in open(f, "rb").read()]
for f in bad:
    print(f"CRLF line endings: {f}")
sys.exit(1 if bad else 0)
