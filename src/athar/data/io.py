"""Reading gold JSONL and hashing items, files and manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from athar.schema import GoldItem


class DatasetError(ValueError):
    """The dataset (or a manifest over it) does not satisfy the contract."""


@dataclass(frozen=True)
class Record:
    """One non-blank JSONL line: the raw object, the parsed item (if valid) and its errors."""

    line: int
    raw: dict[str, Any]
    item: GoldItem | None
    errors: tuple[str, ...] = field(default=())

    @property
    def where(self) -> str:
        rid = self.raw.get("id")
        return f"line {self.line}" + (f" ({rid})" if isinstance(rid, str) else "")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def item_hash(raw: dict[str, Any]) -> str:
    """Hash of the item as written (key order and whitespace do not matter, content does)."""
    return sha256_text(canonical_json(raw))


def _format_error(err: Any) -> str:
    loc = ".".join(str(p) for p in err["loc"])
    msg = err["msg"].removeprefix("Value error, ")
    return f"{loc}: {msg}" if loc else msg


def read_records(path: str | Path) -> list[Record]:
    """Parse every non-blank line; never raises on bad content, errors are kept per record."""
    records: list[Record] = []
    text = Path(path).read_text(encoding="utf-8-sig")  # tolerate a BOM from spreadsheet tools
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            records.append(Record(lineno, {}, None, (f"invalid JSON: {e.msg} (column {e.colno})",)))
            continue
        if not isinstance(raw, dict):
            records.append(Record(lineno, {}, None, ("line is not a JSON object",)))
            continue
        try:
            records.append(Record(lineno, raw, GoldItem.model_validate(raw)))
        except ValidationError as e:
            records.append(Record(lineno, raw, None, tuple(_format_error(x) for x in e.errors())))
    return records


def load_items(path: str | Path) -> list[GoldItem]:
    """All items of a file that must be fully valid (schema level); raises otherwise."""
    records = read_records(path)
    bad = [r for r in records if r.item is None]
    if bad:
        first = bad[0]
        raise DatasetError(f"{path}: {len(bad)} invalid item(s); first at {first.where}: "
                           f"{first.errors[0]}. Run `athar-data validate {path}`.")
    return [r.item for r in records if r.item is not None]
