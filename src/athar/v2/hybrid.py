"""S9 (hybrid) and S8 replay, from the stored S8 traces (docs/protocol_natural.md, section 5).

S8 runs once per seed on the GPU; its trace keeps every item's tool calls. ``S8Replay`` turns a
trace back into S8's responses exactly as the run did (``athar.agent.to_response``), and ``S9``
combines the same calls with the deterministic core:

- S8 declines -> ABSTAIN;
- otherwise the intent is the tool S8 names (its create call, or the ``tool`` of its
  ``ask_clarification``); when S8 gave no valid tool call (plain text or a malformed call), the
  intent is S1 v1.1's;
- the route comes from the shared policy on the core-v1.1 extraction: a blocking gap -> CLARIFY;
- otherwise ACT, with S8's datetime and reminder text when S8 made the matching create call, and
  core v1.1's otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from athar.agent import ToolCallError, to_response
from athar.normalize import norm
from athar.policy import decide
from athar.respond import build_response, describe_datetime
from athar.schema import GoldItem, Response
from athar.v2 import core
from athar.v2.rules import classify as s1_v11


def read_trace(path: str | Path) -> dict[str, dict[str, Any]]:
    """Trace rows of one S8 seed, by item id."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def s8_response(row: Mapping[str, Any] | None, tz_name: str, system: str = "S8") -> Response | None:
    """S8's response for one trace row; ``None`` (scored as a failure) when it made no valid call."""
    if row is None:
        return None
    try:
        return to_response(row.get("calls") or [], system, tz_name)[0]
    except ToolCallError:
        return None


def s9_response(item: GoldItem, row: Mapping[str, Any] | None, system: str = "S9") -> Response:
    s8 = s8_response(row, item.timezone)
    ext = core.extract(item.text, item.ref_datetime, item.timezone)
    if s8 is not None and s8.route == "ABSTAIN":
        return build_response(system, decide("unsupported", ext), ext)
    intent = s8.intent if s8 is not None else s1_v11(norm(item.text))
    decision = decide(intent, ext)
    if decision.route != "ACT" or s8 is None or s8.route != "ACT" or s8.intent != intent:
        return build_response(system, decision, ext)
    call = s8.tool_call
    assert call is not None
    message = (f"تمام، هفكرك بـ«{call.reminder_text}» {describe_datetime(call.datetime)}." if call.reminder_text
               else f"تمام، المنبه هيرن {describe_datetime(call.datetime)}.")
    return Response(system=system, route="ACT", intent=intent, tool_call=call, message=message,
                    extracted_datetime=call.datetime, extracted_reminder_text=call.reminder_text)


class S8Replay:
    """S8 for one seed, answered from its stored trace (no model is loaded)."""

    def __init__(self, trace: Mapping[str, Mapping[str, Any]], name: str) -> None:
        self.trace, self.name = trace, name

    def respond_items(self, items: Sequence[GoldItem]) -> list[Response | None]:
        return [s8_response(self.trace.get(it.id), it.timezone, self.name) for it in items]


class S9:
    """The hybrid for one S8 seed."""

    def __init__(self, trace: Mapping[str, Mapping[str, Any]], name: str) -> None:
        self.trace, self.name = trace, name

    def respond_items(self, items: Sequence[GoldItem]) -> list[Response | None]:
        return [s9_response(it, self.trace.get(it.id), self.name) for it in items]
