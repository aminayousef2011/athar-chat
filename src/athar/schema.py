"""Typed contracts: gold dataset items and system responses.

Both models enforce the route/intent/argument consistency rules of the task contract, so a
mislabeled gold item fails validation and a malformed system output counts as invalid
(an invalid structured output is a failure under Safe Task Success).
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from athar.timeexpr import GAP_ORDER

Intent = Literal["create_reminder", "create_alarm", "unsupported"]
Route = Literal["ACT", "CLARIFY", "ABSTAIN"]
Gap = Literal["date", "time", "am_pm", "past", "reminder_text"]
#: what a system may ask about: a missing argument, or ``confirm`` from the confidence gate
#: (athar.gate). Gold items never carry ``confirm``: it depends on the system, not the request.
ClarifyTarget = Literal["date", "time", "am_pm", "past", "reminder_text", "confirm"]
ToolName = Literal["create_reminder", "create_alarm"]
UnsupportedReason = Literal[
    "deletion", "recurring", "other_person", "multi_action", "other_tool", "modification", "other"
]
Source = Literal["author", "contributor", "assistant_draft"]
#: who would send the request: a project manager about their work (the main target users) or anyone
Domain = Literal["project_management", "general"]

INTENTS: tuple[str, ...] = ("create_reminder", "create_alarm", "unsupported")
ROUTES: tuple[str, ...] = ("ACT", "CLARIFY", "ABSTAIN")
REQUIRED_GAPS: dict[str, frozenset[str]] = {
    # gaps that block execution for each tool; reminder_text is irrelevant to an alarm
    "create_reminder": frozenset(GAP_ORDER),
    "create_alarm": frozenset(GAP_ORDER) - {"reminder_text"},
}


def _sorted_gaps(v: list[str]) -> list[str]:
    if len(set(v)) != len(v):
        raise ValueError("duplicate clarification target")
    return sorted(v, key=(*GAP_ORDER, "confirm").index)


def _aware(v: dt.datetime | None) -> dt.datetime | None:
    if v is not None and v.tzinfo is None:
        raise ValueError("datetime must include a UTC offset")
    return v


class GoldArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    datetime: dt.datetime | None = None
    reminder_text: str | None = None
    reminder_text_alts: list[str] = Field(default_factory=list)

    check_datetime = field_validator("datetime")(_aware)


class GoldItem(BaseModel):
    """One annotated request. See docs/annotation_guide.md for how each field is decided."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    family_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    text: str = Field(min_length=2)
    ref_datetime: dt.datetime
    timezone: str = "Africa/Cairo"
    intent: Intent
    route: Route
    args: GoldArgs = Field(default_factory=GoldArgs)
    clarify_targets: list[Gap] = Field(default_factory=list)
    unsupported_reason: UnsupportedReason | None = None
    pair_id: str | None = None
    code_switched: bool = False
    source: Source = "author"
    domain: Domain = "general"
    notes: str = ""

    check_ref_datetime = field_validator("ref_datetime")(_aware)
    check_targets = field_validator("clarify_targets")(_sorted_gaps)

    @model_validator(mode="after")
    def _consistent(self) -> GoldItem:
        unsupported = self.intent == "unsupported"
        if unsupported != (self.route == "ABSTAIN"):
            raise ValueError("route ABSTAIN if and only if intent is unsupported")
        if unsupported and self.unsupported_reason is None:
            raise ValueError("unsupported items need unsupported_reason")
        if not unsupported and self.unsupported_reason is not None:
            raise ValueError("unsupported_reason only applies to unsupported items")
        if self.route == "ACT":
            if self.clarify_targets:
                raise ValueError("ACT items have no clarify_targets")
            if self.args.datetime is None:
                raise ValueError("ACT items need args.datetime")
            if self.args.datetime <= self.ref_datetime:
                raise ValueError("ACT datetime must be after ref_datetime")
            if self.intent == "create_reminder" and not self.args.reminder_text:
                raise ValueError("ACT reminders need args.reminder_text")
        if self.route == "CLARIFY" and not self.clarify_targets:
            raise ValueError("CLARIFY items need at least one clarify_target")
        if self.route == "CLARIFY":
            extra = set(self.clarify_targets) - REQUIRED_GAPS[self.intent]
            if extra:
                raise ValueError(f"targets {sorted(extra)} do not apply to {self.intent}")
        if self.intent == "create_alarm" and self.args.reminder_text:
            raise ValueError("alarms carry no reminder_text")
        return self


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: ToolName
    datetime: dt.datetime
    reminder_text: str | None = None

    check_datetime = field_validator("datetime")(_aware)


class Response(BaseModel):
    """The fixed output schema every system must emit."""

    model_config = ConfigDict(extra="forbid")
    system: str
    route: Route
    intent: Intent
    tool_call: ToolCall | None = None
    clarify_targets: list[ClarifyTarget] = Field(default_factory=list)
    message: str
    # diagnostics (not scored): what the extractor saw even when the route is not ACT
    extracted_datetime: dt.datetime | None = None
    extracted_reminder_text: str | None = None
    # probability the recognizer gave its intent (None for S1 / oracle); see athar.gate
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    check_targets = field_validator("clarify_targets")(_sorted_gaps)

    @model_validator(mode="after")
    def _consistent(self) -> Response:
        if (self.route == "ABSTAIN") != (self.intent == "unsupported"):
            raise ValueError("route ABSTAIN if and only if intent is unsupported")
        if (self.route == "ACT") != (self.tool_call is not None):
            raise ValueError("tool_call present if and only if route is ACT")
        if (self.route == "CLARIFY") != bool(self.clarify_targets):
            raise ValueError("clarify_targets non-empty if and only if route is CLARIFY")
        if "confirm" in self.clarify_targets and len(self.clarify_targets) > 1:
            raise ValueError("a confirmation is asked on its own")
        if self.tool_call is not None and self.tool_call.name != self.intent:
            raise ValueError("tool_call.name must equal intent")
        if self.tool_call and self.tool_call.name == "create_reminder" and not self.tool_call.reminder_text:
            raise ValueError("create_reminder needs reminder_text")
        return self
