"""S8 (exploratory): an LLM does the whole task through native tool calling.

S1-S7 only predict the intent; the route, the arguments and the clarification targets come from
the shared deterministic extractor and policy. S8 asks a different question: how safely does a
tool-calling LLM (Jais 3, released 2026-10-15 with native function calling) route Egyptian
Arabic requests *on its own*? It gets the request, the moment it was sent and a summary of the
task contract, and must call exactly one of four tools:

=====================  ==========================  ===================================
tool                   arguments                   becomes
=====================  ==========================  ===================================
``create_reminder``    ``datetime``, ``reminder_text``  ACT, ``create_reminder``
``create_alarm``       ``datetime``                ACT, ``create_alarm``
``ask_clarification``  ``tool``, ``missing``       CLARIFY with targets ``missing``
``decline``            ``reason``                  ABSTAIN (reason kept as a diagnostic)
=====================  ==========================  ===================================

The call is turned into the same :class:`athar.schema.Response` every system emits and scored
with the same Safe Task Success. No call, several calls, an unknown tool or invalid arguments
give no response, which scores as a failure (protocol section 3), and the reason is recorded in
``trace``. A ``datetime`` without a UTC offset is read as wall-clock time in the request's IANA
zone (the prompt asks for that format, so DST arithmetic is not part of the task); one with an
offset is taken as written.

Two backends produce completions: :class:`HFToolBackend` (Transformers ``generate``; any GPU,
including Kaggle T4s) parses the tool calls from the generated text, and
:class:`OpenAIToolBackend` sends the request to an OpenAI-compatible server (vLLM/SGLang with the
model's own ``k2_horizon`` reasoning and tool parsers, the serving setup the Jais 3 card
recommends). The parser understands Jais 3's ``json``, ``xml`` and ``xml_typed`` call formats
and the ``<tool_call>{...}</tool_call>`` format of Qwen-style templates.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from athar.respond import ABSTAIN_MESSAGE, QUESTIONS, describe_datetime
from athar.schema import GoldItem, Response, ToolCall
from athar.timeexpr import GAP_ORDER

TOOL_NAMES = ("create_reminder", "create_alarm", "ask_clarification", "decline")
REASONS = ("deletion", "modification", "recurring", "other_person", "multi_action", "other_tool", "other")
DATETIME_FORMAT = "YYYY-MM-DDTHH:MM"
WEEKDAYS_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style function definitions, as both the chat templates and the servers take them."""
    when = {"type": "string",
            "description": f"Local date and time in the user's time zone, formatted {DATETIME_FORMAT} (24-hour clock)."}

    def fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}}

    return [
        fn("create_reminder", "Create one one-off reminder for the user. Use only when the moment and the content are both clear.",
           {"datetime": when,
            "reminder_text": {"type": "string",
                              "description": "What to remind the user about, in the user's own words, without the time expression or the request words."}},
           ["datetime", "reminder_text"]),
        fn("create_alarm", "Set one one-off alarm for the user. Use only when the moment is clear.",
           {"datetime": when}, ["datetime"]),
        fn("ask_clarification", "Ask the user for what is missing or ambiguous in a supported reminder or alarm request.",
           {"tool": {"type": "string", "enum": ["create_reminder", "create_alarm"],
                     "description": "The tool the user wants."},
            "missing": {"type": "array", "minItems": 1, "uniqueItems": True,
                        "items": {"type": "string", "enum": list(GAP_ORDER)},
                        "description": "Every gap that blocks the tool call."}},
           ["tool", "missing"]),
        fn("decline", "Decline a request the assistant cannot do.",
           {"reason": {"type": "string", "enum": list(REASONS),
                       "description": "Why the request is out of scope (the first that applies, in this order)."}},
           ["reason"]),
    ]


SYSTEM_PROMPT = """You are the tool-routing step of an assistant for Egyptian users. The assistant can do exactly two things: create ONE one-off reminder for the user, or set ONE one-off alarm for the user. The user writes in Egyptian Arabic, sometimes mixed with English.

The user sent the message on {weekday} {date} at {time} ({tz}).

Call exactly one tool, and nothing else:
- create_reminder(datetime, reminder_text): a one-off reminder for the user whose moment and content are both clear.
- create_alarm(datetime): a one-off alarm whose moment is clear.
- ask_clarification(tool, missing): a supported request that cannot be executed yet. List every gap in `missing`:
  - date: a date is mentioned but ambiguous, conflicting or invalid (a day of the month alone such as "يوم 15"; "next week", "الويكند", "آخر الشهر"; a bare weekday that is today's weekday; an impossible date such as 31/2).
  - time: no usable clock time (missing, vague such as "بدري", "بعدين", or a period alone such as "الصبح"; several times; an hour that does not fit its period such as "8 الضهر").
  - am_pm: an hour from 1 to 12 with nothing that says morning or evening.
  - past: everything is clear, but that moment has already passed.
  - reminder_text: a reminder with no content or vague content ("حاجة", "something"). Never for an alarm.
- decline(reason): anything else. Reasons, the first that applies in this order: deletion (deleting or cancelling), modification (changing an existing reminder or alarm), recurring (every day, every week, ...), other_person (a reminder for someone else, e.g. "فكّر أحمد"), multi_action (more than one action in one message), other_tool (anything else: weather, calls, messages, timers, music, ...).

How to read times:
- No date mentioned: the next occurrence of that time (today if it is still ahead, otherwise tomorrow). "بكرة" = tomorrow, "بعد بكرة" = in two days, "النهارده" = today, "الليلة" = tonight.
- A bare weekday = its next occurrence 1-6 days ahead; a weekday with "الجاي" or "next" = 1-7 days ahead; a weekday with "ده" or "this" = 0-6 days ahead.
- Explicit dates are day/month (15/10 = 15 October).
- Periods: "الصبح"/"ص"/am = morning; "الضهر" = around noon (11 to 3); "العصر"/"بعد الضهر" = afternoon; "المغرب" = 5 to 8 pm; "بالليل"/"العشا"/"م"/pm = evening or night. 12 to 5 "بالليل" is after midnight, on the next calendar day.
- Relative times are exact: "بعد ساعتين" = two hours from now, "بعد ربع ساعة" = 15 minutes from now.
Write `datetime` as local time, {fmt}, without a time zone. Write `reminder_text` in the user's own words."""


def system_prompt(ref: dt.datetime, tz_name: str = "Africa/Cairo") -> str:
    local = ref.astimezone(ZoneInfo(tz_name))
    return SYSTEM_PROMPT.format(weekday=WEEKDAYS_EN[local.weekday()], date=local.strftime("%Y-%m-%d"),
                                time=local.strftime("%H:%M"), tz=tz_name, fmt=DATETIME_FORMAT)


def build_messages(text: str, ref: dt.datetime, tz_name: str = "Africa/Cairo") -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt(ref, tz_name)}, {"role": "user", "content": text}]


# --------------------------------------------------------------------------- parsing
#: closing reasoning tags: Jais 3 (three effort levels) and Qwen-style templates
THINK_CLOSE = re.compile(r"</ifm\|think(?:_fast|_faster)?>|</think>")
CALL_BLOCK = re.compile(r"<(ifm\|)?tool_call>(.*?)</(?:ifm\|)?tool_call>", re.DOTALL)
XML_ARG = re.compile(r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*(?:<ifm\|arg_type>(.*?)</ifm\|arg_type>\s*)?"
                     r"<ifm\|arg_value>(.*?)</ifm\|arg_value>", re.DOTALL)


def split_reasoning(text: str) -> tuple[str, str]:
    """(reasoning, answer): everything up to the last closing think tag is reasoning."""
    ends = list(THINK_CLOSE.finditer(text))
    if not ends:
        return "", text
    return text[:ends[-1].start()], text[ends[-1].end():]


def _xml_value(raw: str, typ: str | None) -> Any:
    raw = raw.strip()
    if typ in ("array", "object", "integer", "number", "boolean", "null") or raw[:1] in ("[", "{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def parse_tool_calls(answer: str) -> list[dict[str, Any]]:
    """Tool calls in the answer text, as ``{"name": ..., "arguments": {...}}``. A call whose
    JSON or arguments cannot be read is returned as ``{"name": None, "error": ...}``."""
    calls: list[dict[str, Any]] = []
    for m in CALL_BLOCK.finditer(answer):
        body = m.group(2).strip()
        if body.startswith("{"):
            try:
                obj = json.loads(body)
                args = obj.get("arguments", obj.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append({"name": obj.get("name"), "arguments": args if isinstance(args, dict) else {}})
            except (json.JSONDecodeError, AttributeError) as e:
                calls.append({"name": None, "error": f"unreadable call: {e}"})
        else:
            name, _, rest = body.partition("\n")
            args = {k.strip(): _xml_value(v, t.strip() if t else None) for k, t, v in XML_ARG.findall(rest)}
            calls.append({"name": name.strip() or None, "arguments": args})
    return calls


# --------------------------------------------------------------------------- mapping
class ToolCallError(ValueError):
    """The model's output is not exactly one valid call."""


def parse_datetime(value: Any, tz_name: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ToolCallError("datetime is not a string")
    try:
        when = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as e:
        raise ToolCallError(f"unreadable datetime {value!r}") from e
    return when if when.tzinfo is not None else when.replace(tzinfo=ZoneInfo(tz_name))


def _questions(targets: Sequence[str], tool_name: Any) -> str:
    """The contract's fixed questions (section 6); the AM/PM question names no hour here."""
    parts = []
    for gap in targets:
        q = QUESTIONS[gap]
        if isinstance(q, dict):
            q = q.get(str(tool_name), q["create_reminder"])
        parts.append("الصبح ولا بالليل؟" if gap == "am_pm" else q)
    return " ".join(parts)


def to_response(calls: Sequence[Mapping[str, Any]], system: str, tz_name: str = "Africa/Cairo") -> tuple[Response, str | None]:
    """The response for exactly one valid call, and the decline reason (if any). Raises
    :class:`ToolCallError` for anything else."""
    if not calls:
        raise ToolCallError("no tool call")
    if len(calls) > 1:
        raise ToolCallError(f"{len(calls)} tool calls")
    call = calls[0]
    name, args = call.get("name"), call.get("arguments")
    if name not in TOOL_NAMES:
        raise ToolCallError(call.get("error") or f"unknown tool {name!r}")
    if not isinstance(args, Mapping):
        raise ToolCallError("arguments are not an object")
    try:
        if name in ("create_reminder", "create_alarm"):
            when = parse_datetime(args.get("datetime"), tz_name)
            text = args.get("reminder_text") if name == "create_reminder" else None
            if name == "create_reminder" and (not isinstance(text, str) or not text.strip()):
                raise ToolCallError("create_reminder without reminder_text")
            tool = ToolCall(name=name, datetime=when, reminder_text=text.strip() if text else None)
            message = (f"تمام، هفكرك بـ«{tool.reminder_text}» {describe_datetime(when)}." if tool.reminder_text
                       else f"تمام، المنبه هيرن {describe_datetime(when)}.")
            return Response(system=system, route="ACT", intent=name, tool_call=tool, message=message,
                            extracted_datetime=when, extracted_reminder_text=tool.reminder_text), None
        if name == "ask_clarification":
            missing = args.get("missing")
            if isinstance(missing, str):
                missing = [missing]
            if not isinstance(missing, list) or not missing or not all(m in GAP_ORDER for m in missing):
                raise ToolCallError(f"invalid missing {missing!r}")
            targets = [g for g in GAP_ORDER if g in missing]
            tool_name = args.get("tool")
            return Response.model_validate({"system": system, "route": "CLARIFY", "intent": tool_name,
                                            "clarify_targets": targets,
                                            "message": _questions(targets, tool_name)}), None
        reason = args.get("reason")
        if reason not in REASONS:
            raise ToolCallError(f"invalid reason {reason!r}")
        return Response(system=system, route="ABSTAIN", intent="unsupported", message=ABSTAIN_MESSAGE), reason
    except ValidationError as e:
        raise ToolCallError(f"response schema: {e.errors()[0]['msg']}") from e


# --------------------------------------------------------------------------- backends
@dataclass
class Completion:
    """One model answer. ``tool_calls`` is None when the backend left parsing to us."""

    text: str
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    n_tokens: int = 0
    finish_reason: str = "stop"  # "length": cut at max_new_tokens


class Backend(Protocol):
    def complete(self, conversations: Sequence[list[dict[str, str]]]) -> list[Completion]: ...


class HFToolBackend:
    """Transformers generation with the model's chat template and the tools in it.

    ``template_kwargs`` go to ``apply_chat_template`` (Jais 3: ``reasoning_effort``,
    ``tool_call_format``). ``sampling`` None = greedy; else the ``generate`` sampling arguments
    (the Jais 3 card recommends temperature 0.7, top_p 0.95, top_k 20 for Arabic with
    reasoning). Requests are left-padded and generated in batches; the seed is set once per
    ``complete`` call, so sampled outputs depend on the batch composition (fixed by the order)."""

    def __init__(self, model: Any, tokenizer: Any, *, template_kwargs: Mapping[str, Any] | None = None,
                 max_new_tokens: int = 8192, sampling: Mapping[str, Any] | None = None, seed: int = 0,
                 batch_size: int = 4) -> None:
        self.model, self.tok = model, tokenizer
        self.template_kwargs = dict(template_kwargs or {})
        self.max_new_tokens, self.sampling, self.seed = max_new_tokens, dict(sampling or {}), seed
        self.batch_size = batch_size

    def render(self, messages: list[dict[str, str]]) -> str:
        return str(self.tok.apply_chat_template(messages, tools=tool_schemas(), tokenize=False,
                                                add_generation_prompt=True, **self.template_kwargs))

    def complete(self, conversations: Sequence[list[dict[str, str]]]) -> list[Completion]:
        import torch

        torch.manual_seed(self.seed)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        eos = self.model.generation_config.eos_token_id if hasattr(self.model, "generation_config") else None
        eos_ids = set(eos if isinstance(eos, list) else [eos] if eos is not None else []) | {self.tok.eos_token_id}
        device = next(self.model.parameters()).device
        out: list[Completion] = []
        for start in range(0, len(conversations), self.batch_size):
            prompts = [self.tok(self.render(m), add_special_tokens=False)["input_ids"]
                       for m in conversations[start:start + self.batch_size]]
            width = max(map(len, prompts))
            ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
            kw: dict[str, Any] = {"do_sample": bool(self.sampling), **self.sampling}
            with torch.no_grad():
                gen = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=self.max_new_tokens,
                                          pad_token_id=pad, **kw)
            for row in gen[:, width:].tolist():
                ended = next((i for i, t in enumerate(row) if t in eos_ids), None)
                kept = row if ended is None else row[:ended]
                text = str(self.tok.decode(kept, skip_special_tokens=False))
                reasoning, answer = split_reasoning(text)
                out.append(Completion(text=answer, reasoning=reasoning, n_tokens=len(kept),
                                      finish_reason="stop" if ended is not None else "length"))
        return out


class OpenAIToolBackend:
    """An OpenAI-compatible ``/v1/chat/completions`` server (vLLM or SGLang started with the
    model's reasoning and tool-call parsers). The server parses the calls; requests run in
    ``workers`` parallel threads. ``post`` is injectable for offline tests."""

    def __init__(self, base_url: str, model: str, *, chat_template_kwargs: Mapping[str, Any] | None = None,
                 max_tokens: int = 8192, sampling: Mapping[str, Any] | None = None, seed: int = 0,
                 workers: int = 8, api_key: str = "EMPTY", timeout: float = 600.0,
                 post: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.max_tokens, self.seed, self.workers = model, max_tokens, seed, workers
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.sampling = dict(sampling or {"temperature": 0.0})
        self.api_key, self.timeout = api_key, timeout
        self.post = post or self._post

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        import urllib.request

        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (a configured server)
            return dict(json.loads(resp.read().decode("utf-8")))

    def _one(self, i: int, messages: list[dict[str, str]]) -> Completion:
        body = {"model": self.model, "messages": messages, "tools": tool_schemas(), "tool_choice": "auto",
                "max_tokens": self.max_tokens, "seed": self.seed + i, **self.sampling,
                "chat_template_kwargs": self.chat_template_kwargs}
        choice = self.post(self.url, body)["choices"][0]
        msg = choice.get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            try:
                args = fn.get("arguments")
                calls.append({"name": fn.get("name"), "arguments": json.loads(args) if isinstance(args, str) else args})
            except json.JSONDecodeError as e:
                calls.append({"name": None, "error": f"unreadable call: {e}"})
        content = msg.get("content") or ""
        if not calls:  # a server without a tool parser leaves the calls in the text
            calls = parse_tool_calls(content)
        return Completion(text=content, reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                          tool_calls=calls, finish_reason=choice.get("finish_reason") or "stop")

    def complete(self, conversations: Sequence[list[dict[str, str]]]) -> list[Completion]:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._one, range(len(conversations)), conversations))


# --------------------------------------------------------------------------- the system
@dataclass
class TraceRow:
    id: str
    route: str | None
    intent: str | None
    reason: str | None
    error: str | None
    n_tokens: int
    finish_reason: str
    answer: str
    reasoning_chars: int
    calls: list[dict[str, Any]] = field(default_factory=list)


class ToolCallingSystem:
    """S8 behind the harness: ``respond_items`` returns one response (or None) per item.
    ``predict`` gives the intent of normalized texts sent at ``ood_ref``, for the intent-level
    out-of-distribution checks (``athar.eval.ood``)."""

    def __init__(self, backend: Backend, name: str = "S8",
                 ood_ref: dt.datetime | None = None) -> None:
        self.backend, self.name = backend, name
        self.ood_ref = ood_ref or dt.datetime(2026, 10, 15, 10, 0, tzinfo=ZoneInfo("Africa/Cairo"))
        self.trace: list[TraceRow] = []
        #: the responses of the latest ``respond_items`` call (for :func:`diagnose`)
        self.outputs: list[Response | None] = []

    def respond_items(self, items: Sequence[GoldItem]) -> list[Response | None]:
        completions = self.backend.complete([build_messages(it.text, it.ref_datetime, it.timezone) for it in items])
        if len(completions) != len(items):
            raise ValueError(f"{self.name}: {len(completions)} completions for {len(items)} items")
        out: list[Response | None] = []
        for it, c in zip(items, completions, strict=True):
            calls = c.tool_calls if c.tool_calls is not None else parse_tool_calls(c.text)
            response, reason, error = None, None, None
            if c.finish_reason == "length" and not calls:
                error = "truncated: no answer within the token limit"
            else:
                try:
                    response, reason = to_response(calls, self.name, it.timezone)
                except ToolCallError as e:
                    error = str(e)
            out.append(response)
            self.trace.append(TraceRow(id=it.id, route=response.route if response else None,
                                       intent=response.intent if response else None, reason=reason, error=error,
                                       n_tokens=c.n_tokens, finish_reason=c.finish_reason, answer=c.text,
                                       reasoning_chars=len(c.reasoning), calls=list(calls)))
        self.outputs = out
        return out

    def predict(self, normalized_texts: Sequence[str]) -> list[str]:
        items = [GoldItem(id=f"ood{i}", family_id=f"ood{i}", text=t, ref_datetime=self.ood_ref,
                          intent="unsupported", route="ABSTAIN", unsupported_reason="other")
                 for i, t in enumerate(normalized_texts)]
        return [r.intent if r is not None else "invalid" for r in self.respond_items(items)]

    def trace_rows(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.trace]


def diagnose(items: Sequence[GoldItem], outputs: Sequence[Response | None],
             trace: Sequence[TraceRow]) -> dict[str, Any]:
    """Where S8's failures come from (descriptive, for the error analysis): invalid outputs by
    cause, truncation, decline-reason accuracy on gold ABSTAIN items, and executions with the
    right tool and moment but a reminder text that differs from gold (the strict text match is
    part of Safe Task Success)."""
    from athar.eval.metrics import score_item
    from athar.timeexpr import is_after

    by_id = {r.id: r for r in trace}
    errors: dict[str, int] = {}
    for r in trace:
        if r.error:
            key = r.error.split(":")[0]
            errors[key] = errors.get(key, 0) + 1
    abstain = [it for it in items if it.route == "ABSTAIN"]
    reason_ok = sum(by_id[it.id].reason == it.unsupported_reason for it in abstain if it.id in by_id)
    text_only = 0
    for it, o in zip(items, outputs, strict=True):
        if it.route == "ACT" and o is not None and o.tool_call is not None and o.tool_call.name == it.intent:
            gold_dt = it.args.datetime
            same = gold_dt is not None and not is_after(o.tool_call.datetime, gold_dt) and not is_after(gold_dt, o.tool_call.datetime)
            if same and it.intent == "create_reminder" and not score_item(it, o).success:
                text_only += 1
    n = max(len(trace), 1)
    return {"n": len(trace), "invalid": sum(r.error is not None for r in trace), "invalid_by_cause": errors,
            "truncated": sum(r.finish_reason == "length" for r in trace),
            "mean_new_tokens": sum(r.n_tokens for r in trace) / n,
            "decline_reason_accuracy": reason_ok / len(abstain) if abstain else None,
            "act_wrong_text_only": text_only}
