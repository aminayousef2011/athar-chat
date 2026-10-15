"""S8 end-to-end tool calling (athar.agent), offline: parsing every call format, the mapping
to the response schema, both backends with fakes, and the harness on a perfect and a broken
fake model."""

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from athar import agent
from athar.data.io import load_items
from athar.eval import harness as H
from athar.timeexpr import GAP_ORDER, cairo

PILOT = Path(__file__).resolve().parents[1] / "data" / "pilot" / "pilot.jsonl"
FAST = {"seed": 1, "level": 0.95, "resamples": 100, "permutations": 100, "alpha": 0.05,
        "calibration_bins": 10, "oracle": False, "perturbations": [], "comparisons": []}
CAIRO = ZoneInfo("Africa/Cairo")


def test_tool_schemas_cover_the_contract():
    tools = {t["function"]["name"]: t["function"]["parameters"] for t in agent.tool_schemas()}
    assert tuple(tools) == agent.TOOL_NAMES
    assert tools["create_reminder"]["required"] == ["datetime", "reminder_text"]
    assert tools["ask_clarification"]["properties"]["missing"]["items"]["enum"] == list(GAP_ORDER)
    assert set(tools["decline"]["properties"]["reason"]["enum"]) >= {
        "deletion", "modification", "recurring", "other_person", "multi_action", "other_tool"}
    assert all(p["additionalProperties"] is False for p in tools.values())


def test_system_prompt_states_the_local_moment():
    p = agent.system_prompt(cairo("2026-10-07T10:00"))
    assert "Wednesday 2026-10-07 at 10:00 (Africa/Cairo)" in p
    # after Cairo's DST change the same UTC instant is an hour earlier in local time
    later = dt.datetime(2026, 10, 30, 8, 0, tzinfo=dt.UTC)
    assert "Friday 2026-10-30 at 10:00" in agent.system_prompt(later)
    msgs = agent.build_messages("فكرني", cairo("2026-10-07T10:00"))
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[1]["content"] == "فكرني"


@pytest.mark.parametrize("text, reasoning, answer", [
    ("no tags", "", "no tags"),
    ("<ifm|think>\nhmm</ifm|think>ok", "<ifm|think>\nhmm", "ok"),
    ("x</ifm|think_faster>y", "x", "y"),
    ("<think>a</think>b</think>c", "<think>a</think>b", "c"),
])
def test_split_reasoning(text, reasoning, answer):
    assert agent.split_reasoning(text) == (reasoning, answer)


def test_parse_every_call_format():
    as_json = '<ifm|tool_calls>\n<ifm|tool_call>{"name": "create_alarm", "arguments": {"datetime": "2026-10-08T07:00"}}</ifm|tool_call>\n</ifm|tool_calls>'
    as_xml = ("<ifm|tool_calls>\n<ifm|tool_call>ask_clarification\n<ifm|arg_key>tool</ifm|arg_key>\n"
              "<ifm|arg_value>create_reminder</ifm|arg_value>\n<ifm|arg_key>missing</ifm|arg_key>\n"
              '<ifm|arg_value>["time", "reminder_text"]</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>')
    as_typed = ("<ifm|tool_call>decline\n<ifm|arg_key>reason</ifm|arg_key>\n<ifm|arg_type>string</ifm|arg_type>\n"
                "<ifm|arg_value>recurring</ifm|arg_value>\n</ifm|tool_call>")
    as_hermes = '<tool_call>\n{"name": "create_reminder", "arguments": "{\\"datetime\\": \\"2026-10-08T07:00\\", \\"reminder_text\\": \\"عيش\\"}"}\n</tool_call>'
    assert agent.parse_tool_calls(as_json) == [{"name": "create_alarm", "arguments": {"datetime": "2026-10-08T07:00"}}]
    assert agent.parse_tool_calls(as_xml) == [{"name": "ask_clarification",
                                               "arguments": {"tool": "create_reminder", "missing": ["time", "reminder_text"]}}]
    assert agent.parse_tool_calls(as_typed) == [{"name": "decline", "arguments": {"reason": "recurring"}}]
    assert agent.parse_tool_calls(as_hermes)[0]["arguments"]["reminder_text"] == "عيش"
    assert agent.parse_tool_calls("plain text, no call") == []
    bad = agent.parse_tool_calls("<ifm|tool_call>{not json}</ifm|tool_call>")
    assert bad[0]["name"] is None and "unreadable" in bad[0]["error"]
    assert len(agent.parse_tool_calls(as_json + as_typed)) == 2
    assert agent._xml_value("[broken", None) == "[broken"


def call(name, **arguments):
    return [{"name": name, "arguments": arguments}]


def test_reminder_alarm_clarify_and_decline_map_to_the_schema():
    r, reason = agent.to_response(call("create_reminder", datetime="2026-10-08T07:00", reminder_text=" أشتري عيش "), "S8")
    assert (r.route, r.intent, reason) == ("ACT", "create_reminder", None)
    assert r.tool_call.datetime == dt.datetime(2026, 10, 8, 7, 0, tzinfo=CAIRO)   # naive -> Cairo wall clock
    assert r.tool_call.datetime.utcoffset() == dt.timedelta(hours=3)
    assert r.tool_call.reminder_text == "أشتري عيش" and "أشتري عيش" in r.message
    alarm, _ = agent.to_response(call("create_alarm", datetime="2026-10-08T04:00:00Z", reminder_text="ignored"), "S8")
    assert alarm.tool_call.reminder_text is None and alarm.tool_call.datetime == dt.datetime(2026, 10, 8, 4, tzinfo=dt.UTC)
    assert "المنبه" in alarm.message
    clar, _ = agent.to_response(call("ask_clarification", tool="create_alarm", missing=["am_pm", "date", "am_pm"]), "S8")
    assert (clar.route, clar.intent, clar.clarify_targets) == ("CLARIFY", "create_alarm", ["date", "am_pm"])
    assert "الصبح ولا بالليل؟" in clar.message
    one, _ = agent.to_response(call("ask_clarification", tool="create_reminder", missing="time"), "S8")
    assert one.clarify_targets == ["time"] and one.message == "أفكرك الساعة كام؟"
    dec, reason = agent.to_response(call("decline", reason="other_person"), "S8")
    assert (dec.route, dec.intent, reason) == ("ABSTAIN", "unsupported", "other_person")


@pytest.mark.parametrize("calls, message", [
    ([], "no tool call"),
    (call("create_alarm", datetime="2026-10-08T07:00") * 2, "2 tool calls"),
    (call("send_email"), "unknown tool"),
    ([{"name": None, "error": "unreadable call: x"}], "unreadable call"),
    ([{"name": "decline", "arguments": "reason"}], "not an object"),
    (call("create_reminder", datetime="2026-10-08T07:00"), "without reminder_text"),
    (call("create_reminder", datetime="tomorrow 7", reminder_text="x"), "unreadable datetime"),
    (call("create_alarm", datetime=7), "not a string"),
    (call("ask_clarification", tool="create_alarm", missing=[]), "invalid missing"),
    (call("ask_clarification", tool="create_alarm", missing=["confirm"]), "invalid missing"),
    (call("ask_clarification", tool="weather", missing=["time"]), "response schema"),
    (call("decline", reason="because"), "invalid reason"),
])
def test_anything_but_one_valid_call_is_an_error(calls, message):
    with pytest.raises(agent.ToolCallError, match=message):
        agent.to_response(calls, "S8")


def gold_call(it, fmt):
    """The call a perfect model makes for a gold item, written in one of the call formats."""
    if it.route == "ACT":
        args = {"datetime": it.args.datetime.astimezone(ZoneInfo(it.timezone)).strftime("%Y-%m-%dT%H:%M")}
        if it.intent == "create_reminder":
            args["reminder_text"] = it.args.reminder_text
        name = it.intent
    elif it.route == "CLARIFY":
        name, args = "ask_clarification", {"tool": it.intent, "missing": list(it.clarify_targets)}
    else:
        name, args = "decline", {"reason": it.unsupported_reason}
    if fmt == "xml":
        body = "".join(f"<ifm|arg_key>{k}</ifm|arg_key>\n<ifm|arg_value>"
                       f"{v if isinstance(v, str) else json.dumps(v)}</ifm|arg_value>\n" for k, v in args.items())
        return f"<ifm|tool_calls>\n<ifm|tool_call>{name}\n{body}</ifm|tool_call>\n</ifm|tool_calls>"
    if fmt == "hermes":
        return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"
    return f"<ifm|tool_calls>\n<ifm|tool_call>{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}</ifm|tool_call>\n</ifm|tool_calls>"


class FakeBackend:
    """Answers from the gold labels (by request text) after a short reasoning block."""

    def __init__(self, items, broken=()):
        self.by_text = {it.text: it for it in items}
        self.broken = set(broken)
        self.seen = []

    def complete(self, conversations):
        self.seen += conversations
        out = []
        for i, msgs in enumerate(conversations):
            it = self.by_text[msgs[-1]["content"]]
            if it.id in self.broken:
                out.append(agent.Completion(text="", reasoning="...", n_tokens=50, finish_reason="length"))
                continue
            out.append(agent.Completion(text=gold_call(it, ("json", "xml", "hermes")[i % 3]), reasoning="thinking",
                                        n_tokens=20))
        return out


def test_a_perfect_tool_caller_scores_one_through_the_harness():
    items = load_items(PILOT)
    s8 = agent.ToolCallingSystem(FakeBackend(items), name="S8")
    rep = H.evaluate({"S8": s8}, items, FAST)["systems"]["S8"]
    assert rep["metrics"]["safe_task_success"] == 1.0
    assert rep["calibration"] is None and "gate" not in rep      # no intent probabilities
    diag = agent.diagnose(items, s8.outputs, s8.trace)
    assert len(s8.outputs) == len(items) == len(s8.trace)
    assert diag["invalid"] == 0 and diag["decline_reason_accuracy"] == 1.0 and diag["act_wrong_text_only"] == 0
    first = s8.backend.seen[0]
    assert first[0]["role"] == "system" and "Wednesday 2026-10-07 at 10:00" in first[0]["content"]


def test_invalid_and_truncated_outputs_score_as_failures_with_a_cause():
    items = load_items(PILOT)
    broken = [items[0].id, items[1].id]
    s8 = agent.ToolCallingSystem(FakeBackend(items, broken=broken))
    outputs = H.run(s8, items)
    assert outputs[0] is None and outputs[1] is None and all(o is not None for o in outputs[2:])
    rep = H.evaluate({"S8": agent.ToolCallingSystem(FakeBackend(items, broken=broken))}, items, FAST)
    assert rep["systems"]["S8"]["metrics"]["invalid_outputs"] == 2
    diag = agent.diagnose(items, outputs, s8.trace)
    assert diag["truncated"] == 2 and diag["invalid_by_cause"] == {"truncated": 2}
    rows = s8.trace_rows()
    assert rows[0]["error"].startswith("truncated") and rows[2]["route"] is not None


def test_wrong_text_with_the_right_moment_is_diagnosed():
    items = [it for it in load_items(PILOT) if it.route == "ACT" and it.intent == "create_reminder"][:1]

    class WrongText(FakeBackend):
        def complete(self, conversations):
            out = super().complete(conversations)
            for c in out:
                c.text = c.text.replace(items[0].args.reminder_text, "حاجة تانية خالص")
            return out

    s8 = agent.ToolCallingSystem(WrongText(items))
    outputs = H.run(s8, items)
    assert agent.diagnose(items, outputs, s8.trace)["act_wrong_text_only"] == 1


def test_predict_gives_intents_for_the_ood_checks():
    class Fixed:
        def complete(self, conversations):
            assert all("Thursday 2026-10-15 at 10:00" in m[0]["content"] for m in conversations)
            return [agent.Completion(text=t) for t in (
                '<tool_call>{"name": "decline", "arguments": {"reason": "other_tool"}}</tool_call>',
                '<tool_call>{"name": "create_alarm", "arguments": {"datetime": "2026-10-16T07:00"}}</tool_call>',
                "I cannot help with that.")]

    assert agent.ToolCallingSystem(Fixed()).predict(["شغل مزيكا", "صحيني 7 الصبح", "ازيك"]) == [
        "unsupported", "create_alarm", "invalid"]


def test_the_harness_rejects_a_wrong_number_of_responses():
    class Short:
        name = "S8"

        def respond_items(self, items):
            return []

    with pytest.raises(ValueError, match="0 responses"):
        H.run(Short(), load_items(PILOT)[:2])

    class Lossy:
        def complete(self, conversations):
            return []

    with pytest.raises(ValueError, match="0 completions"):
        agent.ToolCallingSystem(Lossy()).respond_items(load_items(PILOT)[:2])


def test_openai_backend_sends_tools_and_reads_parsed_or_textual_calls():
    bodies = []

    def post(url, body):
        bodies.append((url, body))
        i = body["seed"]
        if i == 0:   # a server with the tool parser: arguments as a JSON string
            msg = {"content": None, "reasoning_content": "r", "tool_calls": [
                {"type": "function", "function": {"name": "create_alarm", "arguments": '{"datetime": "2026-10-08T07:00"}'}}]}
        elif i == 1:  # no tool parser: the call stays in the text
            msg = {"content": '<ifm|tool_call>{"name": "decline", "arguments": {"reason": "deletion"}}</ifm|tool_call>'}
        else:
            msg = {"content": "", "tool_calls": [{"function": {"name": "decline", "arguments": "{oops"}}]}
        return {"choices": [{"message": msg, "finish_reason": "tool_calls" if i != 1 else "stop"}]}

    backend = agent.OpenAIToolBackend("http://server:8000/v1/", "inception42/Jais-3-7B", post=post, workers=2,
                                      chat_template_kwargs={"reasoning_effort": "high"},
                                      sampling={"temperature": 0.7, "top_p": 0.95, "top_k": 20})
    convs = [agent.build_messages(t, cairo("2026-10-07T10:00")) for t in ("صحيني 7 الصبح", "امسح المنبه", "x")]
    out = backend.complete(convs)
    assert out[0].tool_calls == [{"name": "create_alarm", "arguments": {"datetime": "2026-10-08T07:00"}}]
    assert out[0].reasoning == "r"
    assert out[1].tool_calls == [{"name": "decline", "arguments": {"reason": "deletion"}}]
    assert out[2].tool_calls[0]["name"] is None
    url, body = bodies[0]
    assert url == "http://server:8000/v1/chat/completions"
    assert body["chat_template_kwargs"] == {"reasoning_effort": "high"} and body["top_k"] == 20
    assert [t["function"]["name"] for t in body["tools"]] == list(agent.TOOL_NAMES)
