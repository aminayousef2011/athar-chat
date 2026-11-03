"""ATHAR Chat local demo: the frozen systems side by side, the locked test results, and the
pre-registered evaluation on 17 other native speakers.

    uv run --extra ui streamlit run app/streamlit_app.py

CPU only. The LLM systems (S4/S7/S8/S9) need GPUs or stored GPU outputs; their results are read
from the committed result files.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from athar import demo
from athar.intent.train import FreezeError, read_freeze
from athar.respond import describe_datetime

ROOT = Path(__file__).resolve().parents[1]
MODELS, MODELS_V2, RESULTS = ROOT / "models", ROOT / "models_v2", ROOT / "results"
TZ = "Africa/Cairo"

#: plain-language descriptions shown on each answer card
SYSTEM_PLAIN = {
    "S1": "Keyword rules written by hand",
    "S2": "Statistical classifier trained on 298 examples (words)",
    "S3": "Statistical classifier trained on 298 examples (words + letter patterns)",
    "S1-v1.1": "Keyword rules with the fixes from the error analysis",
}
TARGET_PLAIN = {"date": "the day", "time": "the time", "am_pm": "morning or evening (AM/PM)",
                "past": "a later time (the one given has passed)", "reminder_text": "what to remind you about"}
VERDICT = {"ACT": ("✅", "Sets it"), "CLARIFY": ("❓", "Asks first"), "ABSTAIN": ("🚫", "Declines")}

st.set_page_config(page_title="ATHAR Chat demo", page_icon="🗓️", layout="wide")
# Requests are Arabic: type them right to left (bidi keeps embedded English in order).
st.markdown("<style>.stTextArea textarea {direction: rtl; text-align: right; font-size: 1.15rem;}</style>",
            unsafe_allow_html=True)


@st.cache_resource
def systems() -> list[demo.Frozen]:
    return [*demo.load_systems(MODELS), demo.load_v11(MODELS_V2)]


def rtl(text: str, size: str = "1.1rem") -> None:
    st.markdown(f'<div dir="rtl" style="text-align:right;font-size:{size};line-height:1.7">{html.escape(text)}</div>',
                unsafe_allow_html=True)


def verdict(a: demo.Answer) -> str:
    """One plain sentence for an answer."""
    r = a.response
    icon, _ = VERDICT[r.route]
    if r.route == "ACT" and r.tool_call is not None:
        return f"{icon} **Sets {'a reminder' if r.tool_call.name == 'create_reminder' else 'an alarm'}**"
    if r.route == "CLARIFY":
        return f"{icon} **Asks for {', '.join(TARGET_PLAIN.get(t, t) for t in r.clarify_targets)}**"
    return f"{icon} **Declines** (outside what the assistant can do)"


def decision_key(a: demo.Answer) -> str:
    return json.dumps([a.response.route, demo.tool_call_view(a.response), list(a.response.clarify_targets)],
                      ensure_ascii=False)


st.title("ATHAR Chat")
st.markdown("##### An assistant that turns an Egyptian Arabic message into a reminder or an alarm, "
            "and knows when to ask or to say no.")
st.caption("A course project. Nothing is really created: reminders and alarms are simulated. "
           "Full results: docs/report.md.")

try:
    loaded = systems()
    freeze_id = read_freeze(MODELS)["freeze_sha256"][:12]
    freeze_v2 = read_freeze(MODELS_V2)["freeze_sha256"][:12]
except FreezeError as e:
    st.error(f"The frozen models failed verification, so the demo will not run them: {e}")
    st.stop()

with st.sidebar:
    st.subheader("🕒 When is the message sent?")
    st.caption("Words like «بكرة» (tomorrow) or «الخميس» (Thursday) are worked out from this moment, "
               "Cairo time.")
    use_now = st.checkbox("Use the current time", value=False)
    if use_now:
        ref = dt.datetime.now(ZoneInfo(TZ)).replace(second=0, microsecond=0)
    else:
        day = st.date_input("Date", value=dt.date(2026, 10, 19))
        clock = st.time_input("Time", value=dt.time(10, 0))
        ref = dt.datetime.combine(day, clock, tzinfo=ZoneInfo(TZ))
    st.markdown(f"**{ref.strftime('%A %d %B %Y, %H:%M')}**")
    rtl(describe_datetime(ref), "1rem")
    with st.expander("Technical details"):
        st.caption(f"Reference time `{ref.isoformat()}`")
        st.caption(f"Frozen models verified: `models/freeze.json` {freeze_id}…, `models_v2/freeze.json` {freeze_v2}…")
        st.caption("Confidence gate τ (chosen on dev): " +
                   ", ".join(f"{s.name} {s.tau if s.tau is not None else 'n/a'}" for s in loaded))

try_tab, test_tab, natural_tab, help_tab = st.tabs(
    ["💬 Try a message", "📊 Results: author's test", "📊 Results: 17 new writers", "❔ Help"])

with try_tab:
    c1, c2, c3 = st.columns(3)
    c1.info("**1. Pick or type a message**\n\nin Egyptian Arabic; English words are fine.")
    c2.info("**2. Four systems read it**\n\nEach decides what the user wants and whether it has every detail.")
    c3.info("**3. Each answer is one of three**\n\n✅ sets it · ❓ asks first · 🚫 declines")

    labels = ["(write your own)"] + [e.text for e in demo.EXAMPLES]
    pick = st.selectbox("Example message", labels, index=2)
    example = next((e for e in demo.EXAMPLES if e.text == pick), None)
    if example is not None:
        st.caption(f"What this example shows: {example.shows}")
    text = st.text_area("Message", value="" if example is None else example.text, height=80,
                        placeholder="مثال: فكرني بكرة الساعة 9 الصبح ابعت التقرير")
    if text.strip():
        answers = demo.answer_all(text, ref, loaded, TZ)
        groups: dict[str, list[str]] = {}
        for a in answers:
            groups.setdefault(decision_key(a), []).append(a.system)
        if len(groups) > 1:
            st.warning("The systems disagree on this message: " +
                       " vs ".join(" + ".join(g) for g in groups.values()) + ".")
        else:
            st.success("All four systems give the same answer.")
        for col, a, s in zip(st.columns(len(answers)), answers, loaded, strict=True):
            with col, st.container(border=True):
                st.markdown(f"**{a.system}**")
                st.caption(SYSTEM_PLAIN.get(a.system, demo.SYSTEM_LABELS.get(a.system, a.system)))
                st.markdown(verdict(a))
                call = a.response.tool_call
                if call is not None:
                    st.markdown(f"🗓️ {call.datetime.strftime('%A %d %B, %H:%M')}")
                    if call.reminder_text:
                        st.markdown("Reminder text:")
                        rtl(f"«{call.reminder_text}»")
                st.markdown("Reply to the user:")
                rtl(a.response.message)
                with st.expander("Technical details"):
                    conf = "rules (no probability)" if a.confidence is None else f"{a.confidence:.2f}"
                    st.markdown(f"route `{a.response.route}` · intent `{a.intent}` · confidence {conf}")
                    if a.response.clarify_targets:
                        st.markdown("targets: " + ", ".join(f"`{t}`" for t in a.response.clarify_targets))
                    if call is not None:
                        st.json(demo.tool_call_view(a.response))
                    st.caption(f"confidence gate: {demo.gate_note(a, s.tau)}")

with test_tab:
    st.subheader("Test 1: 120 messages written by the author, scored once")
    st.markdown("**Safe Task Success (STS)** = the share of messages handled exactly right: the right "
                "reminder or alarm at the right time with the right text, or the right question, or a decline "
                "when the request is out of scope.")
    summary = {r["system"]: r for r in demo.test_summary(RESULTS)}
    m1, m2, m3, m4 = st.columns(4)
    for col, name in zip((m1, m2, m3, m4), ("S1 keyword rules", "S2 word TF-IDF + LR", "S3 + character n-grams",
                                            "oracle (gold intent)"), strict=True):
        col.metric(name, f"{summary[name]['STS']:.3f}")
    st.markdown("**Finding:** the hand-written rules beat both trained classifiers. The training data were mostly "
                "AI-drafted and phrased reminders differently: 88% of drafted reminders use «فكرني», but only 50% "
                "of the test's, where 42% use the noun «تذكير». The test author also wrote the rules (see the "
                "report's threats to validity), which is why Test 2 exists.")
    with st.expander("Full table and pre-registered tests", expanded=True):
        st.dataframe(pd.DataFrame(demo.test_summary(RESULTS)), hide_index=True)
        st.markdown("Pre-registered hypotheses (family-level paired randomization, Holm over H1-H3)")
        st.dataframe(pd.DataFrame(demo.test_comparisons(RESULTS)), hide_index=True)
    with st.expander("Failure browser: every message a system got wrong"):
        rows = demo.failures(RESULTS)
        names = sorted({r["system"] for r in rows}, key=lambda n: (not n.startswith(("S1", "S2", "S3", "oracle")), n))
        system = st.selectbox("System", names)
        counts = demo.cause_counts(rows, system)
        cause = st.selectbox("Failure type (by outcome)", ["all"] + [f"{c} ({n})" for c, n in counts.items()])
        shown = [r for r in rows if r["system"] == system and (cause == "all" or cause.startswith(r["cause"] + " ("))]
        cols = ["id", "text", "cause", "gold_route", "pred_route", "gold_intent", "pred_intent",
                "gold_canonical", "pred_canonical", "gold_targets", "pred_targets"]
        st.dataframe(pd.DataFrame(shown)[cols], hide_index=True)
        st.caption("Rebuilt exactly from the locked runs by scripts/error_analysis.py. Grouped by outcome; report "
                   "section 7.3 groups the oracle's same 23 failures by root cause.")

with natural_tab:
    facts = demo.natural_facts(RESULTS)
    tables = demo.natural_tables(RESULTS)
    hyp = {r["hypothesis"]: r for r in demo.natural_hypotheses(RESULTS)}
    st.subheader(f"Test 2: {facts['confirmatory']} messages from {facts['writers']} other native speakers")
    st.markdown("Planned and frozen before any message was labelled; two people labelled every message "
                "independently; each system was scored once.")
    plain = {"N1 S1 vs S2": "Rules beat the classifier", "N2 S1-v1.1 vs S1": "Fixes from Test 1 help",
             "N3 S9 vs S8": "A safety policy improves LLM tool calling"}
    for col, (h, label) in zip(st.columns(3), plain.items(), strict=True):
        r = hyp[h]
        col.metric(f"{h.split()[0]}: {label}", f"{r['difference']:+.3f} STS",
                   "confirmed" if r["supported"] else "not confirmed", delta_color="normal" if r["supported"] else "off",
                   delta_arrow="off")
    st.markdown(
        "- **Rules vs classifier:** the hand-written rules won again, by more than in Test 1.\n"
        "- **Fixes from Test 1:** they changed one message out of 238. Fixes found by studying one test did "
        "not carry over to other people's wording.\n"
        "- **Safety policy:** letting fixed rules decide when Jais-3-7B's tool calls may act (S9) beat tool "
        "calling alone (S8), mostly by asking when a detail was missing.\n"
        "- **Checked afterwards:** both confirmed results hold with either annotator's labels before "
        "adjudication and with a lenient reminder-text rule (report section 8.7).")
    with st.expander("Pre-registered tests", expanded=True):
        st.dataframe(pd.DataFrame(demo.natural_hypotheses(RESULTS)), hide_index=True)
    with st.expander("Every system (238 messages)"):
        st.markdown(tables["Main results (238 confirmatory items)"])
    with st.expander("Test 1 vs Test 2, per system"):
        st.markdown(tables[next(k for k in tables if k.startswith("v1 locked test"))])
    with st.expander("Same scenarios: the author's messages vs the contributors'"):
        st.markdown(tables[next(k for k in tables if k.startswith("Same scenario slots"))])
    with st.expander("Where the safety policy gains over tool calling"):
        st.markdown(tables[next(k for k in tables if k.startswith("S9 vs S8"))])
    with st.expander("Per writer"):
        st.markdown(tables[next(k for k in tables if k.startswith("Per writer"))])
    st.caption(f"Tables from results/natural/tables.md (scripts/natural_tables.py); locked set "
               f"data/natural/manifest.json {facts['manifest']}….")

with help_tab:
    st.markdown(
        """
### How to use this demo

1. **Set when the message is sent** in the sidebar. Words like «بكرة», «الخميس» or «بعد ساعتين» are worked out
   from that moment, Cairo time. Tick *Use the current time* to use your clock.
2. **Pick an example or type your own message** in Egyptian Arabic; English words are fine
   («remind me بكرة الساعة 3 العصر»). Each example says what it shows.
3. **Read the four answer cards.** Each says, in plain words, what that system would do, and shows its
   Arabic reply. *Technical details* opens the raw output.
4. **Open the results tabs** for the two evaluations.

### The three possible answers

| Answer | When | Example |
|---|---|---|
| ✅ **Sets it** (ACT) | every detail is known | «فكرني بكرة الساعة 9 الصبح ابعت التقرير» |
| ❓ **Asks first** (CLARIFY) | something is missing: the day, the time, morning or evening, or what to remind about | «فكرني اكلم العميل» → asks for the time |
| 🚫 **Declines** (ABSTAIN) | the assistant cannot do it: deleting or changing a reminder, repeating ones, reminders for someone else, several at once, other apps | «امسح التذكير بتاع بكرة» |

### Glossary

| Term | Meaning |
|---|---|
| **S1** | hand-written keyword rules |
| **S2, S3** | statistical classifiers (TF-IDF + logistic regression) trained on 298 examples |
| **S1 v1.1** | S1 with the fixes found by studying Test 1's errors |
| **S4** | prompted LLMs (Jais 2, Jais 3, Nile-Chat, Qwen3) that only decide the intent |
| **S7** | Jais 2 fine-tuned with QLoRA |
| **S8** | Jais-3-7B doing the whole task through tool calls |
| **S9** | S8 with fixed rules deciding when it may act |
| **oracle** | the correct intent fed into the shared rules: the best the shared rules can do |
| **STS** | Safe Task Success: share of messages handled exactly right |
| **confidence** | a classifier's probability for its answer (the rules have none) |

### Good to know

- Only S1, S2, S3 and S1 v1.1 run here; the LLM systems need GPUs, so their results come from the
  committed outputs of their single scored runs.
- The demo loads a system only after its frozen files verify.
- Nothing is created: every reminder and alarm is simulated.
- The reminder text keeps the user's own words («اراجع الميزانية»), as the task contract requires.
"""
    )
