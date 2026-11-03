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

ROOT = Path(__file__).resolve().parents[1]
MODELS, MODELS_V2, RESULTS = ROOT / "models", ROOT / "models_v2", ROOT / "results"
TZ = "Africa/Cairo"
ROUTE_COLOR = {"ACT": "#1a7f37", "CLARIFY": "#9a6700", "ABSTAIN": "#cf222e"}

st.set_page_config(page_title="ATHAR Chat demo", layout="wide")
# Requests are Arabic: type them right to left (bidi keeps embedded English in order).
st.markdown("<style>.stTextArea textarea {direction: rtl; text-align: right; font-size: 1.1rem;}</style>",
            unsafe_allow_html=True)


@st.cache_resource
def systems() -> list[demo.Frozen]:
    return [*demo.load_systems(MODELS), demo.load_v11(MODELS_V2)]


def rtl(text: str) -> None:
    st.markdown(f'<div dir="rtl" style="text-align:right;font-size:1.1rem">{html.escape(text)}</div>',
                unsafe_allow_html=True)


def badge(route: str) -> str:
    return (f'<span style="background:{ROUTE_COLOR.get(route, "#57606a")};color:white;'
            f'padding:2px 8px;border-radius:6px;font-weight:600">{route}</span>')


st.title("ATHAR Chat: safe tool routing for Egyptian Arabic")
st.caption("Reminders and alarms are simulated: nothing is created. Course project; results in docs/report.md. "
           "New here? Open the Help tab.")

try:
    loaded = systems()
    freeze_id = read_freeze(MODELS)["freeze_sha256"][:12]
    freeze_v2 = read_freeze(MODELS_V2)["freeze_sha256"][:12]
except FreezeError as e:
    st.error(f"The frozen models failed verification, so the demo will not run them: {e}")
    st.stop()

with st.sidebar:
    st.subheader("Reference time")
    st.caption("Relative expressions («بكرة», «الخميس») are resolved from this moment, in Africa/Cairo.")
    use_now = st.checkbox("Use the current time", value=False)
    if use_now:
        ref = dt.datetime.now(ZoneInfo(TZ)).replace(second=0, microsecond=0)
    else:
        day = st.date_input("Date", value=dt.date(2026, 10, 19))
        clock = st.time_input("Time", value=dt.time(10, 0))
        ref = dt.datetime.combine(day, clock, tzinfo=ZoneInfo(TZ))
    st.write(f"`{ref.isoformat()}`")
    st.divider()
    st.caption(f"Frozen models verified: `models/freeze.json` {freeze_id}…, `models_v2/freeze.json` {freeze_v2}…")
    st.caption("Confidence gate τ (chosen on dev): " +
               ", ".join(f"{s.name} {s.tau if s.tau is not None else 'n/a'}" for s in loaded))

try_tab, test_tab, natural_tab, help_tab = st.tabs(
    ["Try a request", "Locked test (author, 120)", "New writers (17 speakers, 238)", "Help"])

with try_tab:
    labels = ["(write your own)"] + [e.text for e in demo.EXAMPLES]
    pick = st.selectbox("Example request", labels, index=2)
    example = next((e for e in demo.EXAMPLES if e.text == pick), None)
    if example is not None:
        st.info(example.shows)
    text = st.text_area("Request (Egyptian Arabic, code-switching allowed)",
                        value="" if example is None else example.text, height=80)
    if text.strip():
        answers = demo.answer_all(text, ref, loaded, TZ)
        decisions = {(a.intent, a.response.route, json.dumps(demo.tool_call_view(a.response), ensure_ascii=False),
                      tuple(a.response.clarify_targets)) for a in answers}
        if len(decisions) > 1:
            st.warning("The systems disagree on this request.")
        for col, a, s in zip(st.columns(len(answers)), answers, loaded, strict=True):
            with col:
                st.markdown(f"**{demo.SYSTEM_LABELS[a.system]}**")
                st.markdown(badge(a.response.route), unsafe_allow_html=True)
                conf = "rules (no probability)" if a.confidence is None else f"{a.confidence:.2f}"
                st.markdown(f"intent `{a.intent}` · confidence {conf}")
                rtl(a.response.message)
                call = demo.tool_call_view(a.response)
                if call is not None:
                    st.json(call)
                if a.response.clarify_targets:
                    st.markdown("asks for: " + ", ".join(f"`{t}`" for t in a.response.clarify_targets))
                st.caption(f"confidence gate: {demo.gate_note(a, s.tau)}")

with test_tab:
    st.subheader("Locked test: 120 author-written requests, scored once")
    st.dataframe(pd.DataFrame(demo.test_summary(RESULTS)), hide_index=True)
    st.markdown("**Pre-registered hypotheses** (family-level paired randomization, Holm over H1-H3)")
    st.dataframe(pd.DataFrame(demo.test_comparisons(RESULTS)), hide_index=True)
    st.markdown(
        "The keyword rules beat both classifiers. In the AI-drafted training data 88% of reminders use "
        "«فكرني» and 5% the noun «تذكير»; in the test, 50% and 42%. The classifiers learned the drafting "
        "style. The test author also wrote S1's rules: see the report's threats to validity.")
    st.divider()
    st.subheader("Failure browser (error analysis)")
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
    st.subheader(f"Pre-registered evaluation: {facts['confirmatory']} confirmatory messages from "
                 f"{facts['writers']} other native speakers")
    st.markdown(
        "Hypotheses, systems and analysis code were frozen before any message was labelled; two annotators "
        "labelled every message independently; every system was scored once. Design: "
        "`docs/protocol_natural.md`; full write-up: report section 8. "
        f"Locked set `data/natural/manifest.json` {facts['manifest']}….")
    st.markdown("**Pre-registered hypotheses** (family-level paired randomization, Holm over N1-N3)")
    st.dataframe(pd.DataFrame(demo.natural_hypotheses(RESULTS)), hide_index=True)
    st.markdown(
        "- **N1:** the keyword rules beat TF-IDF again, by more than on the author's test.\n"
        "- **N2:** the rule fixes found by studying the locked test's failures changed one item here.\n"
        "- **N3:** letting the deterministic policy decide when Jais-3-7B's tool calls may act (S9) beats "
        "tool calling alone (S8), mostly by asking when a detail is missing.")
    st.markdown("**Every system** (238 items; multi-seed systems averaged per item)")
    st.markdown(tables["Main results (238 confirmatory items)"])
    with st.expander("Locked test vs new writers, per system (descriptive)"):
        st.markdown(tables[next(k for k in tables if k.startswith("v1 locked test"))])
    with st.expander("Same scenarios, author's items vs contributors' items"):
        st.markdown(tables[next(k for k in tables if k.startswith("Same scenario slots"))])
    with st.expander("Where the hybrid gains over tool calling"):
        st.markdown(tables[next(k for k in tables if k.startswith("S9 vs S8"))])
    with st.expander("Per writer"):
        st.markdown(tables[next(k for k in tables if k.startswith("Per writer"))])
    st.caption("Tables from results/natural/tables.md (scripts/natural_tables.py), read from the committed outputs.")

with help_tab:
    st.markdown(
        """
### How to use this demo

1. **Set the reference time** in the sidebar. Relative words («بكرة», «الخميس», «بعد ساعتين») are resolved
   from this moment in Africa/Cairo. Tick *Use the current time* to use your clock.
2. **Pick an example or type your own request** in Egyptian Arabic; English words are fine
   («remind me بكرة الساعة 3 العصر»). Each example says what it illustrates.
3. **Read the four answers side by side.** Each column shows the route, the recognized intent, the
   Arabic reply, and, for ACT, the simulated tool call. A warning appears when the systems disagree.
4. **Browse the results tabs**: the locked test written by the author, and the evaluation on 17 other
   native speakers.

### What each answer means

| Route | Meaning |
|---|---|
| **ACT** | Every detail is known: a simulated `create_reminder(datetime, reminder_text)` or `create_alarm(datetime)` |
| **CLARIFY** | Something is missing (`date`, `time`, `am_pm`, `past`, `reminder_text`): the system asks instead of guessing |
| **ABSTAIN** | Out of scope (deletion, modification, recurrence, another person, several actions, other tools): a polite decline |

*Confidence* is the classifier's probability for its intent (rules have none). *Confidence gate* says
what the dev-chosen threshold would do; the answers themselves are ungated, as in the metrics.

### What runs here

- **S1**: keyword rules; **S2**: word TF-IDF + logistic regression; **S3**: S2 + character n-grams. Loaded from
  `models/` after `models/freeze.json` verifies.
- **S1 v1.1**: S1 with the extractor fixes and extra out-of-scope verbs from the error analysis (core v1.1),
  loaded after `models_v2/freeze.json` verifies.
- All four share the normalizer, time and argument extractor, execution policy and Arabic reply templates;
  S1-S3 use the frozen v1 core.

The LLM systems (Jais 2, Jais 3, Nile-Chat, Qwen3, QLoRA Jais 2, Jais 3 tool calling and the S9 hybrid)
need GPUs. Their results are shown from the committed outputs of their single scored runs.

### Good to know

- Requests are typed right to left; mixed Arabic and English stays in order.
- Nothing is created: the tool calls are simulated.
- The reminder text is stored in the user's own words («اراجع الميزانية»), as the task contract requires.
- Every number in the results tabs comes from a committed file; the report (`docs/report.md`) gives the
  confidence intervals, the statistics and the limitations.
"""
    )
