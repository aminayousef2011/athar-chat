"""ATHAR Chat local demo: the frozen systems side by side, and the locked test results.

    uv run --extra ui streamlit run app/streamlit_app.py

CPU only. The LLM systems (S4/S7/S8) need GPUs; their locked-test failures are browsable in the
second tab from the committed result files.
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
MODELS, RESULTS = ROOT / "models", ROOT / "results"
TZ = "Africa/Cairo"
ROUTE_COLOR = {"ACT": "#1a7f37", "CLARIFY": "#9a6700", "ABSTAIN": "#cf222e"}

st.set_page_config(page_title="ATHAR Chat demo", layout="wide")
# Requests are Arabic: type them right to left (bidi keeps embedded English in order).
st.markdown("<style>.stTextArea textarea {direction: rtl; text-align: right; font-size: 1.1rem;}</style>",
            unsafe_allow_html=True)


@st.cache_resource
def systems() -> list[demo.Frozen]:
    return demo.load_systems(MODELS)


def rtl(text: str) -> None:
    st.markdown(f'<div dir="rtl" style="text-align:right;font-size:1.1rem">{html.escape(text)}</div>',
                unsafe_allow_html=True)


def badge(route: str) -> str:
    return (f'<span style="background:{ROUTE_COLOR.get(route, "#57606a")};color:white;'
            f'padding:2px 8px;border-radius:6px;font-weight:600">{route}</span>')


st.title("ATHAR Chat: safe tool routing for Egyptian Arabic")
st.caption("Reminders and alarms are simulated: nothing is created. Course project; results in docs/report.md.")

try:
    loaded = systems()
    freeze_id = read_freeze(MODELS)["freeze_sha256"][:12]
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
    st.caption(f"Frozen models verified: `models/freeze.json` {freeze_id}…")
    st.caption("Confidence gate τ (chosen on dev): " +
               ", ".join(f"{s.name} {s.tau if s.tau is not None else 'n/a'}" for s in loaded))

try_tab, test_tab, about_tab = st.tabs(["Try a request", "Locked test results", "About"])

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
    cause = st.selectbox("Cause", ["all"] + [f"{c} ({n})" for c, n in counts.items()])
    shown = [r for r in rows if r["system"] == system and (cause == "all" or cause.startswith(r["cause"] + " ("))]
    cols = ["id", "text", "cause", "gold_route", "pred_route", "gold_intent", "pred_intent",
            "gold_canonical", "pred_canonical", "gold_targets", "pred_targets"]
    st.dataframe(pd.DataFrame(shown)[cols], hide_index=True)
    st.caption("Rebuilt exactly from the locked runs by scripts/error_analysis.py (report section 7.3).")

with about_tab:
    st.markdown(
        """
**What runs here.** S1 (keyword rules), S2 (word TF-IDF + logistic regression) and S3 (S2 + character
n-grams), loaded from `models/` after the freeze is verified, followed by the shared normalizer, time and
argument extractor, execution policy and Arabic reply templates. All three systems share everything except
intent recognition. Answers are shown ungated, as in the pre-registered metrics.

**What does not run here.** The LLM systems (Jais 2, Jais 3, Nile-Chat, Qwen3; QLoRA Jais 2; Jais 3 tool
calling) need GPUs. Their locked-test failures are in the failure browser; full results are in
`docs/report.md`.

**Routes.** ACT executes a simulated `create_reminder` or `create_alarm`. CLARIFY asks for the missing
slots. ABSTAIN declines out-of-scope requests (deletion, modification, recurrence, another person,
several actions, other tools).
"""
    )
