"""The GPU systems of protocol v3, section 6, as fixed by amendment A5 (descriptive).

- ``S8-fair``: Jais-3-7B end to end through native tool calling, with **greedy** decoding. The
  prompt is the frozen S8 system prompt (``athar.agent.SYSTEM_PROMPT``), plus the annotation
  guide's section 5 reminder-text rule verbatim, plus 6 worked examples from author-written
  train/dev items (:func:`select_tool_exemplars`). The trace has the same format as S8's, so
  ``athar.v2.hybrid`` replays it as S8 and builds S9 from it.
- ``S4-jais3-7b-inline-norm`` / ``-raw``: 6-shot ``prior`` label scoring with the exemplars inline,
  on normalized text (as every system's input) and on the text as written (request, exemplars
  and prior texts alike). Label probabilities are stored.

Everything that shapes the prompts is fixed here and hashed into ``models_v3/freeze_gpu.json``
before the run; the rendered prompts are hashed too.

    python -m athar.v3.gpu run                        # on a GPU (Kaggle T4 x2)
    python -m athar.v3.gpu render                     # print the S8-fair prompt for one item (CPU)
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from athar import agent
from athar.data.split import load_split
from athar.intent import llm
from athar.intent.train import source_hash
from athar.normalize import norm
from athar.schema import INTENTS, GoldItem, Response
from athar.v3 import amend
from athar.v3 import data as D
from athar.v3 import systems as S

MODEL_ID = "inception42/Jais-3-7B"
REVISION = "94d69f49b4af702d42fcee122cc06e493a6fe7e1"   # as pinned in scripts/make_notebook.py
PRECISION = "auto:fp16"                                  # as S4/S8 Jais-3-7B before
S8_TEMPLATE = {"reasoning_effort": "high", "tool_call_format": "xml"}
S8_MAX_NEW_TOKENS = 8192
S8_BATCH = 4
SHOTS_PER_LABEL, N_PRIOR = 2, 60
SOURCES = ("src/athar/v3/gpu.py", "src/athar/v3/amend.py", "src/athar/agent.py", "src/athar/intent/llm.py",
           *S.V3_SOURCES)
TRACE_DIR = S.REPO / "results" / "v3" / "traces"

#: annotation guide, section 5 (docs/annotation_guide.md), verbatim; tests check it against the guide
GUIDE_REMINDER_TEXT = """The reminder text is the request minus:
1. the request framing (`فكرني`, `ممكن تفكرني`, `اعملي تذكير`, `remind me`, ...);
2. politeness and fillers (`لو سمحت`, `please`, `معلش`);
3. every date and time expression (`بكرة`, `الساعة 7 الصبح`, `بعد ساعة`);
4. function words left at either edge (`إن/إني`, `عشان`, `في`, `to`, ...);
5. the `ب` of a leading `بال...` (`بالاجتماع` → `الاجتماع`).

Keep everything inside, in the user's own spelling, including English words:

| Request | `reminder_text` |
|---|---|
| `فكرني بكرة الساعة ٧ الصبح أشتري عيش` | `أشتري عيش` |
| `فكرني إني أكلم ماما بكرة الساعة 5 العصر` | `أكلم ماما` |
| `ممكن تفكرني أكلم أحمد بكرة الساعة 5 العصر عشان عيد ميلاده` | `أكلم أحمد عشان عيد ميلاده` |
| `فكرني بكرة الساعة 3 العصر عندي meeting مع الـ team` | `عندي meeting مع الـ team` |"""

CATEGORIES = (("ACT", "create_reminder"), ("ACT", "create_alarm"), ("CLARIFY", ("time",)),
              ("CLARIFY", ("am_pm",)), ("ABSTAIN", "recurring"), ("ABSTAIN", "deletion"))


# --------------------------------------------------------------------------- S8-fair prompt
def _category(it: GoldItem) -> tuple[str, Any]:
    if it.route == "ACT":
        return ("ACT", it.intent)
    if it.route == "CLARIFY":
        return ("CLARIFY", tuple(it.clarify_targets))
    return ("ABSTAIN", it.unsupported_reason)


def select_tool_exemplars(pool: Sequence[GoldItem], seed: int = 0) -> list[GoldItem]:
    """One author-written item per category of :data:`CATEGORIES`, distinct families, seeded."""
    rng = random.Random(seed)
    used: set[str] = set()
    picked = []
    for cat in CATEGORIES:
        cands = sorted((it for it in pool if it.source == "author" and _category(it) == cat), key=lambda x: x.id)
        rng.shuffle(cands)
        it = next((c for c in cands if c.family_id not in used), None)
        if it is None:
            raise ValueError(f"no author-written exemplar for {cat}")
        used.add(it.family_id)
        picked.append(it)
    return picked


def _local(t: dt.datetime, tz: str) -> dt.datetime:
    return t.astimezone(ZoneInfo(tz))


def gold_call(it: GoldItem) -> str:
    """The exact call the contract expects for ``it``, written as a function call."""
    if it.route == "ACT":
        assert it.args.datetime is not None, f"{it.id}: ACT without a gold datetime"
        when = _local(it.args.datetime, it.timezone).strftime("%Y-%m-%dT%H:%M")
        if it.intent == "create_reminder":
            return f'create_reminder(datetime="{when}", reminder_text="{it.args.reminder_text}")'
        return f'create_alarm(datetime="{when}")'
    if it.route == "CLARIFY":
        return f"ask_clarification(tool=\"{it.intent}\", missing={json.dumps(list(it.clarify_targets))})"
    return f'decline(reason="{it.unsupported_reason}")'


def exemplar_block(exemplars: Sequence[GoldItem]) -> str:
    lines = ["Worked examples (each with the moment it was sent; the correct call follows the arrow):"]
    for it in exemplars:
        t = _local(it.ref_datetime, it.timezone)
        lines.append(f"- Sent {agent.WEEKDAYS_EN[t.weekday()]} {t:%Y-%m-%d} at {t:%H:%M}: «{it.text}» -> {gold_call(it)}")
    return "\n".join(lines)


def fair_system_prompt(ref: dt.datetime, tz_name: str, exemplars: Sequence[GoldItem]) -> str:
    return (agent.system_prompt(ref, tz_name)
            + "\n\nHow to write reminder_text (annotation guide, section 5):\n" + GUIDE_REMINDER_TEXT
            + "\n\n" + exemplar_block(exemplars))


def fair_messages(text: str, ref: dt.datetime, tz_name: str, exemplars: Sequence[GoldItem]) -> list[dict[str, str]]:
    return [{"role": "system", "content": fair_system_prompt(ref, tz_name, exemplars)},
            {"role": "user", "content": text}]


def respond(items: Sequence[D.V3Item], backend: agent.Backend, exemplars: Sequence[GoldItem],
            name: str = "S8-fair") -> list[agent.TraceRow]:
    """S8's mapping (``agent.to_response``) on the fair prompt; one trace row per item."""
    completions = backend.complete([fair_messages(it.text, it.ref_datetime, it.timezone, exemplars) for it in items])
    if len(completions) != len(items):
        raise ValueError(f"{len(completions)} completions for {len(items)} items")
    rows = []
    for it, c in zip(items, completions, strict=True):
        calls = c.tool_calls if c.tool_calls is not None else agent.parse_tool_calls(c.text)
        response: Response | None = None
        reason = error = None
        if c.finish_reason == "length" and not calls:
            error = "truncated: no answer within the token limit"
        else:
            try:
                response, reason = agent.to_response(calls, name, it.timezone)
            except agent.ToolCallError as e:
                error = str(e)
        rows.append(agent.TraceRow(id=it.id, route=response.route if response else None,
                                   intent=response.intent if response else None, reason=reason, error=error,
                                   n_tokens=c.n_tokens, finish_reason=c.finish_reason, answer=c.text,
                                   reasoning_chars=len(c.reasoning), calls=list(calls)))
    return rows


# --------------------------------------------------------------------------- S4 inline, raw vs normalized
def shot_items(train: Sequence[GoldItem], per_label: int = SHOTS_PER_LABEL, seed: int = 0) -> list[GoldItem]:
    """The items behind ``llm.select_shots`` (same rule, same order), so the raw variant can show
    the same exemplars as written."""
    rng = random.Random(seed)
    by_label: dict[str, list[GoldItem]] = {lab: [] for lab in INTENTS}
    for it in sorted(train, key=lambda x: x.id):
        by_label[it.intent].append(it)
    used: set[str] = set()
    picked: dict[str, list[GoldItem]] = {}
    for lab in INTENTS:
        pool = by_label[lab][:]
        rng.shuffle(pool)
        picked[lab] = []
        for it in pool:
            if len(picked[lab]) < per_label and it.family_id not in used:
                picked[lab].append(it)
                used.add(it.family_id)
    return [picked[lab][k] for k in range(per_label) for lab in INTENTS if k < len(picked[lab])]


def s4_inputs(variant: str, seed: int) -> tuple[list[tuple[str, str]], list[str]]:
    """(shots, prior texts) for ``variant`` ``norm`` or ``raw``; the same items either way."""
    train = list(load_split(S.MAIN, "train"))
    def f(text: str) -> str:
        return norm(text) if variant == "norm" else text

    shots: list[tuple[str, str]] = [(f(it.text), str(it.intent)) for it in shot_items(train, seed=seed)]
    prior: list[str] = [f(it.text) for it in random.Random(seed).sample(train, N_PRIOR)]
    return shots, prior


# --------------------------------------------------------------------------- run + freeze
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_text_once(path: Path, text: str) -> tuple[str, str]:
    if path.exists():
        raise S.FreezeError(f"{path} exists: GPU outputs are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return str(path.relative_to(S.REPO)).replace("\\", "/"), hashlib.sha256(path.read_bytes()).hexdigest()


def run(out: Path = S.MODELS_V3, systems: Sequence[str] = ("S4-norm", "S4-raw", "S8-fair")) -> dict[str, Any]:
    """Run the GPU systems on the unlabelled v3 texts and write ``models_v3/freeze_gpu.json``."""
    from athar.intent.neural import use_system_certificates

    use_system_certificates()
    if (out / "freeze_gpu.json").exists():
        raise S.FreezeError(f"{out}/freeze_gpu.json exists")
    eval_seed = int(yaml.safe_load((S.REPO / "configs" / "eval.yaml").read_text(encoding="utf-8"))["seed"])
    items = D.load()
    pool = list(load_split(S.MAIN, "train")) + list(load_split(S.MAIN, "dev"))
    exemplars = select_tool_exemplars(pool, seed=0)
    model, tok = llm.load_llm(MODEL_ID, PRECISION, revision=REVISION, trust_remote_code=True)
    prof = llm.chat_profile(MODEL_ID)
    record: dict[str, Any] = {"model": MODEL_ID, "revision": REVISION, "precision": llm.resolve_precision(PRECISION),
                              "outputs": {}, "prompts_sha256": {}}
    for variant in ("norm", "raw"):
        name = f"S4-jais3-7b-inline-{variant}"
        if f"S4-{variant}" not in systems:
            continue
        shots, prior = s4_inputs(variant, eval_seed)
        rec = llm.LLMIntent(model, tok, name, shots=shots, method="prior", prior_texts=prior,
                            **{**prof.intent_kwargs(), "shot_style": "inline"})
        texts = [norm(it.text) if variant == "norm" else it.text for it in items]
        record["prompts_sha256"][name] = _sha("\n\x00\n".join(rec.prompt(t) for t in texts))
        proba = rec.predict_proba(texts, list(INTENTS))
        labels = rec.predict(texts)
        body = {it.id: {"intent": lab, "proba": [round(p, 6) for p in pr]}
                for it, lab, pr in zip(items, labels, proba, strict=True)}
        rel, h = _write_text_once(S.PRED_DIR / f"{name}.json",
                                  json.dumps(dict(sorted(body.items())), ensure_ascii=False, indent=1) + "\n")
        record["outputs"][name] = {"file": rel, "sha256": h, "labels": list(INTENTS)}
    if "S8-fair" in systems:
        backend = agent.HFToolBackend(model, tok, template_kwargs=S8_TEMPLATE, max_new_tokens=S8_MAX_NEW_TOKENS,
                                      sampling=None, seed=0, batch_size=S8_BATCH)
        msgs = [fair_messages(it.text, it.ref_datetime, it.timezone, exemplars) for it in items]
        record["prompts_sha256"]["S8-fair"] = _sha("\n\x00\n".join(backend.render(m) for m in msgs))
        rows = respond(items, backend, exemplars)
        rel, h = _write_text_once(TRACE_DIR / "S8-fair.jsonl",
                                  "".join(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n" for r in rows))
        record["outputs"]["S8-fair"] = {"file": rel, "sha256": h, "decoding": "greedy", "template": S8_TEMPLATE,
                                        "max_new_tokens": S8_MAX_NEW_TOKENS,
                                        "exemplars": [it.id for it in exemplars]}
    common = S._common(items)
    common["sources"] = {rel_: source_hash(rel_) for rel_ in SOURCES}
    return S._write_once(out / "freeze_gpu.json", {**common, "amendment": "A5", **amend.snapshot(),
                                                   "gate": "G3 (GPU systems, descriptive)", **record})


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = list(argv if argv is not None else sys.argv[1:])
    if args[:1] == ["render"]:
        pool = list(load_split(S.MAIN, "train")) + list(load_split(S.MAIN, "dev"))
        ex = select_tool_exemplars(pool)
        ref = dt.datetime(2026, 10, 18, 9, 0, tzinfo=ZoneInfo("Africa/Cairo"))
        print(fair_system_prompt(ref, "Africa/Cairo", ex))
        return 0
    if args[:1] == ["run"]:
        fz = run()
        print(f"models_v3/freeze_gpu.json  freeze_sha256={fz['freeze_sha256']}")
        return 0
    print("usage: python -m athar.v3.gpu render | run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
