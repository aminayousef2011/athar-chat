"""Logic behind the local demo (``app/streamlit_app.py``): the frozen systems side by side, and the
locked test results read from the committed result files.

Nothing here changes a frozen system. ``load_systems`` goes through ``load_recognizer``, which
re-verifies ``models/freeze.json`` (model hashes and every frozen source) before loading.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from athar.gate import confidences
from athar.intent.train import FreezeError, load_recognizer, read_freeze
from athar.normalize import norm
from athar.pipeline import respond_with_intent
from athar.schema import Response
from athar.v2 import core as core_v11
from athar.v2 import gcv
from athar.v2.rules import KeywordIntentV11

CORE = ("S1", "S2", "S3")
SYSTEM_LABELS = {"S1": "S1 keyword rules", "S2": "S2 word TF-IDF + LR", "S3": "S3 + character n-grams",
                 "S1-v1.1": "S1 v1.1 (rules + core v1.1)", "oracle": "oracle (gold intent)"}


@dataclass(frozen=True)
class Example:
    text: str
    shows: str


#: Requests written for the demo (not test items), each illustrating a finding of docs/report.md.
EXAMPLES: tuple[Example, ...] = (
    Example("فكرني بكرة الساعة 9 الصبح ابعت التقرير لمديري",
            "Complete reminder in the drafts' style («فكرني»): every system acts."),
    Example("حطلي تذكير يوم الخميس الساعة 7 الصبح عشان اراجع خطة السبرينت",
            "Noun-form reminder («حطلي تذكير»), frequent in the author-written test and rare in "
            "training (report section 3.2)."),
    Example("عايز تذكير بكرة الساعة 5 العصر علشان اراجع الميزانية",
            "The shared extractor's framing gap: «عايز» is not in the framing lexicon, so «عايز علشان» "
            "stays in the reminder text (report section 7.3)."),
    Example("فكرني اكلم العميل", "Incomplete request: no time, so the system asks."),
    Example("صحيني الساعة 7", "Alarm with an hour but no AM/PM."),
    Example("remind me بكرة الساعة 3 العصر اعمل review للـ PR", "Arabic-English code-switching."),
    Example("فكرني كل يوم الساعة 9 اشرب مية", "Recurring reminder: out of scope by the task contract."),
    Example("امسح التذكير بتاع بكرة", "Deletion: out of scope."),
    Example("فكرني بحاجة بكرة الساعة 8", "Vague content and no AM/PM: two slots are asked for in one question."),
    Example("الغيلي المنبه بتاع بكرة",
            "Deletion with a suffixed verb: S1's lexicon lacks «الغيلي», S1 v1.1 adds it (report sections "
            "7.5 and 8.1)."),
    Example("فكرني اجدد الرخصة بعد اسبوع",
            "A day-scale offset. The systems ask only for the time; the contract does not resolve «بعد "
            "اسبوع», so the new writers' labels treat the day as missing too (report section 9)."),
)


@dataclass(frozen=True)
class Frozen:
    """A frozen intent recognizer and its confidence-gate threshold (chosen on dev)."""
    name: str
    recognizer: Any
    tau: float | None


def load_systems(models_dir: str | Path = "models", names: Sequence[str] = CORE) -> list[Frozen]:
    """The frozen systems, freeze-verified (raises ``FreezeError`` if anything changed)."""
    freeze = read_freeze(models_dir)
    out = []
    for n in names:
        gate = freeze["systems"].get(n, {}).get("gate") or {}
        out.append(Frozen(n, load_recognizer(n, models_dir), gate.get("threshold")))
    return out


def load_v11(models_v2_dir: str | Path = "models_v2") -> Frozen:
    """S1 v1.1 (stateless rules + core v1.1), after the v2 freeze verifies (raises ``FreezeError``)."""
    problems = gcv.verify(models_v2_dir)
    if problems:
        raise FreezeError("; ".join(problems))
    return Frozen("S1-v1.1", KeywordIntentV11(), None)


@dataclass(frozen=True)
class Answer:
    system: str
    intent: str
    confidence: float | None
    response: Response


def answer_all(text: str, ref: dt.datetime, systems: Sequence[Frozen], tz_name: str = "Africa/Cairo") -> list[Answer]:
    """Each system's answer to one request: intent, its confidence, and the shared pipeline's
    response. Ungated, as in the pre-registered metrics; ``gate_note`` says what the gate would do."""
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=ZoneInfo(tz_name))
    normalized = [norm(text)]
    out = []
    for s in systems:
        intent = s.recognizer.predict(normalized)[0]
        conf = confidences(s.recognizer, normalized, [intent])[0]
        respond = core_v11.respond_with_intent if s.name.endswith("-v1.1") else respond_with_intent
        out.append(Answer(s.name, intent, conf, respond(intent, text, ref, tz_name, s.name)))
    return out


def gate_note(answer: Answer, tau: float | None) -> str:
    """What the frozen confidence gate does to this answer (it only ever turns an execution into
    a confirmation question)."""
    if answer.response.route != "ACT" or answer.confidence is None or not tau:
        return "no effect"
    return "would ask to confirm" if answer.confidence < tau else "passes"


def tool_call_view(r: Response) -> dict[str, Any] | None:
    if r.tool_call is None:
        return None
    call = {"name": r.tool_call.name, "datetime": r.tool_call.datetime.isoformat()}
    if r.tool_call.reminder_text is not None:
        call["reminder_text"] = r.tool_call.reminder_text
    return call


# --------------------------------------------------------------------------- locked test results
def test_summary(results_dir: str | Path = "results") -> list[dict[str, Any]]:
    """The locked core run (S1-S3, oracle): STS with its CI and the main rates."""
    d = json.loads((Path(results_dir) / "test_results.json").read_text(encoding="utf-8"))
    rows = []
    for name, rep in d["systems"].items():
        m, ci = rep["metrics"], rep["ci"]["safe_task_success"]
        rows.append({"system": SYSTEM_LABELS.get(name, name), "STS": round(m["safe_task_success"], 3),
                     "95% CI": f"[{ci['low']:.3f}, {ci['high']:.3f}]",
                     "intent accuracy": round(m["intent_accuracy"], 3),
                     "unsafe execution": round(m["unsafe_execution_rate"], 3)})
    return rows


def test_comparisons(results_dir: str | Path = "results") -> list[dict[str, Any]]:
    """H1-H3 as tested in the locked run."""
    d = json.loads((Path(results_dir) / "test_results.json").read_text(encoding="utf-8"))
    return [{"comparison": k, "difference": round(v["difference"], 3),
             "95% CI": f"[{v['ci']['low']:.3f}, {v['ci']['high']:.3f}]",
             "Holm p": round(v["p_holm"], 4), "significant": v["significant"]}
            for k, v in d["comparisons"].items()]


def failures(results_dir: str | Path = "results") -> list[dict[str, str]]:
    """Every failed test item of the error analysis (``scripts/error_analysis.py``)."""
    path = Path(results_dir) / "error_analysis" / "failures.csv"
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- natural-phrasing set
def natural_hypotheses(results_dir: str | Path = "results") -> list[dict[str, Any]]:
    """N1-N3 as tested once on the natural-phrasing set (``athar-v2 evaluate``)."""
    d = json.loads((Path(results_dir) / "natural" / "evaluation.json").read_text(encoding="utf-8"))
    wcl = d["sensitivity"].get("writer_clustered", {})
    return [{"hypothesis": h, "difference": round(t["difference"], 3),
             "95% CI": f"[{t['ci']['low']:.3f}, {t['ci']['high']:.3f}]", "Holm p": round(t["p_holm"], 4),
             "writer-clustered p": round(wcl[h]["p_value"], 4) if h in wcl else None,
             "supported": bool(t["significant"])}
            for h, t in d["tests"].items()]


def natural_facts(results_dir: str | Path = "results") -> dict[str, Any]:
    d = json.loads((Path(results_dir) / "natural" / "evaluation.json").read_text(encoding="utf-8"))
    return {"items": d["n_items"], "confirmatory": d["confirmatory_items"], "writers": len(d["writers"]),
            "manifest": d["natural_manifest"][:12]}


def natural_tables(results_dir: str | Path = "results") -> dict[str, str]:
    """The sections of ``results/natural/tables.md`` (generated by ``scripts/natural_tables.py``),
    keyed by heading, so the app shows exactly the report's numbers."""
    text = (Path(results_dir) / "natural" / "tables.md").read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for block in text.split("\n## ")[1:]:
        head, _, body = block.partition("\n")
        out[head.strip()] = body.strip()
    return out


def cause_counts(rows: Sequence[Mapping[str, str]], system: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        if r["system"] == system:
            counts[r["cause"]] = counts.get(r["cause"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
