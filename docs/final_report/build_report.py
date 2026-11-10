"""Build the final course report (A4 .docx, and .pdf through Microsoft Word).

Every table is filled from the committed result files, and the commit hash printed in the
Deliverables section is read from git, so the report always names the commit it was built from.

    uv run --with python-docx python docs/final_report/build_report.py            # .docx only
    uv run --with python-docx python docs/final_report/build_report.py --pdf      # + .pdf (Windows, Word)

The build refuses to run on a working tree with uncommitted changes (``--allow-dirty`` overrides),
because the printed commit would not match the files the report describes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/aminayousef2011/athar-chat"
FONT = "Times New Roman"
MONO = "Consolas"


# ---------------------------------------------------------------- data -------------------------

def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


TEST = load("results/test_results.json")
NAT = load("results/natural/evaluation.json")
ENC = load("results/posthoc/encoders_run1.json")
HUM = load("results/posthoc/human_training.json")
SENS = load("results/natural/sensitivity.json")
# S7's natural-set score is the per-item mean over three seeds; the generated table holds it at full
# precision (the per-seed values stored with the LLM results are rounded).
S7_NAT = float(re.search(r"\| S7 Jais 2 \+ QLoRA \(3 seeds\) \| ([0-9.]+)",
                         (ROOT / "results/natural/tables.md").read_text(encoding="utf-8")).group(1))


def f3(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def signed(x: float) -> str:
    return f"{x:+.3f}".replace("-", "−")


def ci(c: dict) -> str:
    return f"[{c['low']:.3f}, {c['high']:.3f}]".replace("-", "−")


def test_metrics(name: str) -> dict:
    return TEST["systems"][name]["metrics"]


def nat_metrics(name: str) -> dict:
    return NAT["systems"][name]["metrics"]


def nat_seed_mean(prefix: str, key: str) -> float:
    return st.mean(nat_metrics(f"{prefix}-s{i}")[key] for i in range(3))


def nat_seed_route(prefix: str, route: str) -> float:
    return st.mean(nat_metrics(f"{prefix}-s{i}")["success_by_gold_route"][route] for i in range(3))


def s4_test(model: str) -> float:
    paths = {"jais2-8b": "7b-others", "nile-4b": "7b-others", "qwen3-8b": "7b-others",
             "jais3-7b": "7a-jais3", "jais3-3.7b": "7a-jais3"}
    d = load(f"results/s4/{paths[model]}/s4_results/{model}.json")
    return d["systems"][f"S4-{model}-6shot-prior"]["metrics"]["safe_task_success"]


def s7_test() -> list[float]:
    return [load(f"results/s4/7cd-s7-s8/s4_results/s7_jais2_qlora_s{i}_test.json")["systems"]
            [f"S7-jais2-qlora-s{i}-prior"]["metrics"]["safe_task_success"] for i in range(3)]


def s8_test() -> list[float]:
    d = load("results/s4/7cd-s7-s8/s4_results/s8_jais3-7b.json")["systems"]
    return [d[f"S8-jais3-7b-s{i}"]["metrics"]["safe_task_success"] for i in range(3)]


def s4_nat(model: str) -> float:
    return NAT["llm_with_core_v1.1"][f"S4-{model}-6shot-prior"]["stored"]


def s7_nat() -> list[float]:
    return [NAT["llm_with_core_v1.1"][f"S7-jais2-qlora-s{i}-prior"]["stored"] for i in range(3)]


def nat_confirmatory_families() -> int:
    conf = {i["id"] for i in NAT["systems"]["S1"]["items"]}
    return len({i["family_id"] for i in NAT["systems"]["S1"]["items"] if i["id"] in conf})


# ---------------------------------------------------------------- docx helpers -----------------

def _is_arabic(ch: str) -> bool:
    return "؀" <= ch <= "ۿ" or "ݐ" <= ch <= "ݿ" or "ﭐ" <= ch <= "﻿"


def _segments(text: str) -> list[tuple[str, bool]]:
    """Split text into Arabic and non-Arabic runs. Digits, spaces and punctuation that follow
    Arabic stay in the Arabic run (so «الساعة 7 الصبح» keeps its order); a Latin letter or a
    bracket/guillemet ends it."""
    segs: list[tuple[str, bool]] = []
    for ch in text:
        ar = _is_arabic(ch)
        if not ar and segs and segs[-1][1] and not ch.isalpha() and ch not in "«»()[]":
            ar = True
        if segs and segs[-1][1] == ar:
            segs[-1] = (segs[-1][0] + ch, ar)
        else:
            segs.append((ch, ar))
    fixed: list[tuple[str, bool]] = []
    for s, ar in segs:  # trailing spaces and ASCII punctuation belong to the English text
        core = s.rstrip(" ,.;:")
        if ar and core != s:
            fixed.append((core, True))
            fixed.append((s[len(core):], False))
        else:
            fixed.append((s, ar))
    return [(s, a) for s, a in fixed if s]


def _set_fonts(run, name: str, size: float) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    rpr = run._r.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)
    szcs = OxmlElement("w:szCs")
    szcs.set(qn("w:val"), str(int(size * 2)))
    rpr.append(szcs)


def add_runs(p, text: str, size: float = 12, font: str = FONT, bold: bool = False,
             italic: bool = False, rtl_all: bool = False) -> None:
    """Add text with **bold**, *italic* and `code` markup; Arabic runs are marked right-to-left.

    Arabic quoted inside an English paragraph must not contain Latin words: the bidi algorithm
    would split it. Mixed messages go into right-to-left paragraphs (table cells starting "@")."""
    # kinds: "t" text, "c" code
    parts: list[tuple[str, bool, bool, str]] = []
    i, b, it, buf = 0, bold, italic, ""
    while i < len(text):
        ch = text[i]
        if text.startswith("**", i):
            parts.append((buf, b, it, "t"))
            buf = ""
            b = not b
            i += 2
            continue
        if ch == "*":
            parts.append((buf, b, it, "t"))
            buf = ""
            it = not it
            i += 1
            continue
        if ch == "`":
            j = text.index("`", i + 1)
            parts.append((buf, b, it, "t"))
            buf = ""
            parts.append((text[i + 1:j], b, False, "c"))
            i = j + 1
            continue
        buf += ch
        i += 1
    parts.append((buf, b, it, "t"))

    def run(seg: str, cb: bool, ci_: bool, rtl: bool, fnt: str, sz: float) -> None:
        r = p.add_run(seg)
        _set_fonts(r, fnt, sz)
        r.bold, r.italic = cb, ci_
        rpr = r._r.get_or_add_rPr()
        if cb:
            rpr.append(OxmlElement("w:bCs"))
        if ci_:
            rpr.append(OxmlElement("w:iCs"))
        if rtl:
            rpr.append(OxmlElement("w:rtl"))

    for chunk, cb, ci_, kind in parts:
        if not chunk:
            continue
        if kind == "c":
            run(chunk, cb, False, False, MONO, max(10.0, size - 1))
        elif rtl_all:  # right-to-left paragraph: English words stay left-to-right runs
            for seg, ar in _segments(chunk):
                if ar or not any(c.isalpha() for c in seg):
                    run(seg, cb, ci_, True, font, size)
                    continue
                core = seg.strip()
                lead, trail = seg[:len(seg) - len(seg.lstrip())], seg[len(seg.rstrip()):]
                for piece, rtl in ((lead, True), (core, False), (trail, True)):
                    if piece:
                        run(piece, cb, ci_, rtl, font, size)
        else:
            for seg, ar in _segments(chunk):
                run(seg, cb, ci_, ar, font, size)


def fmt(p, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before: float = 0, after: float = 6,
        line: float = 1.15, keep_next: bool = False) -> None:
    pf = p.paragraph_format
    pf.alignment = align
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line
    pf.first_line_indent = Cm(0)
    pf.keep_with_next = keep_next


class Report:
    def __init__(self) -> None:
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Cm(2.5))
        for name, size in (("Normal", 12), ("Heading 1", 14), ("Heading 2", 12)):
            style = self.doc.styles[name]
            style.font.name, style.font.size = FONT, Pt(size)
            style.font.color.rgb = RGBColor(0, 0, 0)
            if name != "Normal":
                style.font.bold, style.font.italic = True, False
            rpr = style.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                fonts.set(qn(attr), FONT)
            for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                if fonts.get(qn(attr)) is not None:
                    del fonts.attrib[qn(attr)]
        self._footer(sec)
        self.table_no = 0

    def _footer(self, sec) -> None:
        p = sec.footer.paragraphs[0]
        fmt(p, align=WD_ALIGN_PARAGRAPH.RIGHT, after=0, line=1.0)
        r = p.add_run()
        _set_fonts(r, FONT, 12)
        for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            r._r.append(el)

    # text blocks
    def title(self, text: str, name: str) -> None:
        p = self.doc.add_paragraph()
        fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
        add_runs(p, text, size=16, bold=True)
        p = self.doc.add_paragraph()
        fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
        add_runs(p, name, size=12)

    def h1(self, text: str) -> None:
        p = self.doc.add_paragraph(style="Heading 1")
        fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT, before=12, after=6, line=1.15, keep_next=True)
        add_runs(p, text, size=14, bold=True)
        for r in p.runs:
            r.font.color.rgb = RGBColor(0, 0, 0)

    def h2(self, text: str) -> None:
        p = self.doc.add_paragraph(style="Heading 2")
        fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT, before=12, after=6, line=1.15, keep_next=True)
        add_runs(p, text, size=12, bold=True)
        for r in p.runs:
            r.font.color.rgb = RGBColor(0, 0, 0)

    def para(self, text: str) -> None:
        p = self.doc.add_paragraph()
        fmt(p)
        add_runs(p, text)

    def bullets(self, items: list[str]) -> None:
        for it in items:
            p = self.doc.add_paragraph()
            fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT, after=6)
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.first_line_indent = Cm(-0.35)
            add_runs(p, "• " + it)

    def code(self, lines: list[str]) -> None:
        for i, line in enumerate(lines):
            p = self.doc.add_paragraph()
            fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT, after=6 if i == len(lines) - 1 else 0, line=1.0)
            add_runs(p, line, size=10, font=MONO)

    def link_para(self, before: str, url: str, after: str = "") -> None:
        p = self.doc.add_paragraph()
        fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT, after=6)
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.first_line_indent = Cm(-0.35)
        add_runs(p, "• " + before)
        hl = OxmlElement("w:hyperlink")
        hl.set(qn("r:id"), p.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/"
                                                  "2006/relationships/hyperlink", is_external=True))
        r = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        fonts = OxmlElement("w:rFonts")
        for attr in ("w:ascii", "w:hAnsi", "w:cs"):
            fonts.set(qn(attr), FONT)
        color = OxmlElement("w:color")
        color.set(qn("w:val"), "0563C1")
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), "24")
        for el in (fonts, color, u, sz):
            rpr.append(el)
        t = OxmlElement("w:t")
        t.text = url
        r.append(rpr)
        r.append(t)
        hl.append(r)
        p._p.append(hl)
        if after:
            add_runs(p, after)

    def page_break(self) -> None:
        p = self.doc.add_paragraph()
        fmt(p, after=0)
        p.add_run().add_break(WD_BREAK.PAGE)

    # tables
    def table(self, caption: str, header: list[str], rows: list[list[str]],
              widths: list[float] | None = None, bold_rows: set[int] | None = None) -> None:
        self.table_no += 1
        cap = self.doc.add_paragraph()
        fmt(cap, align=WD_ALIGN_PARAGRAPH.LEFT, before=3, after=3, line=1.0, keep_next=True)
        add_runs(cap, f"**Table {self.table_no}.** {caption}", size=10)
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = widths is None
        for ri, row in enumerate([header, *rows]):
            for cj, val in enumerate(row):
                cell = t.cell(ri, cj)
                if widths:
                    cell.width = Cm(widths[cj])
                bold = ri == 0 or (bold_rows is not None and ri - 1 in bold_rows)
                for k, line in enumerate(val.split("\n")):
                    p = cell.paragraphs[0] if k == 0 else cell.add_paragraph()
                    rtl = line.startswith("@")
                    fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT if cj == 0 else WD_ALIGN_PARAGRAPH.CENTER,
                        after=0, line=1.0)
                    if rtl:  # a right-to-left paragraph keeps mixed Arabic/English in reading order
                        line = line[1:]
                        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        p._p.get_or_add_pPr().insert(0, OxmlElement("w:bidi"))
                    p.paragraph_format.keep_with_next = ri < len(rows)  # keep the table on one page
                    add_runs(p, line, size=10, bold=bold, rtl_all=rtl)
                tcpr = cell._tc.get_or_add_tcPr()
                mar = OxmlElement("w:tcMar")
                for side in ("top", "bottom"):
                    m = OxmlElement(f"w:{side}")
                    m.set(qn("w:w"), "15")
                    m.set(qn("w:type"), "dxa")
                    mar.append(m)
                for side in ("left", "right"):
                    m = OxmlElement(f"w:{side}")
                    m.set(qn("w:w"), "60")
                    m.set(qn("w:type"), "dxa")
                    mar.append(m)
                tcpr.append(mar)
                if ri == 0:
                    borders = OxmlElement("w:tcBorders")
                    b = OxmlElement("w:bottom")
                    b.set(qn("w:val"), "single")
                    b.set(qn("w:sz"), "6")
                    borders.append(b)
                    tcpr.append(borders)
            if ri == 0:
                trpr = t.rows[0]._tr.get_or_add_trPr()
                trpr.append(OxmlElement("w:tblHeader"))
        tblpr = t._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for side in ("top", "bottom"):
            b = OxmlElement(f"w:{side}")
            b.set(qn("w:val"), "single")
            b.set(qn("w:sz"), "8")
            borders.append(b)
        tblpr.append(borders)
        spacer = self.doc.add_paragraph()
        fmt(spacer, after=0, line=1.0)
        mark = OxmlElement("w:rPr")  # a small empty paragraph separates the table from the text
        for tag in ("w:sz", "w:szCs"):
            el = OxmlElement(tag)
            el.set(qn("w:val"), "10")
            mark.append(el)
        spacer._p.get_or_add_pPr().append(mark)

    def save(self, path: Path) -> None:
        self.doc.save(path)


# ---------------------------------------------------------------- content ----------------------

def build(commit: str) -> Report:
    R = Report()
    t1, t2, t3, o = (test_metrics(s) for s in ("S1", "S2", "S3", "oracle"))
    n1, n2, n3, no = (nat_metrics(s) for s in ("S1", "S2", "S3", "oracle"))
    cmp_ = TEST["comparisons"]
    tests = NAT["tests"]
    s8m, s9m = NAT["summary"]["S8 (seed mean)"], NAT["summary"]["S9 (seed mean)"]
    s6 = ENC["s6_seed_mean"]
    hum = HUM["systems"]

    R.title("ATHAR Chat: Safe Tool Routing for Egyptian Arabic", "Amina Yousef")

    R.h1("Abstract")
    R.para(
        "A reminder assistant must decide whether to execute a request, ask for missing information or "
        "decline it. We study this decision (ACT, CLARIFY, ABSTAIN) for single-turn Egyptian Arabic "
        "reminder and alarm requests, often code-switched with English. Sharing one rule-based "
        "extractor and deterministic policy, keyword rules (S1), TF-IDF logistic regression (S2) and "
        "its character n-gram extension (S3) are compared by Safe Task Success (STS). On a locked 120-item test the "
        f"rules outperformed both classifiers ({f3(t1['safe_task_success'])} vs "
        f"{f3(t2['safe_task_success'])} and {f3(t3['safe_task_success'])}). A pre-registered second "
        "evaluation on 238 messages from 17 other native speakers confirmed this "
        f"(+{tests['N1 S1 vs S2']['difference']:.3f} STS, Holm p = {tests['N1 S1 vs S2']['p_holm']:.4f}) "
        "and showed that the same deterministic policy makes end-to-end Jais-3-7B tool calling safer "
        f"(STS {s8m['sts']:.3f} → {s9m['sts']:.3f}). Exploratory analyses trace the classifiers' "
        "deficit to their mostly AI-drafted training data and show that the shared extractor bounds "
        f"every system ({f3(no['safe_task_success'])} with gold intents).")

    # 1 ------------------------------------------------------------------------------------------
    R.h1("1. Motivation and objectives")
    R.para(
        "A reminder assistant fails in two costly ways: it executes a request whose time or content is "
        "missing or ambiguous, or it asks again for what the user already gave. It must also decline "
        "what it cannot do, such as deleting a reminder. Egyptian Arabic makes "
        "this harder: messages are informal («اعمللي تذكير بعد بكرة الساعة 9 بالليل عشان اخد الدوا», "
        "\"set me a reminder the day after tomorrow at 9 pm to take my medicine\"), inconsistently "
        "spelled and mixed with English (*timesheet*, *sprint backlog*). Out-of-scope requests are hard "
        "for intent classifiers [6], and When2Call [1] showed for English that deciding *when* to call "
        "a tool is a problem of its own; no Egyptian Arabic dataset labels this decision with tool "
        "arguments.")
    R.para(
        "**Intended users.** Developers and students building assistants for under-resourced dialects "
        "with little compute, who need evidence on how far inexpensive methods go; and Egyptian "
        "professionals who write mixed Arabic and English (187 of the 238 scored independent messages "
        "concern project management).")
    R.para(
        "**Research question.** When all systems share the same argument extractor and execution policy, "
        "does lightweight supervised intent classification improve Safe Task Success over keyword rules? "
        "**Objectives:** (O1) a task contract and safety-aware metric; (O2) a reviewed Egyptian Arabic "
        "benchmark with leakage-free splits; (O3) rules vs. TF-IDF classifiers in one shared pipeline; "
        "(O4) a test on writers other than the author; (O5) the errors that limit the pipeline.")

    # 2 ------------------------------------------------------------------------------------------
    R.h1("2. Deliverables")
    R.link_para("**Repository:** ", REPO_URL, f" (branch `report-tier1`), commit `{commit}`.")
    R.bullets([
        "**Code:** package `athar` (normalizer, extractor, policy, systems S1-S9, evaluation), "
        "command-line tools, 366 unit tests, dependencies locked in `uv.lock` (Python 3.11).",
        "**Data:** `data/athar.jsonl` (458 items; splits `data/splits/manifest.json`), "
        "`data/natural/natural.jsonl` (247), data card, task contract and annotation guide (`docs/`), "
        "shared with the contributors' consent.",
        "**Models, outputs, demo:** frozen classifiers (`models/`, `models_v2/`), stored predictions "
        "(`results/`), pinned Hugging Face models (Appendix C), a Streamlit demo "
        "(`app/streamlit_app.py`) and the technical report `docs/report.md`.",
    ])

    # 3 ------------------------------------------------------------------------------------------
    R.h1("3. Methods and implementation")
    R.h2("3.1 Task contract and metric")
    R.para(
        "The simulated tools are `create_reminder(datetime, reminder_text)` and `create_alarm(datetime)`; "
        "each request has a sending time. The contract resolves explicit dates, today/tomorrow, "
        "weekdays with «الجاي» (next), clock times and period words such as «الصبح». **ACT** needs every "
        "argument resolved to one future instant. **CLARIFY** names the missing slots (date, time, "
        "AM/PM, past time, reminder text). **ABSTAIN** covers deletion, modification, recurrence, other "
        "people's items, several actions and other tools. **Safe Task Success** counts an item only if "
        "an ABSTAIN item is declined, a CLARIFY item asks for exactly the gold slots, or an ACT item "
        "calls the gold tool at the gold instant with the gold reminder text (exact after "
        "normalization and removal of request words and fillers). Invalid outputs fail.")

    R.h2("3.2 Data")
    nat_fams = nat_confirmatory_families()
    R.table(
        "Datasets. Splits are by scenario family; no family crosses splits.",
        ["Set", "Items", "Families", "ACT/CLAR/ABST", "Written by", "Use"],
        [["train", "298", "137", "120 / 127 / 51", "261 AI-drafted, 37 author", "fitting"],
         ["dev", "40", "17", "16 / 17 / 7", "39 AI-drafted, 1 author", "selection"],
         ["test (locked)", "120", "71", "50 / 50 / 20", "author (native speaker)", "evaluation 1"],
         ["natural (locked)", "238", str(nat_fams), "91 / 88 / 59", "17 other native speakers",
          "evaluation 2"]],
        widths=[2.7, 1.3, 1.6, 3.2, 4.4, 2.8])
    R.para(
        "The test was written by the author, a native speaker, from scenario cards; a second native "
        "speaker labelled 60 items "
        "independently (intent κ = 0.916, route κ = 1.000). The natural set was collected after the "
        "systems were frozen: contributors wrote, without AI tools, a complete and an incomplete message "
        "for each scenario. After screening out copies (474 messages, 34 writers), a seeded draw of two "
        "writers per scenario block gave 247 messages from 17 writers, 238 of them in families absent "
        "from train and dev. Two annotators labelled all of them (intent κ = 0.938, route κ = 0.913); "
        "the author adjudicated the 66 disagreements "
        "(`results/natural/adjudication.json`).")

    R.h2("3.3 Pipeline and systems")
    R.para(
        "The normalizer unifies alef, ya and taa marbuta variants, strips diacritics and elongation and "
        "maps Arabic-Indic digits. A pattern-based extractor parses the "
        "supported time expressions and takes as reminder text what remains after removing request "
        "framing, time expressions and fillers. The policy maps an unsupported intent to ABSTAIN, a "
        "supported one with all required slots to ACT, and otherwise to CLARIFY, so systems predict "
        "only the intent. **S1** uses hand-written trigger and out-of-scope lexicons. **S2** is word "
        "TF-IDF with class-balanced logistic regression [4]; **S3** adds character 2-4-grams for "
        "spelling variation. Vocabularies are fitted on train; C and n-gram ranges were chosen by dev "
        "intent macro-F1 (S2: C = 0.3; S3: C = 0.1). On a laptop CPU, fitting takes under a second, "
        "inference under 0.3 ms per item, peak memory 141 MB. The **oracle** feeds gold intents "
        "to the same pipeline. **Extensions:** prompted LLM intent recognizers (**S4**: Jais 2 8B "
        "[5, 11], Jais-3-7B, Jais-3-3.7B [12], Nile-Chat 4B [13], Qwen3 8B [14]); fine-tuned MARBERTv2 [2] "
        "(**S6**, five seeds, recipe after [10]); Jais 2 with QLoRA [3] (**S7**); Jais-3-7B performing "
        "the whole task by native tool calling (**S8**); and **S9**, which keeps S8's declines and "
        "arguments but lets the policy decide when to execute, falling back to S1 v1.1 where S8 made no valid "
        "call (23-25 of 238 items per seed).")

    R.h2("3.4 Changes from the proposal")
    R.para(
        "**Training data:** 261 of 298 training items were AI-drafted (and marked), because the human "
        "requests available at the freeze were needed for an all-human test. **Evaluation:** a "
        "screened contributor collection became a second, pre-registered evaluation. **Scope:** 458 items and 34 test pairs (planned: 300, 20); "
        "LLM and encoder baselines on Kaggle 2×T4 GPUs, adding Jais 3 (released after the freeze); "
        "S1-S3 on CPU.")

    # 4 ------------------------------------------------------------------------------------------
    R.h1("4. Evaluation and results")
    R.h2("4.1 Evaluation data and metrics")
    R.para(
        "**Adequacy.** Both evaluation sets hold native-written requests for the two tools, balanced "
        "across routes, with incomplete variants and in-domain out-of-scope requests. Coverage is "
        "uneven: code-switching marks 64% of training items but 20% of "
        "the test and 35% of the natural set, and 79% of natural messages concern project-management "
        "work. Test intervals span about ±0.09 STS, so "
        "the larger natural set carries the main conclusion. "
        "**Separation.** Splits follow scenario families under a hash-locked manifest; vocabularies and "
        "hyper-parameters come from train/dev only; the test was scored once, by a command that refuses "
        "changed models; natural-set systems were frozen before labelling. **Limitations.** The "
        "test is one writer's templated phrasing (52 sentence frames for 120 items) by the author of "
        "S1's lexicon. "
        "**Metrics.** Besides STS: intent and route macro-F1; unsafe execution (a wrong or unwarranted "
        "tool call) over all items and over predicted ACT outputs (all, ACT); unnecessary clarification (Unn. clar.: "
        "gold ACT answered with CLARIFY, set mainly by the shared extractor); and minimal pairs with both versions correct. Intervals are "
        "family-level bootstraps (10,000 resamples) [8]; paired tests swap whole families [7], "
        "Holm-corrected [9]. **Hypotheses** (fixed before labelling): "
        "N1, S1 beats S2; N2, the extractor fix drawn from the test errors (S1 v1.1) beats S1; N3, S9 "
        "beats S8.")

    R.h2("4.2 Results")
    h1_, h2_ = cmp_["S2 vs S1"], cmp_["S3 vs S1"]
    R.para(
        "On the locked test (Table 2) both classifiers scored below the rules "
        f"(S2 − S1 = {signed(h1_['difference'])} {ci(h1_['ci'])}, Holm p = {h1_['p_holm']:.3f}; "
        f"S3 − S1 = {signed(h2_['difference'])}, Holm p = {h2_['p_holm']:.3f}; S3 vs S2: "
        f"p = {cmp_['S3 vs S2']['p_value']:.2f}), so here the answer to the research question is "
        "negative. Clustering items by shared sentence frame keeps the differences but raises Holm p "
        "to 0.055-0.114.")
    rows = []
    for name, m, c in (("oracle (gold intent)", o, TEST["systems"]["oracle"]),
                       ("S1 keyword rules", t1, TEST["systems"]["S1"]),
                       ("S2 TF-IDF + LR", t2, TEST["systems"]["S2"]),
                       ("S3 + char n-grams", t3, TEST["systems"]["S3"])):
        rows.append([name, f"{m['safe_task_success']:.3f} {ci(c['ci']['safe_task_success'])}",
                     f3(m["intent_macro_f1"]), f3(m["route_macro_f1"]), f3(m["unsafe_execution_rate"]),
                     f3(1 - m["execution_precision"]),
                     f3(m["unnecessary_clarification_rate"]), f3(m["pair_both_success"])])
    R.table("Locked test (120 items, 34 pairs). Metrics as defined in §4.1.",
            ["System", "STS [95% CI]", "Intent F1", "Route F1", "Unsafe (all)", "Unsafe (ACT)",
             "Unn. clar.", "Pairs"], rows, widths=[3.3, 3.3, 1.5, 1.5, 1.6, 1.7, 1.6, 1.3],
            bold_rows={1})

    nat_rows = []
    for name, key in (("oracle (gold intent)", "oracle"), ("S1 keyword rules", "S1"),
                      ("S2 TF-IDF + LR", "S2"), ("S3 + char n-grams", "S3")):
        m = nat_metrics(key)
        c = NAT["systems"][key]["ci"]["safe_task_success"]
        r = m["success_by_gold_route"]
        nat_rows.append([name, f"{m['safe_task_success']:.3f} {ci(c)}", f3(m["route_macro_f1"]),
                         f3(m["unsafe_execution_rate"]), f3(1 - m["execution_precision"]),
                         f3(m["unnecessary_clarification_rate"]),
                         f3(m["pair_both_success"]),
                         f"{r['ACT']:.2f} / {r['CLARIFY']:.2f} / {r['ABSTAIN']:.2f}"])
    for name, key, summ in (("S8 Jais-3-7B tool calls", "S8", s8m), ("S9 S8 + policy", "S9", s9m)):
        nat_rows.append([name, f"{summ['sts']:.3f} {ci(summ['ci'])}",
                         f3(nat_seed_mean(key, "route_macro_f1")),
                         f3(nat_seed_mean(key, "unsafe_execution_rate")),
                         f3(1 - nat_seed_mean(key, "execution_precision")),
                         f3(nat_seed_mean(key, "unnecessary_clarification_rate")),
                         f3(nat_seed_mean(key, "pair_both_success")),
                         " / ".join(f"{nat_seed_route(key, r):.2f}" for r in ("ACT", "CLARIFY", "ABSTAIN"))])
    R.table("Natural set (238 items, 17 writers, 64 pairs; S8/S9: 3-seed means; last: success by gold "
            "route).",
            ["System", "STS [95% CI]", "Route F1", "Unsafe (all)", "Unsafe (ACT)", "Unn. clar.",
             "Pairs", "ACT / CLAR / ABST"], nat_rows, widths=[3.4, 3.2, 1.2, 1.3, 1.4, 1.2, 1.1, 3.2],
            bold_rows={1})
    N1, N3 = tests["N1 S1 vs S2"], tests["N3 S9 vs S8"]
    R.para(
        "**N1 was confirmed** with a larger gap: "
        f"{signed(N1['difference'])} {ci(N1['ci'])}, Holm p = {N1['p_holm']:.4f}; S1 led for 13 of 17 "
        "writers and under every alternative labelling (Table 7). **N3 was confirmed:** "
        f"{signed(N3['difference'])} {ci(N3['ci'])}, Holm p = {N3['p_holm']:.3f}. **N2 was not:** the "
        "extractor fix changed one natural item. S4 Jais-3-3.7B "
        f"({f3(s4_nat('jais3-3.7b'))}) and S7 ({S7_NAT:.3f}) score above S1, within its interval "
        "(untested; Table 4). Even the best systems are unsafe: "
        f"{1 - t1['execution_precision']:.0%} of S1's test executions and "
        f"{1 - n1['execution_precision']:.0%} of its natural ones were wrong, though 13 of its 16 wrong "
        "test calls missed only the exact reminder text.")

    R.h2("4.3 Analysis and errors")
    R.para(
        "**Training data.** In the drafted training set 88% of reminders use the verb «فكرني» and 5% "
        "the noun «تذكير»; in the test, 50% and 42%. S2 sent 15 of the 62 test reminders to the alarm "
        "tool, mostly noun forms. In exploratory "
        "leave-one-writer-out "
        "training, about 215 added human messages raised S3 on new writers from "
        f"{hum['S3 | drafted+author']['sts']:.3f} to {hum['S3 | drafted+author+contributors']['sts']:.3f}, "
        "within 0.025 of the rules (S1 v1.1, p = 0.49), while MARBERTv2 trained on drafted data tied the "
        f"rules on the test ({s6['locked_test_core_v1']:.3f}) but reached only "
        f"{s6['natural_core_v1']:.3f} on new writers. Human training data mattered more than the model "
        "class.")
    R.para(
        "**Failure types.** S2's 20 extra test failures are intent errors S1 never makes: 9 wrong "
        "tools, 7 declined in-scope requests, 6 wrong-intent clarifications (Table 9). "
        "**Shared pipeline.** With gold intents, 15 of the 23 test failures trace to request framing "
        "the extractor does not strip («عايز تذكير» or «اعمللي»). New writers failed elsewhere: emphasis "
        "words after the verb stay in the text («ضروري» in «فكرني ضروري بكره…»), and day-scale expressions («بعد اسبوع») were hardest (oracle 0.58 on 31 "
        "items); hence N2's small effect; S9's gain over S8 is mostly on incomplete requests (Table 3).")

    # 5 ------------------------------------------------------------------------------------------
    R.h1("5. Conclusion")
    R.para(
        "We built a task contract, a safety-aware metric, a 458-item benchmark and a 247-message "
        "independent evaluation set. Keyword rules beat TF-IDF classifiers on both locked evaluations "
        "(+0.206 STS on new writers), a gap exploratory analyses trace to AI-drafted training data. A "
        "deterministic policy makes LLM tool calling safer (+0.102 STS). The shared extractor limits "
        "every system (a third of the rules' calls were wrong, mostly in the reminder text); a learned "
        "text extractor and a third, untouched evaluation are next.")

    # References ----------------------------------------------------------------------------------
    R.h1("References")
    refs = [
        "[1] Ross, H., Mahabaleshwarkar, A. S. and Suhara, Y. (2025). When2Call: When (not) to Call "
        "Tools. *NAACL-HLT*, 3391-3409.",
        "[2] Abdul-Mageed, M., Elmadany, A. and Nagoudi, E. M. B. (2021). ARBERT & MARBERT: Deep "
        "bidirectional transformers for Arabic. *ACL*.",
        "[3] Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L. (2023). QLoRA: Efficient "
        "finetuning of quantized LLMs. *NeurIPS*.",
        "[4] Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR* 12, "
        "2825-2830.",
        "[5] Sengupta, N. et al. (2023). Jais and Jais-chat: Arabic-centric foundation and "
        "instruction-tuned open generative large language models. *arXiv:2308.16149*.",
        "[6] Larson, S. et al. (2019). An evaluation dataset for intent classification and "
        "out-of-scope prediction. *EMNLP*.",
        "[7] Berg-Kirkpatrick, T., Burkett, D. and Klein, D. (2012). An empirical investigation of "
        "statistical significance in NLP. *EMNLP*.",
        "[8] Field, C. A. and Welsh, A. H. (2007). Bootstrapping clustered data. *JRSS B* 69(3).",
        "[9] Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian "
        "Journal of Statistics* 6(2).",
        "[10] Mosbach, M., Andriushchenko, M. and Klakow, D. (2021). On the stability of fine-tuning "
        "BERT. *ICLR*.",
        "[11] Jais-2-8B-Chat, model card. Hugging Face: inception42/Jais-2-8B-Chat.",
        "[12] Jais-3-7B and Jais-3-3.7B, model cards. Hugging Face: inception42/Jais-3-7B, "
        "inception42/Jais-3-3.7B.",
        "[13] Nile-Chat-4B, model card. Hugging Face: MBZUAI-Paris/Nile-Chat-4B.",
        "[14] Yang, A. et al. (2025). Qwen3 technical report. *arXiv:2505.09388*.",
    ]
    for ref in refs:
        p = R.doc.add_paragraph()
        fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT, after=6)
        add_runs(p, ref)

    # Appendices -----------------------------------------------------------------------------------
    R.page_break()
    R.h1("Appendix A. Additional results")
    s7t, s8t, s7n = s7_test(), s8_test(), s7_nat()
    ext = [
        ["S1 keyword rules (reference)", f3(t1["safe_task_success"]), f3(n1["safe_task_success"])],
        ["S4 Jais 2 8B (6-shot)", f3(s4_test("jais2-8b")), f3(s4_nat("jais2-8b"))],
        ["S4 Jais-3-7B (6-shot)", f3(s4_test("jais3-7b")), f3(s4_nat("jais3-7b"))],
        ["S4 Jais-3-3.7B (6-shot)", f3(s4_test("jais3-3.7b")), f3(s4_nat("jais3-3.7b"))],
        ["S4 Nile-Chat 4B (6-shot)", f3(s4_test("nile-4b")), f3(s4_nat("nile-4b"))],
        ["S4 Qwen3 8B (6-shot)", f3(s4_test("qwen3-8b")), f3(s4_nat("qwen3-8b"))],
        ["S5 multilingual-e5-small + LR", f3(ENC["systems"]["S5"]["locked_test_core_v1"]["sts"]),
         f3(ENC["systems"]["S5"]["natural_core_v1"]["sts"])],
        ["S6 MARBERTv2 fine-tuned (5 seeds)",
         f"{s6['locked_test_core_v1']:.3f} ± {ENC['s6_seed_sd']['locked_test_core_v1']:.3f}",
         f"{s6['natural_core_v1']:.3f} ± {ENC['s6_seed_sd']['natural_core_v1']:.3f}"],
        ["S7 Jais 2 + QLoRA (3 seeds)", f"{st.mean(s7t):.3f} (" + " / ".join(f3(x) for x in s7t) + ")",
         f"{S7_NAT:.3f} (" + " / ".join(f3(x) for x in s7n) + ")"],
        ["S8 Jais-3-7B tool calling (3 seeds)", f"{st.mean(s8t):.3f} (" + " / ".join(f3(x) for x in s8t) + ")",
         f"{s8m['sts']:.3f}"],
        ["S9 S8 + deterministic policy (3 seeds)", "not run", f"{s9m['sts']:.3f}"],
    ]
    R.table("Safe Task Success of the extension systems on both locked sets (S4: primary setting, "
            "6 examples with prior calibration; S5/S6 fitted after both evaluations, exploratory).",
            ["System", "Locked test (120)", "Natural (238)"], ext, widths=[7.0, 4.5, 4.5])
    R.para(
        "Out-of-scope requests separate the learned systems on new writers: S6, Jais 2 and Nile-Chat "
        "declined only about half of them (0.48, 0.48 and 0.53), against 0.85 for the rules. These "
        "requests sit inside the reminder domain (recurring, deleting or changing a reminder), where the "
        "rules' explicit out-of-scope verbs decide better.")

    hrows = []
    for model in ("S2", "S3", "S5"):
        hrows.append([model] + [f"{hum[f'{model} | {pool}']['sts']:.3f} ({hum[f'{model} | {pool}']['intent_accuracy']:.3f})"
                                for pool in ("drafted+author", "contributors only",
                                             "drafted+author+contributors")])
    hrows.append(["S1 rules (reference)", "", "", f"{hum['S1-v1.1']['sts']:.3f} (0.933)"])
    R.table("Exploratory leave-one-writer-out training on the natural set: STS (intent accuracy). Each "
            "held-out writer's scenario families are removed from training; about 215 contributor "
            "messages per fold; S2/S3 use grouped-CV settings; S1 is core v1.1.",
            ["Classifier", "Drafted + author (338)", "Contributors only (~215)", "Both"],
            hrows, widths=[3.5, 4.2, 4.2, 4.1])
    tsv = list(csv.DictReader((ROOT / "results/posthoc/template_sensitivity.csv").open(encoding="utf-8")))
    srows = []
    for r in tsv:
        if r["hypothesis"] in ("H1", "H2"):
            srows.append([f"{r['clustering']}, {r['comparison']}", r["clusters"],
                          f"{float(r['difference']):+.3f}".replace("-", "−"),
                          f"[{float(r['ci_low']):.3f}, {float(r['ci_high']):.3f}]".replace("-", "−"),
                          f"{float(r['p_holm']):.3f}"])
    R.table("Locked test: sensitivity of the rules-vs-classifier differences to the clustering unit "
            "(frames: items sharing a sentence frame).",
            ["Clustering, comparison", "Clusters", "Difference", "95% CI", "Holm p"], srows,
            widths=[6.6, 1.8, 2.2, 3.2, 2.2])
    vrows = []
    for label, v in SENS.items():
        a, b = v["N1 S1 vs S2"], v["N3 S9 vs S8"]
        vrows.append([label, str(v["n"]), f"{a['difference']:+.3f} (p {a['p_value']:.4f})",
                      f"{b['difference']:+.3f} (p {b['p_value']:.4f})"])
    R.table("Natural set: confirmed differences under alternative labels and scoring rules "
            "(unadjusted p).",
            ["Labels / rule", "n", "N1 S1 − S2", "N3 S9 − S8"], vrows, widths=[7.4, 1.2, 3.7, 3.7])

    R.h1("Appendix B. Representative errors and failure types")
    R.table("Real outputs of the shared pipeline with the gold intent (oracle). Arabic cells read "
            "right to left; the second line of a cell is the reminder text.",
            ["Message", "Gold", "Pipeline output", "Cause"],
            [["@عايز تذكير بكرة الساعة 5 العصر علشان احدث سجل المخاطر", "ACT\n@احدث سجل المخاطر",
              "ACT\n@عايز علشان احدث سجل المخاطر", "want-verb framing kept (test)"],
             ["@اعمللي تذكير لو ينفع", "CLARIFY time, reminder text", "CLARIFY time",
              "filler read as content (test)"],
             ["@فكرني بكرة الساعة 8 الصبح بحاجة محتاجه ضروري", "CLARIFY reminder text", "ACT",
              "vague content accepted (test)"],
             ["@فكرني اعدل الrisk register ضروري بكره الساعه 7 الصبح", "ACT\n@اعدل الrisk register",
              "ACT\n@اعدل الrisk register ضروري", "emphasis word kept (new writer)"],
             ["@محتاج تفكرني اراجع الjira tickets بكره الساعه 10 الصبح قبل ميعاد ال sprint review",
              "ACT\n@اراجع الjira tickets قبل ميعاد ال sprint review",
              "ACT\n@محتاج اراجع الjira tickets قبل ميعاد ال sprint review",
              "second-person framing (new writer)"]],
            widths=[5.4, 3.6, 3.8, 3.2])
    fails = {r["system"]: r for r in csv.DictReader(
        (ROOT / "results/error_analysis/summary.csv").open(encoding="utf-8"))}
    cats = ["wrong reminder text only", "wrong tool", "executed an incomplete request",
            "asked although complete", "asked for the wrong slots", "clarified with the wrong intent",
            "asked instead of declining", "declined an in-scope request"]
    frows = [[c[0].upper() + c[1:]] + [fails[s][c] for s in ("oracle", "S1", "S2", "S3")] for c in cats]
    frows.append(["All failures"] + [fails[s]["failures"] for s in ("oracle", "S1", "S2", "S3")])
    R.table("Locked test: failures by type (counts of 120 items; from results/error_analysis).",
            ["Failure type", "Oracle", "S1", "S2", "S3"], frows, widths=[7.6, 2.1, 2.1, 2.1, 2.1],
            bold_rows={len(frows) - 1})

    R.h1("Appendix C. Reproducing the results")
    R.para("All commands run from the repository root on a CPU. The first four rebuild Tables 2, 3, 6 "
           "and 7 from the stored outputs and reproduce the committed files exactly (Table 7 also needs "
           "the private labelling sheets in `data/raw/private/`, which are not shared); the encoder "
           "commands retrain S5/S6 (slow on CPU). The GPU systems were run with the notebooks in "
           "`notebooks/`.")
    R.code([
        "uv sync",
        "uv run pytest                                               # unit tests",
        "uv run python scripts/error_analysis.py --out results/error_analysis   # T2",
        "uv run python scripts/natural_tables.py                     # Table 3",
        "uv run python scripts/posthoc_analysis.py --out results/posthoc        # T6",
        "uv run --with openpyxl python scripts/natural_sensitivity.py           # T7",
        "uv run --extra neural python scripts/posthoc_encoders_human.py encoders",
        "uv run --extra neural python scripts/posthoc_encoders_human.py human   # T4-5",
        "uv run --extra ui streamlit run app/streamlit_app.py        # demo",
    ])
    R.table("Pretrained models used, with the pinned Hugging Face revisions.",
            ["System", "Hugging Face model", "Revision"],
            [["S4, S7", "inception42/Jais-2-8B-Chat", "da0e1639cd92b508b24120f7f77f5270a8465dc4"],
             ["S4, S8, S9", "inception42/Jais-3-7B", "94d69f49b4af702d42fcee122cc06e493a6fe7e1"],
             ["S4", "inception42/Jais-3-3.7B", "3d5754d5db87cd3a59f0f5b09c7ac39b351ff5ed"],
             ["S4", "MBZUAI-Paris/Nile-Chat-4B", "2a365667659bbd6786c2760d868e37c506d76ce3"],
             ["S4", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"],
             ["S5", "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"],
             ["S6", "UBC-NLP/MARBERTv2", "fe88db9db8ccdb0c4e1627495f405c44a5f89066"]],
            widths=[2.2, 5.6, 8.2])

    R.h1("Appendix D. Use of generative AI")
    R.para(
        "The 300 AI-drafted training and development requests were generated with an AI assistant and "
        "are marked `source: assistant_draft`; none is in either evaluation set. The code, tests and "
        "documentation were developed with Claude Code as a programming assistant. During adjudication "
        "of the natural set, the assistant checked the author's final labels against the written "
        "guide and flagged 24 that broke a rule; the author approved each correction. No message was "
        "labelled from scratch by an AI system. The author designed the study, wrote the test requests "
        "and their labels, recruited the contributors, ran the experiments and is responsible for every "
        "claim in this report.")
    return R


def to_pdf(docx: Path, pdf: Path) -> None:
    ps = (f"$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
          f"$d = $w.Documents.Open('{docx}'); $d.SaveAs([ref]'{pdf}', [ref]17); "
          f"$p = $d.ComputeStatistics(2); "
          f"$r = $d.Content; $f = $r.Find; $null = $f.Execute('References'); "
          f"Write-Output \"pages=$p references_on_page=$($r.Information(3))\"; "
          f"$d.Close([ref]0); $w.Quit()")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", action="store_true", help="also export a PDF with Microsoft Word")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if git("status", "--porcelain") and not a.allow_dirty:
        sys.exit("working tree has uncommitted changes; commit first so the printed hash matches")
    commit = git("rev-parse", "HEAD")
    docx = OUT / "ATHAR_Chat_Final_Report.docx"
    build(commit).save(docx)
    print(f"wrote {docx.relative_to(ROOT)} for commit {commit}")
    if a.pdf:
        to_pdf(docx, docx.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
