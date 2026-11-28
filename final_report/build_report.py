"""Build the final course report (A4 .docx, and .pdf through Microsoft Word).

Every table is filled from the committed data and result files, and so are the counts quoted in the
text; the few figures typed into a sentence are checked against those files when the report is built
(``require``), so a stale number stops the build. The commit hash printed in the Deliverables
section is read from git, so the report always names the commit it was built from.

    uv run --with python-docx python final_report/build_report.py            # .docx only
    uv run --with python-docx python final_report/build_report.py --pdf      # + .pdf (Windows, Word)
    uv run --with python-docx python final_report/build_report.py --pdf --allow-dirty --out-dir DIR   # draft

The build refuses to run on a working tree with uncommitted changes (``--allow-dirty`` overrides),
because the printed commit would not match the files the report describes, and it warns when that
commit is not on the remote yet (a commit that was never pushed cannot be opened by a reader).
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
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/aminayousef2011/athar-chat"
FONT = "Times New Roman"
MONO = "Consolas"
MAIN_PAGE_LIMIT = 4  # abstract to conclusion; references and appendices do not count


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
LVR = load("results/posthoc/llm_vs_rules.json")  # post hoc: S4/S7 vs S1 (scripts/posthoc_llm_vs_rules.py)
V3 = load("results/v3/evaluation.json")          # protocol v3, gate G5 (registered H-a, H-b)
V3G = load("results/v3/evaluation_gpu.json")     # protocol v3 GPU systems (descriptive, amendment A7)
C12 = load("results/posthoc/core_v12.json")      # post hoc: core v1.2 extractor fixes (scripts/posthoc_core_v12.py)
CA = load("results/posthoc/ceiling_ambiguity.json")  # post hoc: ceiling on agreed vs disagreed items (scripts/posthoc_ceiling_ambiguity.py)
LT = load("results/posthoc/learned_text.json")   # post hoc: learned reminder-text tagger (scripts/posthoc_learned_text.py)
AGREE_V3 = load("results/v3/agreement.json")
V3_MAN = load("data/v3/manifest.json")
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


S4_DIRS = {"jais2-8b": "7b-others", "nile-4b": "7b-others", "qwen3-8b": "7b-others",
           "jais3-7b": "7a-jais3", "jais3-3.7b": "7a-jais3"}


def precision(rel: str) -> str:
    """Numeric format a GPU result recorded for its model (2x T4 has no bf16)."""
    return str(load(rel)["model"]["precision"])


def s4_nat(model: str) -> float:
    return NAT["llm_with_core_v1.1"][f"S4-{model}-6shot-prior"]["stored"]


def s7_nat() -> list[float]:
    return [NAT["llm_with_core_v1.1"][f"S7-jais2-qlora-s{i}-prior"]["stored"] for i in range(3)]


def nat_confirmatory_families() -> int:
    conf = {i["id"] for i in NAT["systems"]["S1"]["items"]}
    return len({i["family_id"] for i in NAT["systems"]["S1"]["items"] if i["id"] in conf})


def jsonl(rel: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / rel).read_text(encoding="utf-8").splitlines() if line.strip()]


def csv_rows(rel: str) -> list[dict]:
    with (ROOT / rel).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def require(ok: bool, claim: str) -> None:
    """Stop the build when the committed files no longer support a sentence of the report."""
    if not ok:
        sys.exit(f"report text is out of date with the committed files: {claim}")


SPLITS = load("data/splits/manifest.json")["splits"]
_FAMILY_SPLIT = {fam: name for name, sp in SPLITS.items() for fam in sp["families"]}
ITEMS: dict[str, list[dict]] = {name: [] for name in SPLITS}
for _item in jsonl("data/athar.jsonl"):
    ITEMS[_FAMILY_SPLIT[_item["family_id"]]].append(_item)
NAT_ITEMS = jsonl("data/natural/natural.jsonl")
NAT_CONF_IDS = {i["id"] for i in NAT["systems"]["S1"]["items"]}
NAT_CONF = [i for i in NAT_ITEMS if i["id"] in NAT_CONF_IDS]
AGREE_TEST = load("docs/agreement.json")
AGREE_NAT = load("results/natural/agreement.json")
FREEZE = load("models/freeze.json")
FAILS = {r["system"]: r for r in csv_rows("results/error_analysis/summary.csv")}
# one real failure of the shared pipeline with the gold intent, quoted in section 4.3
EXAMPLE = next(r for r in csv_rows("results/error_analysis/failures.csv")
               if r["system"] == "oracle" and r["id"] == "c014-a")


def share(items: list[dict], key: str, value: object = True) -> float:
    return sum(i.get(key) == value for i in items) / len(items)


def drafted(split: str) -> int:
    return sum(i.get("source") == "assistant_draft" for i in ITEMS[split])


def split_row(split: str) -> list[str]:
    c = SPLITS[split]["counts"]
    return [str(len(ITEMS[split])), str(len(SPLITS[split]["families"])),
            f"{c['ACT']} / {c['CLARIFY']} / {c['ABSTAIN']}"]


def reminder_share(split: str, word: str) -> float:
    rem = [i for i in ITEMS[split] if i["intent"] == "create_reminder"]
    return sum(word in i["text"] for i in rem) / len(rem)


def count_tests() -> int:
    """Number of tests pytest collects in this checkout."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    n = sum(int(m.group(1)) for m in re.finditer(r"^tests[/\\]\S+: (\d+)\s*$", out, re.M))
    require(n > 0, "pytest collected no tests (run the build inside the project environment)")
    return n


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
            fmt(p)
            add_runs(p, "• " + it)

    def code(self, lines: list[str]) -> None:
        for i, line in enumerate(lines):
            p = self.doc.add_paragraph()
            fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT, after=6 if i == len(lines) - 1 else 0, line=1.0,
                keep_next=i < len(lines) - 1)  # keep the listing on one page
            add_runs(p, line, size=10, font=MONO)

    def link_para(self, *parts: str | tuple[str, str], align=WD_ALIGN_PARAGRAPH.LEFT) -> None:
        """A bullet made of text parts and (url, shown text) links."""
        p = self.doc.add_paragraph()
        fmt(p, align=align, after=6)
        add_runs(p, "• ")
        for part in parts:
            if isinstance(part, str):
                add_runs(p, part)
            else:
                self._hyperlink(p, *part)

    @staticmethod
    def _hyperlink(p, url: str, shown: str, mono: bool | None = None, size: float | None = None) -> None:
        if mono is None:
            mono = shown != url  # the commit hash and file paths are set like the other identifiers
        hl = OxmlElement("w:hyperlink")
        hl.set(qn("r:id"), p.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/"
                                                  "2006/relationships/hyperlink", is_external=True))
        r = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        fonts = OxmlElement("w:rFonts")
        for attr in ("w:ascii", "w:hAnsi", "w:cs"):
            fonts.set(qn(attr), MONO if mono else FONT)
        color = OxmlElement("w:color")
        color.set(qn("w:val"), "0563C1")
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), str(int((size or (11 if mono else 12)) * 2)))
        for el in (fonts, color, u, sz):
            rpr.append(el)
        t = OxmlElement("w:t")
        t.text = shown
        r.append(rpr)
        r.append(t)
        hl.append(r)
        p._p.append(hl)

    def page_break_before_last(self) -> None:
        """Start the last paragraph (a heading) on a new page, without an empty paragraph."""
        self.doc.paragraphs[-1].paragraph_format.page_break_before = True

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
                    link = re.fullmatch(r"\[\[(.+?)\|(.+?)\]\]", line)
                    if link:  # [[url|shown text]]
                        self._hyperlink(p, link.group(1), link.group(2), mono=False, size=10)
                    else:
                        add_runs(p, line, size=10, bold=bold, rtl_all=rtl)
                tcpr = cell._tc.get_or_add_tcPr()
                mar = OxmlElement("w:tcMar")
                for side in ("top", "bottom"):
                    m = OxmlElement(f"w:{side}")
                    # the last row gets 6 pt below it, so text after a table does not touch its rule
                    m.set(qn("w:w"), "120" if side == "bottom" and ri == len(rows) else "15")
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
        # no empty spacer paragraph after the table: the brief allows no extra blank lines

    def save(self, path: Path) -> None:
        self.doc.save(path)


# ---------------------------------------------------------------- content ----------------------

def build(commit: str, branch: str) -> Report:
    R = Report()
    n_tests = count_tests()
    t1, t2, t3, o = (test_metrics(s) for s in ("S1", "S2", "S3", "oracle"))
    n1, n2, n3, no = (nat_metrics(s) for s in ("S1", "S2", "S3", "oracle"))
    cmp_ = TEST["comparisons"]
    tests = NAT["tests"]
    s8m, s9m = NAT["summary"]["S8 (seed mean)"], NAT["summary"]["S9 (seed mean)"]
    s6 = ENC["s6_seed_mean"]
    hum = HUM["systems"]
    N1, N3 = tests["N1 S1 vs S2"], tests["N3 S9 vs S8"]
    n_all = sum(len(v) for v in ITEMS.values())
    n_nat, n_conf, n_writers = len(NAT_ITEMS), len(NAT_CONF), len(NAT["writers"])
    n_drafted = drafted("train") + drafted("dev")
    n_fit = len(ITEMS["train"]) + len(ITEMS["dev"])
    s8_invalid_nat = [nat_metrics(f"S8-s{i}")["invalid_outputs"] for i in range(3)]

    # figures typed into the text below, checked against the committed files
    fits = [FREEZE["systems"][s]["resources"] for s in ("S2", "S3")]
    require(max(f["final_fit"]["cpu_seconds"] for f in fits) < 0.3
            and max(f["predict_ms_per_item"] for f in fits) < 0.3
            and round(FREEZE["process_peak_rss_mb"]) == 141, "S2/S3 fit time, inference time, peak memory")
    require(load("results/posthoc/frame_overlap_summary.json")["distinct_frames_A"] == 52, "52 sentence frames")
    require(int(FAILS["S2"]["failures"]) - int(FAILS["S1"]["failures"]) == 20
            and [FAILS["S2"][c] for c in ("wrong tool", "declined an in-scope request",
                                          "clarified with the wrong intent")] == ["9", "7", "6"],
            "S2 failure types")
    require(round(t1["unsafe_execution_rate"] * t1["n"]) == 16
            and FAILS["S1"]["wrong reminder text only"] == "13", "13 of S1's 16 wrong test calls")
    fix = {r["system"]: r for r in csv_rows("results/posthoc/extractor_fix.csv")}
    require(fix["oracle"]["fixed"] == "14" and FAILS["oracle"]["failures"] == "23", "14 of 23 oracle failures")
    require(EXAMPLE["cause"] == "wrong reminder text only" and EXAMPLE["pred_route"] == "ACT"
            and EXAMPLE["gold_datetime"] == EXAMPLE["pred_datetime"], "the quoted pipeline failure")
    require(n_drafted == sum(i.get("source") == "assistant_draft" for v in ITEMS.values() for i in v),
            "AI-drafted items are only in train and dev")
    protocol = (ROOT / "docs/protocol_natural.md").read_text(encoding="utf-8")
    tables = (ROOT / "results/natural/tables.md").read_text(encoding="utf-8")
    phenomena = (ROOT / "results/posthoc/phenomena.md").read_text(encoding="utf-8")
    require("S1 > S2 for 13 of 17 writers" in tables, "S1 led for 13 of 17 writers")
    require("474 confirmatory messages from 34 writers" in protocol and "corrected 24 rows" in protocol,
            "screened pool of 474 messages by 34 writers; 24 adjudication corrections")
    tech = (ROOT / "docs/report.md").read_text(encoding="utf-8")
    require("19 of 21 early submissions shared one writing pattern; several codes gave identical answers"
            in protocol.replace("\n", " ")
            and "Google Form gathered 21 responses" in tech and "(protocol log 2026-10-10)" in tech,
            "first Google Form: 21 responses, 19 sharing one writing pattern, discarded on 10 October")
    require(re.search(r"\| day-scale expression[^|]*\| 31 \| 0\.581 \|", phenomena) is not None,
            "oracle 0.58 on 31 day-scale items")
    frame_p = [float(r["p_holm"]) for r in csv_rows("results/posthoc/template_sensitivity.csv")
               if r["hypothesis"] in ("H1", "H2") and not r["clustering"].startswith("family (")]
    require(f"{min(frame_p):.3f}-{max(frame_p):.3f}" == "0.055-0.526", "Holm p under frame clustering")
    half = [(c["high"] - c["low"]) / 2 for c in (TEST["systems"][s]["ci"]["safe_task_success"]
                                                 for s in ("S1", "S2", "S3"))]
    require(all(0.08 <= h < 0.095 for h in half), "test intervals of about ±0.09")
    s6_abst = st.mean(ENC["systems"][f"S6-s{i}"]["natural_core_v1"]["by_route"]["ABSTAIN"] for i in range(5))
    require(f"{s6_abst:.2f}" == "0.48" and "0.659 / 0.475 |" in tables and "0.727 / 0.525 |" in tables
            and f"{n1['success_by_gold_route']['ABSTAIN']:.2f}" == "0.85",
            "declined out-of-scope requests: S6, Jais 2, Nile-Chat, S1")
    v3_n = 236
    jais3_doc = (ROOT / "docs/jais3.md").read_text(encoding="utf-8")
    require("Jais 2 8B stays the pre-registered S4 model"
            in (ROOT / "docs/evaluation_protocol.md").read_text(encoding="utf-8")
            and "On **2026-10-15**" in jais3_doc
            and "after the models were frozen (2026-10-12)" in jais3_doc
            and "**before the test split was opened**" in jais3_doc,
            "Jais 2 pre-registered; Jais 3 released on 2026-10-15, after the freeze, before the test")
    require(FREEZE["environment"]["platform"].startswith("Windows")
            and all(set(load(f"results/s4/{d}/s4_results/environment.json")["gpus"]) == {"Tesla T4"}
                    for d in ("7a-jais3", "7b-others", "7cd-s7-s8")),
            "S1-S3 frozen on the author's Windows machine; the LLM runs on Kaggle T4 GPUs")
    require(f"**{v3_n} messages from 17 writers" in (ROOT / "docs/protocol_v3.md").read_text(encoding="utf-8")
            and V3["confirmatory_items"] == V3["n_items"] == v3_n == V3G["confirmatory_items"]
            and len(V3["writers"]) == 17 and V3["hypothesis_tests"].startswith("confirmatory")
            and V3_MAN["dataset"]["n_items"] == v3_n,
            "protocol v3: 236 confirmatory items from 17 writers, locked and scored once")
    v3a, v3b = V3["primary"]["H-a S3-human vs S3-drafted"], V3["primary"]["H-b S6-human vs S6-drafted (seed means)"]
    v3c = V3["secondary"]["H-c ABSTAIN success, S6-human vs S6-drafted (seed means)"]
    v3s, v3e = V3["summary"], V3["estimation"]
    require(v3c["n"] == V3_MAN["splits"]["test"]["counts"]["ABSTAIN"] and v3c["p_value"] < 0.05
            and abs(v3c["difference"] - (v3s["S6-human (seed mean)"]["by_route"]["ABSTAIN"]
                                         - v3s["S6-drafted (seed mean)"]["by_route"]["ABSTAIN"])) < 1e-3,
            "v3 H-c: the out-of-scope change quoted in section 4.3 is the registered secondary test")
    v3_lowo = V3["sensitivity"]["leave_one_writer_out"]
    v3_lowo_wins = [sum(v[a] > v[b] for v in v3_lowo.values()) for a, b in
                    (("S3-human", "S3-drafted"), ("S6-human (seed mean)", "S6-drafted (seed mean)"))]
    require(v3a["significant"] and v3b["significant"] and v3a["difference"] > 0 and v3b["difference"] > 0
            and all(t["p_value"] < 0.05 for t in V3["sensitivity"]["writer_clustered"].values())
            and v3_lowo_wins == [17, 17],
            "v3: H-a and H-b confirmed, also writer-clustered and in every leave-one-writer-out analysis")
    v3_ctrl = V3["descriptive"]["size-matched control (A4): S3-contrib-only vs S3-drafted"]
    require(v3s["S3-contrib-only"]["sts"] == v3s["S3-human"]["sts"] and v3_ctrl["p_value"] < 0.05,
            "v3 control: contributors only reach S3-human")
    v3_s3r, v3_s6r = v3e["S3-human minus S1-v1.1"], v3e["S6-human (seed mean) minus S1-v1.1"]
    require(v3_s3r["within_0.10_of_the_rules"] and v3_s6r["within_0.10_of_the_rules"]
            and v3_s6r["ci"]["low"] > 0 and v3_s3r["ci"]["low"] < 0 < v3_s3r["ci"]["high"],
            "v3 estimation: S3-human level with the rules, S6-human interval above zero (estimate only)")
    v3_adj = load("results/v3/adjudication.json")["adjudicated"]
    nat_writers = NAT["writers"] if isinstance(NAT["writers"], list) else list(NAT["writers"])
    require(not set(V3["writers"]) & set(nat_writers) and len(nat_writers) == n_writers,
            "natural-set and v3 writers are disjoint (34 other writers in all)")
    g_cmp = V3G["comparisons"]
    require(g_cmp["S4 raw vs normalized text"]["p_value"] > 0.05 and g_cmp["S9-fair vs S8-fair"]["p_value"] > 0.05
            and V3G["s8_fair_no_valid_call"] == 8 and V3G["s4_norm_raw_same_intent"] == 230,
            "v3 GPU systems: raw text and the policy make no significant difference (descriptive)")
    # where S9's gain over S8 comes from: items S8 answered with no valid call go to the rule fallback
    s9_fallback = s9_policy = 0
    for i in range(3):
        a = {x["id"]: x for x in NAT["systems"][f"S8-s{i}"]["items"]}
        b = {x["id"]: x for x in NAT["systems"][f"S9-s{i}"]["items"]}
        for k, x in a.items():
            gain = int(b[k]["success"]) - int(x["success"])
            if x["valid"]:
                s9_policy += gain
            else:
                s9_fallback += gain
    s9_gain = s9_fallback + s9_policy
    require(abs(s9_gain / (3 * n_conf) - N3["difference"]) < 1e-9 and s9_fallback > s9_policy > 0,
            "S9's gain over S8 split into fallback and policy items")
    frames = load("results/posthoc/frame_overlap_summary.json")
    require(frames["test_items"] == len(ITEMS["test"]) and frames["frame_seen_in_dev"] == 0,
            "sentence-frame overlap of the test with train and dev")
    n_human = 215  # the figure used in docs/report.md and the registered protocol v3
    require(abs(st.mean(f["contributor_training"] for f in HUM["folds"].values()) - n_human) < 1,
            "about 215 contributor messages per training fold")
    # test -> natural change of each learned recognizer (the Table 5 values), discussed in section 4.3
    s7t_mean = st.mean(s7_test())
    shift = {"S1": n1["safe_task_success"] - t1["safe_task_success"],
             "S2": n2["safe_task_success"] - t2["safe_task_success"],
             "S5": ENC["systems"]["S5"]["natural_core_v1"]["sts"] - ENC["systems"]["S5"]["locked_test_core_v1"]["sts"],
             "S6": s6["natural_core_v1"] - s6["locked_test_core_v1"],
             "S7": S7_NAT - s7t_mean}
    s1_ci_t, s1_ci_n = TEST["systems"]["S1"]["ci"]["safe_task_success"], NAT["systems"]["S1"]["ci"]["safe_task_success"]
    require(s6["locked_test_core_v1"] > t1["safe_task_success"] and s7t_mean > t1["safe_task_success"]
            and s1_ci_t["low"] <= s7t_mean <= s1_ci_t["high"] and s1_ci_n["low"] <= S7_NAT <= s1_ci_n["high"]
            and min(shift.values()) == shift["S5"] and shift["S6"] < shift["S2"] < shift["S7"] < shift["S1"] < 0,
            "S6/S7 match S1 on the test; drops from test to new writers ordered S5 < S6 < S2 < S7 < S1")
    # wrong executions split by severity: wrong reminder text only vs wrong tool / instant / should not act
    exec_t = sum(i["executed"] for i in TEST["systems"]["S1"]["items"])
    wrong_t = round(t1["unsafe_execution_rate"] * t1["n"])
    harm_t = int(FAILS["S1"]["harmful_executions"])
    require(wrong_t - harm_t == int(FAILS["S1"]["wrong reminder text only"]), "S1 test: wrong executions = text-only + harmful")
    sens_md = (ROOT / "results/natural/sensitivity.md").read_text(encoding="utf-8")
    act_fail = [float(x) for x in re.search(r"^\| S1 \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$",
                                            sens_md, re.M).groups()]  # wrong tool, wrong instant, text only, not executed
    exec_n = sum(i["executed"] for i in NAT["systems"]["S1"]["items"])
    wrong_n = sum(i["executed"] and not i["success"] for i in NAT["systems"]["S1"]["items"])
    harm_n = wrong_n - round(act_fail[2])
    require(harm_n == round(act_fail[0] + act_fail[1])
            + sum(i["executed"] and i["gold_route"] != "ACT" for i in NAT["systems"]["S1"]["items"]),
            "S1 natural: harmful executions = wrong tool + wrong instant + executed a non-ACT item")
    # selecting C: dev (frozen) vs grouped CV on train, post hoc sweep of C alone
    FREEZE_C = {s: float(FREEZE["systems"][s]["params"]["C"]) for s in ("S2", "S3")}
    sweep = csv_rows("results/posthoc/regularization_sweep.csv")
    gcv_c = {r["system"]: r for r in sweep if r["groupcv_choice"] == "True"}
    require(all(abs(float(r["test_sts"]) - test_metrics(r["system"])["safe_task_success"]) < 1e-3
                for r in sweep if r["frozen_choice"] == "True")
            and all(float(gcv_c[s]["test_sts"]) < t1["safe_task_success"] for s in ("S2", "S3"))
            and all(float(gcv_c[s]["C"]) > FREEZE_C[s] for s in ("S2", "S3")),
            "regularization sweep: grouped CV picks weaker regularization, still below S1 on the test")

    # post hoc LLM recognizers vs the rules: the claims of section 4.2 and the conclusion
    lv_n, lv_t = LVR["natural"], LVR["locked_test"]
    require(abs(lv_n["S1"]["sts"] - n1["safe_task_success"]) < 1e-9 and lv_n["n"] == n_conf
            and abs(lv_t["S1"]["sts"] - t1["safe_task_success"]) < 1e-9 and lv_t["n"] == len(ITEMS["test"]),
            "post hoc LLM comparison uses the same S1 items as Tables 2 and 3")
    lv_j2, lv_j33, lv_s7 = (lv_n["comparisons"][k] for k in
                            ("S4 Jais 2 8B vs S1", "S4 Jais-3-3.7B vs S1", "S7 Jais 2 + QLoRA vs S1"))
    lv_sig = [k for k, v in lv_n["comparisons"].items() if v["p_holm"] < 0.05]
    require(lv_sig == ["S4 Jais 2 8B vs S1"] and lv_j2["difference"] < 0
            and not any(v["difference"] > 0 and v["p_holm"] < 0.05
                        for s in (lv_n, lv_t) for v in s["comparisons"].values())
            and max(v["difference"] for v in lv_n["comparisons"].values()) == lv_j33["difference"]
            and all(v["ci"]["low"] < 0 < v["ci"]["high"] for v in (lv_j33, lv_s7)),
            "no LLM recognizer beats S1 significantly; on new writers only Jais 2 differs (worse); "
            "Jais-3-3.7B is the best and, with S7, level with S1")
    require(abs(lv_j33["sts"] - s4_nat("jais3-3.7b")) < 5e-4 and abs(lv_s7["sts"] - S7_NAT) < 5e-4,
            "post hoc STS values equal those of Table 5")
    nat_tables = (ROOT / "results/natural/tables.md").read_text(encoding="utf-8")
    for label, rec in (("S4 Jais-3-3.7B", lv_j33), (r"S7 Jais 2 \+ QLoRA \(3 seeds\)", lv_s7)):
        row = re.search(rf"^\| {label} \|[^|]*\|[^|]*\| ([0-9.]+) \|", nat_tables, re.M)
        require(row is not None and f"{rec['unsafe_execution_rate']:.3f}" == row.group(1),
                f"post hoc unsafe rate of {label} equals results/natural/tables.md")
    len_s1 = SENS["locked gold, lenient reminder text"]["systems"]["S1"]
    lenient_items = round(act_fail[2])  # S1's text-only ACT failures on the locked gold
    require(lenient_items > 0 and abs(n1["safe_task_success"] + lenient_items / n_conf - len_s1) < 5e-4,
            "S1 natural items rescued by the lenient reminder-text rule")

    with (ROOT / "contrib/scenario_cards.csv").open(encoding="utf-8-sig") as fh:
        cards = list(csv.DictReader(fh))
    require(0.65 <= sum(c["domain"] == "project_management" for c in cards) / len(cards) < 0.75,
            "about 70% of the scenario cards are project management")
    # v3 is harder for the shared extractor: compare the core v1.1 ceiling and rules on both sets
    nat_or11, nat_s111 = NAT["summary"]["oracle-v1.1"]["sts"], NAT["summary"]["S1-v1.1"]["sts"]
    v3_or11, v3_s111 = v3s["oracle-v1.1"]["sts"], v3s["S1-v1.1"]["sts"]
    require(v3_or11 < nat_or11 and v3_s111 < nat_s111 and v3_s111 == v3s["S3-human"]["sts"],
            "v3 ceiling and rules both lower than on the natural set; S3-human equals S1 v1.1 on v3")
    require(min(v3s["oracle-v1.1"]["by_route"], key=v3s["oracle-v1.1"]["by_route"].get) == "CLARIFY",
            "CLARIFY is the oracle's weakest route on v3")

    route_n = V3_MAN["splits"]["test"]["counts"]
    for lo, hi in (("S3-drafted", "S3-human"), ("S6-drafted (seed mean)", "S6-human (seed mean)")):
        gain = {r: (v3s[hi]["by_route"][r] - v3s[lo]["by_route"][r]) * route_n[r] for r in route_n}
        require(gain["ABSTAIN"] > sum(gain.values()) / 2, f"{hi}: most of the gain is declined out-of-scope requests")

    # post hoc core v1.2: its v1.1 baselines must be the registered numbers, and the v3 gain held out
    c12n, c12v = C12["natural (development)"], C12["v3 (held-out)"]
    require(abs(c12v["oracle"]["sts_v1.1"] - v3_or11) < 5e-4 and abs(c12v["S1 rules"]["sts_v1.1"] - v3_s111) < 5e-4
            and abs(c12v["S3 + human data"]["sts_v1.1"] - v3s["S3-human"]["sts"]) < 5e-4
            and abs(c12v["S6 MARBERTv2 + human data (5 seeds)"]["sts_v1.1"] - v3s["S6-human (seed mean)"]["sts"]) < 5e-3
            and abs(c12n["oracle"]["sts_v1.1"] - nat_or11) < 5e-4 and c12v["oracle"]["n"] == v3_n
            and c12n["oracle"]["n"] == n_conf,
            "core v1.2 analysis starts from the registered core v1.1 scores")
    require(all(r["difference"] > 0 and r["ci"]["low"] > 0 and r["p_value"] < 0.05 for r in c12v.values()),
            "core v1.2 raises every v3 system, interval above zero")
    c12_lo, c12_hi = min(r["difference"] for r in c12v.values()), max(r["difference"] for r in c12v.values())

    # post hoc learned reminder-text tagger: its baseline is the registered oracle v1.1, and the
    # variant with core v1.1's word lists changes no score
    lt, lt_rules = LT["runs"]["learned"], LT["runs"]["learned + rules"]
    lt_hv, lt_dv, lt_dn = lt["natural (human) -> v3"], lt["drafted + author -> v3"], lt["drafted + author -> natural"]
    require(abs(lt_hv["oracle_rules_sts"] - v3_or11) < 5e-4 and abs(lt_dn["oracle_rules_sts"] - nat_or11) < 5e-4
            and lt_hv["n"] == v3_n and lt_dn["n"] == n_conf
            and all(lt[k][f] == lt_rules[k][f] for k in lt for f in ("oracle_learned_sts", "text_learned"))
            and LT["v3_families_shared_with_natural"] == V3["families"]
            and lt_hv["ci"]["low"] > 0 and lt_dv["p_value"] > 0.05,
            "learned tagger: v1.1 baselines, rule lists change nothing, v3 shares all cards with the natural set")

    R.title("ATHAR Chat: Safe Tool Routing for Egyptian Arabic", "Amina Yousef")

    R.h1("Abstract")
    R.para(
        "Asked in Egyptian Arabic for a reminder or alarm, an assistant "
        "must execute (ACT), ask for missing details (CLARIFY) or decline "
        f"(ABSTAIN). We built a {n_all}-request benchmark and compared keyword rules with "
        "TF-IDF intent classifiers in one shared pipeline, scored by Safe Task "
        "Success (STS: right decision and arguments). "
        f"On {n_conf} messages from {n_writers} new writers, the rules won (STS "
        f"{f3(n1['safe_task_success'])} vs {f3(n2['safe_task_success'])}, Holm p = "
        f"{tests['N1 S1 vs S2']['p_holm']:.4f}). In a pre-registered test on {v3_n} messages from "
        f"{len(V3['writers'])} further writers, adding in-domain human messages raised TF-IDF by "
        f"{signed(v3a['difference'])} and MARBERTv2 by {signed(v3b['difference'])} (Holm p ≤ "
        f"{max(v3a['p_holm'], v3b['p_holm']):.3f}); TF-IDF then matched the rules ({v3_s111:.3f}). "
        "As these share the test's scenarios, this shows the value of in-distribution data, not "
        "of authorship. With gold intents STS is only "
        f"{f3(no['safe_task_success'])}: the hand-written extractor limits every system; a reminder-text "
        f"tagger learned from {lt_hv['training']['messages']} human messages raised this ceiling on v3 by "
        f"{lt_hv['difference']:.3f} (exploratory).")

    # 1 ------------------------------------------------------------------------------------------
    R.h1("1. Motivation and objectives")
    R.para(
        "A reminder assistant must not execute a request whose time or content is missing or "
        "ambiguous, re-ask for what the user gave, or attempt what it cannot do. Egyptian Arabic makes "
        "this harder: messages are informal («اعمللي تذكير بعد بكرة الساعة 9 بالليل عشان اخد الدوا», "
        "\"set me a reminder the day after tomorrow at 9 pm to take my medicine\"), inconsistently "
        "spelled and mixed with English. When2Call [1] showed for English that deciding *when* to call "
        "a tool is its own problem, and out-of-scope requests are hard for classifiers [6]; "
        "no Egyptian Arabic dataset labels this decision. **Intended users:** Egyptian project managers "
        f"asking for work reminders in dialect ({share(NAT_CONF, 'domain', 'project_management'):.0%} of "
        "contributor messages), low-compute developers and researchers.")
    R.para(
        "**Research questions.** (RQ1) When all systems share the same argument extractor and execution "
        "policy, does lightweight supervised intent classification improve Safe Task Success over keyword "
        "rules? (RQ2) Does human-written instead of AI-drafted training data change the answer? "
        "**Objectives:** (O1) a task contract and safety-aware metric; (O2) a reviewed benchmark; "
        "(O3) rules vs. TF-IDF; (O4) new-writer tests; (O5) error analysis.")

    # 2 ------------------------------------------------------------------------------------------
    R.h1("2. Deliverables")
    R.link_para("**Repository:** ", (REPO_URL, REPO_URL), f" (branch `{branch}`), commit ",
                (f"{REPO_URL}/tree/{commit}", commit),
                "; this PDF is built from it.")
    blob = f"{REPO_URL}/blob/{commit}/"
    R.link_para(f"**Data:** the {n_all}-item benchmark is in ", (blob + "data/athar.jsonl", "data/athar.jsonl"),
                " with its splits in ", (blob + "data/splits/manifest.json", "data/splits/manifest.json"),
                f"; the natural set ({n_nat}) is in ", (blob + "data/natural/natural.jsonl", "data/natural/natural.jsonl"),
                " and v3 in `data/v3/manifest.json` (texts private); data card and guides are in `docs/`; "
                "consent covers this project, no public licence yet.",
                align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    # S7 adapters published as a GitHub release; the downloaded files were checked against these
    # hashes (recorded at the freeze in docs/jais3.md and in the stored S7 results)
    s7_hashes = ("64e87a48a6c24f6ff73cd4cfc62946ba28104f218aa9a63a74d196224bb6c62c",
                 "30abde5994ea9b010f017b8c802a4c1fa19e1275456021d16de63c2d2a86735d",
                 "8ada29076edce065ac0c6ac38c8d8d0ad9075ac447d740a6e9724b443d16c9ab")
    require(all(h in jais3_doc for h in s7_hashes), "S7 release hashes equal those recorded at the freeze")
    R.link_para(f"**Code, models, outputs:** package `athar` with {n_tests} tests and locked dependencies "
                "(`uv.lock`); frozen classifiers (`models/`, `models_v2/`), stored predictions (`results/`), "
                "pinned Hugging Face models (Table 14), a Streamlit demo (`app/`) and the technical report "
                "(`docs/report.md`); S7 adapters: release ",
                (f"{REPO_URL}/releases/tag/s7-adapters", "s7-adapters"),
                " (hash-checked); S6 weights not stored.",
                align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # 3 ------------------------------------------------------------------------------------------
    R.h1("3. Methods and implementation")
    R.h2("3.1 Task contract and metric")
    R.para(
        "Each request has a sending time and may call one of two simulated tools, "
        "`create_reminder(datetime, reminder_text)` or `create_alarm(datetime)`. The contract resolves explicit dates, today/tomorrow, "
        "weekdays with «الجاي» (next), clock times and period words such as «الصبح». **ACT** needs every "
        "argument resolved to one future instant; **CLARIFY** names the missing slots (date, time, "
        "AM/PM, past time, reminder text); **ABSTAIN** covers deletion, modification, recurrence, other "
        "people's items, several actions and other tools. **Safe Task Success** counts an item only if "
        "ABSTAIN is declined, CLARIFY asks for exactly the gold slots, or ACT "
        "calls the gold tool at the gold instant with the gold reminder text (exact after "
        "normalization, request words and fillers removed); invalid outputs fail.")

    R.h2("3.2 Data")
    nat_fams = nat_confirmatory_families()
    R.table(
        "Datasets. Train, dev and test share no scenario family; natural and v3 reuse the test's.",
        ["Set", "Items", "Families", "ACT/CLAR/ABST", "Written by", "Use"],
        [["train", *split_row("train"),
          f"{drafted('train')} AI-drafted, {len(ITEMS['train']) - drafted('train')} author", "fitting"],
         ["dev", *split_row("dev"),
          f"{drafted('dev')} AI-drafted, {len(ITEMS['dev']) - drafted('dev')} author", "selection"],
         ["test (locked)", *split_row("test"), "author (native speaker)", "evaluation 1"],
         ["natural (locked)", str(n_conf), str(nat_fams),
          " / ".join(str(sum(i["route"] == r for i in NAT_CONF)) for r in ("ACT", "CLARIFY", "ABSTAIN")),
          f"{n_writers} other native speakers", "evaluation 2"],
         ["v3 (locked)", str(v3_n), str(V3["families"]),
          " / ".join(str(V3_MAN["splits"]["test"]["counts"][r]) for r in ("ACT", "CLARIFY", "ABSTAIN")),
          f"{len(V3['writers'])} further native speakers", "evaluation 3"]],
        widths=[2.7, 1.3, 1.6, 3.2, 4.4, 2.8])
    R.para(
        "The author wrote the test from scenario cards; a second native "
        f"speaker labelled {AGREE_TEST['n_compared']} items "
        f"independently (intent κ = {AGREE_TEST['intent']['kappa']:.3f}, route κ = "
        f"{AGREE_TEST['route']['kappa']:.3f}). For the **natural set**, collected after the freeze, "
        "contributors wrote without AI tools a complete and an incomplete message per test "
        "scenario; a seeded draw of two writers per scenario group (474 screened messages, 34 "
        f"writers) gave {n_nat}, {n_conf} in families unseen in train/dev. Two annotators "
        f"labelled all (intent κ = {AGREE_NAT['intent']['kappa']:.3f}, route κ = "
        f"{AGREE_NAT['route']['kappa']:.3f}) but chose the same slots on only "
        f"{AGREE_NAT['targets_exact_set']['agreement']:.0%} of {AGREE_NAT['targets_exact_set']['n']} "
        f"CLARIFY items and the same instant on {AGREE_NAT['datetime_same_instant']['agreement']:.0%} "
        f"of {AGREE_NAT['datetime_same_instant']['n']} ACT items; the author, who also wrote S1's lexicon, "
        f"adjudicated the {load('results/natural/adjudication.json')['adjudicated']} disagreements alone "
        "(see Table 7). "
        f"**v3** is the other {len(V3['writers'])} writers' {v3_n} "
        f"messages, labelled after its systems were frozen; {v3_adj} were adjudicated jointly (route κ = "
        f"{AGREE_V3['route']['kappa']:.3f}, same slots {AGREE_V3['targets_exact_set']['agreement']:.0%}).")

    R.h2("3.3 Pipeline and systems")
    R.para(
        "**Shared pipeline:** message → normalizer → *intent recognizer* (the only part that differs) "
        "→ extractor → policy → route. The normalizer unifies alef, ya and taa marbuta variants, strips "
        "diacritics and elongation and maps Arabic-Indic digits; a pattern-based extractor parses the "
        "supported time expressions; the reminder text is what remains once request framing, "
        "times and fillers are removed. The policy sends an unsupported intent to ABSTAIN, a supported "
        "one with all slots to ACT, otherwise to CLARIFY, so STS differences come from intent. "
        "Intent is isolated because out-of-scope detection is hard [6]; lexical models fit laptop CPUs. "
        "Extraction is rule-based because resolving a time is date arithmetic and no span-labelled data "
        "existed at the freeze (a learned tagger is tested in Finding 3).")
    R.para(
        "**Systems compared.** **S1:** hand-written trigger and out-of-scope lexicons, untrained. **S2:** word TF-IDF "
        "with class-balanced logistic regression [4]; **S3:** S2 plus character 2-4-grams for spelling "
        "variation; both fitted on train, with C and n-grams chosen by dev intent macro-F1 "
        f"(S2: C = {FREEZE_C['S2']:g}; S3: C = {FREEZE_C['S3']:g}). **Oracle:** gold intents."
        " **Core versions:** the shared normalizer, extractor and policy form the *core*, frozen "
        "as v1.0; v1.1 adds one extractor fix drawn from the test errors (test oracle "
        f"{f3(o['safe_task_success'])} → {C12['locked test']['oracle']['sts_v1.1']:.3f}), so the test then "
        "becomes development data and v1.1 is scored only on new writers. **Extensions** (Appendix A): "
        "S4-S7 replace only the intent recognizer (LLM prompts, embeddings, QLoRA [3]; **S6:** fine-tuned "
        "MARBERTv2 [2]); **S8:** Jais-3-7B [12] does the whole "
        "task by tool calling; **S9:** the policy checks S8's call (else S1).")

    require("split and frozen 2026-10-12" in (ROOT / "docs/data_card.md").read_text(encoding="utf-8")
            and "Recruiting native contributors was not possible in the time available"
            in (ROOT / "docs/evaluation_protocol.md").read_text(encoding="utf-8")
            and "arrived two weeks later" in tech
            and "so that no system would be scored on its" in tech.replace("\n", " ")
            and n_all - n_drafted == 158   # data card: 158 written by the author, 300 assistant-drafted
            and "158 written by the author" in (ROOT / "docs/data_card.md").read_text(encoding="utf-8")
            and not any(i.get("source") == "assistant_draft" for i in ITEMS["test"]),
            "section 3.4: freeze date, contributors two weeks later, all-human test, author-text count")
    env_s6 = load("results/v3/environment_s6.json")
    require(env_s6.get("device") == "cuda" and env_s6.get("gpu") == "Tesla T4"
            and "per S6 seed on CPU" in (ROOT / "docs/report_supplement.md").read_text(encoding="utf-8"),
            "section 3.4 hardware: S6 for v3 on a Kaggle T4; S6 on the locked test and natural set on CPU")
    R.h2("3.4 Changes from the proposal")
    R.para(
        "**Training data:** the proposal planned only hand-written items, but native writers could not be "
        "recruited in time for the freeze (12 October); 19 of the first Google Form's 21 responses "
        f"shared one pattern, likely AI-generated, and were discarded. The author's {n_all - n_drafted} requests, the only human texts, "
        f"supplied the all-human test ({len(ITEMS['test'])}) so that no system is scored on its own training "
        f"data; {drafted('train')} of {len(ITEMS['train'])} training items were therefore AI-drafted "
        f"(`source: assistant_draft`; {n_all} items, {t1['pairs']} test pairs; planned "
        "300, 20). Contributor messages arrived two weeks later and became two more test sets (RQ2). "
        "**Hardware:** S1-S3, S5 and S6 (test, natural set) on a laptop CPU; S4, S7-S9 and S6 (v3) "
        "on Kaggle T4s.")

    # 4 ------------------------------------------------------------------------------------------
    R.h1("4. Evaluation and results")
    R.h2("4.1 Evaluation data and metrics")
    R.para(
        "**Evaluation data.** All three sets are native-written and cover all routes, incomplete "
        "variants and in-domain out-of-scope requests; "
        "most concern project management (§1). Test intervals span about "
        "±0.09 STS, so the new-writer sets carry the conclusions. "
        "**Training data** is relevant (same tools, routes and contract) but "
        f"inadequate in quality and coverage: "
        f"{share(ITEMS['train'], 'code_switched'):.0%} are code-switched against "
        f"{share(ITEMS['test'], 'code_switched'):.0%} of the test and "
        f"{share(NAT_CONF, 'code_switched'):.0%} of the natural set; "
        f"{reminder_share('train', 'فكرني'):.0%} of training reminders use «فكرني» and "
        f"{reminder_share('train', 'تذكير'):.0%} «تذكير», against "
        f"{reminder_share('test', 'فكرني'):.0%} and {reminder_share('test', 'تذكير'):.0%} in the test. "
        "**Separation.** Splits follow scenario families under a hash-locked manifest; vocabularies "
        "and hyper-parameters come from train/dev only; each set was scored once, after its systems were "
        "frozen. **Limitations.** The test is one writer's templated phrasing "
        f"(52 sentence frames; {frames['frame_seen_in_train']} items share a frame with train) by the "
        "author of S1's lexicon; natural and v3 test new writers, not new scenarios. C was tuned "
        f"on {len(ITEMS['dev'])} mostly drafted dev items; grouped CV on train picks a larger C, still below "
        "S1 on the test.")
    R.para(
        "**Metrics.** Besides STS: intent and route macro-F1 and the safety measures defined under Table 2. Intervals are family-level bootstraps (10,000 "
        "resamples) [8]; paired tests permute families [7], Holm-corrected [9]. **Pre-registered "
        "hypotheses.** Natural set: rules beat TF-IDF; core v1.1 helps; S9 beats S8. v3: human training "
        "data help TF-IDF and MARBERTv2.")

    R.h2("4.2 Results")
    h1_, h2_ = cmp_["S2 vs S1"], cmp_["S3 vs S1"]
    R.para(
        "**Finding 1: rules beat drafted-data classifiers.** On new writers (Table 2) "
        f"rules lead TF-IDF by {signed(N1['difference'])} {ci(N1['ci'])} STS (Holm p = "
        f"{N1['p_holm']:.4f}; 13 of 17 writers; Table 7). The "
        f"author's test agrees (Holm p ≤ {max(h1_['p_holm'], h2_['p_holm']):.3f}) but alone is "
        "inconclusive (Holm p 0.055-0.526 clustering by sentence frame, Table 6). "
        "TF-IDF's extra test failures are all intent errors (Tables 10, 11).")
    # Tables 2 (locked test) and natural set share one table: same columns, one caption and header
    def main_row(name: str, m: dict, c: dict) -> list[str]:
        r = m["success_by_gold_route"]
        return [name, f"{m['safe_task_success']:.3f} {ci(c)}", f3(m["intent_macro_f1"]),
                f3(m["route_macro_f1"]), f3(m["unsafe_execution_rate"]), f3(1 - m["execution_precision"]),
                f3(m["unnecessary_clarification_rate"]), f3(m["pair_both_success"]),
                f"{r['ACT']:.2f}/{r['CLARIFY']:.2f}/{r['ABSTAIN']:.2f}"]

    systems4 = (("oracle", "oracle"), ("S1 rules", "S1"), ("S2 TF-IDF", "S2"), ("S3 + char", "S3"))
    main_rows = [[f"Locked test: the author, {t1['n']} items, {t1['pairs']} pairs (core v1.0)"] + [""] * 8]
    main_rows += [main_row(n, test_metrics(k), TEST["systems"][k]["ci"]["safe_task_success"]) for n, k in systems4]
    main_rows += [[f"Natural set: {n_writers} new writers, {n_conf} items, {n1['pairs']} pairs"] + [""] * 8]
    main_rows += [main_row(n, nat_metrics(k), NAT["systems"][k]["ci"]["safe_task_success"]) for n, k in systems4]
    require(abs(wrong_n / exec_n - (1 - n1["execution_precision"])) < 1e-9
            and wrong_n - harm_n == lenient_items, "S1 natural executions: wrong = harmful + wording only")
    require(round((nat_or11 - no["safe_task_success"]) * n_conf) == 1
            and round((NAT["summary"]["S1-v1.1"]["sts"] - n1["safe_task_success"]) * n_conf) == 1,
            "core v1.1 changed a single natural item")
    R.para(
        "**Other pre-registered tests.** Core v1.1 changed one natural item (oracle "
        f"{f3(no['safe_task_success'])} → {nat_or11:.3f}). S9 beat S8 by {signed(N3['difference'])} (Holm p = "
        f"{N3['p_holm']:.3f}), mostly as a fallback: in {s9_fallback} of the {s9_gain} item gains S1 answered "
        "where S8 made no valid call. "
        f"**Safety.** {harm_n} of S1's {exec_n} natural calls were harmful (wrong tool or time, or should "
        f"not run), {lenient_items} more had wrong wording (test: {harm_t} of {exec_t} harmful).")
    R.table("Main results. Unsafe: wrong or unneeded calls over all items (all) or calls made (ACT); Unn. "
            "clar.: gold ACT answered with CLARIFY; Pairs: both pair members correct; last: success by gold route.",
            ["System", "STS [95% CI]", "Intent F1", "Route F1", "Unsafe (all)", "Unsafe (ACT)",
             "Unn. clar.", "Pairs", "ACT/CLAR/ ABST"], main_rows,
            widths=[2.55, 3.25, 1.2, 1.2, 1.3, 1.3, 1.15, 1.15, 2.9], bold_rows={0, 2, 5, 7})
    tbl = R.doc.tables[-1]
    for ri in (1, 6):  # the set rows span the whole table
        tbl.cell(ri, 0).merge(tbl.cell(ri, 8))

    s3h = HUM["tests"]["S3 +contributors vs S1-v1.1"]
    s3h = {**s3h, "difference": -s3h["difference"],   # report as rules minus S3, the gap to close
           "ci": {"low": -s3h["ci"]["high"], "high": -s3h["ci"]["low"]}}
    def v3row(label: str, data: str, key: str) -> list[str]:
        s = v3s[key]
        r = s["by_route"]
        return [label, data, f"{s['sts']:.3f} {ci(s['ci'])}",
                f"{r['ACT']:.2f} / {r['CLARIFY']:.2f} / {r['ABSTAIN']:.2f}"]

    R.table(f"Registered v3 test ({v3_n} items, 17 further writers). Human data: natural-set messages "
            "added to training.",
            ["System", "Training data", "STS [95% CI]", "Success ACT / CLAR / ABST"],
            [v3row("oracle (gold intent)", "–", "oracle-v1.1"),
             v3row("S1 keyword rules", "none", "S1-v1.1"),
             v3row("S3 TF-IDF + char", "drafted + author", "S3-drafted"),
             v3row("S3 TF-IDF + char", "+ human data", "S3-human"),
             v3row("S6 MARBERTv2", "drafted + author", "S6-drafted (seed mean)"),
             v3row("S6 MARBERTv2", "+ human data", "S6-human (seed mean)")],
            widths=[3.4, 3.0, 4.1, 5.5], bold_rows={3, 5})
    R.para(
        "**Finding 2: in-domain human data close the gap.** On v3, adding human "
        f"messages raised S3 by {signed(v3a['difference'])} {ci(v3a['ci'])} (Holm p = "
        f"{v3a['p_holm']:.3f}) and MARBERTv2 by {signed(v3b['difference'])} {ci(v3b['ci'])} (Holm "
        f"p = {v3b['p_holm']:.4f}), mostly on out-of-scope requests. Not just more data: about "
        f"{n_human} human messages alone beat the {n_fit} drafted and author items "
        f"({signed(v3_ctrl['difference'])}, p = {v3_ctrl['p_value']:.3f}, descriptive). Retrained "
        f"S3 equals the rules ({signed(v3_s3r['difference'])} {ci(v3_s3r['ci'])}); retrained MARBERTv2 "
        f"exceeds them ({signed(v3_s6r['difference'])} {ci(v3_s6r['ci'])}, estimate only). "
        "**Limitation:** only the human data share v3's scenario cards, so this shows that "
        "in-distribution data help, not that authorship does.")
    require(all((1 - orc) / (1 - s1) > 0.5 for orc, s1 in (
        (o["safe_task_success"], t1["safe_task_success"]),
        (no["safe_task_success"], n1["safe_task_success"]),
        (v3_or11, v3_s111))), "most of S1's failures remain with gold intents on all three sets")
    # extractor accuracy with gold intents on the natural set's complete requests (sensitivity.md)
    or_act = [round(float(x)) for x in re.search(r"^\| oracle \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$",
                                                 sens_md, re.M).groups()]  # wrong tool, instant, text only, not executed
    n_act = sum(i["route"] == "ACT" for i in NAT_CONF)
    or_act_ok = round(no["success_by_gold_route"]["ACT"] * n_act)
    require(or_act[0] == 0 and or_act_ok + sum(or_act) == n_act,
            "oracle on gold-ACT natural items: full successes + wording + instant + asked = all")
    ca_or = CA["systems"]["oracle"]
    require(all(CA["validation"].values()) and CA["n"] == n_conf
            and CA["items"]["agreed"] + CA["items"]["disagreed"] == n_conf
            and ca_or["failures"] == round((1 - no["safe_task_success"]) * n_conf)
            and ca_or["agreed"]["failures"] + ca_or["disagreed"]["failures"] == ca_or["failures"]
            and ca_or["agreed"]["failures"] > ca_or["failures"] / 2
            and CA["both_act_datetime"]["n"] == AGREE_NAT["datetime_same_instant"]["n"]
            and abs(CA["both_act_datetime"]["agreement_as_written"]
                    - AGREE_NAT["datetime_same_instant"]["agreement"]) < 1e-9,
            "ceiling split: validated against the adjudication; most oracle failures on agreed items")
    R.para(
        "**Finding 3: the extractor sets the ceiling.** With gold intents "
        f"STS is {f3(no['safe_task_success'])} (natural) and "
        f"{v3_or11:.3f} (v3). Of {n_act} complete "
        f"natural requests the extractor gets {or_act_ok} fully right; {or_act[2]} fail on wording alone, "
        f"{or_act[1]} on the instant, {or_act[3]} are re-asked. Label ambiguity explains little: "
        f"{ca_or['agreed']['failures']} of the oracle's {ca_or['failures']} natural failures fall on the "
        f"{ca_or['agreed']['n']} items both annotators labelled identically. Four post hoc rule fixes "
        "raised every system on unseen v3 by "
        f"{c12_lo:.3f}-{c12_hi:.3f} (core v1.2, Table 12). A learned "
        "word tagger (logistic regression, no word lists, Table 13) trained on "
        f"{lt_hv['training']['messages']} natural-set reminders got "
        f"{lt_hv['text_learned']['correct']} of v3's {lt_hv['text_learned']['n']} complete reminders' text right "
        f"(rules: {lt_hv['text_rules']['correct']}), raising the v3 oracle by "
        f"{signed(lt_hv['difference'])} {ci(lt_hv['ci'])} (p = {lt_hv['p_value']:.3f}, exploratory); "
        f"trained on drafted data, {signed(lt_dv['difference'])}.")

    # 5 ------------------------------------------------------------------------------------------
    R.h1("5. Conclusion")
    R.para(
        f"We met O1-O5 with a {n_all}-item benchmark and tests by {n_writers + len(V3['writers'])} other "
        "writers. **RQ1:** no; trained on mostly AI-drafted data, TF-IDF lost to the rules by "
        f"{N1['difference']:.3f} STS. **RQ2:** yes; in-domain human data closed the gap (in-distribution "
        "data; authorship not isolated). The extractor, not label ambiguity, limits every "
        "system; learning it from in-domain data is the next step.")

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
        fmt(p)
        add_runs(p, ref)

    # Appendices -----------------------------------------------------------------------------------
    R.h1("Appendix A. Additional results")
    R.page_break_before_last()
    s7t, s8t, s7n = s7_test(), s8_test(), s7_nat()
    s7_file, s8_file = ("results/s4/7cd-s7-s8/s4_results/s7_jais2_qlora_s0_test.json",
                         "results/s4/7cd-s7-s8/s4_results/s8_jais3-7b.json")
    s4p = {m: precision(f"results/s4/{d}/s4_results/{m}.json") for m, d in S4_DIRS.items()}
    ext = [
        ["S1 keyword rules (reference)", "CPU", f3(t1["safe_task_success"]), f3(n1["safe_task_success"])],
        ["S4 Jais 2 8B (6-shot)", s4p["jais2-8b"], f3(s4_test("jais2-8b")), f3(s4_nat("jais2-8b"))],
        ["S4 Jais-3-7B (6-shot)", s4p["jais3-7b"], f3(s4_test("jais3-7b")), f3(s4_nat("jais3-7b"))],
        ["S4 Jais-3-3.7B (6-shot)", s4p["jais3-3.7b"], f3(s4_test("jais3-3.7b")), f3(s4_nat("jais3-3.7b"))],
        ["S4 Nile-Chat 4B (6-shot)", s4p["nile-4b"], f3(s4_test("nile-4b")), f3(s4_nat("nile-4b"))],
        ["S4 Qwen3 8B (6-shot)", s4p["qwen3-8b"], f3(s4_test("qwen3-8b")), f3(s4_nat("qwen3-8b"))],
        ["S5 multilingual-e5-small + LR", "CPU", f3(ENC["systems"]["S5"]["locked_test_core_v1"]["sts"]),
         f3(ENC["systems"]["S5"]["natural_core_v1"]["sts"])],
        ["S6 MARBERTv2 fine-tuned (5 seeds)", "CPU",
         f"{s6['locked_test_core_v1']:.3f} ± {ENC['s6_seed_sd']['locked_test_core_v1']:.3f}",
         f"{s6['natural_core_v1']:.3f} ± {ENC['s6_seed_sd']['natural_core_v1']:.3f}"],
        ["S7 Jais 2 + QLoRA (3 seeds)", precision(s7_file),
         f"{st.mean(s7t):.3f} (" + " / ".join(f3(x) for x in s7t) + ")",
         f"{S7_NAT:.3f} (" + " / ".join(f3(x) for x in s7n) + ")"],
        ["S8 Jais-3-7B tool calling (3 seeds)", precision(s8_file),
         f"{st.mean(s8t):.3f} (" + " / ".join(f3(x) for x in s8t) + ")", f"{s8m['sts']:.3f}"],
        ["S9 S8 + deterministic policy (3 seeds)", precision(s8_file), "not run¹", f"{s9m['sts']:.3f}"],
    ]
    ext = [row + ["–"] for row in ext]
    ext[0][-1] = f"{v3s['S1-v1.1']['sts']:.3f}"
    gs = V3G["summary"]
    v3_prec = load("models_v3/freeze_gpu.json")["precision"]
    ext += [["S4 Jais-3-7B 6-shot inline, normalized / raw text", v3_prec, "–", "–",
             f"{gs['S4-jais3-7b-inline-norm']['sts']:.3f} / {gs['S4-jais3-7b-inline-raw']['sts']:.3f}"],
            ["S8 Jais-3-7B greedy, with exemplars (1 run)", v3_prec, "–", "–", f"{gs['S8-fair']['sts']:.3f}"],
            ["S9 from that S8", v3_prec, "–", "–", f"{gs['S9-fair']['sts']:.3f}"]]
    require(abs(gs["S1-v1.1"]["sts"] - v3s["S1-v1.1"]["sts"]) < 1e-9, "same S1 v1.1 score in both v3 result files")
    R.table("Safe Task Success of the extension systems on the three locked sets, with the numeric format "
            "each GPU model ran in, or CPU (S4: primary setting, 6 examples with prior calibration; S5/S6 fitted after "
            "both evaluations, exploratory; the registered v3 S6 runs of Table 3 were trained on a Kaggle T4). "
            "¹S9 uses core v1.1, whose fix was drawn from the test errors. "
            "v3: core v1.1; the GPU rows were frozen before labelling (amendments A5, A7), descriptive.",
            ["System", "Format", "Locked test (120)", "Natural (238)", f"v3 ({v3_n})"], ext,
            widths=[5.0, 1.7, 3.4, 3.4, 2.5])
    j3 = load("results/s4/7a-jais3/s4_results/jais3-7b.json")["systems"]
    j3_inline = j3["S4-jais3-7b-6shot-inline-prior"]["metrics"]["safe_task_success"]
    j3_declined = FAILS["S4-jais3-7b-6shot-prior"]["declined an in-scope request"]
    s8_invalid = [load(s8_file)["systems"][f"S8-jais3-7b-s{i}"]["metrics"]["invalid_outputs"] for i in range(3)]
    R.para(
        "**Systems S4-S7.** They replace only the intent recognizer; the extractor and policy are "
        "shared. **S4** prompts an LLM for the intent with 6 examples (Jais 2 8B [5, 11], Jais-3-7B, "
        "Jais-3-3.7B [12], Nile-Chat 4B [13], Qwen3 8B [14]). **S5** is logistic regression on frozen "
        "multilingual-e5-small embeddings. **S6** is fine-tuned MARBERTv2 [2] (five seeds, recipe after "
        "[10]). **S7** is Jais 2 with QLoRA [3] (three seeds). No LLM recognizer beat the rules "
        "significantly on any set (Tables 4, 8).")
    R.para(
        "**Precision.** Kaggle's T4 GPUs have no bf16, so the models ran in different formats: Jais 2 "
        "overflowed in fp16 and ran 4-bit, Jais-3-7B and Qwen3 ran in fp16, the smaller models in fp32. "
        "Differences between these models may therefore partly reflect precision. **Jais-3-7B** was "
        "released after the freeze but before the test was opened; Jais 2 stayed the pre-registered "
        "LLM. In its "
        f"primary setting it declined {j3_declined} in-scope test requests; with the exemplars listed in "
        f"the user message (a secondary analysis declared before the test was opened) it scored "
        f"{j3_inline:.3f}. As S8 it made no valid call on " + ", ".join(map(str, s8_invalid[:-1]))
        + f" and {s8_invalid[-1]} of the 120 test items.")
    R.para(
        f"**Protocol v3** (`docs/protocol_v3.md`, amendments A1-A9). Registered on 6 November before "
        f"any v3 system existed; systems and GPU outputs frozen before labelling; gold locked "
        f"(manifest `{V3_MAN['manifest_sha256'][:8]}`) and scored once (`results/v3/evaluation.json`, "
        "`evaluation_gpu.json`). Both registered human-data gains also hold with writers as clusters and in all 17 "
        "leave-one-writer-out analyses; the exploratory natural-set analysis agreed (Table 5). "
        "GPU systems (descriptive): raw instead of normalized text did not change S4 "
        f"({signed(g_cmp['S4 raw vs normalized text']['difference'])}, p = "
        f"{g_cmp['S4 raw vs normalized text']['p_value']:.2f}; same intent on "
        f"{V3G['s4_norm_raw_same_intent']} of {v3_n}); the policy added "
        f"{signed(g_cmp['S9-fair vs S8-fair']['difference'])} to greedy Jais 3 with exemplars "
        f"(p = {g_cmp['S9-fair vs S8-fair']['p_value']:.2f}). Greedy S8 made no valid "
        f"call on {V3G['s8_fair_no_valid_call']} of {v3_n} items (sampled S8 on the natural set: "
        f"{min(s8_invalid_nat)}-{max(s8_invalid_nat)} of {n_conf} a seed).")
    R.para(
        "**Cost.** On a laptop CPU, each final S2/S3 fit takes under 0.3 s, inference under 0.3 ms per "
        "item, and peak memory is 141 MB (`models/freeze.json`).")
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
    hrows.append(["S1 rules (reference)", "", "", f"{hum['S1-v1.1']['sts']:.3f} ({hum['S1-v1.1']['intent_accuracy']:.3f})"])
    hrows = [[f"{r[0]}, natural" if r[0] != "S1 rules (reference)" else "S1 rules, natural", *r[1:]] for r in hrows]
    vm = V3["metrics"]

    def v3cell(name: str) -> str:
        if name.startswith("S6"):
            acc = st.mean(vm[f"{name}-s{s}"]["intent_accuracy"] for s in range(5))
            return f"{v3s[name + ' (seed mean)']['sts']:.3f} ({acc:.3f})"
        return f"{v3s[name]['sts']:.3f} ({vm[name]['intent_accuracy']:.3f})"

    hrows += [["**S3, v3 (registered)**", v3cell("S3-drafted"), v3cell("S3-contrib-only"), v3cell("S3-human")],
              ["**S6 MARBERTv2, v3 (5 seeds)**", v3cell("S6-drafted"), "", v3cell("S6-human")],
              ["S1 rules, v3", "", "", v3cell("S1-v1.1")]]
    R.table("Training on human-written messages: STS (intent accuracy). Natural set: exploratory "
            "leave-one-writer-out (each held-out writer's scenarios removed; about "
            f"{n_human} contributor messages per fold). v3: registered test on {v3_n} unseen messages, "
            "trained per scenario block on the natural set minus that block's cards (registered for S3 and S6). "
            "S2/S3 use grouped-CV settings; S1 is core v1.1.",
            ["Classifier, set", f"Drafted + author ({n_fit})", f"Contributors only (~{n_human})", "Both"],
            hrows, widths=[4.3, 3.9, 3.9, 3.9])
    tsv = csv_rows("results/posthoc/template_sensitivity.csv")
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
            ["Labels / rule", "n", "Rules − TF-IDF (S1 − S2)", "Policy gain (S9 − S8)"], vrows,
            widths=[7.4, 1.2, 3.7, 3.7])
    lrows = []
    for key in lv_n["comparisons"]:
        t_, n_ = lv_t["comparisons"][key], lv_n["comparisons"][key]
        lrows.append([key.removesuffix(" vs S1"),
                      f"{signed(t_['difference'])} {ci(t_['ci'])}", f"{t_['p_holm']:.3f}",
                      f"{signed(n_['difference'])} {ci(n_['ci'])}", f"{n_['p_holm']:.3f}",
                      f"{n_['unsafe_execution_rate']:.3f}"])
    lrows.append(["S1 keyword rules (reference)", f"STS {lv_t['S1']['sts']:.3f}", "",
                  f"STS {lv_n['S1']['sts']:.3f}", "", f"{lv_n['S1']['unsafe_execution_rate']:.3f}"])
    R.table("Post hoc: LLM intent recognizers minus S1 in STS, same pipeline and stored outputs, "
            "family-level paired test and bootstrap as pre-registered, Holm over the six comparisons "
            "of each set (`scripts/posthoc_llm_vs_rules.py`). S7: per-item mean over three seeds. "
            "Last column: unsafe execution rate on the natural set.",
            ["Recognizer", "Test: Δ [95% CI]", "Holm p", "Natural: Δ [95% CI]", "Holm p", "Unsafe"],
            lrows, widths=[3.9, 3.9, 1.4, 3.9, 1.4, 1.5])

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
    fails = FAILS
    cats = ["wrong reminder text only", "wrong tool", "executed an incomplete request",
            "asked although complete", "asked for the wrong slots", "clarified with the wrong intent",
            "asked instead of declining", "declined an in-scope request"]
    frows = [[c[0].upper() + c[1:]] + [fails[s][c] for s in ("oracle", "S1", "S2", "S3")] for c in cats]
    frows.append(["All failures"] + [fails[s]["failures"] for s in ("oracle", "S1", "S2", "S3")])
    R.table("Locked test: failures by type (counts of 120 items; from results/error_analysis). S2's "
            "20 failures beyond S1's are all intent errors.",
            ["Failure type", "Oracle", "S1", "S2", "S3"], frows, widths=[7.6, 2.1, 2.1, 2.1, 2.1],
            bold_rows={len(frows) - 1})

    nat_cs = {i["id"] for i in NAT_ITEMS if i.get("code_switched")}
    crow = []
    for name, key in (("oracle (gold intent)", "oracle"), ("S1 keyword rules", "S1"),
                      ("S2 TF-IDF + LR", "S2"), ("S3 + char n-grams", "S3")):
        pc = test_metrics(key)["intent_per_class"]
        cs_t = TEST["systems"][key]["subgroups"]["code_switching"]["code-switched"]
        cs_n = [i["success"] for i in NAT["systems"][key]["items"] if i["id"] in nat_cs]
        crow.append([name] + [f"{pc[c]['precision']:.2f} / {pc[c]['recall']:.2f}"
                              for c in ("create_reminder", "create_alarm", "unsupported")]
                    + [f"{cs_t['safe_task_success']['estimate']:.3f}", f"{sum(cs_n) / len(cs_n):.3f}"])
    cm = test_metrics("S2")["intent_confusion"]["matrix"]
    R.table("Locked test: intent precision / recall per class, and STS on code-switched items "
            f"(test n = {cs_t['n']}, natural n = {len(cs_n)}; exploratory, small subgroups). S2 labelled "
            f"the {sum(cm[0])} gold reminders as {cm[0][0]} reminder, {cm[0][1]} alarm, {cm[0][2]} unsupported.",
            ["System", "Reminder P / R", "Alarm P / R", "Unsupported P / R", "CS test", "CS natural"],
            crow, widths=[3.4, 2.6, 2.6, 3.4, 1.8, 2.2])

    c12rows = []
    for part, label in (("natural (development)", "natural (dev.)"), ("v3 (held-out)", "v3 (held-out)")):
        for name, r in C12[part].items():
            name = name.replace(" MARBERTv2", "").replace(" data", "")
            c12rows.append([label, name, f"{r['sts_v1.1']:.3f}", f"{r['sts_v1.2']:.3f}",
                            f"{signed(r['difference'])} {ci(r['ci'])}",
                            f"{r['p_value']:.3f}" if r["p_value"] >= 0.001 else "< 0.001",
                            f"{len(r['items_fixed'])} / {len(r['items_broken'])}"])
    c12t = C12["locked test"]["oracle"]
    R.table("Post hoc core v1.2 (`src/athar/v2/core12.py`, `scripts/posthoc_core_v12.py`): STS of each "
            "system with its frozen intents through core v1.1 and v1.2. The fixes were written from the "
            "natural-set oracle failures (in-sample there); v3 was not inspected meanwhile. Paired family-level "
            "permutation test and bootstrap, unadjusted p. One natural item breaks because its gold keeps "
            f"«ضروري» inside the text, unlike the others; the two broken v3 items were not examined, to "
            f"keep v3 unread. Locked test: no change ({c12t['sts_v1.1']:.3f} "
            "with core v1.1, whose fixes came from its errors).",
            ["Set", "System", "Core v1.1", "Core v1.2", "Δ [95% CI]", "p", "Fixed / broken"],
            c12rows, widths=[2.4, 3.9, 1.5, 1.5, 3.7, 1.4, 1.6])

    ltrows = []
    for key, label in (("drafted + author -> natural", "drafted + author → natural"),
                       ("drafted + author -> v3", "drafted + author → v3"),
                       ("natural (human) -> v3", "natural (human) → v3")):
        r = lt[key]
        ltrows.append([label, str(r["training"]["messages"]),
                       f"{r['text_rules']['correct']} / {r['text_learned']['correct']} of {r['text_rules']['n']}",
                       f"{r['oracle_rules_sts']:.3f} → {r['oracle_learned_sts']:.3f}",
                       f"{signed(r['difference'])} {ci(r['ci'])}",
                       f"{r['p_value']:.3f}", f"{r['items_gained']} / {r['items_lost']}"])
    R.table("Post hoc: learned reminder-text tagger (`scripts/posthoc_learned_text.py`) in place of the "
            "hand-written framing and filler lists, with gold intents; times are still resolved by core "
            "v1.1. Per-word logistic regression (C = 1, threshold 0.5, not tuned) on the word, its "
            "neighbours, affixes, position and the time parser's spans; trained on the reminder messages "
            "of each training set. Text: gold-ACT reminders with the right reminder text, rules / learned. "
            "Paired family-level test against the rule extractor, unadjusted p. All v3 scenario cards also "
            "appear in the natural set (as in Finding 2). Adding core v1.1's word lists as a feature and "
            "output filter changed no score.",
            ["Training → set", "Train msgs", "Text right", "Oracle STS", "Δ [95% CI]", "p", "Gained / lost"],
            ltrows, widths=[4.4, 1.3, 2.0, 2.4, 3.4, 1.1, 1.4])

    R.h1("Appendix C. Reproducing the results")
    R.para("All commands run from the repository root on a CPU; T2 stands for Table 2. The five "
           "analysis scripts (T2-T10) rebuild their tables from the stored outputs and reproduce the "
           "committed files exactly (Table 3 is read from `results/v3/evaluation.json`; Table 7 also needs the private labelling sheets in "
           "`data/raw/private/`, which are not shared); the encoder commands retrain S5/S6 (slow on "
           "CPU). The GPU systems (S4, S7, S8) were run, and S7 was trained, with "
           "`notebooks/kaggle_s4_llm.ipynb`.")
    R.code([
        "uv sync",
        "uv run pytest                                               # tests",
        "uv run python scripts/error_analysis.py --out results/error_analysis   # T2, T10",
        "uv run python scripts/natural_tables.py                     # T2",
        "uv run python scripts/posthoc_analysis.py --out results/posthoc        # T6",
        "uv run python scripts/posthoc_llm_vs_rules.py               # T8",
        "uv run --with openpyxl python scripts/natural_sensitivity.py           # T7",
        "uv run --extra neural python scripts/posthoc_encoders_human.py encoders # T4",
        "uv run --extra neural python scripts/posthoc_encoders_human.py human   # T5",
        "uv run python scripts/posthoc_core_v12.py   # T12 (needs the private v3 texts)",
        "uv run python scripts/posthoc_learned_text.py  # T13 (needs the private v3 texts)",
        "uv run --with openpyxl python scripts/posthoc_ceiling_ambiguity.py     # 4.2 (private sheets)",
        "uv run --extra ui streamlit run app/streamlit_app.py        # demo",
    ])
    R.table("Pretrained models used (linked), with the pinned Hugging Face revisions.",
            ["System", "Hugging Face model", "Revision"],
            [["S4, S7", "[[https://huggingface.co/inception42/Jais-2-8B-Chat|inception42/Jais-2-8B-Chat]]", "da0e1639cd92b508b24120f7f77f5270a8465dc4"],
             ["S4, S8, S9", "[[https://huggingface.co/inception42/Jais-3-7B|inception42/Jais-3-7B]]", "94d69f49b4af702d42fcee122cc06e493a6fe7e1"],
             ["S4", "[[https://huggingface.co/inception42/Jais-3-3.7B|inception42/Jais-3-3.7B]]", "3d5754d5db87cd3a59f0f5b09c7ac39b351ff5ed"],
             ["S4", "[[https://huggingface.co/MBZUAI-Paris/Nile-Chat-4B|MBZUAI-Paris/Nile-Chat-4B]]", "2a365667659bbd6786c2760d868e37c506d76ce3"],
             ["S4", "[[https://huggingface.co/Qwen/Qwen3-8B|Qwen/Qwen3-8B]]", "b968826d9c46dd6066d109eabc6255188de91218"],
             ["S5", "[[https://huggingface.co/intfloat/multilingual-e5-small|intfloat/multilingual-e5-small]]", "614241f622f53c4eeff9890bdc4f31cfecc418b3"],
             ["S6", "[[https://huggingface.co/UBC-NLP/MARBERTv2|UBC-NLP/MARBERTv2]]", "fe88db9db8ccdb0c4e1627495f405c44a5f89066"]],
            widths=[2.2, 5.6, 8.2])

    R.h1("Appendix D. Use of generative AI")
    R.para(
        f"The {n_drafted} AI-drafted training and development requests were generated with an AI "
        "assistant and "
        "are marked `source: assistant_draft`; none is in either evaluation set. The code, tests and "
        "documentation were developed with Claude Code as a programming assistant. During adjudication "
        "of the natural set, the assistant checked the author's final labels against the written "
        "guide and flagged 24 that broke a rule; the author approved each correction. The core v1.2 "
        "fixes were written with the assistant from the natural-set oracle failures, without reading "
        "the v3 texts; the v3 scores were computed once and only in aggregate. The learned "
        "reminder-text tagger (Table 13) was written with the assistant with its settings fixed in "
        "advance and nothing tuned on an evaluation set; only aggregates were read on v3. No message was "
        "labelled from scratch by an AI system. The author designed the study, wrote the test requests "
        "and their labels, recruited the contributors, ran the experiments and is responsible for every "
        "claim in this report.")
    return R


def export_text(doc, path: Path) -> list[tuple[str, str]]:
    """Write the report as plain text (paragraphs and table rows in order) so that `git diff` shows
    every word a revision changed. Returns (kind, text) blocks in document order."""
    blocks: list[tuple[str, str]] = []
    for el in doc.element.body.iterchildren():
        if el.tag == qn("w:p"):
            text = "".join(t.text or "" for t in el.iter(qn("w:t"))).strip()
            if text:
                style = el.find(f"{qn('w:pPr')}/{qn('w:pStyle')}")
                heading = style is not None and style.get(qn("w:val"), "").startswith("Heading")
                blocks.append(("heading" if heading else "para", text))
        elif el.tag == qn("w:tbl"):
            for tr in el.iter(qn("w:tr")):
                cells = ["".join(t.text or "" for t in tc.iter(qn("w:t"))).strip() for tc in tr.iter(qn("w:tc"))]
                blocks.append(("row", " | ".join(cells)))
    lines = []
    for kind, text in blocks:
        lines.append(f"\n## {text}\n" if kind == "heading" else (f"| {text} |" if kind == "row" else f"\n{text}\n"))
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return blocks


def _numbers_in_files() -> set[str]:
    """Every number stored in the committed result files, written with 2, 3 and 4 decimals."""
    found: set[str] = set()

    def add(x: float) -> None:
        found.add(f"{abs(x):g}")
        for d in (2, 3, 4):
            found.add(f"{abs(x):.{d}f}")

    def walk(v: object) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            add(float(v))
        elif isinstance(v, str) and re.fullmatch(r"-?\d+(\.\d+)?", v):
            add(float(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    files = [*ROOT.glob("results/**/*.json"), ROOT / "docs/agreement.json", ROOT / "models/freeze.json"]
    for f in files:
        try:
            walk(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    for f in [*ROOT.glob("results/**/*.csv"), *ROOT.glob("results/**/*.md")]:
        for m in re.finditer(r"-?\d+\.\d+", f.read_text(encoding="utf-8", errors="ignore")):
            add(float(m.group()))
    return found


# decimals typed in the main text that are not stored in a result file; each one has a require()
AUDIT_ALLOWED = {
    "0.055", "0.526",   # Holm p range under frame clustering (require: "Holm p under frame clustering")
    "0.09",             # test interval half-width (require: "test intervals of about ±0.09")
    "3.11",             # Python version
}


def audit_numbers(blocks: list[tuple[str, str]]) -> None:
    """Stop the build if a decimal in the main report (abstract to conclusion) cannot be traced to a
    committed result file or to a table of the report."""
    stored = _numbers_in_files()
    table_text = " ".join(t for k, t in blocks if k == "row")
    stored |= {m.group() for m in re.finditer(r"\d+\.\d+", table_text)}
    main: list[str] = []
    for kind, text in blocks:
        if kind == "heading" and text == "References":
            break
        if kind == "para":
            main.append(text)
    untraced = sorted({m.group() for t in main for m in re.finditer(r"(?<![\w.§])\d+\.\d+(?![\w.])", t)}
                      - stored - AUDIT_ALLOWED)
    if untraced:
        sys.exit("number audit: these decimals in the main report are not in any result file or table: "
                 + ", ".join(untraced))
    print("number audit: every decimal in the main report is traceable")


def to_pdf(docx: Path, pdf: Path) -> None:
    ps = (f"$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
          f"$d = $w.Documents.Open('{docx}'); $d.SaveAs([ref]'{pdf}', [ref]17); "
          f"$p = $d.ComputeStatistics(2); "
          f"$r = $d.Content; $f = $r.Find; $f.MatchCase = $true; $null = $f.Execute('References'); "
          f"$r.Move(1, -2) | Out-Null; "  # the last character of the Conclusion
          f"Write-Output \"pages=$p main_report_ends_on_page=$($r.Information(3))\"; "
          f"$d.Close([ref]0); $w.Quit()")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())
    end = re.search(r"main_report_ends_on_page=(\d+)", out.stdout)
    if end and int(end.group(1)) > MAIN_PAGE_LIMIT:
        sys.exit(f"the main report ends on page {end.group(1)}; the limit is {MAIN_PAGE_LIMIT} pages")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", action="store_true", help="also export a PDF with Microsoft Word")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT, help="where to write (default: next to this script)")
    a = ap.parse_args()
    dirty = bool(git("status", "--porcelain"))
    if dirty and not a.allow_dirty:
        sys.exit("working tree has uncommitted changes; commit first so the printed hash matches")
    commit, branch = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    docx = a.out_dir.resolve() / "ATHAR_Chat_Final_Report.docx"
    report = build(commit, branch)
    blocks = export_text(report.doc, a.out_dir.resolve() / "report_text.md")
    audit_numbers(blocks)
    report.save(docx)
    print(f"wrote {docx} for commit {commit}")
    if dirty:
        print("DRAFT: uncommitted changes, so the printed commit does not hold these files; do not submit")
    if not git("branch", "-r", "--contains", commit):
        print(f"WARNING: commit {commit[:7]} is not on the remote yet; push before submitting")
    if a.pdf:
        to_pdf(docx, docx.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
