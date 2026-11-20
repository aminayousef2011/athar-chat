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
            and (ROOT / "models_v3/freeze_s3.json").exists() and (ROOT / "models_v3/freeze_s6.json").exists()
            and not any((ROOT / "results/v3").glob("evaluation*")),
            "protocol v3: systems frozen, not yet evaluated")
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
    # test -> natural change of each learned recognizer (the Table 4 values), discussed in section 4.3
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
            "post hoc STS values equal those of Table 4")
    nat_tables = (ROOT / "results/natural/tables.md").read_text(encoding="utf-8")
    for label, rec in (("S4 Jais-3-3.7B", lv_j33), (r"S7 Jais 2 \+ QLoRA \(3 seeds\)", lv_s7)):
        row = re.search(rf"^\| {label} \|[^|]*\|[^|]*\| ([0-9.]+) \|", nat_tables, re.M)
        require(row is not None and f"{rec['unsafe_execution_rate']:.3f}" == row.group(1),
                f"post hoc unsafe rate of {label} equals results/natural/tables.md")
    len_s1 = SENS["locked gold, lenient reminder text"]["systems"]["S1"]
    len_or = SENS["locked gold, lenient reminder text"]["systems"]["oracle"]
    lenient_items = round(act_fail[2])  # S1's text-only ACT failures on the locked gold
    require(lenient_items > 0 and abs(n1["safe_task_success"] + lenient_items / n_conf - len_s1) < 5e-4,
            "S1 natural items rescued by the lenient reminder-text rule")

    R.title("ATHAR Chat: Safe Tool Routing for Egyptian Arabic", "Amina Yousef")

    R.h1("Abstract")
    R.para(
        "An assistant that sets reminders and alarms from Egyptian Arabic messages must decide whether "
        "to execute, ask for missing details or decline (ACT, CLARIFY, ABSTAIN). With one shared "
        "rule-based extractor and execution policy, we compare keyword rules (S1) with TF-IDF logistic "
        "regression (S2, S3) by Safe Task Success (STS). In a pre-registered evaluation on "
        f"{n_conf} messages from {n_writers} other native speakers, the rules beat the classifier "
        f"({f3(n1['safe_task_success'])} vs {f3(n2['safe_task_success'])}, Holm p = "
        f"{tests['N1 S1 vs S2']['p_holm']:.4f}), as on the author's locked test. Letting the policy, "
        "with a rule fallback, decide when Jais-3-7B tool calls are executed raised STS from "
        f"{s8m['sts']:.3f} to {s9m['sts']:.3f}, mostly by replacing invalid calls. Exploratory "
        "analyses trace the classifiers' deficit to their mostly AI-drafted training data. No system "
        f"is deployable: the shared extractor caps all ({f3(no['safe_task_success'])} with gold intents).")

    # 1 ------------------------------------------------------------------------------------------
    R.h1("1. Motivation and objectives")
    R.para(
        "A reminder assistant must not execute a request whose time or content is missing or "
        "ambiguous, must not ask again for what the user already gave, and must decline what it "
        "cannot do. Egyptian Arabic makes "
        "this harder: messages are informal («اعمللي تذكير بعد بكرة الساعة 9 بالليل عشان اخد الدوا», "
        "\"set me a reminder the day after tomorrow at 9 pm to take my medicine\"), inconsistently "
        "spelled and mixed with English. Out-of-scope requests are hard "
        "for intent classifiers [6], and When2Call [1] showed for English that deciding *when* to call "
        "a tool is a problem of its own; we know of no Egyptian Arabic dataset labelling this decision. "
        "**Intended users:** teams building assistants for under-resourced dialects with little "
        "compute, choosing between rules, classifiers and LLMs, and researchers who need an Egyptian "
        "Arabic benchmark for when to call a tool.")
    R.para(
        "**Research question.** When all systems share the same argument extractor and execution policy, "
        "does lightweight supervised intent classification improve Safe Task Success over keyword rules? "
        "**Objectives:** (O1) a task contract and safety-aware metric; (O2) a reviewed Egyptian Arabic "
        "benchmark split by scenario; (O3) rules vs. TF-IDF classifiers in one pipeline; "
        "(O4) a test on writers other than the author; (O5) the errors limiting the pipeline.")

    # 2 ------------------------------------------------------------------------------------------
    R.h1("2. Deliverables")
    R.link_para("**Repository:** ", (REPO_URL, REPO_URL), f" (branch `{branch}`), commit ",
                (f"{REPO_URL}/tree/{commit}", commit), ".")
    blob = f"{REPO_URL}/blob/{commit}/"
    R.link_para("**Data:** ", (blob + "data/athar.jsonl", "data/athar.jsonl"),
                f" ({n_all} items; splits ", (blob + "data/splits/manifest.json", "data/splits/manifest.json"),
                "), ", (blob + "data/natural/natural.jsonl", "data/natural/natural.jsonl"),
                f" ({n_nat}), data card and guides (`docs/`); consent covers this project, no public "
                "licence yet.",
                align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    R.bullets([
        f"**Code, models, outputs:** package `athar` (pipeline, systems, evaluation) with {n_tests} tests and locked dependencies (`uv.lock`, Python 3.11); "
        "frozen classifiers (`models/`, `models_v2/`), stored predictions (`results/`), pinned Hugging "
        "Face models (Table 12), a Streamlit demo (`app/`) and the full technical report "
        "(`docs/report.md`); S6/S7 weights are not stored (Appendix C).",
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
        "ABSTAIN is declined, CLARIFY asks for exactly the gold slots, or ACT "
        "calls the gold tool at the gold instant with the gold reminder text (exact after "
        "normalization, request words and fillers removed); invalid outputs fail.")

    R.h2("3.2 Data")
    nat_fams = nat_confirmatory_families()
    R.table(
        "Datasets. No scenario family crosses train, dev and test; the natural set reuses the test's "
        "scenarios.",
        ["Set", "Items", "Families", "ACT/CLAR/ABST", "Written by", "Use"],
        [["train", *split_row("train"),
          f"{drafted('train')} AI-drafted, {len(ITEMS['train']) - drafted('train')} author", "fitting"],
         ["dev", *split_row("dev"),
          f"{drafted('dev')} AI-drafted, {len(ITEMS['dev']) - drafted('dev')} author", "selection"],
         ["test (locked)", *split_row("test"), "author (native speaker)", "evaluation 1"],
         ["natural (locked)", str(n_conf), str(nat_fams),
          " / ".join(str(sum(i["route"] == r for i in NAT_CONF)) for r in ("ACT", "CLARIFY", "ABSTAIN")),
          f"{n_writers} other native speakers", "evaluation 2"]],
        widths=[2.7, 1.3, 1.6, 3.2, 4.4, 2.8])
    R.para(
        "The author wrote the test from scenario cards; a second native "
        f"speaker labelled {AGREE_TEST['n_compared']} items "
        f"independently (intent κ = {AGREE_TEST['intent']['kappa']:.3f}, route κ = "
        f"{AGREE_TEST['route']['kappa']:.3f}). The natural set was collected after the "
        "systems were frozen: contributors wrote, without AI tools, a complete and an incomplete message "
        "for each test scenario. Screening out copies left 474 messages by 34 writers; a seeded draw of two "
        f"writers per scenario group gave {n_nat} messages from {n_writers} writers, {n_conf} "
        "in families absent from train and dev. Two annotators labelled all of them (intent κ = "
        f"{AGREE_NAT['intent']['kappa']:.3f}, route κ = {AGREE_NAT['route']['kappa']:.3f}). "
        "Slot-level agreement was lower (same slots on "
        f"{AGREE_NAT['targets_exact_set']['agreement']:.0%} of {AGREE_NAT['targets_exact_set']['n']} "
        f"items both labelled CLARIFY, same instant on {AGREE_NAT['datetime_same_instant']['agreement']:.0%} "
        f"of {AGREE_NAT['datetime_same_instant']['n']} both labelled ACT). The author adjudicated the "
        f"{load('results/natural/adjudication.json')['adjudicated']} disagreements, with no recorded "
        "joint discussion (Table 7 uses each annotator's own labels).")

    R.h2("3.3 Pipeline and systems")
    R.para(
        "**Shared pipeline.** The normalizer unifies alef, ya and taa marbuta variants, strips "
        "diacritics and elongation and maps Arabic-Indic digits. A pattern-based extractor parses the "
        "supported time expressions; the reminder text is what remains after request framing, time "
        "expressions and fillers are removed. The policy sends an unsupported intent to ABSTAIN, a "
        "supported one with all required slots to ACT, and otherwise to CLARIFY. A system therefore "
        "predicts only the intent, so STS differences between S1-S7 come from intent recognition alone.")
    R.para(
        "**Systems.** **S1:** hand-written trigger and out-of-scope lexicons, no training data "
        "(**S1 v1.1:** S1 with an extractor fix drawn from the test errors, §4.3). **S2:** word TF-IDF "
        "with class-balanced logistic regression [4]; **S3:** S2 plus character 2-4-grams for spelling "
        "variation. Both are fitted on train, with C and n-gram ranges chosen by dev intent macro-F1 "
        "(S2: C = 0.3; S3: C = 0.1). **Oracle:** gold intents in the "
        "same pipeline, its ceiling. **S4-S7** replace only the intent recognizer (Appendix A): "
        "few-shot LLMs (S4), an embedding classifier (S5), fine-tuned MARBERTv2 [2] (S6) and "
        "QLoRA-tuned Jais 2 [3] (S7). **S8:** Jais-3-7B [12] does the whole task by native tool "
        "calling. **S9:** the policy decides whether S8's calls are executed or asked about (S8's "
        "declines and arguments are kept; S1 v1.1 gives the intent where S8 made no valid call, "
        f"{min(s8_invalid_nat)}-{max(s8_invalid_nat)} of {n_conf} items a seed).")

    R.h2("3.4 Changes from the proposal")
    R.para(
        "**Training data:** the proposal planned manually written items only; "
        f"{drafted('train')} of {len(ITEMS['train'])} training items were AI-drafted, because the human "
        "requests available at the freeze were needed for an all-human test "
        f"({n_all} items, {t1['pairs']} test pairs; planned: 300, 20). **Evaluation:** a "
        "screened contributor collection became a second, pre-registered evaluation. **Systems:** "
        "S1-S3 ran on a laptop CPU as proposed; the open-weight extensions S4-S9 ran on Kaggle T4 "
        "GPUs. Jais 2 stayed the pre-registered LLM; Jais 3 (released after the freeze, before the "
        "test was opened) joined S4 unchanged and added S8-S9.")

    # 4 ------------------------------------------------------------------------------------------
    R.h1("4. Evaluation and results")
    R.h2("4.1 Evaluation data and metrics")
    R.para(
        "**Evaluation data.** Both sets hold native-written requests for the two tools, covering "
        "all routes, with incomplete variants and in-domain out-of-scope requests; "
        f"{share(NAT_CONF, 'domain', 'project_management'):.0%} of natural messages are about project "
        "management. Test intervals span about ±0.09 STS, so the natural set carries the main "
        "conclusion. **Training data** is relevant (same tools, routes and contract) but not adequate "
        f"in quality or coverage: {drafted('train')} of {len(ITEMS['train'])} items are AI-drafted; "
        f"code-switching marks {share(ITEMS['train'], 'code_switched'):.0%} of them but "
        f"{share(ITEMS['test'], 'code_switched'):.0%} of the test and "
        f"{share(NAT_CONF, 'code_switched'):.0%} of the natural set; "
        f"{reminder_share('train', 'فكرني'):.0%} of training reminders use the verb «فكرني» and "
        f"{reminder_share('train', 'تذكير'):.0%} the noun «تذكير», against "
        f"{reminder_share('test', 'فكرني'):.0%} and {reminder_share('test', 'تذكير'):.0%} in the test. "
        "Its phrasing, rather than its size, appears to limit the classifiers (§4.3). "
        "**Separation.** Splits follow scenario families under a hash-locked manifest; vocabularies "
        "and hyper-parameters come from train/dev only; the test was scored once; natural-set systems "
        "were frozen before labelling. **Limitations.** The test is one writer's templated phrasing "
        f"(52 sentence frames; {frames['frame_seen_in_train']} items share a frame with train) by the "
        "author of S1's lexicon; the natural set has new writers but the same scenarios. C was chosen "
        f"on {len(ITEMS['dev'])} mostly drafted dev items; retuned by family-grouped cross-validation on "
        f"train+dev, S2 and S3 still score only {f3(NAT['summary']['S2-gcv']['sts'])} and "
        f"{f3(NAT['summary']['S3-gcv']['sts'])} on the natural set.")
    R.para(
        "**Metrics.** Besides STS: intent and route macro-F1; unsafe execution (a wrong or unwarranted "
        "tool call) over all items and over executions; unnecessary clarification (Unn. clar.: "
        "gold ACT answered with CLARIFY); and pairs with both versions correct. Intervals are "
        "family-level bootstraps (10,000 resamples) [8]; paired tests swap whole families [7], "
        "Holm-corrected [9]. **Hypotheses** (fixed before labelling): "
        "N1, S1 beats S2; N2, S1 v1.1 beats S1; N3, S9 beats S8.")

    R.h2("4.2 Results")
    h1_, h2_ = cmp_["S2 vs S1"], cmp_["S3 vs S1"]
    R.para(
        "On the locked test (Table 2) both classifiers scored below the rules "
        f"(S2 − S1 = {signed(h1_['difference'])} {ci(h1_['ci'])}, Holm p = {h1_['p_holm']:.3f}; "
        f"S3 − S1 = {signed(h2_['difference'])}, Holm p = {h2_['p_holm']:.3f}; S3 vs S2: "
        f"p = {cmp_['S3 vs S2']['p_value']:.2f}), so here the answer to the research question is "
        "negative. Clustering items by shared sentence frame keeps the differences but raises Holm p "
        "to 0.055-0.526 (Table 6).")
    rows = []
    for name, m, c in (("oracle (gold intent)", o, TEST["systems"]["oracle"]),
                       ("S1 keyword rules", t1, TEST["systems"]["S1"]),
                       ("S2 TF-IDF + LR", t2, TEST["systems"]["S2"]),
                       ("S3 + char n-grams", t3, TEST["systems"]["S3"])):
        rows.append([name, f"{m['safe_task_success']:.3f} {ci(c['ci']['safe_task_success'])}",
                     f3(m["intent_macro_f1"]), f3(m["route_macro_f1"]), f3(m["unsafe_execution_rate"]),
                     f3(1 - m["execution_precision"]),
                     f3(m["unnecessary_clarification_rate"]), f3(m["pair_both_success"])])
    R.table(f"Locked test ({t1['n']} items, {t1['pairs']} pairs). Metrics as defined in §4.1.",
            ["System", "STS [95% CI]", "Intent F1", "Route F1", "Unsafe (all)", "Unsafe (ACT)",
             "Unn. clar.", "Pairs"], rows, widths=[3.3, 3.3, 1.5, 1.5, 1.6, 1.7, 1.6, 1.3],
            bold_rows={1})

    nat_rows = []
    for name, key in (("oracle", "oracle"), ("S1 rules", "S1"),
                      ("S2 TF-IDF", "S2"), ("S3 + char", "S3")):
        m = nat_metrics(key)
        c = NAT["systems"][key]["ci"]["safe_task_success"]
        r = m["success_by_gold_route"]
        nat_rows.append([name, f"{m['safe_task_success']:.3f} {ci(c)}", f3(m["intent_macro_f1"]),
                         f3(m["route_macro_f1"]),
                         f3(m["unsafe_execution_rate"]), f3(1 - m["execution_precision"]),
                         f3(m["unnecessary_clarification_rate"]),
                         f3(m["pair_both_success"]),
                         f"{r['ACT']:.2f}/{r['CLARIFY']:.2f}/{r['ABSTAIN']:.2f}"])
    for name, key, summ in (("S8 Jais-3 tools", "S8", s8m), ("S9 S8 + policy", "S9", s9m)):
        nat_rows.append([name, f"{summ['sts']:.3f} {ci(summ['ci'])}",
                         f3(nat_seed_mean(key, "intent_macro_f1")),
                         f3(nat_seed_mean(key, "route_macro_f1")),
                         f3(nat_seed_mean(key, "unsafe_execution_rate")),
                         f3(1 - nat_seed_mean(key, "execution_precision")),
                         f3(nat_seed_mean(key, "unnecessary_clarification_rate")),
                         f3(nat_seed_mean(key, "pair_both_success")),
                         "/".join(f"{nat_seed_route(key, r):.2f}" for r in ("ACT", "CLARIFY", "ABSTAIN"))])
    R.table(f"Natural set ({n_conf} items, {n_writers} writers, {n1['pairs']} pairs); S8/S9: 3-seed "
            "means; last column: success by gold route.",
            ["System", "STS [95% CI]", "Intent F1", "Route F1", "Unsafe (all)", "Unsafe (ACT)",
             "Unn. clar.", "Pairs", "ACT/CLAR/ ABST"], nat_rows,
            widths=[2.55, 3.25, 1.2, 1.2, 1.3, 1.3, 1.15, 1.15, 2.9], bold_rows={1})
    R.para(
        "**N1 was confirmed** with a larger gap: "
        f"{signed(N1['difference'])} {ci(N1['ci'])}, Holm p = {N1['p_holm']:.4f}; S1 led for 13 of 17 "
        "writers and under every alternative labelling (Table 7). **N3 was confirmed:** "
        f"{signed(N3['difference'])} {ci(N3['ci'])}, Holm p = {N3['p_holm']:.3f}, but {s9_fallback} "
        f"of the {s9_gain} items gained (three seeds) were S8's invalid calls answered by the rule "
        f"fallback. **N2 was not:** the extractor fix changed one natural item. **Safety.** "
        f"{1 - t1['execution_precision']:.0%} of S1's test and {1 - n1['execution_precision']:.0%} of "
        f"its natural executions were wrong; harmful ones (wrong tool or time, or should have asked or "
        f"declined) rose from {harm_t} of {exec_t} to {harm_n} of {exec_n}, the rest differing only in "
        f"reminder wording. Accepting any reminder text, natural STS is {len_s1:.3f} (S1) and "
        f"{len_or:.3f} (oracle). No LLM recognizer beat the rules significantly (post hoc, Table 8).")

    R.h2("4.3 Analysis and errors")
    s3h = HUM["tests"]["S3 +contributors vs S1-v1.1"]
    s3h = {**s3h, "difference": -s3h["difference"],   # report as rules minus S3, the gap to close
           "ci": {"low": -s3h["ci"]["high"], "high": -s3h["ci"]["low"]}}
    R.para(
        "**Training data (exploratory, Table 5).** In leave-one-writer-out training, about "
        f"{n_human} added human messages raised S3 on new writers from "
        f"{hum['S3 | drafted+author']['sts']:.3f} to {hum['S3 | drafted+author+contributors']['sts']:.3f} "
        f"(gap to the rules {signed(s3h['difference'])} {ci(s3h['ci'])}); trained on them alone "
        f"({hum['S3 | contributors only']['sts']:.3f}) it beat the {n_fit} drafted and author items, so "
        "size does not explain the gap, but they come from the natural set's collection. "
        "**Failure types.** S2's 20 extra test failures are all intent errors (Tables 10, 11). **Shared pipeline.** With "
        "gold intents, 14 of 23 test failures vanish once two framings, such as the want-verb «عايز», "
        "are stripped (Table 9). New writers failed instead on emphasis words («ضروري») "
        "and day-scale expressions (oracle 0.58 on 31 items), hence N2.")

    # 5 ------------------------------------------------------------------------------------------
    R.h1("5. Conclusion")
    R.para(
        f"We built a task contract, a safety-aware metric, a {n_all}-item benchmark and a {n_conf}-message "
        f"evaluation by {n_writers} other writers on the same scenarios and two tools. Rules beat TF-IDF classifiers trained on mostly "
        f"AI-drafted data on both sets ({signed(N1['difference'])} STS on new writers), owing more to "
        "the data than the method (exploratory). A policy with a rule fallback raised "
        f"LLM tool calling by {signed(N3['difference'])} STS, mostly by recovering invalid calls. No "
        "intent layer beat the cheap rules significantly, yet none is deployable: the extractor "
        f"caps all systems and {100 * (1 - t1['execution_precision']):.0f}-"
        f"{1 - n1['execution_precision']:.0%} of S1's executions were wrong.")

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
    R.table("Safe Task Success of the extension systems on both locked sets, with the numeric format "
            "each GPU model ran in, or CPU (S4: primary setting, 6 examples with prior calibration; S5/S6 fitted after "
            "both evaluations, exploratory). ¹S9 uses core v1.1, whose fix was drawn from the test errors.",
            ["System", "Format", "Locked test (120)", "Natural (238)"], ext,
            widths=[5.8, 2.0, 4.1, 4.1])
    j3 = load("results/s4/7a-jais3/s4_results/jais3-7b.json")["systems"]
    j3_inline = j3["S4-jais3-7b-6shot-inline-prior"]["metrics"]["safe_task_success"]
    j3_declined = FAILS["S4-jais3-7b-6shot-prior"]["declined an in-scope request"]
    s8_invalid = [load(s8_file)["systems"][f"S8-jais3-7b-s{i}"]["metrics"]["invalid_outputs"] for i in range(3)]
    R.para(
        "**Systems S4-S7.** They replace only the intent recognizer; the extractor and policy are "
        "shared. **S4** prompts an LLM for the intent with 6 examples (Jais 2 8B [5, 11], Jais-3-7B, "
        "Jais-3-3.7B [12], Nile-Chat 4B [13], Qwen3 8B [14]). **S5** is logistic regression on frozen "
        "multilingual-e5-small embeddings. **S6** is fine-tuned MARBERTv2 [2] (five seeds, recipe after "
        "[10]). **S7** is Jais 2 with QLoRA [3] (three seeds).")
    R.para(
        "**Precision.** Kaggle's T4 GPUs have no bf16, so the models ran in different formats: Jais 2 "
        "overflowed in fp16 and ran 4-bit, Jais-3-7B and Qwen3 ran in fp16, the smaller models in fp32. "
        "Differences between these models may therefore partly reflect precision. **Jais-3-7B.** In its "
        f"primary setting it declined {j3_declined} in-scope test requests; with the exemplars listed in "
        f"the user message (a secondary analysis declared before the test was opened) it scored "
        f"{j3_inline:.3f}. As S8 it made no valid call on " + ", ".join(map(str, s8_invalid[:-1]))
        + f" and {s8_invalid[-1]} of the 120 test items.")
    R.para(
        f"**Protocol v3.** A registered test of the training-data explanation on {v3_n} unseen "
        "contributor messages (`docs/protocol_v3.md`) has frozen systems and awaits labelling; no "
        "result from it is reported.")
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
    R.table("Exploratory leave-one-writer-out training on the natural set: STS (intent accuracy). Each "
            f"held-out writer's scenario families are removed from training; about {n_human} contributor "
            "messages per fold; S2/S3 use grouped-CV settings; S1 is core v1.1.",
            ["Classifier", f"Drafted + author ({n_fit})", f"Contributors only (~{n_human})", "Both"],
            hrows, widths=[3.5, 4.2, 4.2, 4.1])
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
            ["Labels / rule", "n", "N1 S1 − S2", "N3 S9 − S8"], vrows, widths=[7.4, 1.2, 3.7, 3.7])
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
    R.table("Locked test: failures by type (counts of 120 items; from results/error_analysis).",
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

    R.h1("Appendix C. Reproducing the results")
    R.para("All commands run from the repository root on a CPU; T2 stands for Table 2. The five "
           "analysis scripts (T2-T10) rebuild their tables from the stored outputs and reproduce the "
           "committed files exactly (Table 7 also needs the private labelling sheets in "
           "`data/raw/private/`, which are not shared); the encoder commands retrain S5/S6 (slow on "
           "CPU). The GPU systems (S4, S7, S8) were run, and S7 was trained, with "
           "`notebooks/kaggle_s4_llm.ipynb`.")
    R.code([
        "uv sync",
        "uv run pytest                                               # tests",
        "uv run python scripts/error_analysis.py --out results/error_analysis   # T2, T10",
        "uv run python scripts/natural_tables.py                     # T3",
        "uv run python scripts/posthoc_analysis.py --out results/posthoc        # T6",
        "uv run python scripts/posthoc_llm_vs_rules.py               # T8",
        "uv run --with openpyxl python scripts/natural_sensitivity.py           # T7",
        "uv run --extra neural python scripts/posthoc_encoders_human.py encoders # T4",
        "uv run --extra neural python scripts/posthoc_encoders_human.py human   # T5",
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
        "guide and flagged 24 that broke a rule; the author approved each correction. No message was "
        "labelled from scratch by an AI system. The author designed the study, wrote the test requests "
        "and their labels, recruited the contributors, ran the experiments and is responsible for every "
        "claim in this report.")
    return R


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
    build(commit, branch).save(docx)
    print(f"wrote {docx} for commit {commit}")
    if dirty:
        print("DRAFT: uncommitted changes, so the printed commit does not hold these files; do not submit")
    if not git("branch", "-r", "--contains", commit):
        print(f"WARNING: commit {commit[:7]} is not on the remote yet; push before submitting")
    if a.pdf:
        to_pdf(docx, docx.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
