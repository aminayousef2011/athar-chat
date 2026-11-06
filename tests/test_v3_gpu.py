"""Protocol v3 amendments A1, A4 and A5: append-only check, the size-matched control, and the GPU
systems' prompts (CPU only, no model is loaded)."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from athar import agent
from athar.data.split import load_split
from athar.intent import llm
from athar.normalize import norm
from athar.v2 import hybrid
from athar.v3 import amend, controls, gpu
from athar.v3 import data as D
from athar.v3 import systems as S

REPO = Path(__file__).resolve().parents[1]
EVAL_SEED = int(yaml.safe_load((REPO / "configs" / "eval.yaml").read_text(encoding="utf-8"))["seed"])


def _pool() -> list:
    return list(load_split(S.MAIN, "train")) + list(load_split(S.MAIN, "dev"))


# --------------------------------------------------------------------------- A1 append-only
def test_amendments_may_grow_but_not_change(tmp_path: Path) -> None:
    f = tmp_path / "amend.md"
    f.write_text("A1 first\n", encoding="utf-8")
    snap = amend.snapshot(f)
    f.write_text("A1 first\nA2 second\n", encoding="utf-8")
    assert amend.check(snap, f) is None
    f.write_text("A1 edited\nA2 second\n", encoding="utf-8")
    assert "edited" in (amend.check(snap, f) or "")
    f.write_text("A1\n", encoding="utf-8")
    assert "shorter" in (amend.check(snap, f) or "")


def test_amendments_check_ignores_line_endings(tmp_path: Path) -> None:
    f = tmp_path / "amend.md"
    f.write_bytes(b"A1 first\n")
    snap = amend.snapshot(f)
    f.write_bytes(b"A1 first\r\nA2\r\n")
    assert amend.check(snap, f) is None


# --------------------------------------------------------------------------- A4 control
def test_contributors_only_pool_has_no_drafted_or_author_item() -> None:
    class It:
        def __init__(self, fam: str) -> None:
            self.family_id = fam

    natural = [It("fam-c001"), It("fam-c009")]
    pool = controls.contributors_pool("block1", {"block1": {"fam-c001"}}, natural)
    assert [i.family_id for i in pool] == ["fam-c009"]


# --------------------------------------------------------------------------- A5 prompts
def test_guide_text_is_the_annotation_guides_section_5_verbatim() -> None:
    guide = (REPO / "docs" / "annotation_guide.md").read_text(encoding="utf-8")
    section = re.search(r"## 5\. Reminder text\n\n(.*?)\n\nScoring compares", guide, re.S)
    assert section is not None
    assert section.group(1) == gpu.GUIDE_REMINDER_TEXT


def test_tool_exemplars_are_author_written_one_per_category_and_deterministic() -> None:
    ex = gpu.select_tool_exemplars(_pool())
    assert [gpu._category(it) for it in ex] == list(gpu.CATEGORIES)
    assert all(it.source == "author" for it in ex)
    assert len({it.family_id for it in ex}) == 6
    assert [it.id for it in ex] == [it.id for it in gpu.select_tool_exemplars(_pool())]


def test_gold_calls_parse_back_to_the_gold_route() -> None:
    for it in gpu.select_tool_exemplars(_pool()):
        call = gpu.gold_call(it)
        assert call.split("(")[0] in agent.TOOL_NAMES
    act = next(it for it in gpu.select_tool_exemplars(_pool()) if it.route == "ACT")
    local = act.args.datetime.astimezone(ZoneInfo(act.timezone)).strftime("%Y-%m-%dT%H:%M")
    assert local in gpu.gold_call(act)


def test_fair_prompt_extends_the_frozen_s8_prompt() -> None:
    ref = dt.datetime(2026, 10, 18, 9, 0, tzinfo=ZoneInfo("Africa/Cairo"))
    ex = gpu.select_tool_exemplars(_pool())
    prompt = gpu.fair_system_prompt(ref, "Africa/Cairo", ex)
    assert prompt.startswith(agent.system_prompt(ref, "Africa/Cairo"))
    assert gpu.GUIDE_REMINDER_TEXT in prompt
    assert all(it.text in prompt for it in ex)


def test_s4_shot_items_match_the_notebooks_shots() -> None:
    train = list(load_split(S.MAIN, "train"))
    assert [(norm(it.text), it.intent) for it in gpu.shot_items(train, seed=EVAL_SEED)] == \
        llm.select_shots(train, per_label=2, seed=EVAL_SEED)


def test_s4_raw_and_normalized_variants_use_the_same_items() -> None:
    shots_n, prior_n = gpu.s4_inputs("norm", EVAL_SEED)
    shots_r, prior_r = gpu.s4_inputs("raw", EVAL_SEED)
    assert [(norm(t), lab) for t, lab in shots_r] == shots_n
    assert [norm(t) for t in prior_r] == prior_n
    assert len(prior_n) == 60


def test_s8_fair_trace_replays_as_s8_and_s9() -> None:
    class Fake:
        def complete(self, conversations):  # noqa: ANN001, ANN201
            assert conversations[0][0]["role"] == "system"
            call = {"name": "create_alarm", "arguments": {"datetime": "2026-10-19T07:00"}}
            return [agent.Completion(text=f"<tool_call>{json.dumps(call)}</tool_call>") for _ in conversations]

    ref = dt.datetime(2026, 10, 18, 9, 0, tzinfo=ZoneInfo("Africa/Cairo"))
    items = [D.V3Item("c001-a-x", "صحيني بكرة الساعة 7 الصبح", ref, "Africa/Cairo", "fam-c001", "block1", "X")]
    rows = gpu.respond(items, Fake(), gpu.select_tool_exemplars(_pool()))
    assert rows[0].route == "ACT" and rows[0].error is None
    trace = {r.id: json.loads(json.dumps(r.__dict__, ensure_ascii=False)) for r in rows}
    s8 = hybrid.s8_response(trace["c001-a-x"], "Africa/Cairo")
    assert s8 is not None and s8.tool_call is not None and s8.tool_call.name == "create_alarm"


def test_gpu_run_refuses_an_existing_freeze(tmp_path: Path) -> None:
    (tmp_path / "freeze_gpu.json").write_text("{}", encoding="utf-8")
    with pytest.raises(S.FreezeError):
        gpu.run(out=tmp_path)
