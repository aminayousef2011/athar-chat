"""Protocol v3, GPU systems: descriptive scoring (athar.v3.gpu_score) and its refusals."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from athar.schema import GoldItem
from athar.v3 import evaluate as E
from athar.v3 import gpu_score as G

CFG: dict[str, Any] = {"seed": 1, "level": 0.95, "resamples": 200, "permutations": 200, "alpha": 0.05}
REF = datetime.fromisoformat("2026-11-01T08:00:00+02:00")
KINDS = [
    ("فكرني بكرة الساعة 9 الصبح اراجع التقرير", "create_reminder", "ACT",
     {"args": {"datetime": "2026-11-02T09:00:00+02:00", "reminder_text": "اراجع التقرير"}},
     {"name": "create_reminder", "arguments": {"datetime": "2026-11-02 09:00", "reminder_text": "اراجع التقرير"}}),
    ("صحيني بكرة الساعة 7", "create_alarm", "CLARIFY", {"clarify_targets": ["am_pm"]},
     {"name": "create_alarm", "arguments": {"datetime": "2026-11-02 07:00"}}),   # executes a guess
    ("امسح كل التذكيرات", "unsupported", "ABSTAIN", {"unsupported_reason": "deletion"},
     {"name": "decline", "arguments": {"reason": "deletion"}}),
]


def _items(n: int = 30) -> list[GoldItem]:
    out = []
    for k in range(n):
        text, intent, route, extra, _ = KINDS[k % 3]
        out.append(GoldItem.model_validate({
            "id": f"x{k:03d}-a-w{k % 4}", "family_id": f"fam-x{k:03d}", "text": text, "ref_datetime": REF,
            "intent": intent, "route": route, "source": "contributor", **extra}))
    return out


def test_gpu_systems_are_scored_like_every_other_system() -> None:
    items = _items()
    gold = {it.id: it.intent for it in items}
    s4 = {G.S4_NAMES[0]: dict(gold), G.S4_NAMES[1]: {**gold, items[0].id: "create_alarm"}}
    trace = {it.id: {"id": it.id, "calls": [KINDS[k % 3][4]]} for k, it in enumerate(items)}
    trace[items[3].id] = {"id": items[3].id, "calls": []}            # no valid call
    r = G.analyse_gpu(items, s4, trace, training_families=set(), cfg=CFG)
    assert r["confirmatory_items"] == 30 and r["s8_fair_no_valid_call"] == 1
    assert r["s4_norm_raw_same_intent"] == 29
    # S8-fair executes the incomplete alarm requests; S9 lets the policy ask instead
    assert r["summary"]["S8-fair"]["by_route"]["CLARIFY"] == 0.0
    assert r["summary"]["S9-fair"]["by_route"]["CLARIFY"] == 1.0
    assert r["comparisons"]["S9-fair vs S8-fair"]["difference"] > 0
    assert r["comparisons"]["S4 raw vs normalized text"]["difference"] < 0
    assert all(c["p_adjustment"].startswith("none") for c in r["comparisons"].values())
    assert set(r["items"]) == {*G.S4_NAMES, "S8-fair", "S9-fair", "S1-v1.1"}


def test_scoring_refuses_before_g5_and_runs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "evaluation_gpu.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(E.EvaluationError, match="scored once"):
        G.run(tmp_path / "manifest.json", out=out, cfg=CFG)
    monkeypatch.setattr(E, "OUT", tmp_path / "absent.json")
    with pytest.raises(E.EvaluationError, match="run gate G5 first"):
        G.run(tmp_path / "manifest.json", out=tmp_path / "new.json", cfg=CFG)


def test_the_committed_gpu_freeze_verifies() -> None:
    s4, trace, sha = G.frozen_gpu()
    assert set(s4) == set(G.S4_NAMES) and len(trace) == len(s4[G.S4_NAMES[0]]) == 236 and len(sha) == 64


def test_a_changed_gpu_output_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    real = E._freeze

    def tampered(name: str) -> dict[str, Any]:
        fz = real(name)
        if name == "freeze_gpu.json":
            fz["outputs"][G.S8_NAME] = {**fz["outputs"][G.S8_NAME], "sha256": "0" * 64}
        return fz

    monkeypatch.setattr(E, "_freeze", tampered)
    with pytest.raises(E.EvaluationError, match="S8-fair"):
        G.frozen_gpu()


def test_run_scores_once_after_g5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = _items()
    gold = {it.id: it.intent for it in items}
    trace = {it.id: {"id": it.id, "calls": [KINDS[k % 3][4]]} for k, it in enumerate(items)}
    g5 = tmp_path / "evaluation.json"
    g5.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(E, "OUT", g5)
    monkeypatch.setattr(G, "frozen_gpu", lambda: ({n: dict(gold) for n in G.S4_NAMES}, trace, "f" * 64))
    monkeypatch.setattr(G, "load_split", lambda manifest, split, unlock_test=False: items)
    monkeypatch.setattr(E, "check_gold", lambda gold_items, frozen_hash, preds: None)
    monkeypatch.setattr(E, "_freeze", lambda name: {"v3_set_sha256": "frozen"})
    monkeypatch.setattr(G, "read_manifest", lambda path: {"manifest_sha256": "m",
                                                          "splits": {"train": {"families": []},
                                                                     "dev": {"families": []}}})
    out = tmp_path / "evaluation_gpu.json"
    r = G.run(tmp_path / "manifest.json", out=out, cfg=CFG)
    assert out.exists() and r["freeze_gpu_sha256"] == "f" * 64 and r["confirmatory_items"] == 30
    with pytest.raises(E.EvaluationError, match="scored once"):
        G.run(tmp_path / "manifest.json", out=out, cfg=CFG)


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert G.main(["--manifest", str(tmp_path / "m.json"), "--check"]) == 0
    assert "ready" in capsys.readouterr().out
    monkeypatch.setattr(E, "OUT", tmp_path / "absent.json")
    monkeypatch.setattr(G, "OUT", tmp_path / "evaluation_gpu.json")
    assert G.main(["--manifest", str(tmp_path / "m.json")]) == 1
    assert "run gate G5 first" in capsys.readouterr().out
