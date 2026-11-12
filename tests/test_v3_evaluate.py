"""Protocol v3, gate G5: the scoring run's analysis and its refusals (athar.v3.evaluate)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from athar.schema import GoldItem
from athar.v3 import data as D
from athar.v3 import evaluate as E

CFG: dict[str, Any] = {"seed": 1, "level": 0.95, "resamples": 200, "permutations": 200, "alpha": 0.05}
REF = datetime.fromisoformat("2026-11-01T08:00:00+02:00")


def _items(writers: int = 8, per_writer: int = 20) -> list[GoldItem]:
    """ACT reminders, CLARIFY alarms and ABSTAIN requests from several writers and families."""
    kinds = [
        ("فكرني بكرة الساعة 9 الصبح اراجع التقرير", "create_reminder", "ACT",
         {"args": {"datetime": "2026-11-02T09:00:00+02:00", "reminder_text": "اراجع التقرير"}}),
        ("صحيني بكرة الساعة 7", "create_alarm", "CLARIFY", {"clarify_targets": ["am_pm"]}),
        ("امسح كل التذكيرات", "unsupported", "ABSTAIN", {"unsupported_reason": "deletion"}),
    ]
    out = []
    for w in range(writers):
        for k in range(per_writer):
            text, intent, route, extra = kinds[k % 3]
            out.append(GoldItem.model_validate({
                "id": f"x{k:03d}-a-w{w}", "family_id": f"fam-x{k:03d}", "text": text, "ref_datetime": REF,
                "intent": intent, "route": route, "source": "contributor", **extra}))
    return out


def _preds(items: list[GoldItem], wrong_every: dict[str, int]) -> dict[str, dict[str, str]]:
    """Gold intents, except every n-th item for a system gets a wrong one."""
    names = ["S3-drafted", "S3-human", "S3-contrib-only", *(f"S6-{k}-s{s}" for k in ("drafted", "human") for s in E.SEEDS)]
    out = {}
    for name in names:
        n = wrong_every.get(name.split("-s")[0] if name.startswith("S6") else name, 0)
        out[name] = {it.id: ("create_alarm" if it.intent != "create_alarm" else "unsupported")
                     if n and k % n == 0 else it.intent for k, it in enumerate(items)}
    return out


def test_analysis_follows_the_registered_plan() -> None:
    items = _items()
    preds = _preds(items, {"S3-drafted": 2, "S3-human": 0, "S3-contrib-only": 3, "S6-drafted": 2, "S6-human": 0})
    r = E.analyse(items, preds, training_families=set(), cfg=CFG)
    assert r["confirmatory_items"] == 160 and len(r["writers"]) == 8
    assert r["hypothesis_tests"].startswith("confirmatory")
    ha, hb = r["primary"]["H-a S3-human vs S3-drafted"], r["primary"]["H-b S6-human vs S6-drafted (seed means)"]
    assert ha["difference"] > 0 and hb["difference"] > 0
    assert "p_holm" in ha and ha["p_holm"] >= ha["p_value"] and "mcnemar" in ha
    hc = r["secondary"]["H-c ABSTAIN success, S6-human vs S6-drafted (seed means)"]
    assert hc["n"] == sum(it.route == "ABSTAIN" for it in items) and "p_holm" not in hc
    assert set(r["estimation"]) == {"S3-human minus S1-v1.1", "S6-human (seed mean) minus S1-v1.1"}
    assert "size-matched control (A4): S3-contrib-only vs S3-drafted" in r["descriptive"]
    assert set(r["sensitivity"]) == {"writer_clustered", "leave_one_writer_out"}
    assert r["summary"]["oracle-v1.1"]["by_route"]["ABSTAIN"] == 1.0


def test_small_sets_are_descriptive_only() -> None:
    items = _items(writers=3, per_writer=20)
    r = E.analyse(items, _preds(items, {}), training_families=set(), cfg=CFG)
    assert r["hypothesis_tests"].startswith("descriptive only")
    assert all("p_holm" not in t for t in r["primary"].values())


def test_training_families_are_not_confirmatory() -> None:
    items = _items()
    r = E.analyse(items, _preds(items, {}), training_families={"fam-x000"}, cfg=CFG)
    assert r["confirmatory_items"] == len(items) - 8


def test_reading_rule_needs_the_lower_bound_above_minus_ten_points() -> None:
    items = _items()
    worse = _preds(items, {"S3-human": 2, "S6-human": 2})   # half the items wrong
    r = E.analyse(items, worse, training_families=set(), cfg=CFG)
    est = r["estimation"]["S3-human minus S1-v1.1"]
    assert est["ci"]["low"] <= -E.MARGIN and not est["within_0.10_of_the_rules"]
    assert est["reading"].startswith("not shown")


def test_gold_must_be_the_frozen_set() -> None:
    items = _items(writers=1, per_writer=3)
    frozen = D.set_hash([D.V3Item(it.id, it.text, it.ref_datetime, it.timezone, it.family_id, "", "w0")
                         for it in items])
    preds = {"S3-human": {it.id: it.intent for it in items}}
    E.check_gold(items, frozen, preds)
    edited = [items[0].model_copy(update={"text": items[0].text + " بسرعة"}), *items[1:]]
    with pytest.raises(E.EvaluationError, match="not the frozen v3 set"):
        E.check_gold(edited, frozen, preds)
    with pytest.raises(E.EvaluationError, match="one prediction per gold item"):
        E.check_gold(items, frozen, {"S3-human": {items[0].id: "unsupported"}})


def test_the_scoring_run_happens_once(tmp_path: Path) -> None:
    out = tmp_path / "evaluation.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(E.EvaluationError, match="happens once"):
        E.evaluate(tmp_path / "manifest.json", out=out, cfg=CFG)


def test_the_committed_freezes_hold_the_registered_systems() -> None:
    preds, hashes = E.frozen_predictions()
    assert len(preds) == 13 and all(len(p) == 236 for p in preds.values())
    assert hashes["freeze_gpu.json"].startswith("absent")


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """The run's inputs replaced by synthetic ones: frozen predictions and a locked gold set."""
    items = _items()
    preds = _preds(items, {"S3-drafted": 2, "S6-drafted": 3})
    monkeypatch.setattr(E, "frozen_predictions", lambda: (preds, {"freeze_s3.json": "h"}))
    monkeypatch.setattr(E, "_freeze", lambda name: {"v3_set_sha256": "frozen", "freeze_sha256": "h"})
    monkeypatch.setattr(E, "check_gold", lambda gold, frozen_hash, p: None)
    monkeypatch.setattr(E, "load_split", lambda manifest, split, unlock_test=False: items)
    real_read = E.read_manifest
    monkeypatch.setattr(E, "read_manifest",
                        lambda path: {"manifest_sha256": "m"} if Path(path).name == "v3.json" else real_read(path))
    monkeypatch.setattr(E, "_cfg", lambda: CFG)
    monkeypatch.setattr(E, "OUT", tmp_path / "evaluation.json")
    return {"items": items, "out": tmp_path / "evaluation.json", "manifest": tmp_path / "v3.json"}


def test_scoring_run_writes_its_record_once(frozen: dict[str, Any]) -> None:
    r = E.evaluate(frozen["manifest"], out=frozen["out"], cfg=CFG)
    saved = json.loads(frozen["out"].read_text(encoding="utf-8"))
    assert saved["v3_manifest"] == "m" and saved["confirmatory_items"] == r["confirmatory_items"] == 160
    assert saved["primary"]["H-a S3-human vs S3-drafted"]["difference"] > 0
    with pytest.raises(E.EvaluationError):
        E.evaluate(frozen["manifest"], out=frozen["out"], cfg=CFG)


def test_command_line_check_run_and_refusal(frozen: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert E.main(["--manifest", str(frozen["manifest"]), "--check"]) == 0
    assert not frozen["out"].exists()
    assert E.main(["--manifest", str(frozen["manifest"])]) == 0
    assert frozen["out"].exists()
    assert E.main(["--manifest", str(frozen["manifest"])]) == 1
    out = capsys.readouterr().out
    assert "ready:" in out and "H-a S3-human vs S3-drafted" in out and "refused:" in out


def test_unregistered_prediction_sets_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E.S, "verify", lambda: [])
    monkeypatch.setattr(E.amend, "check", lambda recorded: None)
    real = E._freeze
    def one_extra(name: str) -> dict[str, Any]:
        fz = real(name)
        if name == "freeze_controls.json":
            fz["predictions"] = {**fz["predictions"], "S3-extra": fz["predictions"]["S3-contrib-only"]}
        return fz
    monkeypatch.setattr(E, "_freeze", one_extra)
    with pytest.raises(E.EvaluationError, match="not the registered systems"):
        E.frozen_predictions()
    monkeypatch.setattr(E.S, "verify", lambda: ["freeze_s3.json was edited"])
    with pytest.raises(E.EvaluationError, match="do not verify"):
        E.frozen_predictions()
